from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import pydantic

from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    from mlflow.entities.issue import Issue
    from mlflow.entities.trace import Trace
    from mlflow.genai.discovery.entities import DiscoverIssuesResult

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas for LLM structured output
# ---------------------------------------------------------------------------


class _AgentDescription(pydantic.BaseModel):
    description: str = pydantic.Field(
        description="What the agent does — a concise summary of its purpose"
    )
    capabilities: list[str] = pydantic.Field(
        description="Tools, skills, or knowledge areas the agent has"
    )
    limitations: list[str] = pydantic.Field(
        description="Known constraints, boundaries, or things the agent cannot do"
    )


class _TestCase(pydantic.BaseModel):
    goal: str = pydantic.Field(description="What the simulated user is trying to accomplish")
    persona: str = pydantic.Field(description="A short description of who the simulated user is")
    simulation_guidelines: list[str] = pydantic.Field(
        description="Instructions for how the simulated user should behave"
    )


class _TestCaseList(pydantic.BaseModel):
    test_cases: list[_TestCase] = pydantic.Field(description="List of test cases to simulate")


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@experimental(version="3.11.0")
@dataclass
class FindBugsResult:
    """
    Result of :func:`find_bugs`.

    Attributes:
        issues: Issues discovered across all simulated conversations.
        test_cases: Test cases that were generated and simulated.
        agent_description: Natural-language description of the agent
            produced by Step 1.
        simulation_traces: Per-test-case lists of traces produced by
            the conversation simulator.
        discover_issues_result: Full result from the underlying
            :func:`~mlflow.genai.discover_issues` call.
    """

    issues: list[Issue]
    test_cases: list[dict[str, str]]
    agent_description: str
    simulation_traces: list[list[Trace]]
    discover_issues_result: DiscoverIssuesResult


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DESCRIBE_AGENT_SYSTEM_PROMPT = """\
You are an expert at analysing AI agents. Given the agent's own response to \
"describe yourself", extract a structured description."""

_DESCRIBE_AGENT_FROM_TRACES_SYSTEM_PROMPT = """\
You are an expert at analysing AI agents. Given conversation traces from an \
AI agent, extract a structured description of what the agent does, its \
capabilities, and its limitations."""

_GENERATE_TEST_CASES_SYSTEM_PROMPT = """\
You are an expert QA engineer for AI agents. Given a description of an agent, \
generate diverse test cases that will stress-test the agent and probe for bugs.

Each test case should target a specific capability or limitation. Focus on:
- Edge cases and boundary conditions
- Error handling and graceful degradation
- Multi-step reasoning that could go wrong
- Ambiguous or conflicting requests
- Attempts to exceed the agent's stated limitations
- Unusual personas that might reveal bias or inconsistency"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_agent_response_text(predict_fn: Callable[..., Any]) -> str | None:
    """
    Call *predict_fn* with a self-description prompt and return the
    assistant's response as a plain string, or ``None`` on failure.
    """
    from mlflow.genai.utils.trace_utils import (
        extract_outputs_from_trace,
        parse_outputs_to_str,
    )

    prompt = [
        {
            "role": "user",
            "content": (
                "What can you do? Describe your capabilities, tools, and limitations in detail."
            ),
        }
    ]

    sig = inspect.signature(predict_fn)
    params = list(sig.parameters.keys())

    try:
        if params and params[0] == "messages":
            result = predict_fn(messages=prompt)
        else:
            result = predict_fn(input=prompt)
    except Exception:
        _logger.debug("predict_fn raised when asked to self-describe", exc_info=True)
        return None

    if isinstance(result, str):
        return result

    # Try to extract text the same way the simulator does
    text = parse_outputs_to_str(result)
    if text and text.strip():
        return text

    # Last resort: check the latest trace
    try:
        import mlflow

        if trace := mlflow.get_last_active_trace():
            if outputs := extract_outputs_from_trace(trace):
                return parse_outputs_to_str(outputs)
    except Exception:
        pass

    return None


def _describe_agent_from_response(
    response_text: str,
    model: str,
) -> _AgentDescription:
    from mlflow.genai.judges.utils import (
        get_chat_completions_with_structured_output,
    )
    from mlflow.types.llm import ChatMessage

    messages = [
        ChatMessage(role="system", content=_DESCRIBE_AGENT_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=f"Agent's self-description:\n\n{response_text}",
        ),
    ]
    return get_chat_completions_with_structured_output(
        model_uri=model,
        messages=messages,
        output_schema=_AgentDescription,
    )


def _describe_agent_from_traces(
    traces: list[Trace],
    model: str,
) -> _AgentDescription:
    from mlflow.genai.discovery.extraction import (
        extract_execution_paths_for_session,
    )
    from mlflow.genai.discovery.utils import group_traces_by_session
    from mlflow.genai.judges.utils import (
        get_chat_completions_with_structured_output,
    )
    from mlflow.genai.utils.trace_utils import (
        extract_available_tools_from_trace,
        resolve_conversation_from_session,
    )
    from mlflow.types.llm import ChatMessage

    # Build context from traces
    sessions = group_traces_by_session(traces)
    context_parts: list[str] = []

    # Sample up to 5 sessions to keep prompt size manageable
    for session_id, session_traces in list(sessions.items())[:5]:
        if conversation := resolve_conversation_from_session(session_traces):
            formatted = "\n".join(f"  {m['role']}: {m['content']}" for m in conversation)
            context_parts.append(f"Conversation ({session_id}):\n{formatted}")

        paths = extract_execution_paths_for_session(session_traces)
        if paths and paths != "(no routing)":
            context_parts.append(f"Execution paths: {paths}")

    # Extract tools from the first trace that has them
    tools_desc = ""
    for trace in traces[:10]:
        if tools := extract_available_tools_from_trace(trace, model=model):
            tool_names = [t.function.name for t in tools if t.function]
            tools_desc = f"Available tools: {', '.join(tool_names)}"
            break

    if tools_desc:
        context_parts.append(tools_desc)

    messages = [
        ChatMessage(
            role="system",
            content=_DESCRIBE_AGENT_FROM_TRACES_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content="\n\n".join(context_parts) if context_parts else "(no traces)",
        ),
    ]
    return get_chat_completions_with_structured_output(
        model_uri=model,
        messages=messages,
        output_schema=_AgentDescription,
    )


def _generate_test_cases(
    agent_desc: _AgentDescription,
    model: str,
    num_test_cases: int | None = None,
) -> list[dict[str, Any]]:
    from mlflow.genai.judges.utils import (
        get_chat_completions_with_structured_output,
    )
    from mlflow.types.llm import ChatMessage

    count_instruction = ""
    if num_test_cases is not None:
        count_instruction = f"\n\nGenerate exactly {num_test_cases} test cases."
    else:
        count_instruction = (
            "\n\nGenerate 5-10 diverse test cases that cover different "
            "capabilities and potential failure modes."
        )

    user_content = (
        f"Agent description: {agent_desc.description}\n\n"
        f"Capabilities:\n"
        + "\n".join(f"- {c}" for c in agent_desc.capabilities)
        + "\n\nLimitations:\n"
        + "\n".join(f"- {l}" for l in agent_desc.limitations)
        + count_instruction
    )

    messages = [
        ChatMessage(role="system", content=_GENERATE_TEST_CASES_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]
    result = get_chat_completions_with_structured_output(
        model_uri=model,
        messages=messages,
        output_schema=_TestCaseList,
    )

    test_cases = [tc.model_dump() for tc in result.test_cases]
    if num_test_cases is not None:
        test_cases = test_cases[:num_test_cases]
    return test_cases


def _load_traces(
    experiment_id: str | None,
    traces: list[Trace] | None,
) -> list[Trace] | None:
    if traces is not None:
        return traces

    if experiment_id is None:
        return None

    import mlflow

    found = mlflow.search_traces(
        experiment_ids=[experiment_id],
        max_results=50,
        return_type="list",
    )
    return found or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@experimental(version="3.11.0")
def find_bugs(
    predict_fn: Callable[..., Any],
    *,
    experiment_id: str | None = None,
    traces: list[Trace] | None = None,
    model: str | None = None,
    max_turns: int = 10,
    max_issues: int = 20,
    num_test_cases: int | None = None,
) -> FindBugsResult:
    """
    Automatically stress-test a conversational AI agent and discover bugs.

    Runs a multi-step pipeline:

    1. **Describe** — asks the agent to describe itself (falls back to
       analysing existing traces when available).
    2. **Generate test cases** — uses an LLM to create diverse,
       targeted test scenarios from the agent description.
    3. **Simulate conversations** — runs each test case through the
       :class:`~mlflow.genai.simulators.ConversationSimulator`.
    4. **Discover issues** — analyses simulation traces with
       :func:`~mlflow.genai.discover_issues`.

    Args:
        predict_fn: Agent function compatible with
            :class:`~mlflow.genai.simulators.ConversationSimulator`.
            Must accept either ``input`` or ``messages`` for conversation
            history.
        experiment_id: Optional experiment containing existing traces to
            help describe the agent. Ignored when ``traces`` is provided.
        traces: Optional list of existing traces to help describe the
            agent.
        model: LLM used for analysis, test generation, and simulation.
            Defaults to :func:`~mlflow.genai.simulators.utils.get_default_simulation_model`.
        max_turns: Maximum conversation turns per test case.
        max_issues: Maximum number of issues to report.
        num_test_cases: Number of test cases to generate. When ``None``
            the LLM decides (typically 5-10).

    Returns:
        A :class:`FindBugsResult` containing discovered issues, generated
        test cases, the agent description, simulation traces, and the
        full :class:`~mlflow.genai.discovery.entities.DiscoverIssuesResult`.
    """
    from mlflow.genai.discovery import discover_issues
    from mlflow.genai.simulators import ConversationSimulator
    from mlflow.genai.simulators.utils import get_default_simulation_model

    model = model or get_default_simulation_model()

    # ------------------------------------------------------------------
    # Step 1: Describe the agent
    # ------------------------------------------------------------------
    _logger.info("Step 1/4: Describing the agent")
    agent_desc: _AgentDescription | None = None

    # Primary: ask the agent directly
    response_text = _get_agent_response_text(predict_fn)
    if response_text and len(response_text.strip()) > 20:
        agent_desc = _describe_agent_from_response(response_text, model)

    # Fallback: analyse existing traces
    if agent_desc is None or not agent_desc.capabilities:
        if existing_traces := _load_traces(experiment_id, traces):
            agent_desc = _describe_agent_from_traces(existing_traces, model)

    if agent_desc is None:
        # Last resort: use whatever thin response we got
        if response_text:
            agent_desc = _describe_agent_from_response(response_text, model)
        else:
            agent_desc = _AgentDescription(
                description="A conversational AI agent",
                capabilities=["general conversation"],
                limitations=["unknown"],
            )

    description_str = (
        f"{agent_desc.description}\n\n"
        f"Capabilities: {', '.join(agent_desc.capabilities)}\n"
        f"Limitations: {', '.join(agent_desc.limitations)}"
    )
    _logger.info("Agent description: %s", agent_desc.description)

    # ------------------------------------------------------------------
    # Step 2: Generate test cases
    # ------------------------------------------------------------------
    _logger.info("Step 2/4: Generating test cases")
    test_cases = _generate_test_cases(agent_desc, model, num_test_cases)
    _logger.info("Generated %d test cases", len(test_cases))

    # ------------------------------------------------------------------
    # Step 3: Simulate conversations
    # ------------------------------------------------------------------
    _logger.info("Step 3/4: Simulating conversations")
    simulator = ConversationSimulator(
        test_cases=test_cases,
        max_turns=max_turns,
        user_model=model,
    )
    simulation_traces = simulator.simulate(predict_fn)

    # ------------------------------------------------------------------
    # Step 4: Discover issues
    # ------------------------------------------------------------------
    _logger.info("Step 4/4: Discovering issues")
    flat_traces = [t for session in simulation_traces for t in session]
    discover_result = discover_issues(
        traces=flat_traces,
        model=model,
        max_issues=max_issues,
    )

    return FindBugsResult(
        issues=discover_result.issues,
        test_cases=test_cases,
        agent_description=description_str,
        simulation_traces=simulation_traces,
        discover_issues_result=discover_result,
    )
