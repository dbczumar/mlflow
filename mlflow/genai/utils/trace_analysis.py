import logging
from dataclasses import dataclass
from typing import Any

import pydantic

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.utils.invocation_utils import get_chat_completions_with_structured_output
from mlflow.genai.utils.trace_utils import extract_inputs_from_trace, extract_outputs_from_trace
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types.llm import ChatMessage, FunctionToolDefinition, ToolDefinition, ToolParamsSchema

_logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """
    A reference to a trace and feedback that supports an issue.

    Attributes:
        trace_id: ID of the trace.
        span_id: Optional span ID within the trace.
        content: The actual evidence content (e.g., quote, observation, data).
    """

    trace_id: str
    span_id: str | None
    content: str


@dataclass
class Analysis:
    """
    Evidence and root cause analysis for an issue.

    Attributes:
        evidence: Description of the evidence supporting this issue.
        root_cause: Root cause analysis explaining why the issue occurred.
        citations: Specific citations with content from traces supporting the analysis.
    """

    evidence: str
    root_cause: str
    citations: list[Citation]


@dataclass
class Issue:
    """
    Represents an issue found in a session through trace analysis.

    Attributes:
        description: Concise summary of what the issue is.
        analysis: Detailed evidence and root cause analysis.
    """

    description: str
    analysis: Analysis


class TraceToolWrapper(JudgeTool):
    """
    Wrapper that adds trace_id as a parameter to existing trace tools.

    This allows an LLM to dynamically select which trace to inspect by
    providing the trace_id in tool calls.
    """

    def __init__(self, original_tool: JudgeTool):
        self.original_tool = original_tool

    @property
    def name(self) -> str:
        return self.original_tool.name

    def get_definition(self) -> ToolDefinition:
        """Get tool definition with trace_id added as the first parameter."""
        original_def = self.original_tool.get_definition()

        # Create new properties dict with trace_id as first parameter
        new_properties = {
            "trace_id": {
                "type": "string",
                "description": "ID of the trace to inspect",
            }
        }

        # Add original parameters
        if original_def.function.parameters and original_def.function.parameters.properties:
            new_properties.update(original_def.function.parameters.properties)

        # Update required fields to include trace_id
        original_required = (
            original_def.function.parameters.required if original_def.function.parameters else []
        )
        new_required = ["trace_id"] + (original_required or [])

        return ToolDefinition(
            function=FunctionToolDefinition(
                name=original_def.function.name,
                description=original_def.function.description,
                parameters=ToolParamsSchema(
                    type="object",
                    properties=new_properties,
                    required=new_required,
                ),
            ),
            type="function",
        )

    def invoke(self, trace_id: str, **kwargs) -> Any:
        """Fetch the trace by ID and call the original tool."""
        # Fetch the trace using MlflowClient
        trace = mlflow.MlflowClient().get_trace(trace_id)

        # Call original tool with the trace
        return self.original_tool.invoke(trace, **kwargs)


def _get_wrapped_trace_tools() -> list[TraceToolWrapper]:
    """
    Get all registered trace inspection tools wrapped with trace_id parameter.

    Returns:
        List of TraceToolWrapper instances for all registered judge tools.
    """
    from mlflow.genai.judges.tools.registry import list_judge_tools

    original_tools = list_judge_tools()
    return [TraceToolWrapper(tool) for tool in original_tools]


def _extract_conversation_with_trace_ids(
    session: list[Trace],
) -> list[dict[str, Any]]:
    """
    Extract conversation history from traces in session, including trace IDs for each turn.

    Args:
        session: List of traces from the same session.

    Returns:
        List of conversation messages in the format:
        [{"role": "user"|"assistant", "content": str, "trace_id": str}].
        Each trace contributes user input and assistant output messages with trace IDs.
    """
    sorted_traces = sorted(session, key=lambda t: t.info.timestamp_ms)

    conversation = []
    for trace in sorted_traces:
        trace_id = trace.info.trace_id

        if inputs := extract_inputs_from_trace(trace):
            from mlflow.genai.utils.trace_utils import parse_inputs_to_str

            user_content = parse_inputs_to_str(inputs)
            if user_content and user_content.strip():
                conversation.append(
                    {
                        "role": "user",
                        "content": user_content,
                        "trace_id": trace_id,
                    }
                )

        if outputs := extract_outputs_from_trace(trace):
            from mlflow.genai.utils.trace_utils import parse_outputs_to_str

            assistant_content = parse_outputs_to_str(outputs)
            if assistant_content and assistant_content.strip():
                conversation.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "trace_id": trace_id,
                    }
                )

    return conversation


class FeedbackSummary(pydantic.BaseModel):
    """Schema for LLM summary of negative feedback."""

    summary: str = pydantic.Field(
        description=(
            "A concise summary of all negative feedback with trace ID citations in parentheses"
        )
    )


class IssueCitation(pydantic.BaseModel):
    """Schema for a citation supporting an issue."""

    trace_id: str = pydantic.Field(description="ID of the trace that supports this issue")
    span_id: str | None = pydantic.Field(
        description="Optional span ID within the trace", default=None
    )
    content: str = pydantic.Field(
        description="The actual evidence content (e.g., quote, observation, data)"
    )


class IssueAnalysis(pydantic.BaseModel):
    """Schema for evidence and root cause analysis."""

    evidence: str = pydantic.Field(description="Description of the evidence supporting this issue")
    root_cause: str = pydantic.Field(
        description="Root cause analysis explaining why the issue occurred"
    )
    citations: list[IssueCitation] = pydantic.Field(
        description="Specific citations with content from traces supporting the analysis"
    )


class IssueDescription(pydantic.BaseModel):
    """Schema for a single issue identified in the session."""

    description: str = pydantic.Field(
        description="Concise summary of what the issue is (1-2 sentences)"
    )
    analysis: IssueAnalysis = pydantic.Field(
        description="Detailed evidence and root cause analysis"
    )


class SessionIssueAnalysis(pydantic.BaseModel):
    """Schema for complete issue analysis results."""

    issues: list[IssueDescription] = pydantic.Field(
        description="List of distinct issues identified in the session"
    )


def _collect_negative_feedback(session: list[Trace], model: str | None = None) -> str:
    """
    Collect and summarize all negative feedback from traces in the session using LLM.

    Args:
        session: List of traces from the same session.
        model: Optional model URI to use for feedback analysis. If None, uses default model.

    Returns:
        String summary of all negative feedback with trace IDs in parentheticals.
    """
    if model is None:
        from mlflow.genai.judges.utils import get_default_model

        model = get_default_model()

    # First, collect all negative feedback
    negative_feedback_list = []

    for trace in session:
        trace_id = trace.info.trace_id

        for assessment in trace.info.assessments:
            if not isinstance(assessment, Feedback):
                continue

            if _is_negative_feedback(assessment, model=model):
                negative_feedback_list.append((trace_id, assessment))

    if not negative_feedback_list:
        return ""

    # Build detailed feedback context for LLM
    feedback_details = "Negative feedback found:\n\n"
    for trace_id, feedback in negative_feedback_list:
        feedback_details += f"Trace ID: {trace_id}\n"
        feedback_details += f"Feedback name: {feedback.name}\n"
        feedback_details += f"Value: {feedback.value}\n"
        if feedback.rationale:
            feedback_details += f"Rationale: {feedback.rationale}\n"
        feedback_details += "\n"

    # Use LLM to summarize
    system_message = ChatMessage(
        role="system",
        content=(
            "You are an expert at summarizing feedback for AI systems. "
            "Summarize the negative feedback concisely, grouping similar feedback together. "
            "IMPORTANT: You MUST include trace ID citations in parentheses after each point. "
            "Format: 'Feedback summary (trace ID: abc123)' or "
            "'Feedback summary (trace IDs: abc123, def456)' for multiple traces with "
            "similar feedback."
        ),
    )

    user_message = ChatMessage(
        role="user",
        content=(
            f"Summarize this negative feedback with parenthetical trace ID citations:\n\n"
            f"{feedback_details}"
        ),
    )

    result = get_chat_completions_with_structured_output(
        model_uri=model,
        messages=[system_message, user_message],
        output_schema=FeedbackSummary,
    )
    _logger.info(f"Feedback summary: {result.summary}")
    return result.summary


class FeedbackAnalysis(pydantic.BaseModel):
    """Schema for LLM analysis of whether feedback is negative."""

    is_negative: bool = pydantic.Field(
        description="True if the feedback indicates a problem, issue, or negative outcome"
    )
    reasoning: str = pydantic.Field(
        description="Brief explanation of why this feedback is considered negative or positive"
    )


def _is_negative_feedback(feedback: Feedback, model: str | None = None) -> bool:
    """
    Determine if feedback is negative using an LLM.

    Args:
        feedback: Feedback assessment to check.
        model: Optional model URI to use for analysis. If None, uses default model.

    Returns:
        True if the feedback is considered negative, False otherwise.
    """
    if model is None:
        from mlflow.genai.judges.utils import get_default_model

        model = get_default_model()

    feedback_info = f"""
Feedback name: {feedback.name}
Feedback value: {feedback.value}
Feedback rationale: {feedback.rationale or "Not provided"}
"""

    system_message = ChatMessage(
        role="system",
        content=(
            "You are an expert at analyzing feedback assessments for AI systems. "
            "Determine if the given feedback indicates a negative outcome, problem, or issue."
        ),
    )

    user_message = ChatMessage(
        role="user",
        content=f"Analyze this feedback and determine if it's negative:\n\n{feedback_info}",
    )

    analysis = get_chat_completions_with_structured_output(
        model_uri=model,
        messages=[system_message, user_message],
        output_schema=FeedbackAnalysis,
    )
    _logger.debug(
        f"Feedback '{feedback.name}' with value '{feedback.value}' "
        f"analyzed as {'negative' if analysis.is_negative else 'positive'}: "
        f"{analysis.reasoning}"
    )
    return analysis.is_negative


def _find_issues_in_session(
    session: list[Trace],
    model: str | None = None,
) -> list[Issue]:
    """
    Find and analyze issues in a session based on negative feedback.

    Analyzes traces with negative feedback to root cause underlying issues. Root cause
    analysis may include examining span statuses and errors, analyzing inputs and outputs,
    identifying failed components, correlating feedback with trace execution details, and
    other investigative techniques.

    Args:
        session: List of Trace objects representing a session.
        model: Optional model URI to use for feedback analysis and issue detection.
               If None, uses default model.

    Returns:
        List of Issue objects found in the session. Returns empty list if no
        negative feedback is found.

    Raises:
        MlflowException: If session is empty or invalid.
    """
    if not session:
        raise MlflowException(
            "Session must contain at least one trace", error_code=INVALID_PARAMETER_VALUE
        )

    if not isinstance(session, list):
        raise MlflowException(
            f"Session must be a list of Trace objects, got {type(session).__name__}",
            error_code=INVALID_PARAMETER_VALUE,
        )

    for i, trace in enumerate(session):
        if not isinstance(trace, Trace):
            raise MlflowException(
                f"All elements in session must be Trace objects, "
                f"got {type(trace).__name__} at index {i}",
                error_code=INVALID_PARAMETER_VALUE,
            )

    # Step 1: Extract conversation history with trace IDs
    conversation = _extract_conversation_with_trace_ids(session)

    # Step 2: Collect and summarize all negative feedback with trace ID parentheticals
    feedback_summary = _collect_negative_feedback(session, model=model)

    if not feedback_summary:
        return []

    # Step 3: Use LLM with trace inspection tools to identify and analyze issues
    # Get wrapped trace tools that accept trace_id as a parameter
    wrapped_tools = _get_wrapped_trace_tools()

    # Build context for the LLM
    trace_ids = [trace.info.trace_id for trace in session]

    conversation_context = "Conversation history:\n"
    for i, turn in enumerate(conversation, 1):
        role = turn["role"]
        content = turn["content"]
        trace_id = turn["trace_id"]
        # Truncate long content for readability
        content_preview = content[:200] + "..." if len(content) > 200 else content
        conversation_context += f"  Turn {i} ({role}, trace {trace_id}): {content_preview}\n"

    system_message = ChatMessage(
        role="system",
        content=(
            "You are an expert at analyzing AI system traces to identify root causes of issues. "
            "You have been given a summary of negative feedback and a conversation history. "
            "Your task is to explore the traces using the available tools to identify the "
            "underlying issues that caused the negative feedback. "
            "\n\n"
            "In order to identify issues precisely and correctly, you must think "
            "methodically and explain your reasoning before taking actions and act "
            "step-by-step. Your goal is to identify evidence-based root causes that are as "
            "specific as possible. There must be evidence backing the root causes; if you do "
            "not have evidence, more general root causes are acceptable."
            "You MUST follow these guidelines:\n\n"
            "1. You MUST use the tools provided to you to analyze traces in detail before "
            "confirming that there's an issue. This includes reading information about a "
            "trace, feedback on a trace, analyzing span details, and examining inputs and "
            "outputs.\n"
            "2. You must carefully read and analyze the information you've gathered from the "
            "traces\n."
            "3. Think critically about whether you have enough information to confirm that an "
            "issue occurred. If you don't have enough information, use the tools to gather "
            "more evidence.\n"
            "4. Think critically about whether you've root caused the issue as deeply as "
            "possible with the evidence you have. If you don't have evidence for a specific "
            "root cause, try to verify a more general root cause using the evidence. Only "
            "return issues that you can confirm with evidence from the traces.\n"
            "\n\n"
            f"Available trace IDs to inspect: {trace_ids}"
        ),
    )

    user_message = ChatMessage(
        role="user",
        content=(
            f"===== Negative Feedback Summary ====="
            f"{feedback_summary}\n\n"
            f"===== Conversation History =====\n\n"
            f"{conversation_context}\n\n"
            "Please explore the traces using the available tools to identify issues that "
            "caused the negative feedback. Provide a detailed analysis of what went wrong."
        ),
    )

    if model is None:
        from mlflow.genai.judges.utils import get_default_model

        model = get_default_model()

    # Call LLM with wrapped tools to analyze traces
    analysis = get_chat_completions_with_structured_output(
        model_uri=model,
        messages=[system_message, user_message],
        output_schema=SessionIssueAnalysis,
        tools=wrapped_tools,
    )

    # Convert Pydantic models to dataclass instances
    issues = []
    for issue_desc in analysis.issues:
        citations = [
            Citation(
                trace_id=cite.trace_id,
                span_id=cite.span_id,
                content=cite.content,
            )
            for cite in issue_desc.analysis.citations
        ]
        analysis_obj = Analysis(
            evidence=issue_desc.analysis.evidence,
            root_cause=issue_desc.analysis.root_cause,
            citations=citations,
        )
        issues.append(Issue(description=issue_desc.description, analysis=analysis_obj))

    _logger.info(f"Found {len(issues)} issues through trace analysis")
    return issues
