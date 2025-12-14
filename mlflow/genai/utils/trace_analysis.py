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
        title: Brief title following pattern "X occurred due to Y".
        context: Context about the user's query or environment where issue occurred.
        description: Concise summary of what the issue is.
        root_causes: List of contributing root causes with evidence and citations.
    """

    title: str
    context: str
    description: str
    root_causes: list[Analysis]


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


class IssueClarification(pydantic.BaseModel):
    """Schema for LLM-clarified issue with concise title and refined description."""

    title: str = pydantic.Field(
        description=(
            "Concise title focusing on WHAT happened, without explaining WHY. "
            "Should be brief and action-focused. "
            "GOOD examples (focus on what): "
            "'Agent deleted all saved memories without confirmation', "
            "'Agent provided incorrect stock information', "
            "'Agent asked to save a memory instead of providing instructions'. "
            "BAD examples (too verbose, includes why): "
            "'Agent deleted memories due to missing confirmation and speed instructions', "
            "'Agent provided incorrect stock information due to missing tool call parameters', "
            "'User received unhelpful response due to failed database query'."
        )
    )
    description: str = pydantic.Field(
        description=(
            "Refined description incorporating WHY the issue occurred based on root causes. "
            "Should maintain the original issue description's clarity while weaving in "
            "high-level explanations from root causes to provide context. Keep it concise "
            "(2-3 sentences). Focus on what happened and why it matters, incorporating the "
            "key contributing factors without losing information from the original description."
        )
    )


class IssueClarificationList(pydantic.BaseModel):
    """Schema for batch clarification of multiple issues."""

    issues: list[IssueClarification] = pydantic.Field(
        description=(
            "List of clarified issues with concise titles and refined descriptions. "
            "Each issue should have a distinct, non-overlapping title that clearly "
            "differentiates it from other issues in the list."
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


class RootCauseAnalysis(pydantic.BaseModel):
    """Schema for a single root cause with evidence and citations."""

    root_cause: str = pydantic.Field(
        description=(
            "Root cause analysis explaining one contributing factor to why the issue occurred"
        )
    )
    evidence: str = pydantic.Field(
        description="Description of the evidence supporting this specific root cause"
    )
    citations: list[IssueCitation] = pydantic.Field(
        description="Specific citations with content from traces supporting this root cause"
    )


class IssueDescription(pydantic.BaseModel):
    """Schema for a single issue identified in the session."""

    context: str = pydantic.Field(
        description=(
            "Context explaining the type/nature of the query or environment in which the "
            "issue occurred. This MUST describe what the query was trying to accomplish or "
            "what scenario the agent was in, NOT system processing steps or internal "
            "component behavior. "
            "GOOD examples (describe a scenario): "
            "'User asked about S&P 500 status', "
            "'Agent was asked about holiday schedule'. "
            "'Documents about HR policies were submitted to agent for processing'. "
            "BAD examples (describe system processing - DO NOT USE): "
            "'After receiving tool outputs that look like live data, ...', "
            "'When the agent processed the financial data, ...', "
            "'Following the database query failure, ...'."
        )
    )
    description: str = pydantic.Field(
        description=(
            "Concise summary of what the issue is and why it matters (1-2 sentences). "
            "Should describe the problem and its impact."
        )
    )
    root_causes: list[RootCauseAnalysis] = pydantic.Field(
        description=(
            "List of likely contributing root causes for this issue. Each root cause should "
            "represent a distinct contributing factor with its own evidence and citations. "
            "If multiple factors contributed to the same problem, list them all here rather "
            "than creating separate issues."
        )
    )


class SessionIssueAnalysis(pydantic.BaseModel):
    """Schema for complete issue analysis results."""

    guidelines_adherence_explanation: str = pydantic.Field(
        description=(
            "Detailed explanation of how you followed each guideline in your analysis process. "
            "For each guideline, explicitly describe the actions you took and evidence "
            "you gathered to adhere to it."
        )
    )
    issues: list[IssueDescription] = pydantic.Field(
        description="List of distinct issues identified in the session"
    )


def _collect_negative_feedback(
    session: list[Trace],
    model: str | None = None,
    model_params: dict[str, Any] | None = None,
) -> str:
    """
    Collect and summarize all negative feedback from traces in the session using LLM.

    Args:
        session: List of traces from the same session.
        model: Optional model URI to use for feedback analysis. If None, uses default model.
        model_params: Optional dictionary of model parameters to pass to the LLM.

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

            if _is_negative_feedback(assessment, model=model, model_params=model_params):
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
        inference_params=model_params,
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


def _clarify_issues(
    issues_data: list[tuple[str, str, list[tuple[str, str]]]],
    model: str | None = None,
    model_params: dict[str, Any] | None = None,
) -> list[IssueClarification]:
    """
    Clarify and refine multiple issues by generating concise titles and improved descriptions.

    Processes all issues together so the LLM can ensure titles are distinct and non-overlapping.
    Uses an LLM to create shorter, more focused titles that describe "what" happened,
    and refines descriptions to incorporate the "why" details from root causes.

    Args:
        issues_data: List of tuples (context, description, root_causes) for each issue,
                     where root_causes is a list of (root_cause, evidence) tuples.
        model: Optional model URI to use. If None, uses default model.
        model_params: Optional dictionary of model parameters to pass to the LLM.

    Returns:
        List of IssueClarification objects with concise titles and refined descriptions.
    """
    if model is None:
        from mlflow.genai.judges.utils import get_default_model

        model = get_default_model()

    # Build detailed issue information for all issues
    all_issues_text = ""
    for i, (context, description, root_causes) in enumerate(issues_data, 1):
        root_causes_text = ""
        for j, (root_cause, evidence) in enumerate(root_causes, 1):
            root_causes_text += f"\n  Root Cause {j}: {root_cause}\n  Evidence {j}: {evidence}\n"

        all_issues_text += f"""
Issue {i}:
---------
Context: {context}

Description: {description}
{root_causes_text}
"""

    system_message = ChatMessage(
        role="system",
        content=(
            "You are an expert at clarifying and refining AI system issue descriptions. "
            "Your task is to create concise titles and refined descriptions for multiple issues. "
            "\n\n"
            "IMPORTANT: You are processing ALL issues at once, so you MUST ensure that each "
            "issue's title is DISTINCT and clearly differentiates it from other issues. "
            "If two issues are related, make sure their titles highlight what makes each unique. "
            "\n\n"
            "TITLE: Focus on WHAT happened, not WHY. Keep it brief and action-focused. "
            "Do NOT include explanations of causes. The title should be focused on the "
            "agent, not the user (e.g., 'agent did X', not 'user received Y'). "
            "Always use the term 'agent', not synonyms like 'system', 'model', or 'AI'. "
            "\n\n"
            "DESCRIPTION: Refine the provided description by incorporating WHY the issue "
            "occurred based on the root causes. Maintain the original meaning but weave in "
            "high-level explanations from root causes to provide context. Keep it concise "
            "(2-3 sentences). You MUST NOT lose ANY information from the original description."
        ),
    )

    user_message = ChatMessage(
        role="user",
        content=f"Clarify and refine these issues:\n\n{all_issues_text}",
    )

    result = get_chat_completions_with_structured_output(
        model_uri=model,
        messages=[system_message, user_message],
        output_schema=IssueClarificationList,
        inference_params=model_params,
    )
    return result.issues


def _is_negative_feedback(
    feedback: Feedback,
    model: str | None = None,
    model_params: dict[str, Any] | None = None,
) -> bool:
    """
    Determine if feedback is negative using an LLM.

    Args:
        feedback: Feedback assessment to check.
        model: Optional model URI to use for analysis. If None, uses default model.
        model_params: Optional dictionary of model parameters to pass to the LLM.

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
        inference_params=model_params,
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
    context: str | None = None,
    model_params: dict[str, Any] | None = None,
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
        context: Optional context about the agent/use case to help characterize issues
                more accurately (e.g., "This is a customer support chatbot that helps
                users with product questions").
        model_params: Optional dictionary of model parameters (e.g., temperature, top_p,
                     max_tokens) to pass to the LLM for all completion calls.

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
    feedback_summary = _collect_negative_feedback(session, model=model, model_params=model_params)

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

    # Build system message with context awareness
    system_content = (
        "You are an expert at analyzing AI system traces to identify root causes of issues. "
        "You have been given a summary of negative feedback and a conversation history. "
    )

    if context:
        system_content += (
            "You have also been provided with context about the agent/use case. Use this "
            "context to better understand what the agent is supposed to do and to identify "
            "when behavior deviates from expectations. "
        )

    system_content += (
        "Your task is to explore the traces using the available tools to identify the "
        "top underlying issues that caused the negative feedback."
        "\n\n"
        "In order to identify issues precisely and correctly, you must think "
        "methodically and explain your reasoning before taking actions and act "
        "step-by-step. Your goal is to identify evidence-based root causes that are as "
        "specific as possible. There must be evidence backing the root causes; if you do "
        "not have evidence, more general root causes are acceptable."
        "IMPORTANT: You MUST follow these guidelines AND explain your reasoning about "
        "how they're met:\n\n"
        "1. You MUST use the tools provided to you to analyze traces in detail before "
        "confirming that there's an issue. This includes reading information about a "
        "trace and feedback on a trace, reading the full list of spans executed, "
        "reading relevant span details including inputs and outputs, and listing all "
        "tool calls & analyzing relevant ones deeply. It's okay if this takes time - "
        "be thorough!\n"
        "2. You must carefully read and analyze the information you've gathered from the "
        "traces\n."
        "3. Think critically about whether you have enough information to confirm that an "
        "issue occurred. If you don't have enough information, use the tools to gather "
        "more evidence.\n"
        "4. Think critically about whether you've root caused the issue as deeply as "
        "possible with the evidence you have. If you don't have evidence for a specific "
        "root cause, try to verify a more general root cause using the evidence. Only "
        "return issues that you can confirm with evidence from the traces.\n"
        "5. Avoid speculation. If you don't have concrete evidence to confirm an issue, "
        "pursue a different root cause, broadening if necessary. For example, do not "
        "make claims about computer system / code base threading or process state if "
        "you do not have direct information about threads or processes from the traces."
        "Another example: do not assume that the agent has capability X unless you observe "
        "it in the traces or context (it's *strongly encouraged* to look for the capability by "
        "doing additional analysis of the context and traces if you are unsure!).\n"
        "6. IF the negative feedback is specific, issues MUST be directly relevant to the negative"
        " feedback.\n"
        "7. Issues MUST be INDEPENDENT. Do NOT return overlapping issues. "
        "CRITICAL patterns that signal NON-independence: "
        "(a) If one issue's context describes a SCENARIO (e.g., 'User asked about X') and "
        "another's context describes SYSTEM PROCESSING (e.g., 'After receiving tool outputs...', "
        "'When the orchestrator processed...'), they are likely the SAME issue - one describes "
        "the problem, the other describes an internal root cause. MERGE them into ONE issue. "
        "(b) If Issue A is 'X was missing' and Issue B is 'System didn't handle missing X', "
        "these are ONE issue with root cause 'Y occurred due to missing X and system not "
        "handling it'. "
        "Examples: "
        "BAD (two non-independent issues): 'Tool missing metadata' + 'Agent didn't compensate "
        "for missing metadata'. "
        "GOOD (one issue): 'Agent provided incorrect status due to tool missing metadata and "
        "agent not validating/compensating'. "
        "Before responding, you MUST RIGOROUSLY VERIFY that each issue is independent AND is "
        "NOT a root cause of another issue. If you suspect non-independence, MERGE the issues "
        "and adjust, regroup, or continue the analysis process as necessary.\n"
        "8. Draw clear boundaries between distinct issues. Example: if an agent performed an "
        "operation incorrectly *and* without permission, that would be two distinct issues. "
        "Counterexample: if an agent returned an unhelpful response because information was "
        "missing from a tool call, the missing information would be part of the root cause of the "
        "unhelpful response, not a separate issue."
        "\n\n"
        f"Available trace IDs to inspect: {trace_ids}"
    )

    system_message = ChatMessage(role="system", content=system_content)

    # Build user message with optional context
    user_content = ""
    if context:
        user_content += f"===== Agent/Use Case Context =====\n{context}\n\n"

    user_content += (
        f"===== Negative Feedback Summary =====\n"
        f"{feedback_summary}\n\n"
        f"===== Conversation History =====\n\n"
        f"{conversation_context}\n\n"
        "Please explore the traces using the available tools to identify issues that "
        "caused the negative feedback. Provide a detailed analysis of what went wrong."
    )

    user_message = ChatMessage(role="user", content=user_content)

    if model is None:
        from mlflow.genai.judges.utils import get_default_model

        model = get_default_model()

    # Call LLM with wrapped tools to analyze traces
    analysis = get_chat_completions_with_structured_output(
        model_uri=model,
        messages=[system_message, user_message],
        output_schema=SessionIssueAnalysis,
        tools=wrapped_tools,
        inference_params=model_params,
    )

    # Log guidelines adherence explanation
    _logger.info(f"Guidelines adherence explanation:\n{analysis.guidelines_adherence_explanation}")

    # Convert Pydantic models to dataclass instances and prepare for batch clarification
    issues_for_clarification = []
    issues_root_causes_lists = []

    for issue_desc in analysis.issues:
        # Convert each root cause analysis to an Analysis dataclass
        root_causes_list = []
        root_cause_tuples = []

        for rc in issue_desc.root_causes:
            citations = [
                Citation(
                    trace_id=cite.trace_id,
                    span_id=cite.span_id,
                    content=cite.content,
                )
                for cite in rc.citations
            ]
            root_cause_obj = Analysis(
                evidence=rc.evidence,
                root_cause=rc.root_cause,
                citations=citations,
            )
            root_causes_list.append(root_cause_obj)
            root_cause_tuples.append((rc.root_cause, rc.evidence))

        # Store for batch clarification
        issues_for_clarification.append(
            (issue_desc.context, issue_desc.description, root_cause_tuples)
        )
        issues_root_causes_lists.append((issue_desc.context, root_causes_list))

    # Clarify all issues at once so LLM can ensure distinct titles
    clarifications = _clarify_issues(
        issues_data=issues_for_clarification,
        model=model,
        model_params=model_params,
    )

    # Build final Issue objects with clarified titles and descriptions
    issues = []
    for clarification, (context, root_causes_list) in zip(clarifications, issues_root_causes_lists):
        issues.append(
            Issue(
                title=clarification.title,
                context=context,
                description=clarification.description,
                root_causes=root_causes_list,
            )
        )

    _logger.info(f"Found {len(issues)} issues through trace analysis")
    return issues
