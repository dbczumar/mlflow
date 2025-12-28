"""
Issue Judge for detecting issues in traces.

This module provides functionality to create LLM judges that detect whether
traces contain specific issues identified by the Trace Insights system.
"""

import logging
from dataclasses import asdict
from typing import Any

import pydantic
from pydantic import PrivateAttr

import mlflow
from mlflow.entities.assessment import Issue as IssueAssessment
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import Judge, JudgeField
from mlflow.genai.judges.constants import _RATIONALE_FIELD_DESCRIPTION
from mlflow.genai.judges.utils import (
    invoke_judge_model,
    validate_judge_model,
)
from mlflow.genai.scorers.base import (
    _SERIALIZATION_VERSION,
    ScorerKind,
    SerializedScorer,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)

# Fixed model for issue judges - no configuration needed
_ISSUE_JUDGE_MODEL = "openai:/gpt-5"
_DEFAULT_MAX_REFERENCE_TRACES = 5

# Meta-prompt for generating the issue judge prompt
_PROMPT_GENERATOR_SYSTEM = """You are an expert at creating evaluation prompts for LLM judges that detect quality issues in traces.

Your task is to generate a comprehensive evaluation prompt that will help another LLM detect a specific issue in new traces.

You have access to tools to inspect reference traces that exhibit this issue. Use these tools to understand:
1. What makes these traces problematic
2. What specific patterns or signals indicate this issue
3. What relevant information should be extracted from traces to detect this issue

Based on the issue type, determine what information is most relevant:
- For latency/performance issues: Include span hierarchy, timing data, bottlenecks
- For output quality issues: Include inputs, outputs, and response content
- For error/failure issues: Include error messages, stack traces, status codes
- For tool/function call issues: Include tool names, arguments, results
- For conversation/context issues: Include message history, context handling

Generate a structured prompt following the exact format below."""

_PROMPT_GENERATOR_USER = """## Issue to Analyze
**Issue ID**: {issue_id}
**Issue Name**: {issue_name}
**Issue Description**: {issue_description}

## Reference Trace IDs
These traces have been confirmed to exhibit this issue:
{trace_ids_list}

## Instructions
1. Use the available tools to inspect each reference trace
2. Identify what specific patterns or signals indicate this issue
3. For each trace, extract the MOST RELEVANT information based on the issue type
4. Generate an evaluation prompt in the following JSON format:

```json
{{
    "issue_section": "A clear description of the issue and what to look for. 3~5 sentences.",
    "reference_traces": [
        {{
            "trace_id": "the trace ID",
            "summary": "The most relevant information for detecting this issue in this trace. Should be detailed, but do not include unrelated information to the evaluation."
        }}
    ],
    "task_definition": "Clear instructions for the evaluator on how to determine if a trace has this issue",
    "detection_criteria": "Specific criteria or patterns that indicate this issue is present"
}}
```

Important: The "summary" field for each trace should contain information that is SPECIFICALLY relevant to detecting this type of issue. For example:
- For latency issues: span timing breakdown, slowest operations
- For quality issues: input/output content, response quality indicators
- For error issues: error details, failure points

Now analyze the traces and generate the prompt."""


class IssueJudge(Judge):
    """
    A judge that detects whether traces contain a specific issue.

    This judge is created from an Issue entity and evaluates traces to determine
    if they exhibit the same issue. Unlike regular judges that return Feedback
    assessments, IssueJudge returns Issue assessments when an issue is detected.
    """

    _issue_id: str = PrivateAttr()
    _issue_name: str = PrivateAttr()
    _system_prompt: str = PrivateAttr()
    _model: str = PrivateAttr()

    def __init__(
        self,
        issue_id: str,
        issue_name: str,
        system_prompt: str,
        model: str | None = None,
        **kwargs,
    ):
        """
        Initialize the IssueJudge.

        Args:
            issue_id: The unique identifier of the issue to detect
            issue_name: Human-readable name of the issue
            system_prompt: The system prompt for issue detection
            model: The model to use for evaluation (defaults to databricks-gpt-5)
        """
        # Name the scorer after the issue
        name = f"issue-detector-{issue_id[:8]}"
        description = f"Detects whether traces exhibit the issue: {issue_name}"

        super().__init__(name=name, description=description, aggregations=[], **kwargs)

        if not issue_id:
            raise MlflowException(
                "issue_id must be a non-empty string",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if not issue_name:
            raise MlflowException(
                "issue_name must be a non-empty string",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if not system_prompt:
            raise MlflowException(
                "system_prompt must be a non-empty string",
                error_code=INVALID_PARAMETER_VALUE,
            )

        self._issue_id = issue_id
        self._issue_name = issue_name
        self._system_prompt = system_prompt
        self._model = model or _ISSUE_JUDGE_MODEL

        self._validate_model_format()

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.INSTRUCTIONS

    @property
    def instructions(self) -> str:
        """Plain text instructions of what this judge evaluates."""
        return self._system_prompt

    @property
    def issue_id(self) -> str:
        """Get the issue ID this judge detects."""
        return self._issue_id

    @property
    def issue_name(self) -> str:
        """Get the issue name."""
        return self._issue_name

    @property
    def model(self) -> str:
        """Get the model for this judge."""
        return self._model

    def get_input_fields(self) -> list[JudgeField]:
        """Get the input fields for this judge."""
        return [JudgeField(name="trace_id", description="ID of the trace to evaluate for the issue")]

    def get_output_fields(self) -> list[JudgeField]:
        """Get the output fields for this judge."""
        # Use "result" as the field name to be compatible with invoke_judge_model
        return [
            JudgeField(
                name="result",
                description="Whether the trace exhibits this issue (true/false)",
                value_type=bool,
            ),
            JudgeField(
                name="rationale",
                description=_RATIONALE_FIELD_DESCRIPTION,
                value_type=str,
            ),
        ]

    def _build_user_message(self, trace: Trace) -> str:
        """Build the user message with the trace to evaluate."""
        return f"Trace ID to evaluate: {trace.info.trace_id}"

    def __call__(
        self,
        *,
        trace: Trace,
    ) -> IssueAssessment | None:
        """
        Evaluate a trace to determine if it contains the issue.

        Args:
            trace: The trace to evaluate

        Returns:
            An Issue assessment if the issue is detected, None otherwise
        """
        if not isinstance(trace, Trace):
            raise MlflowException(
                f"'trace' must be a Trace object, got {type(trace).__name__}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        user_content = self._build_user_message(trace)

        from mlflow.types.llm import ChatMessage

        messages = [
            ChatMessage(role="system", content=self._system_prompt),
            ChatMessage(role="user", content=user_content),
        ]

        response_format = self._create_response_format_model()

        # Invoke the judge model - this returns a Feedback, but we'll convert it
        feedback = invoke_judge_model(
            model_uri=self._model,
            prompt=messages,
            assessment_name=self.name,
            trace_id=trace.info.trace_id,
            response_format=response_format,
            use_case="issue_detection",
        )

        # Parse the response to determine if issue was detected
        # The feedback.value contains the "result" field (boolean)
        # The feedback.rationale contains the explanation
        has_issue = bool(feedback.value) if feedback.value is not None else False
        rationale = feedback.rationale or ""

        return IssueAssessment(
            issue_id=self._issue_id,
            issue_name=self._issue_name,
            value=has_issue,
            trace_id=trace.info.trace_id if trace.info else None,
            rationale=rationale,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=self.name,
            ),
        )

    def _create_response_format_model(self) -> type[pydantic.BaseModel]:
        """Create a Pydantic model for structured output."""
        output_fields = self.get_output_fields()

        fields = {}
        for field in output_fields:
            fields[field.name] = (
                field.value_type,
                pydantic.Field(description=field.description),
            )

        return pydantic.create_model("IssueDetectionResponse", **fields)

    def _validate_model_format(self) -> None:
        """Validate that the model is in a valid format."""
        validate_judge_model(self._model)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Override model_dump to serialize as a SerializedScorer with issue metadata."""
        pydantic_data = {
            "issue_id": self._issue_id,
            "issue_name": self._issue_name,
            "system_prompt": self._system_prompt,
            "model": self._model,
        }

        serialized_scorer = SerializedScorer(
            name=self.name,
            description=self.description,
            aggregations=self.aggregations,
            is_session_level_scorer=False,
            mlflow_version=mlflow.__version__,
            serialization_version=_SERIALIZATION_VERSION,
            # Store issue judge data in instructions_judge_pydantic_data
            # with a special marker to identify it as an issue judge
            instructions_judge_pydantic_data={
                "is_issue_judge": True,
                **pydantic_data,
            },
            builtin_scorer_class=None,
            builtin_scorer_pydantic_data=None,
            call_source=None,
            call_signature=None,
            original_func_name=None,
        )
        return asdict(serialized_scorer)

    def __repr__(self) -> str:
        """Return string representation of the IssueJudge."""
        prompt_preview = self._system_prompt[:50] + "..." if len(self._system_prompt) > 50 else self._system_prompt
        return (
            f"IssueJudge(issue_id='{self._issue_id}', "
            f"issue_name='{self._issue_name}', "
            f"model='{self._model}', "
            f"system_prompt='{prompt_preview}')"
        )


class _ReferenceTraceInfo(pydantic.BaseModel):
    """Information about a reference trace for the generated prompt."""
    trace_id: str = pydantic.Field(description="The trace ID")
    summary: str = pydantic.Field(description="Relevant summary for detecting this issue")


class _GeneratedPromptData(pydantic.BaseModel):
    """Structure of the LLM-generated prompt data."""
    issue_section: str = pydantic.Field(description="Description of the issue and what to look for")
    reference_traces: list[_ReferenceTraceInfo] = pydantic.Field(description="Reference traces with relevant summaries")
    task_definition: str = pydantic.Field(description="Instructions for the evaluator")
    detection_criteria: str = pydantic.Field(description="Specific criteria indicating the issue")


def _format_generated_prompt(prompt_data: _GeneratedPromptData) -> str:
    """Format the LLM-generated prompt data into a final prompt string."""
    lines = [
        "You are an expert at detecting quality issues in LLM agent traces.",
        "",
        "## Important: Avoid False Positives",
        "Your primary goal is PRECISION over recall. It is much better to miss an issue than to incorrectly flag one.",
        "- Only flag an issue when you have CLEAR, STRONG evidence that matches the specific patterns described below",
        "- If you are unsure or the evidence is ambiguous, return `result: false`",
        "- The trace must exhibit the SAME problematic pattern as the reference traces, not just superficial similarities",
        "- Do NOT flag based on speculation or potential issues - only flag based on concrete evidence in the trace",
        "",
        "## Issue to Detect",
        prompt_data.issue_section,
        "",
        "## Reference Traces",
        "The following traces have been confirmed to exhibit this issue. "
        "Use these as examples of what the issue looks like - the trace you evaluate must show similar concrete problems:",
        "",
    ]

    for i, ref_trace in enumerate(prompt_data.reference_traces, 1):
        lines.extend([
            f"### Reference Trace {i}",
            f"- **Trace ID**: {ref_trace.trace_id}",
            f"- **Relevant Summary**:",
            ref_trace.summary,
            "",
        ])

    lines.extend([
        "## Task Definition",
        prompt_data.task_definition,
        "",
        "**Detection Criteria**:",
        prompt_data.detection_criteria,
        "",
        "## Response Format",
        "You must respond with:",
        "- `result`: true ONLY if the trace clearly exhibits this issue with strong evidence, false otherwise",
        "- `rationale`: A brief explanation of your decision, referencing specific evidence from the trace. "
        "If returning false, briefly explain why the trace does NOT have this issue or why evidence is insufficient.",
    ])

    return "\n".join(lines)


def _generate_prompt_with_llm(
    issue_id: str,
    issue_name: str,
    issue_description: str,
    trace_ids: list[str],
    model: str | None = None,
) -> str:
    """
    Use LLM with tools to generate the issue detection prompt.

    Args:
        issue_id: The issue ID
        issue_name: The issue name
        issue_description: The issue description
        trace_ids: List of reference trace IDs
        model: The model URI to use for prompt generation (e.g., "openai:/gpt-4").
               If not provided, uses the default issue judge model.

    Returns:
        The generated prompt string
    """
    import json

    from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm_and_handle_tools
    from mlflow.metrics.genai.model_utils import _parse_model_uri
    from mlflow.types.llm import ChatMessage

    # Parse the model URI to get provider and model name
    model_uri = model or _ISSUE_JUDGE_MODEL
    provider, model_name = _parse_model_uri(model_uri)

    # Build the user message with trace IDs
    trace_ids_formatted = "\n".join(f"- {tid}" for tid in trace_ids)

    user_message = _PROMPT_GENERATOR_USER.format(
        issue_id=issue_id,
        issue_name=issue_name,
        issue_description=issue_description or "(No description provided)",
        trace_ids_list=trace_ids_formatted if trace_ids else "(No reference traces available)",
    )

    messages = [
        ChatMessage(role="system", content=_PROMPT_GENERATOR_SYSTEM),
        ChatMessage(role="user", content=user_message),
    ]

    # Use the first trace ID for tool access (if available)
    first_trace_id = trace_ids[0] if trace_ids else None

    # Invoke LLM with tool access to inspect traces
    response, _ = _invoke_litellm_and_handle_tools(
        provider=provider,
        model_name=model_name,
        messages=messages,
        trace_id=first_trace_id,
        num_retries=3,
        response_format=_GeneratedPromptData,
    )

    # Parse the JSON response
    try:
        # Clean up markdown code blocks if present
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        prompt_data = _GeneratedPromptData.model_validate_json(cleaned)
        return _format_generated_prompt(prompt_data)
    except (json.JSONDecodeError, pydantic.ValidationError) as e:
        _logger.warning(f"Failed to parse LLM-generated prompt: {e}. Using raw response.")
        # Fall back to a basic prompt if parsing fails
        return f"""You are an expert at detecting quality issues in LLM agent traces.

## Issue to Detect
**Name**: {issue_name}
**Description**: {issue_description}

## Your Task
Analyze the provided trace and determine if it exhibits this issue.

You must respond with:
- `result`: true if the trace has this issue, false otherwise
- `rationale`: A brief explanation of your decision, referencing specific evidence from the trace
"""


@experimental(version="3.10.0")
def make_judge_from_issue(
    issue_id: str,
    max_reference_traces: int = _DEFAULT_MAX_REFERENCE_TRACES,
    model: str | None = None,
    prompt_writer_model: str | None = None,
) -> IssueJudge:
    """
    Create an LLM judge that detects whether traces contain a specific issue.

    This function fetches the issue entity and its linked traces (evidence),
    uses an LLM to analyze the traces and generate an appropriate detection prompt,
    then creates an IssueJudge configured to detect that issue in new traces.

    The LLM-generated prompt includes:
    1. Issue to Detect - description of the issue and what to look for
    2. Reference Traces - with trace ID, latency, state, and a relevant summary
       tailored to the issue type (e.g., span hierarchy for latency issues,
       output content for quality issues)
    3. Task Definition - clear instructions and detection criteria

    Args:
        issue_id: The ID of the issue to detect
        max_reference_traces: Maximum number of reference traces to include
            in the prompt for few-shot learning (default: 5)
        model: The model identifier to use for issue detection evaluation
            (e.g., "openai:/gpt-4"). If not provided, uses the default model.
        prompt_writer_model: The model identifier to use for generating the
            issue detection prompt. If not provided, defaults to the value of
            ``model``. This allows using a different (potentially more capable)
            model for authoring the judge prompt while using a faster/cheaper
            model for the actual issue detection.

    Returns:
        An IssueJudge instance configured to detect the specified issue

    Raises:
        MlflowException: If the issue is not found or cannot be accessed

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.judges import make_judge_from_issue

            # Create a judge from an existing issue
            judge = make_judge_from_issue(issue_id="issue-123")

            # Create a judge with a custom model
            judge = make_judge_from_issue(
                issue_id="issue-123",
                model="openai:/gpt-4",
            )

            # Use different models for prompt generation vs detection
            judge = make_judge_from_issue(
                issue_id="issue-123",
                model="openai:/gpt-4.1-mini",  # Faster model for detection
                prompt_writer_model="openai:/gpt-4",  # Better model for prompt authoring
            )

            # Evaluate traces for this issue
            traces = mlflow.search_traces(experiment_ids=["1"], return_type="list")
            for trace in traces:
                assessment = judge(trace=trace)
                if assessment:
                    print(f"Issue detected in trace {trace.info.trace_id}")
                    mlflow.log_assessment(trace.info.trace_id, assessment)
    """
    client = mlflow.MlflowClient()

    # Fetch the issue entity
    issue = client.get_issue(issue_id)
    if not issue:
        raise MlflowException.invalid_parameter_value(
            f"Issue with ID '{issue_id}' not found"
        )

    # Fetch traces that have this issue (reference traces)
    trace_ids = []
    try:
        # Search for traces with this issue assessment
        filter_string = f"issue.`{issue_id}` = 'true'"
        traces = mlflow.search_traces(
            locations=[issue.experiment_id],
            filter_string=filter_string,
            max_results=max_reference_traces,
            return_type="list",
        )

        trace_ids = [
            trace.info.trace_id
            for trace in traces
            if trace.info and trace.info.trace_id
        ]
    except Exception as e:
        _logger.warning(f"Could not fetch reference traces for issue {issue_id}: {e}")

    # Determine the model for prompt generation
    # If prompt_writer_model is specified, use it; otherwise fall back to model or default
    writer_model = prompt_writer_model or model

    # Generate the prompt using LLM with tool access
    system_prompt = _generate_prompt_with_llm(
        issue_id=issue.issue_id,
        issue_name=issue.name,
        issue_description=issue.description or "",
        trace_ids=trace_ids,
        model=writer_model,
    )

    # Determine the model for the judge
    judge_model = model or _ISSUE_JUDGE_MODEL

    # Create and return the IssueJudge
    return IssueJudge(
        issue_id=issue.issue_id,
        issue_name=issue.name,
        system_prompt=system_prompt,
        model=judge_model,
    )


__all__ = ["IssueJudge", "make_judge_from_issue"]
