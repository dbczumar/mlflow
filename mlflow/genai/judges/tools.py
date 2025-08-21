"""
Tools that MLflow's judges can use when judging a trace.

This module provides utility functions and tools that judges can leverage
to analyze traces more effectively, including trace inspection utilities,
context extraction, and analysis helpers.
"""

from typing import Any

from mlflow.entities.trace import Trace
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
def extract_trace_inputs(trace: Trace) -> dict[str, Any]:
    """
    Extract input data from the root span of a trace.

    Args:
        trace: The trace object to extract inputs from

    Returns:
        Dictionary containing the input data from the trace's root span

    Example:
        >>> trace = mlflow.get_trace("trace_id")
        >>> inputs = extract_trace_inputs(trace)
        >>> print(inputs)
        {"question": "What is the capital of France?"}
    """
    if not trace or not trace.data or not trace.data.spans:
        return {}

    root_span = trace.data.spans[0]
    return root_span.inputs or {}


@experimental(version="3.4.0")
def extract_trace_outputs(trace: Trace) -> Any:
    """
    Extract output data from the root span of a trace.

    Args:
        trace: The trace object to extract outputs from

    Returns:
        The output data from the trace's root span

    Example:
        >>> trace = mlflow.get_trace("trace_id")
        >>> outputs = extract_trace_outputs(trace)
        >>> print(outputs)
        "The capital of France is Paris."
    """
    if not trace or not trace.data or not trace.data.spans:
        return None

    root_span = trace.data.spans[0]
    return root_span.outputs


@experimental(version="3.4.0")
def get_span_by_name(trace: Trace, span_name: str) -> Any | None:
    """
    Find the first span in a trace with the specified name.

    Args:
        trace: The trace object to search
        span_name: Name of the span to find

    Returns:
        The first span with the matching name, or None if not found

    Example:
        >>> trace = mlflow.get_trace("trace_id")
        >>> retrieval_span = get_span_by_name(trace, "retrieval")
    """
    if not trace or not trace.data or not trace.data.spans:
        return None

    for span in trace.data.spans:
        if span.name == span_name:
            return span
    return None


@experimental(version="3.4.0")
def get_spans_by_type(trace: Trace, span_type: str) -> list[Any]:
    """
    Find all spans in a trace with the specified type.

    Args:
        trace: The trace object to search
        span_type: Type of spans to find (e.g., "RETRIEVER", "LLM")

    Returns:
        List of spans with the matching type

    Example:
        >>> trace = mlflow.get_trace("trace_id")
        >>> llm_spans = get_spans_by_type(trace, "LLM")
    """
    if not trace or not trace.data or not trace.data.spans:
        return []

    matching_spans = []
    for span in trace.data.spans:
        if hasattr(span, "span_type") and span.span_type == span_type:
            matching_spans.append(span)
    return matching_spans


@experimental(version="3.4.0")
def extract_span_metadata(span: Any) -> dict[str, Any]:
    """
    Extract metadata from a span for analysis.

    Args:
        span: The span object to extract metadata from

    Returns:
        Dictionary containing span metadata including name, duration, status, etc.

    Example:
        >>> span = get_span_by_name(trace, "llm_call")
        >>> metadata = extract_span_metadata(span)
        >>> print(metadata["duration_ms"])
        1250
    """
    if not span:
        return {}

    metadata = {
        "name": getattr(span, "name", None),
        "span_id": getattr(span, "span_id", None),
        "parent_id": getattr(span, "parent_id", None),
        "start_time": getattr(span, "start_time_ns", None),
        "end_time": getattr(span, "end_time_ns", None),
        "duration_ms": getattr(span, "duration_ms", None),
        "status": getattr(span, "status", None),
        "span_type": getattr(span, "span_type", None),
    }

    # Remove None values
    return {k: v for k, v in metadata.items() if v is not None}


@experimental(version="3.4.0")
def calculate_trace_duration(trace: Trace) -> float | None:
    """
    Calculate the total duration of a trace in milliseconds.

    Args:
        trace: The trace object to analyze

    Returns:
        Total trace duration in milliseconds, or None if cannot be calculated

    Example:
        >>> trace = mlflow.get_trace("trace_id")
        >>> duration = calculate_trace_duration(trace)
        >>> print(f"Trace took {duration}ms")
        Trace took 1500.5ms
    """
    if not trace or not trace.data or not trace.data.spans:
        return None

    # Find the root span (first span) and use its duration
    root_span = trace.data.spans[0]
    if hasattr(root_span, "duration_ms") and root_span.duration_ms is not None:
        return root_span.duration_ms

    # Fallback: calculate from start and end times if available
    if (
        hasattr(root_span, "start_time_ns")
        and hasattr(root_span, "end_time_ns")
        and root_span.start_time_ns is not None
        and root_span.end_time_ns is not None
    ):
        duration_ns = root_span.end_time_ns - root_span.start_time_ns
        return duration_ns / 1_000_000  # Convert nanoseconds to milliseconds

    return None


@experimental(version="3.4.0")
def count_spans_by_type(trace: Trace) -> dict[str, int]:
    """
    Count the number of spans by their type in a trace.

    Args:
        trace: The trace object to analyze

    Returns:
        Dictionary mapping span types to their counts

    Example:
        >>> trace = mlflow.get_trace("trace_id")
        >>> counts = count_spans_by_type(trace)
        >>> print(counts)
        {"LLM": 2, "RETRIEVER": 1, "TOOL": 3}
    """
    if not trace or not trace.data or not trace.data.spans:
        return {}

    type_counts = {}
    for span in trace.data.spans:
        span_type = getattr(span, "span_type", "UNKNOWN")
        type_counts[span_type] = type_counts.get(span_type, 0) + 1

    return type_counts


@experimental(version="3.4.0")
def has_errors_in_trace(trace: Trace) -> bool:
    """
    Check if any span in the trace has an error status.

    Args:
        trace: The trace object to check

    Returns:
        True if any span has an error status, False otherwise

    Example:
        >>> trace = mlflow.get_trace("trace_id")
        >>> has_errors = has_errors_in_trace(trace)
        >>> if has_errors:
        ...     print("Trace contains errors")
    """
    if not trace or not trace.data or not trace.data.spans:
        return False

    for span in trace.data.spans:
        status = getattr(span, "status", None)
        if status and hasattr(status, "status_code"):
            # Check for ERROR status code (typically 2 in OpenTelemetry)
            if status.status_code == 2:
                return True

    return False


@experimental(version="3.4.0")
def extract_tool_calls(trace: Trace) -> list[dict[str, Any]]:
    """
    Extract information about tool calls from a trace.

    Args:
        trace: The trace object to analyze

    Returns:
        List of dictionaries containing tool call information

    Example:
        >>> trace = mlflow.get_trace("trace_id")
        >>> tools = extract_tool_calls(trace)
        >>> for tool in tools:
        ...     print(f"Tool: {tool['name']}, Duration: {tool['duration_ms']}ms")
    """
    tool_spans = get_spans_by_type(trace, "TOOL")
    tool_calls = []

    for span in tool_spans:
        tool_info = {
            "name": getattr(span, "name", "unknown"),
            "span_id": getattr(span, "span_id", None),
            "duration_ms": getattr(span, "duration_ms", None),
            "inputs": getattr(span, "inputs", {}),
            "outputs": getattr(span, "outputs", None),
        }
        tool_calls.append(tool_info)

    return tool_calls
