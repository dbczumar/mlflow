"""
Get root span tool for MLflow GenAI judges.

This module provides a tool for retrieving the root span of a trace,
which contains the top-level inputs and outputs.
"""

import json
from dataclasses import dataclass

from mlflow.entities.trace import Trace
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.types.llm import (
    FunctionToolDefinition,
    ToolDefinition,
    ToolParamsSchema,
)
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
@dataclass
class GetRootSpanResult:
    """Result from getting the root span."""

    span_id: str | None
    content: str | None  # JSON string of root span content
    content_size_bytes: int
    error: str | None = None


@experimental(version="3.4.0")
class GetRootSpanTool(JudgeTool):
    """
    Tool for retrieving the root span from a trace.

    The root span contains the top-level inputs to the agent and final outputs.
    """

    @property
    def name(self) -> str:
        return "get_root_span"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name="get_root_span",
                description=(
                    "Retrieve the root span of the trace, which contains the top-level inputs "
                    "to the agent and final outputs. Note that in some traces, the root span "
                    "may not contain outputs, but it typically should. If the root span doesn't "
                    "have outputs, you may need to look at other spans to find the final results. "
                    "The content is returned as a JSON string with all span details."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={},
                    required=[],
                ),
            ),
            type="function",
        )

    def invoke(self, trace: Trace) -> GetRootSpanResult:
        """
        Get the root span from the trace.

        Args:
            trace: The MLflow trace object to analyze

        Returns:
            GetRootSpanResult with the root span content as JSON string
        """
        if not trace or not trace.data or not trace.data.spans:
            return GetRootSpanResult(
                span_id=None, content=None, content_size_bytes=0, error="Trace has no spans"
            )

        # Find the root span (parent_id is None)
        root_span = None
        for span in trace.data.spans:
            if span.parent_id is None:
                root_span = span
                break

        if not root_span:
            return GetRootSpanResult(
                span_id=None,
                content=None,
                content_size_bytes=0,
                error="No root span found in trace",
            )

        # Calculate timing information
        start_time_ms = root_span.start_time_ns / 1_000_000
        end_time_ms = root_span.end_time_ns / 1_000_000
        duration_ms = end_time_ms - start_time_ms

        # Build root span data
        span_data = {
            "span_id": root_span.span_id,
            "name": root_span.name,
            "span_type": root_span.span_type,
            "is_root": True,
            "start_time_ms": start_time_ms,
            "end_time_ms": end_time_ms,
            "duration_ms": duration_ms,
            "status": {
                "status_code": root_span.status.status_code,
                "description": root_span.status.description,
            },
            "inputs": root_span.inputs,
            "outputs": root_span.outputs,
            "attributes": root_span.attributes,
            "events": [
                {
                    "name": event.name,
                    "timestamp_ms": event.timestamp_ns / 1_000_000 if event.timestamp_ns else None,
                    "attributes": event.attributes,
                }
                for event in (root_span.events or [])
            ],
            "has_outputs": root_span.outputs is not None and len(str(root_span.outputs)) > 0,
            "child_span_count": sum(
                1 for s in trace.data.spans if s.parent_id == root_span.span_id
            ),
        }

        # Convert to JSON string
        content = json.dumps(span_data, default=str, indent=2)
        content_size = len(content.encode("utf-8"))

        return GetRootSpanResult(
            span_id=root_span.span_id, content=content, content_size_bytes=content_size, error=None
        )
