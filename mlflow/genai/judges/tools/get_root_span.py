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
    content: str | None  # JSON string of root span content (may be truncated)
    content_size_bytes: int
    page_token: str | None = None  # Token to get next page if content was truncated
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
                    "The content is returned as a JSON string. Large content may be paginated."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={
                        "max_content_length": {
                            "type": "integer",
                            "description": "Maximum content size in bytes (default: 100000)",
                        },
                        "page_token": {
                            "type": "string",
                            "description": "Token to retrieve the next page of content",
                        },
                    },
                    required=[],
                ),
            ),
            type="function",
        )

    def invoke(
        self, trace: Trace, max_content_length: int = 100000, page_token: str | None = None
    ) -> GetRootSpanResult:
        """
        Get the root span from the trace.

        Args:
            trace: The MLflow trace object to analyze
            max_content_length: Maximum content size in bytes to return
            page_token: Token to retrieve the next page (offset in bytes)

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

        # Parse page token to get offset
        offset = 0
        if page_token:
            try:
                offset = int(page_token)
            except (ValueError, TypeError):
                offset = 0

        # Convert span directly to JSON
        full_content = json.dumps(root_span.to_dict(), default=str, indent=2)
        total_size = len(full_content.encode("utf-8"))

        # Get the chunk for this page
        end_offset = min(offset + max_content_length, total_size)
        content_chunk = full_content[offset:end_offset]

        # Determine if there's more content
        next_page_token = str(end_offset) if end_offset < total_size else None

        return GetRootSpanResult(
            span_id=root_span.span_id,
            content=content_chunk,
            content_size_bytes=len(content_chunk.encode("utf-8")),
            page_token=next_page_token,
            error=None,
        )
