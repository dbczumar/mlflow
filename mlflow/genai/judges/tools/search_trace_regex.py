"""
Tool for searching traces using regex patterns.

This module provides functionality to search through all spans in a trace
using regular expressions with case-insensitive matching.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from mlflow.entities.trace import Trace
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.registry import register_judge_tool
from mlflow.types.llm import ToolDefinition


@dataclass
class RegexMatch:
    """Represents a single regex match found in a span."""

    span_id: str
    span_name: str
    field_path: str  # e.g., "inputs.message" or "outputs.response"
    matched_text: str
    context_before: str
    context_after: str
    match_start: int
    match_end: int


@dataclass
class SearchTraceRegexResult:
    """Result of searching a trace with a regex pattern."""

    pattern: str
    total_matches: int
    matches: list[RegexMatch]
    error: str | None = None


class SearchTraceRegexTool(JudgeTool):
    """
    Tool for searching through all spans in a trace using regex patterns.

    Performs case-insensitive regex search across all span fields including
    inputs, outputs, and attributes. Returns matched text with surrounding
    context to help understand where matches occur.
    """

    @property
    def name(self) -> str:
        """Return the tool name."""
        return "search_trace_regex"

    def get_definition(self) -> ToolDefinition:
        """Get the tool definition for LiteLLM/OpenAI function calling."""
        return ToolDefinition(
            type="function",
            function={
                "name": self.name,
                "description": (
                    "Search through all spans in the trace using a regular expression pattern. "
                    "Performs case-insensitive matching and returns all matches with surrounding "
                    "context. Useful for finding specific patterns, values, or text across the "
                    "entire trace."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": (
                                "Regular expression pattern to search for. The search is "
                                "case-insensitive. Examples: 'track.*number' to find tracking "
                                "numbers, '\\d{3}-\\d{4}' for phone patterns, 'error|fail' to "
                                "find errors or failures."
                            ),
                        },
                        "context_chars": {
                            "type": "integer",
                            "description": (
                                "Number of characters to include before and after each match "
                                "for context. Default is 50."
                            ),
                            "default": 50,
                        },
                        "max_matches": {
                            "type": "integer",
                            "description": (
                                "Maximum number of matches to return. Default is 100. "
                                "Set to a lower value if you only need a few examples."
                            ),
                            "default": 100,
                        },
                    },
                    "required": ["pattern"],
                },
            },
        )

    def invoke(
        self, trace: Trace, pattern: str, context_chars: int = 50, max_matches: int = 100
    ) -> SearchTraceRegexResult:
        """
        Search the trace for regex pattern matches.

        Args:
            trace: The trace to search
            pattern: Regular expression pattern (case-insensitive)
            context_chars: Number of context characters before/after match
            max_matches: Maximum number of matches to return

        Returns:
            SearchTraceRegexResult with all matches found
        """
        if not trace or not trace.data or not trace.data.spans:
            return SearchTraceRegexResult(
                pattern=pattern,
                total_matches=0,
                matches=[],
                error="No trace data available",
            )

        try:
            # Compile regex pattern with case-insensitive flag
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return SearchTraceRegexResult(
                pattern=pattern,
                total_matches=0,
                matches=[],
                error=f"Invalid regex pattern: {e}",
            )

        matches = []
        total_matches = 0

        for span in trace.data.spans:
            if total_matches >= max_matches:
                break

            # Search in various span fields
            fields_to_search = {}

            # Add inputs if present
            if span.inputs:
                fields_to_search["inputs"] = span.inputs

            # Add outputs if present
            if span.outputs:
                fields_to_search["outputs"] = span.outputs

            # Add relevant attributes
            if span.attributes:
                for key, value in span.attributes.items():
                    # Skip internal MLflow attributes
                    if not key.startswith("mlflow."):
                        fields_to_search[f"attributes.{key}"] = value

            # Search each field
            for field_path, field_value in fields_to_search.items():
                if total_matches >= max_matches:
                    break

                # Convert field value to searchable text
                text = self._value_to_text(field_value)

                # Find all matches in this text
                for match in regex.finditer(text):
                    if total_matches >= max_matches:
                        break

                    # Extract context around the match
                    start = max(0, match.start() - context_chars)
                    end = min(len(text), match.end() + context_chars)

                    context_before = text[start : match.start()]
                    context_after = text[match.end() : end]

                    matches.append(
                        RegexMatch(
                            span_id=span.span_id,
                            span_name=span.name,
                            field_path=field_path,
                            matched_text=match.group(0),
                            context_before=context_before,
                            context_after=context_after,
                            match_start=match.start(),
                            match_end=match.end(),
                        )
                    )
                    total_matches += 1

        return SearchTraceRegexResult(
            pattern=pattern,
            total_matches=total_matches,
            matches=matches,
        )

    def _value_to_text(self, value: Any) -> str:
        """
        Convert any value to searchable text.

        Args:
            value: Value to convert

        Returns:
            String representation of the value
        """
        if isinstance(value, str):
            return value
        elif isinstance(value, dict):
            # Convert dict to JSON string for searching
            try:
                return json.dumps(value, default=str, indent=2)
            except Exception:
                return str(value)
        elif isinstance(value, list):
            # Convert list to JSON string for searching
            try:
                return json.dumps(value, default=str, indent=2)
            except Exception:
                return str(value)
        else:
            return str(value)


# Register the tool
register_judge_tool(SearchTraceRegexTool())
