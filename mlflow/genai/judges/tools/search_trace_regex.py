"""
Tool for searching traces using regex patterns.

This module provides functionality to search through all spans in a trace
using regular expressions with case-insensitive matching.
"""

import json
import re
from dataclasses import dataclass

from mlflow.entities.trace import Trace
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.types.llm import ToolDefinition


@dataclass
class RegexMatch:
    """Represents a single regex match found in a span."""

    span_id: str
    matched_text: str
    surrounding_text: str  # Text with ~100 chars before and after, with ellipses


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
                                "for context. Default is 100."
                            ),
                            "default": 100,
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
        self, trace: Trace, pattern: str, context_chars: int = 100, max_matches: int = 100
    ) -> SearchTraceRegexResult:
        """
        Search the trace for regex pattern matches.

        Args:
            trace: The trace to search
            pattern: Regular expression pattern (case-insensitive)
            context_chars: Number of context characters before/after match (default 100)
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

        # Convert entire trace to JSON string for searching
        for span in trace.data.spans:
            if total_matches >= max_matches:
                break

            # Convert span to JSON representation
            span_json = json.dumps(span.to_dict(), default=str, indent=2)

            # Find all matches in this span's JSON
            for match in regex.finditer(span_json):
                if total_matches >= max_matches:
                    break

                # Extract context around the match
                start = max(0, match.start() - context_chars)
                end = min(len(span_json), match.end() + context_chars)

                # Create surrounding text with ellipses
                prefix = "..." if start > 0 else ""
                suffix = "..." if end < len(span_json) else ""
                surrounding = prefix + span_json[start:end] + suffix

                matches.append(
                    RegexMatch(
                        span_id=span.span_id,
                        matched_text=match.group(0),
                        surrounding_text=surrounding,
                    )
                )
                total_matches += 1

        return SearchTraceRegexResult(
            pattern=pattern,
            total_matches=total_matches,
            matches=matches,
        )
