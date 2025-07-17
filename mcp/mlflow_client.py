"""MLflow client wrapper for trace operations."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


@dataclass
class GetSpanPaginationToken:
    """Token for paginating through large span content."""

    offset: int
    total_size: int

    def to_json(self) -> str:
        """Serialize token to JSON string."""
        return json.dumps({"offset": self.offset, "total_size": self.total_size})

    @classmethod
    def from_json(cls, token_str: str) -> "GetSpanPaginationToken":
        """Deserialize token from JSON string."""
        try:
            data = json.loads(token_str)
            return cls(offset=data["offset"], total_size=data["total_size"])
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid pagination token: {token_str}") from e


@dataclass
class ListPaginationToken:
    """Token for paginating through lists of items."""

    offset: int  # Offset in the list (number of items to skip)
    total_items: int  # Total number of items in the complete list

    def to_json(self) -> str:
        """Serialize token to JSON string."""
        return json.dumps({"offset": self.offset, "total_items": self.total_items})

    @classmethod
    def from_json(cls, token_str: str) -> "ListPaginationToken":
        """Deserialize token from JSON string."""
        try:
            data = json.loads(token_str)
            return cls(offset=data["offset"], total_items=data["total_items"])
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid pagination token: {token_str}") from e


class MlflowMCPClient:
    """Wrapper around MLflow client for trace operations."""

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize MLflow client with optional tracking URI."""
        self.client = MlflowClient(tracking_uri=tracking_uri)
        logger.info(f"Initialized MLflow client with tracking URI: {tracking_uri}")

    def _get_lightweight_span_dict(self, span) -> dict[str, Any]:
        """Convert a span to a lightweight dictionary representation."""
        return {
            "span_id": span.span_id,
            "name": span.name,
            "span_type": span.span_type,
            "parent_id": span.parent_id,
            "start_time_ms": span.start_time_ns // 1_000_000,
            "end_time_ms": span.end_time_ns // 1_000_000,
            "duration_ms": (span.end_time_ns - span.start_time_ns) // 1_000_000,
            "status": span.status.status_code,
        }

    def get_trace_info(self, trace_id: str) -> dict[str, Any]:
        """Get comprehensive trace information."""
        trace = self.client.get_trace(trace_id)
        return trace.info.to_dict()

    def list_trace_spans(
        self, trace_id: str, max_results: int = 100, page_token: Optional[str] = None
    ) -> tuple[list[dict[str, Any]], Optional[ListPaginationToken]]:
        """List all spans in a trace with pagination support.

        Retrieves a list of all spans within a trace, returning lightweight
        representations suitable for overview and navigation. Use get_trace_span()
        to retrieve full details for specific spans.

        Args:
            trace_id (str): The unique identifier of the trace whose spans
                to list. Format is typically a UUID string.
            max_results (int, optional): Maximum number of spans to return
                in a single response. Defaults to 100.
            page_token (Optional[str], optional): Serialized ListPaginationToken
                for retrieving next page of results. Use the token from previous
                response. Defaults to None.

        Returns:
            Tuple[List[Dict[str, Any]], Optional[ListPaginationToken]]: A tuple containing:
                - List[Dict[str, Any]]: List of span dictionaries. Each dictionary contains:
                    - span_id (str): Unique identifier for the span
                    - name (str): Human-readable name of the span
                    - span_type (str): Type of operation (LLM, RETRIEVER, etc.)
                    - parent_id (str): ID of parent span, None for root spans
                    - start_time_ms (int): Start timestamp in milliseconds
                    - end_time_ms (int): End timestamp in milliseconds
                    - duration_ms (int): Total execution time in milliseconds
                    - status (str): Status code (OK, ERROR, etc.)
                - Optional[ListPaginationToken]: Token for next page if results
                  were truncated, None if all results were returned

        Raises:
            ValueError: If the trace has no data, no spans, or invalid page token.
            Exception: If the trace cannot be found or if there's an error
                communicating with the MLflow tracking server.
        """
        trace = self.client.get_trace(trace_id)
        if not trace.data:
            raise ValueError(f"No data found for trace {trace_id}")
        if not trace.data.spans:
            return [], None

        # Get all spans as lightweight dicts
        all_spans = [self._get_lightweight_span_dict(span) for span in trace.data.spans]

        # Parse page token to get offset
        offset = 0
        if page_token:
            token = ListPaginationToken.from_json(page_token)
            offset = token.offset

        # Get the page of results
        end_offset = offset + max_results
        page_spans = all_spans[offset:end_offset]

        # Create next page token if there are more spans
        next_token = None
        if end_offset < len(all_spans):
            next_token = ListPaginationToken(offset=end_offset, total_items=len(all_spans))

        return page_spans, next_token

    def get_trace_span(
        self,
        trace_id: str,
        span_id: str,
        max_content_length: int = 100000,
        page_token: Optional[str] = None,
    ) -> tuple[str, Optional[GetSpanPaginationToken]]:
        """Get detailed information about a specific span with pagination support."""
        trace = self.client.get_trace(trace_id)
        if not trace.data:
            raise ValueError(f"No data found for trace {trace_id}")

        for span in trace.data.spans:
            if span.span_id == span_id:
                span_dict = span.to_dict()
                span_json = json.dumps(span_dict, sort_keys=True)
                total_size = len(span_json)

                # Parse page token to get offset
                offset = 0
                if page_token:
                    token = GetSpanPaginationToken.from_json(page_token)
                    offset = token.offset

                # Return slice of JSON string
                end_offset = min(offset + max_content_length, total_size)
                content_slice = span_json[offset:end_offset]

                # Create next page token if there's more content
                next_token = None
                if end_offset < total_size:
                    next_token = GetSpanPaginationToken(offset=end_offset, total_size=total_size)

                return content_slice, next_token

        raise ValueError(f"Span {span_id} not found in trace {trace_id}")

    def search_trace_spans(
        self,
        trace_id: str,
        pattern: Optional[str] = None,
        max_results: int = 100,
        page_token: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], Optional[ListPaginationToken]]:
        """Search spans within a trace using regex pattern matching with pagination support.

        Searches through all spans in a trace, matching against the complete
        JSON representation of each span. This allows searching through span
        names, inputs, outputs, attributes, and any other span data.

        Args:
            trace_id (str): The unique identifier of the trace to search within.
                Format is typically a UUID string.
            pattern (Optional[str], optional): Regular expression pattern to
                search for in span data. If None, returns all spans. The pattern
                is matched against the JSON representation of each span.

                Examples:
                - "error.*timeout" - Find spans with timeout errors
                - "model.*gpt-4" - Find spans using GPT-4
                - "duration_ms.*[5-9][0-9]{3}" - Find spans taking 5+ seconds
                Defaults to None.
            max_results (int, optional): Maximum number of spans to return
                in a single response. Defaults to 100.
            page_token (Optional[str], optional): Serialized ListPaginationToken
                for retrieving next page of results. Use the token from previous
                response. Defaults to None.

        Returns:
            Tuple[List[Dict[str, Any]], Optional[ListPaginationToken]]: A tuple containing:
                - List[Dict[str, Any]]: List of matching span dictionaries. Each contains:
                    - span_id (str): Unique identifier for the span
                    - name (str): Human-readable name of the span
                    - span_type (str): Type of operation
                    - parent_id (str): ID of parent span
                    - start_time_ms (int): Start timestamp in milliseconds
                    - end_time_ms (int): End timestamp in milliseconds
                    - duration_ms (int): Total execution time in milliseconds
                    - status (str): Status code
                - Optional[ListPaginationToken]: Token for next page if results
                  were truncated, None if all results were returned

            Returns empty list if no spans match the pattern.

        Raises:
            ValueError: If the trace has no data, no spans, invalid regex pattern,
                or invalid page token.
            Exception: If the trace cannot be found or if there's an error
                communicating with the MLflow tracking server.
        """
        trace = self.client.get_trace(trace_id)
        if not trace.data:
            raise ValueError(f"No data found for trace {trace_id}")
        if not trace.data.spans:
            return [], None

        # Compile regex pattern if pattern is provided
        regex_pattern = None
        if pattern:
            try:
                regex_pattern = re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}") from e

        # Filter spans based on pattern
        filtered_spans = []
        for span in trace.data.spans:
            # If no pattern provided, include all spans
            if not regex_pattern:
                filtered_spans.append(self._get_lightweight_span_dict(span))
            else:
                # Get full span details for searching
                span_dict = span.to_dict()
                span_json = json.dumps(span_dict, sort_keys=True)

                # Apply regex pattern search on JSON representation
                if regex_pattern.search(span_json):
                    filtered_spans.append(self._get_lightweight_span_dict(span))

        if not filtered_spans:
            return [], None

        # Parse page token to get offset
        offset = 0
        if page_token:
            token = ListPaginationToken.from_json(page_token)
            offset = token.offset

        # Get the page of results
        end_offset = offset + max_results
        page_spans = filtered_spans[offset:end_offset]

        # Create next page token if there are more spans
        next_token = None
        if end_offset < len(filtered_spans):
            next_token = ListPaginationToken(offset=end_offset, total_items=len(filtered_spans))

        return page_spans, next_token

    def search_traces(
        self,
        filter_string: Optional[str] = None,
        span_pattern: Optional[str] = None,
        experiment_ids: Optional[list[str]] = None,
        max_results: int = 100,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], Optional[ListPaginationToken]]:
        """Search traces using MLflow filter_string and optional span regex pattern.

        Searches for traces across experiments using MLflow's native search
        capabilities. Optionally filters results by span content using regex
        pattern matching.

        Args:
            filter_string (Optional[str], optional): MLflow filter expression
                for trace attributes. Uses MLflow's filter syntax.

                Examples:
                - "status = 'ERROR'"
                - "execution_time_ms > 1000"
                - "tags.model_name = 'gpt-4'"
                - "timestamp_ms > 1234567890"
                Defaults to None (no filtering).
            span_pattern (Optional[str], optional): Regular expression pattern
                to match against span content. Only traces containing at least
                one matching span will be returned. Pattern is matched against
                JSON representation of spans. Defaults to None.
            experiment_ids (Optional[List[str]], optional): List of experiment
                IDs to search within. If None, searches all accessible
                experiments. Defaults to None.
            max_results (int, optional): Maximum number of traces to return
                in a single response. Defaults to 100.
            order_by (Optional[List[str]], optional): List of fields to order
                results by. Each field can be prefixed with '+' for ascending
                or '-' for descending order.
                Examples: ["-timestamp_ms"], ["+execution_time_ms"]
                Defaults to None.
            page_token (Optional[str], optional): Serialized ListPaginationToken
                for retrieving next page of results. Use the token from previous
                response. Defaults to None.

        Returns:
            Tuple[List[Dict[str, Any]], Optional[ListPaginationToken]]: A tuple containing:
                - List[Dict[str, Any]]: List of trace information dictionaries.
                  Each dictionary contains trace metadata including:
                    - All fields from trace.info.to_dict()
                    - span_count (int): Number of spans in the trace
                    - root_span_id (str): ID of the root span, if any
                - Optional[ListPaginationToken]: Token for next page if results
                  were truncated, None if all results were returned

            Returns empty list if no traces match the criteria.

        Raises:
            Exception: If there's an error communicating with the MLflow
                tracking server or if the filter_string syntax is invalid.
        """
        # Use MLflow's native search_traces API - get more than needed for pagination
        from mlflow.tracing import search_traces

        # Parse page token to get offset
        offset = 0
        if page_token:
            token = ListPaginationToken.from_json(page_token)
            offset = token.offset

        # Request more traces than needed to implement pagination
        # We'll get offset + max_results + 1 to check if there are more
        search_limit = offset + max_results + 1

        traces = search_traces(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=search_limit,
            order_by=order_by,
            return_type="list",
        )

        # Filter traces by span pattern if provided
        filtered_traces = []
        for trace in traces:
            # Use trace.info.to_dict() and add supplemental fields
            trace_info = trace.info.to_dict()
            trace_info["span_count"] = len(trace.data.spans) if trace.data else 0
            trace_info["root_span_id"] = (
                trace.data.spans[0].span_id if trace.data and trace.data.spans else None
            )

            # If span pattern is provided, filter by spans
            if span_pattern and trace.data and trace.data.spans:
                pattern = re.compile(span_pattern)
                found_match = False

                for span in trace.data.spans:
                    span_dict = span.to_dict()
                    span_json = json.dumps(span_dict, sort_keys=True)
                    if pattern.search(span_json):
                        found_match = True
                        break

                if found_match:
                    filtered_traces.append(trace_info)
            else:
                filtered_traces.append(trace_info)

        # Apply pagination to filtered results
        page_traces = filtered_traces[offset : offset + max_results]

        # Create next page token if there are more traces
        next_token = None
        if len(filtered_traces) > offset + max_results:
            next_token = ListPaginationToken(
                offset=offset + max_results, total_items=len(filtered_traces)
            )

        return page_traces, next_token
