"""MLflow client wrapper for trace operations."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

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
        """Initialize MLflow client."""
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

    def _get_enhanced_trace_info(self, trace) -> dict[str, Any]:
        """Get trace info with additional span count and root span information."""
        trace_info = trace.info.to_dict()
        # Add additional computed fields
        trace_info["span_count"] = len(trace.data.spans) if trace.data else 0
        trace_info["root_span_id"] = (
            trace.data.spans[0].span_id if trace.data and trace.data.spans else None
        )
        return trace_info

    def get_trace_info(self, trace_id: str) -> dict[str, Any]:
        """Get comprehensive trace information."""
        trace = self.client.get_trace(trace_id)
        return self._get_enhanced_trace_info(trace)

    def list_trace_spans(
        self, trace_id: str, max_content_length: int = 100000, page_token: Optional[str] = None
    ) -> Tuple[str, Optional[ListPaginationToken]]:
        """List all spans in a trace with pagination support."""
        trace = self.client.get_trace(trace_id)
        if not trace.data:
            raise ValueError(f"No data found for trace {trace_id}")
        if not trace.data.spans:
            raise ValueError(f"No spans found for trace {trace_id}")

        # Get all spans as lightweight dicts
        all_spans = [self._get_lightweight_span_dict(span) for span in trace.data.spans]

        # Parse page token to get offset
        offset = 0
        if page_token:
            token = ListPaginationToken.from_json(page_token)
            offset = token.offset

        # Get the remaining spans from offset
        remaining_spans = all_spans[offset:]

        # Pop elements until JSON is under limit
        while remaining_spans:
            result_json = json.dumps(remaining_spans, sort_keys=True)
            if len(result_json) <= max_content_length:
                break
            remaining_spans.pop()

        # Create next page token if there are more spans
        next_token = None
        if offset + len(remaining_spans) < len(all_spans):
            next_token = ListPaginationToken(
                offset=offset + len(remaining_spans), total_items=len(all_spans)
            )

        return json.dumps(remaining_spans, sort_keys=True), next_token

    def get_trace_span(
        self,
        trace_id: str,
        span_id: str,
        max_content_length: int = 100000,
        page_token: Optional[str] = None,
    ) -> Tuple[str, Optional[GetSpanPaginationToken]]:
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
        keywords: Optional[list[str]] = None,
        max_content_length: int = 100000,
        page_token: Optional[str] = None,
    ) -> Tuple[str, Optional[ListPaginationToken]]:
        """Search spans within a trace using keyword matching with pagination support."""
        trace = self.client.get_trace(trace_id)
        if not trace.data:
            raise ValueError(f"No data found for trace {trace_id}")
        if not trace.data.spans:
            raise ValueError(f"No spans found for trace {trace_id}")

        # Prepare keywords for case-insensitive search
        search_keywords = None
        if keywords:
            search_keywords = [keyword.lower() for keyword in keywords]

        # Filter spans based on keywords
        filtered_spans = []
        for span in trace.data.spans:
            # If no keywords provided, include all spans
            if not search_keywords:
                filtered_spans.append(self._get_lightweight_span_dict(span))
            else:
                # Build searchable content from key span fields
                searchable_parts = [
                    span.name or "",
                    json.dumps(span.inputs) if span.inputs else "",
                    json.dumps(span.outputs) if span.outputs else "",
                    json.dumps(span.attributes) if span.attributes else "",
                ]
                searchable_content = " ".join(searchable_parts).lower()

                # Check if any keyword matches (OR logic)
                if any(keyword in searchable_content for keyword in search_keywords):
                    filtered_spans.append(self._get_lightweight_span_dict(span))

        if not filtered_spans:
            return json.dumps([]), None

        # Parse page token to get offset
        offset = 0
        if page_token:
            token = ListPaginationToken.from_json(page_token)
            offset = token.offset

        # Get the remaining spans from offset
        remaining_spans = filtered_spans[offset:]

        # Pop elements until JSON is under limit
        while remaining_spans:
            result_json = json.dumps(remaining_spans, sort_keys=True)
            if len(result_json) <= max_content_length:
                break
            remaining_spans.pop()

        # Create next page token if there are more spans
        next_token = None
        if offset + len(remaining_spans) < len(filtered_spans):
            next_token = ListPaginationToken(
                offset=offset + len(remaining_spans), total_items=len(filtered_spans)
            )

        return json.dumps(remaining_spans, sort_keys=True), next_token

    def search_traces(
        self,
        filter_string: Optional[str] = None,
        span_keywords: Optional[list[str]] = None,
        experiment_ids: Optional[list[str]] = None,
        experiment_names: Optional[list[str]] = None,
        max_results: Optional[int] = None,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
    ) -> Tuple[list[dict[str, Any]], Optional[ListPaginationToken]]:
        """Search traces using MLflow filter_string and optional span keyword matching.

        Args:
            filter_string: MLflow filter expression for trace attributes
            span_keywords: Keywords to search for in span content (OR'd together)
            experiment_ids: List of experiment IDs to search within
            experiment_names: List of experiment names to search within (resolved to IDs)
            max_results: Maximum number of traces to return per page (default 1)
            order_by: List of fields to order results by
            page_token: Token for retrieving next page of results

        Returns:
            Tuple of (list of trace info dictionaries, next page token)

        Raises:
            ValueError: If neither experiment_ids nor experiment_names is provided
        """
        # Validate that at least one experiment specification is provided
        if not experiment_ids and not experiment_names:
            raise ValueError("Either experiment_ids or experiment_names must be provided")

        # Resolve experiment names to IDs if provided
        resolved_experiment_ids = experiment_ids or []
        if experiment_names:
            for name in experiment_names:
                try:
                    experiment = self.client.get_experiment_by_name(name)
                    resolved_experiment_ids.append(experiment.experiment_id)
                except Exception as e:
                    raise ValueError(f"Experiment '{name}' not found: {e}") from e

        # Default to ordering by creation time descending if no order specified
        order_by = order_by or ["timestamp_ms DESC"]

        # Parse pagination token if provided
        start_offset = 0
        if page_token:
            token = ListPaginationToken.from_json(page_token)
            start_offset = token.offset

        # Prepare keywords for case-insensitive search if span filtering is needed
        search_keywords = None
        if span_keywords:
            search_keywords = [keyword.lower() for keyword in span_keywords]

        # Default to 1 result per page for best performance
        page_size = max_results or 1
        results = []
        current_offset = start_offset
        mlflow_page_token = None

        # Keep fetching from MLflow until we have enough results for this page OR no more traces
        while len(results) < page_size:
            # Fetch traces from MLflow in batches
            batch_size = 20  # Fixed batch size for MLflow queries

            paged_traces = self.client.search_traces(
                experiment_ids=resolved_experiment_ids,
                filter_string=filter_string,
                max_results=batch_size,
                order_by=order_by,
                page_token=mlflow_page_token,
            )

            # If no more traces found from MLflow, we're completely done
            if not paged_traces:
                break

            # Process each trace in the batch and apply span filtering
            batch_results = []
            for trace in paged_traces:
                trace_info = self._get_enhanced_trace_info(trace)

                # If keywords are provided, filter by span content
                if search_keywords and trace.data and trace.data.spans:
                    found_match = False
                    for span in trace.data.spans:
                        # Build searchable content from key span fields
                        searchable_parts = [
                            span.name or "",
                            json.dumps(span.inputs) if span.inputs else "",
                            json.dumps(span.outputs) if span.outputs else "",
                            json.dumps(span.attributes) if span.attributes else "",
                        ]
                        searchable_content = " ".join(searchable_parts).lower()

                        # Check if any keyword matches (OR logic)
                        if any(keyword in searchable_content for keyword in search_keywords):
                            found_match = True
                            break

                    if found_match:
                        batch_results.append(trace_info)
                else:
                    # No span filtering, include all traces
                    batch_results.append(trace_info)

            # Apply offset and collect results for this page
            if current_offset > 0:
                # Still skipping items due to pagination offset
                items_to_skip = min(current_offset, len(batch_results))
                batch_results = batch_results[items_to_skip:]
                current_offset -= items_to_skip

            # Take what we need for this page
            items_to_take = min(page_size - len(results), len(batch_results))
            results.extend(batch_results[:items_to_take])

            # Update MLflow pagination token for next batch
            mlflow_page_token = getattr(paged_traces, "token", None)

            # If no more MLflow pages, we're done (even if we don't have a full page)
            if not mlflow_page_token:
                break

        # Create next page token only if we have results AND there might be more
        next_token = None
        if len(results) > 0 and (mlflow_page_token or len(results) == page_size):
            # We found some results and there might be more
            next_token = ListPaginationToken(
                offset=start_offset + len(results),
                total_items=-1,  # Unknown total, but there are more
            )

        return results, next_token
