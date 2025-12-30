"""Trace loading utilities for online scoring."""

import logging

from mlflow.entities import Trace, TraceInfo
from mlflow.store.tracking.abstract_store import AbstractStore

_logger = logging.getLogger(__name__)


class OnlineTraceLoader:
    """Loads traces from the tracking store for online scoring."""

    def __init__(self, tracking_store: AbstractStore):
        self._tracking_store = tracking_store

    def fetch_traces(self, trace_ids: list[str]) -> list[Trace]:
        """
        Fetch full traces by their IDs.

        Args:
            trace_ids: List of trace IDs to fetch.

        Returns:
            List of Trace objects (in same order as input, skipping any not found).
        """
        if not trace_ids:
            return []

        traces = self._tracking_store.batch_get_traces(trace_ids)
        trace_map = {t.info.trace_id: t for t in traces}

        # Preserve order, skip missing
        return [trace_map[tid] for tid in trace_ids if tid in trace_map]

    def fetch_trace_infos_between(
        self,
        experiment_id: str,
        start_time_ms: int,
        end_time_ms: int,
        filter_string: str | None = None,
        max_traces: int = 500,
        page_size: int = 100,
    ) -> list[TraceInfo]:
        """
        Fetch trace infos within a time window, optionally filtered.

        Args:
            experiment_id: The experiment ID to search.
            start_time_ms: Start of time window (exclusive).
            end_time_ms: End of time window (inclusive).
            filter_string: Optional additional filter criteria.
            max_traces: Maximum number of traces to return.
            page_size: Number of traces to fetch per API call.

        Returns:
            List of TraceInfo objects matching the criteria.
        """
        time_filter = (
            f"trace.timestamp_ms > {start_time_ms} AND trace.timestamp_ms <= {end_time_ms}"
        )
        # Exclude traces from MLflow runs (training traces)
        combined_filter = f"{time_filter} AND metadata.mlflow.sourceRun IS NULL"
        if filter_string:
            combined_filter = f"{combined_filter} AND {filter_string}"
        _logger.info(f"Fetching traces with filter: {combined_filter}")

        all_trace_infos = []
        page_token = None
        has_more = True

        # Paginate through results
        while has_more and len(all_trace_infos) < max_traces:
            batch_size = min(page_size, max_traces - len(all_trace_infos))

            trace_batch, token = self._tracking_store.search_traces(
                experiment_ids=[experiment_id],
                filter_string=combined_filter,
                max_results=batch_size,
                order_by=["timestamp_ms ASC"],
                page_token=page_token,
            )

            if trace_batch:
                remaining = max_traces - len(all_trace_infos)
                all_trace_infos.extend(trace_batch[:remaining])
                _logger.debug(
                    f"Fetched batch of {len(trace_batch)} traces, total: {len(all_trace_infos)}"
                )
                page_token = token
                if not page_token:
                    has_more = False
            else:
                has_more = False

        _logger.info(
            f"Fetched {len(all_trace_infos)} trace infos between {start_time_ms} and {end_time_ms}"
        )

        # search_traces returns Trace objects, extract TraceInfo
        return [t.info if isinstance(t, Trace) else t for t in all_trace_infos]
