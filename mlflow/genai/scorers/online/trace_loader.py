"""Trace loading utilities for online scoring."""

import logging

from mlflow.entities import Trace
from mlflow.store.tracking.abstract_store import AbstractStore

_logger = logging.getLogger(__name__)


class TraceLoader:
    """Loads traces from the tracking store."""

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
    ) -> list[Trace]:
        """
        Fetch traces within a time window, optionally filtered.

        Args:
            experiment_id: The experiment ID to search.
            start_time_ms: Start of time window (exclusive).
            end_time_ms: End of time window (inclusive).
            filter_string: Optional additional filter criteria.
            max_traces: Maximum number of traces to return.

        Returns:
            List of Trace objects matching the criteria.
        """
        time_filter = (
            f"trace.timestamp_ms > {start_time_ms} AND trace.timestamp_ms <= {end_time_ms}"
        )
        combined_filter = f"({time_filter}) AND ({filter_string})" if filter_string else time_filter

        trace_infos, _ = self._tracking_store.search_traces(
            experiment_ids=[experiment_id],
            filter_string=combined_filter,
            max_results=max_traces,
            order_by=["timestamp_ms ASC"],
        )

        if not trace_infos:
            return []

        trace_ids = [info.trace_id for info in trace_infos]
        return self.fetch_traces(trace_ids)
