"""
Background worker for synchronizing traces from Databricks experiments to local experiments.
"""

import asyncio
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.exceptions import MlflowException
from mlflow.server.databricks_trace_sync import DatabricksTraceSyncConfig
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracing.client import TracingClient
from mlflow.tracking.client import MlflowClient

_logger = logging.getLogger(__name__)


class DatabricksTraceSyncWorker:
    """
    Background worker that synchronizes traces from Databricks experiments to local experiments.
    """

    def __init__(self, config: DatabricksTraceSyncConfig):
        self.config = config
        # For testing, use the same tracking URI for both source and destination
        # In production, source would be Databricks and destination would be local
        self.source_client = TracingClient()  # Use default tracking URI for testing
        self.dest_client = TracingClient()  # Use default local tracking URI
        self.mlflow_client = MlflowClient()  # For experiment management
        self._stop_event = threading.Event()
        self._worker_thread = None

    def start(self):
        """Start the background synchronization worker."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            _logger.warning("Trace sync worker is already running")
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._run_sync_loop,
            name="DatabricksTraceSyncWorker",
            daemon=True,
        )
        self._worker_thread.start()
        _logger.info(
            f"Started trace sync worker for experiment '{self.config.source_experiment_name}'"
        )

    def stop(self):
        """Stop the background synchronization worker."""
        if self._worker_thread is None:
            return

        _logger.info("Stopping trace sync worker...")
        self._stop_event.set()
        self._worker_thread.join(timeout=10)
        if self._worker_thread.is_alive():
            _logger.warning("Trace sync worker did not stop gracefully")
        else:
            _logger.info("Trace sync worker stopped")

    def _run_sync_loop(self):
        """Main synchronization loop that runs in the background thread."""
        while not self._stop_event.is_set():
            try:
                self._sync_traces()
            except Exception as e:
                _logger.error(f"Error in trace sync loop: {e}", exc_info=True)

            # Wait before next sync cycle (30 seconds)
            self._stop_event.wait(30)

    def _sync_traces(self):
        """Synchronize traces from the configured source experiment."""
        # Ensure destination experiment exists
        dest_experiment = self._ensure_experiment_exists(self.config.destination_experiment_name)

        # Get source experiment ID
        try:
            source_exp = self.mlflow_client.get_experiment_by_name(
                self.config.source_experiment_name
            )
            if source_exp:
                source_experiment_id = source_exp.experiment_id
                _logger.debug(
                    f"Found source experiment: {self.config.source_experiment_name} "
                    f"(ID: {source_experiment_id})"
                )
            else:
                _logger.warning(
                    f"Source experiment '{self.config.source_experiment_name}' not found"
                )
                return
        except Exception as e:
            _logger.error(f"Failed to get experiment '{self.config.source_experiment_name}': {e}")
            return

        # Find the last synced trace (cursor) in destination
        cursor_trace = self._find_last_synced_trace(
            dest_experiment.experiment_id, source_exp.experiment_id
        )

        if cursor_trace:
            _logger.info(
                f"Found cursor trace {cursor_trace.info.trace_id} "
                f"(timestamp: {cursor_trace.info.timestamp_ms})"
            )
        else:
            _logger.info("No previously synced traces found, starting from beginning")

        # Search and sync traces from source experiment
        self._search_and_sync_traces(
            [source_experiment_id], dest_experiment.experiment_id, cursor_trace
        )

    def _find_last_synced_trace(
        self, dest_experiment_id: str, source_experiment_id: str
    ) -> Optional[Trace]:
        """
        Find the most recently synced trace in the destination experiment.

        Returns the trace with the latest timestamp that has the source experiment tag.
        """
        try:
            # Search for traces with the source experiment tag
            filter_string = f"tags.mlflow.databricks.sourceExperimentId = '{source_experiment_id}'"

            # Get the most recent trace by ordering by timestamp descending
            traces = self.dest_client.search_traces(
                experiment_ids=[dest_experiment_id],
                filter_string=filter_string,
                max_results=1,
                order_by=["timestamp DESC"],
                include_spans=False,  # We only need trace info
            )

            if traces:
                return traces[0]
            return None

        except Exception as e:
            _logger.error(f"Error finding cursor trace: {e}")
            return None

    def _ensure_experiment_exists(self, experiment_name: str):
        """Ensure the experiment exists, creating it if necessary."""
        try:
            experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
            if experiment is None:
                _logger.info(f"Creating destination experiment '{experiment_name}'")
                experiment_id = self.mlflow_client.create_experiment(experiment_name)
                experiment = self.mlflow_client.get_experiment(experiment_id)
            return experiment
        except Exception as e:
            raise MlflowException(f"Failed to ensure experiment '{experiment_name}' exists: {e}")

    def _search_and_sync_traces(
        self,
        source_experiment_ids: list[str],
        dest_experiment_id: str,
        cursor_trace: Optional[Trace],
    ):
        """Search for traces in source experiments and sync them to destination."""
        page_token = None
        total_synced = 0
        batch_size = 1000  # Process 1000 traces at a time

        # Build filter string for traces after cursor
        filter_string = None
        if cursor_trace:
            # Get the original trace ID and timestamp from the cursor
            source_trace_id = cursor_trace.info.tags.get("mlflow.databricks.sourceTraceId")
            cursor_timestamp = cursor_trace.info.timestamp_ms

            if source_trace_id:
                # Filter for traces created after cursor timestamp
                # or traces with same timestamp but ID > cursor ID (for stable ordering)
                filter_string = (
                    f"(timestamp > {cursor_timestamp}) OR "
                    f"(timestamp = {cursor_timestamp} AND trace_id > '{source_trace_id}')"
                )
                _logger.debug(f"Using filter: {filter_string}")

        while True:
            try:
                # Search for traces in source experiments
                traces_page: PagedList[Trace] = self.source_client.search_traces(
                    experiment_ids=source_experiment_ids,
                    filter_string=filter_string,
                    max_results=batch_size,
                    page_token=page_token,
                    order_by=["timestamp ASC", "trace_id ASC"],  # Ensure stable ordering
                    include_spans=True,  # We need the full trace data
                )

                if not traces_page:
                    break

                # Process traces in parallel
                traces_to_sync = []
                for trace in traces_page:
                    # Apply sampling
                    if random.random() <= self.config.sampling_rate:
                        traces_to_sync.append(trace)

                # Sync traces in parallel
                if traces_to_sync:
                    synced_count = self._sync_traces_batch(traces_to_sync, dest_experiment_id)
                    total_synced += synced_count

                # Check if there are more pages
                page_token = traces_page.token
                if not page_token:
                    break

            except Exception as e:
                _logger.error(f"Error searching/syncing traces: {e}", exc_info=True)
                break

        if total_synced > 0:
            _logger.info(f"Synced {total_synced} traces in this cycle")

    def _sync_traces_batch(self, traces: list[Trace], dest_experiment_id: str) -> int:
        """Sync a batch of traces to the destination in parallel."""
        synced_count = 0

        with ThreadPoolExecutor(max_workers=10, thread_name_prefix="TraceSync") as executor:
            futures = []
            for trace in traces:
                future = executor.submit(self._sync_single_trace, trace, dest_experiment_id)
                futures.append((trace.info.trace_id, future))

            for trace_id, future in futures:
                try:
                    if future.result():
                        synced_count += 1
                except Exception as e:
                    _logger.error(f"Failed to sync trace {trace_id}: {e}")

        return synced_count

    def _sync_single_trace(self, trace: Trace, dest_experiment_id: str) -> bool:
        """
        Sync a single trace to the destination experiment.

        Returns True if successful, False otherwise.
        """
        try:
            # Create a new trace info with the destination experiment ID
            # Note: We preserve the original timestamp to maintain chronological order
            new_trace_info = trace.info._copy_with_overrides(
                experiment_id=dest_experiment_id,
                # Keep original trace ID to maintain traceability
                tags={
                    **trace.info.tags,
                    "mlflow.databricks.sourceTraceId": trace.info.trace_id,
                    "mlflow.databricks.sourceExperimentId": trace.info.experiment_id,
                },
            )

            # 1. Create the trace in the destination
            returned_trace_info = self.dest_client.start_trace(new_trace_info)

            # 2. Log spans using the log_spans API
            if trace.data and trace.data.spans:
                try:
                    # Update span trace IDs to match the new trace
                    updated_spans = []
                    for span in trace.data.spans:
                        updated_span = span._copy_with_overrides(
                            trace_id=returned_trace_info.trace_id
                        )
                        updated_spans.append(updated_span)

                    # Log spans asynchronously
                    asyncio.run(self.dest_client.log_spans(updated_spans))
                except Exception as e:
                    _logger.error(f"Failed to log spans for trace {trace.info.trace_id}: {e}")

            # 3. Upload trace data as artifact
            try:
                # Update trace data with new trace ID
                updated_trace_data = TraceData(
                    spans=[
                        span._copy_with_overrides(trace_id=returned_trace_info.trace_id)
                        for span in trace.data.spans
                    ]
                    if trace.data
                    else []
                )
                self.dest_client._upload_trace_data(returned_trace_info, updated_trace_data)
            except Exception as e:
                _logger.error(f"Failed to upload trace data for trace {trace.info.trace_id}: {e}")

            _logger.debug(
                f"Successfully synced trace {trace.info.trace_id} -> {returned_trace_info.trace_id}"
            )
            return True

        except Exception as e:
            _logger.error(f"Failed to sync trace {trace.info.trace_id}: {e}", exc_info=True)
            return False


# Global worker instance
_sync_worker: Optional[DatabricksTraceSyncWorker] = None


def start_databricks_trace_sync(config: DatabricksTraceSyncConfig):
    """Start the Databricks trace synchronization worker."""
    global _sync_worker

    if _sync_worker is not None:
        _logger.warning("Stopping existing trace sync worker")
        _sync_worker.stop()

    _sync_worker = DatabricksTraceSyncWorker(config)
    _sync_worker.start()


def stop_databricks_trace_sync():
    """Stop the Databricks trace synchronization worker."""
    global _sync_worker

    if _sync_worker is not None:
        _sync_worker.stop()
        _sync_worker = None
