"""
Background worker for synchronizing traces from Databricks experiments to local experiments.
"""

import logging
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# Configure boto3 to use larger connection pools for S3
# This prevents "Connection pool is full" warnings when downloading many traces in parallel
# We need to monkey patch botocore's default config since MLflow doesn't expose this
try:
    import botocore.config

    # Increase max_pool_connections from default 10 to 50
    original_config_init = botocore.config.Config.__init__

    def patched_config_init(self, **kwargs):
        # Ensure max_pool_connections is set to at least 50
        if "max_pool_connections" not in kwargs or kwargs.get("max_pool_connections", 0) < 50:
            kwargs["max_pool_connections"] = 50
        original_config_init(self, **kwargs)

    botocore.config.Config.__init__ = patched_config_init
except ImportError:
    # If botocore is not installed, S3 artifact storage won't be used anyway
    pass

from mlflow.entities.trace import Trace
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
        # Source client connects to Databricks
        self.source_client = TracingClient(tracking_uri=config.tracking_uri)
        # Destination client uses local tracking URI
        self.dest_client = TracingClient()  # Use default local tracking URI
        # MLflow clients - source for Databricks, dest for local
        self.source_mlflow_client = MlflowClient(tracking_uri=config.tracking_uri)
        self.dest_mlflow_client = MlflowClient()  # Local client for destination
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
            # Check if source_experiment_name is actually an ID (numeric)
            if self.config.source_experiment_name.isdigit():
                source_experiment_id = self.config.source_experiment_name
                # Verify the experiment exists
                source_exp = self.source_mlflow_client.get_experiment(source_experiment_id)
                _logger.debug(
                    f"Using source experiment ID: {source_experiment_id} "
                    f"(name: {source_exp.name if source_exp else 'Unknown'})"
                )
            else:
                source_exp = self.source_mlflow_client.get_experiment_by_name(
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
            dest_experiment.experiment_id, source_experiment_id
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
                order_by=["timestamp_ms DESC"],
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
            experiment = self.dest_mlflow_client.get_experiment_by_name(experiment_name)
            if experiment is None:
                _logger.info(f"Creating destination experiment '{experiment_name}'")
                try:
                    experiment_id = self.dest_mlflow_client.create_experiment(experiment_name)
                    experiment = self.dest_mlflow_client.get_experiment(experiment_id)
                except Exception as create_error:
                    # Handle race condition where another process created it
                    if "already exists" in str(create_error):
                        _logger.info(
                            f"Experiment '{experiment_name}' was created by another process"
                        )
                        experiment = self.dest_mlflow_client.get_experiment_by_name(experiment_name)
                        if experiment is None:
                            raise MlflowException(
                                f"Failed to get experiment after creation race: {create_error}"
                            )
                    else:
                        raise create_error
            return experiment
        except MlflowException:
            raise
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
                # Since Databricks doesn't support trace_id in filters, we'll use timestamp only
                filter_string = f"timestamp_ms > {cursor_timestamp}"
                _logger.debug(f"Using filter: {filter_string}")

        current_cursor = cursor_trace

        while True:
            try:
                # Update filter if we have a new cursor from previous batch
                if current_cursor != cursor_trace:
                    source_trace_id = (
                        current_cursor.info.tags.get("mlflow.databricks.sourceTraceId")
                        or current_cursor.info.trace_id
                    )
                    cursor_timestamp = current_cursor.info.timestamp_ms
                    filter_string = f"timestamp_ms > {cursor_timestamp}"
                    _logger.debug(f"Updated filter after batch: {filter_string}")

                # Search for traces in source experiments
                # Note: We search without spans first to avoid loading too much data at once
                traces_page: PagedList[Trace] = self.source_client.search_traces(
                    experiment_ids=source_experiment_ids,
                    filter_string=filter_string,
                    max_results=batch_size,
                    page_token=page_token,
                    order_by=["timestamp_ms ASC"],  # Order by timestamp
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
                    synced_count, last_synced = self._sync_traces_batch(
                        traces_to_sync, dest_experiment_id
                    )
                    total_synced += synced_count

                    # Update cursor to the last successfully synced trace
                    if last_synced:
                        current_cursor = last_synced
                        _logger.debug(
                            f"Updated cursor to trace {last_synced.info.trace_id} "
                            f"(timestamp: {last_synced.info.timestamp_ms})"
                        )

                # Check if there are more pages
                page_token = traces_page.token
                if not page_token:
                    break

            except Exception as e:
                _logger.error(f"Error searching/syncing traces: {e}", exc_info=True)
                break

        if total_synced > 0:
            _logger.info(f"Synced {total_synced} traces in this cycle")

    def _sync_traces_batch(
        self, traces: list[Trace], dest_experiment_id: str
    ) -> tuple[int, Optional[Trace]]:
        """
        Sync a batch of traces to the destination in parallel.

        Returns:
            Tuple of (synced_count, last_synced_trace)
        """
        synced_count = 0
        last_synced_trace = None

        with ThreadPoolExecutor(max_workers=5, thread_name_prefix="TraceSync") as executor:
            futures = []
            for trace in traces:
                future = executor.submit(self._sync_single_trace, trace, dest_experiment_id)
                futures.append((trace, future))

            for trace, future in futures:
                try:
                    if future.result():
                        synced_count += 1
                        # Track the last successfully synced trace
                        # Since we process in timestamp order, the last one is the newest
                        last_synced_trace = trace
                except Exception as e:
                    _logger.error(f"Failed to sync trace {trace.info.trace_id}: {e}")

        return synced_count, last_synced_trace

    def _sync_single_trace(self, trace: Trace, dest_experiment_id: str) -> bool:
        """
        Sync a single trace to the destination experiment.

        Returns True if successful, False otherwise.
        """
        try:
            # Create a new trace info with the destination experiment ID
            # Since TraceInfo doesn't have _copy_with_overrides, we need to create a new object
            from mlflow.entities.trace_info import TraceInfo
            from mlflow.entities.trace_location import (
                MlflowExperimentLocation,
                TraceLocation,
                TraceLocationType,
            )

            # Create new trace location with destination experiment
            new_trace_location = TraceLocation(
                type=TraceLocationType.MLFLOW_EXPERIMENT,
                mlflow_experiment=MlflowExperimentLocation(experiment_id=dest_experiment_id),
            )

            # Create new trace info with updated location and tags
            # Generate a new trace ID for the destination to avoid conflicts
            new_trace_id = f"tr-{uuid.uuid4().hex}"

            # Filter out tags that will be set by the destination system
            tags_to_exclude = {"mlflow.artifactLocation", "mlflow.traceRequestId"}
            filtered_tags = {k: v for k, v in trace.info.tags.items() if k not in tags_to_exclude}

            new_trace_info = TraceInfo(
                trace_id=new_trace_id,
                trace_location=new_trace_location,
                request_time=trace.info.request_time,
                state=trace.info.state,
                request_preview=trace.info.request_preview,
                response_preview=trace.info.response_preview,
                client_request_id=trace.info.client_request_id,
                execution_duration=trace.info.execution_duration,
                trace_metadata=trace.info.trace_metadata.copy(),
                tags={
                    **filtered_tags,
                    "mlflow.databricks.sourceTraceId": trace.info.trace_id,
                    "mlflow.databricks.sourceExperimentId": trace.info.experiment_id,
                },
                assessments=trace.info.assessments.copy(),
            )

            # 1. Create the trace in the destination
            returned_trace_info = self.dest_client.start_trace(new_trace_info)

            # 2. Download trace data from source if not already present
            trace_data = trace.data
            if not trace_data:
                try:
                    # Download the trace data from the source
                    trace_data = self.source_client._download_trace_data(trace.info)
                except Exception as e:
                    # Check if it's a missing data error
                    error_msg = str(e)
                    if "missing span data" in error_msg or "MlflowTraceDataNotFound" in str(
                        type(e)
                    ):
                        _logger.info(
                            f"Trace {trace.info.trace_id} has no span data, "
                            "creating metadata-only trace"
                        )
                    else:
                        _logger.warning(
                            f"Failed to download trace data for trace {trace.info.trace_id}: {e}. "
                            "Creating metadata-only trace."
                        )
                    trace_data = None

            # 3. Upload trace data as artifact if available
            if trace_data:
                try:
                    self.dest_client._upload_trace_data(returned_trace_info, trace_data)
                except Exception as e:
                    _logger.error(
                        f"Failed to upload trace data for trace {trace.info.trace_id}: {e}"
                    )
            else:
                # Even without span data, the trace metadata was successfully synced
                _logger.debug(f"Synced trace metadata for {trace.info.trace_id} (no span data)")

            _logger.debug(
                f"Successfully synced trace {trace.info.trace_id} -> {returned_trace_info.trace_id}"
            )
            return True

        except Exception as e:
            _logger.error(f"Failed to sync trace {trace.info.trace_id}: {e}", exc_info=True)
            return False


# Global worker instance and lock
_sync_worker: Optional[DatabricksTraceSyncWorker] = None
_sync_worker_lock = threading.Lock()


def start_databricks_trace_sync(config: DatabricksTraceSyncConfig):
    """Start the Databricks trace synchronization worker."""
    global _sync_worker

    with _sync_worker_lock:
        if _sync_worker is not None:
            _logger.warning("Trace sync worker already running, skipping initialization")
            return

        _sync_worker = DatabricksTraceSyncWorker(config)
        _sync_worker.start()


def stop_databricks_trace_sync():
    """Stop the Databricks trace synchronization worker."""
    global _sync_worker

    with _sync_worker_lock:
        if _sync_worker is not None:
            _sync_worker.stop()
            _sync_worker = None
