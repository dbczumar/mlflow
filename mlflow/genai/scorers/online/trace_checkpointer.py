"""Checkpoint management for trace-level online scoring."""

import logging
import time
from dataclasses import dataclass

from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.store.tracking.abstract_store import AbstractStore

_logger = logging.getLogger(__name__)

# Checkpoint tag for tracking last processed trace timestamp
TRACE_CHECKPOINT_TAG = "mlflow.latestOnlineScoring.trace.timestampMs"

# Default lookback period when no checkpoint exists (1 hour)
_DEFAULT_LOOKBACK_MS = 60 * 60 * 1000

# Maximum lookback period to prevent getting stuck on old failing traces (1 hour)
_MAX_LOOKBACK_MS = 60 * 60 * 1000


@dataclass
class OnlineTraceScoringTimeWindow:
    """Time window for trace-level online scoring."""

    min_trace_timestamp_ms: int
    max_trace_timestamp_ms: int


class OnlineTraceCheckpointManager:
    """Manages checkpoint timestamps for trace-level online scoring."""

    def __init__(self, tracking_store: AbstractStore, experiment_id: str):
        self._tracking_store = tracking_store
        self._experiment_id = experiment_id

    def get_checkpoint_timestamp(self) -> int | None:
        """
        Get the last processed trace timestamp from the experiment checkpoint tag.

        Returns:
            The checkpoint timestamp in milliseconds, or None if no checkpoint exists.
        """
        try:
            experiment = self._tracking_store.get_experiment(self._experiment_id)
            if checkpoint := experiment.tags.get(TRACE_CHECKPOINT_TAG):
                return int(checkpoint)
        except (TypeError, ValueError):
            pass
        return None

    def update_checkpoint_timestamp(self, timestamp_ms: int) -> None:
        """
        Update the checkpoint tag with a new timestamp.

        Args:
            timestamp_ms: The new checkpoint timestamp in milliseconds.
        """
        self._tracking_store.set_experiment_tag(
            self._experiment_id,
            ExperimentTag(TRACE_CHECKPOINT_TAG, str(timestamp_ms)),
        )

    def calculate_time_window(self) -> OnlineTraceScoringTimeWindow:
        """
        Calculate the time window for trace scoring.

        Enforces a maximum lookback of 1 hour to prevent getting stuck on persistently
        failing traces. If the checkpoint is older than 1 hour, uses current_time - 1 hour
        instead to skip over old problematic traces.

        Returns:
            OnlineTraceScoringTimeWindow with min and max trace timestamps.
            min_trace_timestamp_ms is the checkpoint if it exists and is within the last hour,
            otherwise now - 1 hour.
            max_trace_timestamp_ms is the current time.
        """
        current_time_ms = int(time.time() * 1000)
        current_checkpoint = self.get_checkpoint_timestamp()

        # Start from checkpoint, but never look back more than 1 hour
        min_lookback_time_ms = current_time_ms - _MAX_LOOKBACK_MS

        if current_checkpoint is not None:
            # Use the more recent of: checkpoint or (current_time - 1 hour)
            min_trace_timestamp_ms = max(current_checkpoint, min_lookback_time_ms)
        else:
            min_trace_timestamp_ms = min_lookback_time_ms

        return OnlineTraceScoringTimeWindow(
            min_trace_timestamp_ms=min_trace_timestamp_ms,
            max_trace_timestamp_ms=current_time_ms,
        )
