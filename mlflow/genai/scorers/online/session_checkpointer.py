"""Checkpoint management for session-level online scoring."""

import logging
import time

from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.store.tracking.abstract_store import AbstractStore

_logger = logging.getLogger(__name__)

# Checkpoint tag for tracking last processed session timestamp
SESSION_CHECKPOINT_TAG = "mlflow.latestOnlineScoring.session.timestampMs"

# Default lookback period when no checkpoint exists (1 hour)
_DEFAULT_LOOKBACK_MS = 60 * 60 * 1000


class OnlineSessionCheckpointManager:
    """Manages checkpoint timestamps for session-level online scoring."""

    def __init__(self, tracking_store: AbstractStore, experiment_id: str):
        self._tracking_store = tracking_store
        self._experiment_id = experiment_id

    def get_checkpoint_timestamp(self) -> int | None:
        """
        Get the last processed session timestamp from the experiment checkpoint tag.

        Returns:
            The checkpoint timestamp in milliseconds, or None if no checkpoint exists.
        """
        try:
            experiment = self._tracking_store.get_experiment(self._experiment_id)
            if checkpoint := experiment.tags.get(SESSION_CHECKPOINT_TAG):
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
            ExperimentTag(SESSION_CHECKPOINT_TAG, str(timestamp_ms)),
        )

    def calculate_time_window(self) -> tuple[int, int, int | None]:
        """
        Calculate the time window for session fetching.

        Returns:
            Tuple of (start_timestamp_ms, end_timestamp_ms, current_checkpoint).
            start_timestamp_ms is the checkpoint if it exists, otherwise now - 1 hour.
            end_timestamp_ms is the current time.
        """
        current_time_ms = int(time.time() * 1000)
        current_checkpoint = self.get_checkpoint_timestamp()

        if current_checkpoint is not None:
            start_time_ms = current_checkpoint
        else:
            start_time_ms = current_time_ms - _DEFAULT_LOOKBACK_MS

        return start_time_ms, current_time_ms, current_checkpoint
