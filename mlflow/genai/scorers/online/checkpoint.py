"""Checkpoint management for online scoring."""

import logging
import time
from enum import Enum

from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.store.tracking.abstract_store import AbstractStore

_logger = logging.getLogger(__name__)


class CheckpointType(Enum):
    """Type of checkpoint for online scoring."""

    TRACE = "trace"
    SESSION = "session"


# Checkpoint tag prefix for tracking last processed timestamp
_CHECKPOINT_TAG_PREFIX = "mlflow.latestOnlineScoring"

# Default lookback period when no checkpoint exists (1 hour)
_DEFAULT_LOOKBACK_MS = 60 * 60 * 1000


def _get_checkpoint_tag(checkpoint_type: CheckpointType) -> str:
    """
    Get the checkpoint tag name for a given checkpoint type.

    Args:
        checkpoint_type: The type of checkpoint (TRACE or SESSION).

    Returns:
        The full checkpoint tag name.
    """
    return f"{_CHECKPOINT_TAG_PREFIX}.{checkpoint_type.value}.timestampMs"


class OnlineCheckpointManager:
    """Manages checkpoint timestamps for online scoring progress tracking."""

    def __init__(
        self,
        tracking_store: AbstractStore,
        experiment_id: str,
        checkpoint_type: CheckpointType = CheckpointType.TRACE,
    ):
        self._tracking_store = tracking_store
        self._experiment_id = experiment_id
        self._checkpoint_type = checkpoint_type
        self._checkpoint_tag = _get_checkpoint_tag(checkpoint_type)

    def get_checkpoint_timestamp(self) -> int | None:
        """
        Get the last processed timestamp from the experiment checkpoint tag.

        Returns:
            The checkpoint timestamp in milliseconds, or None if no checkpoint exists.
        """
        try:
            experiment = self._tracking_store.get_experiment(self._experiment_id)
            if checkpoint := experiment.tags.get(self._checkpoint_tag):
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
            ExperimentTag(self._checkpoint_tag, str(timestamp_ms)),
        )

    def calculate_time_window(self) -> tuple[int, int, int | None]:
        """
        Calculate the time window for trace fetching.

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
