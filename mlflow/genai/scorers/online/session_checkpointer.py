"""Checkpoint management for session-level online scoring."""

import logging
import time
from dataclasses import dataclass

from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.genai.scorers.online.constants import (
    MAX_LOOKBACK_MS,
    SESSION_CHECKPOINT_TAG,
    SESSION_COMPLETION_BUFFER_MS,
)
from mlflow.store.tracking.abstract_store import AbstractStore

_logger = logging.getLogger(__name__)


@dataclass
class OnlineSessionScoringTimeWindow:
    """Time window for session-level online scoring."""

    min_last_trace_timestamp_ms: int
    max_last_trace_timestamp_ms: int
    min_session_id: str | None = None


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
                # Parse format: "timestamp_ms:session_id" or just "timestamp_ms" (legacy)
                parts = checkpoint.split(":", 1)
                return int(parts[0])
        except (TypeError, ValueError):
            pass
        return None

    def get_checkpoint_session_id(self) -> str | None:
        """
        Get the last processed session ID from the experiment checkpoint tag.

        Returns:
            The session ID, or None if no checkpoint exists or it's a legacy
            timestamp-only checkpoint.
        """
        try:
            experiment = self._tracking_store.get_experiment(self._experiment_id)
            if checkpoint := experiment.tags.get(SESSION_CHECKPOINT_TAG):
                # Parse format: "timestamp_ms:session_id"
                parts = checkpoint.split(":", 1)
                if len(parts) == 2:
                    return parts[1]
        except (TypeError, ValueError):
            pass
        return None

    def update_checkpoint(self, timestamp_ms: int, session_id: str) -> None:
        """
        Update the checkpoint tag with a new timestamp and session ID.

        Args:
            timestamp_ms: The new checkpoint timestamp in milliseconds.
            session_id: The session ID of the last processed session.
        """
        checkpoint_value = f"{timestamp_ms}:{session_id}"
        self._tracking_store.set_experiment_tag(
            self._experiment_id,
            ExperimentTag(SESSION_CHECKPOINT_TAG, checkpoint_value),
        )

    def update_checkpoint_timestamp(self, timestamp_ms: int) -> None:
        """
        Update the checkpoint tag with a new timestamp (legacy method for backward compatibility).

        Args:
            timestamp_ms: The new checkpoint timestamp in milliseconds.
        """
        self._tracking_store.set_experiment_tag(
            self._experiment_id,
            ExperimentTag(SESSION_CHECKPOINT_TAG, str(timestamp_ms)),
        )

    def calculate_time_window(self) -> OnlineSessionScoringTimeWindow:
        """
        Calculate the time window for session scoring.

        Enforces a maximum lookback period to prevent getting stuck on persistently
        failing sessions. If the checkpoint is older than MAX_LOOKBACK_MS, uses
        current_time - MAX_LOOKBACK_MS instead to skip over old problematic sessions.

        Returns:
            OnlineSessionScoringTimeWindow with min and max last trace timestamps and
            optional min session ID for tiebreaking.
            min_last_trace_timestamp_ms is the checkpoint if it exists and is within
            the lookback period, otherwise now - MAX_LOOKBACK_MS.
            max_last_trace_timestamp_ms is current time - session completion buffer.
            min_session_id is the session ID from checkpoint for handling timestamp ties.
        """
        current_time_ms = int(time.time() * 1000)
        current_checkpoint = self.get_checkpoint_timestamp()
        checkpoint_session_id = self.get_checkpoint_session_id()

        # Start from checkpoint, but never look back more than MAX_LOOKBACK_MS
        min_lookback_time_ms = current_time_ms - MAX_LOOKBACK_MS

        if current_checkpoint is not None:
            min_last_trace_timestamp_ms = max(current_checkpoint, min_lookback_time_ms)
        else:
            min_last_trace_timestamp_ms = min_lookback_time_ms

        max_last_trace_timestamp_ms = current_time_ms - SESSION_COMPLETION_BUFFER_MS

        return OnlineSessionScoringTimeWindow(
            min_last_trace_timestamp_ms=min_last_trace_timestamp_ms,
            max_last_trace_timestamp_ms=max_last_trace_timestamp_ms,
            min_session_id=checkpoint_session_id,
        )
