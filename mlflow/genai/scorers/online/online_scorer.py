"""
Online scorer entities and configuration.

This module contains entities for online scorer configuration used by the store layer
and online scoring infrastructure.
"""

from dataclasses import dataclass


@dataclass
class OnlineScorer:
    """An online scorer with sampling configuration."""

    name: str
    experiment_id: str
    serialized_scorer: str
    sample_rate: float
    filter_string: str | None = None


@dataclass
class CompletedSession:
    """A completed session with metadata."""

    session_id: str
    trace_count: int
    first_trace_timestamp_ms: int
    last_trace_timestamp_ms: int


class OnlineScoringConfig:
    """
    Internal entity representing the online configuration for a scorer.

    This configuration controls how a scorer is applied to traces in an online/real-time
    manner. It defines sampling rates and optional filters for selecting which traces
    should be scored.
    """

    def __init__(
        self,
        online_scoring_config_id: str,
        scorer_id: str,
        sample_rate: float,
        filter_string: str | None = None,
    ):
        self._online_scoring_config_id = online_scoring_config_id
        self._scorer_id = scorer_id
        self._sample_rate = sample_rate
        self._filter_string = filter_string

    @property
    def online_scoring_config_id(self) -> str:
        """The unique identifier for this online config."""
        return self._online_scoring_config_id

    @property
    def scorer_id(self) -> str:
        """The ID of the scorer this config belongs to."""
        return self._scorer_id

    @property
    def sample_rate(self) -> float:
        """The fraction of traces to sample (between 0 and 1)."""
        return self._sample_rate

    @property
    def filter_string(self) -> str | None:
        """Optional filter expression to select which traces should be scored."""
        return self._filter_string

    def __repr__(self):
        return (
            f"<OnlineScoringConfig(online_scoring_config_id={self.online_scoring_config_id}, "
            f"scorer_id='{self.scorer_id}', sample_rate={self.sample_rate})>"
        )

    def to_dict(self) -> dict[str, str | float]:
        """Convert the entity to a dictionary for JSON serialization."""
        result: dict[str, str | float] = {
            "online_scoring_config_id": self.online_scoring_config_id,
            "scorer_id": self.scorer_id,
            "sample_rate": self.sample_rate,
        }
        if self.filter_string is not None:
            result["filter_string"] = self.filter_string
        return result
