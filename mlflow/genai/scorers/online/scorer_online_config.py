"""
Internal entity for scorer online configuration.

This is an internal entity used by the store layer and is not part of the public API.
"""


class ScorerOnlineConfig:
    """
    Internal entity representing the online configuration for a scorer.

    This configuration controls how a scorer is applied to traces in an online/real-time
    manner. It defines sampling rates and optional filters for selecting which traces
    should be scored.
    """

    def __init__(
        self,
        scorer_online_config_id: str,
        scorer_id: str,
        sample_rate: float,
        filter_string: str | None = None,
    ):
        self._scorer_online_config_id = scorer_online_config_id
        self._scorer_id = scorer_id
        self._sample_rate = sample_rate
        self._filter_string = filter_string

    @property
    def scorer_online_config_id(self) -> str:
        """The unique identifier for this online config."""
        return self._scorer_online_config_id

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
            f"<ScorerOnlineConfig(scorer_online_config_id={self.scorer_online_config_id}, "
            f"scorer_id='{self.scorer_id}', sample_rate={self.sample_rate})>"
        )

    def to_dict(self) -> dict[str, str | float]:
        """Convert the entity to a dictionary for JSON serialization."""
        result: dict[str, str | float] = {
            "scorer_online_config_id": self.scorer_online_config_id,
            "scorer_id": self.scorer_id,
            "sample_rate": self.sample_rate,
        }
        if self.filter_string is not None:
            result["filter_string"] = self.filter_string
        return result
