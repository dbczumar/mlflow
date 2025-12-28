"""Dense sampling strategy for online scoring."""

import json
import random

from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.online.config import OnlineScorerConfig


class ScorerSampler:
    """
    Samples scorers for traces using dense sampling strategy.

    Dense sampling ensures traces that are selected get thorough coverage:
    - Sort scorers by sample_rate descending
    - Use conditional probability: if a scorer is rejected, skip all lower-rate scorers
    """

    def __init__(self, configs: list[OnlineScorerConfig]):
        self.configs = configs
        self._sample_rates: dict[str, float] = {c.serialized_scorer: c.sample_rate for c in configs}
        self._scorers: dict[str, Scorer] = {}
        for config in configs:
            if config.serialized_scorer not in self._scorers:
                scorer_dict = json.loads(config.serialized_scorer)
                self._scorers[config.serialized_scorer] = Scorer.model_validate(scorer_dict)

    def get_filter_strings(self) -> set[str | None]:
        """Get all unique filter strings from configs."""
        return {c.filter_string for c in self.configs}

    def get_scorers_for_filter(
        self, filter_string: str | None, session_level: bool
    ) -> list[Scorer]:
        """Get scorers matching the filter string and session level."""
        return [
            self._scorers[c.serialized_scorer]
            for c in self.configs
            if c.filter_string == filter_string
            and self._scorers[c.serialized_scorer].is_session_level_scorer == session_level
        ]

    def sample(self, scorers: list[Scorer]) -> list[Scorer]:
        """
        Apply dense sampling to select scorers.

        Returns a subset of scorers selected via conditional probability waterfall.
        """
        if not scorers:
            return []

        # Sort by sample rate descending
        sorted_scorers = sorted(
            scorers,
            key=lambda s: self._sample_rates.get(s.model_dump_json(), 1.0),
            reverse=True,
        )

        selected = []
        prev_rate = 1.0

        for scorer in sorted_scorers:
            rate = self._sample_rates.get(scorer.model_dump_json(), 1.0)
            conditional_rate = rate / prev_rate if prev_rate > 0 else 0

            if random.random() > conditional_rate:
                break

            selected.append(scorer)
            prev_rate = rate

        return selected
