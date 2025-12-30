"""Dense sampling strategy for online scoring."""

import hashlib
import json
import logging
from typing import TYPE_CHECKING

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.scorer_utils import (
    build_gateway_model,
    extract_endpoint_ref,
    extract_model_from_serialized_scorer,
    is_gateway_model,
    update_model_in_serialized_scorer,
)

if TYPE_CHECKING:
    from mlflow.genai.scorers.online.processor import OnlineScorer
    from mlflow.store.tracking.abstract_store import AbstractStore

_logger = logging.getLogger(__name__)


class OnlineScorerSampler:
    """
    Samples scorers for traces using dense sampling strategy.

    Dense sampling ensures traces that are selected get thorough coverage:
    - Sort scorers by sample_rate descending
    - Use conditional probability: if a scorer is rejected, skip all lower-rate scorers
    """

    def __init__(
        self,
        configs: list["OnlineScorer"],
        tracking_store: "AbstractStore",
    ):
        self.configs = configs
        # Map scorer name -> sample rate and scorer name -> Scorer
        self._sample_rates: dict[str, float] = {}
        self._scorers: dict[str, Scorer] = {}
        for config in configs:
            scorer_dict = json.loads(config.serialized_scorer)

            # Resolve gateway endpoint IDs to names
            model = extract_model_from_serialized_scorer(scorer_dict)
            if is_gateway_model(model):
                endpoint_ref = extract_endpoint_ref(model)
                try:
                    endpoint = tracking_store.get_gateway_endpoint(endpoint_id=endpoint_ref)
                    new_model = build_gateway_model(endpoint.name)
                    scorer_dict = update_model_in_serialized_scorer(scorer_dict, new_model)
                except MlflowException:
                    _logger.warning(
                        f"Skipping scorer '{scorer_dict.get('name')}': "
                        f"failed to resolve gateway endpoint from ID '{endpoint_ref}'"
                    )
                    continue

            scorer = Scorer.model_validate(scorer_dict)
            self._sample_rates[scorer.name] = config.sample_rate
            self._scorers[scorer.name] = scorer

    def get_filter_strings(self) -> set[str | None]:
        """Get all unique filter strings from configs."""
        return {c.filter_string for c in self.configs}

    def get_scorers_for_filter(
        self, filter_string: str | None, session_level: bool
    ) -> list[Scorer]:
        """Get scorers matching the filter string and session level."""
        result = []
        for config in self.configs:
            scorer_dict = json.loads(config.serialized_scorer)
            scorer = self._scorers.get(scorer_dict.get("name"))
            if (
                scorer
                and config.filter_string == filter_string
                and scorer.is_session_level_scorer == session_level
            ):
                result.append(scorer)
        return result

    def sample(self, entity_id: str, scorers: list[Scorer]) -> list[Scorer]:
        """
        Apply dense sampling to select scorers for an entity.

        Args:
            entity_id: The trace ID or session ID to sample for.
            scorers: List of scorers to sample from.

        Returns:
            A subset of scorers selected via conditional probability waterfall.
        """
        if not scorers:
            return []

        # Sort by sample rate descending
        sorted_scorers = sorted(
            scorers,
            key=lambda s: self._sample_rates.get(s.name, 1.0),
            reverse=True,
        )

        selected = []
        prev_rate = 1.0

        for scorer in sorted_scorers:
            rate = self._sample_rates.get(scorer.name, 1.0)
            conditional_rate = rate / prev_rate if prev_rate > 0 else 0

            # Hash entity_id + scorer name to get deterministic value in [0, 1]
            hash_input = f"{entity_id}:{scorer.name}"
            hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16) / (2**256)

            if hash_value > conditional_rate:
                break

            selected.append(scorer)
            prev_rate = rate

        _logger.debug(
            f"Sampled {len(selected)}/{len(scorers)} scorers for entity {entity_id[:8]}..."
        )
        return selected

    def log_sampling_stats(self, sampled_counts: dict[str, int], total_entities: int) -> None:
        """
        Log sampling statistics.

        Args:
            sampled_counts: Dict mapping scorer name to count of entities sampled.
            total_entities: Total number of entities (traces/sessions) considered.
        """
        _logger.info(f"Sampling stats for {total_entities} entities:")
        for scorer_name, count in sampled_counts.items():
            rate = count / total_entities if total_entities > 0 else 0
            _logger.info(f"  {scorer_name}: {count}/{total_entities} ({rate:.1%})")
