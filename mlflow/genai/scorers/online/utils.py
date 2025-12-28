"""Utilities for online scoring execution."""

from typing import Any

from mlflow.genai.scorers.online.checkpoint import CHECKPOINT_TAG
from mlflow.genai.scorers.online.processor import OnlineScoringProcessor
from mlflow.store.tracking.abstract_store import AbstractStore

# Re-export for backwards compatibility
ONLINE_SCORING_CHECKPOINT_TAG = CHECKPOINT_TAG


def execute_online_scoring(
    experiment_id: str,
    scorer_configs: list[dict[str, Any]],
    tracking_store: AbstractStore,
) -> None:
    """
    Execute online scoring for an experiment.

    This is the main entry point for online scoring, called by run_online_scorer_job.

    Args:
        experiment_id: The experiment ID to fetch traces from.
        scorer_configs: List of OnlineScorerConfig dicts specifying which scorers to run.
        tracking_store: The tracking store instance.
    """
    processor = OnlineScoringProcessor.create(
        experiment_id=experiment_id,
        scorer_configs=scorer_configs,
        tracking_store=tracking_store,
    )
    processor.process_traces()
