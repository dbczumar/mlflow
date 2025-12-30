"""Online scoring processor for executing scorers on traces."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from mlflow.entities import Trace
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.harness import _compute_eval_scores, _log_assessments
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.online.constants import EXCLUDE_EVAL_RUN_TRACES_FILTER, MAX_TRACES_PER_JOB
from mlflow.genai.scorers.online.online_scorer import OnlineScorer
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler
from mlflow.genai.scorers.online.trace_checkpointer import OnlineTraceCheckpointManager
from mlflow.genai.scorers.online.trace_loader import OnlineTraceLoader
from mlflow.store.tracking.abstract_store import AbstractStore

_logger = logging.getLogger(__name__)


@dataclass
class TraceScoringTask:
    """A task to score a single trace with multiple scorers."""

    trace: Trace
    scorers: list[Scorer]


class OnlineTraceScoringProcessor:
    """Orchestrates online scoring of individual traces."""

    def __init__(
        self,
        trace_loader: OnlineTraceLoader,
        checkpoint_manager: OnlineTraceCheckpointManager,
        sampler: OnlineScorerSampler,
        experiment_id: str,
    ):
        self._trace_loader = trace_loader
        self._checkpoint_manager = checkpoint_manager
        self._sampler = sampler
        self._experiment_id = experiment_id

    @classmethod
    def create(
        cls,
        experiment_id: str,
        online_scorers: list[OnlineScorer],
        tracking_store: AbstractStore,
    ) -> "OnlineTraceScoringProcessor":
        """
        Factory method to create an OnlineTraceScoringProcessor with dependencies.

        Args:
            experiment_id: The experiment ID to process traces from.
            online_scorers: List of OnlineScorer instances.
            tracking_store: The tracking store instance.

        Returns:
            Configured OnlineTraceScoringProcessor instance.
        """
        return cls(
            trace_loader=OnlineTraceLoader(tracking_store),
            checkpoint_manager=OnlineTraceCheckpointManager(tracking_store, experiment_id),
            sampler=OnlineScorerSampler(online_scorers, tracking_store),
            experiment_id=experiment_id,
        )

    def process_traces(self) -> None:
        """
        Execute online scoring for the experiment.

        Fetches traces since the last checkpoint, applies sampling to select
        scorers, runs scoring in parallel, and updates the checkpoint.
        """
        if not self._sampler.configs:
            _logger.info("No scorer configs provided, skipping")
            return

        time_window = self._checkpoint_manager.calculate_time_window()
        _logger.info(
            f"Online scoring for experiment {self._experiment_id}: "
            f"time window [{time_window.min_trace_timestamp_ms}, "
            f"{time_window.max_trace_timestamp_ms}]"
        )

        tasks = self._fetch_and_sample_traces(
            time_window.min_trace_timestamp_ms, time_window.max_trace_timestamp_ms
        )

        if not tasks:
            _logger.info("No traces selected after sampling, skipping")
            self._checkpoint_manager.update_checkpoint_timestamp(time_window.max_trace_timestamp_ms)
            return

        _logger.info(f"Running scoring: {len(tasks)} trace tasks")

        sampled_trace_ids = list(tasks.keys())
        full_traces = self._trace_loader.fetch_traces(sampled_trace_ids)
        trace_map = {t.info.trace_id: t for t in full_traces}

        for trace_id, task in tasks.items():
            task.trace = trace_map.get(trace_id)

        self._execute_scoring(tasks)

        self._checkpoint_manager.update_checkpoint_timestamp(time_window.max_trace_timestamp_ms)

        _logger.info(f"Online scoring completed for experiment {self._experiment_id}")

    def _fetch_and_sample_traces(
        self,
        start_time_ms: int,
        end_time_ms: int,
    ) -> dict[str, TraceScoringTask]:
        """
        Fetch traces for each filter and apply sampling.

        Returns:
            Dictionary mapping trace_id to TraceScoringTask.
        """
        tasks: dict[str, TraceScoringTask] = {}

        # Group by filter string to fetch matching traces in a single query per filter
        for filter_string in self._sampler.get_filter_strings():
            trace_scorers = self._sampler.get_scorers_for_filter(filter_string, session_level=False)

            combined_filter = (
                f"{EXCLUDE_EVAL_RUN_TRACES_FILTER} AND {filter_string}"
                if filter_string
                else EXCLUDE_EVAL_RUN_TRACES_FILTER
            )
            trace_infos = self._trace_loader.fetch_trace_infos_between(
                self._experiment_id,
                start_time_ms,
                end_time_ms,
                combined_filter,
                MAX_TRACES_PER_JOB,
            )

            if not trace_infos:
                _logger.debug(f"No trace infos found for filter: {filter_string}")
                continue

            _logger.info(f"Found {len(trace_infos)} trace infos for filter: {filter_string}")

            for trace_info in trace_infos:
                trace_id = trace_info.trace_id
                if selected := self._sampler.sample(trace_id, trace_scorers):
                    # Store just the trace_id and scorers - we'll fetch full traces later
                    if trace_id not in tasks:
                        tasks[trace_id] = TraceScoringTask(trace=None, scorers=[])
                    tasks[trace_id].scorers.extend(selected)

        return tasks

    def _execute_scoring(
        self,
        tasks: dict[str, TraceScoringTask],
    ) -> None:
        """
        Execute trace scoring tasks in parallel.

        Args:
            tasks: Trace-level scoring tasks.
        """
        with ThreadPoolExecutor(
            max_workers=MLFLOW_GENAI_EVAL_MAX_WORKERS.get(),
            thread_name_prefix="OnlineScoring",
        ) as executor:
            futures = {}
            for task in tasks.values():
                if task.trace is None:
                    _logger.warning("Skipping task with no trace")
                    continue
                eval_item = EvalItem.from_trace(task.trace)
                future = executor.submit(
                    _compute_eval_scores, eval_item=eval_item, scorers=task.scorers
                )
                futures[future] = task

            for future in as_completed(futures):
                task = futures[future]
                try:
                    if feedbacks := future.result():
                        _log_assessments(run_id=None, trace=task.trace, assessments=feedbacks)
                except Exception as e:
                    _logger.warning(
                        f"Failed to score trace {task.trace.info.trace_id}: {e}",
                        exc_info=True,
                    )
