"""Online scoring processor for executing scorers on traces."""

import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from mlflow.entities import Trace
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.harness import _compute_eval_scores, _log_assessments
from mlflow.genai.evaluation.session_utils import evaluate_session_level_scorers
from mlflow.genai.scorers.online.checkpoint import OnlineCheckpointManager
from mlflow.genai.scorers.online.config import (
    OnlineScorerConfig,
    SessionScoringTask,
    TraceScoringTask,
)
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler
from mlflow.genai.scorers.online.trace_loader import TraceLoader
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.constant import TraceMetadataKey

_logger = logging.getLogger(__name__)

# Maximum traces to process per job run
_MAX_TRACES_PER_JOB = 500


class OnlineScoringProcessor:
    """Orchestrates online scoring of traces."""

    def __init__(
        self,
        trace_loader: TraceLoader,
        checkpoint_manager: OnlineCheckpointManager,
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
        scorer_configs: list[dict[str, Any]],
        tracking_store: AbstractStore,
    ) -> "OnlineScoringProcessor":
        """
        Factory method to create an OnlineScoringProcessor with dependencies.

        Args:
            experiment_id: The experiment ID to process traces from.
            scorer_configs: List of OnlineScorerConfig dicts.
            tracking_store: The tracking store instance.

        Returns:
            Configured OnlineScoringProcessor instance.
        """
        configs = [OnlineScorerConfig(**c) for c in scorer_configs]
        return cls(
            trace_loader=TraceLoader(tracking_store),
            checkpoint_manager=OnlineCheckpointManager(tracking_store, experiment_id),
            sampler=OnlineScorerSampler(configs),
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

        # Calculate time window
        start_time_ms, end_time_ms, current_checkpoint = (
            self._checkpoint_manager.calculate_time_window()
        )
        _logger.info(
            f"Online scoring for experiment {self._experiment_id}: "
            f"time window [{start_time_ms}, {end_time_ms}]"
        )

        # Fetch traces and apply sampling per filter group
        single_turn_tasks, session_tasks = self._fetch_and_sample_traces(start_time_ms, end_time_ms)

        if not single_turn_tasks and not session_tasks:
            _logger.info("No traces selected after sampling, skipping")
            return

        _logger.info(
            f"Running scoring: {len(single_turn_tasks)} single-turn tasks, "
            f"{len(session_tasks)} session tasks"
        )

        # Execute scoring
        self._execute_scoring(single_turn_tasks, session_tasks, current_checkpoint)

        _logger.info(f"Online scoring completed for experiment {self._experiment_id}")

    def _fetch_and_sample_traces(
        self,
        start_time_ms: int,
        end_time_ms: int,
    ) -> tuple[dict[str, TraceScoringTask], dict[str, SessionScoringTask]]:
        """Fetch traces for each filter and apply sampling."""
        single_turn_tasks: dict[str, TraceScoringTask] = {}
        session_tasks: dict[str, SessionScoringTask] = {}

        for filter_string in self._sampler.get_filter_strings():
            single_turn_scorers = self._sampler.get_scorers_for_filter(
                filter_string, session_level=False
            )
            session_scorers = self._sampler.get_scorers_for_filter(
                filter_string, session_level=True
            )

            traces = self._trace_loader.fetch_trace_infos_between(
                self._experiment_id,
                start_time_ms,
                end_time_ms,
                filter_string,
                _MAX_TRACES_PER_JOB,
            )

            if not traces:
                _logger.debug(f"No traces found for filter: {filter_string}")
                continue

            _logger.info(f"Found {len(traces)} traces for filter: {filter_string}")

            # Sample single-turn scorers per trace
            for trace in traces:
                trace_id = trace.info.trace_id
                if selected := self._sampler.sample(trace_id, single_turn_scorers):
                    single_turn_tasks.setdefault(
                        trace_id, TraceScoringTask(trace=trace, scorers=[])
                    ).scorers.extend(selected)

            # Sample session-level scorers per session
            if session_scorers:
                session_groups = self._group_traces_by_session(traces)
                for session_id, session_traces in session_groups.items():
                    if selected := self._sampler.sample(session_id, session_scorers):
                        session_tasks.setdefault(
                            session_id,
                            SessionScoringTask(traces=session_traces, scorers=[]),
                        ).scorers.extend(selected)

        return single_turn_tasks, session_tasks

    def _group_traces_by_session(self, traces: list[Trace]) -> dict[str, list[Trace]]:
        """Group traces by their session ID."""
        session_groups: dict[str, list[Trace]] = defaultdict(list)

        for trace in traces:
            trace_metadata = trace.info.trace_metadata or {}
            if session_id := trace_metadata.get(TraceMetadataKey.TRACE_SESSION):
                session_groups[session_id].append(trace)

        # Sort traces within each session by timestamp
        for session_id in session_groups:
            session_groups[session_id] = sorted(
                session_groups[session_id],
                key=lambda t: t.info.timestamp_ms,
            )

        return dict(session_groups)

    def _execute_scoring(
        self,
        single_turn_tasks: dict[str, TraceScoringTask],
        session_tasks: dict[str, SessionScoringTask],
        current_checkpoint: int | None,
    ) -> None:
        """Execute scoring tasks and update checkpoints."""
        checkpoint_lock = threading.Lock()
        latest_checkpoint = current_checkpoint

        def update_checkpoint_if_needed(timestamp_ms: int | None) -> None:
            nonlocal latest_checkpoint
            if timestamp_ms is None:
                return
            with checkpoint_lock:
                if latest_checkpoint is None or timestamp_ms > latest_checkpoint:
                    self._checkpoint_manager.update_checkpoint_timestamp(timestamp_ms)
                    latest_checkpoint = timestamp_ms

        with ThreadPoolExecutor(
            max_workers=MLFLOW_GENAI_EVAL_MAX_WORKERS.get(),
            thread_name_prefix="OnlineScoring",
        ) as executor:
            # Phase 1: Single-turn scorers
            self._execute_single_turn_tasks(
                executor, single_turn_tasks, update_checkpoint_if_needed
            )

            # Phase 2: Session-level scorers
            self._execute_session_tasks(executor, session_tasks, update_checkpoint_if_needed)

    def _execute_single_turn_tasks(
        self,
        executor: ThreadPoolExecutor,
        tasks: dict[str, TraceScoringTask],
        update_checkpoint: Callable[[int | None], None],
    ) -> None:
        """Execute single-turn scoring tasks."""
        futures = {}
        for task in tasks.values():
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
                update_checkpoint(task.trace.info.timestamp_ms)
            except Exception as e:
                _logger.warning(
                    f"Failed to score trace {task.trace.info.trace_id}: {e}",
                    exc_info=True,
                )

    def _execute_session_tasks(
        self,
        executor: ThreadPoolExecutor,
        tasks: dict[str, SessionScoringTask],
        update_checkpoint: Callable[[int | None], None],
    ) -> None:
        """Execute session-level scoring tasks."""
        futures = {}
        for session_id, task in tasks.items():
            session_items = [EvalItem.from_trace(t) for t in task.traces]
            future = executor.submit(
                evaluate_session_level_scorers,
                session_id=session_id,
                session_items=session_items,
                multi_turn_scorers=task.scorers,
            )
            futures[future] = (session_id, task)

        for future in as_completed(futures):
            session_id, task = futures[future]
            try:
                result = future.result()
                for trace_id, feedbacks in result.items():
                    if feedbacks:
                        trace = next(
                            (t for t in task.traces if t.info.trace_id == trace_id),
                            None,
                        )
                        if trace:
                            _log_assessments(run_id=None, trace=trace, assessments=feedbacks)

                max_ts = max(
                    (t.info.timestamp_ms for t in task.traces if t.info.timestamp_ms),
                    default=None,
                )
                update_checkpoint(max_ts)
            except Exception as e:
                _logger.warning(
                    f"Failed to score session {session_id}: {e}",
                    exc_info=True,
                )
