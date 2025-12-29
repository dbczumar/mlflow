"""Online scoring processor for executing scorers on traces."""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from mlflow.entities import Trace, TraceInfo
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.harness import _compute_eval_scores, _log_assessments
from mlflow.genai.evaluation.session_utils import evaluate_session_level_scorers
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.online.online_scorer import OnlineScorer
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler
from mlflow.genai.scorers.online.trace_checkpointer import OnlineTraceCheckpointManager
from mlflow.genai.scorers.online.trace_loader import OnlineTraceLoader
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.constant import TraceMetadataKey

_logger = logging.getLogger(__name__)

# Maximum traces to process per job run
_MAX_TRACES_PER_JOB = 500


@dataclass
class TraceScoringTask:
    """A task to score a single trace with multiple scorers."""

    trace: Trace
    scorers: list[Scorer]


@dataclass
class SessionScoringTask:
    """A task to score a session (multiple traces) with multiple scorers."""

    traces: list[Trace]
    scorers: list[Scorer]
    trace_infos: list[TraceInfo] | None = None  # Used temporarily during fetching


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
            sampler=OnlineScorerSampler(online_scorers),
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

        # Fetch trace infos and apply sampling per filter group
        single_turn_tasks, session_tasks = self._fetch_and_sample_traces(start_time_ms, end_time_ms)

        if not single_turn_tasks and not session_tasks:
            _logger.info("No traces selected after sampling, skipping")
            self._checkpoint_manager.update_checkpoint_timestamp(end_time_ms)
            return

        _logger.info(
            f"Running scoring: {len(single_turn_tasks)} single-turn tasks, "
            f"{len(session_tasks)} session tasks"
        )

        # Fetch full traces only for sampled trace IDs
        all_sampled_trace_ids = set(single_turn_tasks.keys())
        for task in session_tasks.values():
            all_sampled_trace_ids.update(ti.trace_id for ti in task.trace_infos)

        full_traces = self._trace_loader.fetch_traces(list(all_sampled_trace_ids))
        trace_map = {t.info.trace_id: t for t in full_traces}

        # Populate tasks with full traces
        for trace_id, task in single_turn_tasks.items():
            task.trace = trace_map.get(trace_id)

        for task in session_tasks.values():
            task.traces = [
                trace_map[ti.trace_id] for ti in task.trace_infos if ti.trace_id in trace_map
            ]
            # Sort traces within session by timestamp
            task.traces.sort(key=lambda t: t.info.timestamp_ms)

        # Execute scoring
        self._execute_scoring(single_turn_tasks, session_tasks)

        # Update checkpoint after scoring
        self._checkpoint_manager.update_checkpoint_timestamp(end_time_ms)

        _logger.info(f"Online scoring completed for experiment {self._experiment_id}")

    def _fetch_and_sample_traces(
        self,
        start_time_ms: int,
        end_time_ms: int,
    ) -> tuple[dict[str, TraceScoringTask], dict[str, SessionScoringTask]]:
        """
        Fetch traces for each filter and apply sampling.

        Returns:
            Tuple of (single_turn_tasks, session_tasks).
        """
        single_turn_tasks: dict[str, TraceScoringTask] = {}
        session_tasks: dict[str, SessionScoringTask] = {}

        for filter_string in self._sampler.get_filter_strings():
            single_turn_scorers = self._sampler.get_scorers_for_filter(
                filter_string, session_level=False
            )
            session_scorers = self._sampler.get_scorers_for_filter(
                filter_string, session_level=True
            )

            trace_infos = self._trace_loader.fetch_trace_infos_between(
                self._experiment_id,
                start_time_ms,
                end_time_ms,
                filter_string,
                _MAX_TRACES_PER_JOB,
            )

            if not trace_infos:
                _logger.debug(f"No trace infos found for filter: {filter_string}")
                continue

            _logger.info(f"Found {len(trace_infos)} trace infos for filter: {filter_string}")

            # Sample scorers for each trace
            sampled_trace_ids = []
            for trace_info in trace_infos:
                trace_id = trace_info.trace_id
                if selected := self._sampler.sample(trace_id, single_turn_scorers):
                    sampled_trace_ids.append(trace_id)
                    # Store just the trace_id and scorers - we'll fetch full traces later
                    if trace_id not in single_turn_tasks:
                        single_turn_tasks[trace_id] = TraceScoringTask(trace=None, scorers=[])
                    single_turn_tasks[trace_id].scorers.extend(selected)

            # For session-level scorers, we need to group traces by session
            if session_scorers:
                # Collect trace_ids that have sessions
                session_trace_infos = [
                    ti
                    for ti in trace_infos
                    if ti.trace_metadata and ti.trace_metadata.get(TraceMetadataKey.TRACE_SESSION)
                ]

                # Group by session_id
                session_groups = defaultdict(list)
                for trace_info in session_trace_infos:
                    session_id = trace_info.trace_metadata[TraceMetadataKey.TRACE_SESSION]
                    session_groups[session_id].append(trace_info)

                # Sample session-level scorers per session
                for session_id, session_trace_infos in session_groups.items():
                    if selected := self._sampler.sample(session_id, session_scorers):
                        # Store trace_ids for this session
                        trace_ids = [ti.trace_id for ti in session_trace_infos]
                        sampled_trace_ids.extend(trace_ids)
                        if session_id not in session_tasks:
                            session_tasks[session_id] = SessionScoringTask(traces=[], scorers=[])
                        session_tasks[session_id].scorers.extend(selected)
                        # Store trace_infos (we'll fetch full traces later)
                        session_tasks[session_id].trace_infos = session_trace_infos

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
    ) -> None:
        """
        Execute scoring tasks.

        Args:
            single_turn_tasks: Single-turn scoring tasks.
            session_tasks: Session-level scoring tasks.
        """
        with ThreadPoolExecutor(
            max_workers=MLFLOW_GENAI_EVAL_MAX_WORKERS.get(),
            thread_name_prefix="OnlineScoring",
        ) as executor:
            # Phase 1: Single-turn scorers
            self._execute_single_turn_tasks(executor, single_turn_tasks)

            # Phase 2: Session-level scorers
            self._execute_session_tasks(executor, session_tasks)

    def _execute_single_turn_tasks(
        self,
        executor: ThreadPoolExecutor,
        tasks: dict[str, TraceScoringTask],
    ) -> None:
        """Execute single-turn scoring tasks."""
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

    def _execute_session_tasks(
        self,
        executor: ThreadPoolExecutor,
        tasks: dict[str, SessionScoringTask],
    ) -> None:
        """Execute session-level scoring tasks."""
        futures = {}
        for session_id, task in tasks.items():
            if not task.traces:
                _logger.warning(f"Skipping session {session_id} with no traces")
                continue
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
            except Exception as e:
                _logger.warning(
                    f"Failed to score session {session_id}: {e}",
                    exc_info=True,
                )
