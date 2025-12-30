"""Session-level online scoring processor for executing scorers on completed sessions."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.session_utils import evaluate_session_level_scorers
from mlflow.genai.scorers.online.constants import (
    EXCLUDE_EVAL_RUN_TRACES_FILTER,
    MAX_SESSIONS_PER_JOB,
)
from mlflow.genai.scorers.online.online_scorer import OnlineScorer
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler
from mlflow.genai.scorers.online.session_checkpointer import OnlineSessionCheckpointManager
from mlflow.genai.scorers.online.trace_loader import OnlineTraceLoader
from mlflow.store.tracking.abstract_store import AbstractStore

_logger = logging.getLogger(__name__)


@dataclass
class CompletedSession:
    """A completed session with metadata."""

    session_id: str
    first_trace_timestamp_ms: int
    last_trace_timestamp_ms: int


class OnlineSessionScoringProcessor:
    """Orchestrates online scoring of completed sessions."""

    def __init__(
        self,
        trace_loader: "OnlineTraceLoader",
        checkpoint_manager: OnlineSessionCheckpointManager,
        sampler: OnlineScorerSampler,
        experiment_id: str,
        tracking_store: AbstractStore,
    ):
        self._trace_loader = trace_loader
        self._checkpoint_manager = checkpoint_manager
        self._sampler = sampler
        self._experiment_id = experiment_id
        self._tracking_store = tracking_store

    @classmethod
    def create(
        cls,
        experiment_id: str,
        online_scorers: list[OnlineScorer],
        tracking_store: AbstractStore,
    ) -> "OnlineSessionScoringProcessor":
        """
        Factory method to create an OnlineSessionScoringProcessor with dependencies.

        Args:
            experiment_id: The experiment ID to process sessions from.
            online_scorers: List of OnlineScorer instances.
            tracking_store: The tracking store instance.

        Returns:
            Configured OnlineSessionScoringProcessor instance.
        """
        return cls(
            trace_loader=OnlineTraceLoader(tracking_store),
            checkpoint_manager=OnlineSessionCheckpointManager(tracking_store, experiment_id),
            sampler=OnlineScorerSampler(online_scorers, tracking_store),
            experiment_id=experiment_id,
            tracking_store=tracking_store,
        )

    def process_sessions(self) -> None:
        """
        Execute online scoring for completed sessions in the experiment.

        Finds sessions that have been inactive for the completion buffer duration,
        applies sampling to select scorers, runs scoring in parallel (one thread per
        session), and updates the checkpoint.
        """
        if not self._sampler.configs:
            _logger.info("No scorer configs provided, skipping")
            return

        time_window = self._checkpoint_manager.calculate_time_window()

        _logger.info(
            f"Session scoring for experiment {self._experiment_id}: "
            f"looking for sessions in "
            f"[{time_window.min_last_trace_timestamp_ms}, "
            f"{time_window.max_last_trace_timestamp_ms}]"
        )

        completed_sessions = self._tracking_store.find_completed_sessions(
            experiment_id=self._experiment_id,
            min_last_trace_timestamp_ms=time_window.min_last_trace_timestamp_ms,
            max_last_trace_timestamp_ms=time_window.max_last_trace_timestamp_ms,
            max_results=MAX_SESSIONS_PER_JOB,
        )

        if not completed_sessions:
            _logger.info("No completed sessions found, skipping")
            return

        _logger.info(f"Found {len(completed_sessions)} completed sessions for scoring")

        self._execute_session_scoring(completed_sessions)

        # Update checkpoint to the last processed session's last trace timestamp
        # Sessions are sorted by last_trace_timestamp_ms ASC, so take the last one
        latest_session = completed_sessions[-1]
        self._checkpoint_manager.update_checkpoint_timestamp(latest_session.last_trace_timestamp_ms)

        _logger.info(f"Session scoring completed for experiment {self._experiment_id}")

    def _execute_session_scoring(self, sessions: list[CompletedSession]) -> None:
        """
        Execute session-level scoring tasks in parallel.

        Each thread loads traces for its session independently and runs all applicable
        scorers on that session.

        Args:
            sessions: List of CompletedSession objects to score.
        """
        with ThreadPoolExecutor(
            max_workers=MLFLOW_GENAI_EVAL_MAX_WORKERS.get(),
            thread_name_prefix="SessionScoring",
        ) as executor:
            futures = {}
            for session in sessions:
                future = executor.submit(self._score_session, session)
                futures[future] = session

            for future in as_completed(futures):
                session = futures[future]
                try:
                    future.result()
                except Exception as e:
                    _logger.warning(
                        f"Failed to score session {session.session_id}: {e}",
                        exc_info=True,
                    )

    def _score_session(self, session: CompletedSession) -> None:
        """
        Score a single session by loading its traces and applying all scorers.

        This method runs in a worker thread. It fetches all traces for the session,
        applies sampling to select scorers, and runs the selected scorers.

        Args:
            session: The CompletedSession to score.
        """
        session_filter = f"metadata.`mlflow.trace.session` = '{session.session_id}'"
        combined_filter = f"{EXCLUDE_EVAL_RUN_TRACES_FILTER} AND {session_filter}"
        traces = self._trace_loader.fetch_trace_infos_between(
            experiment_id=self._experiment_id,
            start_time_ms=session.first_trace_timestamp_ms,
            end_time_ms=session.last_trace_timestamp_ms,
            filter_string=combined_filter,
        )

        if not traces:
            _logger.warning(f"No traces found for session {session.session_id}")
            return

        trace_ids = [t.trace_id for t in traces]
        full_traces = self._trace_loader.fetch_traces(trace_ids)

        if not full_traces:
            _logger.warning(f"Failed to fetch full traces for session {session.session_id}")
            return

        full_traces.sort(key=lambda t: t.info.timestamp_ms)

        applicable_scorers = []
        for filter_string in self._sampler.get_filter_strings():
            session_scorers = self._sampler.get_scorers_for_filter(
                filter_string, session_level=True
            )
            if selected := self._sampler.sample(session.session_id, session_scorers):
                applicable_scorers.extend(selected)

        if not applicable_scorers:
            return

        session_items = [EvalItem.from_trace(t) for t in full_traces]

        try:
            result = evaluate_session_level_scorers(
                session_id=session.session_id,
                session_items=session_items,
                multi_turn_scorers=applicable_scorers,
            )

            from mlflow.genai.evaluation.harness import _log_assessments

            for trace_id, feedbacks in result.items():
                if feedbacks and (
                    trace := next((t for t in full_traces if t.info.trace_id == trace_id), None)
                ):
                    _log_assessments(run_id=None, trace=trace, assessments=feedbacks)
        except Exception as e:
            _logger.warning(
                f"Failed to evaluate session {session.session_id}: {e}",
                exc_info=True,
            )
