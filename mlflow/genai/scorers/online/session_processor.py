"""Session-level online scoring processor for executing scorers on completed sessions."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.session_utils import evaluate_session_level_scorers
from mlflow.genai.scorers.online.online_scorer import OnlineScorer
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler
from mlflow.genai.scorers.online.session_checkpoint import OnlineSessionCheckpointManager
from mlflow.genai.scorers.online.trace_loader import OnlineTraceLoader
from mlflow.store.tracking.abstract_store import AbstractStore

_logger = logging.getLogger(__name__)


@dataclass
class CompletedSession:
    """A completed session with metadata."""

    session_id: str
    trace_count: int
    first_trace_timestamp_ms: int
    last_trace_timestamp_ms: int


# Session inactivity buffer: 10 minutes without new traces = session complete
_SESSION_COMPLETION_BUFFER_MS = 10 * 60 * 1000

# Minimum number of sessions to process per job run
_MIN_SESSIONS_PER_JOB = 10

# Maximum traces across all sessions per job run (same as trace processor)
_MAX_TRACES_PER_JOB = 500


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
            sampler=OnlineScorerSampler(online_scorers),
            experiment_id=experiment_id,
            tracking_store=tracking_store,
        )

    def process_sessions(self) -> None:
        """
        Execute online scoring for completed sessions in the experiment.

        Finds sessions that have been inactive for 10+ minutes, applies sampling to
        select scorers, runs scoring in parallel (one thread per session), and updates
        the checkpoint.
        """
        if not self._sampler.configs:
            _logger.info("No scorer configs provided, skipping")
            return

        # Calculate time window for completed sessions
        start_time_ms, end_time_ms, current_checkpoint = (
            self._checkpoint_manager.calculate_time_window()
        )

        # Sessions are "complete" if they haven't had new traces in 10+ minutes
        max_last_activity_timestamp_ms = end_time_ms - _SESSION_COMPLETION_BUFFER_MS

        _logger.info(
            f"Session scoring for experiment {self._experiment_id}: "
            f"looking for sessions in [{start_time_ms}, {max_last_activity_timestamp_ms}]"
        )

        # Find completed sessions
        completed_sessions = self._tracking_store.find_completed_sessions(
            experiment_id=self._experiment_id,
            min_start_timestamp_ms=start_time_ms,
            max_last_activity_timestamp_ms=max_last_activity_timestamp_ms,
        )

        if not completed_sessions:
            _logger.info("No completed sessions found, skipping")
            self._checkpoint_manager.update_checkpoint_timestamp(end_time_ms)
            return

        _logger.info(f"Found {len(completed_sessions)} completed sessions")

        # Select sessions to score based on batching constraints
        sessions_to_score = self._select_sessions_to_score(completed_sessions)

        if not sessions_to_score:
            _logger.info("No sessions selected after batching, skipping")
            self._checkpoint_manager.update_checkpoint_timestamp(end_time_ms)
            return

        _logger.info(
            f"Selected {len(sessions_to_score)} sessions "
            f"({sum(s.trace_count for s in sessions_to_score)} total traces)"
        )

        # Execute scoring in parallel
        self._execute_session_scoring(sessions_to_score)

        # Update checkpoint after scoring
        self._checkpoint_manager.update_checkpoint_timestamp(end_time_ms)

        _logger.info(f"Session scoring completed for experiment {self._experiment_id}")

    def _select_sessions_to_score(
        self, completed_sessions: list[CompletedSession]
    ) -> list[CompletedSession]:
        """
        Select which sessions to score based on batching constraints.

        Selects at least MIN_SESSIONS_PER_JOB sessions, but stops when total traces
        would exceed MAX_TRACES_PER_JOB.

        Args:
            completed_sessions: List of completed sessions sorted by trace_count DESC.

        Returns:
            List of sessions to score.
        """
        selected = []
        total_traces = 0

        for session in completed_sessions:
            # Always include at least MIN_SESSIONS_PER_JOB sessions
            if len(selected) < _MIN_SESSIONS_PER_JOB:
                selected.append(session)
                total_traces += session.trace_count
                continue

            # After min sessions, check trace limit
            if total_traces + session.trace_count > _MAX_TRACES_PER_JOB:
                break

            selected.append(session)
            total_traces += session.trace_count

        return selected

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
        # Fetch traces for this session by searching with session filter
        traces = self._trace_loader.fetch_trace_infos_between(
            experiment_id=self._experiment_id,
            start_time_ms=session.first_trace_timestamp_ms,
            end_time_ms=session.last_trace_timestamp_ms,
            filter_string=f"request_metadata.`mlflow.trace.session` = '{session.session_id}'",
            max_results=session.trace_count + 10,  # Add buffer for concurrent traces
        )

        if not traces:
            _logger.warning(f"No traces found for session {session.session_id}")
            return

        # Fetch full trace data
        trace_ids = [t.trace_id for t in traces]
        full_traces = self._trace_loader.fetch_traces(trace_ids)

        if not full_traces:
            _logger.warning(f"Failed to fetch full traces for session {session.session_id}")
            return

        # Sort traces by timestamp
        full_traces.sort(key=lambda t: t.info.timestamp_ms)

        # Get session-level scorers that apply to this session
        # For now, we'll check all filters and combine scorers
        applicable_scorers = []
        for filter_string in self._sampler.get_filter_strings():
            session_scorers = self._sampler.get_scorers_for_filter(
                filter_string, session_level=True
            )
            if selected := self._sampler.sample(session.session_id, session_scorers):
                applicable_scorers.extend(selected)

        if not applicable_scorers:
            return

        # Convert traces to eval items
        session_items = [EvalItem.from_trace(t) for t in full_traces]

        # Run session-level scorers
        try:
            result = evaluate_session_level_scorers(
                session_id=session.session_id,
                session_items=session_items,
                multi_turn_scorers=applicable_scorers,
            )

            # Log assessments for each trace in the result
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
