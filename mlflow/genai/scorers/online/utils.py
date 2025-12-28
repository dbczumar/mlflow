"""Utilities for online scoring execution."""

import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from mlflow.entities import Trace
from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.harness import _compute_eval_scores, _log_assessments
from mlflow.genai.evaluation.session_utils import evaluate_session_level_scorers
from mlflow.genai.scorers.online.config import (
    OnlineScorerConfig,
    SessionScoringTask,
    TraceScoringTask,
)
from mlflow.genai.scorers.online.sampler import ScorerSampler
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.constant import TraceMetadataKey

# Checkpoint tag for tracking last processed trace timestamp
ONLINE_SCORING_CHECKPOINT_TAG = "mlflow.latestOnlineScoringTimestampMs"

# Default lookback period when no checkpoint exists (1 hour)
_DEFAULT_LOOKBACK_MS = 60 * 60 * 1000

# Maximum traces to process per job run
_MAX_TRACES_PER_JOB = 500


_logger = logging.getLogger(__name__)


def _get_checkpoint_timestamp(
    tracking_store: AbstractStore,
    experiment_id: str,
) -> int | None:
    """
    Get the last processed trace timestamp from the experiment checkpoint tag.

    Returns None if no checkpoint exists.
    """
    try:
        experiment = tracking_store.get_experiment(experiment_id)
        if checkpoint := experiment.tags.get(ONLINE_SCORING_CHECKPOINT_TAG):
            return int(checkpoint)
    except (TypeError, ValueError):
        pass
    return None


def _update_checkpoint_timestamp(
    tracking_store: AbstractStore,
    experiment_id: str,
    timestamp_ms: int,
    current_checkpoint: int | None,
    lock: threading.Lock,
) -> None:
    """
    Update the checkpoint tag if the new timestamp is greater than the current checkpoint.

    Uses a lock to ensure thread-safe updates.
    """
    with lock:
        # Only update if new timestamp is greater
        if current_checkpoint is None or timestamp_ms > current_checkpoint:
            tracking_store.set_experiment_tag(
                experiment_id,
                ExperimentTag(ONLINE_SCORING_CHECKPOINT_TAG, str(timestamp_ms)),
            )


def _calculate_time_window(
    tracking_store: AbstractStore,
    experiment_id: str,
) -> tuple[int, int, int | None]:
    """
    Calculate the time window for trace fetching.

    Returns:
        Tuple of (start_timestamp_ms, end_timestamp_ms, current_checkpoint)
    """
    current_time_ms = int(time.time() * 1000)
    current_checkpoint = _get_checkpoint_timestamp(tracking_store, experiment_id)

    if current_checkpoint is not None:
        start_time_ms = current_checkpoint
    else:
        start_time_ms = current_time_ms - _DEFAULT_LOOKBACK_MS

    return start_time_ms, current_time_ms, current_checkpoint


def _fetch_traces_for_filter(
    tracking_store: AbstractStore,
    experiment_id: str,
    start_time_ms: int,
    end_time_ms: int,
    filter_string: str | None,
    max_traces: int,
) -> list[Trace]:
    """Fetch traces within the time window matching the filter string."""
    # Build filter: timestamp range
    time_filter = f"trace.timestamp_ms > {start_time_ms} AND trace.timestamp_ms <= {end_time_ms}"
    combined_filter = f"({time_filter}) AND ({filter_string})" if filter_string else time_filter

    trace_infos, _ = tracking_store.search_traces(
        experiment_ids=[experiment_id],
        filter_string=combined_filter,
        max_results=max_traces,
        order_by=["timestamp_ms ASC"],
    )

    if not trace_infos:
        return []

    # Fetch full traces
    trace_ids = [info.trace_id for info in trace_infos]
    return tracking_store.batch_get_traces(trace_ids)


def _group_traces_by_session(
    traces: list[Trace],
) -> dict[str, list[Trace]]:
    """
    Group traces by their session ID.

    Traces without session metadata are not included.
    """
    session_groups: dict[str, list[Trace]] = defaultdict(list)

    for trace in traces:
        trace_metadata = trace.info.trace_metadata or {}
        if session_id := trace_metadata.get(TraceMetadataKey.TRACE_SESSION):
            session_groups[session_id].append(trace)

    # Sort traces within each session by timestamp
    for session_id in session_groups:
        session_groups[session_id] = sorted(
            session_groups[session_id],
            key=lambda t: t.info.timestamp_ms or float("inf"),
        )

    return dict(session_groups)


def execute_online_scoring(
    experiment_id: str,
    scorer_configs: list[dict[str, Any]],
    tracking_store: AbstractStore,
) -> None:
    """
    Execute online scoring for an experiment.

    This is the main implementation of online scoring logic, called by run_online_scorer_job.

    The function:
    1. Calculates time window from checkpoint tag or fallback
    2. Groups configs by filter_string and fetches matching traces
    3. Applies dense sampling to select scorers for each trace/session
    4. Runs single-turn scorers in parallel (Phase 1)
    5. Runs session-level scorers in parallel (Phase 2)
    6. Updates checkpoint tag incrementally

    Args:
        experiment_id: The experiment ID to fetch traces from.
        scorer_configs: List of OnlineScorerConfig dicts specifying which scorers to run.
        tracking_store: The tracking store instance.
    """
    configs = [OnlineScorerConfig(**c) for c in scorer_configs]

    if not configs:
        _logger.info("No scorer configs provided, skipping")
        return

    sampler = ScorerSampler(configs)

    # Step 1: Calculate time window
    start_time_ms, end_time_ms, current_checkpoint = _calculate_time_window(
        tracking_store, experiment_id
    )
    _logger.info(
        f"Online scoring for experiment {experiment_id}: "
        f"time window [{start_time_ms}, {end_time_ms}]"
    )

    # Step 2: Fetch traces and apply dense sampling per filter group
    single_turn_tasks: dict[str, TraceScoringTask] = {}
    session_tasks: dict[str, SessionScoringTask] = {}

    for filter_string in sampler.get_filter_strings():
        single_turn_scorers = sampler.get_scorers_for_filter(filter_string, session_level=False)
        session_scorers = sampler.get_scorers_for_filter(filter_string, session_level=True)

        # Fetch traces for this filter
        traces = _fetch_traces_for_filter(
            tracking_store,
            experiment_id,
            start_time_ms,
            end_time_ms,
            filter_string,
            _MAX_TRACES_PER_JOB,
        )

        if not traces:
            _logger.debug(f"No traces found for filter: {filter_string}")
            continue

        _logger.info(f"Found {len(traces)} traces for filter: {filter_string}")

        # Dense sample single-turn scorers per trace
        for trace in traces:
            if selected := sampler.sample(single_turn_scorers):
                single_turn_tasks.setdefault(
                    trace.info.trace_id, TraceScoringTask(trace=trace, scorers=[])
                ).scorers.extend(selected)

        # Dense sample session-level scorers per session
        if session_scorers:
            session_groups = _group_traces_by_session(traces)
            for session_id, session_traces in session_groups.items():
                if selected := sampler.sample(session_scorers):
                    session_tasks.setdefault(
                        session_id, SessionScoringTask(traces=session_traces, scorers=[])
                    ).scorers.extend(selected)

    if not single_turn_tasks and not session_tasks:
        _logger.info("No traces selected after sampling, skipping")
        return

    _logger.info(
        f"Running scoring: {len(single_turn_tasks)} single-turn tasks, "
        f"{len(session_tasks)} session tasks"
    )

    # Step 3: Execute scoring with ThreadPoolExecutor (two phases)
    checkpoint_lock = threading.Lock()
    latest_checkpoint = current_checkpoint

    def update_checkpoint_if_needed(timestamp_ms: int | None) -> None:
        nonlocal latest_checkpoint
        if timestamp_ms is None:
            return
        with checkpoint_lock:
            if latest_checkpoint is None or timestamp_ms > latest_checkpoint:
                _update_checkpoint_timestamp(
                    tracking_store,
                    experiment_id,
                    timestamp_ms,
                    latest_checkpoint,
                    checkpoint_lock,
                )
                latest_checkpoint = timestamp_ms

    with ThreadPoolExecutor(
        max_workers=MLFLOW_GENAI_EVAL_MAX_WORKERS.get(),
        thread_name_prefix="OnlineScoring",
    ) as executor:
        # Phase 1: Single-turn scorers
        single_turn_futures = {}
        for task in single_turn_tasks.values():
            eval_item = EvalItem.from_trace(task.trace)
            future = executor.submit(
                _compute_eval_scores, eval_item=eval_item, scorers=task.scorers
            )
            single_turn_futures[future] = task

        for future in as_completed(single_turn_futures):
            task = single_turn_futures[future]
            try:
                if feedbacks := future.result():
                    _log_assessments(run_id=None, trace=task.trace, assessments=feedbacks)
                # Update checkpoint incrementally
                update_checkpoint_if_needed(task.trace.info.timestamp_ms)
            except Exception as e:
                _logger.warning(
                    f"Failed to score trace {task.trace.info.trace_id}: {e}",
                    exc_info=True,
                )

        # Phase 2: Session-level scorers
        session_futures = {}
        for session_id, task in session_tasks.items():
            session_items = [EvalItem.from_trace(t) for t in task.traces]
            future = executor.submit(
                evaluate_session_level_scorers,
                session_id=session_id,
                session_items=session_items,
                multi_turn_scorers=task.scorers,
            )
            session_futures[future] = (session_id, task)

        for future in as_completed(session_futures):
            session_id, task = session_futures[future]
            try:
                result = future.result()
                # result is {first_trace_id: [feedbacks]}
                for trace_id, feedbacks in result.items():
                    if feedbacks:
                        # Find the trace to log to
                        trace = next(
                            (t for t in task.traces if t.info.trace_id == trace_id),
                            None,
                        )
                        if trace:
                            _log_assessments(run_id=None, trace=trace, assessments=feedbacks)

                # Update checkpoint with max timestamp from session
                max_ts = max(
                    (t.info.timestamp_ms for t in task.traces if t.info.timestamp_ms),
                    default=None,
                )
                update_checkpoint_if_needed(max_ts)
            except Exception as e:
                _logger.warning(
                    f"Failed to score session {session_id}: {e}",
                    exc_info=True,
                )

    _logger.info(f"Online scoring completed for experiment {experiment_id}")
