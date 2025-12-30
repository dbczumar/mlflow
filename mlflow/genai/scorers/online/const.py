"""Constants for online scoring."""

# Checkpoint tags for tracking last processed timestamps
TRACE_CHECKPOINT_TAG = "mlflow.latestOnlineScoring.trace.timestampMs"
SESSION_CHECKPOINT_TAG = "mlflow.latestOnlineScoring.session.timestampMs"

# Maximum lookback period to prevent getting stuck on old failing traces/sessions (1 hour)
MAX_LOOKBACK_MS = 60 * 60 * 1000

# Session inactivity buffer: 10 minutes without new traces = session complete
# TODO: CHANGE BACK TO 10 * 60 * 1000 - Currently set to 15 seconds for testing!
SESSION_COMPLETION_BUFFER_MS = 15 * 1000

# Minimum sessions to process before submitting batch job
MIN_SESSIONS_PER_JOB = 10

# Maximum traces to include in a single scoring job
MAX_TRACES_PER_JOB = 500

# Filter to exclude eval run traces (traces generated from MLflow runs)
EXCLUDE_EVAL_RUN_TRACES_FILTER = "metadata.mlflow.sourceRun IS NULL"
