"""Configuration classes for online scoring."""

from dataclasses import dataclass

from mlflow.entities import Trace
from mlflow.genai.scorers.base import Scorer


@dataclass
class OnlineScorerConfig:
    """Configuration for an online scorer to run against traces."""

    serialized_scorer: str
    sample_rate: float
    filter_string: str | None = None


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
