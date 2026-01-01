"""
Base classes for MLflow GenAI tools that can be used by judges.

This module provides the foundational interfaces for tools that judges can use
to enhance their evaluation capabilities.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import mlflow
from mlflow.entities.trace import Trace
from mlflow.types.llm import ToolDefinition
from mlflow.utils.annotations import experimental


@lru_cache(maxsize=100)
def _get_trace(trace_id: str) -> Trace:
    """
    Fetch a trace by ID from MLflow.

    Args:
        trace_id: The ID of the trace to fetch

    Returns:
        The Trace object

    Raises:
        MlflowException: If the trace is not found
    """
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

    trace = mlflow.get_trace(trace_id)
    if trace is None:
        raise MlflowException(
            f"Trace with ID '{trace_id}' not found",
            error_code=INVALID_PARAMETER_VALUE,
        )
    return trace


@experimental(version="3.4.0")
class JudgeTool(ABC):
    """
    Abstract base class for tools that can be used by MLflow judges.

    Tools provide additional capabilities to judges for analyzing traces,
    performing calculations, or accessing external data sources during evaluation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of the tool.

        Returns:
            Tool name used for registration and invocation
        """

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """
        Get the tool definition in LiteLLM/OpenAI function calling format.

        Returns:
            ToolDefinition object containing the tool specification
        """

    @abstractmethod
    def invoke(self, trace_id: str, **kwargs) -> Any:
        """
        Invoke the tool with the provided trace ID and arguments.

        Args:
            trace_id: The ID of the trace to analyze
            kwargs: Additional keyword arguments for the tool

        Returns:
            Result of the tool execution
        """
