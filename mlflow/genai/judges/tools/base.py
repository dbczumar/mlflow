"""
Base classes for MLflow GenAI tools that can be used by judges.

This module provides the foundational interfaces for tools that judges can use
to enhance their evaluation capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any

from mlflow.types.llm import ToolDefinition
from mlflow.utils.annotations import experimental


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
        Return the name of the tool.

        Returns:
            String identifier for the tool, used in function calls
        """

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """
        Get the tool definition in LiteLLM/OpenAI function calling format.

        Returns:
            ToolDefinition object containing the tool specification
        """

    @abstractmethod
    def invoke(self, *args, **kwargs) -> Any:
        """
        Invoke the tool with the provided arguments.

        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            Result of the tool execution
        """
