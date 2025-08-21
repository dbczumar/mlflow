"""
Base classes for MLflow GenAI tools that can be used by judges.

This module provides the foundational interfaces for tools that judges can use
to enhance their evaluation capabilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
@dataclass
class ToolDefinition:
    """
    Definition of a tool that follows LiteLLM/OpenAI function calling format.

    Attributes:
        type: The type of tool, typically "function"
        function: Dictionary containing function definition with name, description, and parameters

    Example:
        >>> definition = ToolDefinition(
        ...     type="function",
        ...     function={
        ...         "name": "get_weather",
        ...         "description": "Get current weather for a location",
        ...         "parameters": {
        ...             "type": "object",
        ...             "properties": {"location": {"type": "string", "description": "City name"}},
        ...             "required": ["location"],
        ...         },
        ...     },
        ... )
    """

    type: str
    function: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert tool definition to dictionary format for LLM APIs."""
        return {"type": self.type, "function": self.function}


@experimental(version="3.4.0")
class JudgeTool(ABC):
    """
    Abstract base class for tools that can be used by MLflow judges.

    Tools provide additional capabilities to judges for analyzing traces,
    performing calculations, or accessing external data sources during evaluation.

    Example:
        >>> class WeatherTool(JudgeTool):
        ...     @property
        ...     def name(self) -> str:
        ...         return "get_weather"
        ...
        ...     def get_definition(self) -> ToolDefinition:
        ...         return ToolDefinition(
        ...             type="function",
        ...             function={
        ...                 "name": self.name,
        ...                 "description": "Get weather information",
        ...                 "parameters": {
        ...                     "type": "object",
        ...                     "properties": {"location": {"type": "string"}},
        ...                     "required": ["location"],
        ...                 },
        ...             },
        ...         )
        ...
        ...     def invoke(self, location: str) -> str:
        ...         return f"Weather in {location}: Sunny, 75°F"
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
