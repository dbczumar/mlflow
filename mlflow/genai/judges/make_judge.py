"""
Factory function for creating MLflow judge instances.

This module provides a flexible interface for creating custom judges
that can evaluate traces based on user-defined criteria.
"""

from mlflow.genai.judges.base import Judge
from mlflow.genai.judges.instructions_judge import InstructionsJudge
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
def make_judge(name: str, instructions: str, model: str | None = None, **kwargs) -> Judge:
    """
    Create a custom MLflow judge instance.

    Args:
        name: The name of the judge
        instructions: Natural language instructions for evaluation
        model: The model identifier to use for evaluation (e.g., "openai/gpt-4o")
        kwargs: Additional configuration parameters

    Returns:
        An InstructionsJudge instance configured with the provided parameters
    """
    return InstructionsJudge(name=name, instructions=instructions, model=model, **kwargs)
