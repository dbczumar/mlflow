"""
Instructions-based judge implementation for MLflow.

This module provides the InstructionsJudge class that evaluates traces
based on user-provided instructions.
"""

from typing import Any

from mlflow.genai.judges.base import Judge
from mlflow.genai.scorers.base import ScorerKind
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class InstructionsJudge(Judge):
    """
    A judge that evaluates traces based on user-provided instructions.

    This judge uses natural language instructions to guide evaluation,
    making it flexible for various assessment criteria.
    """

    def __init__(self, name: str, instructions: str, model: str | None = None, **kwargs):
        """
        Initialize the InstructionsJudge.

        Args:
            name: The name of the judge
            instructions: Natural language instructions for evaluation
            model: The model identifier to use for evaluation (e.g., "openai/gpt-4o")
            kwargs: Additional configuration parameters
        """
        super().__init__(name=name, **kwargs)
        self.instructions = instructions
        self.model = model

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Any = None,
        **kwargs,
    ) -> Any:
        """
        Evaluate the provided data using the judge's instructions.

        Args:
            inputs: Input data to evaluate
            outputs: Output data to evaluate
            expectations: Expected outcomes or ground truth
            trace: Trace object for evaluation
            kwargs: Additional context for evaluation

        Returns:
            Evaluation results
        """
        raise NotImplementedError("InstructionsJudge evaluation is not yet implemented")

    @property
    def kind(self) -> ScorerKind:
        """Return the kind of scorer this judge represents."""
        raise NotImplementedError("InstructionsJudge kind property is not yet implemented")
