"""
Instructions-based judge implementation for MLflow.

This module provides the InstructionsJudge class that evaluates traces
based on user-provided instructions.
"""

from typing import Any

from mlflow.entities.model_registry.prompt_version import PromptVersion
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

    _RESERVED_INSTRUCTION_TEMPLATE_VARIABLES = [
        "inputs",
        "outputs",
        "trace",
        "expectations",
    ]

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
        # Create a dummy PromptVersion to represent the instructions as a formattable template
        # with an API for variable extraction
        self._prompt_version = PromptVersion(
            name=name,
            version=1,
            template=instructions,
        )

    def __call__(
        self,
        *,
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        expectations: dict[str, Any] | None = None,
        trace: Any = None,
        **kwargs,
    ) -> Any:
        """
        Evaluate the provided data using the judge's instructions.

        Args:
            inputs: List of input dictionaries to evaluate
            outputs: List of output dictionaries to evaluate
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
        return ScorerKind.CLASS

    @property
    def template_variables(self) -> set[str]:
        """Get the template variables from the instructions."""
        return self._prompt_version.variables

    def _validate_template_variables(
        self,
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Validate that all dictionaries in inputs/outputs contain required template variables.

        Args:
            inputs: List of input dictionaries to validate
            outputs: List of output dictionaries to validate

        Raises:
            ValueError: If any dictionary is missing required template variables
        """
        # Get non-reserved template variables
        required_vars = self.template_variables - set(self._RESERVED_INSTRUCTION_TEMPLATE_VARIABLES)

        if not required_vars:
            return  # No validation needed if no non-reserved variables

        # Check inputs
        if inputs is not None:
            for i, input_dict in enumerate(inputs):
                missing_vars = required_vars - set(input_dict.keys())
                if missing_vars:
                    raise ValueError(
                        f"Input at index {i} is missing required template variables: {missing_vars}"
                    )

        # Check outputs
        if outputs is not None:
            for i, output_dict in enumerate(outputs):
                missing_vars = required_vars - set(output_dict.keys())
                if missing_vars:
                    raise ValueError(
                        f"Output at index {i} is missing required template variables: "
                        f"{missing_vars}"
                    )
