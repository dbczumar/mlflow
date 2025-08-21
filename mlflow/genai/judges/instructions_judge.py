"""
Instructions-based judge implementation for MLflow.

This module provides the InstructionsJudge class that evaluates traces
based on user-provided instructions.
"""

from typing import Any

from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import Judge
from mlflow.genai.scorers.base import ScorerKind
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class InstructionsJudge(Judge):
    """
    A judge that evaluates traces based on user-provided instructions.

    This judge uses natural language instructions to guide evaluation,
    making it flexible for various assessment criteria.
    """

    _TEMPLATE_VARIABLE_INPUTS = "inputs"
    _TEMPLATE_VARIABLE_OUTPUTS = "outputs"
    _TEMPLATE_VARIABLE_TRACE = "trace"
    _TEMPLATE_VARIABLE_EXPECTATIONS = "expectations"
    _RESERVED_INSTRUCTION_TEMPLATE_VARIABLES = [
        _TEMPLATE_VARIABLE_INPUTS,
        _TEMPLATE_VARIABLE_OUTPUTS,
        _TEMPLATE_VARIABLE_TRACE,
        _TEMPLATE_VARIABLE_EXPECTATIONS,
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
        self._instructions_prompt = PromptVersion(
            name=name,
            version=1,
            template=instructions,
        )
        self._validate_instructions_template()

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
        return self._instructions_prompt.variables

    def _validate_instructions_template(self) -> None:
        """
        Validate that instructions don't contain both trace and inputs/outputs variables.

        Raises:
            MlflowException: If instructions contain both trace and inputs/outputs variables
        """
        template_vars = self.template_variables

        has_trace = self._TEMPLATE_VARIABLE_TRACE in template_vars
        has_inputs = self._TEMPLATE_VARIABLE_INPUTS in template_vars
        has_outputs = self._TEMPLATE_VARIABLE_OUTPUTS in template_vars

        if has_trace and (has_inputs or has_outputs):
            raise MlflowException(
                "Instructions template cannot contain both 'trace' and 'inputs'/'outputs' "
                "variables. Use either 'trace' for trace-based evaluation or 'inputs'/'outputs' "
                "for field-based evaluation.",
                error_code=INVALID_PARAMETER_VALUE,
            )

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
            MlflowException: If any dictionary is missing required template variables
        """
        # Get non-reserved template variables
        vars_to_check = self.template_variables - set(self._RESERVED_INSTRUCTION_TEMPLATE_VARIABLES)

        if not vars_to_check:
            return  # No validation needed if no non-reserved variables

        # Check inputs
        if inputs is not None:
            for i, input_dict in enumerate(inputs):
                missing_vars = vars_to_check - set(input_dict.keys())
                if missing_vars:
                    raise MlflowException(
                        f"Input at index {i} is missing required template variables: "
                        f"{missing_vars}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )

        # Check outputs
        if outputs is not None:
            for i, output_dict in enumerate(outputs):
                missing_vars = vars_to_check - set(output_dict.keys())
                if missing_vars:
                    raise MlflowException(
                        f"Output at index {i} is missing required template variables: "
                        f"{missing_vars}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
