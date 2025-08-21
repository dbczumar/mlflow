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
        Validate that instructions contain at least one variable and don't contain both
        trace and inputs/outputs variables. Also validate that model is defined when
        using trace or expectations.

        Raises:
            MlflowException: If instructions don't contain any variables, contain both
                trace and inputs/outputs variables, or use trace/expectations without a model
        """
        template_vars = self.template_variables

        # Check that template contains at least one variable
        if not template_vars:
            raise MlflowException(
                "Instructions template must contain at least one variable (e.g., {{inputs}}, "
                "{{outputs}}, {{trace}}, or custom variables).",
                error_code=INVALID_PARAMETER_VALUE,
            )

        has_trace = self._TEMPLATE_VARIABLE_TRACE in template_vars
        has_inputs = self._TEMPLATE_VARIABLE_INPUTS in template_vars
        has_outputs = self._TEMPLATE_VARIABLE_OUTPUTS in template_vars
        has_expectations = self._TEMPLATE_VARIABLE_EXPECTATIONS in template_vars

        if has_trace and (has_inputs or has_outputs):
            raise MlflowException(
                "Instructions template cannot contain both 'trace' and 'inputs'/'outputs' "
                "variables. Use either 'trace' for trace-based evaluation or 'inputs'/'outputs' "
                "for field-based evaluation.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Check that model is defined when using trace or expectations
        if (has_trace or has_expectations) and not self.model:
            raise MlflowException(
                "Model must be specified when using 'trace' or 'expectations' variables in the "
                "instructions template. Specify the model parameter (e.g., model='openai/gpt-4o').",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _validate_template_variables(
        self,
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Validate that required template variables are present in either inputs or outputs
        at each index.

        Args:
            inputs: List of input dictionaries to validate
            outputs: List of output dictionaries to validate

        Raises:
            MlflowException: If any required template variable is missing from both
                inputs and outputs at the same index
        """
        # Get non-reserved template variables
        vars_to_check = self.template_variables - set(self._RESERVED_INSTRUCTION_TEMPLATE_VARIABLES)

        if not vars_to_check:
            return  # No validation needed if no non-reserved variables

        # Determine the length of data to validate
        max_len = 0
        if inputs is not None:
            max_len = len(inputs)
        if outputs is not None:
            max_len = max(max_len, len(outputs))

        # Check that each required variable exists in either inputs or outputs at each index
        for i in range(max_len):
            input_dict = inputs[i] if inputs and i < len(inputs) else {}
            output_dict = outputs[i] if outputs and i < len(outputs) else {}

            # Get all available keys from both inputs and outputs at this index
            available_vars = set(input_dict.keys()) | set(output_dict.keys())

            # Check which required variables are missing from both
            missing_vars = vars_to_check - available_vars

            if missing_vars:
                raise MlflowException(
                    f"At index {i}, required template variables {missing_vars} are missing "
                    f"from both inputs and outputs. Each variable must be present in at least "
                    f"one of them.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
