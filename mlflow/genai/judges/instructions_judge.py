"""
Instructions-based judge implementation for MLflow.

This module provides the InstructionsJudge class that evaluates traces
based on user-provided instructions.
"""

from typing import Any

from pydantic import PrivateAttr

from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import Judge
from mlflow.genai.judges.utils import (
    _DEFAULT_MODEL_DATABRICKS,
    format_prompt,
    get_default_model,
    invoke_judge_model,
)
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

    # Private attributes to store instructions and model
    _instructions: str = PrivateAttr()
    _model: str = PrivateAttr()
    _instructions_prompt: PromptVersion = PrivateAttr()

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
        self._instructions = instructions
        self._model = model or get_default_model()
        # Create a dummy PromptVersion to represent the instructions as a formattable template
        # with an API for variable extraction
        self._instructions_prompt = PromptVersion(
            name=name,
            version=1,
            template=instructions,
        )
        self._validate_instructions_template()

    @property
    def instructions(self) -> str:
        """Get the instructions for this judge."""
        return self._instructions

    @property
    def model(self) -> str:
        """Get the model for this judge."""
        return self._model

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Any:
        """
        Evaluate the provided data using the judge's instructions.

        Args:
            inputs: Input dictionary to evaluate. Cannot be used with 'trace'.
            outputs: Output dictionary to evaluate. Cannot be used with 'trace'.
            expectations: Expected outcomes or ground truth that can be used with any mode.
            trace: Trace object for evaluation. Cannot be used with 'inputs' or 'outputs'.

        Returns:
            Evaluation results

        Raises:
            MlflowException: If both trace and inputs/outputs are specified
        """
        # Validate that trace is not specified together with inputs/outputs/expectations
        if trace is not None and (
            inputs is not None or outputs is not None or expectations is not None
        ):
            raise MlflowException(
                "Cannot specify both 'trace' and 'inputs'/'outputs'/'expectations'. Use either "
                "'trace' for trace-based evaluation or 'inputs'/'outputs'/'expectations' for "
                "field-based evaluation.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Handle field-based evaluation (inputs/outputs)
        if inputs is not None or outputs is not None:
            # Validate that call arguments contain required template fields
            self._validate_call_args_contain_template_fields(inputs, outputs, expectations)

            # Prepare template values by merging inputs, outputs, and expectations
            template_values = {}
            if inputs is not None:
                template_values.update(inputs)
            if outputs is not None:
                template_values.update(outputs)
            if expectations is not None:
                template_values.update(expectations)

            # Format the instructions with the provided values
            formatted_prompt = format_prompt(self._instructions, **template_values)

            # Invoke the judge model
            return invoke_judge_model(self._model, formatted_prompt, self.name)

        # Handle trace-based evaluation
        if trace is not None:
            raise NotImplementedError("Trace-based evaluation is not yet implemented")

        raise MlflowException(
            "Must specify either 'trace' or 'inputs'/'outputs' for evaluation.",
            error_code=INVALID_PARAMETER_VALUE,
        )

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

        # Check if expectations is used as a template variable (not yet supported)
        if has_expectations:
            raise NotImplementedError(
                "The 'expectations' template variable ({{expectations}}) is not yet supported. "
                "This feature will be added in a future release."
            )

        # If trace is used, no other variables (besides reserved ones) should be defined
        if has_trace:
            non_reserved_vars = template_vars - set(self._RESERVED_INSTRUCTION_TEMPLATE_VARIABLES)
            if non_reserved_vars:
                raise MlflowException(
                    f"When using 'trace' variable, no other variables are allowed. "
                    f"Found: {non_reserved_vars}. The 'trace' variable provides complete context "
                    "and should not be mixed with other template variables.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            # Check for inputs/outputs specifically for clearer error message
            if has_inputs or has_outputs:
                raise MlflowException(
                    "Instructions template cannot contain both 'trace' and 'inputs'/'outputs' "
                    "variables. Use either 'trace' for trace-based evaluation or "
                    "'inputs'/'outputs' for field-based evaluation.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        # Check that model is not "databricks" when using trace
        if has_trace and self._model == _DEFAULT_MODEL_DATABRICKS:
            raise MlflowException(
                f"Model cannot be '{_DEFAULT_MODEL_DATABRICKS}' when using 'trace' variable in "
                "the instructions template. Specify a different model "
                "(e.g., model='openai/gpt-4o').",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _validate_call_args_contain_template_fields(
        self,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        expectations: dict[str, Any] | None = None,
    ) -> None:
        """
        Validate that required template variables are present in inputs, outputs, or expectations.

        Args:
            inputs: Input dictionary to validate
            outputs: Output dictionary to validate
            expectations: Expectations dictionary to validate

        Raises:
            MlflowException: If any required template variable is missing from all
                provided dictionaries
        """
        # Get non-reserved template variables
        vars_to_check = self.template_variables - set(self._RESERVED_INSTRUCTION_TEMPLATE_VARIABLES)

        if not vars_to_check:
            return  # No validation needed if no non-reserved variables

        # Get all available keys from inputs, outputs, and expectations
        input_keys = set(inputs.keys()) if inputs is not None else set()
        output_keys = set(outputs.keys()) if outputs is not None else set()
        expectation_keys = set(expectations.keys()) if expectations is not None else set()
        available_vars = input_keys | output_keys | expectation_keys

        # Check which required variables are missing from all sources
        missing_vars = vars_to_check - available_vars

        if missing_vars:
            raise MlflowException(
                f"Required template variables {missing_vars} are missing from inputs, outputs, "
                f"and expectations. Each variable must be present in at least one of them.",
                error_code=INVALID_PARAMETER_VALUE,
            )
