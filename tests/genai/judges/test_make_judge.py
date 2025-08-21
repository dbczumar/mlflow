"""
Tests for make_judge function and InstructionsJudge class.
"""

from unittest import mock

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.instructions_judge import InstructionsJudge


def test_make_judge_returns_instructions_judge():
    """Test that make_judge returns an InstructionsJudge instance."""
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate the {{response}} based on {{criteria}}.",
    )

    assert isinstance(judge, InstructionsJudge)
    assert judge.name == "test_judge"
    assert judge.instructions == "Evaluate the {{response}} based on {{criteria}}."


def test_make_judge_with_explicit_model():
    """Test make_judge with an explicit model specified."""
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate {{response}}.",
        model="openai/gpt-4o",
    )

    assert judge.model == "openai/gpt-4o"


def test_make_judge_with_default_model():
    """Test make_judge uses default model when not specified."""
    with mock.patch("mlflow.genai.judges.instructions_judge.get_default_model") as mock_default:
        mock_default.return_value = "openai:/gpt-4.1-mini"

        judge = make_judge(
            name="test_judge",
            instructions="Evaluate {{response}}.",
        )

        mock_default.assert_called_once()
        assert judge.model == "openai:/gpt-4.1-mini"


def test_field_based_evaluation_with_inputs_outputs():
    """Test field-based evaluation with inputs and outputs."""
    judge = make_judge(
        name="quality_judge",
        instructions="Evaluate if {{query}} is answered well by {{response}}.",
        model="openai/gpt-4o",
    )

    mock_feedback = Feedback(
        name="quality_judge",
        value="good",
        rationale="The response answers the query well.",
    )

    with mock.patch("mlflow.genai.judges.instructions_judge.invoke_judge_model") as mock_invoke:
        mock_invoke.return_value = mock_feedback

        result = judge(
            inputs={"query": "What is MLflow?"},
            outputs={"response": "MLflow is an open source platform for ML lifecycle."},
        )

        assert result == mock_feedback
        mock_invoke.assert_called_once()

        # Verify the formatted prompt contains substituted values
        formatted_prompt = mock_invoke.call_args[0][1]
        assert "What is MLflow?" in formatted_prompt
        assert "MLflow is an open source platform for ML lifecycle." in formatted_prompt

        # Verify model and assessment name
        assert mock_invoke.call_args[0][0] == "openai/gpt-4o"
        assert mock_invoke.call_args[0][2] == "quality_judge"


def test_field_based_evaluation_with_expectations():
    """Test that expectations fields are merged with inputs/outputs."""
    judge = make_judge(
        name="accuracy_judge",
        instructions="Check if {{response}} matches {{expected_answer}} for {{query}}.",
        model="openai/gpt-4o",
    )

    mock_feedback = Feedback(
        name="accuracy_judge",
        value="correct",
        rationale="The response matches the expected answer.",
    )

    with mock.patch("mlflow.genai.judges.instructions_judge.invoke_judge_model") as mock_invoke:
        mock_invoke.return_value = mock_feedback

        result = judge(
            inputs={"query": "What is 2+2?"},
            outputs={"response": "The answer is four"},
            expectations={"expected_answer": "4"},
        )

        assert result == mock_feedback

        # Verify all fields are in the formatted prompt
        formatted_prompt = mock_invoke.call_args[0][1]
        assert "What is 2+2?" in formatted_prompt
        assert "The answer is four" in formatted_prompt  # From response
        assert "4" in formatted_prompt  # From expected_answer


def test_instructions_without_variables_raises_error():
    """Test that instructions without any variables raise MlflowException."""
    with pytest.raises(MlflowException, match="must contain at least one variable") as exc_info:
        make_judge(
            name="invalid_judge",
            instructions="This template has no variables.",
        )

    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "must contain at least one variable" in str(exc_info.value)


def test_instructions_with_both_trace_and_inputs_raises_error():
    """Test that using both trace and inputs/outputs variables raises error."""
    with pytest.raises(
        MlflowException, match="cannot contain both 'trace' and 'inputs'/'outputs'"
    ) as exc_info:
        make_judge(
            name="invalid_judge",
            instructions="Evaluate {{trace}} and {{inputs}} and {{outputs}}.",
        )

    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "cannot contain both 'trace' and 'inputs'/'outputs'" in str(exc_info.value)


def test_missing_required_template_fields_raises_error():
    """Test that missing required template fields in call raises error."""
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate {{query}} and {{response}} based on {{criteria}}.",
    )

    with pytest.raises(MlflowException, match="criteria.*missing from inputs, outputs") as exc_info:
        # Missing 'criteria' field
        judge(
            inputs={"query": "What is MLflow?"},
            outputs={"response": "MLflow is a platform."},
        )

    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "criteria" in str(exc_info.value)
    assert "missing from inputs, outputs, and expectations" in str(exc_info.value)


def test_specifying_both_trace_and_inputs_in_call_raises_error():
    """Test that specifying both trace and inputs/outputs in call raises error."""
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate {{response}}.",
    )

    # Create a mock trace
    mock_trace = mock.Mock()

    with pytest.raises(
        MlflowException, match="Cannot specify both 'trace' and 'inputs'/'outputs'/'expectations'"
    ) as exc_info:
        judge(
            trace=mock_trace,
            inputs={"query": "test"},
        )

    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "Cannot specify both 'trace' and 'inputs'/'outputs'/'expectations'" in str(
        exc_info.value
    )


def test_specifying_both_trace_and_expectations_in_call_raises_error():
    """Test that specifying both trace and expectations in call raises error."""
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate {{response}}.",
    )

    # Create a mock trace
    mock_trace = mock.Mock()

    with pytest.raises(
        MlflowException, match="Cannot specify both 'trace' and 'inputs'/'outputs'/'expectations'"
    ) as exc_info:
        judge(
            trace=mock_trace,
            expectations={"expected_answer": "42"},
        )

    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "Cannot specify both 'trace' and 'inputs'/'outputs'/'expectations'" in str(
        exc_info.value
    )


def test_databricks_model_with_trace_variable_raises_error():
    """Test that databricks model with trace variable raises error."""
    with pytest.raises(
        MlflowException, match="Model cannot be 'databricks' when using 'trace' variable"
    ) as exc_info:
        make_judge(
            name="test_judge",
            instructions="Evaluate the {{trace}}.",
            model="databricks",
        )

    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "Model cannot be 'databricks' when using 'trace' variable" in str(exc_info.value)


def test_trace_with_custom_variables_raises_error():
    """Test that using trace with custom variables raises error."""
    with pytest.raises(
        MlflowException, match="When using 'trace' variable, no other variables are allowed"
    ) as exc_info:
        make_judge(
            name="test_judge",
            instructions="Evaluate the {{trace}} with {{custom_field}} and {{another_field}}.",
            model="openai/gpt-4o",
        )

    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "no other variables are allowed" in str(exc_info.value)
    assert "custom_field" in str(exc_info.value) or "another_field" in str(exc_info.value)


def test_expectations_template_variable_raises_not_implemented():
    """Test that {{expectations}} as a template variable raises NotImplementedError."""
    with pytest.raises(
        NotImplementedError, match="'expectations' template variable.*not yet supported"
    ) as exc_info:
        make_judge(
            name="test_judge",
            instructions="Evaluate {{response}} against {{expectations}}.",
        )

    assert "'expectations' template variable" in str(exc_info.value)
    assert "not yet supported" in str(exc_info.value)


def test_expectations_parameter_provides_field_values():
    """Test that expectations parameter can provide field values."""
    judge = make_judge(
        name="test_judge",
        instructions="Check if {{answer}} equals {{expected_value}}.",
    )

    mock_feedback = Feedback(
        name="test_judge",
        value="correct",
        rationale="Values match.",
    )

    with mock.patch("mlflow.genai.judges.instructions_judge.invoke_judge_model") as mock_invoke:
        mock_invoke.return_value = mock_feedback

        # Provide expected_value through expectations
        result = judge(
            outputs={"answer": "36"},
            expectations={"expected_value": "42"},
        )

        assert result == mock_feedback

        # Verify both fields are in the formatted prompt
        formatted_prompt = mock_invoke.call_args[0][1]
        assert "36" in formatted_prompt
        assert "42" in formatted_prompt


def test_trace_based_evaluation_not_implemented():
    """Test that trace-based evaluation raises NotImplementedError."""
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate the {{trace}}.",
        model="openai/gpt-4o",  # Not databricks
    )

    mock_trace = mock.Mock()

    with pytest.raises(
        NotImplementedError, match="Trace-based evaluation is not yet implemented"
    ) as exc_info:
        judge(trace=mock_trace)

    assert "Trace-based evaluation is not yet implemented" in str(exc_info.value)


def test_neither_trace_nor_inputs_outputs_raises_error():
    """Test that calling without trace or inputs/outputs raises error."""
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate {{response}}.",
    )

    with pytest.raises(
        MlflowException, match="Must specify either 'trace' or 'inputs'/'outputs'"
    ) as exc_info:
        judge()  # No arguments

    assert exc_info.value.error_code == "INVALID_PARAMETER_VALUE"
    assert "Must specify either 'trace' or 'inputs'/'outputs'" in str(exc_info.value)
