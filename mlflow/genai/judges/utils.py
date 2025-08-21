import json
import re
from dataclasses import asdict

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.utils.enum_utils import StrEnum
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.utils.uri import is_databricks_uri

# "endpoints" is a special case for Databricks model serving endpoints.
_NATIVE_PROVIDERS = ["openai", "anthropic", "bedrock", "mistral", "endpoints"]

_DEFAULT_MODEL_DATABRICKS = "databricks"


def get_default_model() -> str:
    if is_databricks_uri(mlflow.get_tracking_uri()):
        return _DEFAULT_MODEL_DATABRICKS
    else:
        return "openai:/gpt-4.1-mini"


def format_prompt(prompt: str, **values) -> str:
    """Format double-curly variables in the prompt template."""
    for key, value in values.items():
        prompt = re.sub(r"\{\{\s*" + key + r"\s*\}\}", str(value), prompt)
    return prompt


def _sanitize_justification(justification: str) -> str:
    # Some judge prompts instruct the model to think step by step.
    return justification.replace("Let's think step by step. ", "")


def invoke_judge_model(
    model_uri: str, prompt: str, assessment_name: str, trace: Trace | None = None
) -> Feedback:
    """
    Invoke the judge model.

    First, try to invoke the judge model via litellm. If litellm is not installed,
    fallback to native parsing using the AI Gateway adapters.

    Args:
        model_uri: The model URI.
        prompt: The prompt to evaluate.
        assessment_name: The name of the assessment.
        trace: Optional trace object for context (default=None).
    """
    from mlflow.metrics.genai.model_utils import (
        _parse_model_uri,
        get_endpoint_type,
        score_model_on_payload,
    )

    provider, model_name = _parse_model_uri(model_uri)

    # Try litellm first for better performance.
    if _is_litellm_available():
        response = _invoke_litellm(provider, model_name, prompt, trace)
    elif trace is not None:
        raise MlflowException(
            "LiteLLM is required for using traces with judge models. "
            "Please install it with `pip install litellm`.",
            error_code=BAD_REQUEST,
        )
    elif provider in _NATIVE_PROVIDERS:
        response = score_model_on_payload(
            model_uri=model_uri,
            payload=prompt,
            endpoint_type=get_endpoint_type(model_uri) or "llm/v1/chat",
        )
    else:
        raise MlflowException(
            f"LiteLLM is required for using '{provider}' LLM. Please install it with "
            "`pip install litellm`.",
            error_code=BAD_REQUEST,
        )

    try:
        response_dict = json.loads(response)
        feedback = Feedback(
            name=assessment_name,
            value=response_dict["result"],
            rationale=_sanitize_justification(response_dict.get("rationale", "")),
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=model_uri,
            ),
        )
    except json.JSONDecodeError as e:
        raise MlflowException(
            f"Failed to parse the response from the judge model. Response: {response}",
            error_code=INVALID_PARAMETER_VALUE,
        ) from e

    return feedback


def _is_litellm_available() -> bool:
    try:
        import litellm  # noqa: F401

        return True
    except ImportError:
        return False


def _invoke_litellm(provider: str, model_name: str, prompt: str, trace: Trace | None) -> str:
    """Invoke the judge model via litellm."""
    import litellm

    from mlflow.genai.judges.tools import list_judge_tools
    from mlflow.genai.judges.tools.registry import _judge_tool_registry
    from mlflow.types.llm import ToolCall

    litellm_model_uri = f"{provider}/{model_name}"
    messages = [{"role": "user", "content": prompt}]

    # Get tools if trace is provided (empty list if no trace or no tools)
    tools = []
    if trace is not None:
        judge_tools = list_judge_tools()
        tools = [asdict(tool.get_definition()) for tool in judge_tools]

    # Main completion loop - handles both tool and non-tool cases
    while True:
        try:
            response = litellm.completion(
                model=litellm_model_uri,
                messages=messages,
                tools=tools if tools else None,  # Only pass tools if we have them
                tool_choice="auto" if tools else None,
            )

            message = response.choices[0].message

            # If no tool calls, we're done
            if not message.tool_calls:
                return message.content

            # Add assistant's message to history
            messages.append(message.model_dump())

            # Execute tool calls
            for tool_call in message.tool_calls:
                try:
                    # Create MLflow ToolCall and invoke
                    mlflow_tool_call = ToolCall(
                        id=tool_call.id,
                        function={
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    )
                    result = _judge_tool_registry.invoke(mlflow_tool_call, trace)

                    # Convert result to string if needed
                    if not isinstance(result, str):
                        result = json.dumps(result)

                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": result,
                        }
                    )
                except Exception as e:
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": f"Error: {e!s}",
                        }
                    )

        except Exception as e:
            raise MlflowException(f"Failed to invoke the judge model via litellm: {e}") from e


class CategoricalRating(StrEnum):
    """
    A categorical rating for an assessment.

    Example:
        .. code-block:: python

            from mlflow.genai.judges import CategoricalRating
            from mlflow.entities import Feedback

            # Create feedback with categorical rating
            feedback = Feedback(
                name="my_metric", value=CategoricalRating.YES, rationale="The metric is passing."
            )
    """

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member == value:
                return member
        return cls.UNKNOWN
