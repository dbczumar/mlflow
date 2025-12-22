from functools import lru_cache
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import REQUEST_LIMIT_EXCEEDED

import pydantic
from typing import Any


def invoke_llm_agent(
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[pydantic.BaseModel] | None = None,
    max_agent_steps: int = 30,
) -> tuple[str, float | None]:
    import litellm

    from mlflow.genai.judges.tools import list_judge_tools
    from mlflow.genai.judges.adapters.litellm_adapter import (
        _invoke_litellm,
        _prune_messages_exceeding_context_window_length,
    )
    from mlflow.genai.judges.utils.tool_calling_utils import _process_tool_calls

    messages = [litellm.Message(role=msg.role, content=msg.content) for msg in messages]

    tools = []
    judge_tools = list_judge_tools()
    tools = [tool.get_definition().to_dict() for tool in judge_tools]

    def _prune_messages_for_context_window() -> list[litellm.Message] | None:
        # For direct providers, use token-counting based pruning.
        try:
            max_context_length = litellm.get_max_tokens(model)
        except Exception:
            max_context_length = None

        return _prune_messages_exceeding_context_window_length(
            messages, model=model, max_tokens=max_context_length or 100000
        )

    iteration_count = 0

    while True:
        iteration_count += 1
        if iteration_count > max_agent_steps:
            raise MlflowException(
                f"Agent step limit of {max_agent_steps} exceeded. "
                f"This usually indicates the model is not powerful enough to effectively "
                f"analyze the trace. Consider using a more intelligent/powerful model. "
                f"In rare cases, for very complex traces where a large number of agent steps "
                f"might be required, you can increase the number of steps by "
                f"modifying the {max_agent_steps} environment variable.",
                error_code=REQUEST_LIMIT_EXCEEDED,
            )
        try:
            try:
                response = _invoke_litellm(
                    litellm_model=model,
                    messages=messages,
                    tools=tools,
                    num_retries=1,
                    response_format=response_format,
                    include_response_format=True,
                )
            except (litellm.BadRequestError, litellm.UnsupportedParamsError) as e:
                error_str = str(e).lower()
                is_context_window_error = (
                    isinstance(e, litellm.ContextWindowExceededError)
                    or "context length" in error_str
                    or "too many tokens" in error_str
                )
                if is_context_window_error:
                    pruned = _prune_messages_for_context_window()
                    if pruned is None:
                        raise MlflowException(
                            "Context window exceeded and there are no tool calls to truncate. "
                            "The initial prompt may be too long for the model's context window."
                        ) from e
                    messages = pruned
                    continue
                else:
                    raise

            message = response.choices[0].message
            if not message.tool_calls:
                return message.content

            messages.append(message)
            tool_response_messages = _process_tool_calls(tool_calls=message.tool_calls, trace=None)
            messages.extend(tool_response_messages)

        except MlflowException:
            raise
        except Exception as e:
            raise MlflowException(f"Failed to invoke the judge via litellm: {e}") from e


