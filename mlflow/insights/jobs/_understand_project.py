import json
import logging
import mlflow
from pydantic import BaseModel

_logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """
Read the experiment description and view a few traces from the experiment to understand

1. What is the main objective of the experiment?
2. What problem do the agents or LLM applications in this experiment try to solve?
3. What are the overall structure of the agents or LLM applications? e.g., is it a pipeline, a workflow, RAG, ReAct agents, etc.

The response should be a summary of the project answers in a JSON format:

{
    "objective": "The main objective of the experiment.",
    "problem": "The problem do the agents or LLM applications in this experiment try to solve.",
    "structure": "The overall structure of the agents or LLM applications."
}
"""

_USER_PROMPT = """
<experiment_description>
{description}
</experiment_description>
"""

class ProjectSummary(BaseModel):
    objective: str
    problem: str
    structure: str

def _understand_project(
    experiment_id: str,
    trace_ids: list[str],
    provider: str,
    model_name: str,
) -> ProjectSummary:
    """
    Understand the project by reading experiment description and a few traces.
    """
    from mlflow.types.llm import ChatMessage
    from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm_and_handle_tools

    _logger.info(f"Reading project with model {model_name} from provider {provider}")

    experiment = mlflow.get_experiment(experiment_id)
    description = experiment.tags.get("mlflow.note.content", "")
    messages = [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=_USER_PROMPT.format(description=description))
    ]

    response, _ = _invoke_litellm_and_handle_tools(
        provider=provider,
        model_name=model_name,
        messages=messages,
        response_format=ProjectSummary,
        trace=mlflow.get_trace(trace_ids[0]),  # TODO: We should sample multiple traces
        num_retries=1,
    )
    return ProjectSummary.model_validate_json(response)