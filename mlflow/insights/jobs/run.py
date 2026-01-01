import json
import logging
import os

import mlflow
from mlflow.entities.assessment import Issue as IssueAssessment
from mlflow.insights.jobs._extract_evidences import extract_trace_evidences
from mlflow.insights.jobs._discover_issues import discover_issues
from mlflow.insights.jobs._understand_project import _understand_project
from mlflow.metrics.genai.model_utils import _parse_model_uri
from mlflow.server.jobs import job

_logger = logging.getLogger(__name__)


@job(
    name="analyze-issues",
    max_workers=1,
    pip_requirements=["litellm"]
)
def analyze_issues(
    filter_string: str,
    experiment_id: str,
    user_question: str,
    model: str,
) -> str:
    """
    Analyze issues for the trace.
    """
    mlflow.set_experiment(experiment_id=experiment_id)
    provider, model_name = _parse_model_uri(model)

    traces = mlflow.search_traces(filter_string=filter_string, return_type="list", include_spans=False)
    trace_ids = [t.info.trace_id for t in traces]
    issues = _analyze_traces(
        experiment_id=experiment_id,
        trace_ids=trace_ids,
        user_question=user_question,
        provider=provider,
        model_name=model_name,
    )

    for issue in issues:
        logged_issue = mlflow.MlflowClient().create_issue(
            experiment_id="3",
            name=issue["name"],
            description=issue["description"],
            state="DRAFT",
        )

        for evidence in issue["evidences"]:
            trace_id = evidence["trace_id"]
            assessment = IssueAssessment(
                issue_id=logged_issue.issue_id,
                issue_name=logged_issue.name,
                trace_id=trace_id,
                rationale=json.dumps(evidence),
            )
            assessment = mlflow.log_assessment(trace_id=trace_id, assessment=assessment)
    return issues



def _analyze_traces(
    experiment_id: str,
    trace_ids: list[str],
    user_question: str,
    provider: str,
    model_name: str,
    enable_tracing: bool = False,
) -> list[dict]:
    """
    Analyze traces to discover issues.

    Args:
        experiment_id: The experiment ID.
        trace_ids: List of trace IDs to analyze.
        user_question: The user's question or focus area for analysis.
        provider: The LLM provider.
        model_name: The model name.

    Returns:
        List of discovered issues with their evidences.
    """
    if enable_tracing:
        os.environ["MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING"] = "true"
        mlflow.litellm.autolog()

    wrap_trace = mlflow.trace if enable_tracing else lambda x: x

    project_summary = wrap_trace(_understand_project)(
        experiment_id=experiment_id,
        trace_ids=trace_ids,
        provider=provider,
        model_name=model_name
    )

    evidences = wrap_trace(extract_trace_evidences)(
        trace_ids=trace_ids,
        user_question=user_question,
        project_summary=project_summary,
        provider=provider,
        model_name=model_name,
    )

    issues = wrap_trace(discover_issues)(
        issue_candidates=evidences,
        project_summary=project_summary,
        user_question=user_question,
        provider=provider,
        model_name=model_name,
    )

    return issues