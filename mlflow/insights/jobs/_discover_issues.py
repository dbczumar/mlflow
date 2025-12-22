import logging
import os
from typing import Any
from pydantic import BaseModel
import litellm

import mlflow
from mlflow.insights.jobs._extract_evidences import IssueCandidate, Evidence
from mlflow.insights.jobs._understand_project import ProjectSummary

_logger = logging.getLogger(__name__)



class Category(BaseModel):
    name: str
    description: str
    evidences: list[Evidence]

class Categories(BaseModel):
    categories: list[Category]



def discover_issues(
    issue_candidates: list[IssueCandidate],
    project_summary: ProjectSummary,
    user_question: str,
    provider: str,
    model_name: str,
) -> list[dict[str, Any]]:
    # To enable tracing for the judge tools
    # os.environ["MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING"] = "true"
    # mlflow.litellm.autolog()

    _logger.info(f"Discovering issues with model {model_name} from provider {provider}")

    response = litellm.completion(
        model=f"{provider}/{model_name}",
        messages=[
            {
                "role": "system",
                "content": _SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": _USER_PROMPT.format(
                    project_summary=project_summary.model_dump_json(),
                    issue_candidates="\n\n".join(issue_candidate.model_dump_json() for issue_candidate in issue_candidates),
                    user_question=user_question),
            },
        ],
        response_format=Categories,
    )

    llm_categories = Categories.model_validate_json(response.choices[0].message.content)
    _logger.info(f"Discovered {len(llm_categories.categories)} issues")

    issues = [
        {
            "issue_id": i,
            "name": category.name,
            "description": category.description,
            "evidences": [e.model_dump() for e in category.evidences],
        }
        for i, category in enumerate(llm_categories.categories)
    ]

    # TODO: Create issue instance in backend and log assessments
    return issues



_SYSTEM_PROMPT = """
## Task: Issue identification
You will be given a list of issue candidates that are extracted from each single trace. Each candidate includes an important information that the quality problems user is interested in.

Your output should be a list of issue categories, each with a name, description, and list of evidence IDs. The category should be concrete and specific. Each category must be an atomic issue that is root caused by a single problem, rather than a generic class of problem that mixes multiple issues.

Maximum number of categories is 10. If some texts are not appropriate for any categories, you can omit them from the output. However, you should try your best to find great set of categories that covers most of the texts yet still not too vague.

Good example:
  - {"name": "Outdated answer", "description": "The referenced documents are from non-latest version of the documentation that misleads users...."}
  - {"name": "Incorrect guidance about MLflow Tracing", "description": "The guidance on MLflow tracing is not correct. Agent confuses tracing and traditional run logging...."}

Bad example:
  - {"name": "The answer is not correct or outdated.", "description": ...} <= This is not an atomic issue. 'Incorrect' and 'outdated' are two separate issues."}
  - {"name": "Unclear description", "description": ...} <= Unclear description is not concrete enough. It is better to be more specific.

Example output:
    {
        "categories": [
            {
               "name": "Outdated answer due to reference to non-latest version of the documentation",
               "description": "The referenced documents are from non-latest version of the documentation that misleads users....",
               "evidences": [
                   {
                       "type": "feedback",
                       "id": "a-12345",
                       "trace_id": "123",
                   },
                   {
                       "type": "feedback",
                       "id": "a-12346",
                       "trace_id": "456",
                   }
               ]
            }
        ]
    }
"""

_USER_PROMPT = """
<project_summary>
{project_summary}
</project_summary>

<user_question>
{user_question}
</user_question>

<issue_candidates>
{issue_candidates}
</issue_candidates>
"""