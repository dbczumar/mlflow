"""
MLflow Issues CLI for managing experiment issues and comments.

This module provides a command-line interface for working with MLflow issues
and issue comments, including creating, retrieving, updating, deleting, and searching.

AVAILABLE COMMANDS:
    get                 Retrieve issue details by ID
    create              Create a new issue for an experiment
    update              Update an existing issue
    delete              Delete an issue by ID
    search              Search issues within an experiment
    get-comment         Retrieve comment details by ID
    create-comment      Add a comment to an issue
    update-comment      Update an existing comment
    delete-comment      Delete a comment by ID
    search-comments     Search comments for an issue

EXAMPLE USAGE:
    # Create a new issue
    mlflow issues create --experiment-id 1 --name "Model performance degradation" \
        --description "Accuracy dropped by 5% after last deployment"

    # Get issue details
    mlflow issues get --issue-id abc123-def456

    # Update issue state
    mlflow issues update --issue-id abc123-def456 --state open

    # Search issues in an experiment
    mlflow issues search --experiment-id 1 --states open,draft

    # Delete an issue
    mlflow issues delete --issue-id abc123-def456

    # Add a comment to an issue
    mlflow issues create-comment --issue-id abc123 --content "Investigating this issue"

    # Search comments for an issue
    mlflow issues search-comments --issue-id abc123

ISSUE STATES:
    - draft: Created by analysis, pending review
    - open: User confirmed as valid issue
    - closed: Issue has been resolved

For detailed help on any command, use:
    mlflow issues COMMAND --help
"""

import json

import click

from mlflow.entities.issue import IssueState
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from mlflow.tracking import MlflowClient
from mlflow.utils.string_utils import _create_table

# Valid issue states for CLI choices
VALID_STATES = [IssueState.DRAFT, IssueState.OPEN, IssueState.CLOSED]

# Define reusable options
EXPERIMENT_ID = click.option(
    "--experiment-id",
    "-x",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    required=True,
    help="Experiment ID. Can be set via MLFLOW_EXPERIMENT_ID env var.",
)
ISSUE_ID = click.option(
    "--issue-id",
    type=click.STRING,
    required=True,
    help="Unique identifier of the issue.",
)
COMMENT_ID = click.option(
    "--comment-id",
    type=click.STRING,
    required=True,
    help="Unique identifier of the comment.",
)


def _issue_to_dict(issue):
    """Convert IssueEntity to dictionary for JSON output."""
    return {
        "issue_id": issue.issue_id,
        "experiment_id": issue.experiment_id,
        "name": issue.name,
        "description": issue.description,
        "state": issue.state,
        "creation_time": issue.creation_time,
        "last_update_time": issue.last_update_time,
        "tags": issue.tags,
    }


def _comment_to_dict(comment):
    """Convert IssueCommentEntity to dictionary for JSON output."""
    return {
        "comment_id": comment.comment_id,
        "issue_id": comment.issue_id,
        "content": comment.content,
        "author": comment.author,
        "creation_time": comment.creation_time,
        "last_update_time": comment.last_update_time,
    }


@click.group("issues")
def commands():
    """
    Manage issues. To manage issues associated with a tracking server, set the
    MLFLOW_TRACKING_URI environment variable to the URL of the desired server.

    \b
    ISSUE SCHEMA:
    issue_id                  # Unique issue identifier (UUID)
    experiment_id             # MLflow experiment ID
    name                      # Issue title/name
    description               # Detailed description
    state                     # Issue state: draft, open, closed
    creation_time             # Creation timestamp (milliseconds)
    last_update_time          # Last update timestamp (milliseconds)
    tags                      # Key-value metadata pairs
    """


@commands.command("get")
@ISSUE_ID
def get_issue(issue_id: str) -> None:
    """
    Get issue details as JSON.

    \b
    Example:
    mlflow issues get --issue-id abc123-def456
    """
    client = MlflowClient()
    issue = client.get_issue(issue_id)
    json_issue = json.dumps(_issue_to_dict(issue), indent=2)
    click.echo(json_issue)


@commands.command("create")
@EXPERIMENT_ID
@click.option(
    "--name",
    type=click.STRING,
    required=True,
    help="Name/title of the issue.",
)
@click.option(
    "--description",
    type=click.STRING,
    help="Detailed description of the issue.",
)
@click.option(
    "--state",
    type=click.Choice(VALID_STATES),
    default=IssueState.DRAFT,
    help="Initial state of the issue (default: draft).",
)
@click.option(
    "--tags",
    type=click.STRING,
    help='Additional metadata tags as JSON string (e.g., \'{"priority": "high"}\').',
)
def create_issue(
    experiment_id: str,
    name: str,
    description: str | None = None,
    state: str = IssueState.DRAFT,
    tags: str | None = None,
) -> None:
    """
    Create a new issue for an experiment.

    \b
    Examples:
    # Create a basic issue
    mlflow issues create --experiment-id 1 --name "Model accuracy drop"

    \b
    # Create with full details
    mlflow issues create --experiment-id 1 \\
        --name "Model performance degradation" \\
        --description "Accuracy dropped by 5% after last deployment" \\
        --state open \\
        --tags '{"priority": "high", "component": "inference"}'
    """
    client = MlflowClient()

    # Parse tags if provided
    tags_dict = json.loads(tags) if tags else None

    issue = client.create_issue(
        experiment_id=experiment_id,
        name=name,
        description=description,
        state=state,
        tags=tags_dict,
    )

    click.echo(f"Created issue '{issue.name}' with ID: {issue.issue_id}")


@commands.command("update")
@ISSUE_ID
@click.option(
    "--name",
    type=click.STRING,
    help="Updated name/title of the issue.",
)
@click.option(
    "--description",
    type=click.STRING,
    help="Updated description of the issue.",
)
@click.option(
    "--state",
    type=click.Choice(VALID_STATES),
    help="Updated state of the issue.",
)
@click.option(
    "--tags",
    type=click.STRING,
    help="Tags to merge with existing tags as JSON string.",
)
def update_issue(
    issue_id: str,
    name: str | None = None,
    description: str | None = None,
    state: str | None = None,
    tags: str | None = None,
) -> None:
    """
    Update an existing issue.

    \b
    Examples:
    # Update issue state
    mlflow issues update --issue-id abc123-def456 --state open

    \b
    # Update multiple fields
    mlflow issues update --issue-id abc123-def456 \\
        --name "Updated issue title" \\
        --description "New description" \\
        --state closed

    \b
    # Add/update tags
    mlflow issues update --issue-id abc123-def456 \\
        --tags '{"resolution": "fixed in v2.0"}'
    """
    client = MlflowClient()

    # Parse tags if provided
    tags_dict = json.loads(tags) if tags else None

    client.update_issue(
        issue_id=issue_id,
        name=name,
        description=description,
        state=state,
        tags=tags_dict,
    )

    click.echo(f"Updated issue {issue_id}.")


@commands.command("delete")
@ISSUE_ID
def delete_issue(issue_id: str) -> None:
    """
    Delete an issue by ID.

    \b
    Example:
    mlflow issues delete --issue-id abc123-def456
    """
    client = MlflowClient()
    client.delete_issue(issue_id)
    click.echo(f"Deleted issue {issue_id}.")


@commands.command("search")
@EXPERIMENT_ID
@click.option(
    "--states",
    type=click.STRING,
    help="Comma-separated list of states to filter by (e.g., 'open,draft'). "
    "If not specified, returns issues in all states.",
)
@click.option(
    "--max-results",
    type=click.INT,
    default=100,
    help="Maximum number of issues to return (default: 100).",
)
@click.option(
    "--page-token",
    type=click.STRING,
    help="Token for pagination from previous search.",
)
@click.option(
    "--output",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format: 'table' (default) or 'json'.",
)
def search_issues(
    experiment_id: str,
    states: str | None = None,
    max_results: int = 100,
    page_token: str | None = None,
    output: str = "table",
) -> None:
    """
    Search issues within an experiment.

    \b
    Examples:
    # Search all issues in an experiment
    mlflow issues search --experiment-id 1

    \b
    # Filter by states
    mlflow issues search --experiment-id 1 --states open,draft

    \b
    # Get JSON output
    mlflow issues search --experiment-id 1 --output json

    \b
    # Paginate results
    mlflow issues search --experiment-id 1 --max-results 10
    """
    client = MlflowClient()

    # Parse states if provided
    states_list = [s.strip() for s in states.split(",")] if states else None

    issues = client.search_issues(
        experiment_id=experiment_id,
        states=states_list,
        max_results=max_results,
        page_token=page_token,
    )

    if output == "json":
        result = {
            "issues": [_issue_to_dict(issue) for issue in issues],
            "next_page_token": issues.token if hasattr(issues, "token") else None,
        }
        click.echo(json.dumps(result, indent=2))
    else:
        # Table output
        headers = ["issue_id", "name", "state", "description", "creation_time"]
        table = []
        for issue in issues:
            desc = issue.description or ""
            # Truncate long descriptions for table display
            if len(desc) > 50:
                desc = desc[:47] + "..."
            table.append(
                [
                    issue.issue_id,
                    issue.name,
                    issue.state,
                    desc,
                    str(issue.creation_time) if issue.creation_time else "",
                ]
            )

        click.echo(_create_table(table, headers=headers))

        if hasattr(issues, "token") and issues.token:
            click.echo(f"\nNext page token: {issues.token}")


# Issue Comment Commands


@commands.command("get-comment")
@COMMENT_ID
def get_issue_comment(comment_id: str) -> None:
    """
    Get comment details as JSON.

    \b
    Example:
    mlflow issues get-comment --comment-id abc123-def456
    """
    client = MlflowClient()
    comment = client.get_issue_comment(comment_id)
    json_comment = json.dumps(_comment_to_dict(comment), indent=2)
    click.echo(json_comment)


@commands.command("create-comment")
@ISSUE_ID
@click.option(
    "--content",
    type=click.STRING,
    required=True,
    help="Comment text content.",
)
@click.option(
    "--author",
    type=click.STRING,
    help="Author name or identifier.",
)
def create_issue_comment(
    issue_id: str,
    content: str,
    author: str | None = None,
) -> None:
    """
    Add a comment to an issue.

    \b
    Examples:
    # Create a simple comment
    mlflow issues create-comment --issue-id abc123 --content "Investigating this issue"

    \b
    # Create with author
    mlflow issues create-comment --issue-id abc123 \\
        --content "Root cause identified" \\
        --author "user@example.com"
    """
    client = MlflowClient()

    comment = client.create_issue_comment(
        issue_id=issue_id,
        content=content,
        author=author,
    )

    click.echo(f"Created comment with ID: {comment.comment_id}")


@commands.command("update-comment")
@COMMENT_ID
@click.option(
    "--content",
    type=click.STRING,
    required=True,
    help="Updated comment content.",
)
def update_issue_comment(
    comment_id: str,
    content: str,
) -> None:
    """
    Update an existing comment.

    \b
    Example:
    mlflow issues update-comment --comment-id abc123-def456 --content "Updated text"
    """
    client = MlflowClient()

    client.update_issue_comment(
        comment_id=comment_id,
        content=content,
    )

    click.echo(f"Updated comment {comment_id}.")


@commands.command("delete-comment")
@COMMENT_ID
def delete_issue_comment(comment_id: str) -> None:
    """
    Delete a comment by ID.

    \b
    Example:
    mlflow issues delete-comment --comment-id abc123-def456
    """
    client = MlflowClient()
    client.delete_issue_comment(comment_id)
    click.echo(f"Deleted comment {comment_id}.")


@commands.command("search-comments")
@ISSUE_ID
@click.option(
    "--max-results",
    type=click.INT,
    default=100,
    help="Maximum number of comments to return (default: 100).",
)
@click.option(
    "--page-token",
    type=click.STRING,
    help="Token for pagination from previous search.",
)
@click.option(
    "--output",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format: 'table' (default) or 'json'.",
)
def search_issue_comments(
    issue_id: str,
    max_results: int = 100,
    page_token: str | None = None,
    output: str = "table",
) -> None:
    """
    Search comments for an issue.

    \b
    Examples:
    # Search all comments for an issue
    mlflow issues search-comments --issue-id abc123

    \b
    # Get JSON output
    mlflow issues search-comments --issue-id abc123 --output json

    \b
    # Paginate results
    mlflow issues search-comments --issue-id abc123 --max-results 10
    """
    client = MlflowClient()

    comments = client.search_issue_comments(
        issue_id=issue_id,
        max_results=max_results,
        page_token=page_token,
    )

    if output == "json":
        result = {
            "comments": [_comment_to_dict(comment) for comment in comments],
            "next_page_token": comments.token if hasattr(comments, "token") else None,
        }
        click.echo(json.dumps(result, indent=2))
    else:
        # Table output
        headers = ["comment_id", "author", "content", "creation_time"]
        table = [
            [
                comment.comment_id,
                comment.author or "",
                comment.content or "",
                str(comment.creation_time) if comment.creation_time else "",
            ]
            for comment in comments
        ]

        click.echo(_create_table(table, headers=headers))

        if hasattr(comments, "token") and comments.token:
            click.echo(f"\nNext page token: {comments.token}")
