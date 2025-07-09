import logging
import os
from typing import Optional

from mlflow.tracking.fluent import _set_active_model
from mlflow.utils.annotations import experimental
from mlflow.utils.git_utils import get_git_branch, get_git_commit, get_git_repo_url

_logger = logging.getLogger(__name__)


def _get_git_dirty_state(path: str) -> bool:
    """
    Check if the git repository at the specified path has uncommitted changes.
    Returns True if the working directory is dirty, False otherwise.
    """
    try:
        from git import Repo
    except ImportError as e:
        _logger.warning(
            "Failed to import Git (the Git executable is probably not on your PATH),"
            " so Git dirty state is not available. Error: %s",
            e,
        )
        return False

    try:
        repo = Repo(path, search_parent_directories=True)
        return repo.is_dirty()
    except Exception:
        return False


def _generate_git_model_name(path: str) -> Optional[str]:
    """
    Generate a model name based on git repository information.

    Args:
        path: Path to the git repository

    Format:
        {repo_name}/{branch}/{commit_short}[-dirty]
    """
    branch = get_git_branch(path)
    commit = get_git_commit(path)

    if not branch or not commit:
        return None

    # Get repository name from URL if available, otherwise use directory name
    repo_url = get_git_repo_url(path)
    if repo_url:
        repo_name = repo_url.split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
    else:
        # Fallback to directory name if no remote URL
        repo_name = os.path.basename(os.path.abspath(path))

    # Use short commit hash (first 7 characters)
    commit_short = commit[:7]

    # Include commit and dirty state
    is_dirty = _get_git_dirty_state(path)
    dirty_suffix = "-dirty" if is_dirty else ""
    return f"{repo_name}/{branch}/{commit_short}{dirty_suffix}"


@experimental(version="3.0.0")
def autolog(
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    log_traces=True,
    enable_git_versioning=False,
):
    """
    Enables (or disables) and configures autologging for GenAI operations to MLflow.

    Args:
        disable: If ``True``, disables the GenAI autologging integration. If ``False``,
            enables the GenAI autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
            If ``False``, autologged content is logged to the active fluent run,
            which may be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of
            GenAI libraries that have not been tested against this version of the MLflow
            client or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during GenAI
            autologging. If ``False``, show all events and warnings during GenAI
            autologging.
        log_traces: If ``True``, traces are logged for GenAI operations. If ``False``, no traces are
            collected during inference. Default to ``True``.
        enable_git_versioning: If ``True``, automatically set the active model based on the
            current git repository state. If ``False``, no git-based model naming is applied.
            Default to ``False``.
    """
    if not disable and enable_git_versioning:
        # Generate model name based on git information
        git_model_name = _generate_git_model_name(os.getcwd())
        if git_model_name:
            try:
                _set_active_model(name=git_model_name, set_by_user=False)
                if not silent:
                    _logger.info(f"Set active model to '{git_model_name}' based on git state")
            except Exception as e:
                if not silent:
                    _logger.warning(f"Failed to set active model based on git state: {e}")
        else:
            if not silent:
                _logger.warning(
                    "Could not generate git-based model name "
                    "(not a git repository or missing git info)"
                )

    _autolog(
        disable=disable,
        exclusive=exclusive,
        disable_for_unsupported_versions=disable_for_unsupported_versions,
        silent=silent,
        log_traces=log_traces,
    )


def _autolog(
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    log_traces=True,
):
    """
    Internal autolog implementation. This is a placeholder for future GenAI-specific
    autologging functionality.
    """
    import mlflow

    mlflow.autolog(
        disable=disable,
        exclusive=exclusive,
        disable_for_unsupported_versions=disable_for_unsupported_versions,
        silent=silent,
        log_traces=log_traces,
    )
