import hashlib
import logging
import os
from typing import Optional

from mlflow.tracking.fluent import _set_active_model
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.git_utils import get_git_branch, get_git_commit, get_git_repo_url

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "genai"


def _get_tracked_files_hash(path: str) -> Optional[str]:
    """
    Generate a hash of all tracked file contents in the repository.
    Returns None if not in a git repository or if there are no tracked files.
    """
    try:
        from git import Repo
    except ImportError as e:
        _logger.warning(
            "Failed to import Git (the Git executable is probably not on your PATH),"
            " so tracked files hash is not available. Error: %s",
            e,
        )
        return None

    try:
        repo = Repo(path, search_parent_directories=True)

        # Get all tracked files in the repository
        tracked_files = []
        for item in repo.index.entries:
            file_path = item[0]
            full_path = os.path.join(repo.working_dir, file_path)
            if os.path.exists(full_path):
                tracked_files.append(file_path)

        if not tracked_files:
            return None

        # Sort files for consistent hash ordering
        tracked_files.sort()

        # Create hash of all tracked file contents
        hasher = hashlib.sha256()
        for file_path in tracked_files:
            full_path = os.path.join(repo.working_dir, file_path)
            try:
                with open(full_path, "rb") as f:
                    # Include file path and content in hash
                    hasher.update(file_path.encode("utf-8"))
                    hasher.update(f.read())
            except (OSError, IOError):
                # Skip files that can't be read
                continue

        return hasher.hexdigest()[:12]  # Use first 12 characters of hash
    except Exception:
        return None


def _generate_git_model_name(path: str) -> Optional[str]:
    """
    Generate a model name based on git repository information and tracked file contents.
    Format: {repo_name}-{branch}-{commit_short}-{content_hash}
    """
    branch = get_git_branch(path)
    commit = get_git_commit(path)
    content_hash = _get_tracked_files_hash(path)

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

    # Include content hash to track file changes
    if content_hash:
        return f"{repo_name}-{branch}-{commit_short}-{content_hash}"
    else:
        return f"{repo_name}-{branch}-{commit_short}"


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
            current git repository state (repo, branch, commit, and tracked file contents hash).
            A new model name is generated every time tracked file contents change. If ``False``,
            no git-based model naming is applied. Default to ``False``.
    """
    if not disable and enable_git_versioning:
        # Generate model name based on git information
        git_model_name = _generate_git_model_name(os.getcwd())
        if git_model_name:
            try:
                _set_active_model(name=git_model_name)
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


# This is required by mlflow.autolog()
autolog.integration_name = FLAVOR_NAME


@autologging_integration(FLAVOR_NAME)
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
    # TODO: Implement GenAI-specific autologging functionality
    # This could include patching various GenAI libraries, frameworks, and APIs
    # for now, this is a minimal implementation focused on the git versioning feature
