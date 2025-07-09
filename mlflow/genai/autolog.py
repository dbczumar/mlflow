import fnmatch
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from mlflow.tracking.fluent import _set_active_model
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.git_utils import get_git_branch, get_git_commit, get_git_repo_url

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "genai"


def _load_mlflowignore_patterns(repo_path: str) -> list[str]:
    """
    Load ignore patterns from .mlflowignore file in the repository root.
    Returns a list of patterns, similar to .gitignore format.
    """
    mlflowignore_path = os.path.join(repo_path, ".mlflowignore")
    patterns = []

    if os.path.exists(mlflowignore_path):
        try:
            with open(mlflowignore_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        patterns.append(line)
        except (OSError, IOError) as e:
            _logger.debug(f"Could not read .mlflowignore file: {e}")

    return patterns


def _should_ignore_file(file_path: str, ignore_patterns: list[str], repo_path: str) -> bool:
    """
    Check if a file should be ignored based on .mlflowignore patterns.
    Supports gitignore-style patterns including:
    - Glob patterns (*.log, *.tmp)
    - Directory patterns (build/, temp/)
    - Recursive patterns (**/__pycache__/*)
    """
    if not ignore_patterns:
        return False

    # Convert to relative path from repo root
    try:
        rel_path = os.path.relpath(file_path, repo_path)
        # Normalize path separators
        rel_path = rel_path.replace(os.sep, "/")
    except ValueError:
        # File is outside repo, don't ignore
        return False

    for pattern in ignore_patterns:
        # Handle directory patterns (ending with /)
        if pattern.endswith("/"):
            dir_pattern = pattern[:-1]
            # Check if file is in this directory
            if rel_path.startswith(dir_pattern + "/") or rel_path == dir_pattern:
                return True

        # Handle glob patterns
        elif fnmatch.fnmatch(rel_path, pattern):
            return True

        # Handle ** patterns (recursive)
        elif "**" in pattern:
            # Use pathlib for ** pattern matching
            if Path(rel_path).match(pattern):
                return True

    return False


def _get_tracked_files_hash(path: str) -> Optional[str]:
    """
    Generate a hash of all tracked file contents in the repository, excluding files
    that match patterns in .mlflowignore (if present).

    .mlflowignore supports gitignore-style patterns:
    - *.log (glob patterns)
    - temp/ (directory patterns)
    - **/__pycache__/** (recursive patterns)

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
        repo_path = repo.working_dir

        # Load .mlflowignore patterns
        ignore_patterns = _load_mlflowignore_patterns(repo_path)

        # Get all tracked files in the repository
        tracked_files = []
        for item in repo.index.entries:
            file_path = item[0]
            full_path = os.path.join(repo_path, file_path)

            # Skip if file doesn't exist or should be ignored
            if not os.path.exists(full_path):
                continue
            if _should_ignore_file(full_path, ignore_patterns, repo_path):
                continue

            tracked_files.append(file_path)

        if not tracked_files:
            return None

        # Sort files for consistent hash ordering
        tracked_files.sort()

        # Create hash of all tracked file contents
        hasher = hashlib.sha256()
        for file_path in tracked_files:
            full_path = os.path.join(repo_path, file_path)
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


def _generate_git_model_name(path: str, versioning_strategy: str = "content") -> Optional[str]:
    """
    Generate a model name based on git repository information.

    Args:
        path: Path to the git repository
        versioning_strategy: Strategy for model versioning:
            - "content": Include file content hash (default)
            - "commit": Only include commit and dirty state

    Format:
        - content: {repo_name}-{branch}-{commit_short}-{content_hash}
        - commit: {repo_name}-{branch}-{commit_short}[-dirty]
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

    if versioning_strategy == "content":
        # Include content hash to track file changes
        content_hash = _get_tracked_files_hash(path)
        if content_hash:
            return f"{repo_name}-{branch}-{commit_short}-{content_hash}"
        else:
            return f"{repo_name}-{branch}-{commit_short}"
    elif versioning_strategy == "commit":
        # Only include commit and dirty state
        is_dirty = _get_git_dirty_state(path)
        dirty_suffix = "-dirty" if is_dirty else ""
        return f"{repo_name}-{branch}-{commit_short}{dirty_suffix}"
    else:
        raise ValueError(f"Unknown versioning strategy: {versioning_strategy}")


@experimental(version="3.0.0")
def autolog(
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    log_traces=True,
    enable_git_versioning=False,
    git_versioning_strategy="content",
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
        git_versioning_strategy: Strategy for git-based model versioning. Options:
            - "content": Generate new model names for every tracked file content change
              (format: {repo}-{branch}-{commit}-{content_hash})
            - "commit": Generate new model names only for commit/branch changes and dirty state
              (format: {repo}-{branch}-{commit}[-dirty])
            Default to "content".
    """
    if not disable and enable_git_versioning:
        # Validate versioning strategy
        if git_versioning_strategy not in ("content", "commit"):
            raise ValueError(
                f"Invalid git_versioning_strategy: {git_versioning_strategy}. "
                "Must be 'content' or 'commit'."
            )

        # Generate model name based on git information
        git_model_name = _generate_git_model_name(os.getcwd(), git_versioning_strategy)
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
