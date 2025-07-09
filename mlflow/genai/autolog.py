import fnmatch
import logging
import os
from pathlib import Path
from typing import Optional

from mlflow.tracking.fluent import _set_active_model
from mlflow.utils.annotations import experimental
from mlflow.utils.git_utils import get_git_branch, get_git_commit, get_git_repo_url

_logger = logging.getLogger(__name__)


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
            # Check if file is in this directory or is the directory itself
            if rel_path.startswith(dir_pattern + "/") or rel_path == dir_pattern:
                return True
            # Also check if any parent directory matches the glob pattern
            path_parts = rel_path.split("/")
            for i in range(len(path_parts)):
                partial_path = "/".join(path_parts[: i + 1])
                if fnmatch.fnmatch(partial_path, dir_pattern):
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


def _get_git_dirty_state(path: str) -> bool:
    """
    Check if the git repository at the specified path has uncommitted changes,
    respecting .mlflowignore patterns.
    Returns True if the working directory is dirty (excluding ignored files), False otherwise.
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
        repo_path = repo.working_dir

        # Load .mlflowignore patterns
        ignore_patterns = _load_mlflowignore_patterns(repo_path)

        # Get all dirty files (modified, added, deleted, renamed) - NOT untracked files
        dirty_files = set()

        # Modified and deleted files (working tree vs index)
        for item in repo.index.diff(None):
            if item.a_path:  # Handle renames/moves
                dirty_files.add(item.a_path)
            if item.b_path and item.b_path != item.a_path:
                dirty_files.add(item.b_path)

        # Staged files (index vs HEAD) - uncommitted changes in index
        try:
            for item in repo.index.diff("HEAD"):
                if item.a_path:
                    dirty_files.add(item.a_path)
                if item.b_path and item.b_path != item.a_path:
                    dirty_files.add(item.b_path)
        except Exception:
            # Handle case where HEAD doesn't exist (initial commit)
            # In this case, all files in index are considered staged
            for item in repo.index.entries:
                dirty_files.add(item[0])

        # Filter out ignored files
        for file_path in dirty_files:
            full_path = os.path.join(repo_path, file_path)
            if not _should_ignore_file(full_path, ignore_patterns, repo_path):
                return True  # Found at least one non-ignored dirty file

        return False  # All dirty files are ignored
    except Exception as e:
        _logger.debug(f"Error checking git dirty state: {e}")
        return False


def _generate_git_model_name(path: str) -> Optional[str]:
    """
    Generate a model name based on git repository information.

    Args:
        path: Path to the git repository

    Format:
        {repo_name}#{branch}-{commit_short}[-dirty]
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
    return f"{repo_name}#{branch}-{commit_short}{dirty_suffix}"


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
            Model names use the format: {repo}#{branch}-{commit}[-dirty]. Default to ``False``.
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
