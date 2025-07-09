import os
import tempfile

import pytest
from git import Repo

import mlflow
import mlflow.genai


@pytest.fixture
def git_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a git repo
        repo = Repo.init(tmpdir)

        # Create and commit an initial file
        test_file = os.path.join(tmpdir, "test.py")
        with open(test_file, "w") as f:
            f.write("# Initial content\ndef hello():\n    return 'world'\n")

        repo.index.add([test_file])
        repo.index.commit("Initial commit")

        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            yield {"repo": repo, "tmpdir": tmpdir, "test_file": test_file}
        finally:
            os.chdir(old_cwd)


class TestGenAIAutolog:
    def test_autolog_without_git_versioning(self):
        """Test basic autolog functionality without git versioning."""
        mlflow.genai.autolog(enable_git_versioning=False)

        # Should not set any active model
        assert mlflow.get_active_model_id() is None

    def test_autolog_with_git_versioning_sets_active_model(self, git_repo):
        """Test that autolog with git versioning sets an active model."""
        with mlflow.start_run():
            mlflow.genai.autolog(enable_git_versioning=True, silent=True)

            # Should have set an active model
            model_id = mlflow.get_active_model_id()
            assert model_id is not None

            # Model name should contain git info
            model = mlflow.get_logged_model(model_id)
            assert "main" in model.name  # branch name
            assert ":" in model.name  # colon-separated format
            parts = model.name.split(":")
            assert len(parts) == 2  # repo:branch-commit format

    def test_autolog_model_name_changes_with_dirty_state(self, git_repo):
        """Test that model name changes when git working directory becomes dirty."""
        test_file = git_repo["test_file"]

        # First autolog call (clean state)
        with mlflow.start_run():
            mlflow.genai.autolog(enable_git_versioning=True, silent=True)
            first_model_id = mlflow.get_active_model_id()
            first_model = mlflow.get_logged_model(first_model_id)
            first_name = first_model.name

        # Modify file content (makes repo dirty)
        with open(test_file, "w") as f:
            f.write("# Modified content\ndef hello():\n    return 'modified world'\n")

        # Second autolog call (dirty state)
        with mlflow.start_run():
            mlflow.genai.autolog(enable_git_versioning=True, silent=True)
            second_model_id = mlflow.get_active_model_id()
            second_model = mlflow.get_logged_model(second_model_id)
            second_name = second_model.name

        # Model names should be different due to dirty state
        assert first_name != second_name
        assert first_model_id != second_model_id

        # First should be clean, second should be dirty
        assert not first_name.endswith("-dirty")
        assert second_name.endswith("-dirty")

        # Both should contain git info in colon-separated format
        assert "main" in first_name
        assert "main" in second_name
        assert ":" in first_name
        assert ":" in second_name

    def test_traces_linked_to_git_model(self, git_repo):
        """Test that active model is set correctly for trace linking."""
        # Setup autolog with git versioning
        with mlflow.start_run():
            mlflow.genai.autolog(enable_git_versioning=True, silent=True)
            active_model_id = mlflow.get_active_model_id()

            # Verify active model is set and has git-based name
            assert active_model_id is not None

            model = mlflow.get_logged_model(active_model_id)
            assert "main" in model.name  # Contains branch name
            assert ":" in model.name  # Colon-separated format

            # Create a simple traced function to verify linking would work
            @mlflow.trace
            def test_function():
                return "test"

            # Call the function - traces will be automatically linked to active model
            result = test_function()
            assert result == "test"

            # Verify the active model is still set after tracing
            assert mlflow.get_active_model_id() == active_model_id

    def test_autolog_handles_non_git_directory(self):
        """Test autolog behavior when not in a git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                # Should not raise an error
                mlflow.genai.autolog(enable_git_versioning=True, silent=True)

                # Should not set an active model
                assert mlflow.get_active_model_id() is None
            finally:
                os.chdir(old_cwd)

    def test_autolog_disable_parameter(self, git_repo):
        """Test that disable=True prevents model setting."""
        with mlflow.start_run():
            mlflow.genai.autolog(disable=True, enable_git_versioning=True, silent=True)

            # Should not set an active model when disabled
            assert mlflow.get_active_model_id() is None

    def test_multiple_autolog_calls_consistent_naming(self, git_repo):
        """Test that multiple autolog calls with same git state produce same model name."""
        with mlflow.start_run():
            # First call
            mlflow.genai.autolog(enable_git_versioning=True, silent=True)
            first_model_id = mlflow.get_active_model_id()
            first_model = mlflow.get_logged_model(first_model_id)
            first_name = first_model.name

            # Clear active model
            mlflow.clear_active_model()

            # Second call with same git state
            mlflow.genai.autolog(enable_git_versioning=True, silent=True)
            second_model_id = mlflow.get_active_model_id()
            second_model = mlflow.get_logged_model(second_model_id)
            second_name = second_model.name

            # Should produce the same model name (and reuse existing model)
            assert first_name == second_name

    def test_commit_strategy_with_commit_changes(self, git_repo):
        """Test that model name changes when committing changes."""
        test_file = git_repo["test_file"]
        repo = git_repo["repo"]

        # Initial state (clean)
        with mlflow.start_run():
            mlflow.genai.autolog(enable_git_versioning=True, silent=True)
            initial_model_id = mlflow.get_active_model_id()
            initial_model = mlflow.get_logged_model(initial_model_id)
            initial_name = initial_model.name

        # Modify file and commit
        with open(test_file, "w") as f:
            f.write("# Committed content\ndef hello():\n    return 'committed'\n")

        repo.index.add([test_file])
        repo.index.commit("Update file content")

        with mlflow.start_run():
            mlflow.genai.autolog(enable_git_versioning=True, silent=True)
            committed_model_id = mlflow.get_active_model_id()
            committed_model = mlflow.get_logged_model(committed_model_id)
            committed_name = committed_model.name

        # Should be different due to new commit
        assert initial_name != committed_name
        assert not committed_name.endswith("-dirty")  # Clean state after commit

        # Should have 2 parts: repo:branch-commit
        assert ":" in committed_name
        parts = committed_name.split(":")
        assert len(parts) == 2
