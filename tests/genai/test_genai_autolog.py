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
            assert len(model.name.split("-")) >= 3  # repo-branch-commit format

    def test_autolog_model_name_changes_with_file_content(self, git_repo):
        """Test that model name changes when tracked file content changes."""
        test_file = git_repo["test_file"]

        # First autolog call
        with mlflow.start_run():
            mlflow.genai.autolog(enable_git_versioning=True, silent=True)
            first_model_id = mlflow.get_active_model_id()
            first_model = mlflow.get_logged_model(first_model_id)
            first_name = first_model.name

        # Modify file content
        with open(test_file, "w") as f:
            f.write("# Modified content\ndef hello():\n    return 'modified world'\n")

        # Second autolog call
        with mlflow.start_run():
            mlflow.genai.autolog(enable_git_versioning=True, silent=True)
            second_model_id = mlflow.get_active_model_id()
            second_model = mlflow.get_logged_model(second_model_id)
            second_name = second_model.name

        # Model names should be different
        assert first_name != second_name
        assert first_model_id != second_model_id

        # Both should contain git info but different hashes
        assert "main" in first_name
        assert "main" in second_name
        first_hash = first_name.split("-")[-1]
        second_hash = second_name.split("-")[-1]
        assert first_hash != second_hash

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

    def test_mlflowignore_functionality(self, git_repo):
        """Test that .mlflowignore patterns are respected when computing hashes."""
        test_file = git_repo["test_file"]
        tmpdir = git_repo["tmpdir"]

        # Create additional files to track
        log_file = os.path.join(tmpdir, "app.log")
        temp_dir = os.path.join(tmpdir, "temp")
        os.makedirs(temp_dir)
        temp_file = os.path.join(temp_dir, "cache.tmp")
        pycache_dir = os.path.join(tmpdir, "src", "__pycache__")
        os.makedirs(pycache_dir)
        pycache_file = os.path.join(pycache_dir, "module.pyc")

        with open(log_file, "w") as f:
            f.write("log content")
        with open(temp_file, "w") as f:
            f.write("temp content")
        with open(pycache_file, "w") as f:
            f.write("cache content")

        # Add all files to git
        repo = git_repo["repo"]
        repo.index.add([log_file, temp_file, pycache_file])
        repo.index.commit("Add files to ignore")

        # Get initial model name (all files included)
        with mlflow.start_run():
            mlflow.genai.autolog(enable_git_versioning=True, silent=True)
            initial_model_id = mlflow.get_active_model_id()
            initial_model = mlflow.get_logged_model(initial_model_id)
            initial_name = initial_model.name

        # Create .mlflowignore file
        mlflowignore_path = os.path.join(tmpdir, ".mlflowignore")
        with open(mlflowignore_path, "w") as f:
            f.write("*.log\n")
            f.write("temp/\n")
            f.write("**/__pycache__/**\n")
            f.write("# Comment line\n")
            f.write("\n")  # Empty line

        # Get model name after adding .mlflowignore (should be different)
        with mlflow.start_run():
            mlflow.genai.autolog(enable_git_versioning=True, silent=True)
            ignored_model_id = mlflow.get_active_model_id()
            ignored_model = mlflow.get_logged_model(ignored_model_id)
            ignored_name = ignored_model.name

        # Model names should be different (ignored files don't affect hash)
        assert initial_name != ignored_name

        # Modify an ignored file - should not change model name
        with open(log_file, "w") as f:
            f.write("modified log content")

        with mlflow.start_run():
            mlflow.genai.autolog(enable_git_versioning=True, silent=True)
            after_ignored_change_model_id = mlflow.get_active_model_id()
            after_ignored_change_model = mlflow.get_logged_model(after_ignored_change_model_id)
            after_ignored_change_name = after_ignored_change_model.name

        # Should be same as ignored_name (ignored file changes don't matter)
        assert ignored_name == after_ignored_change_name

        # Modify a non-ignored file - should change model name
        with open(test_file, "w") as f:
            f.write("# Final modified content\ndef hello():\n    return 'final'\n")

        with mlflow.start_run():
            mlflow.genai.autolog(enable_git_versioning=True, silent=True)
            final_model_id = mlflow.get_active_model_id()
            final_model = mlflow.get_logged_model(final_model_id)
            final_name = final_model.name

        # Should be different from previous names
        assert final_name != ignored_name
        assert final_name != after_ignored_change_name

    def test_mlflowignore_patterns(self, git_repo):
        """Test specific .mlflowignore pattern matching."""
        tmpdir = git_repo["tmpdir"]

        # Import the internal functions for testing
        from mlflow.genai.autolog import _load_mlflowignore_patterns, _should_ignore_file

        # Create .mlflowignore with various patterns
        mlflowignore_path = os.path.join(tmpdir, ".mlflowignore")
        with open(mlflowignore_path, "w") as f:
            f.write("*.log\n")
            f.write("build/\n")
            f.write("**/__pycache__/**\n")
            f.write("*.tmp\n")

        patterns = _load_mlflowignore_patterns(tmpdir)
        assert "*.log" in patterns
        assert "build/" in patterns
        assert "**/__pycache__/**" in patterns
        assert "*.tmp" in patterns

        # Test pattern matching
        test_cases = [
            ("app.log", True),  # matches *.log
            ("debug.log", True),  # matches *.log
            ("build/output.bin", True),  # matches build/
            ("build/subdir/file.txt", True),  # matches build/
            ("src/__pycache__/module.pyc", True),  # matches **/__pycache__/**
            ("deep/src/__pycache__/other.pyc", True),  # matches **/__pycache__/**
            ("cache.tmp", True),  # matches *.tmp
            ("normal.py", False),  # doesn't match any pattern
            ("logs/app.txt", False),  # doesn't match *.log
        ]

        for file_path, should_ignore in test_cases:
            full_path = os.path.join(tmpdir, file_path)
            result = _should_ignore_file(full_path, patterns, tmpdir)
            assert result == should_ignore, f"Pattern matching failed for {file_path}"
