#!/usr/bin/env python3
"""
Test parameter validation - critical for MCP compatibility.

These tests ensure that MCP clients can send parameters in various formats
and they get parsed correctly.
"""

import pytest

from mlflow_mcp_pkg.tools import SearchTracesParams


class TestSearchTracesParams:
    """Test parameter validation for search_traces tool."""

    def test_experiment_ids_json_string(self):
        """Test that JSON string experiment_ids are parsed correctly (MCP format)."""
        # This is the critical case - Claude Code sends JSON strings
        params = SearchTracesParams(experiment_ids='["123", "456"]')
        assert params.experiment_ids == ["123", "456"]

    def test_experiment_ids_single_json_string(self):
        """Test single experiment ID as JSON string."""
        params = SearchTracesParams(experiment_ids='["2082254368706020"]')
        assert params.experiment_ids == ["2082254368706020"]

    def test_experiment_ids_list(self):
        """Test normal list input."""
        params = SearchTracesParams(experiment_ids=["123", "456"])
        assert params.experiment_ids == ["123", "456"]

    def test_experiment_ids_single_string(self):
        """Test single string gets converted to list."""
        params = SearchTracesParams(experiment_ids="123")
        assert params.experiment_ids == ["123"]

    def test_experiment_names_json_string(self):
        """Test experiment names as JSON string."""
        params = SearchTracesParams(experiment_names='["my_exp", "test_exp"]')
        assert params.experiment_names == ["my_exp", "test_exp"]

    def test_experiment_names_single_string(self):
        """Test single experiment name."""
        params = SearchTracesParams(experiment_names="my_experiment")
        assert params.experiment_names == ["my_experiment"]

    def test_order_by_json_string(self):
        """Test order_by as JSON string."""
        params = SearchTracesParams(
            experiment_ids=["123"], order_by='["timestamp_ms DESC", "status ASC"]'
        )
        assert params.order_by == ["timestamp_ms DESC", "status ASC"]

    def test_order_by_single_string(self):
        """Test single order_by string."""
        params = SearchTracesParams(experiment_ids=["123"], order_by="timestamp_ms DESC")
        assert params.order_by == ["timestamp_ms DESC"]

    def test_no_experiments_raises_error(self):
        """Test that missing both experiment_ids and experiment_names raises error."""
        with pytest.raises(
            ValueError, match="Either experiment_ids or experiment_names must be provided"
        ):
            SearchTracesParams(span_pattern="test")

    def test_both_experiments_allowed(self):
        """Test that providing both experiment_ids and experiment_names is allowed."""
        params = SearchTracesParams(experiment_ids=["123"], experiment_names=["my_exp"])
        assert params.experiment_ids == ["123"]
        assert params.experiment_names == ["my_exp"]

    def test_optional_parameters(self):
        """Test that all optional parameters work."""
        params = SearchTracesParams(
            experiment_ids=["123"],
            filter_string="status = 'ERROR'",
            span_pattern=".*timeout.*",
            max_results=50,
            order_by=["timestamp_ms DESC"],
        )
        assert params.experiment_ids == ["123"]
        assert params.filter_string == "status = 'ERROR'"
        assert params.span_pattern == ".*timeout.*"
        assert params.max_results == 50
        assert params.order_by == ["timestamp_ms DESC"]

    def test_malformed_json_fallback(self):
        """Test that malformed JSON falls back to single item."""
        # This should not crash, just treat as single item
        params = SearchTracesParams(experiment_ids="not_valid_json")
        assert params.experiment_ids == ["not_valid_json"]


if __name__ == "__main__":
    # Run tests manually if pytest not available
    test_class = TestSearchTracesParams()

    print("🧪 Running parameter validation tests...")

    test_methods = [method for method in dir(test_class) if method.startswith("test_")]

    passed = 0
    failed = 0

    for method_name in test_methods:
        try:
            method = getattr(test_class, method_name)
            method()
            print(f"✅ {method_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {method_name}: {e}")
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All parameter validation tests passed!")
    else:
        print("💥 Some tests failed!")
