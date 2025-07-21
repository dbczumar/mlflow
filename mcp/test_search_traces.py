#!/usr/bin/env python3
"""
Test script to reproduce the search_traces MCP call issue.

This script simulates the exact MCP call:
mlflow-tracing - search_traces (MCP)(experiment_ids: "[\"2082254368706020\"]", span_pattern: ".*carrot.*")

Usage:
    python test_search_traces.py
"""

import asyncio
import os
import sys

# Add the current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlflow_mcp_pkg.server import MLflowMCPServer
from mlflow_mcp_pkg.tools import SearchTracesParams


async def test_search_traces_call():
    """Test the exact search_traces call that's failing."""

    print("🧪 Testing search_traces MCP call")
    print("=" * 50)

    # Create server instance
    server = MLflowMCPServer()

    # Simulate the exact arguments Claude Code is sending
    # This matches: experiment_ids: "[\"2082254368706020\"]", span_keywords: ["carrot"], max_results: 1
    arguments = {
        "experiment_ids": '["2082254368706020"]',  # JSON string as sent by Claude Code
        "span_keywords": ["carrot"],
        "max_results": 1,
    }

    print("📝 Arguments being tested:")
    print(
        f"   experiment_ids: {arguments['experiment_ids']} (type: {type(arguments['experiment_ids'])})"
    )
    print(f"   span_keywords: {arguments['span_keywords']}")
    print(f"   max_results: {arguments['max_results']}")
    print()

    try:
        print("🔍 Step 1: Testing parameter validation...")

        # Test the parameter model directly first
        params = SearchTracesParams(**arguments)
        print("✅ Parameters parsed successfully:")
        print(f"   experiment_ids: {params.experiment_ids} (type: {type(params.experiment_ids)})")
        print(f"   experiment_names: {params.experiment_names}")
        print(f"   span_keywords: {params.span_keywords}")
        print(f"   max_results: {params.max_results}")
        print()

        print("🔍 Step 2: Testing MCP tool call...")

        # Test the actual tool call
        result = await server.call_tool("search_traces", arguments)
        print("✅ Tool call succeeded!")
        print(f"📊 Result type: {type(result)}")

        if isinstance(result, dict):
            traces = result.get("traces", [])
            next_token = result.get("next_page_token")
            print(f"📊 Number of traces found: {len(traces)}")
            print(f"📋 Next page token: {next_token}")
            if traces:
                print(f"📋 First trace keys: {list(traces[0].keys())}")
        else:
            print(f"📋 Result: {result}")

    except Exception as e:
        print("❌ Error occurred:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print()

        # Print more detailed error info
        import traceback

        print("🔧 Full traceback:")
        traceback.print_exc()

        return False

    return True


def test_parameter_parsing():
    """Test just the parameter parsing logic."""

    print("\n" + "=" * 50)
    print("🧪 Testing Parameter Parsing Only")
    print("=" * 50)

    test_cases = [
        {
            "name": "Claude Code format (JSON string)",
            "experiment_ids": '["2082254368706020"]',
            "span_keywords": ["carrot"],
        },
        {
            "name": "Direct list format",
            "experiment_ids": ["2082254368706020"],
            "span_keywords": ["carrot"],
        },
        {
            "name": "Single string format",
            "experiment_ids": "2082254368706020",
            "span_keywords": ["carrot"],
        },
        {
            "name": "Multiple experiments (JSON string)",
            "experiment_ids": '["123", "456"]',
            "span_keywords": ["test"],
        },
        {
            "name": "Experiment names instead",
            "experiment_names": '["my_experiment"]',
            "span_keywords": ["error"],
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {case['name']}")
        try:
            params = SearchTracesParams(**case)
            print("   ✅ Success!")
            print(f"   📋 experiment_ids: {params.experiment_ids}")
            print(f"   📋 experiment_names: {params.experiment_names}")
            print(f"   📋 span_keywords: {params.span_keywords}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def test_environment_setup():
    """Test that the environment is set up correctly."""

    print("🔧 Environment Check")
    print("=" * 50)

    # Check MLFLOW_TRACKING_URI
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    print(f"📍 MLFLOW_TRACKING_URI: {tracking_uri}")

    if not tracking_uri:
        print("⚠️  WARNING: MLFLOW_TRACKING_URI not set. You may need to run 'source source_env.sh'")

    # Test MLflow import
    try:
        import mlflow

        print(f"✅ MLflow imported successfully (version: {mlflow.__version__})")
    except ImportError as e:
        print(f"❌ MLflow import failed: {e}")

    # Test MCP import
    try:
        from mcp.server import Server

        print("✅ MCP server imported successfully")
    except ImportError as e:
        print(f"❌ MCP import failed: {e}")


if __name__ == "__main__":
    print("🚀 MLflow MCP Search Traces Test Script")
    print("🎯 Reproducing: experiment_ids='[\"2082254368706020\"]', span_keywords=['carrot']")
    print()

    # Test environment first
    test_environment_setup()

    # Test parameter parsing
    test_parameter_parsing()

    # Test the full MCP call
    print("\n" + "=" * 50)
    print("🧪 Testing Full MCP Call")
    print("=" * 50)

    success = asyncio.run(test_search_traces_call())

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The MCP call should work.")
    else:
        print("💥 Test failed. Check the error details above.")
    print("=" * 50)
