#!/usr/bin/env python3
"""
Debug MCP validation issues.
"""

from mlflow_mcp_pkg.tools import SearchTracesParams


def test_validation():
    """Test the exact arguments causing validation errors."""

    print("🔍 Testing MCP validation...")

    # Test the exact arguments from the error
    arguments = {
        "experiment_ids": '["2082254368706020"]',  # JSON string
        "span_keywords": '["carrot"]',  # JSON string
        "page_size": "1",  # String number
    }

    print("📝 Input arguments:")
    for key, value in arguments.items():
        print(f"   {key}: {repr(value)} (type: {type(value)})")
    print()

    try:
        params = SearchTracesParams(**arguments)
        print("✅ Validation succeeded!")
        print("📋 Parsed values:")
        print(f"   experiment_ids: {params.experiment_ids} (type: {type(params.experiment_ids)})")
        print(f"   span_keywords: {params.span_keywords} (type: {type(params.span_keywords)})")
        print(f"   page_size: {params.page_size} (type: {type(params.page_size)})")

    except Exception as e:
        print("❌ Validation failed:")
        print(f"   Error: {e}")
        print(f"   Error type: {type(e)}")

        # Try to get more details
        import traceback

        print("\n🔧 Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    test_validation()
