#!/usr/bin/env python3
"""
Debug JSON schema generation for MCP tools.
"""

import json

from mlflow_mcp_pkg.server import MLflowMCPServer
from mlflow_mcp_pkg.tools import SearchTracesParams


def test_schema_generation():
    """Test the JSON schema generation and MCP compatibility fixes."""

    print("🔍 Testing JSON schema generation...")

    # Get original schema
    original_schema = SearchTracesParams.model_json_schema()
    print("📝 Original schema for page_size:")
    page_size_schema = original_schema.get("properties", {}).get("page_size", {})
    print(json.dumps(page_size_schema, indent=2))
    print()

    # Get schema after MCP compatibility fixes
    server = MLflowMCPServer()
    fixed_schema = SearchTracesParams.model_json_schema()
    server._fix_schema_for_mcp_compatibility(fixed_schema)

    print("📝 Fixed schema for page_size:")
    fixed_page_size_schema = fixed_schema.get("properties", {}).get("page_size", {})
    print(json.dumps(fixed_page_size_schema, indent=2))
    print()

    # Test max_results too
    print("📝 Fixed schema for max_results:")
    fixed_max_results_schema = fixed_schema.get("properties", {}).get("max_results", {})
    print(json.dumps(fixed_max_results_schema, indent=2))


if __name__ == "__main__":
    test_schema_generation()
