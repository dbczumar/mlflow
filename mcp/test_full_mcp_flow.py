#!/usr/bin/env python3
"""
Test the full MCP flow with schema generation and validation.
"""

import asyncio

from mlflow_mcp_pkg.server import MLflowMCPServer


async def test_mcp_flow():
    """Test the complete MCP flow."""

    print("🔍 Testing full MCP flow...")

    server = MLflowMCPServer()

    # Test list_tools to see the schema
    tools = await server.list_tools()
    search_traces_tool = None
    for tool in tools:
        if tool.name == "search_traces":
            search_traces_tool = tool
            break

    if search_traces_tool:
        print("📝 Found search_traces tool")
        page_size_schema = search_traces_tool.inputSchema.get("properties", {}).get("page_size", {})
        print(f"📋 page_size schema in tool: {page_size_schema}")
        print()

    # Test the actual call that was failing
    arguments = {
        "experiment_ids": '["2082254368706020"]',
        "span_keywords": '["carrots"]',
        "page_size": "1",
    }

    print(f"📝 Testing arguments: {arguments}")

    try:
        # This should work now since we fixed the schema
        result = await server.call_tool("search_traces", arguments)
        print("✅ MCP call succeeded!")
        print(f"📊 Result type: {type(result)}")
        if isinstance(result, list):
            print(f"📊 Number of traces: {len(result)}")
    except Exception as e:
        print(f"❌ MCP call failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_mcp_flow())
