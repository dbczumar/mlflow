#!/usr/bin/env python3
"""
Test MCP server functionality.

These tests ensure the MCP server can handle tool calls correctly.
"""

import asyncio

import pytest

from mlflow_mcp_pkg.server import MLflowMCPServer
from mlflow_mcp_pkg.tools import TOOL_DEFINITIONS


class TestMLflowMCPServer:
    """Test MCP server functionality."""

    def setup_method(self):
        """Set up test server."""
        self.server = MLflowMCPServer()

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test that list_tools returns all expected tools."""
        tools = await self.server.list_tools()

        # Should return all tools defined in TOOL_DEFINITIONS
        assert len(tools) == len(TOOL_DEFINITIONS)

        tool_names = [tool.name for tool in tools]
        expected_names = list(TOOL_DEFINITIONS.keys())

        assert set(tool_names) == set(expected_names)

        # Each tool should have required fields
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "inputSchema")
            assert tool.inputSchema is not None

    @pytest.mark.asyncio
    async def test_get_tracing_instructions(self):
        """Test get_tracing_instrumentation_instructions tool."""
        result = await self.server.call_tool("get_tracing_instrumentation_instructions")

        assert "instructions" in result
        assert isinstance(result["instructions"], str)
        assert "MLflow Tracing" in result["instructions"]
        assert "@mlflow.trace" in result["instructions"]

    @pytest.mark.asyncio
    async def test_search_traces_validation_error(self):
        """Test that search_traces raises validation error with no experiments."""
        with pytest.raises(Exception):  # Should be ValueError but wrapped by MCP
            await self.server.call_tool(
                "search_traces",
                {
                    "span_pattern": "test"
                    # Missing experiment_ids and experiment_names
                },
            )

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self):
        """Test that unknown tool name raises error."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await self.server.call_tool("nonexistent_tool")

    def test_server_initialization(self):
        """Test that server initializes correctly."""
        server = MLflowMCPServer()

        # Should have a server instance
        assert hasattr(server, "server")
        assert server.server is not None

        # Should initially have no MLflow client
        assert server.mlflow_client is None


def run_manual_tests():
    """Run tests manually if pytest/asyncio not available."""
    print("🧪 Running MCP server tests...")

    # Test basic initialization
    try:
        server = MLflowMCPServer()
        print("✅ Server initialization")
    except Exception as e:
        print(f"❌ Server initialization: {e}")
        return

    # Test list_tools (sync version)
    try:

        async def test_list_tools():
            tools = await server.list_tools()
            assert len(tools) > 0
            assert all(hasattr(tool, "name") for tool in tools)
            return len(tools)

        tool_count = asyncio.run(test_list_tools())
        print(f"✅ list_tools ({tool_count} tools)")
    except Exception as e:
        print(f"❌ list_tools: {e}")

    # Test get_instructions
    try:

        async def test_instructions():
            result = await server.call_tool("get_tracing_instrumentation_instructions")
            assert "instructions" in result
            assert "MLflow" in result["instructions"]

        asyncio.run(test_instructions())
        print("✅ get_tracing_instrumentation_instructions")
    except Exception as e:
        print(f"❌ get_tracing_instrumentation_instructions: {e}")

    print("🎉 Manual MCP server tests completed!")


if __name__ == "__main__":
    run_manual_tests()
