"""MLflow MCP Server implementation."""

import logging
import os
from typing import Any, Dict, List, Optional

from mcp import ServerCapabilities, ToolsCapability
from mcp.server import InitializationOptions, Server
from mcp.types import Tool
from pydantic import ValidationError

from .mlflow_client import MlflowMCPClient
from .tools import TOOL_DEFINITIONS, get_instrumentation_instructions

logger = logging.getLogger(__name__)


class MLflowMCPServer:
    """MCP server for MLflow trace debugging."""

    def __init__(self):
        """Initialize the MLflow MCP server."""
        self.server = Server("mlflow-tracing")
        self.mlflow_client = None

        # Register handlers using decorators
        @self.server.list_tools()
        async def handle_list_tools():
            return await self.list_tools()

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]] = None):
            return await self.call_tool(name, arguments)

        logger.info("MLflow MCP server initialized")

    async def list_tools(self) -> List[Tool]:
        """List available tools."""
        tools = []

        for tool_name, tool_info in TOOL_DEFINITIONS.items():
            # Add input schema - required for all tools
            if tool_info["params_model"]:
                input_schema = tool_info["params_model"].model_json_schema()
                # Fix schema to allow strings for integer fields (MCP compatibility)
                self._fix_schema_for_mcp_compatibility(input_schema)
            else:
                # Empty schema for tools with no parameters
                input_schema = {"type": "object", "properties": {}}

            tool = Tool(
                name=tool_name, description=tool_info["description"], inputSchema=input_schema
            )

            tools.append(tool)

        logger.info(f"Listed {len(tools)} tools")
        return tools

    def _fix_schema_for_mcp_compatibility(self, schema: dict) -> None:
        """Fix JSON schema to allow strings for integer fields for MCP compatibility.

        MCP clients often send integers as strings, but the generated JSON schema
        only allows the declared types. This method modifies the schema to allow
        both integers and strings for integer fields, since our validators handle
        the conversion.
        """
        if "properties" not in schema:
            return

        integer_fields = ["max_results", "max_content_length"]

        for field_name in integer_fields:
            if field_name in schema["properties"]:
                field_schema = schema["properties"][field_name]

                # If it's an integer field, allow both integer and string
                if field_schema.get("type") == "integer":
                    field_schema["type"] = ["integer", "string"]
                # Handle anyOf patterns for Optional fields
                elif "anyOf" in field_schema:
                    for item in field_schema["anyOf"]:
                        if item.get("type") == "integer":
                            item["type"] = ["integer", "string"]

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a tool."""
        logger.info(f"Calling tool: {name} with arguments: {arguments}")

        # Initialize MLflow client if needed
        if self.mlflow_client is None:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            self.mlflow_client = MlflowMCPClient(tracking_uri=tracking_uri)
            logger.info(f"Initialized MLflow client with tracking URI: {tracking_uri}")

        # Handle tool calls
        try:
            if name == "get_trace_info":
                params = TOOL_DEFINITIONS[name]["params_model"](**arguments)
                return self.mlflow_client.get_trace_info(trace_id=params.trace_id)

            elif name == "list_trace_spans":
                params = TOOL_DEFINITIONS[name]["params_model"](**arguments)
                json_result, next_token = self.mlflow_client.list_trace_spans(
                    trace_id=params.trace_id,
                    max_content_length=params.max_content_length,
                    page_token=params.page_token,
                )
                return {
                    "spans": json_result,
                    "next_page_token": next_token.to_json() if next_token else None,
                }

            elif name == "get_trace_span":
                params = TOOL_DEFINITIONS[name]["params_model"](**arguments)
                json_result, next_token = self.mlflow_client.get_trace_span(
                    trace_id=params.trace_id,
                    span_id=params.span_id,
                    max_content_length=params.max_content_length,
                    page_token=params.page_token,
                )
                return {
                    "span_content": json_result,
                    "next_page_token": next_token.to_json() if next_token else None,
                }

            elif name == "search_trace_spans":
                params = TOOL_DEFINITIONS[name]["params_model"](**arguments)
                json_result, next_token = self.mlflow_client.search_trace_spans(
                    trace_id=params.trace_id,
                    keywords=params.keywords,
                    max_content_length=params.max_content_length,
                    page_token=params.page_token,
                )
                return {
                    "spans": json_result,
                    "next_page_token": next_token.to_json() if next_token else None,
                }

            elif name == "search_traces":
                params = TOOL_DEFINITIONS[name]["params_model"](**arguments)
                traces, next_token = self.mlflow_client.search_traces(
                    filter_string=params.filter_string,
                    span_keywords=params.span_keywords,
                    experiment_ids=params.experiment_ids,
                    experiment_names=params.experiment_names,
                    max_results=params.max_results,
                    order_by=params.order_by,
                    page_token=params.page_token,
                )
                return {
                    "traces": traces,
                    "next_page_token": next_token.to_json() if next_token else None,
                }

            elif name == "get_tracing_instrumentation_instructions":
                return {"instructions": get_instrumentation_instructions()}

            else:
                raise ValueError(f"Unknown tool: {name}")

        except ValidationError as e:
            logger.error(f"Validation error for tool {name}: {e}")
            raise ValueError(f"Invalid parameters for {name}: {e}")
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            raise

    async def run(self):
        """Run the MCP server."""
        logger.info("Starting MLflow MCP server...")

        # Run the server using stdio transport
        from mcp.server.stdio import stdio_server

        # Create server capabilities
        capabilities = ServerCapabilities(tools=ToolsCapability(listChanged=False))

        # Create initialization options
        init_options = InitializationOptions(
            server_name="mlflow-tracing", server_version="0.1.0", capabilities=capabilities
        )

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, init_options)
