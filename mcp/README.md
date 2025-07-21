# MLflow MCP Server

A Model Context Protocol (MCP) server for MLflow trace debugging and analysis.

## Installation

### Option 1: Install from source (Recommended)

```bash
cd /path/to/mlflow/mcp
pip install .
```

### Option 2: Install from wheel

```bash
cd /path/to/mlflow/mcp
python -m build --wheel
pip install dist/mlflow_mcp-0.1.0-py3-none-any.whl
```

## Claude Code Setup

After installation, you can use the MLflow MCP server with Claude Code:

### Step 1: Install the Package

First, install the package using one of the methods above.

### Step 2: Set Environment Variables

Set your MLflow tracking URI:

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

Replace `http://localhost:5000` with your actual MLflow tracking server URL.

### Step 3: Configure Claude Code

Create a `.mcp.json` file in your project root directory:

```json
{
  "mcpServers": {
    "mlflow-tracing": {
      "command": "mlflow-mcp",
      "env": {
        "MLFLOW_TRACKING_URI": "${MLFLOW_TRACKING_URI}"
      },
      "description": "MLflow MCP server for trace debugging"
    }
  }
}
```

This file can be checked into version control for team collaboration.

### Step 4: Verify Installation

Test that the package is working:

```bash
# This should start the MCP server (press Ctrl+C to stop)
mlflow-mcp
```

### Step 5: Start Using

You can now interact with MLflow traces using natural language in Claude Code.

**Important**: When searching traces, you must specify the experiment name or ID you're working with.

Example queries:

```
"Search for traces in experiment 'my_rag_experiment' that have errors"
"Show me the trace with ID abc123"
"List all spans in trace xyz789 with pagination"
"Find all traces in experiment 'chatbot_dev' with span pattern 'timeout'"
"Get instrumentation instructions for MLflow tracing"
```

**Recommended workflow**:

1. Identify your experiment name (e.g., "my_project", "rag_evaluation")
2. Use experiment-scoped searches: "Search traces in experiment 'my_project'"
3. Then drill down into specific traces and spans

## Environment Variables

- `MLFLOW_TRACKING_URI`: MLflow tracking server URI (**Required**)
  - Must be set before starting Claude Code
  - Example: `export MLFLOW_TRACKING_URI="http://localhost:5000"`

### Databricks Configuration

If using MLflow with Databricks, configure these environment variables:

```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-databricks-token"
export MLFLOW_TRACKING_URI="databricks"
```

**Important**:

- Set `MLFLOW_TRACKING_URI=databricks` (not a URL)
- Your Databricks token must have workspace access permissions
- The host should include the full URL with protocol (https://)

## Available Tools

- `get_trace_info`: Get comprehensive trace information
- `list_trace_spans`: List spans in a trace with pagination
- `get_trace_span`: Get detailed span information with pagination
- `search_trace_spans`: Search spans using regex patterns
- `search_traces`: Search traces with MLflow filters and span patterns (**requires experiment name/ID**)
- `get_tracing_instrumentation_instructions`: Get MLflow tracing setup instructions

## Dependencies

- mlflow
- mcp>=0.1.0
- pydantic>=2.0.0
