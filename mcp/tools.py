"""Tool definitions for MLflow MCP server."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class GetTraceInfoParams(BaseModel):
    """Parameters for get_trace_info tool."""

    trace_id: str = Field(description="The unique identifier of the trace to retrieve")
    include_spans: bool = Field(
        default=False, description="Whether to include basic span information in the response"
    )


class ListTraceSpansParams(BaseModel):
    """Parameters for list_trace_spans tool."""

    trace_id: str = Field(description="The unique identifier of the trace whose spans to list")
    max_depth: Optional[int] = Field(
        default=None,
        description="Maximum depth of spans to include (useful for limiting nested spans)",
    )


class GetTraceSpanParams(BaseModel):
    """Parameters for get_trace_span tool."""

    trace_id: str = Field(description="The unique identifier of the trace containing the span")
    span_id: str = Field(description="The unique identifier of the span to retrieve")
    max_content_length: int = Field(
        default=100000,
        description="Maximum content length to return. If content exceeds this, pagination will be used",
    )
    page_token: Optional[str] = Field(
        default=None,
        description="Page token for retrieving next page of content (JSON string with offset)",
    )


class SearchTraceSpansParams(BaseModel):
    """Parameters for search_trace_spans tool."""

    trace_id: str = Field(description="The unique identifier of the trace to search within")
    query: Optional[str] = Field(
        default=None, description="Regex pattern to search for in span JSON representation"
    )


class SearchTracesParams(BaseModel):
    """Parameters for search_traces tool."""

    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description="""Filters to apply to the search. Supported filters:
        - experiment_id: Filter by experiment ID
        - status: Filter by trace status
        - tags: Dictionary of tags to match
        - start_time_after: ISO timestamp for minimum start time
        - start_time_before: ISO timestamp for maximum start time""",
    )
    limit: int = Field(default=100, description="Maximum number of traces to return")
    order_by: Optional[str] = Field(
        default=None,
        description="Field to order results by (e.g., 'timestamp_ms', 'execution_time_ms')",
    )


TOOL_DEFINITIONS = {
    "get_trace_info": {
        "description": """Retrieve comprehensive information about a specific trace.

Returns trace metadata including:
- Trace and run identifiers
- Experiment information
- Timing information (start time, execution time)
- Status (OK, ERROR, etc.)
- Request metadata and tags
- Basic span count and hierarchy info (if requested)

Use this to get an overview of a trace before diving into specific spans.

Example use cases:
- Check if a trace completed successfully
- Get the experiment context of a trace
- Understand the overall execution time
- Access trace-level tags and metadata""",
        "params_model": GetTraceInfoParams,
    },
    "list_trace_spans": {
        "description": """List all spans in a trace with basic information.

Returns a list of spans with:
- Span identifiers and names
- Span types (LLM, RETRIEVER, CHAIN, etc.)
- Parent-child relationships
- Timing information and duration
- Status codes

Use this to understand the structure of a trace and identify spans of interest.

Example use cases:
- View the execution flow of an agent
- Find long-running operations
- Identify failed spans
- Understand the span hierarchy""",
        "params_model": ListTraceSpansParams,
    },
    "get_trace_span": {
        "description": """Get detailed information about a specific span.

Returns comprehensive span data including:
- All basic span information
- Full status with description
- Input and output data
- Span attributes
- Events that occurred during the span

Supports pagination for large content:
- max_content_length: Limit response size (default 100,000 characters)
- page_token: For retrieving subsequent pages
- Returns next_page_token if more content is available

Use this to debug specific operations within a trace.

Example use cases:
- Examine LLM prompts and responses
- Check retriever queries and results
- Debug failed operations
- Analyze span attributes and events""",
        "params_model": GetTraceSpanParams,
    },
    "search_trace_spans": {
        "description": """Search for spans within a trace using regex pattern matching.

Supports:
- Regex search across the entire span JSON representation
- Pattern matching on span names, inputs, outputs, and attributes

Returns matching spans with full details.

Example use cases:
- Find all LLM calls in a trace: query="LLM"
- Search for specific error messages: query="error.*timeout"
- Locate spans processing certain data: query="user_id.*12345"
- Find slow operations: query="duration_ms.*[5-9][0-9]{3}" """,
        "params_model": SearchTraceSpansParams,
    },
    "search_traces": {
        "description": """Search across multiple traces in the MLflow tracking server.

Supports filtering by:
- Experiment ID
- Trace status
- Tags
- Time ranges

Note: This is a stubbed implementation as MLflow doesn't yet have native trace search.
In production, this would query the MLflow backend.

Example use cases:
- Find all failed traces in an experiment
- Search traces by custom tags
- Get recent traces within a time window
- Find traces matching specific criteria""",
        "params_model": SearchTracesParams,
    },
    "get_tracing_instrumentation_instructions": {
        "description": """Get comprehensive instructions for instrumenting code with MLflow Tracing.

Returns detailed documentation covering:
- Basic instrumentation with @mlflow.trace
- Manual span creation
- Context propagation
- Best practices for different scenarios
- Common patterns and examples

Use this when you need to understand how to add tracing to code.""",
        "params_model": None,  # No parameters
    },
}


def get_instrumentation_instructions() -> str:
    """Return comprehensive MLflow tracing instrumentation instructions.

    Provides detailed documentation and examples for instrumenting code with
    MLflow Tracing. This includes decorator usage, manual span creation,
    context propagation, and best practices for different scenarios.

    The instructions cover:
    - Basic @mlflow.trace decorator usage
    - Custom span names and types
    - Manual span creation with context managers
    - Nested span hierarchies
    - LLM and retriever instrumentation patterns
    - Error handling and async support
    - Batch operations and distributed tracing
    - Configuration and integration with MLflow tracking

    Returns:
        str: A comprehensive markdown-formatted guide containing:
            - Overview of MLflow Tracing concepts
            - Step-by-step instrumentation examples
            - Advanced patterns for complex scenarios
            - Best practices and performance tips
            - Common patterns for RAG and agent applications
            - Debugging tips and troubleshooting guidance
    """
    return """# MLflow Tracing Instrumentation Guide

## Overview
MLflow Tracing allows you to instrument your code to capture detailed execution traces, particularly useful for debugging LLM applications and complex workflows.

## Basic Instrumentation

### 1. Automatic Tracing with @mlflow.trace

The simplest way to add tracing is using the @mlflow.trace decorator:

```python
import mlflow

@mlflow.trace
def my_function(param1, param2):
    # Your code here
    result = process_data(param1, param2)
    return result
```

The decorator automatically:
- Creates a span for the function
- Captures inputs and outputs
- Records execution time
- Handles errors gracefully

### 2. Tracing with Custom Span Names

```python
@mlflow.trace(name="custom_operation")
def my_function(data):
    return transform(data)

# Or dynamically:
@mlflow.trace(name=lambda args: f"process_{args[0].type}")
def process_item(item):
    return item.process()
```

### 3. Manual Span Creation

For fine-grained control, create spans manually:

```python
import mlflow
from mlflow.tracing import SpanType, SpanStatusCode

# Using context manager
with mlflow.start_span(name="retrieval_operation", span_type=SpanType.RETRIEVER) as span:
    # Set attributes
    span.set_attributes({
        "retriever.name": "vector_store",
        "retriever.k": 10
    })
    
    # Your retrieval code
    results = retrieve_documents(query)
    
    # Set outputs
    span.set_outputs({"documents": results})
    
    # Set status
    span.set_status(SpanStatusCode.OK)
```

### 4. Nested Spans

MLflow automatically handles span hierarchy:

```python
@mlflow.trace(name="parent_operation")
def parent_function():
    # This creates a parent span
    data = load_data()
    
    # This creates a child span
    processed = process_data(data)
    
    return processed

@mlflow.trace(name="child_operation")
def process_data(data):
    # Automatically nested under parent
    return transform(data)
```

## Advanced Patterns

### 1. Tracing LLM Calls

```python
@mlflow.trace(span_type=SpanType.LLM)
def call_llm(prompt, model="gpt-4", temperature=0.7):
    import openai
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    
    # MLflow automatically captures inputs/outputs
    return response.choices[0].message.content
```

### 2. Tracing Retrieval Operations

```python
@mlflow.trace(span_type=SpanType.RETRIEVER)
def semantic_search(query, index, top_k=5):
    # Perform embedding
    query_embedding = embed(query)
    
    # Search index
    results = index.search(query_embedding, top_k)
    
    # Add custom attributes
    mlflow.get_current_active_span().set_attributes({
        "retriever.index_size": len(index),
        "retriever.top_k": top_k,
        "retriever.similarity_threshold": 0.7
    })
    
    return results
```

### 3. Error Handling

```python
@mlflow.trace
def risky_operation(data):
    try:
        result = process(data)
        return result
    except Exception as e:
        # Errors are automatically captured
        # But you can add context:
        span = mlflow.get_current_active_span()
        if span:
            span.set_attributes({
                "error.type": type(e).__name__,
                "error.message": str(e)
            })
        raise
```

### 4. Async Functions

```python
@mlflow.trace
async def async_operation(data):
    # MLflow handles async automatically
    result = await async_process(data)
    return result
```

### 5. Batch Operations

```python
@mlflow.trace(name="batch_process")
def process_batch(items):
    results = []
    for i, item in enumerate(items):
        # Create child span for each item
        with mlflow.start_span(name=f"process_item_{i}") as span:
            span.set_attributes({"item.id": item.id})
            result = process_single(item)
            results.append(result)
    return results
```

## Best Practices

### 1. Span Naming
- Use descriptive, consistent names
- Include operation type in the name
- Avoid PII in span names

### 2. Span Types
- Use appropriate SpanType enum values:
  - SpanType.LLM for language model calls
  - SpanType.RETRIEVER for retrieval operations
  - SpanType.CHAIN for orchestration logic
  - SpanType.TOOL for tool/function calls
  - SpanType.AGENT for agent operations

### 3. Attributes
- Add relevant metadata as attributes
- Use dot notation for namespacing (e.g., "llm.model", "llm.temperature")
- Avoid large payloads in attributes

### 4. Inputs/Outputs
- Capture enough information for debugging
- Be mindful of sensitive data
- Consider truncating large responses

### 5. Context Propagation
- MLflow automatically propagates context within a process
- For distributed systems, use trace headers:

```python
# Extract trace context
trace_context = mlflow.get_trace_context()

# Pass to another service
headers = {"X-MLflow-Trace-Context": trace_context}

# In the other service, restore context:
mlflow.set_trace_context(headers["X-MLflow-Trace-Context"])
```

## Common Patterns

### RAG Application
```python
@mlflow.trace(name="rag_query")
def answer_question(question):
    # Retrieve relevant documents
    docs = retrieve_documents(question)
    
    # Generate context
    context = format_context(docs)
    
    # Generate answer
    answer = generate_answer(question, context)
    
    return answer

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_documents(query):
    # Retrieval logic
    pass

@mlflow.trace(span_type=SpanType.LLM)
def generate_answer(question, context):
    # LLM generation logic
    pass
```

### Agent with Tools
```python
@mlflow.trace(span_type=SpanType.AGENT)
def agent_execute(task):
    while not task.is_complete():
        # Decide next action
        action = decide_action(task)
        
        # Execute tool
        result = execute_tool(action)
        
        # Update task state
        task.update(result)
    
    return task.result

@mlflow.trace(span_type=SpanType.TOOL)
def execute_tool(action):
    # Tool execution logic
    pass
```

## Configuration

### Enable Tracing
```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("my_experiment")

# Tracing is now active for decorated functions
```

### Disable Tracing
```python
# Temporarily disable
mlflow.tracing.disable()

# Re-enable
mlflow.tracing.enable()

# Or use environment variable:
# MLFLOW_TRACING_ENABLED=false
```

## Integration with MLflow Tracking

Traces are automatically linked to MLflow runs:

```python
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model", "gpt-4")
    
    # Traced functions automatically associate with this run
    result = my_traced_function()
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
```

## Debugging Tips

1. **View Traces in UI**: Use MLflow UI to visualize trace waterfall
2. **Export Traces**: Export to OTLP format for external analysis
3. **Search Spans**: Use the MCP tools to programmatically search traces
4. **Monitor Performance**: Look for slow spans and bottlenecks
5. **Error Analysis**: Filter for failed spans to debug issues

Remember: Good instrumentation is key to understanding and debugging complex AI applications!"""
