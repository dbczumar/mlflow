"""Tool definitions for MLflow MCP server."""

import json
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator


class GetTraceInfoParams(BaseModel):
    """Parameters for get_trace_info tool."""

    trace_id: str = Field(description="The unique identifier of the trace to retrieve")


class ListTraceSpansParams(BaseModel):
    """Parameters for list_trace_spans tool."""

    trace_id: str = Field(description="The unique identifier of the trace whose spans to list")
    max_content_length: int = Field(
        default=100000,
        description="Maximum number of characters in JSON response before pagination",
    )
    page_token: Optional[str] = Field(
        default=None,
        description="Page token for retrieving next page of results",
    )

    @field_validator("max_content_length", mode="before")
    @classmethod
    def parse_max_content_length(cls, v):
        """Parse string numbers to integers for MCP compatibility."""
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"max_content_length must be a valid integer, got: {v}")
        return v


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

    @field_validator("max_content_length", mode="before")
    @classmethod
    def parse_max_content_length(cls, v):
        """Parse string numbers to integers for MCP compatibility."""
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"max_content_length must be a valid integer, got: {v}")
        return v


class SearchTraceSpansParams(BaseModel):
    """Parameters for search_trace_spans tool."""

    trace_id: str = Field(description="The unique identifier of the trace to search within")
    keywords: Optional[Union[list[str], str]] = Field(
        default=None,
        description="Keywords to search for in span CONTENT (inputs, outputs, attributes). Use AS FEW AS POSSIBLE for performance. Keywords are OR'd together. DO NOT use for timing/metadata - those are already shown in the lightweight span list",
    )
    max_content_length: int = Field(
        default=100000,
        description="Maximum number of characters in JSON response before pagination",
    )
    page_token: Optional[str] = Field(
        default=None,
        description="Page token for retrieving next page of results",
    )

    @field_validator("keywords", mode="before")
    @classmethod
    def parse_keywords(cls, v):
        """Parse JSON strings into lists if needed for keywords."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
                else:
                    return [str(parsed)]
            except json.JSONDecodeError:
                return [v]
        return v

    @field_validator("max_content_length", mode="before")
    @classmethod
    def parse_max_content_length(cls, v):
        """Parse string numbers to integers for MCP compatibility."""
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"max_content_length must be a valid integer, got: {v}")
        return v


class SearchTracesParams(BaseModel):
    """Parameters for search_traces tool."""

    # NOTE: Union[list[str], str] is required for MCP compatibility
    # MCP clients may send arrays as JSON strings that need parsing
    experiment_ids: Optional[Union[list[str], str]] = Field(
        default=None,
        description="List of experiment IDs to search within (provide either experiment_ids or experiment_names)",
    )
    experiment_names: Optional[Union[list[str], str]] = Field(
        default=None,
        description="List of experiment names to search within (will be resolved to IDs)",
    )
    filter_string: Optional[str] = Field(
        default=None,
        description="MLflow filter expression for trace attributes. STRONGLY RECOMMENDED: include recent timestamp filter for performance (e.g., \"timestamp_ms > 1735689600000 AND status = 'ERROR'\")",
    )
    span_keywords: Optional[Union[list[str], str]] = Field(
        default=None,
        description="Keywords to search for in span CONTENT (inputs, outputs, attributes). Use AS FEW AS POSSIBLE for best performance. Keywords are OR'd together (any match = found). DO NOT use for timing/metadata - use filter_string for those instead",
    )
    max_results: Optional[int] = Field(
        default=1,
        description="Maximum number of traces to return per page. STRONGLY RECOMMENDED: Use 1 for best performance! Always check if results satisfy your needs before requesting more pages with the returned page token",
    )
    # NOTE: Union[list[str], str] is required for MCP compatibility
    order_by: Optional[Union[list[str], str]] = Field(
        default=None,
        description="List of fields to order results by",
    )
    page_token: Optional[str] = Field(
        default=None,
        description="Token for retrieving next page of results. Omit for first page",
    )

    @field_validator(
        "experiment_ids", "experiment_names", "order_by", "span_keywords", mode="before"
    )
    @classmethod
    def parse_json_lists(cls, v):
        """Parse JSON strings into lists if needed.

        CRITICAL: This parsing logic is required for MCP compatibility.

        Claude Code and other MCP clients often serialize list parameters as JSON strings
        instead of native arrays. For example:
        - Claude Code sends: experiment_ids='["2082254368706020"]' (JSON string)
        - We need to parse it to: ["2082254368706020"] (actual list)

        Without this validator, MCP tool calls fail with "not valid under any of the given schemas".
        DO NOT REMOVE this logic unless you're certain all MCP clients send native arrays.
        """
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                # Ensure parsed items are strings, and convert to list if needed
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
                else:
                    return [str(parsed)]
            except json.JSONDecodeError:
                # If not valid JSON, treat as single item
                return [v]
        return v

    @field_validator("max_results", mode="before")
    @classmethod
    def parse_integer_fields(cls, v):
        """Parse string numbers to integers for MCP compatibility.

        MCP clients may send numbers as strings (e.g., max_results="10").
        Convert these to proper integers.
        """
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"Integer field must be a valid integer, got: {v}")
        return v

    def model_post_init(self, __context):
        """Validate that either experiment_ids or experiment_names is provided."""
        if not self.experiment_ids and not self.experiment_names:
            raise ValueError("Either experiment_ids or experiment_names must be provided")


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
        "description": """Search for spans within a trace using keyword matching.

⚠️  IMPORTANT: keywords search span CONTENT ONLY (inputs, outputs, attributes, names).
    DO NOT use keywords for timing/metadata filters (duration, start/end times, etc.).
    Those are already visible in the lightweight span list from list_trace_spans.

🚀 PERFORMANCE: Use AS FEW keywords AS POSSIBLE for best performance!
   Keywords are OR'd together (finding ANY keyword = match).

Supports:
- Fast keyword search across span names, inputs, outputs, and attributes
- Content-based filtering within a specific trace
- Case-insensitive matching

Returns matching spans with lightweight details.

Example use cases:
- Find spans with specific content: keywords=["hello", "greeting"]
- Search LLM outputs: keywords=["error", "timeout"]
- Locate tool calls: keywords=["calculator", "math"]
- Find model attributes: keywords=["gpt-4", "temperature"]

❌ DON'T use for: keywords=["duration", "1000"] (use list_trace_spans instead)
❌ DON'T use: keywords=["slow", "fast", "quick", "long", "short"] (too many keywords)
✅ DO use for: keywords=["timeout"] (specific content, minimal keywords)
✅ DO use for: keywords=["error", "failed"] (related terms, still minimal)""",
        "params_model": SearchTraceSpansParams,
    },
    "search_traces": {
        "description": """Search across multiple traces in MLflow experiments.

REQUIRED: Must specify either experiment_ids or experiment_names.

🚀 PERFORMANCE CRITICAL: 
   1. STRONGLY recommend specifying a recent minimum timestamp in filter_string 
      for much better performance! Use filter_string="timestamp_ms > RECENT_TIMESTAMP"
      where RECENT_TIMESTAMP is from the last few hours/days, not weeks/months ago.
      Example: filter_string="timestamp_ms > 1735689600000" (last 24 hours)
   
   2. STRONGLY recommend max_results=1 (or other LOW values) for best performance! 
      Higher values can be extremely slow when span filtering is used.
      
   3. ALWAYS check if the returned results satisfy your needs BEFORE requesting 
      more pages! Use the returned next_page_token to fetch additional results 
      only if the current page doesn't contain what you're looking for.

Supports filtering by:
- Experiment ID or name (REQUIRED)
- Trace metadata: status, tags, time ranges (use filter_string)
- Span content matching (use span_keywords)

⚠️  IMPORTANT: span_keywords searches span CONTENT ONLY (inputs, outputs, attributes).
    DO NOT use span_keywords for timing/metadata filters (execution time, timestamps).
    Use filter_string for trace-level metadata filtering instead.

🚀 PERFORMANCE: Use AS FEW keywords AS POSSIBLE for best performance!
   Keywords are OR'd together (finding ANY keyword = match).

📄 PAGINATION WORKFLOW: 
   1. Request with low max_results (ideally 1)
   2. Check if returned traces satisfy your needs
   3. If satisfied: STOP! Don't request more pages
   4. If need more: Use next_page_token to fetch additional results
   5. Repeat until satisfied or no more results

Experiment specification:
- experiment_ids: Direct list of experiment IDs ["1", "2"] 
- experiment_names: List of names ["my_experiment", "test_exp"] (resolved to IDs)

Example workflow:
1. First request: experiment_names=["my_exp"], filter_string="timestamp_ms > 1735689600000", max_results=1
2. Check if returned trace satisfies your needs
3. If yes: Done! If no: Use next_page_token for additional results
4. Next request: same params + page_token from step 1

Example use cases:
- Find recent failed traces: experiment_names=["my_exp"], filter_string="status = 'ERROR' AND timestamp_ms > 1735689600000", max_results=1
- Search recent span content: experiment_names=["rag_exp"], filter_string="timestamp_ms > 1735689600000", span_keywords=["hello"], max_results=1  
- Get latest traces: experiment_ids=["123"], filter_string="timestamp_ms > 1735689600000", order_by=["timestamp_ms DESC"], max_results=1

❌ DON'T: No timestamp filter (slow on large datasets)
❌ DON'T: max_results=20 (extremely slow with span filtering)
❌ DON'T: Request many pages without checking if you're satisfied
❌ DON'T: span_keywords=["duration", "1000"] (use filter_string for timing)
✅ DO: filter_string="timestamp_ms > RECENT_TIMESTAMP" (much faster)
✅ DO: max_results=1, check results, use page_token only if needed
✅ DO: span_keywords=["timeout"] (minimal, specific keywords)""",
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
