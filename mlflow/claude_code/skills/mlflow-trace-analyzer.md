# MLflow Trace Analyzer

You are helping analyze MLflow traces. MLflow traces capture the execution flow of LLM applications, including inputs, outputs, and any errors that occurred.

## Trace Structure

An MLflow trace consists of:

- **Trace ID**: Unique identifier for the trace (e.g., `tr-abc123`)
- **Spans**: Individual operations within the trace, organized in a tree hierarchy
- **Root Span**: The top-level span representing the entire request
- **Child Spans**: Nested operations (LLM calls, retrievers, tools, etc.)

## Span Anatomy

Each span contains:

- **Name**: Operation name (e.g., `ChatOpenAI`, `Retriever`, `ToolCall`)
- **Span Type**: Category (`LLM`, `RETRIEVER`, `TOOL`, `CHAIN`, `AGENT`, `EMBEDDING`)
- **Status**: `OK`, `ERROR`, or `UNSET`
- **Start/End Time**: Timestamps for duration calculation
- **Inputs**: The data passed into the operation
- **Outputs**: The result of the operation
- **Attributes**: Additional metadata (model name, token counts, etc.)
- **Events**: Significant occurrences, including exceptions

## Common Span Types

| Type | Description | Typical Attributes |
|------|-------------|-------------------|
| `LLM` | Language model call | `model`, `temperature`, `max_tokens`, `token_usage` |
| `RETRIEVER` | Document retrieval | `query`, `top_k`, `documents` |
| `TOOL` | Tool/function call | `tool_name`, `arguments`, `result` |
| `CHAIN` | Sequential operations | `chain_type` |
| `AGENT` | Agent reasoning loop | `agent_type`, `iterations` |
| `EMBEDDING` | Text embedding | `model`, `dimensions` |

## Analyzing a Trace

When analyzing a trace:

1. **Start with the root span** - Understand the overall request
2. **Follow the span tree** - Trace the execution flow from parent to children
3. **Check span status** - Look for `ERROR` status spans
4. **Examine inputs/outputs** - Understand data flow between spans
5. **Review attributes** - Check model parameters, token usage, etc.
6. **Look for exceptions** - Exception events contain error details

## Token Usage Analysis

For LLM spans, check token usage in attributes:
- `llm.token_usage.prompt_tokens` - Input token count
- `llm.token_usage.completion_tokens` - Output token count
- `llm.token_usage.total_tokens` - Combined token count

## Latency Analysis

Calculate latency from span timestamps:
- Individual span duration = `end_time - start_time`
- Critical path = longest chain of sequential spans
- Parallel operations = spans with overlapping time ranges

## Best Practices for Analysis

1. **Identify the error span** - Find where the failure occurred
2. **Check the inputs** - Were they valid? Too large? Malformed?
3. **Review parent context** - What led to this operation?
4. **Look for patterns** - Repeated failures? Rate limits? Timeouts?
5. **Suggest actionable fixes** - Provide specific code changes when possible
