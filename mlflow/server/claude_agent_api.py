"""
Claude Agent API endpoints for MLflow Server.

This module provides endpoints for integrating Claude Code Agent SDK with MLflow UI,
enabling AI-powered trace analysis through a chat interface.
"""

import asyncio
import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

_logger = logging.getLogger(__name__)

# Config file location
CLAUDE_CONFIG_FILE = Path.home() / ".mlflow" / "claude-config.json"

# System prompt for the Claude agent
CLAUDE_SYSTEM_PROMPT = """You are an expert MLflow assistant integrated into the MLflow UI.
Your role is to help users understand and troubleshoot their ML experiments, traces,
runs, and models.

## Core Behaviors

1. **Be Concise**: Users are viewing you in a side panel. Keep responses focused and
scannable. Use bullet points and short paragraphs.

2. **Provide Actionable Insights**: Don't just describe what you see - explain what it
means and what the user should do about it.

3. **Ask for Clarification When Needed**: If a question is ambiguous or you need more
context to give a good answer, ask. It's better to clarify than to guess wrong.

4. **Reference Specific Data**: When analyzing traces/runs, reference specific span
names, timestamps, metrics, or parameters to ground your analysis.

## CRITICAL: Fetching Full Trace Details

When analyzing multi-turn sessions, you ONLY see summaries initially. **You MUST use
WebFetch to get full trace details before analyzing any specific turn.**

**To get full trace data, use WebFetch:**
```
URL: http://127.0.0.1:5000/api/claude-agent/trace/{trace_id}
```

**You MUST do this when:**
- User asks about a specific turn (e.g., "what happened in turn 1")
- User asks to debug or analyze a specific trace
- You need to see LLM prompts, tool calls, or span details

**Workflow:**
1. Find the `Trace ID` for the turn from the session context above
2. Call WebFetch with URL: `http://127.0.0.1:5000/api/claude-agent/trace/{trace_id}`
3. Analyze the returned span tree, inputs/outputs, and any errors

**DO NOT guess or speculate** about trace contents without fetching the full data first.

## When to Search MLflow Documentation

For questions about MLflow features, APIs, or best practices:
- Use WebFetch to retrieve documentation from mlflow.org
- **Pro tip**: Add `.md` to any doc URL to get the markdown source (easier to parse)
  - Example: `https://mlflow.org/docs/latest/llms/tracing/index.html.md`
- Search the docs site for specific topics: `https://mlflow.org/docs/latest/search.html?q=<query>`

## Common Scenarios

### Trace Analysis
- Identify slow spans and potential bottlenecks
- Check for errors or exceptions in span events
- Analyze token usage for LLM calls
- Compare input/output patterns

### Multi-Turn Session Debugging
- Fetch individual trace details to see full span trees
- Compare LLM inputs/outputs across turns
- Track how context evolves through the conversation
- Identify where hallucinations or errors originate

### Run Troubleshooting
- Check parameter configurations
- Analyze metric trends
- Compare with other runs
- Identify failed or stuck runs

### General Questions
- Explain MLflow concepts clearly
- Provide code examples when helpful
- Link to relevant documentation

## Response Format

Use markdown formatting:
- **Bold** for key terms and important findings
- `code` for API names, parameters, file paths
- Bulleted lists for multiple points
- Code blocks for examples

Keep the tone professional but approachable. You're a helpful expert, not a formal
documentation system."""

# Custom agents for specialized tasks
CLAUDE_CUSTOM_AGENTS = {
    "mlflow-docs": {
        "description": (
            "Search MLflow documentation to answer questions about MLflow features, "
            "APIs, and best practices"
        ),
        "prompt": """You are a documentation search specialist for MLflow.

When asked about MLflow features, APIs, or best practices:

1. **Search the MLflow documentation** using WebFetch:
   - Main docs: https://mlflow.org/docs/latest/
   - Add `.md` to any URL to get markdown source (cleaner to parse)
   - Example: https://mlflow.org/docs/latest/llms/tracing/index.html.md

2. **Key documentation sections**:
   - Tracing: https://mlflow.org/docs/latest/llms/tracing/index.html.md
   - Tracking: https://mlflow.org/docs/latest/tracking.html.md
   - Model Registry: https://mlflow.org/docs/latest/model-registry.html.md
   - Python API: https://mlflow.org/docs/latest/python_api/index.html.md
   - REST API: https://mlflow.org/docs/latest/rest-api.html.md
   - LLM Evaluation: https://mlflow.org/docs/latest/llms/llm-evaluate/index.html.md

3. **Provide accurate, documentation-backed answers** with:
   - Direct quotes or summaries from the docs
   - Code examples from the documentation
   - Links to relevant sections for further reading

Always cite your sources by including the documentation URL.""",
    }
}

# Session storage directory (file-based to work across workers)
SESSION_DIR = Path(tempfile.gettempdir()) / "mlflow-claude-sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)


def _get_session_file(session_id: str) -> Path:
    """Get path to session file."""
    return SESSION_DIR / f"{session_id}.json"


def _save_session(session_id: str, data: dict[str, Any]) -> None:
    """Save session data to file."""
    session_file = _get_session_file(session_id)
    with open(session_file, "w") as f:
        json.dump(data, f)


def _load_session(session_id: str) -> dict[str, Any] | None:
    """Load session data from file."""
    session_file = _get_session_file(session_id)
    if session_file.exists():
        try:
            with open(session_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


# Create FastAPI router
claude_agent_router = APIRouter(prefix="/api/claude-agent", tags=["claude-agent"])


class AnalyzeRequest(BaseModel):
    """Request body for trace analysis."""

    trace_context: str  # Serialized trace data
    prompt: str | None = None  # Optional user prompt
    session_id: str | None = None  # For follow-up messages


class MessageRequest(BaseModel):
    """Request body for follow-up messages."""

    session_id: str
    message: str


class AnalyzeResponse(BaseModel):
    """Response for analyze endpoint."""

    session_id: str
    stream_url: str


def _load_config() -> dict[str, Any]:
    """Load Claude config from file."""
    if CLAUDE_CONFIG_FILE.exists():
        try:
            with open(CLAUDE_CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _get_claude_path() -> str | None:
    """Get path to Claude CLI executable."""
    return shutil.which("claude")


async def _run_claude_agent(
    prompt: str,
    cwd: str | None = None,
    model: str | None = None,
    session_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Run Claude Agent and stream responses.

    Uses the Claude Agent SDK via subprocess to invoke Claude Code CLI.

    Args:
        prompt: The prompt to send to Claude
        cwd: Working directory for Claude (to read source files)
        model: Model to use (or None for default)
        session_id: Session ID for resume (or None for new session)

    Yields:
        SSE-formatted event strings
    """
    claude_path = _get_claude_path()
    if not claude_path:
        yield "event: error\ndata: Claude CLI not found. Run 'mlflow claude init' first.\n\n"
        return

    # Build command
    # Note: --verbose is required when using --output-format=stream-json with -p
    cmd = [claude_path, "-p", prompt, "--output-format", "stream-json", "--verbose"]

    # Allow WebFetch for localhost to enable Claude to fetch trace details
    cmd.extend(["--allowedTools", "WebFetch(domain:127.0.0.1)"])

    # Add system prompt
    cmd.extend(["--append-system-prompt", CLAUDE_SYSTEM_PROMPT])

    # Add custom agents
    cmd.extend(["--agents", json.dumps(CLAUDE_CUSTOM_AGENTS)])

    if model and model != "default":
        cmd.extend(["--model", model])

    if session_id:
        if session_data := _load_session(session_id):
            if stored_session := session_data.get("claude_session_id"):
                cmd.extend(["--resume", stored_session])

    try:
        # Start the Claude process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        # Stream stdout
        claude_session_id = None
        async for line in process.stdout:
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue

            try:
                data = json.loads(line_str)
                msg_type = data.get("type", "")

                # Extract session ID if present
                if "session_id" in data:
                    claude_session_id = data["session_id"]

                # Handle different message types
                if msg_type == "assistant":
                    content = data.get("message", {}).get("content", [])
                    text_parts = [
                        block.get("text", "") for block in content if block.get("type") == "text"
                    ]
                    if text_parts:
                        text = " ".join(text_parts)
                        yield f"event: message\ndata: {json.dumps({'text': text})}\n\n"

                elif msg_type == "tool_use":
                    # Forward tool usage as status message
                    tool_name = data.get("name", "")
                    tool_input = data.get("input", {})
                    # Create human-readable status based on tool name
                    if tool_name == "Read":
                        file_path = tool_input.get("file_path", "")
                        fname = file_path.split("/")[-1] if file_path else "file"
                        status_text = f"Reading {fname}..."
                    elif tool_name == "Glob":
                        pattern = tool_input.get("pattern", "")
                        status_text = f"Searching for {pattern}..."
                    elif tool_name == "Grep":
                        pattern = tool_input.get("pattern", "")
                        status_text = f"Searching for '{pattern}'..."
                    elif tool_name == "Bash":
                        cmd = tool_input.get("command", "")
                        # Truncate long commands
                        cmd_preview = cmd[:50] + "..." if len(cmd) > 50 else cmd
                        status_text = f"Running: {cmd_preview}"
                    elif tool_name == "Edit":
                        file_path = tool_input.get("file_path", "")
                        fname = file_path.split("/")[-1] if file_path else "file"
                        status_text = f"Editing {fname}..."
                    elif tool_name == "Write":
                        file_path = tool_input.get("file_path", "")
                        fname = file_path.split("/")[-1] if file_path else "file"
                        status_text = f"Writing {fname}..."
                    elif tool_name == "Task":
                        status_text = "Spawning sub-agent..."
                    elif tool_name == "WebFetch":
                        url = tool_input.get("url", "")
                        status_text = f"Fetching {url[:40]}..." if url else "Fetching URL..."
                    elif tool_name == "WebSearch":
                        query = tool_input.get("query", "")
                        status_text = f"Searching web for '{query}'..."
                    else:
                        status_text = f"Using {tool_name}..."
                    status_data = {"status": status_text, "tool": tool_name}
                    yield f"event: status\ndata: {json.dumps(status_data)}\n\n"

                elif msg_type == "result":
                    # Final result - save claude session ID to file
                    if claude_session_id and session_id:
                        if session_data := _load_session(session_id):
                            session_data["claude_session_id"] = claude_session_id
                            _save_session(session_id, session_data)
                    yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

                elif msg_type == "error":
                    error_msg = data.get("error", {}).get("message", "Unknown error")
                    yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"

            except json.JSONDecodeError:
                # Non-JSON output, treat as plain text
                yield f"event: message\ndata: {json.dumps({'text': line_str})}\n\n"

        # Wait for process to complete
        await process.wait()

        if process.returncode != 0:
            stderr = await process.stderr.read()
            default_err = f"Process exited with code {process.returncode}"
            error_msg = stderr.decode("utf-8").strip() or default_err
            yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"

    except Exception as e:
        _logger.exception("Error running Claude agent")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


@claude_agent_router.post("/analyze")
async def analyze_trace(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Start a trace analysis session with Claude.

    This endpoint creates a new session and returns a stream URL
    for receiving Claude's analysis via Server-Sent Events.

    Args:
        request: AnalyzeRequest containing trace context and optional prompt

    Returns:
        AnalyzeResponse with session_id and stream_url
    """
    # Generate or use existing session ID
    session_id = request.session_id or str(uuid.uuid4())

    # Store session data to file (works across workers)
    _save_session(
        session_id,
        {
            "trace_context": request.trace_context,
            "prompt": request.prompt,
            "messages": [],
        },
    )

    return AnalyzeResponse(
        session_id=session_id,
        stream_url=f"/api/claude-agent/stream/{session_id}",
    )


@claude_agent_router.get("/stream/{session_id}")
async def stream_response(session_id: str) -> StreamingResponse:
    """
    Stream Claude's response via Server-Sent Events.

    Args:
        session_id: The session ID returned from /analyze

    Returns:
        StreamingResponse with SSE events
    """
    session = _load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    config = _load_config()

    # Build the prompt
    trace_context = session["trace_context"]
    user_prompt = session.get("prompt", "")

    full_prompt = f"""Analyze this MLflow trace and help identify any issues:

{trace_context}

{user_prompt or "Please analyze this trace and explain what happened."}"""

    # Get config values
    cwd = config.get("projectPath")
    model = config.get("model")

    async def event_generator() -> AsyncGenerator[str, None]:
        async for event in _run_claude_agent(
            prompt=full_prompt,
            cwd=cwd,
            model=model,
            session_id=session_id,
        ):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@claude_agent_router.post("/message")
async def send_message(request: MessageRequest) -> StreamingResponse:
    """
    Send a follow-up message in an existing session.

    Args:
        request: MessageRequest with session_id and message

    Returns:
        StreamingResponse with SSE events
    """
    session = _load_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    config = _load_config()

    # Add message to session history and save
    session["messages"].append({"role": "user", "content": request.message})
    _save_session(request.session_id, session)

    # Get config values
    cwd = config.get("projectPath")
    model = config.get("model")

    async def event_generator() -> AsyncGenerator[str, None]:
        async for event in _run_claude_agent(
            prompt=request.message,
            cwd=cwd,
            model=model,
            session_id=request.session_id,
        ):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@claude_agent_router.get("/trace/{trace_id}")
async def get_trace_details(trace_id: str, request: Request) -> dict[str, Any]:
    """
    Fetch full trace details for Claude to analyze.

    This endpoint returns a detailed, human-readable representation of a trace
    including all spans, inputs/outputs, assessments, and metadata. Claude can
    use this to drill into specific traces when debugging multi-turn sessions.

    Args:
        trace_id: The ID of the trace to fetch
        request: The incoming request (to extract host info)

    Returns:
        Dictionary containing trace details in markdown format
    """
    import mlflow

    try:
        trace = mlflow.get_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found")

        # Serialize the trace to a detailed format
        trace_details = _serialize_trace_for_claude(trace)

        return {
            "trace_id": trace_id,
            "content": trace_details,
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception(f"Error fetching trace {trace_id}")
        raise HTTPException(status_code=500, detail=str(e))


def _serialize_trace_for_claude(trace) -> str:
    """
    Serialize a trace to a detailed, human-readable markdown format for Claude.

    This provides comprehensive trace data including:
    - Trace metadata (ID, status, timing)
    - All spans with hierarchy
    - Inputs/outputs for each span
    - Assessments and feedback
    - Attributes and events

    Args:
        trace: The MLflow Trace object

    Returns:
        Markdown-formatted string with trace details
    """

    text = "# Trace Details\n\n"

    # Trace info
    info = trace.info
    text += f"- **Trace ID**: {info.trace_id}\n"
    text += f"- **Status**: {info.state}\n"
    text += f"- **Request Time**: {info.request_time}\n"
    if info.execution_duration:
        text += f"- **Duration**: {info.execution_duration}ms\n"

    # Request/Response previews
    if info.request_preview:
        text += f"\n## Request Preview\n```\n{_truncate(info.request_preview, 2000)}\n```\n"
    if info.response_preview:
        text += f"\n## Response Preview\n```\n{_truncate(info.response_preview, 2000)}\n```\n"

    # Tags
    if info.tags:
        text += "\n## Tags\n"
        for key, value in info.tags.items():
            text += f"- {key}: {value}\n"

    # Metadata
    if hasattr(info, "trace_metadata") and info.trace_metadata:
        text += "\n## Metadata\n"
        for key, value in info.trace_metadata.items():
            text += f"- {key}: {value}\n"

    # Assessments at trace level
    if hasattr(info, "assessments") and info.assessments:
        if trace_assessments := [a for a in info.assessments if not getattr(a, "span_id", None)]:
            text += "\n## Trace Assessments\n"
            text += _serialize_assessments(trace_assessments)

    # Spans
    text += "\n---\n\n# Spans\n\n"
    spans = trace.data.spans if trace.data else []
    if not spans:
        text += "No span data available.\n"
    else:
        # Build assessment map for span-level assessments
        span_assessments = {}
        if hasattr(info, "assessments") and info.assessments:
            for assessment in info.assessments:
                if span_id := getattr(assessment, "span_id", None):
                    if span_id not in span_assessments:
                        span_assessments[span_id] = []
                    span_assessments[span_id].append(assessment)

        # Serialize spans hierarchically
        text += _serialize_spans(spans, span_assessments)

    return text


def _truncate(text: str, max_length: int = 2000) -> str:
    """Truncate text to max length."""
    if len(text) > max_length:
        return text[:max_length] + "... (truncated)"
    return text


def _serialize_assessments(assessments) -> str:
    """Serialize assessments to markdown."""
    text = ""
    for assessment in assessments:
        source = assessment.source
        source_type = source.source_type if hasattr(source, "source_type") else "UNKNOWN"
        source_id = source.source_id if hasattr(source, "source_id") else ""

        # Get value based on assessment type
        value = "N/A"
        if hasattr(assessment, "feedback") and assessment.feedback:
            fb = assessment.feedback
            if hasattr(fb, "error") and fb.error:
                value = f"ERROR - {fb.error.error_message or fb.error.error_code}"
            elif hasattr(fb, "value"):
                value = str(fb.value)
        elif hasattr(assessment, "expectation") and assessment.expectation:
            exp = assessment.expectation
            if hasattr(exp, "value"):
                value = str(exp.value)
        elif hasattr(assessment, "issue") and assessment.issue:
            value = str(assessment.issue.value)

        text += f"- **{assessment.name}** ({source_type}: {source_id}): {value}\n"
        if hasattr(assessment, "rationale") and assessment.rationale:
            rationale = _truncate(assessment.rationale, 500)
            text += f"  - Rationale: {rationale}\n"

    return text


def _serialize_spans(spans, span_assessments: dict[str, Any]) -> str:
    """Serialize spans hierarchically."""
    # Build hierarchy
    hierarchy: dict[str | None, list[Any]] = {}
    span_map = {}
    for span in spans:
        span_id = span.span_id
        parent_id = span.parent_id
        span_map[span_id] = span
        if parent_id not in hierarchy:
            hierarchy[parent_id] = []
        hierarchy[parent_id].append(span)

    # Serialize recursively
    return _serialize_spans_recursive(hierarchy, span_assessments, None, 0)


def _serialize_spans_recursive(
    hierarchy: dict[str | None, list[Any]],
    span_assessments: dict[str, Any],
    parent_id: str | None,
    level: int,
) -> str:
    """Recursively serialize spans with indentation."""
    text = ""
    spans = hierarchy.get(parent_id, [])
    indent = "  " * level

    for span in spans:
        span_id = span.span_id
        text += f"{indent}## Span: {span.name}\n"
        text += f"{indent}- **ID**: {span_id}\n"
        text += f"{indent}- **Type**: {span.span_type}\n"
        text += f"{indent}- **Status**: {span.status.status_code}\n"

        # Duration
        if hasattr(span, "start_time") and hasattr(span, "end_time"):
            duration_ms = (span.end_time - span.start_time) / 1e6  # ns to ms
            text += f"{indent}- **Duration**: {duration_ms:.2f}ms\n"

        # Inputs
        if span.inputs:
            text += f"{indent}### Inputs\n{indent}```json\n"
            text += _truncate(json.dumps(span.inputs, indent=2, default=str), 2000)
            text += f"\n{indent}```\n"

        # Outputs
        if span.outputs:
            text += f"{indent}### Outputs\n{indent}```json\n"
            text += _truncate(json.dumps(span.outputs, indent=2, default=str), 2000)
            text += f"\n{indent}```\n"

        # Events (exceptions)
        if span.events:
            if exceptions := [e for e in span.events if e.name == "exception"]:
                text += f"{indent}### Exceptions\n"
                for exc in exceptions:
                    exc_type = exc.attributes.get("exception.type", "Unknown")
                    exc_msg = exc.attributes.get("exception.message", "")
                    text += f"{indent}- **{exc_type}**: {exc_msg}\n"

        # Attributes (excluding inputs/outputs)
        if span.attributes:
            excluded = {"mlflow.spanInputs", "mlflow.spanOutputs", "mlflow.spanType"}
            if attrs := {k: v for k, v in span.attributes.items() if k not in excluded}:
                text += f"{indent}### Attributes\n"
                for key, value in attrs.items():
                    text += f"{indent}- {key}: {_truncate(str(value), 200)}\n"

        # Span assessments
        if span_id in span_assessments:
            text += f"{indent}### Assessments\n"
            text += _serialize_assessments(span_assessments[span_id])

        text += "\n"

        # Recursively add children
        text += _serialize_spans_recursive(hierarchy, span_assessments, span_id, level + 1)

    return text


@claude_agent_router.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Health status and whether Claude CLI is available
    """
    claude_available = _get_claude_path() is not None
    config_exists = CLAUDE_CONFIG_FILE.exists()

    return {
        "status": "ok",
        "claude_available": str(claude_available),
        "config_exists": str(config_exists),
    }
