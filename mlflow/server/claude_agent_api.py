"""
Claude Agent API endpoints for MLflow Server.

This module provides endpoints for integrating Claude Code Agent SDK with MLflow UI,
enabling AI-powered trace analysis through a chat interface.
"""

import asyncio
import json
import logging
import shutil
import subprocess
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
CLAUDE_SYSTEM_PROMPT = """You are an expert MLflow assistant integrated into the MLflow UI. Your role is to help users understand and troubleshoot their ML experiments, traces, runs, and models.

## Core Behaviors

1. **Be Concise**: Users are viewing you in a side panel. Keep responses focused and scannable. Use bullet points and short paragraphs.

2. **Provide Actionable Insights**: Don't just describe what you see - explain what it means and what the user should do about it.

3. **Ask for Clarification When Needed**: If a question is ambiguous or you need more context to give a good answer, ask. It's better to clarify than to guess wrong.

4. **Reference Specific Data**: When analyzing traces/runs, reference specific span names, timestamps, metrics, or parameters to ground your analysis.

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

Keep the tone professional but approachable. You're a helpful expert, not a formal documentation system."""

# Custom agents for specialized tasks
CLAUDE_CUSTOM_AGENTS = {
    "mlflow-docs": {
        "description": "Search MLflow documentation to answer questions about MLflow features, APIs, and best practices",
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
        yield f"event: error\ndata: Claude CLI not found. Run 'mlflow claude init' first.\n\n"
        return

    # Build command
    # Note: --verbose is required when using --output-format=stream-json with -p
    cmd = [claude_path, "-p", prompt, "--output-format", "stream-json", "--verbose"]

    # Add system prompt
    cmd.extend(["--append-system-prompt", CLAUDE_SYSTEM_PROMPT])

    # Add custom agents
    cmd.extend(["--agents", json.dumps(CLAUDE_CUSTOM_AGENTS)])

    if model and model != "default":
        cmd.extend(["--model", model])

    if session_id:
        session_data = _load_session(session_id)
        if session_data:
            stored_session = session_data.get("claude_session_id")
            if stored_session:
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
                        block.get("text", "")
                        for block in content
                        if block.get("type") == "text"
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
                        status_text = f"Reading {file_path.split('/')[-1] if file_path else 'file'}..."
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
                        status_text = f"Editing {file_path.split('/')[-1] if file_path else 'file'}..."
                    elif tool_name == "Write":
                        file_path = tool_input.get("file_path", "")
                        status_text = f"Writing {file_path.split('/')[-1] if file_path else 'file'}..."
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
                    yield f"event: status\ndata: {json.dumps({'status': status_text, 'tool': tool_name})}\n\n"

                elif msg_type == "result":
                    # Final result - save claude session ID to file
                    if claude_session_id and session_id:
                        session_data = _load_session(session_id)
                        if session_data:
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
            error_msg = stderr.decode("utf-8").strip() or f"Process exited with code {process.returncode}"
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
    _save_session(session_id, {
        "trace_context": request.trace_context,
        "prompt": request.prompt,
        "messages": [],
    })

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

{user_prompt if user_prompt else "Please analyze this trace and explain what happened."}"""

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
