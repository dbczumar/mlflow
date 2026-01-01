# MLflow Error Debugger

You are helping debug errors in LLM applications by analyzing MLflow traces. This skill helps you identify common error patterns and provide actionable solutions.

## Exception Structure in Traces

Exceptions in MLflow traces appear as events on spans with:
- `exception.type`: The exception class name
- `exception.message`: The error message
- `exception.stacktrace`: Full stack trace (when available)

## Common LLM Error Categories

### 1. Rate Limit Errors

**Symptoms:**
- `RateLimitError`, `429 Too Many Requests`
- High request frequency in trace timeline
- Multiple failed retries

**Common Causes:**
- Too many requests per minute/hour
- Token quota exceeded
- Concurrent request limits

**Solutions:**
```python
# Add exponential backoff
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=4, max=60),
       stop=stop_after_attempt(5))
def call_llm(prompt):
    return client.chat.completions.create(...)

# Or use rate limiting
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=50, period=60)  # 50 calls per minute
def call_llm(prompt):
    return client.chat.completions.create(...)
```

### 2. Token Limit Errors

**Symptoms:**
- `InvalidRequestError: maximum context length exceeded`
- `tokens exceeds maximum`
- Very high `prompt_tokens` in attributes

**Common Causes:**
- Input text too long
- Too many few-shot examples
- Large document retrieval without truncation

**Solutions:**
```python
# Truncate input
def truncate_to_tokens(text, max_tokens=3000):
    # Use tiktoken for accurate counting
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        return enc.decode(tokens[:max_tokens])
    return text

# Or use summarization for long documents
def summarize_if_long(text, max_tokens=3000):
    if count_tokens(text) > max_tokens:
        return summarize(text)
    return text
```

### 3. API Connection Errors

**Symptoms:**
- `ConnectionError`, `TimeoutError`
- `APIConnectionError`
- Network-related exceptions

**Common Causes:**
- Network instability
- API service outage
- Firewall/proxy issues
- DNS resolution failures

**Solutions:**
```python
# Add timeout and retry
import httpx

client = OpenAI(
    timeout=httpx.Timeout(60.0, connect=5.0),
    max_retries=3
)
```

### 4. Invalid Response Errors

**Symptoms:**
- `JSONDecodeError`
- `ValidationError`
- Unexpected response format

**Common Causes:**
- Model output not following expected format
- Missing required fields in structured output
- Prompt not clear about expected format

**Solutions:**
```python
# Use structured outputs (OpenAI)
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: float

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[...],
    response_format=Response
)

# Or add validation with retry
def get_valid_response(prompt, max_retries=3):
    for _ in range(max_retries):
        response = call_llm(prompt)
        try:
            return validate_response(response)
        except ValidationError:
            prompt = f"{prompt}\n\nPrevious response was invalid. Please try again."
    raise ValueError("Failed to get valid response")
```

### 5. Authentication Errors

**Symptoms:**
- `AuthenticationError`
- `401 Unauthorized`
- `Invalid API key`

**Common Causes:**
- Missing or invalid API key
- Expired credentials
- Wrong environment variable

**Solutions:**
```python
# Check API key is set
import os

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Validate key format
if not api_key.startswith("sk-"):
    raise ValueError("Invalid API key format")
```

### 6. Model Not Found Errors

**Symptoms:**
- `NotFoundError`
- `model_not_found`
- `The model does not exist`

**Common Causes:**
- Typo in model name
- Model deprecated or removed
- No access to requested model

**Solutions:**
```python
# Use model constants
MODELS = {
    "gpt4": "gpt-4-turbo-preview",
    "gpt35": "gpt-3.5-turbo",
    "claude": "claude-3-sonnet-20240229"
}

def get_model(alias):
    if alias not in MODELS:
        raise ValueError(f"Unknown model: {alias}. Available: {list(MODELS.keys())}")
    return MODELS[alias]
```

## Debugging Workflow

1. **Locate the error span** - Find the span with `ERROR` status
2. **Read the exception** - Check `exception.type` and `exception.message`
3. **Check the inputs** - Review what was passed to the failing operation
4. **Review parent spans** - Understand the context leading to the error
5. **Check attributes** - Look for relevant metadata (token counts, model, etc.)
6. **Identify the pattern** - Match to one of the categories above
7. **Suggest a fix** - Provide specific, actionable code changes

## When Reading User's Code

If the user provides a project path, you can:
1. Read the source code that caused the error
2. Find the exact line mentioned in the stacktrace
3. Suggest specific changes to their code
4. Check for missing error handling
5. Review configuration and environment setup
