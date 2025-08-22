# MLflow Judges

MLflow Judges provide a flexible framework for evaluating LLM traces and outputs using natural language instructions.

## Quick Start with `make_judge()`

### Trace-Based Evaluation

The most powerful way to use judges is with trace evaluation, where the judge analyzes the complete interaction:

```python
from mlflow.genai.judges import make_judge
from mlflow.tracking import MlflowClient

# Create a judge that evaluates traces
judge = make_judge(
    name="quality_judge",
    instructions="""
    Evaluate the {{trace}} to determine if the assistant provided helpful responses.

    Consider:
    - Was the response accurate and complete?
    - Did the assistant understand the user's intent?
    - Was the interaction efficient?

    Rate as: "excellent", "good", "fair", or "poor"
    """,
    model="openai:/gpt-4o",
)

# Get a trace and evaluate it
client = MlflowClient()
trace = client.get_trace(trace_id)
feedback = judge(trace=trace)

print(f"Rating: {feedback.value}")
print(f"Rationale: {feedback.rationale}")
```

### Using Expectations with Traces

You can provide expected behaviors or guidelines as expectations:

```python
judge = make_judge(
    name="compliance_judge",
    instructions="""
    Evaluate if the {{trace}} follows these guidelines: {{expectations}}

    Check each guideline and determine overall compliance.
    Rate as: "compliant", "partially_compliant", or "non_compliant"
    """,
    model="openai:/gpt-4o",
)

# Evaluate with specific expectations
feedback = judge(
    trace=trace,
    expectations={
        "guidelines": [
            "Must not provide medical advice",
            "Should cite sources when making claims",
            "Must be respectful and professional",
        ]
    },
)
```

### Field-Based Evaluation

For simpler evaluations, you can use arbitrary template variables with inputs/outputs:

```python
# Using standard fields
judge = make_judge(
    name="correctness_judge",
    instructions="""
    Compare the {{answer}} with the {{expected_answer}}.
    Rate as "correct" if they match, "incorrect" otherwise.
    """,
    model="openai:/gpt-4o",
)

feedback = judge(outputs={"answer": "Paris"}, expectations={"expected_answer": "Paris"})
```

### Custom Template Variables

You can define any template variables you need:

```python
judge = make_judge(
    name="translation_judge",
    instructions="""
    Evaluate the translation quality:
    - Source ({{source_lang}}): {{source_text}}
    - Translation ({{target_lang}}): {{translated_text}}

    Consider accuracy, fluency, and preservation of meaning.
    Rate as: "excellent", "good", "acceptable", or "poor"
    """,
    model="openai:/gpt-4o",
)

feedback = judge(
    inputs={
        "source_lang": "English",
        "source_text": "Hello, world!",
        "target_lang": "French",
    },
    outputs={"translated_text": "Bonjour, le monde!"},
)
```

## Key Features

- **Trace Analysis**: When using `{{trace}}`, judges automatically use tools to analyze the complete interaction structure
- **Flexible Templates**: Define any variables you need for your evaluation criteria
- **Expectations Support**: Provide guidelines, ground truth, or expected behaviors
- **Structured Output**: Always returns JSON with `result` and `rationale` fields

## Best Practices

1. **Prefer trace-based evaluation** when you need to understand the full context of an interaction
2. **Use expectations** to provide dynamic evaluation criteria without modifying the judge
3. **Keep instructions clear and specific** about the rating scale and criteria
4. **Specify the model explicitly** for consistent evaluation behavior

## Advanced Usage

See the [full documentation](https://docs.mlflow.org/en/latest/llms/judge.html) for more advanced features including:

- Custom judge implementations
- Batch evaluation
- Integration with MLflow experiments
- Built-in judge functions
