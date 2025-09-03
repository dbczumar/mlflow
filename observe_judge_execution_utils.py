"""
Utility module for MLflow trace judging with automatic logging and tracing.
"""

import mlflow
from contextlib import contextmanager
from typing import Any, Generator


@contextmanager
def observe_judge_execution() -> Generator[None, None, None]:
    """
    Context manager that enables OpenAI autologging and adds judge_overall tracing logic.
    
    This context manager:
    1. Enables MLflow OpenAI autologging
    2. Creates a span for the judge execution
    3. Sets appropriate tags and metadata
    4. Yields control to execute judge code
    5. Allows caller to set inputs/outputs as needed
        
    Usage:
        with observe_judge_execution():
            feedback = judge(trace=trace)
    """
    # Enable OpenAI autologging
    mlflow.openai.autolog()
    
    # Create span for judge execution
    with mlflow.start_span(name="judge_overall") as span:
        mlflow.update_current_trace(tags={
            "judge": "judge_overall", 
            "is_judge": "yes"
        })
        
        try:
            yield
        finally:
            # Note: inputs and outputs should be set by the caller
            pass


def set_outputs_for_current_span(feedback: Any) -> None:
    """
    Helper function to set outputs for the current span after judge execution.
    
    Args:
        feedback: The feedback object returned by the judge
    """
    mlflow.update_current_trace(response_preview=f"{feedback.value}: {feedback.rationale[:100]}")
    # Get the current span and set outputs
    if hasattr(mlflow, '_get_current_span'):
        current_span = mlflow._get_current_span()
        if current_span:
            current_span.set_outputs(feedback)