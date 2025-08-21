"""
Factory function for creating MLflow judge instances.

This module provides a flexible interface for creating custom judges
that can evaluate traces based on user-defined criteria.
"""

from mlflow.genai.judges.base import Judge
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
def make_judge(*args, **kwargs) -> Judge:
    """
    Create a custom MLflow judge instance.

    This is a placeholder function that will be implemented later.
    """
    raise NotImplementedError("make_judge function is not yet implemented")
