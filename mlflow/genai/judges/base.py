from mlflow.genai.scorers.base import Scorer
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class Judge(Scorer):
    """
    Base class for AI-as-a-judge scorers that can be aligned with human feedback.
    """
