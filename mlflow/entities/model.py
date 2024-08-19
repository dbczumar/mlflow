from typing import Any, Dict, List, Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.model_param import ModelParam
from mlflow.entities.model_status import ModelStatus
from mlflow.entities.model_tag import ModelTag


class Model(_MlflowObject):
    """
    MLflow entity representing a Model.
    """

    def __init__(
        self,
        experiment_id: str,  # New field added
        model_id: str,
        name: str,
        creation_timestamp: int,
        last_updated_timestamp: int,
        run_id: Optional[str] = None,
        status: ModelStatus = ModelStatus.READY,
        status_message: Optional[str] = None,
        tags: Optional[List[ModelTag]] = None,
        params: Optional[ModelParam] = None,
    ):
        super().__init__()
        self._experiment_id: str = experiment_id  # New field initialized
        self._model_id: str = model_id
        self._name: str = name
        self._creation_time: int = creation_timestamp
        self._last_updated_timestamp: int = last_updated_timestamp
        self._run_id: Optional[str] = run_id
        self._status: ModelStatus = status
        self._status_message: Optional[str] = status_message
        self._tags: Dict[str, str] = {tag.key: tag.value for tag in (tags or [])}
        self._params: Optional[ModelParam] = params

    @property
    def experiment_id(self) -> str:
        """String. Experiment ID associated with this Model."""
        return self._experiment_id

    @experiment_id.setter
    def experiment_id(self, new_experiment_id: str):
        self._experiment_id = new_experiment_id

    @property
    def model_id(self) -> str:
        """String. Unique ID for this Model."""
        return self._model_id

    @model_id.setter
    def model_id(self, new_model_id: str):
        self._model_id = new_model_id

    @property
    def name(self) -> str:
        """String. Name for this Model."""
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    @property
    def creation_timestamp(self) -> int:
        """Integer. Model creation timestamp (milliseconds since the Unix epoch)."""
        return self._creation_time

    @property
    def last_updated_timestamp(self) -> int:
        """Integer. Timestamp of last update for this Model (milliseconds since the Unix
        epoch).
        """
        return self._last_updated_timestamp

    @last_updated_timestamp.setter
    def last_updated_timestamp(self, updated_timestamp: int):
        self._last_updated_timestamp = updated_timestamp

    @property
    def run_id(self) -> Optional[str]:
        """String. MLflow run ID that generated this model."""
        return self._run_id

    @property
    def status(self) -> ModelStatus:
        """String. Current status of this Model."""
        return self._status

    @status.setter
    def status(self, updated_status: str):
        self._status = updated_status

    @property
    def status_message(self) -> Optional[str]:
        """String. Descriptive message for error status conditions."""
        return self._status_message

    @property
    def tags(self) -> Dict[str, str]:
        """Dictionary of tag key (string) -> tag value for this Model."""
        return self._tags

    @property
    def params(self) -> Optional[ModelParam]:
        """Model parameters."""
        return self._params

    @classmethod
    def _properties(cls) -> List[str]:
        # aggregate with base class properties since cls.__dict__ does not do it automatically
        return sorted(cls._get_properties_helper())

    def _add_tag(self, tag):
        self._tags[tag.key] = tag.value

    def to_dictionary(self) -> Dict[str, Any]:
        model_dict = dict(self)
        model_dict["status"] = str(self.status)
        return model_dict