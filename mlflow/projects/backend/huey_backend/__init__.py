import mlflow

from mlflow.projects import SubmittedRun
from mlflow.projects.backend.abstract_backend import AbstractBackend
from mlflow.projects.backend.huey_backend.config import huey
from mlflow.projects.backend.huey_backend.tasks import run_mlflow_project
from mlflow.tracking.client import MlflowClient

from huey.exceptions import TaskException

class HueyBackend(AbstractBackend):

    def __init__(self, backend_str):
        self.backend_str = backend_str

    def run(self, project_uri, entry_point, params, run_id, experiment_id, 
            tracking_backend_store_uri, version=None, backend_config=None):
        task = run_mlflow_project(
            uri=project_uri,
            entry_point=entry_point,
            version=version,
            parameters=params,
            experiment_id=experiment_id,
            run_id=run_id,
            tracking_backend_store_uri=tracking_backend_store_uri,
            synchronous=True,
        )
        return HueySubmittedRun(run_id, task)


class HueySubmittedRun(SubmittedRun):
    """
    """
    def __init__(self, run_id, task):
        super(HueySubmittedRun, self).__init__()
        self._run_id = run_id
        self._task = task

    @property
    def run_id(self):
        return self._run_id

    def wait(self):
        return self._task.get(blocking=True)

    def cancel(self):
        return self._task.revoke()

    def get_status(self):
        try:
            result = self._task.get(blocking=False)
        except TaskException:
            return "FAILED"
        if result is not None:
            return "COMPLETE"
        else:
            return "RUNNING"
