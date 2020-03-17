import mlflow

from mlflow.projects import SubmittedRun
from mlflow.projects.backend.abstract_backend import AbstractBackend
from mlflow.projects.backend.huey_backend.config import huey
from mlflow.projects.backend.huey_backend.tasks import run_mlflow_project
from mlflow.tracking.client import MlflowClient

class HueyBackend(AbstractBackend):

    def __init__(self, backend_str):
        self.backend_str = backend_str

    def run(self, project_uri, entry_point, params, experiment_id, run_id=None, version=None,
            backend_config=None):
        if run_id is None:
            new_run = MlflowClient().create_run(experiment_id=experiment_id)
            run_id = new_run.info.run_id

        run_mlflow_project(
            uri=project_uri,
            entry_point=entry_point,
            version=version,
            parameters=params,
            experiment_id=experiment_id,
            run_id=run_id,
        )
        return HueySubmittedRun(run_id)


class HueySubmittedRun(SubmittedRun):
    """
    """
    def __init__(self, run_id):
        super(HueySubmittedRun, self).__init__()
        self._run_id = run_id

    @property
    def run_id(self):
        return self._run_id

    def wait(self):
        raise NotImplementedError("Not implemented!")

    def cancel(self):
        raise NotImplementedError("Not implemented!")

    def get_status(self):
        raise NotImplementedError("Not implemented!")
