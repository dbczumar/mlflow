import mlflow
from mlflow.tracking.fluent import set_run
from mlflow.utils.databricks_utils import _get_dbutils


# def fetch_project_context():
#     run_id = get_param("run_id")
#     set_run(run_id)
#

def get_param(param):
    return _get_dbutils().widgets.get(param)


def fetch_project_context():
    run_id = get_param("run_id")
    set_run(run_id)
    return ProjectContext(run_id)


class ProjectParams(object):

    def __init__(self):
        pass

    @staticmethod
    def __getitem__(key):
        return _get_dbutils().widgets.get(key)

    @staticmethod
    def get(key, default=None):
        try:
            return _get_dbutils().widgets.get(key)
        except Exception:
            return default


class ProjectContext(object):

    def __init__(self, run_id):
        self.run = mlflow.get_run(run_id)
        self.params = ProjectParams()

