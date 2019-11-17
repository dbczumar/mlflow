from mlflow.tracking.fluent import set_run
from mlflow.utils.databricks_utils import _get_dbutils


def set_project_context():
    run_id = get_param("run_id")
    set_run(run_id)


def get_param(param):
    return _get_dbutils().widgets.get(param)
