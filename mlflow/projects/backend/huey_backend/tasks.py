import mlflow.projects
from mlflow.projects.backend.huey_backend.config import huey

@huey.task()
def run_mlflow_project(uri, **kwargs):
    submitted_run = mlflow.projects.run(uri, **kwargs)
    # return submitted_run
    return "done" 
