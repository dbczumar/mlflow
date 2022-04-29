import json
import pathlib

import mlflow.utils.databricks_utils as databricks_utils
from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import http_request
from mlflow.utils.file_utils import read_yaml

_PIPELINE_CONFIG_FILE_NAME = "pipeline.yaml"


#REMOVE LATER
import logging
_logger = logging.getLogger(__name__)


def get_pipeline_name(pipeline_root_path: str = None) -> str:
    """
    Obtains the name of the specified pipeline or of the pipeline corresponding to the current
    working directory.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem. If unspecified, the pipeline root directory is
                               resolved from the current working directory, and an
                               ``MlflowException`` is thrown if the current working directory does
                               not correspond to a pipeline.
    :return: The name of the specified pipeline.
    """
    pipeline_root_path = pipeline_root_path or get_pipeline_root_path()
    pipeline_config = read_yaml(root=pipeline_root_path, file_name=_PIPELINE_CONFIG_FILE_NAME)
    return pipeline_config["name"]


def get_pipeline_root_path() -> str:
    """
    Obtains the path of the pipeline corresponding to the current working directory, throwing an
    ``MlflowException`` if the current working directory does not reside within a pipeline
    directory.

    :return: The absolute path of the pipeline root directory on the local filesystem.
    """
    # In the release version of MLflow Pipelines, each pipeline will be its own git repository.
    # To improve developer velocity for now, we choose to treat a pipeline as a directory, which
    # may be a subdirectory of a git repo. The logic for resolving the repository root for
    # development purposes finds the first `pipeline.yaml` file by traversing up the directory
    # tree, while the release version will find the pipeline repository root (commented out below)
    curr_dir_path = pathlib.Path.cwd()

    DEBUGGING_REMOVE_LATER_CURR_ITER = 0
    while True and DEBUGGING_REMOVE_LATER_CURR_ITER < 50:
        _logger.info("CURR DIR", str(curr_dir_path))
        pipeline_yaml_path_to_check = curr_dir_path / _PIPELINE_CONFIG_FILE_NAME
        if pipeline_yaml_path_to_check.exists():
            return str(curr_dir_path.resolve())
        elif curr_dir_path != curr_dir_path.parent:
            curr_dir_path = curr_dir_path.parent
        DEBUGGING_REMOVE_LATER_CURR_ITER += 1
        # else:
    # If curr_dir_path == curr_dir_path.parent,
    # we have reached the root directory without finding
    # the desired pipeline.yaml file
    raise MlflowException(f"Failed to find {_PIPELINE_CONFIG_FILE_NAME}")

    # TODO: Uncomment and fix this pipeline repository root resolution code prior to release,
    #       and replace the current method implementation with this logic
    # TODO: Figure out how to do this in Databricks Jobs (notebook ID isn't available)
    #
    # def _get_databricks_repo_root_path(repo_notebook_id: int) -> str:
    #     """
    #     Obtains the absolute Databricks Workspace Filesystem path to the root directory of the
    #     Databricks Repo associated with the specified Databricks Notebook. This method assumes
    #     that the notebook exists and that it resides in a Databricks Repo.
    #
    #     :param repo_notebook_id: The ID of the Databricks Notebook that resides in a Databricks
    #                              Repo.
    #     :return: The absolute Databricks Workspace Filesystem path to root directory of the
    #              Databricks Repo associated with the specified Databricks Notebook.
    #     """
    #     OBJECT_GIT_INFORMATION_ENDPOINT = "/api/2.0/workspace/get-object-git-information"
    #
    #     repo_notebook_git_info_response = http_request(
    #         host_creds=databricks_utils.get_databricks_host_creds(),
    #         endpoint=OBJECT_GIT_INFORMATION_ENDPOINT,
    #         method="GET",
    #         params={
    #             "object_id": repo_notebook_id,
    #         },
    #     )
    #     repo_notebook_git_info = json.loads(repo_notebook_git_info_response.text)
    #     relative_path = repo_notebook_git_info["relative_path"]
    #     absolute_path = repo_notebook_git_info["absolute_path"]
    #     # Remove the relative path of the current notebook from the absolute path to obtain the
    #     # repo root path. Then, prepend `/Workspace` because repos are mounted to the
    #     # `/Workspaces` directory on the cluster filesystem
    #     return "/Workspace" + absolute_path[: -1 * len(relative_path)]
    #
    # if databricks_utils.is_in_databricks_repo_notebook():
    #     repo_root = _get_databricks_repo_root_path(databricks_utils.get_notebook_id())
    # else:
    #     # Replace with gitpython later if necessary / possible, since this is
    #     # already an MLflow dependency
    #     repo_root = (
    #         subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    #         .decode("utf-8")
    #         .rstrip("\n")
    #     )
    #
    # os.chdir(repo_root)
