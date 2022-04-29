import os
import pathlib
from typing import List

from mlflow.utils.file_utils import read_yaml, write_yaml
from mlflow.utils.process import _exec_cmd


_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR = "MLFLOW_PIPELINES_EXECUTION_DIRECTORY"
_STEPS_SUBDIRECTORY_NAME = "steps"
_STEP_OUTPUTS_SUBDIRECTORY_NAME = "outputs"
_STEP_CONF_YAML_NAME = "conf.yaml"

#REMOVE LATER
import logging
_logger = logging.getLogger(__name__)


def run_pipeline_step(
    pipeline_root_path: str, pipeline_name: str, pipeline_steps: List[str], target_step: str
) -> str:
    """
    Runs the specified step in the specified pipeline, as well as all dependent steps.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem.
    :param pipeline_name: The name of the pipeline.
    :param pipeline_steps: A list of names of all the steps contained in the specified pipeline.
    :param target_step: The name of the step to run.
    :return: The absolute path of the step's execution outputs on the local filesystem.
    """
    execution_dir_path = _get_or_create_execution_directory(
        pipeline_root_path, pipeline_name, pipeline_steps
    )
    _write_updated_step_confs(
        pipeline_root_path=pipeline_root_path,
        pipeline_steps=pipeline_steps,
        execution_directory_path=execution_dir_path,
    )
    _run_make(execution_directory_path=execution_dir_path, rule_name=target_step)
    return _get_step_output_directory_path(
        execution_directory_path=execution_dir_path, step_name=target_step
    )


def clean_execution_state(pipeline_name: str) -> None:
    """
    Removes all execution state for the specified pipeline from the associated execution directory
    on the local filesystem. This method does *not* remove other execution results, such as content
    logged to MLflow Tracking.

    :param pipeline_name: The name of the pipeline.
    """
    execution_dir_path = _get_execution_directory_path(pipeline_name=pipeline_name)
    if os.path.exists(execution_dir_path):
        _run_make(execution_directory_path=execution_dir_path, rule_name="clean")


def _get_or_create_execution_directory(
    pipeline_root_path: str, pipeline_name: str, pipeline_steps: List[str]
) -> str:
    """
    Obtains the path of the execution directory on the local filesystem corresponding to the
    specified pipeline, creating the execution directory and its required contents if they do
    not already exist.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem.
    :param pipeline_name: The name of the pipeline.
    :param pipeline_steps: A list of names of all the steps contained in the specified pipeline.
    :return: The absolute path of the execution directory on the local filesystem for the specified
             pipeline.
    """
    execution_dir_path = _get_execution_directory_path(pipeline_name)

    os.makedirs(execution_dir_path, exist_ok=True)
    _create_makefile(pipeline_root_path, execution_dir_path)
    for step_name in pipeline_steps:
        step_output_subdir_path = _get_step_output_directory_path(execution_dir_path, step_name)
        os.makedirs(step_output_subdir_path, exist_ok=True)

    return execution_dir_path


def _write_updated_step_confs(
    pipeline_root_path: str, pipeline_steps: List[str], execution_directory_path: str
) -> None:
    """
    Compares the in-memory configuration state of the specified pipeline steps with step-specific
    internal configuration files written by prior executions. If updates are found, writes updated
    state to the corresponding files. If no updates are found, configuration state is not
    rewritten.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem.
    :param pipeline_steps: A list of names of all the steps contained in the specified pipeline.
    :param execution_directory_path: The absolute path of the execution directory on the local
                                     filesystem for the specified pipeline. Configuration files are
                                     written to step-specific subdirectories of this execution
                                     directory.
    """
    for step_name in pipeline_steps:
        step_subdir_path = os.path.join(
            execution_directory_path, _STEPS_SUBDIRECTORY_NAME, step_name
        )
        step_conf_path = os.path.join(step_subdir_path, _STEP_CONF_YAML_NAME)
        if os.path.exists(step_conf_path):
            prev_step_conf = read_yaml(root=step_subdir_path, file_name=_STEP_CONF_YAML_NAME)
        else:
            prev_step_conf = None

        # TODO: Extract the conf from the pipeline step
        step_conf = {
            "pipeline_root": pipeline_root_path,
        }

        if prev_step_conf != step_conf:
            write_yaml(
                root=step_subdir_path,
                file_name=_STEP_CONF_YAML_NAME,
                data=step_conf,
                overwrite=True,
                sort_keys=True,
            )


def _get_execution_directory_path(pipeline_name: str) -> str:
    """
    Obtains the path of the execution directory on the local filesystem corresponding to the
    specified pipeline, which may or may not exist.

    :param pipeline_name: The name of the pipeline for which to obtain the associated execution
                          directory path.
    """
    return os.path.abspath(
        os.environ.get(_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR)
        or os.path.join(os.path.expanduser("~"), ".mlflow", "pipelines", pipeline_name)
    )


def _get_step_output_directory_path(execution_directory_path: str, step_name: str) -> str:
    """
    Obtains the path of the local filesystem directory containing outputs for the specified step,
    which may or may not exist.

    :param execution_directory_path: The absolute path of the execution directory on the local
                                     filesystem for the relevant pipeline. The Makefile is created
                                     in this directory.
    :param step_name: The name of the pipeline step for which to obtain the output directory path.
    :return The absolute path of the local filesystem directory containing outputs for the specified
            step.
    """
    return os.path.abspath(
        os.path.join(
            execution_directory_path,
            _STEPS_SUBDIRECTORY_NAME,
            step_name,
            _STEP_OUTPUTS_SUBDIRECTORY_NAME,
        )
    )


def _run_make(execution_directory_path, rule_name: str) -> None:
    """
    Runs the specified pipeline rule with Make. This method assumes that a Makefile named `Makefile`
    exists in the specified execution directory.

    :param execution_directory_path: The absolute path of the execution directory on the local
                                     filesystem for the relevant pipeline. The Makefile is created
                                     in this directory.
    :param rule_name: The name of the Make rule to run.
    """
    _exec_cmd(
        ["make", "-f", "Makefile", rule_name],
        stream_stdout=True,
        synchronous=True,
        cwd=execution_directory_path,
    )


def _create_makefile(pipeline_root_path, execution_directory_path) -> None:
    """
    Creates a Makefile with a set of relevant MLflow Pipelines targets for the specified pipeline,
    overwriting the preexisting Makefile if one exists. The Makefile is created in the specified
    execution directory.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem.
    :param execution_directory_path: The absolute path of the execution directory on the local
                                     filesystem for the specified pipeline. The Makefile is created
                                     in this directory.
    """
    makefile_path = os.path.join(execution_directory_path, "Makefile")
    makefile_contents = _MAKEFILE_FORMAT_STRING.format(
        path=_MakefilePathFormat(os.path.abspath(pipeline_root_path)),
    )
    _logger.info(makefile_contents)
    with open(makefile_path, "w") as f:
        f.write(makefile_contents)


class _MakefilePathFormat:
    """
    Provides platform-agnostic path substitution for execution Makefiles, ensuring that POSIX-style
    relative paths are joined correctly with POSIX-style or Windows-style pipeline root paths.
    For example, given a format string `s = "{path:prp/my/subpath.txt}"`, invoking
    `s.format(path=_MakefilePathFormat("/my/pipeline/root/path"))` on Unix systems or
    `s.format(path=_MakefilePathFormat("C:\my\pipeline\root\path"))`` on Windows systems will
    yield "/my/pipeline/root/path/my/subpath.txt" or "C:/my/pipeline/root/path/my/subpath.txt",
    respectively.
    """

    def __init__(self, pipeline_root_path):
        """
        :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                                   filesystem.
        """
        self.pipeline_root_path = pipeline_root_path

    def __format__(self, path_spec):
        """
        :param path_spec: A substitution path spec of the form `prp/<subpath>`. This method
                          substitutes `prp/` with `<pipeline_root_path>/`.
        """
        root_path_prefix_placeholder = "prp/"
        if path_spec.startswith(root_path_prefix_placeholder):
            subpath = pathlib.PurePosixPath(path_spec.split(root_path_prefix_placeholder)[1])
            pipeline_root_posix_path = pathlib.PurePosixPath(
                pathlib.Path(self.pipeline_root_path).as_posix()
            )
            full_formatted_path = pipeline_root_posix_path / subpath
            return str(full_formatted_path)
        else:
            raise ValueError(f"Invalid Makefile string format path spec: {path_spec}")


# Makefile contents for cache-aware pipeline execution. These contents include variable placeholders
# that need to be formatted (substituted) with the pipeline root directory in order to produce a
# valid Makefile
_MAKEFILE_FORMAT_STRING = r"""\
split_objects = steps/split/outputs/train.parquet steps/split/outputs/test.parquet steps/split/outputs/summary.html

split: $(split_objects)

steps/%/outputs/train.parquet steps/%/outputs/test.parquet steps/%/outputs/summary.html: {path:prp/datasets/autos.parquet}
	python -c "from mlflow.pipelines.split_step import run_split_step; run_split_step(input_path='{path:prp/datasets/autos.parquet}', summary_path='steps/$*/outputs/summary.html', train_output_path='steps/$*/outputs/train.parquet', test_output_path='steps/$*/outputs/test.parquet')"

transform_objects = steps/transform/outputs/transformer.pkl steps/transform/outputs/train_transformed.parquet

transform: $(transform_objects)

steps/%/outputs/transformer.pkl steps/%/outputs/train_transformed.parquet: {path:prp/steps/transform.py} {path:prp/steps/transformer_config.yaml} steps/split/outputs/train.parquet steps/transform/conf.yaml
	python -c "from mlflow.pipelines.transform_step import run_transform_step; run_transform_step(train_data_path='steps/split/outputs/train.parquet', transformer_config_path='{path:prp/steps/transformer_config.yaml}', transformer_output_path='steps/$*/outputs/transformer.pkl', transformed_data_output_path='steps/$*/outputs/train_transformed.parquet', step_config_path='steps/transform/conf.yaml')"

train_objects = steps/train/outputs/pipeline.pkl steps/train/outputs/run_id

train: $(train_objects)

steps/%/outputs/pipeline.pkl steps/%/outputs/run_id: {path:prp/steps/train.py} {path:prp/steps/train_config.yaml} steps/transform/outputs/train_transformed.parquet steps/transform/outputs/transformer.pkl steps/train/conf.yaml
	python -c "from mlflow.pipelines.train_step import run_train_step; run_train_step(transformed_train_data_path='steps/transform/outputs/train_transformed.parquet', train_config_path='{path:prp/steps/train_config.yaml}', transformer_path='steps/transform/outputs/transformer.pkl', tracking_uri='file:/tmp/mlruns', pipeline_output_path='steps/$*/outputs/pipeline.pkl', run_id_output_path='steps/$*/outputs/run_id', step_config_path='steps/train/conf.yaml')"

evaluate_objects = steps/evaluate/outputs/worst_training_examples.parquet steps/evaluate/outputs/metrics.json steps/evaluate/outputs/explanations.html

evaluate: $(evaluate_objects)

steps/%/outputs/worst_training_examples.parquet steps/%/outputs/metrics.json steps/%/outputs/explanations.html: steps/train/outputs/pipeline.pkl steps/split/outputs/train.parquet steps/split/outputs/test.parquet steps/train/outputs/run_id
	python -c "from mlflow.pipelines.evaluate_step import run_evaluate_step; run_evaluate_step(pipeline_path='steps/train/outputs/pipeline.pkl', tracking_uri='file:/tmp/mlruns', run_id_path='steps/train/outputs/run_id', train_data_path='steps/split/outputs/train.parquet', test_data_path='steps/split/outputs/test.parquet', explanations_output_path='steps/$*/outputs/explanations.html', metrics_output_path='steps/$*/outputs/metrics.json', worst_train_examples_output_path='steps/$*/outputs/worst_training_examples.parquet')"

clean:
	rm -rf $(split_objects) $(transform_objects) $(train_objects) $(evaluate_objects)
"""
