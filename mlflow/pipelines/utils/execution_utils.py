import os
from typing import List

from mlflow.utils.file_utils import chdir, read_yaml, write_yaml
from mlflow.utils.process import _exec_cmd


_MLFLOW_PIPELINES_EXECUTION_DIRECTORY_ENV_VAR = "MLFLOW_PIPELINES_EXECUTION_DIRECTORY"
_STEP_OUTPUTS_SUBDIRECTORY_NAME = "outputs"
_STEP_CONF_YAML_NAME = "conf.yaml"


def run_step(
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
    with chdir(execution_dir_path):
        _run_make(target_step)
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
    with chdir(execution_dir_path):
        _run_make("clean")


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
        step_subdir_path = os.path.join(execution_directory_path, step_name)
        step_conf_path = os.path.join(execution_directory_path, step_name, _STEP_CONF_YAML_NAME)
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
        os.path.join(execution_directory_path, step_name, _STEP_OUTPUTS_SUBDIRECTORY_NAME)
    )


def _run_make(rule_name: str) -> None:
    """
    Runs the specified rule with Make. This method assumes that a Makefile named `Makefile` exists
    in the current working directory.

    :param rule_name: The name of the Make rule to run.
    """
    _exec_cmd(["make", "-f", "Makefile", rule_name], stream_stdout=True, synchronous=True)


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
    with open(makefile_path, "w") as f:
        f.write(_MAKEFILE_FORMAT_STRING.format(prp=os.path.abspath(pipeline_root_path)))


# Makefile contents for cache-aware pipeline execution. These contents include variable placeholders
# that need to be formatted (substituted) with the pipeline root directory in order to produce a
# valid Makefile
_MAKEFILE_FORMAT_STRING = """\
split_objects = split/outputs/train.parquet split/outputs/test.parquet split/outputs/summary.html

split: $(split_objects)

%/outputs/train.parquet %/outputs/test.parquet %/outputs/summary.html: {prp}/datasets/autos.parquet
	python -c "from mlflow.pipelines.split_step import run_split_step; run_split_step(input_path='{prp}/datasets/autos.parquet', summary_path='$*/outputs/summary.html', train_output_path='$*/outputs/train.parquet', test_output_path='$*/outputs/test.parquet')"

transform_objects = transform/outputs/transformer.pkl transform/outputs/train_transformed.parquet

transform: $(transform_objects)

%/outputs/transformer.pkl %/outputs/train_transformed.parquet: {prp}/steps/transform.py {prp}/steps/transformer_config.yaml split/outputs/train.parquet transform/conf.yaml
	python -c "from mlflow.pipelines.transform_step import run_transform_step; run_transform_step(train_data_path='split/outputs/train.parquet', transformer_config_path='{prp}/steps/transformer_config.yaml', transformer_output_path='$*/outputs/transformer.pkl', transformed_data_output_path='$*/outputs/train_transformed.parquet', step_config_path='transform/conf.yaml')"

train_objects = train/outputs/pipeline.pkl train/outputs/run_id

train: $(train_objects)

%/outputs/pipeline.pkl %/outputs/run_id: {prp}/steps/train.py {prp}/steps/train_config.yaml transform/outputs/train_transformed.parquet transform/outputs/transformer.pkl train/conf.yaml
	python -c "from mlflow.pipelines.train_step import run_train_step; run_train_step(transformed_train_data_path='transform/outputs/train_transformed.parquet', train_config_path='{prp}/steps/train_config.yaml', transformer_path='transform/outputs/transformer.pkl', tracking_uri='file:/tmp/mlruns', pipeline_output_path='$*/outputs/pipeline.pkl', run_id_output_path='$*/outputs/run_id', step_config_path='train/conf.yaml')"

evaluate_objects = evaluate/outputs/worst_training_examples.parquet evaluate/outputs/metrics.json evaluate/outputs/explanations.html

evaluate: $(evaluate_objects)

%/outputs/worst_training_examples.parquet %/outputs/metrics.json %/outputs/explanations.html: train/outputs/pipeline.pkl split/outputs/train.parquet split/outputs/test.parquet train/outputs/run_id
	python -c "from mlflow.pipelines.evaluate_step import run_evaluate_step; run_evaluate_step(pipeline_path='train/outputs/pipeline.pkl', tracking_uri='file:/tmp/mlruns', run_id_path='train/outputs/run_id', train_data_path='split/outputs/train.parquet', test_data_path='split/outputs/test.parquet', explanations_output_path='$*/outputs/explanations.html', metrics_output_path='$*/outputs/metrics.json', worst_train_examples_output_path='$*/outputs/worst_training_examples.parquet')"

clean:
	rm -rf $(split_objects) $(transform_objects) $(train_objects) $(evaluate_objects)
"""
