import abc
import logging

from mlflow.exceptions import MlflowException
from mlflow.pipelines.step import BaseStep, StepStatus
from mlflow.pipelines.utils import (
    get_pipeline_config,
    get_pipeline_name,
    get_pipeline_root_path,
    get_default_profile,
)
from mlflow.pipelines.utils.execution import (
    clean_execution_state,
    run_pipeline_step,
    get_step_output_path,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, INTERNAL_ERROR, BAD_REQUEST
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.databricks_utils import (
    is_in_databricks_runtime,
    is_running_in_ipython_environment,
)
from mlflow.utils.databricks_utils import is_running_in_ipython_environment
from typing import List

_logger = logging.getLogger(__name__)


class Pipeline:
    def __new__(cls, profile: str = get_default_profile()):
        """
        A factory class to create an instance of MLflow Pipeline based on the template
        defined in pipeline.yaml under the pipeline root.

        :param pipeline_root_path: String file path to the directory where the pipeline is defined.
                                   Infer automatically if not specified.
        :param profile: String defining the profile name used for constructing the pipeline config.
        """
        pipeline_root_path = get_pipeline_root_path()
        pipeline_config = get_pipeline_config(
            pipeline_root_path=pipeline_root_path, profile=profile
        )
        template = pipeline_config.get("template")
        if template is None:
            raise MlflowException(
                "The `template` property needs to be defined in the `pipeline.yaml` file."
                "For example: `template: regression/v1`",
                error_code=INVALID_PARAMETER_VALUE,
            ) from None
        template_path = template.replace("/", ".").replace("@", ".")
        class_name = f"mlflow.pipelines.{template_path}.PipelineImpl"

        try:
            pipeline_class_module = _get_class_from_string(class_name)
        except Exception as e:
            if isinstance(e, ModuleNotFoundError):
                raise MlflowException(
                    f"Failed to find Pipeline {class_name}."
                    f"Please check the correctness of the pipeline template setting: {template}",
                    error_code=INVALID_PARAMETER_VALUE,
                ) from None
            else:
                raise MlflowException(
                    f"Failed to construct Pipeline {class_name}. Error: {repr(e)}",
                    error_code=INTERNAL_ERROR,
                ) from None

        pipeline_name = get_pipeline_name(pipeline_root_path)
        _logger.info(f"Creating MLflow Pipeline '{pipeline_name}' with profile: '{profile}'")
        return pipeline_class_module(pipeline_root_path, profile)


class _BasePipeline:
    def __init__(self, pipeline_root_path: str, profile: str) -> None:
        """
        Pipeline base class.

        :param pipeline_root_path: String path to the directory under which the pipeline template
                                   such as pipeline.yaml, profiles/{profile}.yaml and
                                   steps/{step_name}.py are defined.
        :param profile: String specifying the profile name, with which
                        {pipeline_root_path}/profiles/{profile}.yaml is read and merged with
                        pipeline.yaml to generate the configuration to run the pipeline.
        """
        self._pipeline_root_path = pipeline_root_path
        self._profile = profile
        self._name = get_pipeline_name(pipeline_root_path)
        self._steps = self._resolve_pipeline_steps()

    @property
    def name(self) -> str:
        """Returns the name of the pipeline."""
        return self._name

    @property
    def profile(self) -> str:
        """
        Returns the profile under which the pipeline and its steps will execute.
        """
        return self._profile

    def run(self, step: str = None) -> None:
        """
        Runs a step in the pipeline, or the entire pipeline if a step specified.

        :param step: String name to run a step within the pipeline. The step and its dependencies
                     will be run sequentially. If a step is not specified, the entire pipeline is
                     executed.
        :return: None
        """
        # TODO Record performance here.
        # Always resolve the steps to load latest step modules before execution.
        self._steps = self._resolve_pipeline_steps()
        # Run the last step of the pipeline if no step is specified
        target_step = self._get_step(step) if step else self._steps[-1]
        last_executed_step = run_pipeline_step(
            self._pipeline_root_path,
            self.name,
            self._steps,
            target_step,
        )
        self.inspect(last_executed_step.name)

        last_executed_step_output_directory = get_step_output_path(
            self.name, last_executed_step.name, ""
        )
        last_executed_step_status = last_executed_step.get_execution_state(
            last_executed_step_output_directory
        ).status
        if last_executed_step_status != StepStatus.SUCCEEDED:
            if step is not None:
                raise MlflowException(
                    f"Failed to run step '{step}' of pipeline '{self.name}'."
                    f" An error was encountered while running step '{last_executed_step.name}'.",
                    error_code=BAD_REQUEST,
                )
            else:
                raise MlflowException(
                    f"Failed to run pipeline '{self.name}'."
                    f" An error was encountered while running step '{last_executed_step.name}'.",
                    error_code=BAD_REQUEST,
                )

    def inspect(self, step: str = None) -> None:
        """
        Displays main output from a step, or a pipeline DAG if no step is specified.

        :param step: String name to display a step output within the pipeline. If a step is not
                     specified, the DAG of the pipeline is shown instead.
        :return: None
        """
        if not step:
            from IPython.display import display, HTML

            pipeline_dag_file = self._get_pipeline_dag_file()
            if is_running_in_ipython_environment():
                display(HTML(filename=pipeline_dag_file))
            else:
                import webbrowser

                file_uri = f"file://{pipeline_dag_file}"
                webbrowser.open_new(file_uri)
                _logger.info(f"MLflow regression v1 pipeline snapshot DAG: '{file_uri}'")
        else:
            output_directory = get_step_output_path(self.name, step, "")
            self._get_step(step).inspect(output_directory)

    def clean(self, step: str = None) -> None:
        """
        Cleans the output of a step in the execution directory. If not specified, remove output
        from all steps.

        :param step: String name to clean a step within the pipeline. If a step is not specified,
                     the entire pipeline execution directory is cleaned.
        :return: None
        """
        to_clean = self._steps if not step else [self._get_step(step)]
        clean_execution_state(self.name, to_clean)

    def _get_step(self, step_name) -> BaseStep:
        """Returns a step class object from the pipeline."""
        steps = self._steps or self._resolve_pipeline_steps()
        step_names = [s.name for s in steps]
        if step_name not in step_names:
            raise MlflowException(
                f"Step {step_name} not found in pipeline. Available steps are {step_names}"
            )
        return self._steps[step_names.index(step_name)]

    @abc.abstractmethod
    def _get_step_classes(self) -> List[BaseStep]:
        """
        Returns a list of step classes defined in the pipeline.

        Concrete pipeline class should implement this method.
        """
        pass

    @abc.abstractmethod
    def _get_pipeline_dag_file(self) -> str:
        """
        Returns absolute path to the pipeline DAG representation HTML file.

        Concrete pipeline class should implement this method.
        """
        pass

    def _resolve_pipeline_steps(self) -> List[BaseStep]:
        """
        Constructs and returns all pipeline step objects from the pipeline configuration.
        """
        pipeline_config = get_pipeline_config(self._pipeline_root_path, self._profile)
        return [
            s.from_pipeline_config(pipeline_config, self._pipeline_root_path)
            for s in self._get_step_classes()
        ]
