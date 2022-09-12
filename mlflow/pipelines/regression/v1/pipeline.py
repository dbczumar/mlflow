"""
.. _mlflow-regression-pipeline:

The MLflow Regression Pipeline is an MLflow Pipeline for developing high-quality regression models.
It is designed for developing models using scikit-learn and frameworks that integrate with
scikit-learn, such as the ``XGBRegressor`` API from XGBoost. The corresponding pipeline
template repository is available at https://github.com/mlflow/mlp-regression-template, and the
:py:class:`RegressionPipeline API Documentation <RegressionPipeline>` provides instructions for
executing the pipeline and inspecting its results.

The training pipeline contains the following sequential steps:

**ingest** -> **split** -> **transform** -> **train** -> **evaluate** -> **register**

The batch scoring pipeline contains the following sequential steps:

**ingest** -> **predict**

The pipeline steps are defined as follows:

   - **ingest**
      - The **ingest** step resolves the dataset specified by the |'data' section in pipeline.yaml|
        and converts it to parquet format, leveraging the custom dataset parsing code defined in
        |steps/ingest.py| if necessary. Subsequent steps convert this dataset into training,
        validation, & test sets and use them to develop a model.

        .. note::
            If you make changes to the dataset referenced by the **ingest** step (e.g. by adding
            new records or columns), you must manually re-run the **ingest** step in order to
            use the updated dataset in the pipeline. The **ingest** step does *not* automatically
            detect changes in the dataset.

   .. _mlflow-regression-pipeline-split-step:

   - **split**
      - The **split** step splits the ingested dataset produced by the **ingest** step into
        a training dataset for model training, a validation dataset for model performance
        evaluation  & tuning, and a test dataset for model performance evaluation. The fraction
        of records allocated to each dataset is defined by the ``split_ratios`` attribute of the
        |'split' step definition in pipeline.yaml|. The **split** step also preprocesses the
        datasets using logic defined in |steps/split.py|. Subsequent steps use these datasets
        to develop a model and measure its performance.

   - **transform**
      - The **transform** step uses the training dataset created by **split** to fit
        a transformer that performs the transformations defined in |steps/transform.py|. The
        transformer is then applied to the training dataset and the validation dataset, creating
        transformed datasets that are used by subsequent steps for estimator training and model
        performance evaluation.

   .. _mlflow-regression-pipeline-train-step:

   - **train**
      - The **train** step uses the transformed training dataset output from the **transform**
        step to fit an estimator with the type and parameters defined in |steps/train.py|. The
        estimator is then joined with the fitted transformer output from the **transform** step
        to create a model pipeline. Finally, this model pipeline is evaluated against the
        transformed training and validation datasets to compute performance metrics; custom
        metrics are computed according to definitions in |steps/custom_metrics.py| and the
        |'metrics' section of pipeline.yaml|. The model pipeline and its associated parameters,
        performance metrics, and lineage information are logged to MLflow Tracking, producing
        an MLflow Run.

   - **evaluate**
      - The **evaluate** step evaluates the model pipeline created by the **train** step on
        the test dataset output from the **split** step, computing performance metrics and
        model explanations. Performance metrics are compared against configured thresholds to
        compute a ``model_validation_status``, which indicates whether or not a model is good
        enough to be registered to the MLflow Model Registry by the subsequent **register**
        step. Custom performance metrics are computed according to definitions in
        |steps/custom_metrics.py| and the |'metrics' section of pipeline.yaml|. Model
        performance thresholds are defined in the
        |'validation_criteria' section of the 'evaluate' step definition in pipeline.yaml|. Model
        performance metrics and explanations are logged to the same MLflow Tracking Run used by
        the **train** step.

   - **register**
      - The **register** step checks the ``model_validation_status`` output of the preceding
        **evaluate** step and, if model validation was successful
        (as indicated by the ``'VALIDATED'`` status), registers the model pipeline created by
        the **train** step to the MLflow Model Registry. If the ``model_validation_status`` does
        not indicate that the model passed validation checks (i.e. its value is ``'REJECTED'``),
        the model pipeline is not registered to the MLflow Model Registry.
        If the model pipeline is registered to the MLflow Model Registry, a
        ``registered_model_version`` is produced containing the model name and the model version.

            .. note::
                The model validation status check can be disabled by specifying
                ``allow_non_validated_model: true`` in the
                |'register' step definition of pipeline.yaml|, in which case the model pipeline is
                always registered with the MLflow Model Registry when the **register** step is
                executed.

    - **predict**
      - The **predict** step

.. |'split' step definition in pipeline.yaml| replace:: `'split' step definition in pipeline.yaml <https://github.com/mlflow/mlp-regression-template/blob/35f6f32c7a89dc655fbcfcf731cc1da4685a8ebb/pipeline.yaml#L36-L40>`__
.. |'register' step definition of pipeline.yaml| replace:: `'register' step definition of pipeline.yaml <https://github.com/mlflow/mlp-regression-template/blob/35f6f32c7a89dc655fbcfcf731cc1da4685a8ebb/pipeline.yaml#L57-L63>`__
.. |'data' section in pipeline.yaml| replace:: `'data' section in pipeline.yaml <https://github.com/mlflow/mlp-regression-template/blob/35f6f32c7a89dc655fbcfcf731cc1da4685a8ebb/pipeline.yaml#L15-L32>`__
.. |'metrics' section of pipeline.yaml| replace:: `'metrics' section of pipeline.yaml <https://github.com/mlflow/mlp-regression-template/blob/35f6f32c7a89dc655fbcfcf731cc1da4685a8ebb/pipeline.yaml#L64-L73>`__
.. |'validation_criteria' section of the 'evaluate' step definition in pipeline.yaml| replace:: `'validation_criteria' section of the 'evaluate' step definition in pipeline.yaml <https://github.com/mlflow/mlp-regression-template/blob/35f6f32c7a89dc655fbcfcf731cc1da4685a8ebb/pipeline.yaml#L47-L56>`__
.. |steps/ingest.py| replace:: `steps/ingest.py <https://github.com/mlflow/mlp-regression-template/blob/main/steps/ingest.py>`__
.. |steps/split.py| replace:: `steps/split.py <https://github.com/mlflow/mlp-regression-template/blob/main/steps/split.py>`__
.. |steps/train.py| replace:: `steps/train.py <https://github.com/mlflow/mlp-regression-template/blob/main/steps/train.py>`__
.. |steps/transform.py| replace:: `steps/transform.py <https://github.com/mlflow/mlp-regression-template/blob/main/steps/transform.py>`__
.. |steps/custom_metrics.py| replace:: `steps/custom_metrics.py <https://github.com/mlflow/mlp-regression-template/blob/main/steps/custom_metrics.py>`__
"""
import copy
import os
import logging

from mlflow.pipelines.regression.v1 import dag_help_strings
from mlflow.tracking.client import MlflowClient
from mlflow.pipelines.pipeline import _BasePipeline
from mlflow.pipelines.steps.ingest import IngestStep, IngestScoringStep
from mlflow.pipelines.steps.split import (
    SplitStep,
    _OUTPUT_TRAIN_FILE_NAME,
    _OUTPUT_VALIDATION_FILE_NAME,
    _OUTPUT_TEST_FILE_NAME,
)
from mlflow.pipelines.steps.transform import TransformStep
from mlflow.pipelines.steps.train import TrainStep
from mlflow.pipelines.steps.evaluate import EvaluateStep
from mlflow.pipelines.steps.predict import PredictStep, _SCORED_OUTPUT_FILE_NAME
from mlflow.pipelines.steps.register import RegisterStep, RegisteredModelVersionInfo
from mlflow.pipelines.step import BaseStep
from typing import List, Any, Optional
from mlflow.pipelines.utils import get_pipeline_root_path, get_pipeline_config
from mlflow.pipelines.utils.execution import get_or_create_base_execution_directory
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.tracking._tracking_service.utils import _use_tracking_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.file_utils import write_yaml

_logger = logging.getLogger(__name__)


@experimental
class RegressionPipeline(_BasePipeline):
    """
    A pipeline for developing high-quality regression models. The pipeline is designed for
    developing models using scikit-learn and frameworks that integrate with scikit-learn,
    such as the ``XGBRegressor`` API from XGBoost. The corresponding pipeline
    template repository is available at https://github.com/mlflow/mlp-regression-template.
    The training pipeline contains the following sequential steps:

    **ingest** -> **split** -> **transform** -> **train** -> **evaluate** -> **register**

    while the batch scoring pipeline contains this set of sequential steps:

    **ingest_scoring** -> **predict**

    .. code-block:: python
        :caption: Example

        import os
        from mlflow.pipelines import Pipeline

        os.chdir("~/mlp-regression-template")
        regression_pipeline = Pipeline()
        # Display a visual overview of the pipeline graph
        regression_pipeline.inspect()
        # Run the full pipeline
        regression_pipeline.run()
        # Display a summary of results from the 'train' step, including the trained model
        # and associated performance metrics computed from the training & validation datasets
        regression_pipeline.inspect(step="train")
        # Display a summary of results from the 'evaluate' step, including model explanations
        # computed from the validation dataset and metrics computed from the test dataset
        regression_pipeline.inspect(step="evaluate")
    """

    _PIPELINE_STEPS = (
        # Batch scoring DAG
        IngestScoringStep,
        PredictStep,
        # Training data ingestion DAG
        IngestStep,
        # Model training DAG
        SplitStep,
        TransformStep,
        TrainStep,
        EvaluateStep,
        RegisterStep,
    )

    def _get_step_classes(self) -> List[BaseStep]:
        return self._PIPELINE_STEPS

    def _get_pipeline_dag_file(self) -> str:
        import jinja2

        j2_env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(__file__)))
        pipeline_dag_template = j2_env.get_template("resources/pipeline_dag_template.html").render(
            {
                "pipeline_yaml_help": {
                    "help_string_type": "yaml",
                    "help_string": dag_help_strings.PIPELINE_YAML,
                },
                "ingest_step_help": {
                    "help_string": dag_help_strings.INGEST_STEP,
                    "help_string_type": "text",
                },
                "ingest_user_code_help": {
                    "help_string": dag_help_strings.INGEST_USER_CODE,
                    "help_string_type": "python",
                },
                "ingested_data_help": {
                    "help_string": dag_help_strings.INGESTED_DATA,
                    "help_string_type": "text",
                },
                "split_step_help": {
                    "help_string": dag_help_strings.SPLIT_STEP,
                    "help_string_type": "text",
                },
                "split_user_code_help": {
                    "help_string": dag_help_strings.SPLIT_USER_CODE,
                    "help_string_type": "python",
                },
                "training_data_help": {
                    "help_string": dag_help_strings.TRAINING_DATA,
                    "help_string_type": "text",
                },
                "validation_data_help": {
                    "help_string": dag_help_strings.VALIDATION_DATA,
                    "help_string_type": "text",
                },
                "test_data_help": {
                    "help_string": dag_help_strings.TEST_DATA,
                    "help_string_type": "text",
                },
                "transform_step_help": {
                    "help_string": dag_help_strings.TRANSFORM_STEP,
                    "help_string_type": "text",
                },
                "transform_user_code_help": {
                    "help_string": dag_help_strings.TRANSFORM_USER_CODE,
                    "help_string_type": "python",
                },
                "fitted_transformer_help": {
                    "help_string": dag_help_strings.FITTED_TRANSFORMER,
                    "help_string_type": "text",
                },
                "transformed_training_and_validation_data_help": {
                    "help_string": dag_help_strings.TRANSFORMED_TRAINING_AND_VALIDATION_DATA,
                    "help_string_type": "text",
                },
                "train_step_help": {
                    "help_string": dag_help_strings.TRAIN_STEP,
                    "help_string_type": "text",
                },
                "train_user_code_help": {
                    "help_string": dag_help_strings.TRAIN_USER_CODE,
                    "help_string_type": "python",
                },
                "fitted_model_help": {
                    "help_string": dag_help_strings.FITTED_MODEL,
                    "help_string_type": "text",
                },
                "mlflow_run_help": {
                    "help_string": dag_help_strings.MLFLOW_RUN,
                    "help_string_type": "text",
                },
                "custom_metrics_user_code_help": {
                    "help_string": dag_help_strings.CUSTOM_METRICS_USER_CODE,
                    "help_string_type": "python",
                },
                "evaluate_step_help": {
                    "help_string": dag_help_strings.EVALUATE_STEP,
                    "help_string_type": "text",
                },
                "model_validation_status_help": {
                    "help_string": dag_help_strings.MODEL_VALIDATION_STATUS,
                    "help_string_type": "text",
                },
                "register_step_help": {
                    "help_string": dag_help_strings.REGISTER_STEP,
                    "help_string_type": "text",
                },
                "registered_model_version_help": {
                    "help_string": dag_help_strings.REGISTERED_MODEL_VERSION,
                    "help_string_type": "text",
                },
                "predict_step_help": {
                    "help_string": dag_help_strings.PREDICT_STEP,
                    "help_string_type": "text",
                },
            }
        )

        pipeline_dag_file = os.path.join(
            get_or_create_base_execution_directory(self._pipeline_root_path), "pipeline_dag.html"
        )
        with open(pipeline_dag_file, "w") as f:
            f.write(pipeline_dag_template)

        return pipeline_dag_file

    # def configure(self, target_col=None, metrics=None):
    #     config = get_pipeline_config(self._pipeline_root_path)
    #
    #     if metrics is not None:
    #         metrics = copy.deepcopy(metrics)
    #         for metric_info in (metrics or []).get("custom", []):
    #             metric_function = metric_info.get("function")
    #             if metric_function is None:
    #                 raise MlflowException(
    #                     "All custom metrics definitions must include a 'function' key"
    #                 )
    #
    #             if not hasattr(metric_function, "mlp_fn_name"):
    #                 raise MlflowException(
    #                     f"Unrecognized method for MLP custom metric: '{metric_function.__name__}'."
    #                     " Please ensure that the method is defined within a notebook cell that"
    #                     " applies the %%mlp_code IPython magic."
    #                 )
    #
    #             metric_info["function"] = getattr(metric_function, "mlp_fn_name")
    #
    #         config["metrics"] = metrics
    #
    #     if target_col is not None:
    #         config["target_col"] = target_col
    #
    #     write_yaml(self._pipeline_root_path, "pipeline.yaml", config, overwrite=True, sort_keys=False)


    def run(self, step: str = None) -> None:
        """
        Runs the full pipeline or a particular pipeline step, producing outputs and displaying a
        summary of results upon completion. Step outputs are cached from previous executions, and
        steps are only re-executed if configuration or code changes have been made to the step or
        to any of its dependent steps (e.g. changes to the pipeline's ``pipeline.yaml`` file or
        ``steps/ingest.py`` file) since the previous execution.

        :param step: String name of the step to run within the regression pipeline. The step and
                     its dependencies are executed sequentially. If a step is not specified, the
                     entire pipeline is executed. Supported steps, in their order of execution, are:

                     - ``"ingest"``: resolves the dataset specified by the ``data/training`` section
                       in the pipeline's configuration file (``pipeline.yaml``) and converts it to
                       parquet format.

                     - ``"ingest_scoring"``: resolves the dataset specified by the ``data/scoring``
                       section in the pipeline's configuration file (``pipeline.yaml``) and converts
                       it to parquet format.

                     - ``"split"``: splits the ingested dataset produced by the **ingest** step into
                       a training dataset for model training, a validation dataset for model
                       performance evaluation & tuning, and a test dataset for model performance
                       evaluation.

                     - ``"transform"``: uses the training dataset created by the **split** step to
                       fit a transformer that performs the transformations defined in the
                       pipeline's ``steps/transform.py`` file. Then, applies the transformer to the
                       training dataset and the validation dataset, creating transformed datasets
                       that are used by subsequent steps for estimator training and model
                       performance evaluation.

                     - ``"train"``: uses the transformed training dataset output from the
                       **transform** step to fit an estimator with the type and parameters defined
                       in in the pipeline's ``steps/train.py`` file. Then, joins the estimator with
                       the fitted transformer output from the **transform** step to create a model
                       pipeline. Finally, evaluates the model pipeline against the transformed
                       training and validation datasets to compute performance metrics.

                     - ``"evaluate"``: evaluates the model pipeline created by the **train** step
                       on the validation and test dataset outputs from the **split** step, computing
                       performance metrics and model explanations. Then, compares performance
                       metrics against thresholds configured in the pipeline's ``pipeline.yaml``
                       configuration file to compute a ``model_validation_status``, which indicates
                       whether or not the model is good enough to be registered to the MLflow Model
                       Registry by the subsequent **register** step.

                     - ``"register"``: checks the ``model_validation_status`` output of the
                       preceding **evaluate** step and, if model validation was successful (as
                       indicated by the ``'VALIDATED'`` status), registers the model pipeline
                       created by the **train** step to the MLflow Model Registry.

                     - ``"predict"``: uses the ingested dataset for scoring created by the
                       **ingest_scoring** step and applies the specified model to the dataset.

        .. code-block:: python
            :caption: Example

            import os
            from mlflow.pipelines import Pipeline

            os.chdir("~/mlp-regression-template")
            regression_pipeline = Pipeline()
            # Run the 'train' step and preceding steps
            regression_pipeline.run(step="train")
            # Run the 'register' step and preceding steps; the 'train' step and all steps
            # prior to 'train' are not re-executed because their outputs are already cached
            regression_pipeline.run(step="register")
            # Run all pipeline steps; equivalent to running 'register'; no steps are re-executed
            # because the outputs of all steps are already cached
            regression_pipeline.run()
        """
        return super().run(step=step)


    def ingest(self, location=None, format="parquet", sql=None, custom_loader_method=None, use_cached=False):
        ingest_config = {}
        ingest_config["location"] = location
        ingest_config["sql"] = sql
        ingest_config["use_cached"] = use_cached

        if custom_loader_method is not None:
            if callable(custom_loader_method):
                if not hasattr(custom_loader_method, "mlp_fn_name"):
                    raise MlflowException(
                        f"Unrecognized method for MLP step: '{custom_loader_method.__name__}'."
                        " Please ensure that the method is defined within a notebook cell that"
                        " applies the %%mlp_code IPython magic"
                    )

                custom_loader_method = getattr(custom_loader_method, "mlp_fn_name")

            ingest_config["custom_loader_method"] = custom_loader_method

        if sql is None:
          ingest_config["format"] = format

        ingest_config = {
          key: value
          for key, value in ingest_config.items()
          if value is not None
        }

        config = get_pipeline_config(self._pipeline_root_path)
        config["steps"]["ingest"] = ingest_config
        write_yaml(self._pipeline_root_path, "pipeline.yaml", config, overwrite=True, sort_keys=False)

        self.run("ingest")

        return self.get_artifact("ingested_data")

    def split(self, split_ratios, target_col, post_split_method=None):
        split_step_config = {
            "split_ratios": split_ratios,
            "target_col": target_col,
        }

        if post_split_method is not None:
            if callable(post_split_method):
                if not hasattr(post_split_method, "mlp_fn_name"):
                    raise MlflowException(
                        f"Unrecognized method for MLP step: '{post_split_method.__name__}'."
                        " Please ensure that the method is defined within a notebook cell that"
                        " applies the %%mlp_code IPython magic"
                    )

                post_split_method = getattr(post_split_method, "mlp_fn_name")

            split_step_config["post_split_method"] = post_split_method

        config = get_pipeline_config(self._pipeline_root_path)
        config["steps"]["split"] = split_step_config
        write_yaml(self._pipeline_root_path, "pipeline.yaml", config, overwrite=True, sort_keys=False)

        self.run("split")

        return self.get_artifact("training_data"), self.get_artifact("validation_data"), self.get_artifact("test_data")


    def transform(self, transformer_method, target_col):
        if callable(transformer_method):
            if not hasattr(transformer_method, "mlp_fn_name"):
                raise MlflowException(
                    f"Unrecognized method for MLP step: '{transformer_method.__name__}'."
                    " Please ensure that the method is defined within a notebook cell that"
                    " applies the %%mlp_code IPython magic"
                )

            transformer_method = getattr(transformer_method, "mlp_fn_name")

        transform_step_config = {
            "transformer_method": transformer_method,
            "target_col": target_col,
        }

        config = get_pipeline_config(self._pipeline_root_path)
        config["steps"]["transform"] = transform_step_config
        write_yaml(self._pipeline_root_path, "pipeline.yaml", config, overwrite=True, sort_keys=False)

        self.run("transform")

        return (
            self.get_artifact("transformed_training_data"),
            self.get_artifact("transformed_validation_data"),
        )

    def train(self, estimator_method, target_col, metrics=None):
        if callable(estimator_method):
            if not hasattr(estimator_method, "mlp_fn_name"):
                raise MlflowException(
                    f"Unrecognized method for MLP step: '{estimator_method.__name__}'."
                    " Please ensure that the method is defined within a notebook cell that"
                    " applies the %%mlp_code IPython magic."
                )

            estimator_method = getattr(estimator_method, "mlp_fn_name")

        train_step_config = {
          "estimator_method": estimator_method,
          "target_col": target_col,
        }

        metrics = copy.deepcopy(metrics)
        for metric_info in (metrics or []).get("custom", []):
            metric_function = metric_info.get("function")
            if metric_function is None:
                raise MlflowException(
                    "All custom metrics definitions must include a 'function' key"
                )

            if not hasattr(metric_function, "mlp_fn_name"):
                raise MlflowException(
                    f"Unrecognized method for MLP custom metric: '{metric_function.__name__}'."
                    " Please ensure that the method is defined within a notebook cell that"
                    " applies the %%mlp_code IPython magic."
                )

            metric_info["function"] = getattr(metric_function, "mlp_fn_name")

        train_step_config["metrics"] = metrics

        config = get_pipeline_config(self._pipeline_root_path)
        config["steps"]["train"] = train_step_config
        write_yaml(self._pipeline_root_path, "pipeline.yaml", config, overwrite=True, sort_keys=False)

        self.run("train")

        return self.get_artifact("model")

    def evaluate(self, target_col, metrics=None, validation_criteria=None):
        evaluate_step_config = {
            "target_col": target_col,
        }

        metrics = copy.deepcopy(metrics)
        for metric_info in (metrics or []).get("custom", []):
            metric_function = metric_info.get("function")
            if metric_function is None:
                raise MlflowException(
                    "All custom metrics definitions must include a 'function' key"
                )

            if not hasattr(metric_function, "mlp_fn_name"):
                raise MlflowException(
                    f"Unrecognized method for MLP custom metric: '{metric_function.__name__}'."
                    " Please ensure that the method is defined within a notebook cell that"
                    " applies the %%mlp_code IPython magic."
                )

            metric_info["function"] = getattr(metric_function, "mlp_fn_name")

        if metrics is not None:
            evaluate_step_config["metrics"] = metrics
        if validation_criteria is not None:
            evaluate_step_config["validation_criteria"] = validation_criteria

        config = get_pipeline_config(self._pipeline_root_path)
        config["steps"]["evaluate"] = evaluate_step_config
        write_yaml(self._pipeline_root_path, "pipeline.yaml", config, overwrite=True, sort_keys=False)

        self.run("evaluate")

    def register(self, model_name, allow_non_validated_model=False):
        register_step_config = {
            "allow_non_validated_model": allow_non_validated_model,
            "model_name": model_name,
        }

        config = get_pipeline_config(self._pipeline_root_path)
        config["steps"]["register"] = register_step_config 
        write_yaml(self._pipeline_root_path, "pipeline.yaml", config, overwrite=True, sort_keys=False) 

        self.run("register")

        return self.get_artifact("registered_model_version")

    @experimental
    def get_artifact(self, artifact_name: str) -> Optional[Any]:
        """
        Reads an artifact from the pipeline's outputs. Supported artifact names can be obtained by
        examining the pipeline graph visualization displayed by
        :py:func:`RegressionPipeline.inspect()`.

        :param artifact_name: The string name of the artifact. Supported artifact values are:

                         - ``"ingested_data"``: returns the ingested dataset created in the
                           **ingest** step as a pandas DataFrame.

                         - ``"training_data"``: returns the training dataset created in the
                           **split** step as a pandas DataFrame.

                         - ``"validation_data"``: returns the validation dataset created in the
                           **split** step as a pandas DataFrame.

                         - ``"test_data"``: returns the test dataset created in the **split** step
                           as a pandas DataFrame.

                         - ``"ingested_scoring_data"``: returns the scoring dataset created in the
                           **ingest_scoring** step as a pandas DataFrame.

                         - ``"transformed_training_data"``: returns the transformed training dataset
                           created in the **transform** step as a pandas DataFrame.

                         - ``"transformed_validation_data"``: returns the transformed validation
                           dataset created in the **transform** step as a pandas DataFrame.

                         - ``"model"``: returns the MLflow Model pipeline created in the **train**
                           step as a :py:class:`PyFuncModel <mlflow.pyfunc.PyFuncModel>` instance.

                         - ``"transformer"``: returns the scikit-learn transformer created in the
                           **transform** step.

                         - ``"run"``: returns the
                           :py:class:`MLflow Tracking Run <mlflow.entities.Run>` containing the
                           model pipeline created in the **train** step and its associated
                           parameters, as well as performance metrics and model explanations created
                           during the **train** and **evaluate** steps.

                         - ``"registered_model_version``": returns the MLflow Model Registry
                           :py:class:`ModelVersion <mlflow.entities.model_registry.ModelVersion>`
                           created by the **register** step.

                         - ``"scored_data"``: returns the scored dataset created in the
                           **predict** step as a pandas DataFrame.

        :return: An object representation of the artifact corresponding to the specified name,
                 as described in the ``artifact_name`` parameter docstring. If the artifact is
                 not present because its corresponding step has not been executed or its output
                 cache has been cleaned, ``None`` is returned.

        .. code-block:: python
            :caption: Example

            import os
            import pandas as pd
            from mlflow.pipelines import Pipeline
            from mlflow.pyfunc import PyFuncModel

            os.chdir("~/mlp-regression-template")
            regression_pipeline = Pipeline()
            regression_pipeline.run()
            train_df: pd.DataFrame = regression_pipeline.get_artifact("training_data")
            trained_model: PyFuncModel = regression_pipeline.get_artifact("model")
        """
        import mlflow.pyfunc

        (
            ingest_scoring_step,
            predict_step,
            ingest_step,
            split_step,
            transform_step,
            train_step,
            _,
            register_step,
        ) = self._steps

        def log_artifact_not_found_warning(artifact_name, step_name):
            _logger.warning(
                f"The artifact with name '{artifact_name}' was not found."
                f" Re-run the '{step_name}' step to generate it."
            )

        def read_run_id():
            train_output_dir = get_step_output_path(self._pipeline_root_path, train_step.name, "")
            run_id_file_path = os.path.join(train_output_dir, "run_id")
            if os.path.exists(run_id_file_path):
                with open(run_id_file_path, "r") as f:
                    return f.read().strip()
            else:
                return None

        train_step_tracking_uri = train_step.tracking_config.tracking_uri
        pipeline_root_path = get_pipeline_root_path()

        def read_dataframe_from_path(artifact_path, step_name):
            import pandas as pd

            if os.path.exists(artifact_path):
                return pd.read_parquet(artifact_path)
            else:
                log_artifact_not_found_warning(artifact_name, step_name)
                return None

        artifact_path = self._get_artifact_path(
            artifact_name
        )  # path may or may not exist, error handling is in this function

        if artifact_name == "ingested_data":
            return read_dataframe_from_path(artifact_path, ingest_step.name)

        elif artifact_name == "training_data":
            return read_dataframe_from_path(artifact_path, split_step.name)

        elif artifact_name == "validation_data":
            return read_dataframe_from_path(artifact_path, split_step.name)

        elif artifact_name == "test_data":
            return read_dataframe_from_path(artifact_path, split_step.name)

        elif artifact_name == "transformed_training_data":
            return read_dataframe_from_path(artifact_path, transform_step.name)

        elif artifact_name == "transformed_validation_data":
            return read_dataframe_from_path(artifact_path, transform_step.name)

        elif artifact_name == "model":
            run_id = read_run_id()
            if run_id:
                with _use_tracking_uri(train_step_tracking_uri, pipeline_root_path):
                    return mlflow.pyfunc.load_model(f"runs:/{run_id}/{train_step.name}/model")
            else:
                log_artifact_not_found_warning("model", train_step.name)
                return None

        elif artifact_name == "transformer":
            run_id = read_run_id()
            if run_id:
                with _use_tracking_uri(train_step_tracking_uri, pipeline_root_path):
                    return mlflow.sklearn.load_model(
                        f"runs:/{run_id}/{transform_step.name}/transformer"
                    )
            else:
                log_artifact_not_found_warning("transformer", train_step.name)
                return None

        elif artifact_name == "run":
            run_id = read_run_id()
            if run_id:
                with _use_tracking_uri(train_step_tracking_uri, pipeline_root_path):
                    return MlflowClient().get_run(run_id)
            else:
                log_artifact_not_found_warning("mlflow run", train_step.name)
                return None

        elif artifact_name == "registered_model_version":
            if os.path.exists(artifact_path):
                registered_model_info = RegisteredModelVersionInfo.from_json(path=artifact_path)
                with _use_tracking_uri(train_step_tracking_uri, pipeline_root_path):
                    return MlflowClient().get_model_version(
                        name=registered_model_info.name, version=registered_model_info.version
                    )
            else:
                log_artifact_not_found_warning("registered_model_version", register_step.name)
                return None

        elif artifact_name == "ingested_scoring_data":
            return read_dataframe_from_path(artifact_path, ingest_scoring_step.name)

        elif artifact_name == "scored_data":
            return read_dataframe_from_path(artifact_path, predict_step.name)

        else:
            raise MlflowException(
                f"The artifact with name '{artifact_name}' is not supported.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _get_artifact_path(self, artifact_name: str) -> Optional[str]:
        """
        Returns a path to an artifact, which may or may not exist depending on whether or not the
        corresponding pipeline step has been run.
        """
        (
            ingest_scoring_step,
            predict_step,
            ingest_step,
            split_step,
            transform_step,
            train_step,
            _,
            register_step,
        ) = self._steps

        if artifact_name == "ingested_data":
            ingest_output_dir = get_step_output_path(self._pipeline_root_path, ingest_step.name, "")
            return os.path.join(ingest_output_dir, IngestStep._DATASET_OUTPUT_NAME)
        elif artifact_name == "training_data":
            split_output_dir = get_step_output_path(self._pipeline_root_path, split_step.name, "")
            return os.path.join(split_output_dir, _OUTPUT_TRAIN_FILE_NAME)
        elif artifact_name == "validation_data":
            split_output_dir = get_step_output_path(self._pipeline_root_path, split_step.name, "")
            return os.path.join(split_output_dir, _OUTPUT_VALIDATION_FILE_NAME)
        elif artifact_name == "test_data":
            split_output_dir = get_step_output_path(self._pipeline_root_path, split_step.name, "")
            return os.path.join(split_output_dir, _OUTPUT_TEST_FILE_NAME)
        elif artifact_name == "transformed_training_data":
            transform_output_dir = get_step_output_path(
                self._pipeline_root_path, transform_step.name, ""
            )
            return os.path.join(transform_output_dir, "transformed_training_data.parquet")
        elif artifact_name == "transformed_validation_data":
            transform_output_dir = get_step_output_path(
                self._pipeline_root_path, transform_step.name, ""
            )
            return os.path.join(transform_output_dir, "transformed_validation_data.parquet")
        elif artifact_name == "model":
            train_output_dir = get_step_output_path(self._pipeline_root_path, train_step.name, "")
            return os.path.join(train_output_dir, "model", "model.pkl")
        elif artifact_name == "transformer":
            transform_output_dir = get_step_output_path(
                self._pipeline_root_path, transform_step.name, ""
            )
            return os.path.join(transform_output_dir, "transformer.pkl")
        elif artifact_name == "run":
            train_output_dir = get_step_output_path(self._pipeline_root_path, train_step.name, "")
            return os.path.join(train_output_dir, "run_id")
        elif artifact_name == "registered_model_version":
            register_output_dir = get_step_output_path(
                self._pipeline_root_path, register_step.name, ""
            )
            return os.path.join(register_output_dir, "registered_model_version.json")
        elif artifact_name == "ingested_scoring_data":
            ingest_scoring_output_dir = get_step_output_path(
                self._pipeline_root_path, ingest_scoring_step.name, ""
            )
            return os.path.join(ingest_scoring_output_dir, IngestScoringStep._DATASET_OUTPUT_NAME)
        elif artifact_name == "scored_data":
            predict_output_dir = get_step_output_path(
                self._pipeline_root_path, predict_step.name, ""
            )
            return os.path.join(predict_output_dir, _SCORED_OUTPUT_FILE_NAME)
        else:
            raise MlflowException(
                f"The artifact with name '{artifact_name}' is not supported.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def clean(self, step: str = None) -> None:
        """
        Removes all pipeline outputs from the cache, or removes the cached outputs of a particular
        pipeline step if specified. After cached outputs are cleaned for a particular step, the
        step will be re-executed in its entirety the next time it is run.

        :param step: String name of the step to clean within the pipeline. If not specified,
                     cached outputs are removed for all pipeline steps.

        .. code-block:: python
            :caption: Example

            import os
            from mlflow.pipelines import Pipeline

            os.chdir("~/mlp-regression-template")
            regression_pipeline = Pipeline()
            # Run the 'train' step and preceding steps
            regression_pipeline.run(step="train")
            # Clean the cache of the 'transform' step
            regression_pipeline.clean(step="transform")
            # Run the 'split' step; outputs are still cached because 'split' precedes
            # 'transform' & 'train'
            regression_pipeline.run(step="split")
            # Run the 'train' step again; the 'transform' and 'train' steps are re-executed because:
            # 1. the cache of the preceding 'transform' step was cleaned and 2. 'train' occurs after
            # 'transform'. The 'ingest' and 'split' steps are not re-executed because their outputs
            # are still cached
            regression_pipeline.run(step="train")
        """
        super().clean(step=step)

    def inspect(self, step: str = None) -> None:
        """
        Displays a visual overview of the pipeline graph, or displays a summary of results from
        a particular pipeline step if specified. If the specified step has not been executed,
        nothing is displayed.

        :param step: String name of the pipeline step for which to display a results summary. If
                     unspecified, a visual overview of the pipeline graph is displayed.

        .. code-block:: python
            :caption: Example

            import os
            from mlflow.pipelines import Pipeline

            os.chdir("~/mlp-regression-template")
            regression_pipeline = Pipeline()
            # Display a visual overview of the pipeline graph.
            regression_pipeline.inspect()
            # Run the 'train' pipeline step
            regression_pipeline.run(step="train")
            # Display a summary of results from the preceding 'transform' step
            regression_pipeline.inspect(step="transform")
        """
        super().inspect(step=step)
