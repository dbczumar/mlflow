import datetime
import logging
import time
from typing import Dict, Any

import mlflow
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.pipelines.utils.tracking import (
    get_pipeline_tracking_config,
    apply_pipeline_tracking_config,
    TrackingConfig,
)
from mlflow.projects.utils import get_databricks_env_vars
from mlflow.tracking.client import MlflowClient
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

_logger = logging.getLogger(__name__)


class RegisterStep(BaseStep):
    def __init__(self, step_config: Dict[str, Any], pipeline_root: str):
        super(RegisterStep, self).__init__(step_config, pipeline_root)
        self.status = "UNKNOWN"
        self.tracking_config = TrackingConfig.from_dict(step_config)
        self.run_end_time = None
        self.execution_duration = None
        self.num_dropped_rows = None
        self.model_url = None
        self.model_uri = None
        self.model_details = None
        self.alerts = None
        self.version = None

        if "model_name" not in self.step_config:
            raise MlflowException(
                "Missing 'model_name' config in register step config.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self.register_model_name = self.step_config.get("model_name")
        self.allow_non_validated_model = self.step_config.get("allow_non_validated_model", False)

    def _run(self, output_directory):
        try:
            run_start_time = time.time()
            run_id_path = get_step_output_path(
                pipeline_name=self.pipeline_name,
                step_name="train",
                relative_path="run_id",
            )
            with open(run_id_path, "r") as f:
                run_id = f.read()

            model_validation_path = get_step_output_path(
                pipeline_name=self.pipeline_name,
                step_name="evaluate",
                relative_path="model_validation_status",
            )
            with open(model_validation_path, "r") as f:
                model_validation = f.read()
            artifact_path = "model"
            if model_validation == "VALIDATED" or (
                model_validation == "UNKNOWN" and self.allow_non_validated_model
            ):
                apply_pipeline_tracking_config(self.tracking_config)
                # TODO: Figure out how populate self.model_url
                self.model_uri = "runs:/{run_id}/{artifact_path}".format(
                    run_id=run_id, artifact_path=artifact_path
                )
                self.model_details = mlflow.register_model(
                    model_uri=self.model_uri,
                    name=self.register_model_name,
                    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
                )
                final_status = self._get_model_version_status(self.model_details.version)
                self.version = self.model_details.version
                if final_status == ModelVersionStatus.READY:
                    self.status = "Done"
                else:
                    self.alerts = f"Model failed to register.  Status: {final_status}"
                    self.status = "Failed"
            else:
                self.alerts = (
                    "Model registration skipped.  Please check the validation "
                    "result from Evaluate step."
                )
                self.status = "Done"

        except Exception:
            self.status = "Failed"
            raise
        finally:
            self.run_end_time = time.time()
            self.execution_duration = self.run_end_time - run_start_time
            try:
                return self._build_card(output_directory)
            except Exception as e:
                # swallow exception raised during building profiles and card.
                _logger.warning(f"Build card failed: {repr(e)}")
                # When log level is DEBUG, also log the error stack trace.
                _logger.debug("", exc_info=True)

    def _get_model_version_status(self, model_version: str) -> ModelVersionStatus:
        client = MlflowClient()
        model_version_details = client.get_model_version(
            name=self.register_model_name,
            version=model_version,
        )
        return ModelVersionStatus.from_string(model_version_details.status)

    def _build_card(self, output_directory: str) -> None:
        from mlflow.pipelines.regression.v1.cards.register import RegisterCard

        # Build card
        card = RegisterCard()

        run_end_datetime = datetime.datetime.fromtimestamp(self.run_end_time)
        final_markdown = []
        if self.model_url is not None:
            final_markdown.append(f"**Model URL:** `{self.model_url}`")
        if self.model_uri is not None:
            final_markdown.append(f"**Model URI:** `{self.model_uri}`")
        if self.version is not None:
            final_markdown.append(f"**Model Version:** `{self.version}`")
        if self.alerts is not None:
            final_markdown.append(f"**Alerts:** `{self.alerts}`")
        final_markdown.append(
            f"**Last run completed at:** `{run_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}`"
        )
        final_markdown.append(f"**Execution duration (s):** `{self.execution_duration:.2f}`")
        final_markdown.append(f"**Run status:** `{self.status}`")
        card.add_markdown("REGISTER_SUMMARY", "<br>\n".join(final_markdown))
        return card

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        try:
            step_config = pipeline_config["steps"]["register"]
            step_config.update(
                get_pipeline_tracking_config(
                    pipeline_root_path=pipeline_root,
                    pipeline_config=pipeline_config,
                ).to_dict()
            )
        except KeyError:
            raise MlflowException(
                "Config for register step is not found.", error_code=INVALID_PARAMETER_VALUE
            )
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "register"

    @property
    def environment(self):
        return get_databricks_env_vars(tracking_uri=self.tracking_config.tracking_uri)
