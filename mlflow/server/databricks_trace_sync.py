"""
Configuration and utilities for Databricks trace synchronization.
"""

import os
from dataclasses import dataclass
from typing import Optional

import yaml

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


@dataclass
class DatabricksTraceSyncConfig:
    """
    Configuration for syncing traces from Databricks experiments.
    """

    source_experiment_names: list[str]
    destination_experiment_name: str
    sampling_rate: float = 1.0
    tracking_uri: Optional[str] = None  # Default to "databricks" if not specified


def parse_databricks_trace_sync_config(
    config_path: Optional[str],
) -> Optional[DatabricksTraceSyncConfig]:
    """
    Parse the databricks trace sync configuration from a YAML file.

    Expected YAML format:
    ```
    source_experiment_names:
      - exp1
      - exp2
    destination_experiment_name: local_exp
    sampling_rate: 0.5  # optional, default 1.0
    databricks_tracking_uri: databricks  # optional, default "databricks"
    ```

    Args:
        config_path: Path to YAML configuration file or None

    Returns:
        DatabricksTraceSyncConfig object or None if no config provided

    Raises:
        MlflowException: If the configuration file or format is invalid
    """
    if not config_path:
        return None

    try:
        # Check if file exists
        if not os.path.exists(config_path):
            raise ValueError(f"Configuration file not found: {config_path}")

        # Load YAML file
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if not isinstance(config_data, dict):
            raise ValueError("Configuration file must contain a YAML dictionary")

        # Parse required fields
        if "source_experiment_names" not in config_data:
            raise ValueError("'source_experiment_names' is required in configuration")

        source_experiments = config_data["source_experiment_names"]
        if not isinstance(source_experiments, list) or not source_experiments:
            raise ValueError("'source_experiment_names' must be a non-empty list")

        if "destination_experiment_name" not in config_data:
            raise ValueError("'destination_experiment_name' is required in configuration")

        dest_experiment = config_data["destination_experiment_name"]
        if not isinstance(dest_experiment, str) or not dest_experiment.strip():
            raise ValueError("'destination_experiment_name' must be a non-empty string")

        # Parse optional fields
        sampling_rate = config_data.get("sampling_rate", 1.0)
        if not isinstance(sampling_rate, (int, float)) or not 0.0 < sampling_rate <= 1.0:
            raise ValueError("'sampling_rate' must be a number between 0 and 1")

        databricks_tracking_uri = config_data.get("databricks_tracking_uri", "databricks")
        if not isinstance(databricks_tracking_uri, str):
            raise ValueError("'databricks_tracking_uri' must be a string")

        return DatabricksTraceSyncConfig(
            source_experiment_names=[str(exp).strip() for exp in source_experiments],
            destination_experiment_name=dest_experiment.strip(),
            sampling_rate=float(sampling_rate),
            tracking_uri=databricks_tracking_uri,
        )

    except Exception as e:
        raise MlflowException(
            f"Failed to parse databricks-trace-sync configuration: {e}",
            error_code=INVALID_PARAMETER_VALUE,
        )


def get_databricks_trace_sync_config() -> Optional[DatabricksTraceSyncConfig]:
    """
    Get the databricks trace sync configuration from environment variable.

    Returns:
        DatabricksTraceSyncConfig object or None if not configured
    """
    from mlflow.server import DATABRICKS_TRACE_SYNC_ENV_VAR

    config_path = os.getenv(DATABRICKS_TRACE_SYNC_ENV_VAR)
    return parse_databricks_trace_sync_config(config_path)
