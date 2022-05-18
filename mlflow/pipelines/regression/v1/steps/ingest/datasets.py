import importlib
import logging
import os
import pathlib
import sys
from abc import abstractmethod
from typing import Dict, Any, TypeVar

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, BAD_REQUEST, INTERNAL_ERROR
from mlflow.utils.file_utils import (
    TempDir,
    get_local_path_or_none,
    local_file_uri_to_path,
    write_pandas_df_as_parquet,
    read_parquet_as_pandas_df,
)
from mlflow.utils._spark_utils import _get_active_spark_session

_logger = logging.getLogger(__name__)

_DatasetType = TypeVar("_Dataset")


class _Dataset:
    """
    Base class representing an ingestable dataset.
    """

    def __init__(self, dataset_format: str):
        """
        :param dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).
        """
        self.dataset_format = dataset_format

    @abstractmethod
    def resolve_to_parquet(self, dst_path: str):
        """
        Fetches the dataset, converts it to parquet, and stores it at the specified `dst_path`.

        :param dst_path: The local filesystem path at which to store the resolved parquet dataset
                         (e.g. `<execution_directory_path>/steps/ingest/outputs/dataset.parquet`).
        """
        pass

    @classmethod
    def from_config(cls, dataset_config: Dict[str, Any], pipeline_root: str) -> _DatasetType:
        """
        Constructs a dataset instance from the specified dataset configuration
        and pipeline root path.

        :param dataset_config: Dictionary representation of the pipeline dataset configuration
                               (i.e. the `data` section of pipeline.yaml).
        :param pipeline_root: The absolute path of the associated pipeline root directory on the
                              local filesystem.
        :return: A `_Dataset` instance representing the configured dataset.
        """
        if not cls.matches_format(dataset_config.get("format")):
            raise MlflowException(
                f"Invalid format {dataset_config.get('format')} for dataset {cls}",
                error_code=INTERNAL_ERROR,
            )
        return cls._from_config(dataset_config, pipeline_root)

    @classmethod
    @abstractmethod
    def _from_config(cls, dataset_config, pipeline_root) -> _DatasetType:
        """
        Constructs a dataset instance from the specified dataset configuration
        and pipeline root path.

        :param dataset_config: Dictionary representation of the pipeline dataset configuration
                               (i.e. the `data` section of pipeline.yaml).
        :param pipeline_root: The absolute path of the associated pipeline root directory on the
                              local filesystem.
        :return: A `_Dataset` instance representing the configured dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def matches_format(dataset_format: str) -> bool:
        """
        Determines whether or not the dataset class is a compatible representation of the
        specified dataset format.

        :param dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).
        :return: `True` if the dataset class is a compatible representation of the specified
                 dataset format, `False` otherwise.
        """
        pass

    @classmethod
    def _get_required_config(cls, dataset_config: Dict[str, Any], key: str) -> Any:
        """
        Obtains the value associated with the specified dataset configuration key, first verifying
        that the key is present in the config and throwing if it is not.

        :param dataset_config: Dictionary representation of the pipeline dataset configuration
                               (i.e. the `data` section of pipeline.yaml).
        :param key: The key within the dataset configuration for which to fetch the associated
                    value.
        :return: The value associated with the specified configuration key.
        """
        try:
            return dataset_config[key]
        except KeyError:
            raise MlflowException(
                f"The `{key}` configuration key must be specified for dataset with"
                f" format '{dataset_config.get('format')}'"
            ) from None


class _LocationBasedDataset(_Dataset):
    """
    Base class representing an ingestable dataset with a configurable `location` attribute.
    """

    def __init__(self, location: str, dataset_format: str, pipeline_root: str):
        """
        :param location: The location of the dataset
                         (e.g. '/tmp/myfile.parquet', './mypath', 's3://mybucket/mypath', ...).
        :param dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).
        :param pipeline_root: The absolute path of the associated pipeline root directory on the
                              local filesystem.
        """
        super().__init__(dataset_format=dataset_format)
        self.location = _LocationBasedDataset._sanitize_local_dataset_location_if_necessary(
            dataset_location=location,
            pipeline_root=pipeline_root,
        )

    @abstractmethod
    def resolve_to_parquet(self, dst_path: str):
        pass

    @classmethod
    def _from_config(cls, dataset_config: Dict[str, Any], pipeline_root: str) -> _DatasetType:
        return cls(
            location=cls._get_required_config(dataset_config=dataset_config, key="location"),
            pipeline_root=pipeline_root,
            dataset_format=cls._get_required_config(dataset_config=dataset_config, key="format"),
        )

    @staticmethod
    def _sanitize_local_dataset_location_if_necessary(
        dataset_location: str, pipeline_root: str
    ) -> str:
        """
        Checks whether or not the specified `dataset_location` is a local filesystem location and,
        if it is, converts it to an absolute path if it is not already absolute.

        :param dataset_location: The dataset location from the pipeline dataset configuration.
        :param pipeline_root: The absolute path of the pipeline root directory on the local
                              filesystem.
        :return: The sanitized dataset location.
        """
        local_dataset_path_or_uri_or_none = get_local_path_or_none(path_or_uri=dataset_location)
        if local_dataset_path_or_uri_or_none is None:
            return dataset_location

        # If the local dataset path is a file: URI, convert it to a filesystem path
        local_dataset_path = local_file_uri_to_path(uri=local_dataset_path_or_uri_or_none)
        local_dataset_path = pathlib.Path(local_dataset_path)
        if local_dataset_path.is_absolute():
            return str(local_dataset_path)
        else:
            # Use pathlib to join the local dataset relative path with the pipeline root
            # directory to correctly handle the case where the root path is Windows-formatted
            # and the local dataset relative path is POSIX-formatted
            return str(pathlib.Path(pipeline_root) / local_dataset_path)

    @staticmethod
    @abstractmethod
    def matches_format(dataset_format: str) -> bool:
        pass


class _PandasParseableDataset(_LocationBasedDataset):
    """
    Base class representing a location-based ingestable dataset that can be parsed and converted to
    parquet using a series of Pandas DataFrame ``read_*`` and ``concat`` operations.
    """

    def resolve_to_parquet(self, dst_path: str):
        import pandas as pd

        with TempDir(chdr=True) as tmpdir:
            _logger.info("Resolving input data from '%s'", self.location)
            local_dataset_path = download_artifacts(
                artifact_uri=self.location, dst_path=tmpdir.path()
            )

            if os.path.isdir(local_dataset_path):
                # NB: Sort the file names alphanumerically to ensure a consistent
                # ordering across invocations
                data_file_paths = sorted(
                    list(pathlib.Path(local_dataset_path).glob(f"*.{self.dataset_format}"))
                )
                if len(data_file_paths) == 0:
                    raise MlflowException(
                        message=(
                            "Did not find any data files with the specified format"
                            f" '{self.dataset_format}' in the resolved data directory with path"
                            f" '{local_dataset_path}'. Directory contents:"
                            f" {os.listdir(local_dataset_path)}."
                        ),
                        error_code=INVALID_PARAMETER_VALUE,
                    )
            else:
                if not local_dataset_path.endswith(f".{self.dataset_format}"):
                    raise MlflowException(
                        message=(
                            f"Resolved data file with path '{local_dataset_path}' does not have the"
                            f" expected format '{self.dataset_format}'."
                        ),
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                data_file_paths = [local_dataset_path]

            _logger.info("Resolved input data to '%s'", local_dataset_path)
            _logger.info("Converting dataset to parquet format, if necessary")
            aggregated_dataframe = None
            for data_file_path in data_file_paths:
                data_file_as_dataframe = self._load_file_as_pandas_dataframe(
                    local_data_file_path=data_file_path,
                )
                aggregated_dataframe = (
                    pd.concat([aggregated_dataframe, data_file_as_dataframe])
                    if aggregated_dataframe is not None
                    else data_file_as_dataframe
                )

            write_pandas_df_as_parquet(df=aggregated_dataframe, data_parquet_path=dst_path)

    @abstractmethod
    def _load_file_as_pandas_dataframe(self, local_data_file_path: str):
        """
        Loads the specified file as a Pandas DataFrame.

        :param local_data_file_path: The local filesystem path of the file to load.
        :return: A Pandas DataFrame representation of the specified file.
        """
        pass

    @staticmethod
    @abstractmethod
    def matches_format(dataset_format: str) -> bool:
        pass


class ParquetDataset(_PandasParseableDataset):
    """
    Representation of a dataset in parquet format with files having the `.parquet` extension.
    """

    def _load_file_as_pandas_dataframe(self, local_data_file_path: str):
        return read_parquet_as_pandas_df(data_parquet_path=local_data_file_path)

    @staticmethod
    def matches_format(dataset_format: str) -> bool:
        return dataset_format == "parquet"


class CustomDataset(_PandasParseableDataset):
    """
    Representation of a location-based dataset with files containing a consistent, custom
    extension (e.g. 'csv', 'csv.gz', 'json', ...), as well as a custom function used to load
    and convert the dataset to parquet format.
    """

    def __init__(
        self, location: str, dataset_format: str, custom_loader_method: str, pipeline_root: str
    ):
        """
        :param location: The location of the dataset
                         (e.g. '/tmp/myfile.parquet', './mypath', 's3://mybucket/mypath', ...).
        :param dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).
        :param custom_loader_method: The fully qualified name of the custom loader method used to
                                     load and convert the dataset to parquet format, e.g.
                                     `steps.ingest.load_file_as_dataframe`.
        :param pipeline_root: The absolute path of the associated pipeline root directory on the
                              local filesystem.
        """
        super().__init__(
            location=location, dataset_format=dataset_format, pipeline_root=pipeline_root
        )
        self.pipeline_root = pipeline_root
        (
            self.custom_loader_module_name,
            self.custom_loader_method_name,
        ) = custom_loader_method.rsplit(".", 1)

    def _load_file_as_pandas_dataframe(self, local_data_file_path: str):
        try:
            sys.path.append(self.pipeline_root)
            custom_loader_method = getattr(
                importlib.import_module(self.custom_loader_module_name),
                self.custom_loader_method_name,
            )
        except Exception as e:
            raise MlflowException(
                message=(
                    "Failed to import custom dataset loader function"
                    f" '{self.custom_loader_module_name}.{self.custom_loader_method_name}' for"
                    f" ingesting dataset with format '{self.dataset_format}'. Exception: {e}",
                ),
                error_code=BAD_REQUEST,
            ) from None

        try:
            return custom_loader_method(local_data_file_path, self.dataset_format)
        except NotImplementedError:
            raise MlflowException(
                message=(
                    f"Unable to load data file at path '{local_data_file_path}' with format"
                    f" '{self.dataset_format}' using custom loader method"
                    f" '{custom_loader_method.__name__}' because it is not"
                    " supported. Please update the custom loader method to support this"
                    " format."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            ) from None
        except Exception as e:
            raise MlflowException(
                message=(
                    f"Unable to load data file at path '{local_data_file_path}' with format"
                    f" '{self.dataset_format}' using custom loader method"
                    f" '{custom_loader_method.__name__}'. Encountered exception: {e}"
                ),
                error_code=BAD_REQUEST,
            ) from None

    @classmethod
    def _from_config(cls, dataset_config: Dict[str, Any], pipeline_root: str) -> _DatasetType:
        return cls(
            location=cls._get_required_config(dataset_config=dataset_config, key="location"),
            dataset_format=cls._get_required_config(dataset_config=dataset_config, key="format"),
            custom_loader_method=cls._get_required_config(
                dataset_config=dataset_config, key="custom_loader_method"
            ),
            pipeline_root=pipeline_root,
        )

    @staticmethod
    def matches_format(dataset_format: str) -> bool:
        return dataset_format is not None


class _SparkDatasetMixin:
    """
    Mixin class providing Spark-related utilities for Datasets that use Spark for resolution
    and conversion to parquet format.
    """

    def _get_spark_session(self):
        """
        Obtains the active Spark session, throwing if a session does not exist.

        :return: The active Spark session.
        """
        try:
            return _get_active_spark_session()
        except Exception as e:
            raise MlflowException(
                message=(
                    f"Encountered an error while searching for an active Spark session to"
                    f" load the dataset with format '{self.dataset_format}'. Please create a"
                    f" Spark session and try again."
                ),
                error_code=BAD_REQUEST,
            ) from e


class DeltaTableDataset(_SparkDatasetMixin, _LocationBasedDataset):
    """
    Representation of a dataset in delta format with files having the `.delta` extension.
    """

    def resolve_to_parquet(self, dst_path: str):
        spark_session = self._get_spark_session()
        spark_df = spark_session.read.format("delta").load(self.location)
        if len(spark_df.columns) > 0:
            # Sort across columns in hopes of achieving a consistent ordering at ingest
            spark_df = spark_df.orderBy(spark_df.columns)
        spark_df.write.parquet(dst_path)

    @staticmethod
    def matches_format(dataset_format: str) -> bool:
        return dataset_format == "delta"


class SparkSqlDataset(_SparkDatasetMixin, _Dataset):
    """
    Representation of a Spark SQL dataset defined by a Spark SQL query string
    (e.g. `SELECT * FROM my_spark_table`).
    """

    def __init__(self, sql: str, dataset_format: str):
        """
        :param location: The Spark SQL query string that defines the dataset
                         (e.g. 'SELECT * FROM my_spark_table').
        :param dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).
        """
        super().__init__(dataset_format=dataset_format)
        self.sql = sql

    def resolve_to_parquet(self, dst_path: str):
        spark_session = self._get_spark_session()
        spark_df = spark_session.sql(self.sql)
        spark_df.write.parquet(dst_path)

    @classmethod
    def _from_config(cls, dataset_config: Dict[str, Any], pipeline_root: str) -> _DatasetType:
        return cls(
            sql=cls._get_required_config(dataset_config=dataset_config, key="sql"),
            dataset_format=cls._get_required_config(dataset_config=dataset_config, key="format"),
        )

    @staticmethod
    def matches_format(dataset_format: str) -> bool:
        return dataset_format == "spark_sql"
