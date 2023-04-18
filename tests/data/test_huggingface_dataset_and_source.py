import json
import os

import datasets
import pandas as pd
import pytest

import mlflow.data
import mlflow.data.huggingface_dataset
from mlflow.data.dataset_source_registry import get_dataset_source_from_json
from mlflow.data.huggingface_dataset import HuggingFaceDataset
from mlflow.data.huggingface_dataset_source import HuggingFaceDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema


def test_from_huggingface_dataset_constructs_expected_dataset():
    ds = datasets.load_dataset("rotten_tomatoes", split="train")
    mlflow_ds = mlflow.data.from_huggingface(ds, path="rotten_tomatoes")

    assert isinstance(mlflow_ds, HuggingFaceDataset)
    assert mlflow_ds.ds == ds
    assert mlflow_ds.schema == _infer_schema(ds.to_pandas())
    assert mlflow_ds.profile == {
        "num_rows": ds.num_rows,
        "dataset_size": ds.dataset_size,
        "size_in_bytes": ds.size_in_bytes,
    }

    assert isinstance(mlflow_ds.source, HuggingFaceDatasetSource)
    reloaded_ds = mlflow_ds.source.load()
    assert reloaded_ds.builder_name == ds.builder_name
    assert reloaded_ds.config_name == ds.config_name
    assert reloaded_ds.split == ds.split == "train"
    assert reloaded_ds.num_rows == ds.num_rows

    reloaded_mlflow_ds = mlflow.data.from_huggingface(reloaded_ds, path="rotten_tomatoes")
    assert reloaded_mlflow_ds.digest == mlflow_ds.digest


def test_from_huggingface_dataset_constructs_expected_dataset_with_revision():
    ds_new = datasets.load_dataset(
        "rotten_tomatoes", split="train", revision="c33cbf965006dba64f134f7bef69c53d5d0d285d"
    )
    # NB: Newer versions of the rotten tomatoes dataset define a text-classification task template
    assert ds_new.task_templates

    ds_old = datasets.load_dataset(
        "rotten_tomatoes", split="train", revision="8ca2693371541a5ba2b23981de4222be3bef149f"
    )
    # NB: Older versions of the rotten tomatoes dataset don't define any task templates
    assert not ds_old.task_templates

    mlflow_ds_new = mlflow.data.from_huggingface(
        ds_new, path="rotten_tomatoes", revision="c33cbf965006dba64f134f7bef69c53d5d0d285d"
    )
    reloaded_ds_new = mlflow_ds_new.source.load()
    assert reloaded_ds_new.task_templates

    mlflow_ds_old = mlflow.data.from_huggingface(
        ds_old, path="rotten_tomatoes", revision="8ca2693371541a5ba2b23981de4222be3bef149f"
    )
    reloaded_ds_old = mlflow_ds_old.source.load()
    assert not reloaded_ds_old.task_templates


def test_from_huggingface_dataset_constructs_expected_dataset_with_task():
    ds_text_class = datasets.load_dataset(
        "rotten_tomatoes", split="train", task="text-classification"
    )
    # NB: Specifying the 'text-classification' task transforms the "label" column of the
    # dataset features and renames it to "labels"
    assert "labels" in ds_text_class.features

    ds_no_text_class = datasets.load_dataset("rotten_tomatoes", split="train")
    assert "labels" not in ds_no_text_class.features

    mlflow_ds_text_class = mlflow.data.from_huggingface(
        ds_text_class, path="rotten_tomatoes", task="text-classification"
    )
    reloaded_ds_text_class = mlflow_ds_text_class.source.load()
    assert "labels" in reloaded_ds_text_class.features

    mlflow_ds_no_text_class = mlflow.data.from_huggingface(ds_text_class, path="rotten_tomatoes")
    reloaded_ds_no_text_class = mlflow_ds_no_text_class.source.load()
    assert "labels" not in reloaded_ds_no_text_class.features


def test_from_huggingface_dataset_constructs_expected_dataset_with_data_files():
    data_files = {"validation": "en/c4-validation.00001-of-00008.json.gz"}
    ds = datasets.load_dataset("allenai/c4", data_files=data_files, split="validation")
    mlflow_ds = mlflow.data.from_huggingface(ds, path="allenai/c4", data_files=data_files)

    assert isinstance(mlflow_ds, HuggingFaceDataset)
    assert mlflow_ds.ds == ds
    assert mlflow_ds.schema == _infer_schema(ds.to_pandas())
    assert mlflow_ds.profile == {
        "num_rows": ds.num_rows,
        "dataset_size": ds.dataset_size,
        "size_in_bytes": ds.size_in_bytes,
    }

    assert isinstance(mlflow_ds.source, HuggingFaceDatasetSource)
    reloaded_ds = mlflow_ds.source.load()
    assert reloaded_ds.builder_name == ds.builder_name
    assert reloaded_ds.config_name == ds.config_name
    assert reloaded_ds.split == ds.split == "validation"
    assert reloaded_ds.num_rows == ds.num_rows

    reloaded_mlflow_ds = mlflow.data.from_huggingface(
        reloaded_ds, path="allenai/c4", data_files=data_files
    )
    assert reloaded_mlflow_ds.digest == mlflow_ds.digest


def test_from_huggingface_dataset_constructs_expected_dataset_with_data_dir(tmp_path):
    df = pd.DataFrame.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
    data_dir = "data"
    os.makedirs(tmp_path / data_dir)
    df.to_csv(tmp_path / data_dir / "my_data.csv")
    ds = datasets.load_dataset(str(tmp_path), data_dir=data_dir, name="csv", split="train")
    mlflow_ds = mlflow.data.from_huggingface(ds, path=str(tmp_path), data_dir=data_dir)

    assert mlflow_ds.ds == ds
    assert mlflow_ds.schema == _infer_schema(ds.to_pandas())
    assert mlflow_ds.profile == {
        "num_rows": ds.num_rows,
        "dataset_size": ds.dataset_size,
        "size_in_bytes": ds.size_in_bytes,
    }

    assert isinstance(mlflow_ds.source, HuggingFaceDatasetSource)
    reloaded_ds = mlflow_ds.source.load()
    assert reloaded_ds.builder_name == ds.builder_name
    assert reloaded_ds.config_name == ds.config_name
    assert reloaded_ds.split == ds.split == "train"
    assert reloaded_ds.num_rows == ds.num_rows

    reloaded_mlflow_ds = mlflow.data.from_huggingface(
        reloaded_ds, path=str(tmp_path), data_dir=data_dir
    )
    assert reloaded_mlflow_ds.digest == mlflow_ds.digest


def test_from_huggingface_dataset_respects_user_specified_name_and_digest():
    ds = datasets.load_dataset("rotten_tomatoes", split="train")
    mlflow_ds = mlflow.data.from_huggingface(
        ds, path="rotten_tomatoes", name="myname", digest="mydigest"
    )
    assert mlflow_ds.name == "myname"
    assert mlflow_ds.digest == "mydigest"


def test_from_huggingface_dataset_digest_is_consistent_for_large_ordered_datasets(tmp_path):
    assert (
        mlflow.data.huggingface_dataset._MAX_ROWS_FOR_DIGEST_COMPUTATION_AND_SCHEMA_INFERENCE
        < 200000
    )

    df = pd.DataFrame.from_dict(
        {
            "a": list(range(200000)),
            "b": list(range(200000)),
        }
    )
    data_dir = "data"
    os.makedirs(tmp_path / data_dir)
    df.to_csv(tmp_path / data_dir / "my_data.csv")

    ds = datasets.load_dataset(str(tmp_path), data_dir=data_dir, name="csv", split="train")
    mlflow_ds = mlflow.data.from_huggingface(ds, path=str(tmp_path), data_dir=data_dir)
    assert mlflow_ds.digest == "1dda4ce8"


def test_from_huggingface_dataset_throws_for_dataset_dict():
    ds = datasets.load_dataset("rotten_tomatoes")
    assert isinstance(ds, datasets.DatasetDict)

    with pytest.raises(
        MlflowException, match="must be an instance of `datasets.Dataset`.*DatasetDict"
    ):
        mlflow.data.from_huggingface(ds, path="rotten_tomatoes")


def test_dataset_conversion_to_json():
    ds = datasets.load_dataset("rotten_tomatoes", split="train")
    mlflow_ds = mlflow.data.from_huggingface(ds, path="rotten_tomatoes")

    dataset_json = mlflow_ds.to_json()
    parsed_json = json.loads(dataset_json)
    assert parsed_json.keys() <= {"name", "digest", "source", "source_type", "schema", "profile"}
    assert parsed_json["name"] == mlflow_ds.name
    assert parsed_json["digest"] == mlflow_ds.digest
    assert parsed_json["source"] == mlflow_ds.source.to_json()
    assert parsed_json["source_type"] == mlflow_ds.source._get_source_type()
    assert parsed_json["profile"] == json.dumps(mlflow_ds.profile)

    schema_json = json.dumps(json.loads(parsed_json["schema"])["mlflow_colspec"])
    assert Schema.from_json(schema_json) == mlflow_ds.schema


def test_dataset_source_conversion_to_json():
    ds = datasets.load_dataset(
        "rotten_tomatoes",
        split="train",
        revision="c33cbf965006dba64f134f7bef69c53d5d0d285d",
        task="text-classification",
    )
    mlflow_ds = mlflow.data.from_huggingface(
        ds,
        path="rotten_tomatoes",
        revision="c33cbf965006dba64f134f7bef69c53d5d0d285d",
        task="text-classification",
    )
    source = mlflow_ds.source

    source_json = source.to_json()
    parsed_source = json.loads(source_json)
    assert parsed_source["revision"] == "c33cbf965006dba64f134f7bef69c53d5d0d285d"
    assert parsed_source["task"] == "text-classification"
    assert parsed_source["split"] == "train"
    assert parsed_source["config_name"] == "default"
    assert parsed_source["path"] == "rotten_tomatoes"
    assert not parsed_source["data_dir"]
    assert not parsed_source["data_files"]

    reloaded_source = HuggingFaceDatasetSource.from_json(source_json)
    assert json.loads(reloaded_source.to_json()) == parsed_source

    reloaded_source = get_dataset_source_from_json(
        source_json, source_type=source._get_source_type()
    )
    assert isinstance(reloaded_source, HuggingFaceDatasetSource)
    assert type(source) == type(reloaded_source)
    assert reloaded_source.uri == source.uri
