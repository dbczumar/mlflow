import json

from mlflow.types.schema import Schema

import pandas as pd
import numpy as np

from tests.resources.data.dataset_source import TestDatasetSource
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncInputsOutputs


def test_conversion_to_json():
    source_uri = "test:/my/test/uri"
    source = TestDatasetSource._resolve(source_uri)

    dataset = PandasDataset(
        df=pd.DataFrame([1, 2, 3], columns=["Numbers"]),
        source=source,
        name="testname",
    )

    dataset_json = dataset.to_json()
    parsed_json = json.loads(dataset_json)
    assert parsed_json.keys() <= {"name", "digest", "source", "source_type", "schema", "profile"}
    assert parsed_json["name"] == dataset.name
    assert parsed_json["digest"] == dataset.digest
    assert parsed_json["source"] == dataset.source.to_json()
    assert parsed_json["source_type"] == dataset.source._get_source_type()
    assert parsed_json["profile"] == json.dumps(dataset.profile)

    schema_json = json.dumps(json.loads(parsed_json["schema"])["mlflow_colspec"])
    assert Schema.from_json(schema_json) == dataset.schema


def test_digest_property_has_expected_value():
    source_uri = "test:/my/test/uri"
    source = TestDatasetSource._resolve(source_uri)
    dataset = PandasDataset(
        df=pd.DataFrame([1, 2, 3], columns=["Numbers"]),
        source=source,
        name="testname",
    )
    assert dataset.digest == dataset._compute_digest()
    assert dataset.digest == "31ccce44"


def test_df_property():
    source_uri = "test:/my/test/uri"
    source = TestDatasetSource._resolve(source_uri)
    df = pd.DataFrame([1, 2, 3], columns=["Numbers"])
    dataset = PandasDataset(
        df=df,
        source=source,
        name="testname",
    )
    assert dataset.df.equals(df)


def test_to_pyfunc():
    source_uri = "test:/my/test/uri"
    source = TestDatasetSource._resolve(source_uri)
    df = pd.DataFrame([1, 2, 3], columns=["Numbers"])
    dataset = PandasDataset(
        df=df,
        source=source,
        name="testname",
    )
    assert isinstance(dataset.to_pyfunc(), PyFuncInputsOutputs)


def test_to_pyfunc_with_outputs():
    source_uri = "test:/my/test/uri"
    source = TestDatasetSource._resolve(source_uri)
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    dataset = PandasDataset(
        df=df,
        source=source,
        targets="c",
        name="testname",
    )
    input_outputs = dataset.to_pyfunc()
    assert isinstance(input_outputs, PyFuncInputsOutputs)
    assert input_outputs.inputs.equals(pd.DataFrame([[1, 2], [1, 2]], columns=["a", "b"]))
    assert input_outputs.outputs.equals(pd.Series([3, 3], name="c"))
