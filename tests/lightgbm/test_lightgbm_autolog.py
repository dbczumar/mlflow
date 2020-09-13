import os
import json
import pytest
import yaml
import numpy as np
import pandas as pd
from sklearn import datasets
import lightgbm as lgb
import matplotlib as mpl

import mlflow
import mlflow.lightgbm
from mlflow.models import Model
from mlflow.models.utils import _read_example

mpl.use("Agg")


def get_latest_run():
    client = mlflow.tracking.MlflowClient()
    return client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


@pytest.fixture(scope="session")
def bst_params():
    return {
        "objective": "multiclass",
        "num_class": 3,
    }


@pytest.fixture(scope="session")
def train_set():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target
    # set free_raw_data False to use raw data later.
    return lgb.Dataset(X, y, free_raw_data=False)


@pytest.mark.large
def test_lgb_autolog_ends_auto_created_run(bst_params, train_set):
    mlflow.lightgbm.autolog()
    lgb.train(bst_params, train_set, num_boost_round=1)
    assert mlflow.active_run() is None


@pytest.mark.large
def test_lgb_autolog_persists_manually_created_run(bst_params, train_set):
    mlflow.lightgbm.autolog()
    with mlflow.start_run() as run:
        lgb.train(bst_params, train_set, num_boost_round=1)
        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


@pytest.mark.large
def test_lgb_autolog_logs_default_params(bst_params, train_set):
    mlflow.lightgbm.autolog()
    lgb.train(bst_params, train_set)
    run = get_latest_run()
    params = run.data.params

    expected_params = {
        "num_boost_round": 100,
        "feature_name": "auto",
        "categorical_feature": "auto",
        "verbose_eval": True,
        "keep_training_booster": False,
    }
    expected_params.update(bst_params)

    for key, val in expected_params.items():
        assert key in params
        assert params[key] == str(val)

    unlogged_params = [
        "params",
        "train_set",
        "valid_sets",
        "valid_names",
        "fobj",
        "feval",
        "init_model",
        "evals_result",
        "learning_rates",
        "callbacks",
    ]

    for param in unlogged_params:
        assert param not in params


@pytest.mark.large
def test_lgb_autolog_logs_specified_params(bst_params, train_set):
    mlflow.lightgbm.autolog()
    expected_params = {
        "num_boost_round": 10,
        "early_stopping_rounds": 5,
        "verbose_eval": False,
    }
    lgb.train(bst_params, train_set, valid_sets=[train_set], **expected_params)
    run = get_latest_run()
    params = run.data.params

    expected_params.update(bst_params)

    for key, val in expected_params.items():
        assert key in params
        assert params[key] == str(val)

    unlogged_params = [
        "params",
        "train_set",
        "valid_sets",
        "valid_names",
        "fobj",
        "feval",
        "init_model",
        "evals_result",
        "learning_rates",
        "callbacks",
    ]

    for param in unlogged_params:
        assert param not in params


@pytest.mark.large
def test_lgb_autolog_logs_metrics_with_validation_data(bst_params, train_set):
    mlflow.lightgbm.autolog()
    evals_result = {}
    lgb.train(
        bst_params,
        train_set,
        num_boost_round=10,
        valid_sets=[train_set],
        valid_names=["train"],
        evals_result=evals_result,
    )
    run = get_latest_run()
    data = run.data
    client = mlflow.tracking.MlflowClient()
    metric_key = "train-multi_logloss"
    metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
    assert metric_key in data.metrics
    assert len(metric_history) == 10
    assert metric_history == evals_result["train"]["multi_logloss"]


@pytest.mark.large
def test_lgb_autolog_logs_metrics_with_multi_validation_data(bst_params, train_set):
    mlflow.lightgbm.autolog()
    evals_result = {}
    # If we use [train_set, train_set] here, LightGBM ignores the first dataset.
    # To avoid that, create a new Dataset object.
    valid_sets = [train_set, lgb.Dataset(train_set.data)]
    valid_names = ["train", "valid"]
    lgb.train(
        bst_params,
        train_set,
        num_boost_round=10,
        valid_sets=valid_sets,
        valid_names=valid_names,
        evals_result=evals_result,
    )
    run = get_latest_run()
    data = run.data
    client = mlflow.tracking.MlflowClient()
    for valid_name in valid_names:
        metric_key = "{}-multi_logloss".format(valid_name)
        metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
        assert metric_key in data.metrics
        assert len(metric_history) == 10
        assert metric_history == evals_result[valid_name]["multi_logloss"]


@pytest.mark.large
def test_lgb_autolog_logs_metrics_with_multi_metrics(bst_params, train_set):
    mlflow.lightgbm.autolog()
    evals_result = {}
    params = {"metric": ["multi_error", "multi_logloss"]}
    params.update(bst_params)
    valid_sets = [train_set]
    valid_names = ["train"]
    lgb.train(
        params,
        train_set,
        num_boost_round=10,
        valid_sets=valid_sets,
        valid_names=valid_names,
        evals_result=evals_result,
    )
    run = get_latest_run()
    data = run.data
    client = mlflow.tracking.MlflowClient()
    for metric_name in params["metric"]:
        metric_key = "{}-{}".format(valid_names[0], metric_name)
        metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
        assert metric_key in data.metrics
        assert len(metric_history) == 10
        assert metric_history == evals_result["train"][metric_name]


@pytest.mark.large
def test_lgb_autolog_logs_metrics_with_multi_validation_data_and_metrics(bst_params, train_set):
    mlflow.lightgbm.autolog()
    evals_result = {}
    params = {"metric": ["multi_error", "multi_logloss"]}
    params.update(bst_params)
    valid_sets = [train_set, lgb.Dataset(train_set.data)]
    valid_names = ["train", "valid"]
    lgb.train(
        params,
        train_set,
        num_boost_round=10,
        valid_sets=valid_sets,
        valid_names=valid_names,
        evals_result=evals_result,
    )
    run = get_latest_run()
    data = run.data
    client = mlflow.tracking.MlflowClient()
    for valid_name in valid_names:
        for metric_name in params["metric"]:
            metric_key = "{}-{}".format(valid_name, metric_name)
            metric_history = [
                x.value for x in client.get_metric_history(run.info.run_id, metric_key)
            ]
            assert metric_key in data.metrics
            assert len(metric_history) == 10
            assert metric_history == evals_result[valid_name][metric_name]


@pytest.mark.large
def test_lgb_autolog_logs_metrics_with_early_stopping(bst_params, train_set):
    mlflow.lightgbm.autolog()
    evals_result = {}
    params = {"metric": ["multi_error", "multi_logloss"]}
    params.update(bst_params)
    valid_sets = [train_set, lgb.Dataset(train_set.data)]
    valid_names = ["train", "valid"]
    model = lgb.train(
        params,
        train_set,
        num_boost_round=10,
        early_stopping_rounds=5,
        valid_sets=valid_sets,
        valid_names=valid_names,
        evals_result=evals_result,
    )
    run = get_latest_run()
    data = run.data
    client = mlflow.tracking.MlflowClient()
    assert "best_iteration" in data.metrics
    assert int(data.metrics["best_iteration"]) == model.best_iteration
    assert "stopped_iteration" in data.metrics
    assert int(data.metrics["stopped_iteration"]) == len(evals_result["train"]["multi_logloss"])

    for valid_name in valid_names:
        for metric_name in params["metric"]:
            metric_key = "{}-{}".format(valid_name, metric_name)
            metric_history = [
                x.value for x in client.get_metric_history(run.info.run_id, metric_key)
            ]
            assert metric_key in data.metrics

            best_metrics = evals_result[valid_name][metric_name][model.best_iteration - 1]
            assert metric_history == evals_result[valid_name][metric_name] + [best_metrics]


@pytest.mark.large
def test_lgb_autolog_logs_feature_importance(bst_params, train_set):
    mlflow.lightgbm.autolog()
    model = lgb.train(bst_params, train_set, num_boost_round=10)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace("file://", "")
    client = mlflow.tracking.MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id)]

    for imp_type in ["split", "gain"]:
        plot_name = "feature_importance_{}.png".format(imp_type)
        assert plot_name in artifacts

        json_name = "feature_importance_{}.json".format(imp_type)
        assert json_name in artifacts

        json_path = os.path.join(artifacts_dir, json_name)
        with open(json_path, "r") as f:
            loaded_imp = json.load(f)

        features = model.feature_name()
        importance = model.feature_importance(importance_type=imp_type)
        imp = {ft: imp for ft, imp in zip(features, importance.tolist())}

        assert loaded_imp == imp


@pytest.mark.large
def test_no_figure_is_opened_after_logging(bst_params, train_set):
    mlflow.lightgbm.autolog()
    lgb.train(bst_params, train_set, num_boost_round=10)
    assert mpl.pyplot.get_fignums() == []


@pytest.mark.large
def test_lgb_autolog_loads_model_from_artifact(bst_params, train_set):
    mlflow.lightgbm.autolog()
    model = lgb.train(bst_params, train_set, num_boost_round=10)
    run = get_latest_run()
    run_id = run.info.run_id

    loaded_model = mlflow.lightgbm.load_model("runs:/{}/model".format(run_id))
    np.testing.assert_array_almost_equal(
        model.predict(train_set.data), loaded_model.predict(train_set.data)
    )


@pytest.mark.large
def test_lgb_autolog_gets_input_example(bst_params):
    # we need to check the example input against the initial input given to train function.
    # we can't use the train_set fixture for this as it defines free_raw_data=False but this
    # feature should work even if it is True
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target
    dataset = lgb.Dataset(X, y, free_raw_data=True)

    mlflow.lightgbm.autolog()
    lgb.train(bst_params, dataset)
    run = get_latest_run()

    model_path = os.path.join(run.info.artifact_uri, "model")
    model_conf = Model.load(os.path.join(model_path, "MLmodel"))

    input_example = _read_example(model_conf, model_path)

    assert input_example.equals(X[:5])

    pyfunc_model = mlflow.pyfunc.load_model(os.path.join(run.info.artifact_uri, "model"))

    # make sure reloading the input_example and predicting on it does not error
    pyfunc_model.predict(input_example)


@pytest.mark.large
def test_lgb_autolog_infers_model_signature_correctly(bst_params):
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target
    dataset = lgb.Dataset(X, y, free_raw_data=True)

    mlflow.lightgbm.autolog()
    lgb.train(bst_params, dataset)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace("file://", "")
    client = mlflow.tracking.MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id, "model")]

    ml_model_filename = "MLmodel"
    assert str(os.path.join("model", ml_model_filename)) in artifacts
    ml_model_path = os.path.join(artifacts_dir, "model", ml_model_filename)

    data = None
    with open(ml_model_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    assert data is not None
    assert "signature" in data
    signature = data["signature"]
    assert signature is not None

    assert "inputs" in signature
    assert json.loads(signature["inputs"]) == [
        {"name": "sepal length (cm)", "type": "double"},
        {"name": "sepal width (cm)", "type": "double"},
    ]

    assert "outputs" in signature
    assert json.loads(signature["outputs"]) == [
        {"type": "double"},
        {"type": "double"},
        {"type": "double"},
    ]


@pytest.mark.large
def test_lgb_autolog_continues_logging_even_if_signature_inference_fails(tmpdir):
    tmp_csv = tmpdir.join("data.csv")
    tmp_csv.write("2,6.4,2.8,5.6,2.2\n")
    tmp_csv.write("1,5.0,2.3,3.3,1.0\n")
    tmp_csv.write("2,4.9,2.5,4.5,1.7\n")
    tmp_csv.write("0,4.9,3.1,1.5,0.1\n")
    tmp_csv.write("0,5.7,3.8,1.7,0.3\n")

    # signature and input example inference should fail here since the dataset is given
    #   as a file path
    dataset = lgb.Dataset(tmp_csv.strpath)

    bst_params = {
        "objective": "multiclass",
        "num_class": 3,
    }

    mlflow.lightgbm.autolog()
    lgb.train(bst_params, dataset)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace("file://", "")
    client = mlflow.tracking.MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id, "model")]

    ml_model_filename = "MLmodel"
    assert os.path.join("model", ml_model_filename) in artifacts
    ml_model_path = os.path.join(artifacts_dir, "model", ml_model_filename)

    data = None
    with open(ml_model_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    assert data is not None
    assert "run_id" in data
    assert "signature" not in data
