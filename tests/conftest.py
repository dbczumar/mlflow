import os

import pytest

import mlflow
from mlflow.utils.file_utils import path_to_local_sqlite_uri


@pytest.fixture
def reset_mock():
    cache = []

    def set_mock(obj, attr, mock):
        cache.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, mock)

    yield set_mock

    for obj, attr, value in cache:
        setattr(obj, attr, value)
    cache[:] = []


@pytest.fixture(autouse=True)
def tracking_uri_mock(tmpdir, request):
    try:
        if "notrackingurimock" not in request.keywords:
            tracking_uri = path_to_local_sqlite_uri(os.path.join(tmpdir.strpath, "mlruns"))
            mlflow.set_tracking_uri(tracking_uri)
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        yield tmpdir
    finally:
        mlflow.set_tracking_uri(None)
        if "notrackingurimock" not in request.keywords:
            del os.environ["MLFLOW_TRACKING_URI"]


@pytest.fixture(autouse=True, scope="session")
def autologging_integration_test_mode():
    try:
        prev_env_var_value = os.environ.pop("MLFLOW_AUTOLOGGING_TESTING", None)
        os.environ["MLFLOW_AUTOLOGGING_TESTING"] = "true"
        yield
    finally:
        if prev_env_var_value:
            os.environ["MLFLOW_AUTOLOGGING_TESTING"] = prev_env_var_value
        else:
            del os.environ["MLFLOW_AUTOLOGGING_TESTING"]
