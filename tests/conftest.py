import os

import pytest

import mlflow
from mlflow.tracking.fluent import _active_experiment_id


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
    from mlflow.utils.file_utils import path_to_local_sqlite_uri

    try:
        if "notrackingurimock" not in request.keywords:
            tracking_uri = path_to_local_sqlite_uri(os.path.join(tmpdir.strpath, "mlruns"))
            mlflow.set_tracking_uri(tracking_uri)
            global _active_experiment_id
            _active_experiment_id = None
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        yield tmpdir
    finally:
        mlflow.set_tracking_uri(None)
        global _active_experiment_id
        _active_experiment_id = None
        if "notrackingurimock" not in request.keywords:
            del os.environ["MLFLOW_TRACKING_URI"]


@pytest.fixture
def reset_active_experiment():
    yield
    mlflow.tracking.fluent._active_experiment_id = None
