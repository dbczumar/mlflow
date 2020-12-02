import pytest
from unittest import mock
import inspect

import mlflow

import tensorflow
import keras
import fastai
import sklearn
import xgboost
import lightgbm
import statsmodels
import mxnet.gluon
import pyspark
import pytorch_lightning

library_to_mlflow_module_without_pyspark = {
    tensorflow: "tensorflow",
    keras: "keras",
    fastai: "fastai",
    sklearn: "sklearn",
    xgboost: "xgboost",
    lightgbm: "lightgbm",
    statsmodels: "statsmodels",
    mxnet.gluon: "gluon",
    pytorch_lightning: "pytorch",
}

library_to_mlflow_module = {**library_to_mlflow_module_without_pyspark, pyspark: "spark"}


@pytest.fixture(autouse=True)
def reset_global_states():
    for integration_name in library_to_mlflow_module.keys():
        try:
            del mlflow.utils.import_hooks._post_import_hooks[integration_name.__name__]
        except Exception:  # pylint: disable=broad-except
            pass

    assert mlflow.utils.import_hooks._post_import_hooks == {}

    yield

    for integration_name in library_to_mlflow_module.keys():
        try:
            del mlflow.utils.import_hooks._post_import_hooks[integration_name.__name__]
        except Exception:  # pylint: disable=broad-except
            pass

    assert mlflow.utils.import_hooks._post_import_hooks == {}


# We are pretending the module is not already imported (in reality it is, at the top of this file),
#   and is only imported when we call wrapt.notify_module_loaded in the tests below. Normally,
#   notify_module_loaded would be called by register_post_import_hook if it sees that the module
#   is already loaded.
def only_register(callback_fn, module, overwrite):  # pylint: disable=unused-argument
    mlflow.utils.import_hooks._post_import_hooks[module] = [callback_fn]


@pytest.fixture(autouse=True)
def disable_new_import_hook_firing_if_module_already_exists():
    with mock.patch("mlflow.tracking.fluent.register_post_import_hook", wraps=only_register):
        yield


@pytest.mark.large
@pytest.mark.parametrize("library,mlflow_module", library_to_mlflow_module.items())
def test_universal_autolog_does_not_throw_if_specific_autolog_throws(library, mlflow_module):
    with mock.patch("mlflow." + mlflow_module + ".autolog") as autolog_mock:
        autolog_mock.side_effect = Exception("asdf")
        mlflow.autolog()
        if library != pyspark:
            autolog_mock.assert_not_called()
        mlflow.utils.import_hooks.notify_module_loaded(library)
        autolog_mock.assert_called_once()


@pytest.mark.large
@pytest.mark.parametrize("library,mlflow_module", library_to_mlflow_module_without_pyspark.items())
def test_universal_autolog_calls_specific_autologs_correctly(library, mlflow_module):
    integrations_with_config = [xgboost, lightgbm, statsmodels, sklearn]

    # modify the __signature__ of the mock to contain the needed parameters
    args = (
        {"log_input_examples": bool, "log_model_signatures": bool, "log_models": bool}
        if library in integrations_with_config
        else {"log_models": bool}
    )
    params = [
        inspect.Parameter(param, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=type_)
        for param, type_ in args.items()
    ]
    with mock.patch(
        "mlflow." + mlflow_module + ".autolog", wraps=getattr(mlflow, mlflow_module).autolog
    ) as autolog_mock:
        autolog_mock.__signature__ = inspect.Signature(params)

        autolog_mock.assert_not_called()

        # this should attach import hooks to each library
        mlflow.autolog(log_input_examples=True, log_model_signatures=True, log_models=True)

        autolog_mock.assert_not_called()

        mlflow.utils.import_hooks.notify_module_loaded(library)

        # after each library is imported, its corresponding autolog function should have been called
        if library in integrations_with_config:
            autolog_mock.assert_called_once_with(
                log_input_examples=True, log_model_signatures=True, log_models=True
            )
        else:
            autolog_mock.assert_called_once_with(log_models=True)


@pytest.mark.large
def test_universal_autolog_calls_pyspark_immediately():
    library = pyspark
    mlflow_module = "spark"

    with mock.patch(
        "mlflow." + mlflow_module + ".autolog", wraps=getattr(mlflow, mlflow_module).autolog
    ) as autolog_mock:
        autolog_mock.assert_not_called()

        mlflow.autolog()

        # pyspark autolog should NOT wait for pyspark to be imported
        # it should instead initialize autologging immediately
        autolog_mock.assert_called_once_with()

        # there should also be no import hook on pyspark
        mlflow.utils.import_hooks.notify_module_loaded(library)
        autolog_mock.assert_called_once_with()


@pytest.mark.large
def test_universal_autolog_attaches_pyspark_import_hook_if_pyspark_isnt_installed():
    library = pyspark
    mlflow_module = "spark"

    with mock.patch(
        "mlflow." + mlflow_module + ".autolog", wraps=getattr(mlflow, mlflow_module).autolog
    ) as autolog_mock:
        # simulate pyspark not being installed
        autolog_mock.side_effect = ImportError("no module named pyspark blahblah")

        mlflow.autolog()
        autolog_mock.assert_called_once()  # it was called once and failed

        # now the user installs pyspark
        autolog_mock.side_effect = None

        mlflow.utils.import_hooks.notify_module_loaded(library)

        # assert autolog is called again once pyspark is imported
        assert autolog_mock.call_count == 2
