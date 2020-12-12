# pylint: disable=unused-argument

import importlib
import pytest
from unittest import mock

import mlflow
from mlflow.utils.autologging_utils import (
    safe_patch, get_autologging_config, autologging_is_disabled,
)


pytestmark = pytest.mark.large


AUTOLOGGING_INTEGRATIONS_TO_TEST = {
    mlflow.sklearn: "sklearn",
    mlflow.keras: "keras",
    mlflow.xgboost: "xgboost",
}

for library_module in AUTOLOGGING_INTEGRATIONS_TO_TEST.values():
    importlib.import_module(library_module)


@pytest.fixture(autouse=True)
def disable_autologging_at_test_end():
    yield
    mlflow.autolog(disable=True)


def test_autologging_integrations_expose_configs_and_support_disablement():
    mlflow.autolog()

    for integration in AUTOLOGGING_INTEGRATIONS_TO_TEST:
        assert not autologging_is_disabled(integration.FLAVOR_NAME)
        assert not get_autologging_config(integration.FLAVOR_NAME, "disable", True)

        integration.autolog(disable=True)

        assert autologging_is_disabled(integration.FLAVOR_NAME)
        assert get_autologging_config(integration.FLAVOR_NAME, "disable", False)


def test_autologging_integrations_use_safe_patch_for_monkey_patching():
    from mlflow.utils import gorilla

    for integration in AUTOLOGGING_INTEGRATIONS_TO_TEST:
        with mock.patch("mlflow.utils.gorilla.apply", wraps=gorilla.apply) as gorilla_mock,\
                mock.patch(integration.__name__ + ".safe_patch", wraps=safe_patch) as safe_patch_mock:
            integration.autolog(disable=False)
            assert safe_patch_mock.call_count > 0
            assert safe_patch_mock.call_count == gorilla_mock.call_count
