import pytest
import mlflow
from mlflow.utils.autologging_utils import safe_patch


from unittest import mock


TESTED_INTEGRATIONS = [
    mlflow.keras.autolog,
    mlflow.sklearn.autolog,
    mlflow.xgboost.autolog,
]


@pytest.mark.parametrize("autologging_integration", TESTED_INTEGRATIONS)
def test_integration_applies_safe_patches_with_safe_functions_and_objects(autologging_integration):
    patches = []

    def safe_patch_with_mock_destination(autologging_integration, destination, function_name, function):
        class MockDest:
            pass

        def dummy_original(*args, **kwargs):
            print("BAHHH")
            return

        setattr(MockDest, function_name, dummy_original)
        patches.append((MockDest, function_name, dummy_original))

        return safe_patch(autologging_integration, MockDest, function_name, function)

    with mock.patch(autologging_integration.__module__ + ".safe_patch", wraps=safe_patch_with_mock_destination) as mock_safe_patch:
        autologging_integration()

    for clazz, fn_name, original_fn in patches:
        patch_fn = getattr(clazz, fn_name)
        patch_fn()
    # for call_args in mock_safe_patch.call_args_list:
    #     _, patched_class, patched_fn_name, _ = call_args[0]
    #     patched_fn = getattr(patched_class, patched_fn_name)
    #     print(patched_fn)
        # patched_fn()
    #     with mock
    #
    #
    #     print(patch_info)

    #
    # with mock.patch("mlflow.utils.autologging_utils.safe_patch") as mock_safe_patch:
    #     autologging_integration()
    #     mock_safe_patch.assert_called()
    #
    #
