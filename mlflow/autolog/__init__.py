import functools

import mlflow
from mlflow.entities.run_status import RunStatus
from mlflow.utils import gorilla
from mlflow.utils.autologging_utils import wrap_patch
from mlflow.utils.autologging_utils import try_mlflow_log


AUTOLOGGING_INTEGRATIONS = {}


def apply_patch(autologging_integration, destination, name, function):

    def patched_train(inst, *args, **kwargs):
        preexisting_run = mlflow.active_run()
        failed_during_original = False
        original_result = None

        def get_original():
            original = gorilla.get_original_attribute(destination, name)

            def wrapped_original(inst, *args, **kwargs):
                try:
                    nonlocal original_result
                    original_result = original(inst, *args, **kwargs)
                    return original_result
                except Exception:
                    nonlocal failed_during_original
                    failed_during_original = True
                    raise

            return wrapped_original

        original = gorilla.get_original_attribute(destination, name)
        config = AUTOLOGGING_INTEGRATIONS[autologging_integration]
        if config.get("disable", False):
            return original(inst, *args, **kwargs)

        try:
            return function(get_original, inst, *args, **kwargs)
        except Exception as e:
            if not preexisting_run and mlflow.active_run():
                try_mlflow_log(mlflow.end_run, RunStatus.to_string(RunStatus.FAILED))

            if failed_during_original:
                raise
            
            print("Encountered unexpected error during autologging: " + str(e)) 

            if original_result:
                return original
            else:
                return original(inst, *args, **kwargs)

    wrap_patch(destination, name, patched_train)


def autologging_integration(name):

    AUTOLOGGING_INTEGRATIONS[name] = {}

    def wrapper(_autolog):

        def autolog(*args, **kwargs):
            AUTOLOGGING_INTEGRATIONS[name] = kwargs
            
            try:
                _autolog(**kwargs)
            except Exception as e:
                print("Failed to set up autologging: " + str(e))

        wrapped_autolog = functools.wraps(_autolog)(autolog)
        return wrapped_autolog

    return wrapper

class TrainingExecutionException(Exception):
    pass
