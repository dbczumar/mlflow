from typing import Optional

from mlflow.tracking.fluent import autolog as _autolog


def autolog(
    disable: bool = False,
    silent: bool = False,
    exclude_flavors: Optional[list[str]] = None,
) -> None:
    return _autolog(
        log_traces=True,
        disable=disable,
        exclude_flavors=exclude_flavors,
        log_input_examples=False,
        log_model_signatures=False,
        log_models=False,
        log_datasets=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False,
        extra_tags=None,
    )
