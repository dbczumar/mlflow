"""
The ``mlflow.spacy`` module provides an API for logging and loading SpaCy
models. This module exports SpaCy models with the following flavors:

SpaCy (native) format
    This is the main flavor that can be loaded back into SpaCy.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

import os
import shutil

import pandas as pd
import spacy
import yaml

import mlflow.tracking
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.utils.environment import _mlflow_conda_env 
from mlflow.utils.model_utils import _get_flavor_configuration


DEFAULT_CONDA_ENV = _mlflow_conda_env(
    additional_conda_deps=[
        "spacy={}".format(spacy.__version__),
    ],
    additional_pip_deps=None,
    additional_conda_channels=None,
)


FLAVOR_NAME = "spacy"


def log_model(spacy_model, artifact_path, conda_env=None, **kwargs):
    """
    Log a SpaCy model as an MLflow artifact for the current run.

    :param spacy_model: SpaCy model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Path to a Conda environment file. If provided, this
                      defines the environment for the model. At minimum, it
                      should specify python, spacy, and mlflow with
                      appropriate versions.
    :param kwargs: kwargs to pass to ``spacy_model.to_disk()`` method.
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.spacy,
              spacy_model=spacy_model, conda_env=conda_env, **kwargs)


def save_model(spacy_model, path, conda_env=None, mlflow_model=Model(), **kwargs):
    """
    Save a SpaCy model to a path on the local file system.

    :param spacy_model: SpaCy model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Path to a Conda environment file. If provided, this
                      decribes the environment this model should be run in.
                      At minimum, it should specify python, spacy, and mlflow
                      with appropriate versions.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param kwargs: kwargs to pass to ``spacy_model.to_disk`` method.
    """
    path = os.path.abspath(path)
    if os.path.exists(path):
        raise RuntimeError("Path '{}' already exists".format(path))
    os.makedirs(path)
    model_path = os.path.join(path, "model.pth")

    # Save spacy model
    spacy_model.to_disk(model_path, **kwargs)
    model_file = os.path.basename(model_path)

    conda_env_subpath = "conda.yaml"
    if conda_env is not None:
        shutil.copyfile(conda_env, os.path.join(path, conda_env_subpath))
    else:
        with open(os.path.join(path, conda_env_subpath), "w") as f:
            yaml.safe_dump(DEFAULT_CONDA_ENV, stream=f, default_flow_style=False)

    mlflow_model.add_flavor(FLAVOR_NAME, model_data=model_file,
                            spacy_version=spacy.__version__)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.spacy",
                        data=model_file, env=conda_env_subpath)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def _load_model(path, **kwargs):
    return spacy.load(path, **kwargs)


def load_model(path, run_id=None, **kwargs):
    """
    Load a SpaCy model from a local file (if ``run_id`` is ``None``) or a run.

    :param path: Local filesystem path or run-relative artifact path to the model saved
                 by :py:func:`mlflow.spacy.log_model`.
    :param run_id: Run ID. If provided, combined with ``path`` to identify the model.
    :param kwargs: kwargs to pass to ``spacy.load`` method.
    """
    if run_id is not None:
        path = mlflow.tracking.utils._get_model_log_dir(model_name=path, run_id=run_id)
    path = os.path.abspath(path)
    flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME)
    spacy_model_artifacts_path = os.path.join(path, flavor_conf['model_data'])
    return _load_model(path=spacy_model_artifacts_path, **kwargs)


def _load_pyfunc(path, **kwargs):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return _SpaCyWrapper(_load_model(os.path.abspath(path), **kwargs))


class _SpaCyWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """
    def __init__(self, spacy_model):
        self.spacy_model = spacy_model

    def predict(self, data):
        pass
        # if not isinstance(data, pd.DataFrame):
        #     raise TypeError("Input data should be pandas.DataFrame")
        #
        # if "mode" not in data:
        #     raise ValueError(
        #         "Please specify an additional column called mode with one "
        #         "of the following values: classification, ner, parsing, tagging",
        #     )
        # mode = data["mode"][data["mode"].notna()].tolist()
        # if not mode:
        #     raise ValueError(
        #         "Please specify a value for \"mode\". You can choose one "
        #         "of the following values: classification, ner, parsing, tagging",
        #     )
        # mode = mode[0]
        #
        # if set(data.columns) != {"text", "mode"}:
        #     raise ValueError(
        #         "Data has columns that are not only \"text\" and \"mode\"."
        #         "Found columns: {}".format(set(data.columns)),
        #     )
        #
        # # Drop the extra row from the "mode" column
        # texts = data[data["mode"].isna()]["text"]
        #
        # if mode == "classification":
        #     return pd.DataFrame({
        #         "predictions": texts.apply(lambda text: self.spacy_model(text).cats)
        #     })
        # if mode == "ner":
        #     entities = texts.apply(lambda text: self.spacy_model(text).ents)
        #     return pd.DataFrame({
        #         "predictions": entities.apply(lambda ents: [{
        #             "text": ent.text,
        #             "label": ent.label_,
        #             "start_char": ent.start_char,
        #             "end_char": ent.end_char,
        #         } for ent in ents])
        #     })
        # if mode == "parser":
        #     # TODO
        #     return
        # if mode == "tagger":
        #     # TODO
        #     return
        #
        # raise ValueError("{} is not a supported mode".format(mode))
