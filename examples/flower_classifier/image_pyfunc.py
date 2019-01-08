"""
Example of a custom python function implementing image classifier with image preprocessing embedded
in the model.
"""
import base64
import cloudpickle
from io import BytesIO
import keras
import numpy as np
import os
import pandas as pd
import PIL
from PIL import Image
import yaml
import tensorflow as tf

import mlflow
import mlflow.keras
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.file_utils import TempDir
from mlflow.utils.environment import _mlflow_conda_env


def decode_and_resize_image(raw_bytes, size):
    """
    Read, decode and resize raw image bytes (e.g. raw content of a jpeg file).

    :param raw_bytes: Image bits, e.g. jpeg image.
    :param size: requested output dimensions
    :return: Multidimensional numpy array representing the resized image.
    """
    return np.asarray(Image.open(BytesIO(raw_bytes)).resize(size), dtype=np.float32)


class KerasImageClassifier2(mlflow.pyfunc.PythonModel):

    def __init__(self, context):
        super(KerasImageClassifier2, self).__init__(context)
        self._tf_graph = tf.Graph()
        self._tf_sess = tf.Session(graph=self._tf_graph)
        probs_names = ["p({})".format(x) for x in context.parameters["domain"]]
        self._column_names = ["predicted_label", "predicted_label_id"] + probs_names
        with self._tf_graph.as_default():
            keras.backend.set_session(self._tf_sess)
            self._keras_model = mlflow.keras.load_model(context.artifacts["keras_model"])


    def predict(self, model_input):
        def decode_img(x):
            return pd.Series(base64.decodebytes(bytearray(x[0], encoding="utf8")))

        images = input.apply(axis=1, func=decode_img)
        probs = self._predict_images(images)
        m, n = probs.shape
        label_idx = np.argmax(probs, axis=1)
        labels = np.array([
            self.context.parameters["domain"][i] for i in label_idx], dtype=np.str).reshape(m, 1)
        output_data = np.concatenate((labels, label_idx.reshape(m, 1), probs), axis=1)
        res = pd.DataFrame(columns=self._column_names, data=output_data)
        res.index = input.index
        return res


    def _predict_images(self, images):
        def preprocess_f(z):
            return decode_and_resize_image(z, self.parameters["image_dims"][:2])

        x = np.array(
            images[images.columns[0]].apply(preprocess_f).tolist())
        with self._tf_graph.as_default():
            with self._tf_sess.as_default():
                return self._keras_model.predict(x)


def log_model(keras_model, artifact_path, image_dims, domain):
    """
    Log a KerasImageClassifierPyfunc model as an MLflow artifact for the current run.

    :param keras_model: Keras model to be saved.
    :param artifact_path: Run-relative artifact path this model is to be saved to.
    :param image_dims: Image dimensions the Keras model expects.
    :param domain: Labels for the classes this model can predict.
    """

    with TempDir() as tmp:
        keras_model_path = tmp.path("keras_model")
        mlflow.keras.save_model(keras_model, path=keras_model_path)

        conda_env = tmp.path("conda_env.yaml")
        with open(conda_env, "w") as f:
            f.write(conda_env_template.format(python_version=PYTHON_VERSION,
                                              cloudpickle_version=cloudpickle.__version__,
                                              keras_version=keras.__version__,
                                              tf_name=tf.__name__,  # can have optional -gpu suffix
                                              tf_version=tf.__version__,
                                              pillow_version=PIL.__version__))

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            model_class=KerasImageClassifier2,
            artifacts={
                "keras_model": keras_model_path,
            },
            parameters={
                "image_dims": image_dims,
                "domain": domain,
            },
            code_paths=[__file__])


conda_env_template = """        
name: flower_classifier
channels:
  - defaults
  - anaconda
dependencies:
  - python=={python_version}
  - cloudpickle=={cloudpickle_version}
  - keras=={keras_version} 
  - {tf_name}=={tf_version} 
  - pip:    
    - pillow=={pillow_version}
"""
