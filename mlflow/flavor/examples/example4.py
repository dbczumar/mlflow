import os
import collections
import tempfile

import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.datasets as datasets

import mlflow
import mlflow.pyfunc as pyfunc
from mlflow.flavor import Flavor
from mlflow.utils.environment import _mlflow_conda_env

SavedModelInfo = collections.namedtuple(
        "SavedModelInfo",
        ["path", "meta_graph_tags", "signature_def_key", "inference_df", "expected_results_df"])


def save_tf_iris_model(dst_dir):
    """
    Trains a Tensorflow model on the Iris dataset and saves it in the specified directory.
    The model is saved in Tensorflow's `SavedModel` persistence format.
    """
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features
    y = iris.target
    trainingFeatures = {}
    for i in range(0, 2):
        # TensorFlow is fickle about feature names, so we remove offending characters
        iris.feature_names[i] = iris.feature_names[i].replace(" ", "")
        iris.feature_names[i] = iris.feature_names[i].replace("(", "")
        iris.feature_names[i] = iris.feature_names[i].replace(")", "")
        trainingFeatures[iris.feature_names[i]] = iris.data[:, i:i+1]
    tf_feat_cols = []
    feature_names = iris.feature_names[:2]
    # Create Tensorflow-specific numeric columns for input.
    for col in iris.feature_names[:2]:
        tf_feat_cols.append(tf.feature_column.numeric_column(col))
    # Create a training function for the estimator
    input_train = tf.estimator.inputs.numpy_input_fn(trainingFeatures,
                                                     y,
                                                     shuffle=False,
                                                     batch_size=1)
    estimator = tf.estimator.DNNRegressor(feature_columns=tf_feat_cols,
                                          hidden_units=[1])
    # Train the estimator and obtain expected predictions on the training dataset
    estimator.train(input_train, steps=10)
    estimator_preds = np.array([s["predictions"] for s in estimator.predict(input_train)]).ravel()
    estimator_preds_df = pd.DataFrame({"predictions": estimator_preds})

    # Define a function for estimator inference
    feature_spec = {}
    for name in feature_names:
        feature_spec[name] = tf.placeholder("float", name=name, shape=[150])
    receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    # Save the estimator and its inference function
    saved_estimator_path = str(dst_dir)
    saved_estimator_path = estimator.export_savedmodel(saved_estimator_path,
                                                       receiver_fn).decode("utf-8")
    return SavedModelInfo(path=saved_estimator_path,
                          meta_graph_tags=[tf.saved_model.tag_constants.SERVING],
                          signature_def_key="predict",
                          inference_df=pd.DataFrame(data=X, columns=feature_names),
                          expected_results_df=estimator_preds_df)


if __name__ == "__main__":
    # Train and save a Tensorflow model in Tensorflow's `SavedModel` format
    model_dst_dir = tempfile.mkdtemp()
    tf_saved_model = save_tf_iris_model(dst_dir=model_dst_dir)

    # Load the tensorflow flavor from source 
    flavor_module = Flavor.from_source("mlflow.tensorflow", flavor_name="tensorflow")
   
    # Use the flavor to save the tensorflow model in MLflow format
    mlflow_model_path = "model"
    with mlflow.start_run():
        flavor_module.log_model(
                path=mlflow_model_path, tf_saved_model_dir=tf_saved_model.path, 
                tf_meta_graph_tags=tf_saved_model.meta_graph_tags, 
                tf_signature_def_key=tf_saved_model.signature_def_key)
        run_id = mlflow.active_run().info.run_uuid

    # Load the flavor from the serialized model and use it to deserialize the
    # the model in native format as well as pyfunc format
    flavor_module = Flavor.from_model("tensorflow", model_path=mlflow_model_path, run_id=run_id)

    # Use the loaded flavor to load the orginal model in a Tensorflow graph
    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)
    with tf_graph.as_default():
        sigdef = flavor_module.load_model(path=mlflow_model_path, run_id=run_id, tf_sess=tf_sess)
        print(sigdef)

    # Use the loaded flavor to load the model as a python function
    tf_pyfunc = pyfunc.load_pyfunc(path=mlflow_model_path, run_id=run_id)
    print(tf_pyfunc)
        

    

