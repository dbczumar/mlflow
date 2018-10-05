import os
import cloudpickle 
import shutil

import mlflow
import mlflow.spark as mspark
import mlflow.pyfunc as pyfunc
from mlflow.models import Model
from mlflow.tracking.utils import _get_model_log_dir

FLAVOR_NAME = "FUNCTION_MODEL"


class Predictor(object):
    """
    A wrapper around a Spark model and a function that uses the model to evaluate a query.
    """

    def __init__(self, function, model):
        """
        :param function: A python function
        :param model: A Spark model
        """
        self.function = function
        self.model = model


    def predict(self, *args, **kwargs):
        """
        Evaluates a query using the predictor's model and function.
        """
        return self.function(self.model, *args, **kwargs)


    @classmethod
    def load(cls, function_path, model_path):
        """
        Constructs a `Predictor` object from the serialized function and serialized model
        located at `function_path` and `model_path`, respectively.

        :param function_path: The path to the serialized function.
        :param model_path: The path to the serialized model.
        """
        with open(function_path, "rb") as f:
            function = cloudpickle.load(f)

        model = mspark._load_pyfunc(model_path)
        return cls(function=function, model=model)


    def save(self, function_path, model_path):
        """
        :param function_path: The path to which to save the predictor's function
        :param model_path: The path to which to save the predictor's model
        """
        with open(function_path, "wb") as f:
            cloudpickle.dump(self.function, f)

        mspark._save_model(self.model, model_path)


def load_model(path, run_id=None):
    """
    Loads a serialized mlflow.funcmodel.Predictor model

    :param path: local filesystem path or run-relative artifact path to the model.
    :param run_id: Run ID. If provided, combined with ``path`` to identify the model.
    """
    if run_id is not None:
        path = _get_model_log_dir(model_name=path, run_id=run_id)
    m = Model.load(os.path.join(path, 'MLmodel'))
    if FLAVOR_NAME not in m.flavors:
        raise Exception("Model does not have {} flavor".format(FLAVOR_NAME))
    conf = m.flavors[FLAVOR_NAME]
    function_path = os.path.join(path, conf["function_path"])
    model_path = os.path.join(path, conf["model_path"])
    return Predictor.load(function_path=function_path, model_path=model_path)


def save_model(model, path, mlflow_model=Model(), conda_env=None):
    """
    :param model: An mlflow.funcmodel.Predictor object
    :param path: The local path to which to save the model
    :param mlflow_model: MLflow model config this flavor is being added to.
    :param conda_env: The path to a conda environment defining dependencies for the model
    """
    os.makedirs(path)
    function_path_sub = "function.pkl"
    function_path = os.path.join(path, function_path_sub)
    model_path_sub = "model"
    model_path = os.path.join(path, model_path_sub)

    model.save(function_path=function_path, model_path=model_path)

    model_conda_env = None
    if conda_env:
        model_conda_env = os.path.basename(os.path.abspath(conda_env))
        shutil.copyfile(conda_env, os.path.join(path, model_conda_env))

    mlflow_model.add_flavor(
            FLAVOR_NAME,
            function_path=function_path_sub,
            model_path=model_path_sub)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.funcmodel", env=model_conda_env)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(model, artifact_path, conda_env=None):
    """
    :param model: An mlflow.funcmodel.Predictor object
    :param artifact_path: The run-relative path to which to save the model
    :param conda_env: The path to a conda environment defining dependencies for the model
    """
    return Model.log(artifact_path=artifact_path, flavor=mlflow.funcmodel, model=model,
                     conda_env=conda_env)


def _load_pyfunc(path):
    return load_model(path=path)


if __name__ == "__main__":
    import mlflow.funcmodel
    import mlflow.pyfunc

    # Train a sample model (this one just performs logistic regression)
    from pyspark.sql import SparkSession, SQLContext
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import Tokenizer, HashingTF
    spark = SparkSession.builder \
        .master('local') \
        .appName('sample') \
        .getOrCreate()
    sql_context = SQLContext(spark.sparkContext)
    sample_df = sql_context.createDataFrame([("sample", 0), ("df", 1)], schema=["text", "label"])
    pipeline = Pipeline(stages=[
        Tokenizer(inputCol="text", outputCol="words"),
        HashingTF(inputCol="words", outputCol="features"),
        LogisticRegression(maxIter=20)])
    model = pipeline.fit(sample_df)


    # Define a `predict` function that uses the model to render a prediction, with optional
    # pre-processing and post-processing logic.
    def predict(spark_model, pandas_df):
        """
        :param spark_model: A PySpark model, represented as a generic python function
                            wrapper with a `predict` method.
        :param pandas_df: A pandas dataframe input to be evaluated.
        """

        # Preprocess the input dataframe here.
        print("Did some preprocessing!")

        model_result = spark_model.predict(pandas_df)
        print("Model result: {}".format(model_result))

        # Postprocess the prediction result here.
        print("Did some postprocessing!")

        return model_result

    # Construct a funcmodel `Predictor` object using the Spark model and the
    # `predict` function. Save the predictor object using `funcmodel.log_model`.
    predictor = Predictor(function=predict, model=model)
    with mlflow.start_run(experiment_id=0):
        model_path = "model"
        mlflow.funcmodel.log_model(model=predictor, artifact_path=model_path)
        run_id = mlflow.active_run().info.run_uuid

    # Load the predictor as a generic python function wrapper. This is what the Docker container
    # will do when serving the model.
    pyfunc_predictor = mlflow.pyfunc.load_pyfunc(path=model_path, run_id=run_id)

    # Perform inference using the function wrapper's `predict` method. This is what the Docker
    # container will do when responding to a query during model serving.
    sample_input = sample_df.toPandas()
    pyfunc_predictor.predict(sample_input)
