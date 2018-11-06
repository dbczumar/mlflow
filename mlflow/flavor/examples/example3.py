import os
import tempfile

from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import FunctionTransformer as SKFunctionTransformer

import mlflow.sklearn
import mlflow.pyfunc as pyfunc
from mlflow.flavor import Flavor
from mlflow.utils.environment import _mlflow_conda_env

if __name__ == "__main__":
    # Load the sklearn flavor by passing its `save_model`, `load_model`, and `_load_pyfunc`
    # functions as constructor arguments
    flavor_module = Flavor("sklearn", 
                           mlflow.sklearn.save_model, 
                           mlflow.sklearn.load_model, 
                           mlflow.sklearn._load_pyfunc,
                           simple=False)
    
    # Use the flavor to save a scikit-learn model
    def transform(vec):
        return vec + 1
    pipeline = SKPipeline([("name", SKFunctionTransformer(transform, validate=True))])
    model_path = tempfile.mktemp()
    flavor_module.save_model(path=model_path, sk_model=pipeline)

    # Load the flavor from the serialized model and use it to deserialize the
    # the model in native format as well as pyfunc format
    flavor_module = Flavor.from_model("sklearn", model_path)

    # Use the loaded flavor to load the orginal model
    sk_model = flavor_module.load_model(model_path)
    print(sk_model)

    # Use the loaded flavor to load the model as a python function
    sk_pyfunc = pyfunc.load_pyfunc(model_path)
    print(sk_pyfunc)
