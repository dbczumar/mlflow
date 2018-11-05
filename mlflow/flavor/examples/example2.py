import os
import tempfile

from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import FunctionTransformer as SKFunctionTransformer

import mlflow.pyfunc as pyfunc
from mlflow.flavor import Flavor
from mlflow.utils.environment import _mlflow_conda_env

if __name__ == "__main__":
    flavor_module = Flavor.from_source(uri=".", module_name="mlflow.sklearn", flavor_name="sklearn")

    def transform(vec):
        return vec + 1
    pipeline = SKPipeline([("name", SKFunctionTransformer(transform, validate=True))])

    model_path = tempfile.mktemp()
    flavor_module.save_model(path=model_path, sk_model=pipeline)

    flavor_module = Flavor.from_model("sklearn", model_path)

    sk_model = flavor_module.load_model(model_path)
    print(sk_model)

    sk_pyfunc = pyfunc.load_pyfunc(model_path)
    print(sk_pyfunc)
