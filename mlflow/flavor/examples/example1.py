import os
import tempfile

from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import FunctionTransformer as SKFunctionTransformer

import mlflow.pyfunc as pyfunc
from mlflow.flavor import Flavor
from mlflow.utils.environment import _mlflow_conda_env


if __name__ == "__main__":
    # Define a custom flavor by implementing `save_fn`, `load_fn`, and `load_pyfunc_fn`
    # In this case, we are re-implementing the sklearn flavor
    def save_model(path, sk_model):
        import cloudpickle
        with open(os.path.join(path, "skmodel.pkl"), "wb") as f:
            cloudpickle.dump(sk_model, f)

    def load_model(path):
        import cloudpickle
        with open(os.path.join(path, "skmodel.pkl"), "rb") as f:
            return cloudpickle.load(f)

    def load_pyfunc(path):
        return load_model(path)

    flavor_module = Flavor("sklearncustom", save_model, load_model, load_pyfunc)

    # Use the custom flavor to save a scikit-learn model
    def transform(vec):
        return vec + 1
    pipeline = SKPipeline([("name", SKFunctionTransformer(transform, validate=True))])
    model_path = tempfile.mktemp()
    flavor_module.save_model(path=model_path, sk_model=pipeline)

    # Load the custom flavor from the serialized model and use it to deserialize the
    # the model in native format as well as pyfunc format
    flavor_module = Flavor.from_model("sklearncustom", model_path)

    # Use the loaded flavor to load the orginal model
    sk_model = flavor_module.load_model(model_path)
    print(sk_model)

    # Use the loaded flavor to load the model as a python function
    sk_pyfunc = pyfunc.load_pyfunc(model_path)
    print(sk_pyfunc)
