from __future__ import absolute_import

import os

import cloudpickle
import yaml
import tempfile

from mlflow import pyfunc
from mlflow.exceptions import MlflowException 
from mlflow.models import Model
from mlflow.tracking.utils import _get_model_log_dir


def _load_pyfunc(path, **kwargs):
    flavor_module = FlavorModule.load(name=kwargs["flavor_module_name"], path=path)
    return flavor_module.load_pyfunc(path=path)


class FlavorModule:

    def __init__(self, name, save_fn, load_fn, load_pyfunc_fn=None, conda_env=None):
        self.name = name
        self.save_fn = save_fn
        self.load_fn = load_fn
        self.load_pyfunc_fn = load_pyfunc_fn
        self.conda_env = (self._parse_conda_env(conda_env_path=conda_env) 
                if conda_env is not None else None)

    def save_model(self, path, mlflow_model=Model(), **kwargs):
        if os.path.exists(path):
            raise MlflowException("Path '{}' already exists".format(path))

        data_subpath = "data"
        data_path = os.path.join(path, data_subpath)
        os.makedirs(data_path)
        self.save_fn(path=data_path, **kwargs)

        flavor_module_subpath = os.path.join("flavor_modules", self.name)
        os.makedirs(os.path.join(path, flavor_module_subpath))
        save_fn_subpath = os.path.join(flavor_module_subpath, "save.pkl") 
        with open(os.path.join(path, save_fn_subpath), "wb") as save_out:
            cloudpickle.dump(self.save_fn, save_out)
        load_fn_subpath = os.path.join(flavor_module_subpath, "load.pkl")
        with open(os.path.join(path, load_fn_subpath), "wb") as load_out:
            cloudpickle.dump(self.load_fn, load_out)

        flavor_kwargs = {
                "data": data_subpath,
                "flavor_saver_path": save_fn_subpath,
                "flavor_loader_path": load_fn_subpath,
        }

        if self.conda_env is not None:
            conda_env_subpath = "conda.yaml"
            with open(os.path.join(path, conda_env_subpath), "w") as f:
                yaml.safe_dump(self.conda_env, f, default_flow_style=False)
            flavor_kwargs["conda_env_path"] = conda_env_subpath
        else:
            conda_env_subpath = None
        

        if self.load_pyfunc_fn is not None:
            load_pyfunc_fn_subpath = os.path.join(flavor_module_subpath, "load_pyfunc.pkl")
            with open(os.path.join(path, load_pyfunc_fn_subpath), "wb") as load_pyfunc_out:
                cloudpickle.dump(self.load_pyfunc_fn, load_pyfunc_out)
            flavor_kwargs["flavor_load_pyfunc_path"] = load_pyfunc_fn_subpath 

            pyfunc.add_to_model(mlflow_model, 
                                loader_module="mlflow.flavor", 
                                flavor_module_name=self.name,
                                env=conda_env_subpath)

        mlflow_model.add_flavor(self.name, **flavor_kwargs)
        mlflow_model.save(os.path.join(path, "MLmodel"))

    def log_model(self, artifact_path, **kwargs):
        Model.log(artifact_path=artifact_path, flavor=self, **kwargs)

    def load_model(self, path, run_id=None):
        if run_id is not None:
            path = _get_model_log_dir(path, run_id)
        model_config = Model.load(os.path.join(path, "MLmodel"))
        return self.load_fn(os.path.join(path, model_config.flavors[self.name]["data"]))

    def load_pyfunc(self, path, run_id=None):
        if self.load_pyfunc_fn is None:
            raise MlflowException("This flavor does not support being loaded as a python function.")

        if run_id is not None:
            path = _get_model_log_dir(path, run_id)
        model_config = Model.load(os.path.join(path, "MLmodel"))
        return self.load_pyfunc_fn(os.path.join(path, model_config.flavors[self.name]["data"]))

    def get_conda_env(self, path, run_id=None):
        if run_id is not None:
            path = _get_model_log_dir(path, run_id)
        model_config = Model.load(os.path.join(path, "MLmodel"))
        with open(os.path.join(path, model_config.flavors[self.name]["conda_env_path"]), "r") as f:
            return f.read()

    def to_py_module(self, output_path):
        import inspect
        save_fn_source = inspect.getsource(self.save_fn)
        load_fn_source = inspect.getsource(self.load_fn)

        module_text = "\n\n".join([save_fn_source, load_fn_source])
        with open(output_path, "w") as f:
            f.write(module_text)

    def _parse_conda_env(self, conda_env_path):
        with open(conda_env_path, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def load(cls, name, path, run_id=None):
        if run_id is not None:
            path = _get_model_log_dir(path, run_id)

        model_config = Model.load(os.path.join(path, "MLmodel"))
        flavor_config = model_config.flavors.get(name, None)
        if flavor_config is None:
            raise MlflowException(
                    "The model at the specified path does not contain a flavor with name:"
                    " {flavor_name}".format(flavor_name=name))

        save_fn_subpath = flavor_config.get("flavor_saver_path", None)
        if save_fn_subpath is None:
            raise MlflowException(
                    "The configuration for the specified flavor does not contain a `load_model`"
                    " function.")
        with open(os.path.join(path, save_fn_subpath), "rb") as save_in:
            save_fn = cloudpickle.load(save_in)

        load_fn_subpath = flavor_config.get("flavor_loader_path", None)
        if load_fn_subpath is None:
            raise MlflowException(
                    "The configuration for the specified flavor does not contain a `save_model`"
                    " function.")
        with open(os.path.join(path, load_fn_subpath), "rb") as load_in:
            load_fn = cloudpickle.load(load_in)

        conda_env_path = flavor_config.get("conda_env_path", None)
        if conda_env_path is not None:
            conda_env_path = os.path.join(path, conda_env_path)
            

        load_pyfunc_fn_subpath = flavor_config.get("flavor_load_pyfunc_path", None)
        if load_pyfunc_fn_subpath is not None:
            with open(os.path.join(path, load_pyfunc_fn_subpath), "rb") as load_pyfunc_in:
                load_pyfunc_fn = cloudpickle.load(load_pyfunc_in)
        else:
            load_pyfunc_fn = None

        return cls(name=name, save_fn=save_fn, load_fn=load_fn, load_pyfunc_fn=load_pyfunc_fn, 
                   conda_env=conda_env_path)

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

    from mlflow.utils.environment import _mlflow_conda_env

    conda_path = "conda.yaml"
    _mlflow_conda_env(conda_path, additional_pip_deps=["scikit-learn"])

    flavor_module = FlavorModule("sklearncustom", save_model, load_model, load_pyfunc, conda_path)

    # Use the custom flavor to save a scikit-learn model
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.preprocessing import FunctionTransformer as SKFunctionTransformer
    def transform(vec):
        return vec + 1
    pipeline = SKPipeline([("name", SKFunctionTransformer(transform, validate=True))])

    model_path = tempfile.mktemp()
    flavor_module.save_model(path=model_path, sk_model=pipeline)

    # Load the custom flavor from the serialized model and use it to deserialize the
    # the model in native format as well as pyfunc format
    flavor_module = FlavorModule.load("sklearncustom", model_path)

    sk_model = flavor_module.load_model(model_path)

    sk_pyfunc = pyfunc.load_pyfunc(model_path)

    print(sk_pyfunc)
    print(flavor_module.get_conda_env(model_path))

