import os

import cloudpickle

from mlflow import pyfunc
from mlflow.exceptions import MlflowException 
from mlflow.models import Model
from mlflow.tracking.utils import _get_model_log_dir


def _load_pyfunc(path, **kwargs):
    flavor_module = FlavorModule.load(name=kwargs["flavor_module_name"], path=path)
    return flavor_module.load_pyfunc(path=path)


class FlavorModule:

    def __init__(self, name, save_fn, load_fn, to_pyfunc_fn=None):
        self.name = name
        self.save_fn = save_fn
        self.load_fn = load_fn
        self.to_pyfunc_fn = to_pyfunc_fn

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

        if self.to_pyfunc_fn is not None:
            to_pyfunc_fn_subpath = os.path.join(flavor_module_subpath, "to_pyfunc.pkl")
            with open(os.path.join(path, to_pyfunc_fn_subpath), "wb") as to_pyfunc_out:
                cloudpickle.dump(self.to_pyfunc_fn, to_pyfunc_out)
            flavor_kwargs["flavor_to_pyfunc_path"] = to_pyfunc_fn_subpath

            pyfunc.add_to_model(mlflow_model, 
                                loader_module="mlflow.flavor", 
                                flavor_module_name=self.name)

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
        if self.to_pyfunc_fn is None:
            raise MlflowException("This flavor does not support being loaded as a python function.")

        return self.to_pyfunc_fn(self.load_model(path=path, run_id=run_id))

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

        to_pyfunc_fn_subpath = flavor_config.get("flavor_to_pyfunc_path", None)
        if to_pyfunc_fn_subpath is not None:
            with open(os.path.join(path, to_pyfunc_fn_subpath), "rb") as to_pyfunc_in:
                to_pyfunc_fn = cloudpickle.load(to_pyfunc_in)
        else:
            to_pyfunc_fn = None

        return cls(name=name, save_fn=save_fn, load_fn=load_fn, to_pyfunc_fn=to_pyfunc_fn)

if __name__ == "__main__":
    # def save_model(path, sk_model):
    #     import cloudpickle
    #     with open(os.path.join(path, "skmodel.pkl"), "wb") as f:
    #         cloudpickle.dump(sk_model, f)
    #
    # def load_model(path):
    #     import cloudpickle
    #     with open(os.path.join(path, "skmodel.pkl"), "rb") as f:
    #         return cloudpickle.load(f)
    #
    # def to_pyfunc(sk_model):
    #     return sk_model
    #
    # flavor_module = FlavorModule("sklearncustom", save_model, load_model, to_pyfunc)
    #
    # from sklearn.pipeline import Pipeline as SKPipeline
    # from sklearn.preprocessing import FunctionTransformer as SKFunctionTransformer
    # def transform(vec):
    #     return vec + 1
    # pipeline = SKPipeline([("name", SKFunctionTransformer(transform, validate=True))])
    # flavor_module.save_model(path="/tmp/cat", sk_model=pipeline)

    # sk_model = FlavorModule.load("sklearncustom", "/tmp/cat").load_model("/tmp/cat")
    # print(sk_model)
    #
    sk_pyfunc = pyfunc.load_pyfunc("/tmp/cat")
    print(sk_pyfunc)
