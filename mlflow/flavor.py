import os

import yaml
import cloudpickle

from mlflow.exceptions import MlflowException 
from mlflow.models import Model
from mlflow.tracking.utils import _get_model_log_dir


LOADER_MODULE_KEY = "LOADER"


def _load_pyfunc(path):
    model_config = Model.load(os.path.join(path, "MLmodel"))


class FlavorModule:

    def __init__(self, name, save_fn, load_fn, to_pyfunc_fn=None):
        self.name = name
        self.save_fn = save_fn
        self.load_fn = load_fn
        self.to_pyfunc_fn = to_pyfunc_fn

    def save_model(self, path, mlflow_model=Model(), **kwargs):
        if os.path.exists(path):
            raise Exception("Path '{}' already exists".format(path))

        data_subpath = "data"
        data_path = os.path.join(path, data_subpath)
        os.makedirs(data_path)
        self.save_fn(path=data_path, **kwargs)

        flavor_module_subpath = os.path.join("flavor_modules", self.name)
        os.makedirs(os.path.join(path, flavor_module_subpath))
        save_fn_subpath = os.path.join(flavor_module_subpath, "save.pkl") 
        load_fn_subpath = os.path.join(flavor_module_subpath, "load.pkl")
        with open(os.path.join(path, save_fn_subpath), "w") as save_out:
            cloudpickle.dump(self.save_fn, save_out)
        with open(os.path.join(path, load_fn_subpath), "w") as load_out:
            cloudpickle.dump(self.load_fn, load_out)

        mlflow_model.add_flavor(model_data=data_subpath,
                                flavor_saver_path=save_fn_subpath,
                                flavor_loader_path=load_fn_subpath, )
        mlflow_model.save(os.path.join(path, "MLmodel"))

    def log_model(self, artifact_path, **kwargs):
        Model.log(artifact_path=artifact_path, flavor=self, **kwargs)

    def load_model(self, path, run_id=None):
        if run_id is not None:
            path = _get_model_log_dir(path, run_id)
        model_config = Model.load(os.path.join(path, "MLmodel"))
        return self.load_fn(os.path.join(path, model_config.flavors[self.name]["model_data"]))

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

        flavor_saver_subpath = flavor_config.get("flavor_saver_path", None)
        if flavor_saver_subpath is None:
            raise MlflowException(
                    "The configuration for the specified flavor does not contain a `save_model`"
                    " function.")

        flavor_loader_subpath = flavor_config.get("flavor_loader_path", None)
        if flavor_loader_subpath is None:
            raise MlflowException(
                    "The configuration for the specified flavor does not contain a `load_model`"
                    " function.")

        with open(os.path.join(path, flavor_saver_subpath), "r") as save_in:
            save_fn = cloudpickle.load(save_in)

        with open(os.path.join(path, flavor_loader_subpath), "r") as load_in:
            load_fn = cloudpickle.load(load_in)

        return cls(name=name, save_fn=save_fn, load_fn=load_fn)

if __name__ == "__main__":
    pass
