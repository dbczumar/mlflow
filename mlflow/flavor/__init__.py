from __future__ import absolute_import

import os
from functools import update_wrapper

import cloudpickle
import yaml
import tempfile
import shutil

from mlflow import pyfunc
from mlflow.exceptions import MlflowException 
from mlflow.models import Model
from mlflow.tracking.utils import _get_model_log_dir


def _load_pyfunc(path, **kwargs):
    flavor_module = Flavor.from_model(flavor_name=kwargs["flavor_name"], model_path=path)
    return flavor_module._load_pyfunc(path=path)


class Chdir:

    def __init__(self, dir_path):
        self.dir_path = os.path.abspath(dir_path)
        self.prev_dir = None

    def __enter__(self):
        self.prev_dir = os.getcwd()
        os.chdir(self.dir_path)

    def __exit__(self, *args, **kwargs): 
        os.chdir(self.prev_dir)


class Flavor:

    def __init__(self, name, save_fn, load_fn, load_pyfunc_fn=None, simple=True):
        if simple:
            save_fn, load_fn, load_pyfunc_fn = Flavor._convert_simple_functions(
                    flavor_name=name, save_fn=save_fn, load_fn=load_fn, 
                    load_pyfunc_fn=load_pyfunc_fn)

        self.save_model = Flavor._wrap_save_function(
                flavor_name=name, save_fn=save_fn, load_fn=load_fn, load_pyfunc_fn=load_pyfunc_fn)
        update_wrapper(self.save_model, save_fn)
        self.load_model = load_fn
        self._load_pyfunc = load_pyfunc_fn

    @staticmethod
    def _convert_simple_functions(flavor_name, save_fn, load_fn, load_pyfunc_fn=None):
        def save_model(path, mlflow_model=Model(), conda_env=None, **kwargs):
            if os.path.exists(path):
                raise MlflowException("Path '{}' already exists".format(path))

            data_subpath = "data"
            data_path = os.path.join(path, data_subpath)
            os.makedirs(data_path)
            save_fn(path=data_path, **kwargs)

            model_conda_env = None
            if conda_env:
                model_conda_env = os.path.basename(os.path.abspath(conda_env))
                shutil.copyfile(conda_env, os.path.join(path, model_conda_env))
            
            mlflow_model.add_flavor(flavor_name, data=data_subpath)
            if load_pyfunc_fn is not None:
                pyfunc.add_to_model(mlflow_model, 
                                    loader_module="mlflow.flavor", 
                                    flavor_name=flavor_name,
                                    env=model_conda_env)

            mlflow_model.save(os.path.join(path, "MLmodel"))

        def load_model(path, run_id=None):
            if run_id is not None:
                path = _get_model_log_dir(path, run_id)
            model_config = Model.load(os.path.join(path, "MLmodel"))
            return load_fn(os.path.join(path, model_config.flavors[flavor_name]["data"]))

        if load_pyfunc_fn is not None:
            def _load_pyfunc(path, run_id=None):
                if run_id is not None:
                    path = _get_model_log_dir(path, run_id)
                model_config = Model.load(os.path.join(path, "MLmodel"))
                return load_pyfunc_fn(
                        os.path.join(path, model_config.flavors[flavor_name]["data"]))
        else:
            _load_pyfunc = None

        return save_model, load_model, _load_pyfunc


    @staticmethod
    def _wrap_save_function(flavor_name, save_fn, load_fn, load_pyfunc_fn=None):
        def save_model(path, *args, **kwargs):
            result = save_fn(path=path, *args, **kwargs)

            # Override the model configuration's `pyfunc` configuration so that the pickled
            # `load_pyfunc_fn` can be used to load the model as a python function
            model_conf = Model.load(os.path.join(path, "MLmodel"))
            flavor_conf = model_conf.flavors.get(pyfunc.FLAVOR_NAME, None)
            if flavor_conf is not None:
                flavor_conf["loader_module"] = "mlflow.flavor"
                flavor_kwargs = flavor_conf["kwargs"] if "kwargs" in flavor_conf else {}
                flavor_kwargs["flavor_name"] = flavor_name
                flavor_conf["kwargs"] = flavor_kwargs


            # Save the flavor's `save`, `load`, and `load_pyfunc` functions with the model
            flavor_module_subpath = os.path.join("flavor_modules", "python", flavor_name)
            os.makedirs(os.path.join(path, flavor_module_subpath))

            save_fn_subpath = os.path.join(flavor_module_subpath, "save.pkl")
            with open(os.path.join(path, save_fn_subpath), "wb") as save_out:
                cloudpickle.dump(save_fn, save_out)
            load_fn_subpath = os.path.join(flavor_module_subpath, "load.pkl")
            with open(os.path.join(path, load_fn_subpath), "wb") as load_out:
                cloudpickle.dump(load_fn, load_out)
            if load_pyfunc_fn is not None:
                load_pyfunc_subpath = os.path.join(flavor_module_subpath, "load_pyfunc.pkl")
                with open(os.path.join(path, load_pyfunc_subpath), "wb") as load_pyfunc_out:
                    cloudpickle.dump(load_pyfunc_fn, load_pyfunc_out)

            return result

        return save_model

    @staticmethod
    def _validate_source_module(flavor_module):
        if not hasattr(flavor_module, "save_model"):
            raise MlflowException("Flavor source module must define a `save_model` method!")
        if not hasattr(flavor_module, "load_model"):
            raise MlflowException("Flavor source module must define a `load_model` method!")

    @classmethod
    def from_source(cls, module_name, uri=None, flavor_name=None, git_username=None, 
                    git_password=None):
        import importlib

        if uri is None:
            uri = os.getcwd()
        uri = os.path.abspath(uri)
        with Chdir(uri):
            flavor_module = importlib.import_module(module_name)
        Flavor._validate_source_module(flavor_module=flavor_module)

        if flavor_name is None:
            flavor_name = module_name

        return cls(name=flavor_name, 
                   save_fn=flavor_module.save_model, 
                   load_fn=flavor_module.load_model,
                   load_pyfunc_fn=(
                       flavor_module._load_pyfunc if hasattr(flavor_module, "_load_pyfunc") 
                       else None),
                   simple=False)

    @classmethod
    def from_model(cls, flavor_name, model_path, run_id=None):
        if run_id is not None:
            model_path = _get_model_log_dir(model_path, run_id)
        flavor_module_path = os.path.join(model_path, "flavor_modules", "python", flavor_name)

        if not os.path.exists(flavor_module_path):
            raise MlflowException(
                    "The specified model does not contain a serialized Flavor with name:"
                    " {flavor_name}".format(flavor_name=flavor_name))

        save_fn_subpath = os.path.join(flavor_module_path, "save.pkl")
        if not os.path.exists(save_fn_subpath):
            raise MlflowException(
                    "The configuration for the specified flavor does not contain a `save_model`"
                    " function.")
        with open(os.path.join(model_path, save_fn_subpath), "rb") as save_in:
            save_fn = cloudpickle.load(save_in)

        load_fn_subpath = os.path.join(flavor_module_path, "load.pkl")
        if not os.path.exists(load_fn_subpath):
            raise MlflowException(
                    "The configuration for the specified flavor does not contain a `load_model`"
                    " function.")
        with open(os.path.join(model_path, load_fn_subpath), "rb") as load_in:
            load_fn = cloudpickle.load(load_in)

        load_pyfunc_fn_subpath = os.path.join(flavor_module_path, "load_pyfunc.pkl")
        if os.path.exists(load_pyfunc_fn_subpath):
            with open(os.path.join(model_path, load_pyfunc_fn_subpath), "rb") as load_pyfunc_in:
                load_pyfunc_fn = cloudpickle.load(load_pyfunc_in)
        else:
            load_pyfunc_fn = None


        return cls(name=flavor_name, 
                   save_fn=save_fn, 
                   load_fn=load_fn, 
                   load_pyfunc_fn=load_pyfunc_fn,
                   simple=False)
