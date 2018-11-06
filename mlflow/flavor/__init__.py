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
        """
        :param save_fn: A `save_model` function that persists a model to a specified location.
                        If `simple` is True, this function only needs to output serialized
                        model data. If `simple` is False, this function also needs to construct
                        an "MLmodel" configuration, (optionally) add the `pyfunc` flavor to the
                        configuration, and persist the configuration to the specified location.
                        Required function arguments:
                            - `path`: The absolute path to which to persist the specified model

        :param load_fn: A `load_model` function that loads a persisted model MLflow from a specified
                        location. If `simple` is True, this function only needs to load serialized
                        model data. If `simple` is False, this function first needs to parse the 
                        model's "MLmodel" configuration, perform all the necessary work to resolve 
                        serialized model data paths, and load the serialized data.
                        Required function arguments:
                            - `path`: The run-relative path to the MLflow model 
                            - (If `simple` is `False`) `run_id`: The run id of the MLflow model 

        :param load_pyfunc_fn: A `_load_pyfunc` function that loads a persisted model as a python
                               function from a specified location. If `simple` is True, this
                               function only needs to load serialized model data as a python 
                               function wrapper. If `simple` is False, this function first needs to
                               parse the model's "MLmodel" configuration, perform all the necessary 
                               work to resolve serialized model data paths, and load the serialized 
                               data.        
                               Required function arguments:
                                    - `path`: If `simple` is `True`, this is the path to the
                                              serialized model data. If `simple` is `False`,
                                              the path will depend on the model's pyfunc
                                              configuration format, as defined by `save_fn`.

        :param simple: If `True`, interprets `save_fn`, `load_fn`, and `load_pyfunc_fn` as functions
                       that only operate on model data; these functions will be wrapped to 
                       produce complete `save_model`, `load_model`, and `_load_pyfunc` functions
                       that also output "MLmodel" configurations and provide portability guarantees.
                       If `False`, interpets `save_fn`, `load_fn`, and `load_pyfunc_fn` as complete
                       `save_model`, `load_model`, and `_load_pyfunc` functions respectively; these
                       functions are assumed to produce "MLmodle" configurations and provide
                       model portability.
        
        >>> ### `simple=True` example ###
        >>> def save_model(path, sk_model):
        >>>     import cloudpickle
        >>>     with open(os.path.join(path, "skmodel.pkl"), "wb") as f:
        >>>         cloudpickle.dump(sk_model, f)
        >>>
        >>> def load_model(path):
        >>>     import cloudpickle
        >>>     with open(os.path.join(path, "skmodel.pkl"), "rb") as f:
        >>>         return cloudpickle.load(f)
        >>>
        >>> def load_pyfunc(path):
        >>>     return load_model(path)
        >>> 
        >>> # These functions only operate on model data. Their `path` arguments are assumed
        >>> # to be direct input/output paths referencing sklearn model data. Therefore, they
        >>> # conform to the `simple` format.
        >>> simple_flavor = Flavor(name="sklearnsimple", save_fn=save_model, load_fn=load_model,
        >>>                        load_pyfunc_fn=load_pyfunc, simple=True)
        >>>
        >>> ### `simple=False` example ###
        >>> import mlflow.sklearn
        >>>
        >>> # The `save_model`, `load_model`, and `_load_pyfunc` functions in the `mlflow.sklearn`
        >>> # model are "complete" in the sense that, in addition to serializing model data, they 
        >>> # produce model configurations, handle absolute path resolution from run_ids /
        >>> # run-relative paths, and provide portability. Therefore, they do not conform to the
        >>> # `simple` format.
        >>> complete_flavor = Flavor(
        >>>     name="sklearncomplete", save_fn=mlflow.sklearn.save_model,
        >>>     load_fn=mlflow.sklearn.load_model, load_pyfunc_fn=mlflow.sklearn._load_pyfunc,
        >>>     simple=False)
        """
        if simple:
            save_fn, load_fn, load_pyfunc_fn = Flavor._convert_simple_functions(
                    flavor_name=name, save_fn=save_fn, load_fn=load_fn, 
                    load_pyfunc_fn=load_pyfunc_fn)

        self.save_model = Flavor._wrap_save_function(
                flavor_name=name, save_fn=save_fn, load_fn=load_fn, load_pyfunc_fn=load_pyfunc_fn)
        update_wrapper(self.save_model, save_fn)
        self.load_model = load_fn
        self._load_pyfunc = load_pyfunc_fn

    def log_model(self, path, **kwargs):
        return Model.log(artifact_path=path, flavor=self, **kwargs) 


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
