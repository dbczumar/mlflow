import os
import shutil
from importlib import import_module

import cloudpickle

import mlflow.pyfunc
from mlflow.models import Model
from mlflow.pyfunc.wrappers import BaseModelWrapper
from mlflow.tracking.utils import _get_model_log_dir
from mlflow.utils.file_utils import TempDir 
from mlflow.utils.file_utils import _copy_file_or_tree 
from mlflow.utils.model_utils import _get_flavor_configuration


class ModelExtender:

    _contexts = 0

    def __init__(self, model_path, run_id=None):
        if run_id:
            model_path = _get_model_log_dir(model_path, run_id)
        else:
            model_path = os.path.abspath(model_path)

        self.model_path = model_path
        self.run_id = run_id
        self.pyfunc_model = None
        
        self.entered = False 
        self.tmp_model_dir = TempDir()
        self.tmp_model_path = None

        self.predict_fn = None
        self.objects = {}

    def __enter__(self):
        if ModelExtender._contexts > 0:
            raise Exception("Nested extension is not supported")
        ModelExtender._contexts += 1

        if self.entered:
            raise Exception("Context cannot be re-entered!")
        self.entered = True

        # Load the pyfunc model
        self.pyfunc_model = mlflow.pyfunc.load_pyfunc(self.model_path)
        self.predict_fn = lambda model, objects, input_data: self.pyfunc_model.predict(input_data)

        # Create a temporary directory housing model artifacts
        self.tmp_model_dir.__enter__()
        tmp_model_base_path = self.tmp_model_dir.path()
        tmp_model_subpath = _copy_file_or_tree(src=self.model_path, dst=tmp_model_base_path, dst_dir="")
        self.tmp_model_path = os.path.join(tmp_model_base_path, tmp_model_subpath)

        return self

    def __exit__(self, *args, **kwargs):
        ModelExtender._contexts -= 1
        self.tmp_model_dir.__exit__(*args, **kwargs)

    def set_predict(self, predict_fn):
        self.predict_fn = predict_fn

    def add_object(self, name, path, loader_module):
        self.objects[name] = { "path": path, "loader_module": loader_module }

    def predict(self, input_data):
        if not self.predict_fn:
            raise Exception
        
        return self.predict_fn(self.pyfunc_model, self.objects, input_data)

    def save(self, path, code=None, conda_env=None):
        path = os.path.abspath(path)
        if os.path.exists(path):
            raise Exception

        parent_pyfunc_conf = _get_flavor_configuration(
                model_path=self.tmp_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME)

        os.makedirs(path)
        parent_model_subpath = _copy_file_or_tree(
                src=self.tmp_model_path, dst=path, dst_dir="parent_model")
        parent_model_path = os.path.join(path, parent_model_subpath)

        if conda_env is None and mlflow.pyfunc.ENV in parent_pyfunc_conf:
            conda_env = os.path.join(parent_model_path, mlflow.pyfunc.ENV)
        if conda_env is not None:
            conda_env_subpath = "conda_env.yaml"
            shutil.copy(src=conda_env, dst=os.path.join(path, conda_env_subpath))
        else:
            conda_env_subpath = None

        if code is not None:
            code_subpath = "code"
            for code_path in code:
                _copy_file_or_tree(src=code_path, dst=path, dst_dir=code_subpath)
        else:
            code_subpath = None

        predict_fn_subpath = "predict.pkl"
        with open(os.path.join(path, predict_fn_subpath), "wb") as f:
            cloudpickle.dump(self.predict_fn, f)

        model_conf = Model()
        pyfunc_conf_kwargs = {
            "model": model_conf,
            "loader_module": __name__,
            "parent_model": parent_model_subpath,
            "predict_fn": predict_fn_subpath,
            "env": conda_env_subpath,
            "data": None,
        }

        objects_subpath = "objects"

        saved_objects = {}
        for object_name, object_info in self.objects.items():
            object_src_path = object_info["path"]

            object_dst_subdir = os.path.join(objects_subpath, object_name)
            object_dst_dir = os.path.join(path, object_dst_subdir)
            os.makedirs(object_dst_dir)
            object_dst_subpath = os.path.join(object_dst_subdir, _copy_file_or_tree(src=object_src_path, dst=object_dst_dir))
            saved_objects[object_name] = {
                "path": object_dst_subpath,
            }
                
            object_loader_module = object_info["loader_module"]
            if callable(object_loader_module):
                saved_loader_module_basename = "loader_module.pkl"
                with open(os.path.join(object_dst_dir, saved_loader_module_basename), "wb") as f:
                    cloudpickle.dump(object_loader_module, f)
                saved_objects[object_name]["loader_module_type"] = "function"
                saved_objects[object_name]["loader_module"] = os.path.join(object_dst_subdir, saved_loader_module_basename) 
            elif isinstance(object_loader_module, str) or isinstance(object_loader_module, unicode):
                saved_objects[object_name]["loader_module_type"] = "string"
                saved_objects[object_name]["loader_module"] = object_loader_module
            else:
                raise Exception

        if len(saved_objects) > 0:
            pyfunc_conf_kwargs["objects"] = saved_objects
        mlflow.pyfunc.add_to_model(**pyfunc_conf_kwargs)
        model_conf.save(os.path.join(path, "MLmodel"))

    def log(self, artifact_path):
        pass


class WrappedPyfuncModel(BaseModelWrapper):

    def __init__(self, parent_model, objects, predict_fn):
        super(WrappedPyfuncModel, self).__init__()
        self.parent_model = parent_model 
        self.objects = objects 
        self.predict_fn = predict_fn

    def predict(self, input_data):
        return self.predict_fn(self.parent_model, self.objects, input_data)

    @property
    def base_model(self):
        return self.parent_model

    @property
    def base_flavor(self):
        return mlflow.pyfunc.FLAVOR_NAME


def _load_pyfunc(model_root_path):
    pyfunc_conf = _get_flavor_configuration(model_root_path, mlflow.pyfunc.FLAVOR_NAME)

    assert "parent_model" in pyfunc_conf
    parent_model_path = os.path.join(model_root_path, pyfunc_conf["parent_model"])
    parent_model = mlflow.pyfunc.load_pyfunc(parent_model_path)

    assert "predict_fn" in pyfunc_conf
    predict_fn_path = os.path.join(model_root_path, pyfunc_conf["predict_fn"])
    with open(predict_fn_path, "rb") as f:
        predict_fn = cloudpickle.load(f)

    saved_objects = pyfunc_conf.get("objects", {})
    objects = {}
    for object_name, object_info in saved_objects.items():
        if object_info["loader_module_type"] == "function":
            with open(os.path.join(model_root_path, object_info["loader_module"]), "rb") as f:
                loader_module = cloudpickle.load(f)
        elif object_info["loader_module_type"] == "string":
            split_loader_module_name = object_info["loader_module"].split(".")
            module = ".".join(split_loader_module_name[:-1])
            function_name = split_loader_module_name[-1]
            loader_module = getattr(import_module(module), function_name)
        else:
            raise Exception
        objects[object_name] = loader_module(os.path.join(model_root_path, object_info["path"]))

    return WrappedPyfuncModel(parent_model=parent_model, objects=objects, predict_fn=predict_fn)
