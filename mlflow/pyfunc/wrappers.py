import mlflow.pyfunc

from abc import ABCMeta, abstractmethod

class BaseModelWrapper(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.model_path = None 
        self.run_id = None 

    @property
    @abstractmethod
    def base_model(self):
        pass

    @property
    @abstractmethod
    def base_flavor(self):
        pass

    def extend(self, predict_fn):
        if self.model_path is None:
            raise ValueError("Model path must be specified!")
        return ExtendedModel(parent_model=self, predict_fn=predict_fn, 
                             parent_model_path=self.model_path, parent_run_id=self.run_id)

    def _set_model_path(self, model_path):
        self.model_path = model_path

    def _set_run_id(self, run_id):
        self.run_id = run_id


def extend_model(model_path, run_id=None):
    return ModelExtender(model_path, run_id)

import os
import shutil

from mlflow.models import Model
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
        self.data = {}

    def __enter__(self):
        if ModelExtender._contexts > 0:
            raise Exception("Nested extension is not supported")
        ModelExtender._contexts += 1

        if self.entered:
            raise Exception("Context cannot be re-entered!")
        self.entered = True

        # Load the pyfunc model
        self.pyfunc_model = mlflow.pyfunc.load_pyfunc(self.model_path)
        self.predict_fn = lambda model, data, input_data: self.pyfunc_model.predict(input_data)

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

    def add_data(self, name, path, loader_module):
        self.data[name] = { "path": path, "loader_module": loader_module}

    def predict(self, input_data):
        if not self.predict_fn:
            raise Exception
        
        return self.predict_fn(self.model, self.data, input_data)

    def save(self, path, code=None, conda_env=None):
        path = os.path.abspath(path)
        if os.path.exists(path):
            raise Exception

        parent_pyfunc_conf = _get_flavor_configuration(
                model_path=self.tmp_model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME)

        os.makedirs(path)
        parent_model_subpath = _copy_file_or_tree(
                src=self.tmp_model_path, dst=path, dst_dir="base_model")
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

        model_conf = Model()
        mlflow.pyfunc.add_to_model(
                model=model_conf, loader_module=__name__, 
                env=conda_env_subpath, code=code_subpath, data=None)
        model_conf.save(os.path.join(path, "MLmodel"))

    def log(self, artifact_path):
        pass
