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


class ExtendedModel(BaseModelWrapper):

    def __init__(self, parent_model, predict_fn, parent_model_path, parent_run_id):
        super(ExtendedModel, self).__init__()
        self.parent_model = parent_model
        self.predict_fn = predict_fn
        self.parent_model_path = parent_model_path
        self.parent_run_id = parent_run_id

    def predict(self, input_data):
        return self.predict_fn(self.parent_model, input_data)

    @property
    def base_flavor(self):
        return mlflow.pyfunc.FLAVOR_NAME

    @property
    def base_model(self):
        return self.parent_model

    def get_parent_model_path(self):
        """
        Docs
        """
        return self.parent_model_path

    def get_parent_run_id(self):
        return self.parent_run_id

    # def save(self, path, code=None, conda_env=None):
    #     path = os.path.abspath(path)
    #     if os.path.exists(path):
    #         raise Exception
    #
    #     if run_id is not None:
    #         base_model_path = tracking.utils._get_model_log_dir(base_model_path, run_id)
    #
    #     base_pyfunc_conf = _get_flavor_configuration(
    #             model_path=base_model_path, flavor_name=FLAVOR_NAME)
    #
    #     os.makedirs(path)
    #     base_model_subpath = _copy_file_or_tree(src=base_model_path, dst=path, dst_dir="base_model")
    #     base_model_path = os.path.join(path, base_model_subpath)
    #
    #     if conda_env is None and ENV in base_pyfunc_conf:
    #         conda_env = os.path.join(base_model_path, ENV)
    #     if conda_env is not None:
    #         conda_env_subpath = "conda_env.yaml"
    #         shutil.copy(src=conda_env, dst=os.path.join(path, conda_env_subpath))
    #     else:
    #         conda_env_subpath = None
    #
    #     if code is not None:
    #         code_subpath = "code"
    #         for code_path in code:
    #             _copy_file_or_tree(src=code_path, dst=path, dst_dir=code_subpath)
    #     else:
    #         code_subpath = None
    #
    #     model_conf = Model()
    #     add_to_model(model=model_conf, loader_module="mlflow.pyfunc.wrapped_model", 
    #                  env=conda_env_subpath, code=code_subpath, data=None)
    #     model_conf.save(os.path.join(path, "MLmodel"))

    
    # def log(
    #


# def _load_pyfunc(model_root_path):
#     pyfunc_conf = _get_flavor_configuration(model_root_path, mlflow.pyfunc.FLAVOR_NAME) 
#
#     parent_model_path = os.path.join(model_root_path, pyfunc_conf["parent_model"])
#     parent_model = mlflow.pyfunc.load_pyfunc(path=parent_model_path)
#
#     predict_fn_path = os.path.join(model_root_path, pyfunc_conf["predict_fn"])
#     with open(predict_fn_path, "rb") as f:
#         predict_fn = cloudpickle.load(f)
#
#     return PyfuncModel(parent_model=parent_model, predict_fn=predict_fn)


def extend_model(model_path, run_id=None):
    return ModelExtender(model_path, run_id)


class ModelExtender:

    _contexts = 0

    def __init__(self, model_path, run_id=None):
        self.model_path = model_path
        self.run_id = run_id

    def __enter__(self):
        # Create a temporary directory housing model artifacts
        if ModelExtender._contexts > 0:
            raise Exception("Nested extension is not supported")
        ModelExtender._contexts += 1
        return self

    def __exit__(self, *args, **kwargs):
        ModelExtender._contexts -= 1
        pass
