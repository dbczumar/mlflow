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

    def _set_model_path(self, model_path):
        self.model_path = model_path

    def _set_run_id(self, run_id):
        self.run_id = run_id
