import entrypoints
import logging
import warnings

from mlflow.exceptions import MlflowException
from mlflow.projects.backend.rest_backend import _get_rest_backend
from mlflow.projects.backend.huey_backend import HueyBackend 
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.uri import get_uri_scheme

PROJECT_BACKENDS_ENTRYPOINT_GROUP_NAME = "mlflow.mlproject_backend"

__logger__ = logging.getLogger(__name__)


class UnsupportedProjectBackendException(MlflowException):
    """Exception thrown when attempting to fetch an MLflow Project background that does not exist"""
    def __init__(self, unsupported_backend, supported_backends):
        message = ("Unsupported Project backend '{}'. Supported backends are: {}.").format(
            unsupported_backend, supported_backends)
        super(UnsupportedProjectBackendException, self).__init__(
            message, error_code=INVALID_PARAMETER_VALUE)
        self.supported_backends = supported_backends 


class ProjectBackendRegistry:
    """
    """

    def __init__(self):
        self._registry = {}

    def register(self, backend_str, backend_builder):
        self._registry[backend_str] = backend_builder

    def register_entrypoints(self):
        """Register tracking stores provided by other packages"""
        for entrypoint in entrypoints.get_group_all(PROJECT_BACKENDS_ENTRYPOINT_GROUP_NAME):
            try:
                self.register(entrypoint.name, entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    'Failure attempting to register backend "{}": {}'.format(
                        entrypoint.name, str(exc)
                    ),
                    stacklevel=2
                )

    def get_backend(self, backend_str):
        """
        """
        if backend_str in self._registry:
            return self._registry[backend_str](backend_str)
            
        backend_scheme = get_uri_scheme(backend_str)
        try:
            return self._registry[backend_scheme](backend_str)
        except KeyError:
            raise UnsupportedProjectBackendException(
                unsupported_backend=backend_str,
                supported_backends=list(self._registry.keys()),
            )


# Statically register backends defined in mlflow
_project_backend_registry = ProjectBackendRegistry()
_project_backend_registry.register("mlflow", _get_rest_backend)
_project_backend_registry.register("http", _get_rest_backend)
_project_backend_registry.register("https", _get_rest_backend)
_project_backend_registry.register("huey", HueyBackend)


def load_backend(backend_str):
    # Static backends
    return _project_backend_registry.get_backend(backend_str)
