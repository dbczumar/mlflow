import os

from mlflow.projects import SubmittedRun
from mlflow.projects.backend.abstract_backend import AbstractBackend
from mlflow.protos.projects_pb2 import (
    ProjectsService, ProjectParameter, RunProject,
)
from mlflow.tracking._tracking_service.utils import (
    _TRACKING_USERNAME_ENV_VAR, _TRACKING_PASSWORD_ENV_VAR, _TRACKING_TOKEN_ENV_VAR,
    _TRACKING_INSECURE_TLS_ENV_VAR,
)
from mlflow.utils import rest_utils
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import call_endpoint, extract_api_info_for_service

_PATH_PREFIX = "/api/2.0"
_METHOD_TO_INFO = extract_api_info_for_service(ProjectsService, _PATH_PREFIX)


def _get_rest_backend(backend_uri):
    def get_default_host_creds():
        return rest_utils.MlflowHostCreds(
            host=backend_uri,
            username=os.environ.get(_TRACKING_USERNAME_ENV_VAR),
            password=os.environ.get(_TRACKING_PASSWORD_ENV_VAR),
            token=os.environ.get(_TRACKING_TOKEN_ENV_VAR),
            ignore_tls_verification=os.environ.get(_TRACKING_INSECURE_TLS_ENV_VAR) == 'true',
        )
    return RestBackend(get_default_host_creds)


class RestBackend(AbstractBackend):

    def __init__(self, get_host_creds):
        self.get_host_creds = get_host_creds

    def run(self, project_uri, entry_point, params, experiment_id, tracking_backend_store_uri,
            run_id=None, version=None, backend_config=None):
        param_protos = []
        for key, value in params.items():
            param_proto = ProjectParameter(key=key, value=value)
            param_protos.append(param_proto)

        run_project = RunProject(
            project=project_uri,
            parameters=param_protos,
            entry_point=entry_point,
            version=version,
            experiment_id=experiment_id,
            run_id=run_id,
            config=backend_config,
        )
        req_body = message_to_json(run_project)
        response = self._call_endpoint(RunProject, req_body)
        return RestSubmittedRun(response.run.run_id)

    def _call_endpoint(self, api, json_body):
        endpoint, method = _METHOD_TO_INFO[api]
        response_proto = api.Response()
        return call_endpoint(
            self.get_host_creds(), endpoint, method, json_body, response_proto)


class RestSubmittedRun(SubmittedRun):
    """
    """
    def __init__(self, run_id):
        super(RestSubmittedRun, self).__init__()
        self._run_id = run_id

    @property
    def run_id(self):
        return self._run_id

    def wait(self):
        raise NotImplementedError("Not implemented!")

    def cancel(self):
        raise NotImplementedError("Not implemented!")

    def get_status(self):
        raise NotImplementedError("Not implemented!")
