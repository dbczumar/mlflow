import json

from six.moves import urllib

from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.store.artifact_repo import ArtifactRepository

from mlflow.protos.model_registry_pb2 import ModelVersionDetailed, \
    GetModelVersionDetails, RegisteredModel, ModelVersion, ModelRegistryService, \
    GetRegisteredModelDetails, GetLatestVersions

from mlflow.utils.rest_utils import http_request, verify_rest_response
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.tracking import utils

from mlflow.store.rest_store import RestStore


def _get_path(endpoint_path):
    return "/api/2.0{}".format(endpoint_path)


def _api_method_to_info():
    """ Return a dictionary mapping each API method to a tuple (path, HTTP method)"""
    service_methods = ModelRegistryService.DESCRIPTOR.methods
    res = {}
    for service_method in service_methods:
        endpoints = service_method.GetOptions().Extensions[databricks_pb2.rpc].endpoints
        endpoint = endpoints[0]
        endpoint_path = _get_path(endpoint.path)
        res[ModelRegistryService().GetRequestClass(service_method)] =\
            (endpoint_path, endpoint.method)
    return res


_METHOD_TO_INFO = _api_method_to_info()


class ModelRegistryArtifactRepository(ArtifactRepository):

    URI_TYPE_STAGE = "stage"
    URI_TYPE_VERSION = "version"

    def __init__(self, artifact_uri):
        self.get_host_creds = self._get_host_creds_from_default_store()

        from mlflow.store.artifact_repository_registry import get_artifact_repository
        model_name, stage_or_version, uri_type =\
            ModelRegistryArtifactRepository.parse_uri(artifact_uri)
        source_uri = self.get_model_source_uri(
            model_name=model_name,
            stage_or_version=stage_or_version,
            uri_type=uri_type)
        assert urllib.parse.urlparse(source_uri).scheme != "models"  # avoid an infinite loop
        super(ModelRegistryArtifactRepository, self).__init__(artifact_uri)
        self.repo = get_artifact_repository(source_uri)

    def _get_host_creds_from_default_store(self):
        store = utils._get_store()
        if not isinstance(store, RestStore):
            raise MlflowException('Failed to get credentials for DBFS; they are read from the ' +
                                  'Databricks CLI credentials or MLFLOW_TRACKING* environment ' +
                                  'variables.')
        return store.get_host_creds

    def _call_endpoint(self, api, json_body):
        endpoint, method = _METHOD_TO_INFO[api]
        response_proto = api.Response()
        # Convert json string to json dictionary, to pass to requests
        if json_body:
            json_body = json.loads(json_body)
        host_creds = self.get_host_creds()

        if method == 'GET':
            response = http_request(
                host_creds=host_creds, endpoint=endpoint, method=method, params=json_body)
        else:
            response = http_request(
                host_creds=host_creds, endpoint=endpoint, method=method, json=json_body)

        response = verify_rest_response(response, endpoint)

        js_dict = json.loads(response.text)
        parse_dict(js_dict=js_dict, message=response_proto)
        return response_proto

    def get_model_source_uri(self, model_name, stage_or_version, uri_type):
        if uri_type == ModelRegistryArtifactRepository.URI_TYPE_VERSION:
            req_body = message_to_json(
                GetModelVersionDetails(
                    model_version=(
                        ModelVersion(
                            version=int(stage_or_version),
                            registered_model=RegisteredModel(
                                name=model_name,
                            )
                        )
                    )
                )
            )

            response_proto = self._call_endpoint(GetModelVersionDetails, req_body)
            return response_proto.model_version_detailed.source

        elif uri_type == ModelRegistryArtifactRepository.URI_TYPE_STAGE:
            req_body = message_to_json(
               GetLatestVersions(
                   registered_model=RegisteredModel(
                       name=model_name,
                   ),
                   stages=[stage_or_version],
                )
            )

            response_proto = self._call_endpoint(GetLatestVersions, req_body)
            model_versions = response_proto.model_versions_detailed

            if len(model_versions) > 0:
                return model_versions[0].source
            else:
                raise MlflowException(
                    "Found no registered models matching stage: {}".format(stage_or_version))

        else:
            raise MlflowException("Invalid models URI") 


    @staticmethod
    def parse_uri(uri):
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "models":
            raise MlflowException(
                "Not a proper models:/ URI: %s. " % uri +
                "Models URIs must be of the form 'models:/<model_name>/<stage/version>/<stage_or_version>'")

        path = parsed.path
        if not path.startswith('/') or len(path) <= 1:
            raise MlflowException(
                "Not a proper models:/ URI: %s. " % uri +
                "Models URIs must be of the form 'models:/<model_name>/<stage/version>/<stage_or_version>'")

        path = path[1:]
        path_parts = path.split('/')
        if len(path_parts) != 3:
            raise MlflowException(
                "Not a proper models:/ URI: %s. " % uri +
                "Models URIs must be of the form 'models:/<model_name>/<stage/version>/<stage_or_version>'")

        uri_type = path_parts[1].lower()
        if uri_type in [
            ModelRegistryArtifactRepository.URI_TYPE_STAGE,
            ModelRegistryArtifactRepository.URI_TYPE_VERSION
        ]:
            return path_parts[0], path_parts[2], uri_type
        else:
            raise MlflowException(
                "Not a proper models:/ URI: %s. " % uri +
                "Models URIs must be of the form 'models:/<model_name>/<stage/version>/<stage_or_version>'")


    def log_artifact(self, local_file, artifact_path=None):
        """
        Log a local file as an artifact, optionally taking an ``artifact_path`` to place it in
        within the run's artifacts. Run artifacts can be organized into directories, so you can
        place the artifact in a directory this way.

        :param local_file: Path to artifact to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifact
        """
        self.repo.log_artifact(local_file, artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.

        :param local_dir: Directory of local artifacts to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifacts
        """
        self.repo.log_artifacts(local_dir, artifact_path)

    def list_artifacts(self, path):
        """
        Return all the artifacts for this run_id directly under path. If path is a file, returns
        an empty list. Will error if path is neither a file nor directory.

        :param path: Relative source path that contain desired artifacts

        :return: List of artifacts as FileInfo listed directly under path.
        """
        return self.repo.list_artifacts(path)

    def download_artifacts(self, artifact_path, dst_path=None):
        """
        Download an artifact file or directory to a local directory if applicable, and return a
        local path for it.
        The caller is responsible for managing the lifecycle of the downloaded artifacts.

        :param artifact_path: Relative source path to the desired artifacts.
        :param dst_path: Absolute path of the local filesystem destination directory to which to
                         download the specified artifacts. This directory must already exist.
                         If unspecified, the artifacts will either be downloaded to a new
                         uniquely-named directory on the local filesystem or will be returned
                         directly in the case of the LocalArtifactRepository.

        :return: Absolute path of the local filesystem location containing the desired artifacts.
        """
        return self.repo.download_artifacts(artifact_path, dst_path)

    def _download_file(self, remote_file_path, local_path):
        """
        Download the file at the specified relative remote path and saves
        it at the specified local path.

        :param remote_file_path: Source path to the remote file, relative to the root
                                 directory of the artifact repository.
        :param local_path: The path to which to save the downloaded file.
        """
        self.repo._download_file(remote_file_path, local_path)
