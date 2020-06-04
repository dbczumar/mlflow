from azure.storage.blob import BlobClient
from azure.core.exceptions import ClientAuthenticationError

import os
import uuid
import base64
import logging
import requests
import posixpath

from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.protos.databricks_artifacts_pb2 import DatabricksMlflowArtifactsService, \
    GetCredentialsForWrite, GetCredentialsForRead, ArtifactCredentialType
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.service_pb2 import MlflowService, ListArtifacts
from mlflow.utils.uri import extract_and_normalize_path, is_databricks_acled_artifacts_uri
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.file_utils import relative_path_to_artifact_path, yield_file_in_chunks
from mlflow.utils.rest_utils import call_endpoint, extract_api_info_for_service
from mlflow.utils.databricks_utils import get_databricks_host_creds

_logger = logging.getLogger(__name__)
_PATH_PREFIX = "/api/2.0"
_AZURE_MAX_BLOCK_CHUNK_SIZE = 100000000  # Max. size of each block allowed is 100 MB in stage_block
_DOWNLOAD_CHUNK_SIZE = 100000000
_SERVICE_AND_METHOD_TO_INFO = {
    service: extract_api_info_for_service(service, _PATH_PREFIX)
    for service in [MlflowService, DatabricksMlflowArtifactsService]
}


class DatabricksArtifactRepository(ArtifactRepository):
    """
    Performs storage operations on artifacts in the access-controlled
    `dbfs:/databricks/mlflow-tracking` location.

    Signed access URIs for S3 / Azure Blob Storage are fetched from the MLflow service and used to
    read and write files from/to this location.

    The artifact_uri is expected to be of the form
    dbfs:/databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/artifacts/
    """

    def __init__(self, artifact_uri):
        super(DatabricksArtifactRepository, self).__init__(artifact_uri)
        if not artifact_uri.startswith('dbfs:/'):
            raise MlflowException(message='DatabricksArtifactRepository URI must start with dbfs:/',
                                  error_code=INVALID_PARAMETER_VALUE)
        if not is_databricks_acled_artifacts_uri(artifact_uri):
            raise MlflowException(message=('Artifact URI incorrect. Expected path prefix to be'
                                           ' databricks/mlflow-tracking/path/to/artifact/..'),
                                  error_code=INVALID_PARAMETER_VALUE)
        self.run_id = self._extract_run_id(self.artifact_uri)

    @staticmethod
    def _extract_run_id(artifact_uri):
        """
        The artifact_uri is expected to be
        dbfs:/databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/artifacts/<path>
        Once the path from the input uri is extracted and normalized, it is
        expected to be of the form
        databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/artifacts/<path>

        Hence the run_id is the 4th element of the normalized path.

        :return: run_id extracted from the artifact_uri
        """
        artifact_path = extract_and_normalize_path(artifact_uri)
        return artifact_path.split('/')[3]

    def _call_endpoint(self, service, api, json_body):
        endpoint, method = _SERVICE_AND_METHOD_TO_INFO[service][api]
        response_proto = api.Response()
        return call_endpoint(get_databricks_host_creds(),
                             endpoint, method, json_body, response_proto)

    def _get_write_credentials(self, run_id, path=None):
        json_body = message_to_json(GetCredentialsForWrite(run_id=run_id, path=path))
        return self._call_endpoint(DatabricksMlflowArtifactsService,
                                   GetCredentialsForWrite, json_body)

    def _get_read_credentials(self, run_id, path=None):
        json_body = message_to_json(GetCredentialsForRead(run_id=run_id, path=path))
        return self._call_endpoint(DatabricksMlflowArtifactsService,
                                   GetCredentialsForRead, json_body)

    def _extract_headers_from_credentials(self, credential):
        headers = dict()
        for header in credential.headers:
            headers[header.name] = header.value
        print(headers)
        return headers

    def _azure_upload_file(self, credentials, local_file, artifact_path):
        """
        Uploads a file to a given Azure storage location.

        The function uses a file chunking generator with 100 MB being the size limit for each chunk.
        This limit is imposed by the stage_block API in azure-storage-blob.
        In the case the file size is large and the upload takes longer than the validity of the
        given credentials, a new set of credentials are generated and the operation continues. This
        is the reason for the first nested try-except block

        Finally, since the prevailing credentials could expire in the time between the last
        stage_block and the commit, a second try-except block refreshes credentials if needed.
        """
        try:
            headers = self._extract_headers_from_credentials(credentials)
            service = BlobClient.from_blob_url(blob_url=credentials.signed_uri, credential=None,
                                               headers=headers)
            uploading_block_list = list()
            for chunk in yield_file_in_chunks(local_file, _AZURE_MAX_BLOCK_CHUNK_SIZE):
                block_id = base64.b64encode(uuid.uuid4().hex.encode())
                try:
                    service.stage_block(block_id, chunk, headers=headers)
                except ClientAuthenticationError:
                    _logger.warning(
                        "Failed to authorize request, possibly due to credential expiration."
                        "Refreshing credentials and trying again..")
                    credentials = self._get_write_credentials(self.run_id,
                                                              artifact_path).credentials.signed_uri
                    service = BlobClient.from_blob_url(blob_url=credentials, credential=None)
                    service.stage_block(block_id, chunk, headers=headers)
                uploading_block_list.append(block_id)
            try:
                service.commit_block_list(uploading_block_list)
            except ClientAuthenticationError:
                _logger.warning(
                    "Failed to authorize request, possibly due to credential expiration."
                    "Refreshing credentials and trying again..")
                credentials = self._get_write_credentials(self.run_id,
                                                          artifact_path).credentials.signed_uri
                service = BlobClient.from_blob_url(blob_url=credentials, credential=None)
                service.commit_block_list(uploading_block_list)
        except Exception as err:
            raise MlflowException(err)

    def _aws_upload_file(self, credentials, local_file):
        try:
            headers = self._extract_headers_from_credentials(credentials)
            signed_write_uri = credentials.signed_uri
            print(signed_write_uri)
            # headers = {
            #   "x-amz-server-side-encryption": "AES256",
            # }
            # signed_write_uri = "https://databricks-dev-storage-oregon.s3-us-west-2.amazonaws.com/mlflow-tracking/MlflowArtifactManagerIntegrationSuite-1161411340/myExperiment/myRun/my/file?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200603T213808Z&X-Amz-SignedHeaders=host%3Bx-amz-server-side-encryption&X-Amz-Expires=899&X-Amz-Credential=AKIA2JMHUIXTV5R3VBGC%2F20200603%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Signature=a2e3d996a6a2126a1b5c41b62a258ec902ed0ab3fecec41fc539fda6d273834c"
            # signed_write_uri = "https://databricks-dev-storage-oregon.s3-us-west-2.amazonaws.com/mlflow-tracking/MlflowArtifactManagerIntegrationSuite-9486667483/myExperiment/myRun/my/file?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200603T221413Z&X-Amz-SignedHeaders=host%3Bx-amz-server-side-encryption&X-Amz-Expires=899&X-Amz-Credential=AKIA2JMHUIXT5ZZ3XNPN%2F20200603%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Signature=bb7a788fd4936fa403b1d6277134e18914b55b1c27535dd1c09de6b5ef792057"
            with open(local_file, 'rb') as file:
                put_request = requests.put(signed_write_uri, headers=headers, data=file)
                put_request.raise_for_status()
        except Exception as err:
            raise MlflowException(err)

    def _upload_to_cloud(self, cloud_credentials, local_file, artifact_path):
        if cloud_credentials.credentials.type == ArtifactCredentialType.AZURE_SAS_URI:
            self._azure_upload_file(cloud_credentials.credentials, local_file, artifact_path)
        elif cloud_credentials.credentials.type == ArtifactCredentialType.AWS_PRESIGNED_URL:
            self._aws_upload_file(cloud_credentials.credentials, local_file)
        else:
            raise MlflowException('Not implemented yet')

    def _download_from_cloud(self, cloud_credential, local_file_path):
        """
        Downloads a file from the input `cloud_credential` and save it to `local_path`.

        Since the download mechanism for both cloud services, i.e., Azure and AWS is the same,
        a single download method is sufficient.

        The default working of `requests.get` is to download the entire response body immediately.
        However, this could be inefficient for large files. Hence the parameter `stream` is set to
        true. This only downloads the response headers at first and keeps the connection open,
        allowing content retrieval to be made via `iter_content`.
        In addition, since the connection is kept open, refreshing credentials is not required.
        """
        if cloud_credential.type not in [ArtifactCredentialType.AZURE_SAS_URI,
                                         ArtifactCredentialType.AWS_PRESIGNED_URL]:
            raise MlflowException(message='Cloud provider not supported.',
                                  error_code=INVALID_PARAMETER_VALUE)
        try:
            signed_read_uri = cloud_credential.signed_uri
            with requests.get(signed_read_uri, stream=True) as response:
                response.raise_for_status()
                with open(local_file_path, "wb") as output_file:
                    for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                        if not chunk:
                            break
                        output_file.write(chunk)
        except Exception as err:
            raise MlflowException(err)

    def log_artifact(self, local_file, artifact_path=None):
        basename = os.path.basename(local_file)
        artifact_path = artifact_path or ""
        artifact_path = posixpath.join(artifact_path, basename)
        write_credentials = self._get_write_credentials(self.run_id, artifact_path)
        self._upload_to_cloud(write_credentials, local_file, artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        artifact_path = artifact_path or ""
        for (dirpath, _, filenames) in os.walk(local_dir):
            artifact_subdir = artifact_path
            if dirpath != local_dir:
                rel_path = os.path.relpath(dirpath, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                artifact_subdir = posixpath.join(artifact_path, rel_path)
            for name in filenames:
                file_path = os.path.join(dirpath, name)
                self.log_artifact(file_path, artifact_subdir)

    def list_artifacts(self, path=None):
        json_body = message_to_json(ListArtifacts(run_id=self.run_id, path=path))
        artifact_list = self._call_endpoint(MlflowService, ListArtifacts, json_body).files
        # If `path` is a file, ListArtifacts returns a single list element with the
        # same name as `path`. The list_artifacts API expects us to return an empty list in this
        # case, so we do so here.
        if len(artifact_list) == 1 and artifact_list[0].path == path \
                and not artifact_list[0].is_dir:
            return []
        infos = list()
        for file in artifact_list:
            artifact_size = None if file.is_dir else file.file_size
            infos.append(FileInfo(file.path, file.is_dir, artifact_size))
        return infos

    def _download_file(self, remote_file_path, local_path):
        read_credentials = self._get_read_credentials(self.run_id, remote_file_path)
        self._download_from_cloud(read_credentials.credentials, local_path)

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException('Not implemented yet')
