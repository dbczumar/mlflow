import os

import posixpath
from six.moves import urllib

from mlflow import data
from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import relative_path_to_artifact_path


class S3PresignedArtifactRepository(ArtifactRepository):
    """Stores artifacts on Amazon S3."""

    def __init__(self, artifact_uri):
        from concurrent.futures import ThreadPoolExecutor

        super(S3PresignedArtifactRepository, self).__init__(artifact_uri)
        self.executor = ThreadPoolExecutor(max_workers=4)

    @staticmethod
    def parse_s3_uri(uri):
        """Parse an S3 URI, returning (bucket, path)"""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "s3":
            raise Exception("Not an S3 URI: %s" % uri)
        path = parsed.path
        if path.startswith('/'):
            path = path[1:]
        return parsed.netloc, path

    def _get_s3_client(self):
        import boto3
        s3_endpoint_url = os.environ.get('MLFLOW_S3_ENDPOINT_URL')
        return boto3.client('s3', endpoint_url=s3_endpoint_url)

    def log_artifact(self, local_file, artifact_path=None):
        (bucket, dest_path) = data.parse_s3_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(
            dest_path, os.path.basename(local_file))
        s3_client = self._get_s3_client()
        s3_client.upload_file(local_file, bucket, dest_path)

    def _create_presigned_url_expanded(self, client_method_name, method_parameters=None,
                                       expiration=3600, http_method=None):
        """Generate a presigned URL to invoke an S3.Client method

        Not all the client methods provided in the AWS Python SDK are supported.

        :param client_method_name: Name of the S3.Client method, e.g., 'list_buckets'
        :param method_parameters: Dictionary of parameters to send to the method
        :param expiration: Time in seconds for the presigned URL to remain valid
        :param http_method: HTTP method to use (GET, etc.)
        :return: Presigned URL as string. If error, returns None.
        """
        import logging
        import boto3
        from botocore.exceptions import ClientError

        # Generate a presigned URL for the S3 client method
        s3_client = boto3.client('s3')
        try:
            response = s3_client.generate_presigned_url(ClientMethod=client_method_name,
                                                        Params=method_parameters,
                                                        ExpiresIn=expiration,
                                                        HttpMethod=http_method)
        except ClientError as e:
            logging.error(e)
            return None

        # The response contains the presigned URL
        return response

    def _upload_artifact(self, local_path, bucket, key):
        presigned_url = self._create_presigned_url_expanded(
            client_method_name="put_object",
            method_parameters={
                "Bucket": bucket,
                "Key": key,
            })

        print("Uploading '{}' to '{}'".format(local_path, presigned_url))

        import requests
        with open(local_path, "rb") as f_handle:
            requests.put(presigned_url, data=f_handle)

    def log_artifacts(self, local_dir, artifact_path=None):
        (bucket, dest_path) = data.parse_s3_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)

        s3_client = self._get_s3_client()
        local_dir = os.path.abspath(local_dir)
        for (root, _, filenames) in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                upload_path = posixpath.join(dest_path, rel_path)
            for f in filenames:
                # CONCURRENT 
                dst_key = posixpath.join(upload_path, f)
                local_path = os.path.join(root, f)
                self.executor.submit(self._upload_artifact, local_path, bucket, dst_key) 

                
                # SINGLE THREAD
                # fpath = os.path.join(root, f)
                # presigned_url = self._create_presigned_url_expanded(
                #     client_method_name="put_object",
                #     method_parameters={
                #         "Bucket": bucket,
                #         "Key": posixpath.join(upload_path, f),
                #     })
                #
                # print("Uploading '{}' to '{}'".format(fpath, presigned_url))
                #
                # import requests
                # with open(fpath, "rb") as f_handle:
                #     requests.put(presigned_url, data=f_handle)

    def list_artifacts(self, path=None):
        (bucket, artifact_path) = data.parse_s3_uri(self.artifact_uri)
        dest_path = artifact_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        infos = []
        prefix = dest_path + "/" if dest_path else ""
        s3_client = self._get_s3_client()
        paginator = s3_client.get_paginator("list_objects_v2")
        results = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')
        for result in results:
            # Subdirectories will be listed as "common prefixes" due to the way we made the request
            for obj in result.get("CommonPrefixes", []):
                subdir_path = obj.get("Prefix")
                self._verify_listed_object_contains_artifact_path_prefix(
                    listed_object_path=subdir_path, artifact_path=artifact_path)
                subdir_rel_path = posixpath.relpath(
                    path=subdir_path, start=artifact_path)
                if subdir_rel_path.endswith("/"):
                    subdir_rel_path = subdir_rel_path[:-1]
                infos.append(FileInfo(subdir_rel_path, True, None))
            # Objects listed directly will be files
            for obj in result.get('Contents', []):
                file_path = obj.get("Key")
                self._verify_listed_object_contains_artifact_path_prefix(
                    listed_object_path=file_path, artifact_path=artifact_path)
                file_rel_path = posixpath.relpath(path=file_path, start=artifact_path)
                file_size = int(obj.get('Size'))
                infos.append(FileInfo(file_rel_path, False, file_size))
        return sorted(infos, key=lambda f: f.path)

    @staticmethod
    def _verify_listed_object_contains_artifact_path_prefix(listed_object_path, artifact_path):
        if not listed_object_path.startswith(artifact_path):
            raise MlflowException(
                "The path of the listed S3 object does not begin with the specified"
                " artifact path. Artifact path: {artifact_path}. Object path:"
                " {object_path}.".format(
                    artifact_path=artifact_path, object_path=listed_object_path))

    def _download_file(self, remote_file_path, local_path):
        (bucket, s3_root_path) = data.parse_s3_uri(self.artifact_uri)
        s3_full_path = posixpath.join(s3_root_path, remote_file_path)
        s3_client = self._get_s3_client()
        s3_client.download_file(bucket, s3_full_path, local_path)

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException('Not implemented yet')
