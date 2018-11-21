import os
import pytest
from collections import namedtuple

import boto3
import numpy as np
from sklearn.linear_model import LogisticRegression
from moto import mock_s3, mock_ecr, mock_sts, mock_iam

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.sagemaker as mfs
from mlflow.models import Model
from mlflow.tracking.utils import _get_model_log_dir

from tests.sagemaker.mock import mock_sagemaker

TrainedModel = namedtuple("TrainedModel", ["model_path", "run_id"])


@pytest.fixture
def pretrained_model():
    model_path = "model"
    with mlflow.start_run():
        X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1, 0])
        lr = LogisticRegression()
        lr.fit(X, y)
        mlflow.sklearn.log_model(lr, model_path)
        run_id = mlflow.active_run().info.run_uuid
        return TrainedModel(model_path, run_id)


@pytest.fixture
def sagemaker_client():
    return boto3.client("sagemaker", region_name="us-west-2")


@pytest.fixture(scope='session', autouse=True)
def set_boto_credentials():
    os.environ["AWS_ACCESS_KEY_ID"] = "NotARealAccessKey"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "NotARealSecretAccessKey"
    os.environ["AWS_SESSION_TOKEN"] = "NotARealSessionToken"


def mock_sagemaker_aws_services(fn):
    import decorator

    @mock_ecr
    @mock_iam
    @mock_s3
    @mock_sagemaker
    @mock_sts
    def mock_wrapper(func, *args, **kwargs):
        # Create an ECR repository for the `mlflow-pyfunc` SageMaker docker image
        ecr_client = boto3.client("ecr", region_name="us-west-2")
        ecr_client.create_repository(repositoryName=mfs.DEFAULT_IMAGE_NAME)
        
        # Create the moto IAM role
        role_policy = """
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "*",
                    "Resource": "*"
                }
            ]
        }
        """
        iam_client = boto3.client("iam", region_name="us-west-2")
        iam_client.create_role(RoleName="moto", AssumeRolePolicyDocument=role_policy)

        return func(*args, **kwargs)

    return decorator.decorator(mock_wrapper, fn)


def test_deployment_with_unsupported_flavor_throws_value_error(pretrained_model):
    unsupported_flavor = "this is not a valid flavor"
    with pytest.raises(ValueError):
        mfs.deploy(app_name="bad_flavor",
                   model_path=pretrained_model.model_path,
                   run_id=pretrained_model.run_id,
                   flavor=unsupported_flavor)


def test_deployment_with_missing_flavor_throws_value_error(pretrained_model):
    missing_flavor = "mleap"
    with pytest.raises(ValueError):
        mfs.deploy(app_name="missing-flavor",
                   model_path=pretrained_model.model_path,
                   run_id=pretrained_model.run_id,
                   flavor=missing_flavor)


def test_deployment_of_model_with_no_supported_flavors_throws_value_error(pretrained_model):
    logged_model_path = _get_model_log_dir(pretrained_model.model_path, pretrained_model.run_id)
    model_config_path = os.path.join(logged_model_path, "MLmodel")
    model_config = Model.load(model_config_path)
    del model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    model_config.save(path=model_config_path)

    with pytest.raises(ValueError):
        mfs.deploy(app_name="missing-flavor",
                   model_path=logged_model_path,
                   flavor=None)


def test_validate_deployment_flavor_validates_python_function_flavor_successfully(
        pretrained_model):
    model_config_path = os.path.join(_get_model_log_dir(
        pretrained_model.model_path, pretrained_model.run_id), "MLmodel")
    model_config = Model.load(model_config_path)
    mfs._validate_deployment_flavor(
            model_config=model_config, flavor=mlflow.pyfunc.FLAVOR_NAME)


def test_get_preferred_deployment_flavor_obtains_valid_flavor_from_model(pretrained_model):
    model_config_path = os.path.join(_get_model_log_dir(
        pretrained_model.model_path, pretrained_model.run_id), "MLmodel")
    model_config = Model.load(model_config_path)

    selected_flavor = mfs._get_preferred_deployment_flavor(model_config=model_config)

    assert selected_flavor in mfs.SUPPORTED_DEPLOYMENT_FLAVORS
    assert selected_flavor in model_config.flavors


@mock_sagemaker_aws_services
def test_deploy_creates_sagemaker_resources_with_expected_names(pretrained_model, sagemaker_client):
    app_name = "test-app"
    mfs.deploy(app_name=app_name,
               model_path=pretrained_model.model_path,
               run_id=pretrained_model.run_id,
               mode=mfs.DEPLOYMENT_MODE_CREATE)

    models_response = sagemaker_client.list_models()
    found_matching_model = False
    for model in models_response["Models"]:
        if app_name in model["ModelName"]:
            found_matching_model = True
            break
    assert found_matching_model

    endpoint_configs_response = sagemaker_client.list_endpoint_configs()
    found_matching_config = False
    for config in endpoint_configs_response["EndpointConfigs"]:
        if app_name in config["EndpointConfigName"]:
            found_matching_config = True
            break
    assert found_matching_config

    endpoints_response = sagemaker_client.list_endpoints()
    assert app_name in [endpoint["EndpointName"] for endpoint in endpoints_response["Endpoints"]]


@mock_sagemaker_aws_services
def test_deploying_application_with_preexisting_name_in_create_mode_throws_exception(
        pretrained_model):
    app_name = "test-app"
    mfs.deploy(app_name=app_name,
               model_path=pretrained_model.model_path,
               run_id=pretrained_model.run_id,
               mode=mfs.DEPLOYMENT_MODE_CREATE)

    with pytest.raises(Exception) as exception_info:
        mfs.deploy(app_name=app_name,
                   model_path=pretrained_model.model_path,
                   run_id=pretrained_model.run_id,
                   mode=mfs.DEPLOYMENT_MODE_CREATE)

    assert "an application with the same name already exists" in exception_info.value.message
