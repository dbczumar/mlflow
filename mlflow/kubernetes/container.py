import os

import mlflow.pyfunc as pyfunc
import mlflow.sagemaker.container as sage_container
from mlflow.sagemaker.container import utils as container_utils 
from mlflow.utils.file_utils import TempDir, _copy_file_or_tree
from mlflow.models import Model

def create_conda_env_if_necessary():
    model_config_path = os.path.join(sage_container.MODEL_PATH, "MLmodel")
    model_config = Model.load(model_config_path)
    runtime_flavor = sage_container._get_runtime_flavor(model_config)

    if runtime_flavor != pyfunc.FLAVOR_NAME or pyfunc.ENV not in model_config.flavors[pyfunc.FLAVOR_NAME]:
        return

    env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]
    env_path = os.path.join(sage_container.MODEL_PATH, env_path)
    with TempDir() as tmp:
        env_path = os.path.join(tmp.path(), _copy_file_or_tree(src=env_path, dst=tmp.path(), dst_dir=None))
        container_utils.create_conda_env(env_name=sage_container.CUSTOM_ENV_NAME, env_path=env_path)


