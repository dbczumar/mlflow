import os
from mlflow.utils.process import exec_cmd
from mlflow.version import VERSION as MLFLOW_VERSION

def create_conda_env(env_name, env_path=None):
    if env_path is not None:
        env_creation_cmd = "conda env create -n {env_name} -f {env_path}".format(env_name=env_name, env_path=env_path)
    else:
        env_creation_cmd = "conda create -n {env_name}".format(env_name=env_name)
    exec_cmd(env_creation_cmd.split(" "), stream_output=True)

    exec_cmd("conda install -n {env_name} -c anaconda gunicorn gevent".format(
     env_name=env_name).split(" "), stream_output=True)

    activate_cmd = "source /miniconda/bin/activate {env_name}".format(env_name=env_name)
    install_cmd = ("pip install /opt/mlflow" if container_includes_mlflow_source()\
        else "pip install mlflow=={mlflow_version}".format(
            mlflow_version=MLFLOW_VERSION))

    exec_cmd(["/bin/bash", "-c", "; ".join([activate_cmd, install_cmd])], stream_output=True)


def container_includes_mlflow_source():
    return os.path.isdir("/opt/mlflow")


def activate_environment(env_name):
    conda_activation_cmd = "/bin/bash -c source /miniconda/bin/activate {env_name}".format(
            env_name=env_name)
    exec_cmd(conda_activation_cmd.split(" "), stream_output=True, shell=True)



