import os
from mlflow.utils.process import exec_cmd
from mlflow.version import VERSION as MLFLOW_VERSION

def create_conda_env(env_name, env_path=None):
    exec_cmd("conda create -n {env_name}".format(env_name=env_name).split(" "), stream_output=True)
    if env_path is not None:
        exec_cmd("conda install -n {env_name} --file {env_path}".format(
            env_name=env_name, env_path=env_path).split(" "), stream_output=True)

    exec_cmd("conda install -n {env_name} -c anaconda gunicorn gevent".format(
     env_name=env_name).split(" "), stream_output=True)
    
    activate_environment(env_name=env_name)
    exec_cmd("pip install /opt/mlflow" if container_includes_mlflow_source()\
        else "pip install mlflow=={mlflow_version}".format(
            mlflow_version=MLFLOW_VERSION).split(" "), stream_output=True, shell=True)


def container_includes_mlflow_source():
    return os.path.isdir("/opt/mlflow")


def activate_environment(env_name):
    conda_activation_cmd = "/bin/bash -c source /miniconda/bin/activate {env_name}".format(
            env_name=env_name)
    exec_cmd(conda_activation_cmd.split(" "), stream_output=True, shell=True)



