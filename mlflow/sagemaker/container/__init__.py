"""
Initialize the environment and start model serving on Sagemaker or local Docker container.

To be executed only during the model deployment.

"""
from __future__ import print_function

import multiprocessing
import os
import shutil
import signal
from subprocess import check_call, Popen
import sys
import yaml
import json 

from pkg_resources import resource_filename

import mlflow
import mlflow.version
from mlflow.sagemaker.container import utils as container_utils
from mlflow import pyfunc, mleap
from mlflow.models import Model
from mlflow.utils.logging_utils import eprint
from mlflow.utils.process import exec_cmd 


MODEL_PATH = "/opt/ml/model"
DEFAULT_ENV_NAME = "default_env"
CUSTOM_ENV_NAME = "custom_env"

DEPLOYMENT_CONFIG_KEY_FLAVOR_NAME = "deployment_flavor_name"

DEFAULT_SAGEMAKER_SERVER_PORT = 8080

SUPPORTED_FLAVORS = [
    pyfunc.FLAVOR_NAME,
    mleap.FLAVOR_NAME
]


def _init(cmd):
    """
    Initialize the container and execute command.

    :param cmd: Command param passed by Sagemaker. Can be  "serve" or "train" (unimplemented).
    """
    if cmd == 'serve':
        _serve()
    elif cmd == 'train':
        _train()
    else:
        raise Exception("Unrecognized command {cmd}, full args = {args}".format(cmd=cmd,
                                                                                args=str(sys.argv)))


def _serve():
    """
    Serve the model.

    Read the MLmodel config, initialize the Conda environment if needed and start python server.
    """
    model_config_path = os.path.join(MODEL_PATH, "MLmodel")
    m = Model.load(model_config_path)
    runtime_flavor = _get_runtime_flavor(m)
    if runtime_flavor == pyfunc.FLAVOR_NAME:
        _serve_pyfunc(m)
    elif runtime_flavor == mleap.FLAVOR_NAME:
        _serve_mleap()
    else:
        raise Exception("Unrecognized runtime flavor: {runtime_flavor}".format(
            runtime_flavor=runtime_flavor))


def _get_runtime_flavor(model):
    if DEPLOYMENT_CONFIG_KEY_FLAVOR_NAME in os.environ:
        serving_flavor = os.environ[DEPLOYMENT_CONFIG_KEY_FLAVOR_NAME]
    else:
        # Older versions of mlflow may not specify a deployment configuration
        serving_flavor = pyfunc.FLAVOR_NAME

    if serving_flavor == mleap.FLAVOR_NAME:
        # TODO(dbczumar): Host the scoring Java package on Maven Central so that we no
        # longer require the container source for this flavor.
        if container_utils.container_includes_mlflow_source():
            return mleap.FLAVOR_NAME
        else:
            raise Exception("The container does not support the specified deployment flavor:"
                            " `{mleap_flavor}`. Please build the container with the `mlflow_home`"
                            " parameter specified to enable this feature.".format(
                                mleap_flavor=mleap.FLAVOR_NAME))
    elif pyfunc.FLAVOR_NAME in model.flavors:
        return pyfunc.FLAVOR_NAME
    else:
        raise Exception("This container only supports models with the MLeap or PyFunc flavors.")



def _serve_pyfunc(model):
    conf = model.flavors[pyfunc.FLAVOR_NAME]
    if pyfunc.ENV in conf:
        env_path = conf[pyfunc.ENV]
        if not _has_conda_env(env_name=CUSTOM_ENV_NAME):
            print("creating custom environment")

        env_path_dst = os.path.join("/opt/mlflow/", env_path)
        env_path_dst_dir = os.path.dirname(env_path_dst)
        if not os.path.exists(env_path_dst_dir):
            os.makedirs(env_path_dst_dir)
        # TODO: should we test that the environment does not include any of the server dependencies?
        # Those are gonna be reinstalled. should probably test this on the client side
        shutil.copyfile(os.path.join(MODEL_PATH, env_path), env_path_dst)
        container_utils.create_conda_env(env_name=CUSTOM_ENV_NAME, env_path=env_path)
        
        runtime_env_name = CUSTOM_ENV_NAME
    else:
        runtime_env_name = DEFAULT_ENV_NAME

    print("activating custom environment")
    container_utils.activate_environment(env_name=runtime_env_name)

    nginx_conf = resource_filename(mlflow.sagemaker.__name__, "container/scoring_server/nginx.conf")
    nginx = Popen(['nginx', '-c', nginx_conf])
    # link the log streams to stdout/err so they will be logged to the container logs
    check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
    check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])
    cpu_count = multiprocessing.cpu_count()
    os.system("pip -V")
    os.system("python -V")
    os.system('python -c"from mlflow.version import VERSION as V; print(V)"')
    cmd = ("gunicorn --timeout 60 -k gevent -b unix:/tmp/gunicorn.sock -w {nworkers} " +
           "mlflow.sagemaker.container.scoring_server.wsgi:app").format(nworkers=cpu_count)
    gunicorn = Popen(cmd.split(" "))
    signal.signal(signal.SIGTERM, lambda a, b: _sigterm_handler(pids=[nginx.pid, gunicorn.pid]))
    # If either subprocess exits, so do we.
    awaited_pids = _await_subprocess_exit_any(procs=[nginx, gunicorn])
    _sigterm_handler(awaited_pids)


def _serve_mleap():
    serve_cmd = ["java", "-cp", "/opt/mlflow/mlflow/java/scoring/target/mlflow-scoring-*"
                 "-with-dependencies.jar".format(
                    mlflow_version=mlflow.version.VERSION),
                 "org.mlflow.sagemaker.ScoringServer",
                 MODEL_PATH, str(DEFAULT_SAGEMAKER_SERVER_PORT)]
    # Invoke `Popen` with a single string command in the shell to support wildcard usage
    # with the mlflow jar version.
    serve_cmd = " ".join(serve_cmd)
    mleap = Popen(serve_cmd, shell=True)
    signal.signal(signal.SIGTERM, lambda a, b: _sigterm_handler(pids=[mleap.pid]))
    awaited_pids = _await_subprocess_exit_any(procs=[mleap])
    _sigterm_handler(awaited_pids)


def _has_conda_env(env_name):
    cmd = "conda env list --json"
    _, available_envs, _ = exec_cmd(cmd=cmd.split(" "), stream_output=False)
    available_envs = json.loads(available_envs)["envs"]
    return (env_name in 
            [os.path.basename(available_env_name) for available_env_name in available_envs])


def _train():
    raise Exception("Train is not implemented.")


def _await_subprocess_exit_any(procs):
    pids = [proc.pid for proc in procs]
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break
    return pids


def _sigterm_handler(pids):
    """
    Cleanup when terminating.

    Attempt to kill all launched processes and exit.

    """
    print("Got sigterm signal, exiting.")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    sys.exit(0)
