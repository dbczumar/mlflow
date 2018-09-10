import os
from subprocess import Popen, PIPE, STDOUT

import mlflow
from mlflow.tracking.utils import _get_model_log_dir
from mlflow.utils.file_utils import TempDir, _copy_file_or_tree 
from mlflow.utils.docker import get_template, build_image, push_image
from mlflow.utils.docker import push_image as push_docker_image

MODEL_SERVER_INTERNAL_PORT = 8080

DEFAULT_SERVICE_PORT = 5001

DEPLOYMENT_CONFIG_TEMPLATE = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {model_name} 
  labels:
    app: {model_name} 
spec:
  replicas: 1 
  selector:
    matchLabels:
      app: {model_name} 
  template:
    metadata:
      labels:
        app: {model_name} 
    spec:
      containers:
      - name: {model_name}
        image: {image_uri} 
        args: ["serve"]
        ports:
        - containerPort: {internal_port} 
"""

SERVICE_CONFIG_TEMPLATE = """
apiVersion: v1
kind: Service
metadata:
  name: {model_name}-server
spec:
  type: NodePort
  selector:
    app: {model_name} 
  ports:
  - name: {model_name}-server-port
    protocol: TCP
    port: {service_port} 
    targetPort: {internal_port} 
"""

DEPLOYMENT_SCRIPT_TEMPLATE = """
import os
from mlflow.kubernetes import run_model_server

def main():
    root_server_path = os.path.dirname(os.path.abspath(__file__))
    run_model_server(deployment_config_path=\"{deployment_config_path}\", 
                     service_config_path=\"{service_config_path}\",
                     root_server_path=root_server_path)

if __name__ == "__main__":
    main()
"""


def run_model_server(deployment_config_path, service_config_path, root_server_path=None):
    """
    :param root_server_path: The path to the model server directory containing the specified
                             deployment and service configurations. If `None`, the paths
                             to the deployment and service configurations will be assumed
                             to be absolute. Otherwise, these configuration paths will be
                             assumed to be relative to the specified root path.
    """
    if root_server_path is not None:
        deployment_config_path = os.path.join(root_server_path, deployment_config_path)
        service_config_path = os.path.join(root_server_path, service_config_path) 

    base_cmd = "kubectl create -f {config_path}"
    deployment_proc = Popen(base_cmd.format(config_path=deployment_config_path).split(" "))
    deployment_proc.wait()
    service_proc = Popen(base_cmd.format(config_path=service_config_path).split(" "))
    service_proc.wait()


def build_model_server(model_path, run_id=None, model_name=None, pyfunc_image_uri=None, 
                       target_registry_uri=None, push_image=False, mlflow_home=None, 
                       service_port=None, output_directory=None):
    """
    :param output_directory: The directory to which to write configuration files and scripts
                             for the model server. If `None`, the working directory
                             from which this function was invoked will be used.
    """
    with TempDir() as tmp:
        cwd = tmp.path()
        dockerfile_template = _get_image_template(image_resources_path=cwd, 
                model_path=model_path, run_id=run_id, pyfunc_uri=pyfunc_image_uri, 
                mlflow_home=mlflow_home)

        template_path = os.path.join(cwd, "Dockerfile")
        with open(template_path, "w") as f:
            f.write(dockerfile_template)
        
        if model_name is None:
            model_name = _get_model_name(model_path=model_path, run_id=run_id)
        image_name = "mlflow-model-{model_name}".format(model_name=model_name)
        image_uri = "/".join([target_registry_uri.strip("/"), image_name])
        
        # build_image(image_name=image_uri, template_path=template_path)
        if push_image:
            push_docker_image(image_uri=image_uri)

    output_directory = output_directory if output_directory is not None else os.getcwd()
    os.makedirs(output_directory)

    deployment_config_subpath = "{model_name}-deployment.yaml".format(model_name=model_name)
    service_config_subpath = "{model_name}-service.yaml".format(model_name=model_name)
    deployment_config_fullpath = os.path.join(output_directory, deployment_config_subpath)
    service_config_fullpath = os.path.join(output_directory, service_config_subpath)

    with open(deployment_config_fullpath, "w") as f:
        deployment_config = _get_deployment_config(model_name=model_name, image_uri=image_uri, 
                internal_port=MODEL_SERVER_INTERNAL_PORT)
        f.write(deployment_config)

    service_port = service_port if service_port is not None else DEFAULT_SERVICE_PORT 
    with open(service_config_fullpath, "w") as f:
        service_config = _get_service_config(model_name=model_name, service_port=service_port,
                internal_port=MODEL_SERVER_INTERNAL_PORT)
        f.write(service_config)

    with open(os.path.join(output_directory, "run_server.py"), "w") as f:
        deployment_script = _get_deployment_script(
                deployment_config_subpath=deployment_config_subpath,
                service_config_subpath=service_config_subpath)
        f.write(deployment_script)


def _get_image_template(image_resources_path, model_path, run_id=None, pyfunc_uri=None, 
        mlflow_home=None):
    if pyfunc_uri is not None:
        dockerfile_cmds = ["FROM {base_uri}".format(base_uri=pyfunc_uri)]
    else:
        dockerfile_template = get_template(
                image_resources_path=image_resources_path, mlflow_home=mlflow_home)
        dockerfile_cmds = dockerfile_template.split("\n")

    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
        model_resource_path = _copy_file_or_tree(
                src=model_path, dst=image_resources_path, dst_dir="model")
         

    dockerfile_cmds.append("COPY {host_model_path} {container_model_path}".format(
        host_model_path=model_resource_path, container_model_path="/opt/ml/model")) 
    return "\n".join(dockerfile_cmds)

    
def _get_model_name(model_path, run_id=None):
    return "{mp}-{rid}".format(mp=model_path,
                               rid=(run_id if run_id is not None else "local"))


def _get_deployment_config(model_name, image_uri, internal_port):
    return DEPLOYMENT_CONFIG_TEMPLATE.format(model_name=model_name, image_uri=image_uri, 
            internal_port=internal_port)


def _get_service_config(model_name, service_port, internal_port):
    return SERVICE_CONFIG_TEMPLATE.format(model_name=model_name, service_port=service_port, 
            internal_port=internal_port)


def _get_deployment_script(deployment_config_subpath, service_config_subpath):
    return DEPLOYMENT_SCRIPT_TEMPLATE.format(deployment_config_path=deployment_config_subpath,
                                             service_config_path=service_config_subpath)
