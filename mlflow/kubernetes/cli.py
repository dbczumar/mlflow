import mlflow.kubernetes

import click

@click.group("kubernetes")
def commands():
    """Serve models on Kubernetes clusters"""
    pass

@commands.command("run-server")
@click.option("--server-path", "-s", required=True, 
              help=("The path to the model server directory output by" 
                    "`mlflow.kubernetes.build_model_server`."))
@click.option("--num-replicas", "-n", help="The number of model replicas to serve.", default=1)
def run_model_server(server_path, num_replicas):
    mlflow.kubernetes.run_model_server(server_path, num_replicas)

@commands.command("build-server")
@click.option("--model-path", "-m", required=True, 
              help=("The path to the Mlflow model for which to build a server. If `run_id` is not" 
                    " `None`, this should be an absolute path. Otherwise, it should be a" 
                    " run-relative path."))
@click.option("--run-id", "-r", help="The run id of the Mlflow model for which to build a server.")
@click.option("--model-name", "-n", default=None, 
              help=("The name of the model; this will be used for naming within the" 
                    " Kubernetes deployment and service configurations. If `None`, a name will be" 
                    " created using the specified model path and run id"))
@click.option("--pyfunc-image-uri", "-p", default=None, 
              help=("URI of an `mlflow-pyfunc` base Docker image from which the model server" 
                    " Docker image will be built. If `None`, the base image will be"
                    " built from scratch."))
@click.option("--mlflow-home", "-h", default=None,
              help=("Path to the Mlflow root directory. This will only be used if the container"
                     " base image is being built from scratch (if `pyfunc_image_uri` is `None`)."
                     " If `mlflow_home` is `None`, the base image will install Mlflow from pip"
                     " during the build. Otherwise, it will install Mlflow from the specified"
                     " directory."))
@click.option("--target-registry-uri", "-t", default=None,
              help=("The URI of the docker registry that Kubernetes will use to pull the model"
                     " server Docker image. If `None`, the default docker registry (docker.io) will"
                     " be used. Otherwise, the model server image will be tagged using the"
                     " specified registry uri."))
@click.option("--push-image", is_flag=True,
              help=("If specified, the model server Docker image will be pushed to the registry"
                    " specified by `target_registry_uri` (or docker.io if `target_registry_uri` is"
                    " `None`). If unspecified, the model server Docker image will not be pushed to"
                    " a registry."))
@click.option("--image-pull-secret", "-s", default=None,
              help=("The name of a Kubernetes secret that will be used to pull images from the "
                    " Docker registry specified by `target_registry_uri`"))
@click.option("--port", default=None,
              help=("The cluster node port on which to expose the Kubernetes service for model"
                    " serving. This value will be used for the `port` field of the Kubernetes"
                    " service spec (see mlflow.kubernetes.SERVICE_CONFIG_TEMPLATE for reference)."
                    " If `None`, the port defined by `mlflow.kubernetes.DEFAULT_SERVICE_PORT`"
                    " will be used."))
@click.option("--output-directory", "-o", default=None,
              help=("The directory to which to write configuration files for the model server."
                    " If `None`, the working directory from which this function was invoked will"
                    " be used."))
def build_model_server(model_path, run_id, model_name, pyfunc_image_uri, mlflow_home, 
                       target_registry_uri, push_image, image_pull_secret, port, output_directory):
    mlflow.kubernetes.build_model_server(
            model_path=model_path, run_id=run_id, model_name=model_name, 
            pyfunc_image_uri=pyfunc_image_uri, mlflow_home=mlflow_home, 
            target_registry_uri=target_registry_uri, push_image=push_image,
            image_pull_secret=image_pull_secret, port=port, 
            output_directory=output_directory) 
