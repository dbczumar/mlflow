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
def build_model_server(model_path, run_id, model_name, pyfunc_image_uri, mlflow_home, 
                       target_registry_uri, push_image, image_pull_secret, service_port, 
                       output_directory):
    mlflow.kubernetes.build_model_server(
            model_path=model_path, run_id=run_id, model_name=model_name, 
            pyfunc_image_uri=pyfunc_image_uri, mlflow_home=mlflow_home, 
            target_registry_uri=target_registry_uri, push_image=push_image,
            image_pull_secret=image_pull_secret, service_port=service_port, 
            output_directory=output_directory) 


