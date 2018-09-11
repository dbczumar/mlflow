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

