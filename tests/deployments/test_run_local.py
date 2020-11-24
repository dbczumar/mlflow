from click.testing import CliRunner
from mlflow.deployments import cli


f_model_uri = "fake_model_uri"
f_name = "fake_deployment_name"
f_flavor = "fake_flavor"
f_target = "faketarget"


def test_run_local():
    runner = CliRunner()
    res = runner.invoke(
        cli.run_local, ["-f", f_flavor, "-m", f_model_uri, "-t", f_target, "--name", f_name]
    )
    assert "Deployed locally at the key {}".format(f_name) in res.stdout
    assert "using the model from {}.".format(f_model_uri) in res.stdout
    assert "It's flavor is {} and config is {}".format(f_flavor, str({})) in res.stdout
