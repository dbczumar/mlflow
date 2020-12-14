#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_SKINNY='true'

pytest --verbose tests/test_skinny.py
python -m pip install sqlalchemy alembic sqlparse
pytest --verbose tests/tracking/test_client.py
pytest --verbose tests/tracking/test_tracking.py
pytest --verbose tests/projects/test_projects.py
pytest --verbose tests/deployments/test_cli.py
pytest --verbose tests/deployments/test_deployments.py
pytest --verbose tests/test_cli.py
pytest --verbose tests/projects/test_projects_cli.py

test $err = 0
