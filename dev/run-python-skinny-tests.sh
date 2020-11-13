#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)
export MLFLOW_SKINNY='true'

pytest --verbose tests/test_skinny.py
python -m pip install sqlalchemy
pytest --verbose tests/tracking

test $err = 0
