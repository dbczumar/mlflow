from huey import SqliteHuey

huey = SqliteHuey('mlflow', filename='/tmp/mlflow-huey.db')
