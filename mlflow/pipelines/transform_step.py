import sys
import cloudpickle
import pandas as pd
import importlib
import yaml


def run_transform_step(
    train_data_path,
    transformer_config_path,
    transformer_output_path,
    transformed_data_output_path,
    step_config_path,
):
    """
    :param train_data_path: Path to training data
    :param transformer_config_path: Path to the transformer configuration yaml
    :param transformer_output_path: Output path of the transformer
    :param transformed_data_output_path: Output path of transformed data
    :param step_config_path: Path to the internal transformer step configuration yaml
                             (TODO: Unify `transformer_config_path` and `step_config_path`)
    """
    with open(step_config_path, "r") as f:
        step_config = yaml.safe_load(f)

    pipeline_root = step_config["pipeline_root"]
    sys.path.append(pipeline_root)
    module_name, method_name = (
        yaml.safe_load(open(transformer_config_path, "r")).get("transformer_method").rsplit(".", 1)
    )
    transformer_fn = getattr(importlib.import_module(module_name), method_name)
    transformer = transformer_fn()

    df = pd.read_parquet(train_data_path)

    # TODO: load from conf
    X = df.drop(columns=["price"])
    y = df["price"]

    features = transformer.fit_transform(X)

    with open(transformer_output_path, "wb") as f:
        cloudpickle.dump(transformer, f)

    transformed = pd.DataFrame(data={"features": list(features), "target": y})
    transformed.to_parquet(transformed_data_output_path)
