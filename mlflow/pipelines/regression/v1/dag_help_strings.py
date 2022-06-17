def format_help_string(help_string):
    """
    Formats the specified ``help_string`` to obtain a Mermaid-compatible help string. For example,
    this method replaces quotation marks with their HTML representation.

    :param help_string: The raw help string.
    :return: A Mermaid-compatible help string.
    """
    return help_string.replace("\"", "&bsol;#quot;").replace("'", "&bsol;&#39;")


PIPELINE_YAML = format_help_string("""# pipeline.yaml is the main configuration file for the pipeline. It defines attributes for each step of the regression pipeline, such as the dataset to use (defined in the 'data' section corresponding to the 'ingest' step) and the metrics to compute during model training & evaluation (defined in the 'metrics' section, which is used by the 'train' and 'evaluate' steps). pipeline.yaml files also support value overrides from profiles (located in the 'profiles' subdirectory of the pipeline) using Jinja2 templating syntax. An example pipeline.yaml file is displayed below.\n\n
template: "regression/v1"
# Specifies the dataset to use for model development
data:
  location: {{INGEST_DATA_LOCATION}}
  format: {{INGEST_DATA_FORMAT|default('parquet')}}
  custom_loader_method: steps.ingest.load_file_as_dataframe
target_col: "fare_amount"
steps:
  split:
    split_ratios: {{SPLIT_RATIOS|default([0.75, 0.125, 0.125])}}
    post_split_method: steps.split.process_splits
  transform:
    transform_method: steps.transform.transformer_fn
  train:
    train_method: steps.train.estimator_fn
  evaluate:
    validation_criteria:
      - metric: root_mean_squared_error
        threshold: 10
  register:
    model_name: "taxi_fare_regressor"
    allow_non_validated_model: true
metrics:
  custom:
    - name: weighted_mean_squared_error
      function: weighted_mean_squared_error
      greater_is_better: False
  primary: "root_mean_squared_error"
""")

INGEST_USER_CODE = format_help_string("""\"\"\"\nsteps/ingest.py defines customizable logic for parsing arbitrary dataset formats (i.e. formats that are not natively parsed by MLflow Pipelines) via the `load_file_as_dataframe` function. Note that the Parquet, Delta, and Spark SQL dataset formats are natively parsed by MLflow Pipelines, and you do not need to define custom logic for parsing them. An example `load_file_as_dataframe` implementation is displayed below (note that a different function name or module can be specified via the 'custom_loader_method' attribute of the 'data' section in pipeline.yaml).\n\"\"\"\n
def load_file_as_dataframe(file_path, file_format):
    \"\"\"
    Load content from the specified dataset file as a Pandas DataFrame.

    This method is used to load dataset types that are not natively  managed by MLflow Pipelines (datasets that are not in Parquet, Delta Table, or Spark SQL Table format). This method is called once for each file in the dataset, and MLflow Pipelines automatically combines the resulting DataFrames together.

    :param file_path: The path to the dataset file.
    :param file_format: The file format string, such as "csv".
    :return: A Pandas DataFrame representing the content of the specified file.
    \"\"\"

    if file_format == "csv":
        import pandas

        return pandas.read_csv(file_path, index_col=0)
    else:
        raise NotImplementedError
""")

SPLIT_USER_CODE = format_help_string("""\"\"\"\nsteps/split.py defines customizable logic for preprocessing the training, validation, and test datasets prior to model creation via the `process_splits` function, an example of which is displayed below (note that a different function name or module can be specified via the 'post_split_method' attribute of the 'split' step definition in pipeline.yaml).\n\"\"\"\n
def process_splits(
    train_df: DataFrame, validation_df: DataFrame, test_df: DataFrame
) -> (DataFrame, DataFrame, DataFrame):
    \"\"\"
    Perform additional processing on the split datasets.

    :param train_df: The training dataset.
    :param validation_df: The validation dataset.
    :param test_df: The test dataset.
    :return: A tuple of containing, in order, the processed training dataset, the processed validation dataset, and the processed test dataset.
    \"\"\"

    return train_df.dropna(), validation_df.dropna(), test_df.dropna()
""")
TRANSFORM_USER_CODE = format_help_string("""\"\"\"\nsteps/transform.py defines customizable logic for transforming input data during model inference. Transformations are specified via the via the `transformer_fn` function, an example of which is displayed below (note that a different function name or module can be specified via the 'transform_method' attribute of the 'transform' step definition in pipeline.yaml).\n\"\"\"\n
def transformer_fn():
    \"\"\"
    Returns an *unfit* transformer with ``fit()`` and ``transform()`` methods. The transformer's input and output signatures should be compatible with scikit-learn transformers.
    \"\"\"
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers = [
        (
            "hour_encoder",
            OneHotEncoder(categories="auto", sparse=False),
            ["pickup_hour"],
        ),
        (
            "std_scaler",
            StandardScaler(),
            ["trip_distance"],
        ),
    ]
    return Pipeline(steps=[("encoder", ColumnTransformer(transformers)]))
""")

TRAIN_USER_CODE = format_help_string("""\"\"\"\nsteps/train.py defines customizable logic for specifying your estimator's architecture and parameters that will be used during training. Estimator architectures and parameters are specified via the `estimator_fn` function, an example of which is displayed below (note that a different function name or module can be specified via the 'train_method' attribute of the 'train' step definition in pipeline.yaml).\n\"\"\"\n
def estimator_fn():
    \"\"\"
    Returns an *unfit* estimator that defines ``fit()`` and ``predict()`` methods. The estimator's input and output signatures should be compatible with scikit-learn estimators.
    \"\"\"
    from sklearn.linear_model import SGDRegressor

    return SGDRegressor()
""")
