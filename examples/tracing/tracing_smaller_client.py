"""
This example demonstrates how to create a trace with multiple spans using the low-level MLflow client APIs.
"""

import mlflow
from mlflow.tracing.destination import MlflowExperiment

mlflow.login()

mlflow.set_tracking_uri("databricks")
mlflow.tracing.set_destination(
    MlflowExperiment(
        experiment_id="ID of your experiment"
    )
)

client = mlflow.MlflowClient()

def run(x: int, y: int) -> int:
    # Create a trace. The `start_trace` API returns a root span of the trace.
    root_span = client.start_trace(
        name="my_trace",
        inputs={"x": x, "y": y},
        # Tags are key-value pairs associated with the trace.
        # You can update the tags later using `client.set_trace_tag` API.
        tags={
            "fruit": "apple",
            "vegetable": "carrot",
        },
    )

    z = x + y

    # Request ID is a unique identifier for the trace. You will need this ID
    # to interact with the trace later using the MLflow client.
    request_id = root_span.request_id

    # Create a child span of the root span.
    child_span = client.start_span(
        name="child_span",
        # Specify the request ID to which the child span belongs.
        request_id=request_id,
        # Also specify the ID of the parent span to build the span hierarchy.
        # You can access the span ID via `span_id` property of the span object.
        parent_id=root_span.span_id,
        # Each span has its own inputs.
        inputs={"z": z},
        # Attributes are key-value pairs associated with the span.
        attributes={
            "model": "my_model",
            "temperature": 0.5,
        },
    )

    z = z**2

    # End the child span. Please make sure to end the child span before ending the root span.
    client.end_span(
        request_id=request_id,
        span_id=child_span.span_id,
        # Set the output(s) of the span.
        outputs=z,
        # Set the completion status, such as "OK" (default), "ERROR", etc.
        status="OK",
    )

    z = z + 1

    # End the root span.
    client.end_trace(
        request_id=request_id,
        # Set the output(s) of the span.
        outputs=z,
    )

    return z


assert run(1, 2) == 10

trace = mlflow.get_last_active_trace()
print("Last active trace", trace)

assert trace.info.tags["fruit"] == "apple"
assert trace.info.tags["vegetable"] == "carrot"
