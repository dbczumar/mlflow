import asyncio
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

import mlflow
from mlflow.entities import (
    SpanEvent,
    SpanStatusCode,
    SpanType,
    Trace,
    TraceData,
    TraceInfo,
)
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_TRACKING_USERNAME
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import (
    TRACE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION_KEY,
    SpanAttributeKey,
    TraceMetadataKey,
    TraceTagKey,
)
from mlflow.tracing.export.inference_table import pop_trace
from mlflow.tracing.fluent import TRACE_BUFFER
from mlflow.tracing.provider import _get_trace_exporter, _get_tracer
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.os import is_windows


class DefaultTestModel:
    @mlflow.trace()
    def predict(self, x, y):
        z = x + y
        z = self.add_one(z)
        z = mlflow.trace(self.square)(z)
        return z  # noqa: RET504

    @mlflow.trace(span_type=SpanType.LLM, name="add_one_with_custom_name", attributes={"delta": 1})
    def add_one(self, z):
        return z + 1

    def square(self, t):
        res = t**2
        time.sleep(0.1)
        return res


class DefaultAsyncTestModel:
    @mlflow.trace()
    async def predict(self, x, y):
        z = x + y
        z = await self.add_one(z)
        z = await mlflow.trace(self.square)(z)
        return z  # noqa: RET504

    @mlflow.trace(span_type=SpanType.LLM, name="add_one_with_custom_name", attributes={"delta": 1})
    async def add_one(self, z):
        return z + 1

    async def square(self, t):
        res = t**2
        time.sleep(0.1)
        return res


class StreamTestModel:
    @mlflow.trace(output_reducer=lambda x: sum(x))
    def predict_stream(self, x, y):
        z = x + y
        for i in range(z):
            yield i

        # Generator with a normal func
        for i in range(z):
            yield self.square(i)

        # Nested generator
        yield from self.generate_numbers(z)

    @mlflow.trace
    def square(self, t):
        time.sleep(0.1)
        return t**2

    # No output_reducer -> record the list of outputs
    @mlflow.trace
    def generate_numbers(self, z):
        for i in range(z):
            yield i


class AsyncStreamTestModel:
    @mlflow.trace(output_reducer=lambda x: sum(x))
    async def predict_stream(self, x, y):
        z = x + y
        for i in range(z):
            yield i

        # Generator with a normal func
        for i in range(z):
            yield await self.square(i)

        # Nested generator
        async for number in self.generate_numbers(z):
            yield number

    @mlflow.trace
    async def square(self, t):
        await asyncio.sleep(0.1)
        return t**2

    @mlflow.trace
    async def generate_numbers(self, z):
        for i in range(z):
            yield i


class ErroringTestModel:
    @mlflow.trace()
    def predict(self, x, y):
        return self.some_operation_raise_error(x, y)

    @mlflow.trace()
    def some_operation_raise_error(self, x, y):
        raise ValueError("Some error")


class ErroringAsyncTestModel:
    @mlflow.trace()
    async def predict(self, x, y):
        return await self.some_operation_raise_error(x, y)

    @mlflow.trace()
    async def some_operation_raise_error(self, x, y):
        raise ValueError("Some error")


class ErroringStreamTestModel:
    @mlflow.trace
    def predict_stream(self, x):
        for i in range(x):
            yield self.some_operation_raise_error(i)

    @mlflow.trace
    def some_operation_raise_error(self, i):
        if i >= 1:
            raise ValueError("Some error")
        return i


@pytest.fixture
def mock_client():
    client = mock.MagicMock()
    with mock.patch("mlflow.tracing.fluent.MlflowClient", return_value=client):
        yield client


@pytest.mark.parametrize("with_active_run", [True, False])
@pytest.mark.parametrize("wrap_sync_func", [True, False])
def test_trace(wrap_sync_func, with_active_run, async_logging_enabled):
    model = DefaultTestModel() if wrap_sync_func else DefaultAsyncTestModel()

    if with_active_run:
        with mlflow.start_run() as run:
            model.predict(2, 5) if wrap_sync_func else asyncio.run(model.predict(2, 5))
            run_id = run.info.run_id
    else:
        model.predict(2, 5) if wrap_sync_func else asyncio.run(model.predict(2, 5))

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.request_id is not None
    assert trace.info.experiment_id == "0"  # default experiment
    assert trace.info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace.info.status == SpanStatusCode.OK
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 2, "y": 5}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == "64"
    if with_active_run:
        assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_id

    assert trace.data.request == '{"x": 2, "y": 5}'
    assert trace.data.response == "64"
    assert len(trace.data.spans) == 3

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["predict"]
    # TODO: Trace info timestamp is not accurate because it is not adjusted to exclude the latency
    # assert root_span.start_time_ns // 1e6 == trace.info.timestamp_ms
    assert root_span.parent_id is None
    assert root_span.attributes == {
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanFunctionName": "predict",
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"x": 2, "y": 5},
        "mlflow.spanOutputs": 64,
    }

    child_span_1 = span_name_to_span["add_one_with_custom_name"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "delta": 1,
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanFunctionName": "add_one",
        "mlflow.spanType": "LLM",
        "mlflow.spanInputs": {"z": 7},
        "mlflow.spanOutputs": 8,
    }

    child_span_2 = span_name_to_span["square"]
    assert child_span_2.parent_id == root_span.span_id
    assert child_span_2.start_time_ns <= child_span_2.end_time_ns - 0.1 * 1e6
    assert child_span_2.attributes == {
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanFunctionName": "square",
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"t": 8},
        "mlflow.spanOutputs": 64,
    }


@pytest.mark.parametrize("wrap_sync_func", [True, False])
def test_trace_stream(wrap_sync_func):
    model = StreamTestModel() if wrap_sync_func else AsyncStreamTestModel()

    stream = model.predict_stream(1, 2)

    # Trace should not be logged until the generator is consumed
    assert get_traces() == []
    # The span should not be set to active
    # because the generator is not yet consumed
    assert mlflow.get_current_active_span() is None

    chunks = []
    if wrap_sync_func:
        for chunk in stream:
            chunks.append(chunk)
            # The `predict` span should not be active here.
            assert mlflow.get_current_active_span() is None
    else:

        async def consume_stream():
            async for chunk in stream:
                chunks.append(chunk)
                assert mlflow.get_current_active_span() is None

        asyncio.run(consume_stream())

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.request_id is not None
    assert trace.info.experiment_id == "0"  # default experiment
    assert trace.info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace.info.status == SpanStatusCode.OK
    metadata = trace.info.request_metadata
    assert metadata[TraceMetadataKey.INPUTS] == '{"x": 1, "y": 2}'
    assert metadata[TraceMetadataKey.OUTPUTS] == "11"  # sum of the outputs

    assert len(trace.data.spans) == 5  # 1 root span + 3 square + 1 generate_numbers

    root_span = trace.data.spans[0]
    assert root_span.name == "predict_stream"
    assert root_span.inputs == {"x": 1, "y": 2}
    assert root_span.outputs == 11
    assert len(root_span.events) == 9
    assert root_span.events[0].name == "mlflow.chunk.item.0"
    assert root_span.events[0].attributes == {"mlflow.chunk.value": "0"}
    assert root_span.events[8].name == "mlflow.chunk.item.8"

    # Spans for the chid 'square' function
    for i in range(3):
        assert trace.data.spans[i + 1].name == f"square_{i + 1}"
        assert trace.data.spans[i + 1].inputs == {"t": i}
        assert trace.data.spans[i + 1].outputs == i**2
        assert trace.data.spans[i + 1].parent_id == root_span.span_id

    # Span for the 'generate_numbers' function
    assert trace.data.spans[4].name == "generate_numbers"
    assert trace.data.spans[4].inputs == {"z": 3}
    assert trace.data.spans[4].outputs == [0, 1, 2]  # list of outputs
    assert len(trace.data.spans[4].events) == 3


def test_trace_with_databricks_tracking_uri(
    databricks_tracking_uri, async_logging_enabled, mock_store, monkeypatch
):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")

    mock_experiment = mock.MagicMock()
    mock_experiment.experiment_id = "test_experiment_id"
    monkeypatch.setattr(
        mock_store, "get_experiment_by_name", mock.MagicMock(return_value=mock_experiment)
    )

    model = DefaultTestModel()

    with mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient._upload_trace_data"
    ) as mock_upload_trace_data:
        model.predict(2, 5)
        if async_logging_enabled:
            mlflow.flush_trace_async_logging(terminate=True)

    trace = mlflow.get_last_active_trace()
    trace_info = trace.info
    assert trace_info.request_id == "tr-0"
    assert trace_info.experiment_id == "test_experiment_id"
    assert trace_info.status == TraceStatus.OK
    assert trace_info.request_metadata == {
        TraceMetadataKey.INPUTS: '{"x": 2, "y": 5}',
        TraceMetadataKey.OUTPUTS: "64",
        TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION),
    }
    assert trace_info.tags == {
        "mlflow.traceName": "predict",
        "mlflow.artifactLocation": "test",
        "mlflow.source.name": "test",
        "mlflow.source.type": "LOCAL",
        "mlflow.user": "bob",
    }

    trace_data = trace.data
    assert trace_data.request == '{"x": 2, "y": 5}'
    assert trace_data.response == "64"
    assert len(trace_data.spans) == 3

    mock_store.start_trace.assert_called_once()
    mock_store.end_trace.assert_called_once()
    mock_upload_trace_data.assert_called_once()


# NB: async logging should be no-op for model serving,
# but we test it here to make sure it doesn't break
def test_trace_in_databricks_model_serving(
    mock_databricks_serving_with_tracing_env, async_logging_enabled
):
    # Dummy flask app for prediction
    import flask

    from mlflow.pyfunc.context import Context, set_prediction_context

    app = flask.Flask(__name__)

    @app.route("/invocations", methods=["POST"])
    def predict():
        data = json.loads(flask.request.data.decode("utf-8"))
        request_id = flask.request.headers.get("X-Request-ID")

        with set_prediction_context(Context(request_id=request_id)):
            prediction = TestModel().predict(**data)

        trace = pop_trace(request_id=request_id)

        result = json.dumps(
            {
                "prediction": prediction,
                "trace": trace,
            },
            default=str,
        )
        return flask.Response(response=result, status=200, mimetype="application/json")

    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            z = x + y
            z = self.add_one(z)
            with mlflow.start_span(name="square") as span:
                z = self.square(z)
                span.add_event(SpanEvent("event", 0, attributes={"foo": "bar"}))
            return z

        @mlflow.trace(span_type=SpanType.LLM, name="custom", attributes={"delta": 1})
        def add_one(self, z):
            return z + 1

        def square(self, t):
            return t**2

    # Mimic scoring request
    databricks_request_id = "request-12345"
    response = app.test_client().post(
        "/invocations",
        headers={"X-Request-ID": databricks_request_id},
        data=json.dumps({"x": 2, "y": 5}),
    )

    assert response.status_code == 200
    assert response.json["prediction"] == 64

    trace_dict = response.json["trace"]
    trace = Trace.from_dict(trace_dict)
    assert trace.info.request_id == databricks_request_id
    assert trace.info.request_metadata[TRACE_SCHEMA_VERSION_KEY] == "2"
    assert len(trace.data.spans) == 3

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["predict"]
    assert isinstance(root_span._trace_id, str)
    assert isinstance(root_span.span_id, str)
    assert isinstance(root_span.start_time_ns, int)
    assert isinstance(root_span.end_time_ns, int)
    assert root_span.status.status_code.value == "OK"
    assert root_span.status.description == ""
    assert root_span.attributes == {
        "mlflow.traceRequestId": databricks_request_id,
        "mlflow.spanType": SpanType.UNKNOWN,
        "mlflow.spanFunctionName": "predict",
        "mlflow.spanInputs": {"x": 2, "y": 5},
        "mlflow.spanOutputs": 64,
    }
    assert root_span.events == []

    child_span_1 = span_name_to_span["custom"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "delta": 1,
        "mlflow.traceRequestId": databricks_request_id,
        "mlflow.spanType": SpanType.LLM,
        "mlflow.spanFunctionName": "add_one",
        "mlflow.spanInputs": {"z": 7},
        "mlflow.spanOutputs": 8,
    }
    assert child_span_1.events == []

    child_span_2 = span_name_to_span["square"]
    assert child_span_2.parent_id == root_span.span_id
    assert child_span_2.attributes == {
        "mlflow.traceRequestId": databricks_request_id,
        "mlflow.spanType": SpanType.UNKNOWN,
    }
    assert asdict(child_span_2.events[0]) == {
        "name": "event",
        "timestamp": 0,
        "attributes": {"foo": "bar"},
    }

    # The trace should be removed from the buffer after being retrieved
    assert pop_trace(request_id=databricks_request_id) is None

    # In model serving, the traces should not be stored in the fluent API buffer
    traces = get_traces()
    assert len(traces) == 0


def test_trace_in_model_evaluation(mock_store, monkeypatch, async_logging_enabled):
    from mlflow.pyfunc.context import Context, set_prediction_context

    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")

    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            return x + y

    model = TestModel()

    # mock _upload_trace_data to avoid generating trace data file
    with (
        mock.patch(
            "mlflow.tracking._tracking_service.client.TrackingServiceClient._upload_trace_data"
        ),
        mlflow.start_run() as run,
    ):
        run_id = run.info.run_id
        request_id_1 = "tr-eval-123"
        with set_prediction_context(Context(request_id=request_id_1, is_evaluate=True)):
            model.predict(1, 2)

        request_id_2 = "tr-eval-456"
        with set_prediction_context(Context(request_id=request_id_2, is_evaluate=True)):
            model.predict(3, 4)

    expected_tags = {
        "mlflow.traceName": "predict",
        "mlflow.source.name": "test",
        "mlflow.source.type": "LOCAL",
        "mlflow.user": "bob",
        "mlflow.artifactLocation": "test",
    }

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    trace = mlflow.get_trace(request_id_1)
    assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_id
    assert trace.info.request_metadata[TRACE_SCHEMA_VERSION_KEY] == str(TRACE_SCHEMA_VERSION)
    assert trace.info.tags == {**expected_tags, **{TraceTagKey.EVAL_REQUEST_ID: request_id_1}}

    trace = mlflow.get_trace(request_id_2)
    assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_id
    assert trace.info.tags == {**expected_tags, **{TraceTagKey.EVAL_REQUEST_ID: request_id_2}}

    assert mock_store.start_trace.call_count == 2
    assert mock_store.end_trace.call_count == 2


@pytest.mark.parametrize("sync", [True, False])
def test_trace_handle_exception_during_prediction(sync):
    # This test is to make sure that the exception raised by the main prediction
    # logic is raised properly and the trace is still logged.
    model = ErroringTestModel() if sync else ErroringAsyncTestModel()

    with pytest.raises(ValueError, match=r"Some error"):
        model.predict(2, 5) if sync else asyncio.run(model.predict(2, 5))

    # Trace should be logged even if the function fails, with status code ERROR
    trace = mlflow.get_last_active_trace()
    assert trace.info.request_id is not None
    assert trace.info.status == TraceStatus.ERROR
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 2, "y": 5}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == ""

    assert trace.data.request == '{"x": 2, "y": 5}'
    assert trace.data.response is None
    assert len(trace.data.spans) == 2


def test_trace_handle_exception_during_streaming():
    model = ErroringStreamTestModel()

    stream = model.predict_stream(2)

    chunks = []
    with pytest.raises(ValueError, match=r"Some error"):  # noqa: PT012
        for chunk in stream:
            chunks.append(chunk)

    # The test model raises an error after the first chunk
    assert len(chunks) == 1

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == TraceStatus.ERROR
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 2}'

    # The test model is expected to produce three spans
    # 1. Root span (error - inherited from the child)
    # 2. First chunk span (OK)
    # 3. Second chunk span (error)
    spans = trace.data.spans
    assert len(spans) == 3
    assert spans[0].name == "predict_stream"
    assert spans[0].status.status_code == SpanStatusCode.ERROR
    assert spans[1].name == "some_operation_raise_error_1"
    assert spans[1].status.status_code == SpanStatusCode.OK
    assert spans[2].name == "some_operation_raise_error_2"
    assert spans[2].status.status_code == SpanStatusCode.ERROR

    # One chunk event + one exception event
    assert len(spans[0].events) == 2
    assert spans[0].events[0].name == "mlflow.chunk.item.0"
    assert spans[0].events[1].name == "exception"


@pytest.mark.parametrize(
    "model",
    [
        DefaultTestModel(),
        DefaultAsyncTestModel(),
        StreamTestModel(),
        AsyncStreamTestModel(),
    ],
)
def test_trace_ignore_exception(monkeypatch, model):
    # This test is to make sure that the main prediction logic is not affected
    # by the exception raised by the tracing logic.
    def _call_model_and_assert_output(model):
        if isinstance(model, DefaultTestModel):
            output = model.predict(2, 5)
            assert output == 64
        elif isinstance(model, DefaultAsyncTestModel):
            output = asyncio.run(model.predict(2, 5))
            assert output == 64
        elif isinstance(model, StreamTestModel):
            stream = model.predict_stream(2, 5)
            assert len(list(stream)) == 21
        elif isinstance(model, AsyncStreamTestModel):
            astream = model.predict_stream(2, 5)

            async def _consume_stream():
                return [chunk async for chunk in astream]

            stream = asyncio.run(_consume_stream())
            assert len(list(stream)) == 21
        else:
            raise ValueError("Unknown model type")

    # Exception during starting span: trace should not be logged.
    with mock.patch("mlflow.tracing.provider._get_tracer", side_effect=ValueError("Some error")):
        _call_model_and_assert_output(model)

    assert get_traces() == []

    # Exception during ending span: trace should not be logged.
    tracer = _get_tracer(__name__)

    def _always_fail(*args, **kwargs):
        raise ValueError("Some error")

    monkeypatch.setattr(tracer.span_processor, "on_end", _always_fail)
    _call_model_and_assert_output(model)
    assert len(get_traces()) == 0
    monkeypatch.undo()

    # Exception during inspecting inputs: trace should be logged without inputs field
    with mock.patch("inspect.signature", side_effect=ValueError("Some error")) as mock_input_args:
        _call_model_and_assert_output(model)
        assert mock_input_args.call_count > 0

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == TraceStatus.OK


def test_trace_skip_resolving_unrelated_tags_to_traces():
    with mock.patch("mlflow.tracking.context.registry.DatabricksRepoRunContext") as mock_context:
        mock_context.in_context.return_value = ["unrelated tags"]

        model = DefaultTestModel()
        model.predict(2, 5)

    trace = mlflow.get_last_active_trace()
    assert "unrelated tags" not in trace.info.tags


def test_start_span_context_manager(async_logging_enabled):
    datetime_now = datetime.now()

    class TestModel:
        def predict(self, x, y):
            with mlflow.start_span(name="root_span") as root_span:
                root_span.set_inputs({"x": x, "y": y})
                z = x + y

                with mlflow.start_span(name="child_span", span_type=SpanType.LLM) as child_span:
                    child_span.set_inputs(z)
                    z = z + 2
                    child_span.set_outputs(z)
                    child_span.set_attributes({"delta": 2, "time": datetime_now})

                res = self.square(z)
                root_span.set_outputs(res)
            return res

        def square(self, t):
            with mlflow.start_span(name="child_span") as span:
                span.set_inputs({"t": t})
                res = t**2
                time.sleep(0.1)
                span.set_outputs(res)
                return res

    model = TestModel()
    model.predict(1, 2)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.request_id is not None
    assert trace.info.experiment_id == "0"  # default experiment
    assert trace.info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace.info.status == TraceStatus.OK
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 1, "y": 2}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == "25"

    assert trace.data.request == '{"x": 1, "y": 2}'
    assert trace.data.response == "25"
    assert len(trace.data.spans) == 3

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["root_span"]
    assert root_span.parent_id is None
    assert root_span.attributes == {
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"x": 1, "y": 2},
        "mlflow.spanOutputs": 25,
    }

    # Span with duplicate name should be renamed to have an index number like "_1", "_2", ...
    child_span_1 = span_name_to_span["child_span_1"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "delta": 2,
        "time": str(datetime_now),
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanType": "LLM",
        "mlflow.spanInputs": 3,
        "mlflow.spanOutputs": 5,
    }

    child_span_2 = span_name_to_span["child_span_2"]
    assert child_span_2.parent_id == root_span.span_id
    assert child_span_2.attributes == {
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"t": 5},
        "mlflow.spanOutputs": 25,
    }
    assert child_span_2.start_time_ns <= child_span_2.end_time_ns - 0.1 * 1e6


def test_start_span_context_manager_with_imperative_apis(async_logging_enabled):
    # This test is to make sure that the spans created with fluent APIs and imperative APIs
    # (via MLflow client) are correctly linked together. This usage is not recommended but
    # should be supported for the advanced use cases like using LangChain callbacks as a
    # part of broader tracing.
    class TestModel:
        def __init__(self):
            self._mlflow_client = mlflow.tracking.MlflowClient()

        def predict(self, x, y):
            with mlflow.start_span(name="root_span") as root_span:
                root_span.set_inputs({"x": x, "y": y})
                z = x + y

                child_span = self._mlflow_client.start_span(
                    name="child_span_1",
                    span_type=SpanType.LLM,
                    request_id=root_span.request_id,
                    parent_id=root_span.span_id,
                )
                child_span.set_inputs(z)

                z = z + 2
                time.sleep(0.1)

                child_span.set_outputs(z)
                child_span.set_attributes({"delta": 2})
                child_span.end()

                root_span.set_outputs(z)
            return z

    model = TestModel()
    model.predict(1, 2)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.request_id is not None
    assert trace.info.experiment_id == "0"  # default experiment
    assert trace.info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace.info.status == TraceStatus.OK
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 1, "y": 2}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == "5"

    assert trace.data.request == '{"x": 1, "y": 2}'
    assert trace.data.response == "5"
    assert len(trace.data.spans) == 2

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["root_span"]
    assert root_span.parent_id is None
    assert root_span.attributes == {
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"x": 1, "y": 2},
        "mlflow.spanOutputs": 5,
    }

    child_span_1 = span_name_to_span["child_span_1"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "delta": 2,
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanType": "LLM",
        "mlflow.spanInputs": 3,
        "mlflow.spanOutputs": 5,
    }


def test_mlflow_trace_isolated_from_other_otel_processors():
    # Set up non-MLFlow tracer
    import opentelemetry.sdk.trace as trace_sdk
    from opentelemetry import trace

    class MockOtelExporter(trace_sdk.export.SpanExporter):
        def __init__(self):
            self.exported_spans = []

        def export(self, spans):
            self.exported_spans.extend(spans)

    other_exporter = MockOtelExporter()
    provider = trace_sdk.TracerProvider()
    processor = trace_sdk.export.SimpleSpanProcessor(other_exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    # Create MLflow trace
    with mlflow.start_span(name="mlflow_span"):
        pass

    # Create non-MLflow trace
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("non_mlflow_span"):
        pass

    # MLflow only processes spans created with MLflow APIs
    assert len(TRACE_BUFFER) == 1
    assert mlflow.get_last_active_trace().data.spans[0].name == "mlflow_span"

    # Other spans are processed by the other processor
    assert len(other_exporter.exported_spans) == 1
    assert other_exporter.exported_spans[0].name == "non_mlflow_span"


@mock.patch("mlflow.tracing.export.mlflow.get_display_handler")
def test_get_trace(mock_get_display_handler):
    model = DefaultTestModel()
    model.predict(2, 5)

    trace = mlflow.get_last_active_trace()
    request_id = trace.info.request_id
    mock_get_display_handler.reset_mock()

    # Fetch trace from in-memory buffer
    trace_in_memory = mlflow.get_trace(request_id)
    assert trace.info.request_id == trace_in_memory.info.request_id
    mock_get_display_handler.assert_not_called()

    # Fetch trace from backend
    TRACE_BUFFER.clear()
    trace_from_backend = mlflow.get_trace(request_id)
    assert trace.info.request_id == trace_from_backend.info.request_id
    mock_get_display_handler.assert_not_called()

    # If not found, return None with warning
    with mock.patch("mlflow.tracing.fluent._logger") as mock_logger:
        assert mlflow.get_trace("not_found") is None
        mock_logger.warning.assert_called_once()


@pytest.mark.parametrize(
    "extract_fields",
    [
        ["span.llm.inputs"],
        ["span.llm.inputs.x"],
        ["span.llm.outputs"],
    ],
)
def test_search_traces_invalid_extract_fields(extract_fields):
    with pytest.raises(MlflowException, match="Invalid field type"):
        mlflow.search_traces(extract_fields=extract_fields)


def test_get_last_active_trace():
    assert mlflow.get_last_active_trace() is None

    @mlflow.trace()
    def predict(x, y):
        return x + y

    predict(1, 2)
    predict(2, 5)
    predict(3, 6)

    trace = mlflow.get_last_active_trace()
    assert trace.info.request_id is not None
    assert trace.data.request == '{"x": 3, "y": 6}'

    # Mutation of the copy should not affect the original trace logged in the backend
    trace.info.status = TraceStatus.ERROR
    original_trace = mlflow.MlflowClient().get_trace(trace.info.request_id)
    assert original_trace.info.status == TraceStatus.OK


def test_update_current_trace():
    @mlflow.trace
    def f(x):
        mlflow.update_current_trace(tags={"fruit": "apple", "animal": "dog"})
        return g(x) + 1

    @mlflow.trace
    def g(y):
        with mlflow.start_span():
            mlflow.update_current_trace(tags={"fruit": "orange", "vegetable": "carrot"})
            return y * 2

    f(1)

    expected_tags = {
        "animal": "dog",
        "fruit": "orange",
        "vegetable": "carrot",
    }

    # Validate in-memory trace
    trace = mlflow.get_last_active_trace()
    assert trace.info.status == "OK"
    tags = {k: v for k, v in trace.info.tags.items() if not k.startswith("mlflow.")}
    assert tags == expected_tags

    # Validate backend trace
    traces = mlflow.MlflowClient().search_traces(experiment_ids=[DEFAULT_EXPERIMENT_ID])
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    tags = {k: v for k, v in traces[0].info.tags.items() if not k.startswith("mlflow.")}
    assert tags == expected_tags


def test_non_ascii_characters_not_encoded_as_unicode():
    with mlflow.start_span() as span:
        span.set_inputs({"japanese": "あ", "emoji": "👍"})

    trace = mlflow.MlflowClient().get_trace(span.request_id)
    span = trace.data.spans[0]
    assert span.inputs == {"japanese": "あ", "emoji": "👍"}

    artifact_location = local_file_uri_to_path(trace.info.tags["mlflow.artifactLocation"])
    data = Path(artifact_location, "traces.json").read_text()
    assert "あ" in data
    assert "👍" in data
    assert json.dumps("あ").strip('"') not in data
    assert json.dumps("👍").strip('"') not in data


@pytest.mark.skipif(is_windows(), reason="Otel collector docker image does not support Windows")
def test_export_to_otel_collector(otel_collector, mock_client, monkeypatch):
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:4317/v1/traces")

    # Create a trace
    model = DefaultTestModel()
    model.predict(2, 5)
    time.sleep(10)

    # Tracer should be configured to export to OTLP
    exporter = _get_trace_exporter()
    assert isinstance(exporter, OTLPSpanExporter)
    assert exporter._endpoint == "127.0.0.1:4317"

    # Traces should not be logged to MLflow
    mock_client._start_stacked_trace.assert_not_called()
    mock_client._upload_trace_data.assert_not_called()
    mock_client._upload_ended_trace_info.assert_not_called()

    # Analyze the logs of the collector
    _, output_file = otel_collector
    with open(output_file) as f:
        collector_logs = f.read()

    # 3 spans should be exported
    assert "Span #0" in collector_logs
    assert "Span #1" in collector_logs
    assert "Span #2" in collector_logs
    assert "Span #3" not in collector_logs


_SAMPLE_REMOTE_TRACE = {
    "info": {
        "request_id": "2e72d64369624e6888324462b62dc120",
        "experiment_id": "0",
        "timestamp_ms": 1726145090860,
        "execution_time_ms": 162,
        "status": "OK",
        "request_metadata": {
            "mlflow.trace_schema.version": "2",
            "mlflow.traceInputs": '{"x": 1}',
            "mlflow.traceOutputs": '{"prediction": 1}',
        },
        "tags": {
            "fruit": "apple",
            "food": "pizza",
        },
    },
    "data": {
        "spans": [
            {
                "name": "remote",
                "context": {
                    "span_id": "0x337af925d6629c01",
                    "trace_id": "0x05e82d1fc4486f3986fae6dd7b5352b1",
                },
                "parent_id": None,
                "start_time": 1726145091022155863,
                "end_time": 1726145091022572053,
                "status_code": "OK",
                "status_message": "",
                "attributes": {
                    "mlflow.traceRequestId": '"2e72d64369624e6888324462b62dc120"',
                    "mlflow.spanType": '"UNKNOWN"',
                    "mlflow.spanInputs": '{"x": 1}',
                    "mlflow.spanOutputs": '{"prediction": 1}',
                },
                "events": [
                    {"name": "event", "timestamp": 1726145091022287, "attributes": {"foo": "bar"}}
                ],
            },
            {
                "name": "remote-child",
                "context": {
                    "span_id": "0xa3dde9f2ebac1936",
                    "trace_id": "0x05e82d1fc4486f3986fae6dd7b5352b1",
                },
                "parent_id": "0x337af925d6629c01",
                "start_time": 1726145091022419340,
                "end_time": 1726145091022497944,
                "status_code": "OK",
                "status_message": "",
                "attributes": {
                    "mlflow.traceRequestId": '"2e72d64369624e6888324462b62dc120"',
                    "mlflow.spanType": '"UNKNOWN"',
                },
                "events": [],
            },
        ],
        "request": '{"x": 1}',
        "response": '{"prediction": 1}',
    },
}


def test_add_trace():
    # Mimic a remote service call that returns a trace as a part of the response
    def dummy_remote_call():
        return {"prediction": 1, "trace": _SAMPLE_REMOTE_TRACE}

    @mlflow.trace
    def predict(add_trace: bool):
        resp = dummy_remote_call()

        if add_trace:
            mlflow.add_trace(resp["trace"])
        return resp["prediction"]

    # If we don't call add_trace, the trace from the remote service should be discarded
    predict(add_trace=False)
    trace = mlflow.get_last_active_trace()
    assert len(trace.data.spans) == 1

    # If we call add_trace, the trace from the remote service should be merged
    predict(add_trace=True)
    trace = mlflow.get_last_active_trace()
    request_id = trace.info.request_id
    assert request_id is not None
    assert trace.data.request == '{"add_trace": true}'
    assert trace.data.response == "1"
    # Remote spans should be merged
    assert len(trace.data.spans) == 3
    assert all(span.request_id == request_id for span in trace.data.spans)
    parent_span, child_span, grandchild_span = trace.data.spans
    assert child_span.parent_id == parent_span.span_id
    assert child_span._trace_id == parent_span._trace_id
    assert grandchild_span.parent_id == child_span.span_id
    assert grandchild_span._trace_id == parent_span._trace_id
    # Check if span information is correctly copied
    rs = Trace.from_dict(_SAMPLE_REMOTE_TRACE).data.spans[0]
    assert child_span.name == rs.name
    assert child_span.start_time_ns == rs.start_time_ns
    assert child_span.end_time_ns == rs.end_time_ns
    assert child_span.status == rs.status
    assert child_span.span_type == rs.span_type
    assert child_span.events == rs.events
    # exclude request ID attribute from comparison
    for k in rs.attributes.keys() - {SpanAttributeKey.REQUEST_ID}:
        assert child_span.attributes[k] == rs.attributes[k]


def test_add_trace_no_current_active_trace():
    # Use the remote trace without any active trace
    remote_trace = Trace.from_dict(_SAMPLE_REMOTE_TRACE)

    mlflow.add_trace(remote_trace)

    trace = mlflow.get_last_active_trace()
    assert len(trace.data.spans) == 3
    parent_span, child_span, grandchild_span = trace.data.spans
    assert parent_span.name == "Remote Trace <remote>"
    rs = remote_trace.data.spans[0]
    assert parent_span.start_time_ns == rs.start_time_ns
    assert parent_span.end_time_ns == rs.end_time_ns
    assert child_span.name == rs.name
    assert child_span.parent_id is parent_span.span_id
    assert child_span.start_time_ns == rs.start_time_ns
    assert child_span.end_time_ns == rs.end_time_ns
    assert child_span.status == rs.status
    assert child_span.span_type == rs.span_type
    assert child_span.events == rs.events
    assert grandchild_span.parent_id == child_span.span_id
    # exclude request ID attribute from comparison
    for k in rs.attributes.keys() - {SpanAttributeKey.REQUEST_ID}:
        assert child_span.attributes[k] == rs.attributes[k]


def test_add_trace_specific_target_span():
    client = mlflow.MlflowClient()
    span = client.start_trace(name="parent")
    mlflow.add_trace(_SAMPLE_REMOTE_TRACE, target=span)
    client.end_trace(span.request_id)

    trace = mlflow.get_last_active_trace()
    assert len(trace.data.spans) == 3
    parent_span, child_span, grandchild_span = trace.data.spans
    assert parent_span.span_id == span.span_id
    rs = Trace.from_dict(_SAMPLE_REMOTE_TRACE).data.spans[0]
    assert child_span.name == rs.name
    assert child_span.parent_id is parent_span.span_id
    assert grandchild_span.parent_id == child_span.span_id


def test_add_trace_merge_tags():
    client = mlflow.MlflowClient()

    # Start the parent trace and merge the above trace as a child
    with mlflow.start_span(name="parent") as span:
        client.set_trace_tag(span.request_id, "vegetable", "carrot")
        client.set_trace_tag(span.request_id, "food", "sushi")

        mlflow.add_trace(Trace.from_dict(_SAMPLE_REMOTE_TRACE))

    trace = mlflow.get_last_active_trace()
    custom_tags = {k: v for k, v in trace.info.tags.items() if not k.startswith("mlflow.")}
    assert custom_tags == {
        "fruit": "apple",
        "vegetable": "carrot",
        # Tag value from the parent trace should prevail
        "food": "sushi",
    }


def test_add_trace_raise_for_invalid_trace():
    with pytest.raises(MlflowException, match="Invalid trace object"):
        mlflow.add_trace(None)

    with pytest.raises(MlflowException, match="Failed to load a trace object"):
        mlflow.add_trace({"info": {}, "data": {}})

    in_progress_trace = Trace(
        info=TraceInfo(
            request_id="123",
            status=TraceStatus.IN_PROGRESS,
            experiment_id="0",
            timestamp_ms=0,
            execution_time_ms=0,
        ),
        data=TraceData(),
    )
    with pytest.raises(MlflowException, match="The trace must be ended"):
        mlflow.add_trace(in_progress_trace)

    trace = Trace.from_dict(_SAMPLE_REMOTE_TRACE)
    spans = trace.data.spans
    unordered_trace = Trace(info=trace.info, data=TraceData(spans=[spans[1], spans[0]]))
    with pytest.raises(MlflowException, match="Span with ID "):
        mlflow.add_trace(unordered_trace)


def test_add_trace_in_databricks_model_serving(mock_databricks_serving_with_tracing_env):
    from mlflow.pyfunc.context import Context, set_prediction_context

    # Mimic a remote service call that returns a trace as a part of the response
    def dummy_remote_call():
        return {"prediction": 1, "trace": _SAMPLE_REMOTE_TRACE}

    # The parent function that invokes the dummy remote service
    @mlflow.trace
    def predict():
        resp = dummy_remote_call()
        remote_trace = Trace.from_dict(resp["trace"])
        mlflow.add_trace(remote_trace)
        return resp["prediction"]

    db_request_id = "databricks-request-id"
    with set_prediction_context(Context(request_id=db_request_id)):
        predict()

    # Pop the trace to be written to the inference table
    trace = Trace.from_dict(pop_trace(request_id=db_request_id))

    assert trace.info.request_id == db_request_id
    assert len(trace.data.spans) == 3
    assert all(span.request_id == db_request_id for span in trace.data.spans)
    parent_span, child_span, grandchild_span = trace.data.spans
    assert child_span.parent_id == parent_span.span_id
    assert child_span._trace_id == parent_span._trace_id
    assert grandchild_span.parent_id == child_span.span_id
    assert grandchild_span._trace_id == parent_span._trace_id
    # Check if span information is correctly copied
    rs = Trace.from_dict(_SAMPLE_REMOTE_TRACE).data.spans[0]
    assert child_span.name == rs.name
    assert child_span.start_time_ns == rs.start_time_ns
    assert child_span.end_time_ns == rs.end_time_ns


def test_add_trace_logging_model_from_code():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model",
            python_model="tests/tracing/sample_code/model_with_add_trace.py",
            input_example=[1, 2],
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    # Trace should not be logged while logging / loading
    assert mlflow.get_last_active_trace() is None

    loaded_model.predict(1)
    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert len(trace.data.spans) == 2


@pytest.mark.parametrize(
    "inputs", [{"question": "Does mlflow support tracing?"}, "Does mlflow support tracing?", None]
)
@pytest.mark.parametrize("outputs", [{"answer": "Yes"}, "Yes", None])
@pytest.mark.parametrize(
    "intermediate_outputs",
    [
        {
            "retrieved_documents": ["mlflow documentation"],
            "system_prompt": ["answer the question with yes or no"],
        },
        None,
    ],
)
def test_log_trace_success(inputs, outputs, intermediate_outputs):
    start_time_ms = 1736144700
    execution_time_ms = 5129

    mlflow.log_trace(
        name="test",
        request=inputs,
        response=outputs,
        intermediate_outputs=intermediate_outputs,
        start_time_ms=start_time_ms,
        execution_time_ms=execution_time_ms,
    )

    trace = mlflow.get_last_active_trace()
    if inputs is not None:
        assert trace.data.request == json.dumps(inputs)
    else:
        assert trace.data.request is None
    if outputs is not None:
        assert trace.data.response == json.dumps(outputs)
    else:
        assert trace.data.response is None
    if intermediate_outputs is not None:
        assert trace.data.intermediate_outputs == intermediate_outputs
    spans = trace.data.spans
    assert len(spans) == 1
    root_span = spans[0]
    assert root_span.name == "test"
    assert root_span.start_time_ns == start_time_ms * 1000000
    assert root_span.end_time_ns == (start_time_ms + execution_time_ms) * 1000000


def test_log_trace_fail_within_span_context():
    with pytest.raises(MlflowException, match="Another trace is already set in the global context"):
        with mlflow.start_span("span"):
            mlflow.log_trace(
                request="Does mlflow support tracing?",
                response="Yes",
            )
