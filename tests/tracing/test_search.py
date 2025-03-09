import pytest

import time
from unittest import mock

import mlflow
from mlflow.entities import (
    SpanEvent,
    SpanStatusCode,
    SpanType,
    Trace,
    TraceData,
    TraceInfo,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_TRACES_DEFAULT_MAX_RESULTS

from tests.tracing.helper import create_test_trace_info, get_traces


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


@pytest.fixture
def mock_client():
    client = mock.MagicMock()
    with mock.patch("mlflow.tracing.fluent.MlflowClient", return_value=client):
        yield client


def test_test_search_traces_empty(mock_client):
    mock_client.search_traces.return_value = PagedList([], token=None)

    traces = mlflow.search_traces()
    assert traces.empty

    default_columns = Trace.pandas_dataframe_columns()
    assert traces.columns.tolist() == default_columns

    traces = mlflow.search_traces(extract_fields=["foo.inputs.bar"])
    assert traces.columns.tolist() == [*default_columns, "foo.inputs.bar"]

    mock_client.search_traces.assert_called()


def test_search_traces(mock_client):
    mock_client.search_traces.return_value = PagedList(
        [
            Trace(
                info=create_test_trace_info(f"tr-{i}"),
                data=TraceData([], "", ""),
            )
            for i in range(10)
        ],
        token=None,
    )

    traces = mlflow.search_traces(
        experiment_ids=["1"],
        filter_string="name = 'foo'",
        max_results=10,
        order_by=["timestamp DESC"],
    )

    assert len(traces) == 10
    mock_client.search_traces.assert_called_once_with(
        experiment_ids=["1"],
        run_id=None,
        filter_string="name = 'foo'",
        max_results=10,
        order_by=["timestamp DESC"],
        page_token=None,
        model_id=None,
    )


def test_search_traces_with_pagination(mock_client):
    traces = [
        Trace(
            info=create_test_trace_info(f"tr-{i}"),
            data=TraceData([], "", ""),
        )
        for i in range(30)
    ]

    mock_client.search_traces.side_effect = [
        PagedList(traces[:10], token="token-1"),
        PagedList(traces[10:20], token="token-2"),
        PagedList(traces[20:], token=None),
    ]

    traces = mlflow.search_traces(experiment_ids=["1"])

    assert len(traces) == 30
    common_args = {
        "experiment_ids": ["1"],
        "run_id": None,
        "max_results": SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        "filter_string": None,
        "order_by": None,
    }
    mock_client.search_traces.assert_has_calls(
        [
            mock.call(**common_args, page_token=None, model_id=None),
            mock.call(**common_args, page_token="token-1", model_id=None),
            mock.call(**common_args, page_token="token-2", model_id=None),
        ]
    )


def test_search_traces_with_default_experiment_id(mock_client):
    mock_client.search_traces.return_value = PagedList([], token=None)
    with mock.patch("mlflow.tracing.fluent._get_experiment_id", return_value="123"):
        mlflow.search_traces()

    mock_client.search_traces.assert_called_once_with(
        experiment_ids=["123"],
        run_id=None,
        filter_string=None,
        max_results=SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by=None,
        page_token=None,
        model_id=None,
    )


def test_search_traces_yields_expected_dataframe_contents(monkeypatch):
    model = DefaultTestModel()
    client = mlflow.MlflowClient()
    expected_traces = []
    for _ in range(10):
        model.predict(2, 5)
        time.sleep(0.1)

        # The in-memory trace returned from get_last_active_trace() is not guaranteed to be
        # exactly same as the trace stored in the backend (e.g., tags created by the backend).
        # Therefore, we fetch the trace from the backend to compare the results.
        trace = client.get_trace(mlflow.get_last_active_trace().info.request_id)
        expected_traces.append(trace)

    df = mlflow.search_traces(max_results=10, order_by=["timestamp ASC"])
    assert df.columns.tolist() == [
        "request_id",
        "trace",
        "timestamp_ms",
        "status",
        "execution_time_ms",
        "request",
        "response",
        "request_metadata",
        "spans",
        "tags",
    ]
    for idx, trace in enumerate(expected_traces):
        assert df.iloc[idx].request_id == trace.info.request_id
        assert df.iloc[idx].trace.info.request_id == trace.info.request_id
        assert df.iloc[idx].timestamp_ms == trace.info.timestamp_ms
        assert df.iloc[idx].status == trace.info.status
        assert df.iloc[idx].execution_time_ms == trace.info.execution_time_ms
        assert df.iloc[idx].request == json.loads(trace.data.request)
        assert df.iloc[idx].response == json.loads(trace.data.response)
        assert df.iloc[idx].request_metadata == trace.info.request_metadata
        assert df.iloc[idx].spans == [s.to_dict() for s in trace.data.spans]
        assert df.iloc[idx].tags == trace.info.tags


def test_search_traces_handles_missing_response_tags_and_metadata(monkeypatch):
    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return [
                Trace(
                    info=TraceInfo(
                        request_id=5,
                        experiment_id="test",
                        timestamp_ms=1,
                        execution_time_ms=2,
                        status=TraceStatus.OK,
                    ),
                    data=TraceData(
                        spans=[],
                        request="request",
                        # Response is missing
                    ),
                )
            ]

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces()
    assert df["response"].isnull().all()
    assert df["tags"].tolist() == [{}]
    assert df["request_metadata"].tolist() == [{}]


def test_search_traces_extracts_fields_as_expected(monkeypatch):
    model = DefaultTestModel()
    model.predict(2, 5)

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces(
        extract_fields=["predict.inputs.x", "predict.outputs", "add_one_with_custom_name.inputs.z"]
    )
    assert df["predict.inputs.x"].tolist() == [2]
    assert df["predict.outputs"].tolist() == [64]
    assert df["add_one_with_custom_name.inputs.z"].tolist() == [7]


# Test cases should cover case where there are no spans at all
def test_search_traces_with_no_spans(monkeypatch):
    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return []

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces()
    assert df.empty


# no spans have the input or output with name,
# some span has an input but we’re looking for output,
def test_search_traces_with_input_and_no_output(monkeypatch):
    with mlflow.start_span(name="with_input_and_no_output") as span:
        span.set_inputs({"a": 1})

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces(
        extract_fields=["with_input_and_no_output.inputs.a", "with_input_and_no_output.outputs"]
    )
    assert df["with_input_and_no_output.inputs.a"].tolist() == [1]
    assert df["with_input_and_no_output.outputs"].isnull().all()


# Test case where span content is invalid
def test_search_traces_with_invalid_span_content(monkeypatch):
    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            # Invalid span content
            return [
                Trace(
                    info=TraceInfo(
                        request_id=5,
                        experiment_id="test",
                        timestamp_ms=1,
                        execution_time_ms=2,
                        status=TraceStatus.OK,
                    ),
                    data=TraceData(spans=[None], request="request", response="response"),
                )
            ]

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    with pytest.raises(AttributeError, match="NoneType"):
        mlflow.search_traces()


# Test case where span inputs / outputs aren’t dict
def test_search_traces_with_non_dict_span_inputs_outputs(monkeypatch):
    with mlflow.start_span(name="non_dict_span") as span:
        span.set_inputs(["a", "b"])
        span.set_outputs([1, 2, 3])

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces(
        extract_fields=["non_dict_span.inputs", "non_dict_span.outputs", "non_dict_span.inputs.x"]
    )
    assert df["non_dict_span.inputs"].tolist() == [["a", "b"]]
    assert df["non_dict_span.outputs"].tolist() == [[1, 2, 3]]
    assert df["non_dict_span.inputs.x"].isnull().all()


# Test case where there are multiple spans with the same name
def test_search_traces_with_multiple_spans_with_same_name(monkeypatch):
    class TestModel:
        @mlflow.trace(name="duplicate_name")
        def predict(self, x, y):
            z = x + y
            z = self.add_one(z)
            z = mlflow.trace(self.square)(z)
            return z  # noqa: RET504

        @mlflow.trace(span_type=SpanType.LLM, name="duplicate_name", attributes={"delta": 1})
        def add_one(self, z):
            return z + 1

        def square(self, t):
            res = t**2
            time.sleep(0.1)
            return res

    model = TestModel()
    model.predict(2, 5)

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces(
        extract_fields=[
            "duplicate_name.inputs.y",
            "duplicate_name.inputs.x",
            "duplicate_name.inputs.z",
            "duplicate_name_1.inputs.x",
            "duplicate_name_1.inputs.y",
            "duplicate_name_2.inputs.z",
        ]
    )
    # Duplicate spans would all be null
    assert df["duplicate_name.inputs.y"].isnull().all()
    assert df["duplicate_name.inputs.x"].isnull().all()
    assert df["duplicate_name.inputs.z"].isnull().all()
    assert df["duplicate_name_1.inputs.x"].tolist() == [2]
    assert df["duplicate_name_1.inputs.y"].tolist() == [5]
    assert df["duplicate_name_2.inputs.z"].tolist() == [7]


# Test a field that doesn’t exist for extraction - we shouldn’t throw, just return empty column
def test_search_traces_with_non_existent_field(monkeypatch):
    model = DefaultTestModel()
    model.predict(2, 5)

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces(
        extract_fields=[
            "predict.inputs.k",
            "predict.inputs.x",
            "predict.outputs",
            "add_one_with_custom_name.inputs.z",
        ]
    )
    assert df["predict.inputs.k"].isnull().all()
    assert df["predict.inputs.x"].tolist() == [2]
    assert df["predict.outputs"].tolist() == [64]
    assert df["add_one_with_custom_name.inputs.z"].tolist() == [7]


# Test experiment ID doesn’t need to be specified
def test_search_traces_without_experiment_id(monkeypatch):
    model = DefaultTestModel()
    model.predict(2, 5)

    class MockMlflowClient:
        def search_traces(self, experiment_ids, *args, **kwargs):
            assert experiment_ids == ["0"]
            return get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    mlflow.search_traces()


def test_search_traces_span_and_field_name_with_dot():
    with mlflow.start_span(name="span.name") as span:
        span.set_inputs({"a.b": 0})
        span.set_outputs({"x.y": 1})

    df = mlflow.search_traces(
        extract_fields=[
            "`span.name`.inputs",
            "`span.name`.inputs.`a.b`",
            "`span.name`.outputs",
            "`span.name`.outputs.`x.y`",
        ]
    )

    assert df["span.name.inputs"].tolist() == [{"a.b": 0}]
    assert df["span.name.inputs.a.b"].tolist() == [0]
    assert df["span.name.outputs"].tolist() == [{"x.y": 1}]
    assert df["span.name.outputs.x.y"].tolist() == [1]


def test_search_traces_with_span_name(monkeypatch):
    class TestModel:
        @mlflow.trace(name="span.llm")
        def predict(self, x, y):
            z = x + y
            z = self.add_one(z)
            z = mlflow.trace(self.square)(z)
            return z  # noqa: RET504

        @mlflow.trace(span_type=SpanType.LLM, name="span.invalidname", attributes={"delta": 1})
        def add_one(self, z):
            return z + 1

        def square(self, t):
            res = t**2
            time.sleep(0.1)
            return res

    model = TestModel()
    model.predict(2, 5)

    class MockMlflowClient:
        def search_traces(self, experiment_ids, *args, **kwargs):
            return get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)


def test_search_traces_with_run_id():
    def _create_trace(name, tags=None):
        with mlflow.start_span(name=name) as span:
            for k, v in (tags or {}).items():
                mlflow.MlflowClient().set_trace_tag(request_id=span.request_id, key=k, value=v)
        return span.request_id

    def _get_names(traces):
        tags = traces["tags"].tolist()
        return [tags[i].get(TraceTagKey.TRACE_NAME) for i in range(len(tags))]

    with mlflow.start_run() as run1:
        _create_trace(name="tr-1")
        _create_trace(name="tr-2", tags={"fruit": "apple"})

    with mlflow.start_run() as run2:
        _create_trace(name="tr-3")
        _create_trace(name="tr-4", tags={"fruit": "banana"})
        _create_trace(name="tr-5", tags={"fruit": "apple"})

    traces = mlflow.search_traces()
    assert _get_names(traces) == ["tr-5", "tr-4", "tr-3", "tr-2", "tr-1"]

    traces = mlflow.search_traces(run_id=run1.info.run_id)
    assert _get_names(traces) == ["tr-2", "tr-1"]

    traces = mlflow.search_traces(
        run_id=run2.info.run_id,
        filter_string="tag.fruit = 'apple'",
    )
    assert _get_names(traces) == ["tr-5"]

    with pytest.raises(MlflowException, match="You cannot filter by run_id when it is already"):
        mlflow.search_traces(
            run_id=run2.info.run_id,
            filter_string="metadata.mlflow.sourceRun = '123'",
        )
