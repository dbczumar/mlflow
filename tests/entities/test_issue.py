import pytest

from mlflow.entities.issue import IssueEntity, IssueState
from mlflow.protos.service_pb2 import IssueState as ProtoIssueState


class TestIssueState:
    def test_valid_states(self):
        assert IssueState.DRAFT == "draft"
        assert IssueState.OPEN == "open"
        assert IssueState.CLOSED == "closed"

    def test_is_valid(self):
        assert IssueState.is_valid("draft") is True
        assert IssueState.is_valid("open") is True
        assert IssueState.is_valid("closed") is True
        assert IssueState.is_valid("invalid") is False
        assert IssueState.is_valid("") is False

    def test_from_proto(self):
        assert IssueState.from_proto(ProtoIssueState.DRAFT) == "draft"
        assert IssueState.from_proto(ProtoIssueState.OPEN) == "open"
        assert IssueState.from_proto(ProtoIssueState.CLOSED) == "closed"

    def test_to_proto(self):
        assert IssueState.to_proto("draft") == ProtoIssueState.DRAFT
        assert IssueState.to_proto("open") == ProtoIssueState.OPEN
        assert IssueState.to_proto("closed") == ProtoIssueState.CLOSED


class TestIssueEntity:
    def test_creation(self):
        issue = IssueEntity(
            issue_id="test-issue-id",
            experiment_id="123",
            name="Test Issue",
            description="Test description",
            state=IssueState.DRAFT,
            creation_time=1000,
            last_update_time=2000,
            tags={"key": "value"},
        )

        assert issue.issue_id == "test-issue-id"
        assert issue.experiment_id == "123"
        assert issue.name == "Test Issue"
        assert issue.description == "Test description"
        assert issue.state == "draft"
        assert issue.creation_time == 1000
        assert issue.last_update_time == 2000
        assert issue.tags == {"key": "value"}

    def test_creation_with_defaults(self):
        issue = IssueEntity(
            issue_id="test-issue-id",
            experiment_id="123",
            name="Test Issue",
        )

        assert issue.issue_id == "test-issue-id"
        assert issue.experiment_id == "123"
        assert issue.name == "Test Issue"
        assert issue.description is None
        assert issue.state == "draft"  # Default state
        assert issue.creation_time is None
        assert issue.last_update_time is None
        assert issue.tags == {}

    def test_to_proto(self):
        issue = IssueEntity(
            issue_id="test-issue-id",
            experiment_id="123",
            name="Test Issue",
            description="Test description",
            state=IssueState.OPEN,
            creation_time=1000,
            last_update_time=2000,
            tags={"key": "value"},
        )

        proto = issue.to_proto()

        assert proto.issue_id == "test-issue-id"
        assert proto.experiment_id == "123"
        assert proto.name == "Test Issue"
        assert proto.description == "Test description"
        assert proto.state == ProtoIssueState.OPEN
        assert proto.creation_time == 1000
        assert proto.last_update_time == 2000
        assert dict(proto.tags) == {"key": "value"}

    def test_from_proto(self):
        issue = IssueEntity(
            issue_id="test-issue-id",
            experiment_id="123",
            name="Test Issue",
            description="Test description",
            state=IssueState.CLOSED,
            creation_time=1000,
            last_update_time=2000,
            tags={"key": "value"},
        )

        proto = issue.to_proto()
        restored = IssueEntity.from_proto(proto)

        assert restored.issue_id == issue.issue_id
        assert restored.experiment_id == issue.experiment_id
        assert restored.name == issue.name
        assert restored.description == issue.description
        assert restored.state == issue.state
        assert restored.creation_time == issue.creation_time
        assert restored.last_update_time == issue.last_update_time
        assert restored.tags == issue.tags

    def test_to_proto_without_optional_fields(self):
        issue = IssueEntity(
            issue_id="test-issue-id",
            experiment_id="123",
            name="Test Issue",
        )

        proto = issue.to_proto()

        assert proto.issue_id == "test-issue-id"
        assert proto.experiment_id == "123"
        assert proto.name == "Test Issue"
        assert not proto.HasField("description")
        assert proto.state == ProtoIssueState.DRAFT
        assert proto.creation_time == 0
        assert proto.last_update_time == 0
