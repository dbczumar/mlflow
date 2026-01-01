from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import Issue as ProtoIssueEntity
from mlflow.protos.service_pb2 import IssueState as ProtoIssueState


class IssueState:
    """Enum for issue states."""

    DRAFT = "draft"
    OPEN = "open"
    CLOSED = "closed"

    _VALID_STATES = {DRAFT, OPEN, CLOSED}

    # Mapping from proto enum to string
    _PROTO_TO_STRING = {
        ProtoIssueState.DRAFT: DRAFT,
        ProtoIssueState.OPEN: OPEN,
        ProtoIssueState.CLOSED: CLOSED,
    }

    # Mapping from string to proto enum
    _STRING_TO_PROTO = {
        DRAFT: ProtoIssueState.DRAFT,
        OPEN: ProtoIssueState.OPEN,
        CLOSED: ProtoIssueState.CLOSED,
    }

    @classmethod
    def from_proto(cls, proto_state):
        """Convert proto enum to string state."""
        return cls._PROTO_TO_STRING.get(proto_state, cls.DRAFT)

    @classmethod
    def to_proto(cls, state):
        """Convert string state to proto enum."""
        return cls._STRING_TO_PROTO.get(state, ProtoIssueState.DRAFT)

    @classmethod
    def is_valid(cls, state):
        """Check if state is valid."""
        return state in cls._VALID_STATES


class IssueEntity(_MlflowObject):
    """
    Issue entity representing an identified problem from trace analysis.

    An issue belongs to an experiment and can be in one of three states:
    - draft: Created by analysis, pending review
    - open: User confirmed as valid issue
    - closed: Issue has been resolved

    Note: This class is named IssueEntity to distinguish from the Issue assessment
    class in mlflow.entities.assessment which links traces to issues.
    """

    def __init__(
        self,
        issue_id,
        experiment_id,
        name,
        description=None,
        state=IssueState.DRAFT,
        creation_time=None,
        last_update_time=None,
        tags=None,
    ):
        super().__init__()
        self._issue_id = issue_id
        self._experiment_id = experiment_id
        self._name = name
        self._description = description
        self._state = state
        self._creation_time = creation_time
        self._last_update_time = last_update_time
        self._tags = dict(tags) if tags else {}

    @property
    def issue_id(self):
        """Unique identifier for the issue (UUID)."""
        return self._issue_id

    @property
    def experiment_id(self):
        """ID of the experiment this issue belongs to."""
        return self._experiment_id

    @property
    def name(self):
        """Human-readable name/title of the issue."""
        return self._name

    @property
    def description(self):
        """Detailed description of the issue."""
        return self._description

    @property
    def state(self):
        """Current state of the issue (draft, open, closed)."""
        return self._state

    @property
    def creation_time(self):
        """Creation time in milliseconds since epoch."""
        return self._creation_time

    @property
    def last_update_time(self):
        """Last update time in milliseconds since epoch."""
        return self._last_update_time

    @property
    def tags(self):
        """Additional metadata as key-value pairs."""
        return self._tags

    @classmethod
    def from_proto(cls, proto):
        """Create Issue entity from protobuf message."""
        return cls(
            issue_id=proto.issue_id,
            experiment_id=proto.experiment_id,
            name=proto.name,
            description=proto.description if proto.HasField("description") else None,
            state=IssueState.from_proto(proto.state),
            creation_time=proto.creation_time if proto.creation_time else None,
            last_update_time=proto.last_update_time if proto.last_update_time else None,
            tags=dict(proto.tags) if proto.tags else None,
        )

    def to_proto(self):
        """Convert Issue entity to protobuf message."""
        proto = ProtoIssueEntity()
        proto.issue_id = self.issue_id
        proto.experiment_id = self.experiment_id
        proto.name = self.name
        if self.description is not None:
            proto.description = self.description
        proto.state = IssueState.to_proto(self.state)
        if self.creation_time is not None:
            proto.creation_time = self.creation_time
        if self.last_update_time is not None:
            proto.last_update_time = self.last_update_time
        if self.tags:
            proto.tags.update(self.tags)
        return proto
