from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import IssueComment as ProtoIssueComment


class IssueCommentEntity(_MlflowObject):
    """
    Issue comment entity representing a comment on an issue.

    A comment belongs to an issue and contains text content with
    optional author information and timestamps.
    """

    def __init__(
        self,
        comment_id,
        issue_id,
        content,
        author=None,
        creation_time=None,
        last_update_time=None,
    ):
        super().__init__()
        self._comment_id = comment_id
        self._issue_id = issue_id
        self._content = content
        self._author = author
        self._creation_time = creation_time
        self._last_update_time = last_update_time

    @property
    def comment_id(self):
        """Unique identifier for the comment (UUID)."""
        return self._comment_id

    @property
    def issue_id(self):
        """ID of the issue this comment belongs to."""
        return self._issue_id

    @property
    def content(self):
        """Comment text content."""
        return self._content

    @property
    def author(self):
        """Author name or identifier (optional)."""
        return self._author

    @property
    def creation_time(self):
        """Creation time in milliseconds since epoch."""
        return self._creation_time

    @property
    def last_update_time(self):
        """Last update time in milliseconds since epoch."""
        return self._last_update_time

    @classmethod
    def from_proto(cls, proto):
        """Create IssueComment entity from protobuf message."""
        return cls(
            comment_id=proto.comment_id,
            issue_id=proto.issue_id,
            content=proto.content,
            author=proto.author if proto.HasField("author") else None,
            creation_time=proto.creation_time if proto.creation_time else None,
            last_update_time=proto.last_update_time if proto.last_update_time else None,
        )

    def to_proto(self):
        """Convert IssueComment entity to protobuf message."""
        proto = ProtoIssueComment()
        proto.comment_id = self.comment_id
        proto.issue_id = self.issue_id
        proto.content = self.content
        if self.author is not None:
            proto.author = self.author
        if self.creation_time is not None:
            proto.creation_time = self.creation_time
        if self.last_update_time is not None:
            proto.last_update_time = self.last_update_time
        return proto
