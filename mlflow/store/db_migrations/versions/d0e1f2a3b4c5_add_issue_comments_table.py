"""add issue_comments table

Create Date: 2025-07-10 15:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlIssueComment

# revision identifiers, used by Alembic.
revision = "d0e1f2a3b4c5"
down_revision = "c9e0f1a2b3d4"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlIssueComment.__tablename__,
        sa.Column("comment_id", sa.String(length=36), nullable=False),
        sa.Column("issue_id", sa.String(length=36), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("author", sa.String(length=256), nullable=True),
        sa.Column("creation_time", sa.BigInteger(), nullable=False),
        sa.Column("last_update_time", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["issue_id"],
            ["issues.issue_id"],
            name="fk_issue_comments_issue_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("comment_id", name="issue_comments_pk"),
    )

    with op.batch_alter_table(SqlIssueComment.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            "index_issue_comments_issue_id",
            ["issue_id"],
            unique=False,
        )
        batch_op.create_index(
            "index_issue_comments_creation_time",
            ["creation_time"],
            unique=False,
        )


def downgrade():
    op.drop_table(SqlIssueComment.__tablename__)
