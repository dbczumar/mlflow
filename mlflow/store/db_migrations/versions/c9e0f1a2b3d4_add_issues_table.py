"""add issues table

Create Date: 2025-07-10 14:30:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.entities.issue import IssueState
from mlflow.store.tracking.dbmodels.models import SqlIssue

# revision identifiers, used by Alembic.
revision = "c9e0f1a2b3d4"
down_revision = "2c33131f4dae"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlIssue.__tablename__,
        sa.Column("issue_id", sa.String(length=36), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=500), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("state", sa.String(length=20), nullable=False, default=IssueState.DRAFT),
        sa.Column("creation_time", sa.BigInteger(), nullable=False),
        sa.Column("last_update_time", sa.BigInteger(), nullable=False),
        sa.Column("tags", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_issues_experiment_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("issue_id", name="issues_pk"),
        sa.CheckConstraint(
            "state IN ('draft', 'open', 'closed')",
            name="issues_state",
        ),
    )

    with op.batch_alter_table(SqlIssue.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            "index_issues_experiment_id",
            ["experiment_id"],
            unique=False,
        )
        batch_op.create_index(
            "index_issues_state",
            ["state"],
            unique=False,
        )


def downgrade():
    op.drop_table(SqlIssue.__tablename__)
