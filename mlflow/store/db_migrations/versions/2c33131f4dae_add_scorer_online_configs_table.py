"""add scorer_online_configs table

Create Date: 2025-01-27 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlScorerOnlineConfig

# revision identifiers, used by Alembic.
revision = "2c33131f4dae"
down_revision = "c9d4e5f6a7b8"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlScorerOnlineConfig.__tablename__,
        sa.Column("scorer_online_config_id", sa.String(length=36), nullable=False),
        sa.Column("scorer_id", sa.String(length=36), nullable=False),
        sa.Column("sample_rate", sa.types.Float(precision=53), nullable=False),
        sa.Column("filter_string", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["scorer_id"],
            ["scorers.scorer_id"],
            name="fk_scorer_online_configs_scorer_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("scorer_online_config_id", name="scorer_online_config_pk"),
        sa.UniqueConstraint("scorer_id", name="unique_scorer_online_config_scorer_id"),
    )


def downgrade():
    op.drop_table(SqlScorerOnlineConfig.__tablename__)
