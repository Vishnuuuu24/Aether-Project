"""outcome table — outer-loop outcome capture (docs/11 §3; docs/15 T7.2c)

Adds the append-only `outcome` table so recorded clinical outcomes persist as
queryable rows (joinable to prior outputs + version snapshots for later human-gated
retraining), alongside the `OUTCOME_CAPTURE` audit record. Forward-only per docs/09.

Revision ID: 0003_outcome
Revises: 0002_document_node
Create Date: 2026-07-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003_outcome"
down_revision: str | None = "0002_document_node"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "outcome",
        sa.Column("outcome_id", sa.Uuid(), nullable=False),
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("outcome_type", sa.String(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("detail", sa.String(), nullable=False),
        sa.Column("code", sa.String(), nullable=True),
        sa.Column("linked_output_ids", postgresql.JSONB(), nullable=False),
        sa.Column("versions", postgresql.JSONB(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("outcome_id"),
    )
    op.create_index(op.f("ix_outcome_patient_id"), "outcome", ["patient_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_outcome_patient_id"), table_name="outcome")
    op.drop_table("outcome")
