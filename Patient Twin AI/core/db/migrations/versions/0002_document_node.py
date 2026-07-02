"""document_node + coded-node status — document coding (docs/04 §4; T3.1)

Adds the versioned `document_node` table (provenance for coded documents) and a
`status` column (proposed | committed) to `observation_node` and `allergy_node`, so
sub-threshold coded resources persist as `proposed` uniformly with condition/
medication nodes (which already carry status). Forward-only per docs/09.

Revision ID: 0002_document_node
Revises: 0001_baseline
Create Date: 2026-07-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0002_document_node"
down_revision: str | None = "0001_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_node",
        sa.Column("doc_type", sa.String(), nullable=False),
        sa.Column("uri", sa.String(), nullable=True),
        sa.Column("ocr_ref", sa.String(), nullable=True),
        sa.Column("codes", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("patient_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("supersedes", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_document_node_patient_id"), "document_node", ["patient_id"], unique=False
    )

    # Coded observation/allergy nodes gain a proposed|committed status (docs/04 §4),
    # matching condition/medication nodes. server_default keeps existing rows valid.
    for table in ("observation_node", "allergy_node"):
        op.add_column(
            table,
            sa.Column(
                "status", sa.String(), nullable=False, server_default="committed"
            ),
        )


def downgrade() -> None:
    op.drop_column("allergy_node", "status")
    op.drop_column("observation_node", "status")
    op.drop_index(op.f("ix_document_node_patient_id"), table_name="document_node")
    op.drop_table("document_node")
