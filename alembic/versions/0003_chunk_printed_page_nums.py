"""add printed page numbers to chunks

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-01
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("start_page_num", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("end_page_num", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "end_page_num")
    op.drop_column("chunks", "start_page_num")
