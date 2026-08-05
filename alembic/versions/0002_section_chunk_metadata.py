"""flesh out section and chunk metadata

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

_SECTION_COLUMNS = (
    sa.Column("shamela_title_id", sa.Integer(), nullable=True),
    sa.Column("title_trail", sa.Text(), nullable=True),
    sa.Column("depth", sa.Integer(), nullable=True),
    sa.Column("path_source", sa.String(length=32), nullable=True),
    sa.Column("confidence", sa.String(length=16), nullable=True),
    sa.Column("start_page_id", sa.Integer(), nullable=True),
    sa.Column("end_page_id", sa.Integer(), nullable=True),
)

_CHUNK_COLUMNS = (
    sa.Column("context_header", sa.Text(), nullable=True),
    sa.Column("part", sa.String(length=16), nullable=True),
    sa.Column("start_page_id", sa.Integer(), nullable=True),
    sa.Column("end_page_id", sa.Integer(), nullable=True),
    sa.Column("start_offset", sa.Integer(), nullable=True),
    sa.Column("end_offset", sa.Integer(), nullable=True),
    sa.Column("token_count", sa.Integer(), nullable=True),
)


def upgrade() -> None:
    for column in _SECTION_COLUMNS:
        op.add_column("sections", column)
    for column in _CHUNK_COLUMNS:
        op.add_column("chunks", column)


def downgrade() -> None:
    for column in reversed(_CHUNK_COLUMNS):
        op.drop_column("chunks", column.name)
    for column in reversed(_SECTION_COLUMNS):
        op.drop_column("sections", column.name)
