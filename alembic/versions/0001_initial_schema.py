"""initial schema: books, sections, chunks

Revision ID: 0001
Revises:
Create Date: 2026-08-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "books",
        sa.Column("book_id", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("title_ar", sa.Text(), nullable=True),
        sa.Column("author_name_ar", sa.Text(), nullable=True),
        sa.Column("author_death_hijri", sa.Integer(), nullable=True),
        sa.Column("category_id", sa.Integer(), nullable=True),
        sa.Column("book_type_label", sa.String(length=64), nullable=True),
    )
    op.create_table(
        "sections",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "book_id",
            sa.Integer(),
            sa.ForeignKey("books.book_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_id",
            sa.Integer(),
            sa.ForeignKey("sections.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("title_text", sa.Text(), nullable=True),
        sa.Column("boundary_source", sa.String(length=32), nullable=True),
    )
    op.create_index("ix_sections_book_id", "sections", ["book_id"])
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "book_id",
            sa.Integer(),
            sa.ForeignKey("books.book_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "section_id",
            sa.Integer(),
            sa.ForeignKey("sections.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("content_role", sa.String(length=16), nullable=False),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column("retrieval_text", sa.Text(), nullable=True),
    )
    op.create_index("ix_chunks_book_id", "chunks", ["book_id"])


def downgrade() -> None:
    op.drop_index("ix_chunks_book_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_sections_book_id", table_name="sections")
    op.drop_table("sections")
    op.drop_table("books")
