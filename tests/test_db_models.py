from __future__ import annotations

from shamela_rag.config import Settings
from shamela_rag.db.models import Base


def test_expected_tables_registered() -> None:
    assert {"books", "sections", "chunks"} <= set(Base.metadata.tables)


def test_chunk_keeps_source_and_retrieval_text_separate() -> None:
    cols = Base.metadata.tables["chunks"].columns
    assert "source_text" in cols
    assert "retrieval_text" in cols


def test_sqlalchemy_dsn_uses_psycopg_driver() -> None:
    assert Settings(_env_file=None).sqlalchemy_dsn.startswith("postgresql+psycopg://")
