from __future__ import annotations

import os

import pytest

from shamela_rag.config import Settings


def test_defaults_load(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate from any ambient SHAMELA_* vars (e.g. set in CI) so we test true code defaults.
    for key in list(os.environ):
        if key.startswith("SHAMELA_"):
            monkeypatch.delenv(key, raising=False)
    s = Settings(_env_file=None)
    assert s.postgres_port == 5433
    assert s.qdrant_url.startswith("http")
    assert s.postgres_dsn == "postgresql://shamela:shamela@localhost:5433/shamela_rag"
    assert s.chunk_min_tokens < s.chunk_max_tokens


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHAMELA_POSTGRES_DB", "testdb")
    monkeypatch.setenv("SHAMELA_QDRANT_COLLECTION", "test_coll")
    s = Settings(_env_file=None)
    assert s.postgres_db == "testdb"
    assert "testdb" in s.postgres_dsn
    assert s.qdrant_collection == "test_coll"
