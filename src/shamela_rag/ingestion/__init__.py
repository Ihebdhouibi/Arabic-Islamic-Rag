"""Ingestion pipeline (chunk -> embed -> encode -> Qdrant + Postgres)."""

from __future__ import annotations

from shamela_rag.ingestion.pipeline import (
    BookIngestSummary,
    IngestionService,
    dense_input,
)

__all__ = [
    "BookIngestSummary",
    "IngestionService",
    "dense_input",
]
