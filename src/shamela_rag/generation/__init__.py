"""LLM generation layer (provider interface; prompt/answer assembly come later)."""

from __future__ import annotations

from shamela_rag.generation.provider import GenerationProvider, InMemoryGenerationProvider

__all__ = [
    "GenerationProvider",
    "InMemoryGenerationProvider",
]
