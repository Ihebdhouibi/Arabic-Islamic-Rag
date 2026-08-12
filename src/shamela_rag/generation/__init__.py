"""LLM generation layer (provider interface and prompt template)."""

from __future__ import annotations

from shamela_rag.generation.prompt import PromptPassage, render_general_qa_prompt
from shamela_rag.generation.provider import GenerationProvider, InMemoryGenerationProvider

__all__ = [
    "GenerationProvider",
    "InMemoryGenerationProvider",
    "PromptPassage",
    "render_general_qa_prompt",
]
