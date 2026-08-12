"""LLM generation layer (provider interface, prompt template, answer assembly)."""

from __future__ import annotations

from shamela_rag.generation.answer import Answer, AnswerAssembler, Citation
from shamela_rag.generation.prompt import PromptPassage, render_general_qa_prompt
from shamela_rag.generation.provider import GenerationProvider, InMemoryGenerationProvider

__all__ = [
    "Answer",
    "AnswerAssembler",
    "Citation",
    "GenerationProvider",
    "InMemoryGenerationProvider",
    "PromptPassage",
    "render_general_qa_prompt",
]
