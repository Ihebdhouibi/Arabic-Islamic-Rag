"""LLM generation layer (provider interface, prompt template, answer assembly)."""

from __future__ import annotations

from shamela_rag.generation.answer import Answer, AnswerAssembler, Citation
from shamela_rag.generation.local import LlamaCppGenerationProvider, OllamaGenerationProvider
from shamela_rag.generation.prompt import PromptPassage, render_general_qa_prompt
from shamela_rag.generation.provider import GenerationProvider, InMemoryGenerationProvider
from shamela_rag.generation.service import GeneralQAService

__all__ = [
    "Answer",
    "AnswerAssembler",
    "Citation",
    "GeneralQAService",
    "GenerationProvider",
    "InMemoryGenerationProvider",
    "LlamaCppGenerationProvider",
    "OllamaGenerationProvider",
    "PromptPassage",
    "render_general_qa_prompt",
]
