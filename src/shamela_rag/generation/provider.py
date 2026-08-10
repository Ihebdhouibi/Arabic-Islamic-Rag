"""Generation provider interface and an in-memory test double.

Abstract contract for later local or API LLM backends (doc 09). This module is only the
transport shape: ``generate`` / optional ``generate_stream``. Prompt templates, citation
assembly, and end-to-end Q&A belong in later issues (#44–#46), not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _GenerateCall:
    prompt: str
    max_tokens: int | None
    streamed: bool


class GenerationProvider(ABC):
    """Pluggable LLM backend for answer generation (local weights or remote API)."""

    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Return the full model completion for ``prompt``."""

    def generate_stream(self, prompt: str, *, max_tokens: int | None = None) -> Iterator[str]:
        """Yield completion chunks. Default: one chunk equal to ``generate``."""
        yield self.generate(prompt, max_tokens=max_tokens)


class InMemoryGenerationProvider(GenerationProvider):
    """Deterministic offline provider for tests (no network / model weights).

    Looks up ``prompt`` in ``responses``; if missing, returns ``prefix + prompt``.
    Records each call in ``calls`` for spy assertions. Streaming yields fixed-size chunks
    of the same full string ``generate`` would return.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        *,
        prefix: str = "[generated] ",
        stream_chunk_size: int = 8,
    ) -> None:
        if stream_chunk_size <= 0:
            raise ValueError(f"stream_chunk_size must be positive, got {stream_chunk_size}")
        self._responses = dict(responses) if responses is not None else {}
        self._prefix = prefix
        self._stream_chunk_size = stream_chunk_size
        self.calls: list[_GenerateCall] = []

    def generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
        self.calls.append(_GenerateCall(prompt=prompt, max_tokens=max_tokens, streamed=False))
        text = self._complete(prompt)
        if max_tokens is not None:
            text = text[:max_tokens]
        return text

    def generate_stream(self, prompt: str, *, max_tokens: int | None = None) -> Iterator[str]:
        self.calls.append(_GenerateCall(prompt=prompt, max_tokens=max_tokens, streamed=True))
        text = self._complete(prompt)
        if max_tokens is not None:
            text = text[:max_tokens]
        size = self._stream_chunk_size
        for i in range(0, len(text), size):
            yield text[i : i + size]

    def _complete(self, prompt: str) -> str:
        if prompt in self._responses:
            return self._responses[prompt]
        return f"{self._prefix}{prompt}"
