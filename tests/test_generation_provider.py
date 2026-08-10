from __future__ import annotations

import pytest

from shamela_rag.generation import GenerationProvider, InMemoryGenerationProvider


def test_in_memory_is_generation_provider() -> None:
    provider: GenerationProvider = InMemoryGenerationProvider()
    assert isinstance(provider, GenerationProvider)


def test_in_memory_uses_response_mapping() -> None:
    provider = InMemoryGenerationProvider({"prompt-a": "answer-a"})
    assert provider.generate("prompt-a") == "answer-a"
    assert len(provider.calls) == 1
    assert provider.calls[0].prompt == "prompt-a"
    assert provider.calls[0].streamed is False
    assert provider.calls[0].max_tokens is None


def test_in_memory_falls_back_to_prefix() -> None:
    provider = InMemoryGenerationProvider(prefix="OUT:")
    assert provider.generate("hello") == "OUT:hello"


def test_in_memory_respects_max_tokens() -> None:
    provider = InMemoryGenerationProvider({"p": "abcdefghij"})
    assert provider.generate("p", max_tokens=4) == "abcd"
    assert provider.calls[-1].max_tokens == 4


def test_in_memory_generate_stream_chunks_match_generate() -> None:
    provider = InMemoryGenerationProvider({"p": "abcdefghij"}, stream_chunk_size=3)
    chunks = list(provider.generate_stream("p"))
    assert chunks == ["abc", "def", "ghi", "j"]
    assert "".join(chunks) == provider.generate("p")
    assert provider.calls[0].streamed is True
    assert provider.calls[1].streamed is False


def test_in_memory_stream_respects_max_tokens() -> None:
    provider = InMemoryGenerationProvider({"p": "abcdefghij"}, stream_chunk_size=3)
    assert list(provider.generate_stream("p", max_tokens=5)) == ["abc", "de"]


def test_default_generate_stream_on_base_yields_full_text() -> None:
    """A provider that only implements ``generate`` still streams via the default."""

    class Minimal(GenerationProvider):
        def generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
            text = f"echo:{prompt}"
            return text if max_tokens is None else text[:max_tokens]

    chunks = list(Minimal().generate_stream("x"))
    assert chunks == ["echo:x"]


def test_rejects_non_positive_stream_chunk_size() -> None:
    with pytest.raises(ValueError, match="stream_chunk_size"):
        InMemoryGenerationProvider(stream_chunk_size=0)
