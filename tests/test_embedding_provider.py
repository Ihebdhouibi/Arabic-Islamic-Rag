from __future__ import annotations

import math

import pytest

from shamela_rag.chunking.tokens import HeuristicTokenCounter
from shamela_rag.embeddings import EmbeddingProvider, InMemoryEmbeddingProvider


def test_in_memory_is_embedding_provider() -> None:
    provider: EmbeddingProvider = InMemoryEmbeddingProvider()
    assert provider.dims == 8
    assert provider.query_instruction is None
    assert isinstance(provider.tokenizer, HeuristicTokenCounter)


def test_in_memory_embed_documents_empty() -> None:
    provider = InMemoryEmbeddingProvider()
    assert provider.embed_documents([]) == []


def test_in_memory_embed_documents_dims_and_order() -> None:
    provider = InMemoryEmbeddingProvider(dims=4)
    vectors = provider.embed_documents(["alpha", "beta"])
    assert len(vectors) == 2
    assert all(len(v) == 4 for v in vectors)
    assert vectors[0] != vectors[1]


def test_in_memory_vectors_are_deterministic_and_unit_length() -> None:
    provider = InMemoryEmbeddingProvider(dims=8)
    text = "باب الهمزة"
    a = provider.embed_documents([text])[0]
    b = provider.embed_documents([text])[0]
    assert a == b
    norm = math.sqrt(sum(x * x for x in a))
    assert norm == pytest.approx(1.0)


def test_in_memory_embed_query_matches_document_without_instruction() -> None:
    provider = InMemoryEmbeddingProvider()
    text = "الصلاة"
    assert provider.embed_query(text) == provider.embed_documents([text])[0]


def test_in_memory_query_instruction_is_recorded_and_changes_query_vector() -> None:
    instruction = "Instruct: retrieve relevant passages\nQuery:"
    provider = InMemoryEmbeddingProvider(query_instruction=instruction)
    assert provider.query_instruction == instruction
    text = "ما هي الصلاة"
    with_instruction = provider.embed_query(text)
    plain = InMemoryEmbeddingProvider().embed_query(text)
    assert with_instruction != plain
    assert len(with_instruction) == provider.dims


def test_in_memory_rejects_non_positive_dims() -> None:
    with pytest.raises(ValueError, match="dims"):
        InMemoryEmbeddingProvider(dims=0)


def test_in_memory_tokenizer_counts() -> None:
    provider = InMemoryEmbeddingProvider()
    assert provider.tokenizer.count("hello, world") == 3
