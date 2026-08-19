from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest

from shamela_rag.config import get_settings
from shamela_rag.data.root_dictionary import RootDictionary, load_root_dictionary
from shamela_rag.embeddings.bm25 import Bm25Encoder, SparseVector, tokenize
from shamela_rag.embeddings.root_field import DEFAULT_ROOT_EXPANSION_WEIGHT, RootExpansionEncoder
from shamela_rag.retrieval.results import RetrievedChunk
from shamela_rag.retrieval.service import RetrievalConfig, RetrievalService
from shamela_rag.retrieval.translate import InMemoryTranslator
from shamela_rag.vectorstore.qdrant_store import ROOT_VECTOR_NAME, ChunkPoint, QdrantStore

_FIXTURE = Path(__file__).parent / "fixtures" / "root_dictionary_sample.jsonl"
_DOC_SALAH = "الصلاة واجبة على كل مسلم"
_QUERY_VARIANT = "صلاته"


def _dot(query: SparseVector, document: SparseVector) -> float:
    doc = dict(zip(document.indices, document.values, strict=True))
    return sum(
        weight * doc.get(idx, 0.0) for idx, weight in zip(query.indices, query.values, strict=True)
    )


@pytest.fixture
def dictionary() -> RootDictionary:
    return load_root_dictionary(_FIXTURE)


@pytest.fixture
def encoder(dictionary: RootDictionary) -> RootExpansionEncoder:
    return RootExpansionEncoder(dictionary).fit([_DOC_SALAH, "كتاب يكتب"])


def test_root_terms_map_morphological_variants(encoder: RootExpansionEncoder) -> None:
    salah = set(encoder.root_terms(_DOC_SALAH))
    variant = set(encoder.root_terms(_QUERY_VARIANT))
    assert salah & variant
    assert "صلو" in salah
    assert "صلو" in variant


def test_surface_bm25_does_not_index_roots() -> None:
    surface = Bm25Encoder().fit([_DOC_SALAH])
    assert surface.term_id("صلو") is None
    assert surface.term_id("صلي") is None
    assert tokenize(_DOC_SALAH)


def test_unknown_tokens_yield_no_roots(encoder: RootExpansionEncoder) -> None:
    assert encoder.root_terms("xyzzy") == []
    assert encoder.encode_query("xyzzy").indices == []


def test_weight_scales_sparse_values(dictionary: RootDictionary) -> None:
    full = RootExpansionEncoder(dictionary, weight=1.0).fit([_DOC_SALAH])
    low = RootExpansionEncoder(dictionary, weight=0.25).fit([_DOC_SALAH])
    full_vec = full.encode_document(_DOC_SALAH)
    low_vec = low.encode_document(_DOC_SALAH)
    assert full_vec.indices == low_vec.indices
    assert low_vec.values == pytest.approx([value * 0.25 for value in full_vec.values])
    assert DEFAULT_ROOT_EXPANSION_WEIGHT == 0.25


def test_root_query_matches_variant_surface_does_not(encoder: RootExpansionEncoder) -> None:
    surface = Bm25Encoder().fit([_DOC_SALAH])
    assert _dot(surface.encode_query(_QUERY_VARIANT), surface.encode_document(_DOC_SALAH)) == 0.0
    assert _dot(encoder.encode_query(_QUERY_VARIANT), encoder.encode_document(_DOC_SALAH)) > 0.0


def test_root_encoder_state_roundtrips(encoder: RootExpansionEncoder, tmp_path: Path) -> None:
    path = tmp_path / "root_expansion_state.json"
    encoder.save(path)
    loaded = RootExpansionEncoder.load(path, load_root_dictionary(_FIXTURE))
    assert loaded.encode_query(_QUERY_VARIANT) == encoder.encode_query(_QUERY_VARIANT)
    assert loaded.encode_document(_DOC_SALAH) == encoder.encode_document(_DOC_SALAH)


def test_ab_hook_skips_root_when_disabled() -> None:
    root = _FakeRetriever()
    assert _service(root, use_root=False).retrieve("سؤال") == []
    assert root.called is False


def test_ab_hook_uses_root_when_enabled() -> None:
    root = _FakeRetriever()
    assert _service(root, use_root=True).retrieve("سؤال") == []
    assert root.called is True


def test_root_sparse_search_via_qdrant(encoder: RootExpansionEncoder) -> None:
    store = QdrantStore(
        url=get_settings().qdrant_url, collection=f"test_root_{uuid.uuid4().hex}", dense_dim=2
    )
    try:
        store.client.get_collections()
    except Exception:  # noqa: BLE001 - Qdrant unavailable
        pytest.skip("Qdrant not reachable")
    store.ensure_collection()
    try:
        doc_vec = encoder.encode_document(_DOC_SALAH)
        store.upsert(
            [
                ChunkPoint(
                    point_id=1,
                    dense=[1.0, 0.0],
                    root_sparse_indices=doc_vec.indices,
                    root_sparse_values=doc_vec.values,
                )
            ]
        )
        query = encoder.encode_query(_QUERY_VARIANT)
        hits = store.search_sparse(
            query.indices, query.values, limit=1, vector_name=ROOT_VECTOR_NAME
        )
        assert hits[0].id == 1
    finally:
        store.delete_collection()


class _FakeRetriever:
    def __init__(self) -> None:
        self.called = False

    def search(
        self, query: str, *, limit: int = 10, filters: object | None = None
    ) -> list[RetrievedChunk]:
        self.called = True
        return []


def _service(root: _FakeRetriever, *, use_root: bool) -> RetrievalService:
    idle = _FakeRetriever()
    idle.called = False
    return RetrievalService(
        translator=InMemoryTranslator(),
        dense_retriever=idle,  # type: ignore[arg-type]
        sparse_retriever=idle,  # type: ignore[arg-type]
        reranker=Mock(),
        expander=Mock(),
        session_factory=Mock(),
        config=RetrievalConfig(use_root_expansion=use_root, translate=False),
        root_retriever=root,  # type: ignore[arg-type]
    )
