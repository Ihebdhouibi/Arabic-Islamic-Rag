from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from shamela_rag.config import get_settings
from shamela_rag.embeddings.bm25 import Bm25Encoder, SparseVector, tokenize
from shamela_rag.vectorstore.qdrant_store import ChunkPoint, QdrantStore

_CORPUS = [
    "قال الشافعي في الرسالة إن القياس أصل من أصول الفقه",
    "قال مالك في الموطأ عن نافع عن ابن عمر",
    "باب الطهارة والوضوء وأحكام المياه",
]


def test_bm25_state_roundtrips_through_save_load(tmp_path: Path) -> None:
    encoder = Bm25Encoder().fit(_CORPUS)
    path = tmp_path / "bm25.json"
    encoder.save(path)

    loaded = Bm25Encoder.load(path)
    assert loaded.vocabulary_size == encoder.vocabulary_size
    for text in [*_CORPUS, "ما رأي الشافعي في القياس"]:
        assert loaded.encode_document(text) == encoder.encode_document(text)
        assert loaded.encode_query(text) == encoder.encode_query(text)


def test_bm25_from_dict_matches_original() -> None:
    encoder = Bm25Encoder().fit(_CORPUS)
    clone = Bm25Encoder.from_dict(encoder.to_dict())
    assert clone.encode_query("الشافعي القياس") == encoder.encode_query("الشافعي القياس")


def test_bm25_to_dict_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        Bm25Encoder().to_dict()


def _dot(query: SparseVector, document: SparseVector) -> float:
    doc = dict(zip(document.indices, document.values, strict=True))
    return sum(
        weight * doc.get(idx, 0.0) for idx, weight in zip(query.indices, query.values, strict=True)
    )


def test_tokenize_normalizes_and_splits() -> None:
    assert tokenize("الصَّلَاةُ  والوُضوء") == ["الصلاه", "والوضوء"]


def test_fit_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        Bm25Encoder().fit([])


def test_encode_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        Bm25Encoder().encode_document("نص")


def test_encoder_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="k1"):
        Bm25Encoder(k1=-1.0)
    with pytest.raises(ValueError, match="b must be"):
        Bm25Encoder(b=1.5)


def test_rare_term_outweighs_common_term() -> None:
    encoder = Bm25Encoder().fit(_CORPUS)
    vector = encoder.encode_document(_CORPUS[0])
    weights = dict(zip(vector.indices, vector.values, strict=True))
    rare = encoder.term_id("الشافعي")  # appears in one doc
    common = encoder.term_id("قال")  # appears in two docs
    assert rare is not None and common is not None
    assert weights[rare] > weights[common]


def test_exact_name_query_ranks_correct_chunk() -> None:
    encoder = Bm25Encoder().fit(_CORPUS)
    documents = [encoder.encode_document(text) for text in _CORPUS]
    query = encoder.encode_query("ما رأي الشافعي")
    scores = [_dot(query, document) for document in documents]
    assert max(range(len(scores)), key=lambda i: scores[i]) == 0
    assert scores[0] > 0.0


def test_query_terms_outside_vocabulary_are_dropped() -> None:
    encoder = Bm25Encoder().fit(_CORPUS)
    query = encoder.encode_query("مصطلح غير موجود")
    assert query.indices == []
    assert query.values == []


def test_bm25_sparse_search_returns_right_chunk_via_qdrant() -> None:
    """Integration: fit, upsert sparse vectors, and query through Qdrant. Skips if unreachable."""
    store = QdrantStore(
        url=get_settings().qdrant_url, collection=f"test_bm25_{uuid.uuid4().hex}", dense_dim=2
    )
    try:
        store.client.get_collections()
    except Exception:  # noqa: BLE001 - any connection error means Qdrant is unavailable
        pytest.skip("Qdrant not reachable")

    store.ensure_collection()
    try:
        encoder = Bm25Encoder().fit(_CORPUS)
        points = [
            ChunkPoint(
                point_id=i,
                dense=[1.0, 0.0],
                sparse_indices=encoder.encode_document(text).indices,
                sparse_values=encoder.encode_document(text).values,
                payload={"doc": i},
            )
            for i, text in enumerate(_CORPUS)
        ]
        store.upsert(points)

        query = encoder.encode_query("ما رأي الشافعي")
        hits = store.search_sparse(query.indices, query.values, limit=1)
        assert hits[0].id == 0
    finally:
        store.delete_collection()
