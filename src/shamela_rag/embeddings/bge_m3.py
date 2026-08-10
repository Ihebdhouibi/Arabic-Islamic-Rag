"""BGE-M3 backend for ``EmbeddingProvider``.

Produces dense (1024-d, normalized) embeddings and, when enabled, BGE-M3's learned **lexical
sparse** weights for the hybrid retrieval / M6 ablation. Uses ``FlagEmbedding``; the optional
``[bge]`` extra installs the heavy deps and integration tests skip when weights are absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from shamela_rag.chunking.tokens import TokenCounter
from shamela_rag.embeddings.provider import EmbeddingProvider

BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_DIMS = 1024


@dataclass(frozen=True)
class SparseEmbedding:
    """A learned sparse vector as parallel token-id / weight arrays (Qdrant sparse format)."""

    indices: list[int]
    values: list[float]


def lexical_weights_to_sparse(weights: dict[str, float]) -> SparseEmbedding:
    """Convert BGE-M3 ``lexical_weights`` (``{token_id: weight}``) to a ``SparseEmbedding``."""
    indices: list[int] = []
    values: list[float] = []
    for token_id, weight in weights.items():
        value = float(weight)
        if value == 0.0:
            continue
        indices.append(int(token_id))
        values.append(value)
    return SparseEmbedding(indices=indices, values=values)


def _load_bge_m3(model_id: str, *, device: str | None, use_fp16: bool) -> Any:
    try:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            'BgeM3EmbeddingProvider requires optional deps: pip install "shamela-rag[bge]"'
        ) from exc

    kwargs: dict[str, Any] = {"use_fp16": use_fp16}
    if device is not None:
        kwargs["devices"] = device
    return BGEM3FlagModel(model_id, **kwargs)


class _HFTokenizerCounter:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


class BgeM3EmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        model_id: str = BGE_M3_MODEL_ID,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 8192,
        use_fp16: bool = False,
        enable_sparse: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        self._model = _load_bge_m3(model_id, device=device, use_fp16=use_fp16)
        self._batch_size = batch_size
        self._max_length = max_length
        self._enable_sparse = enable_sparse
        self._tokenizer_counter: TokenCounter = _HFTokenizerCounter(self._model.tokenizer)

    @property
    def dims(self) -> int:
        return BGE_M3_DIMS

    @property
    def tokenizer(self) -> TokenCounter:
        return self._tokenizer_counter

    @property
    def query_instruction(self) -> str | None:
        return None  # BGE-M3 retrieval uses no query instruction prefix

    @property
    def sparse_enabled(self) -> bool:
        return self._enable_sparse

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        output = self._encode(list(texts), sparse=False)
        return self._dense_rows(output["dense_vecs"])

    def embed_query(self, text: str) -> list[float]:
        output = self._encode([text], sparse=False)
        return self._dense_rows(output["dense_vecs"])[0]

    def embed_documents_sparse(self, texts: Sequence[str]) -> list[SparseEmbedding]:
        self._require_sparse()
        if not texts:
            return []
        output = self._encode(list(texts), sparse=True)
        return [lexical_weights_to_sparse(weights) for weights in output["lexical_weights"]]

    def embed_query_sparse(self, text: str) -> SparseEmbedding:
        self._require_sparse()
        output = self._encode([text], sparse=True)
        return lexical_weights_to_sparse(output["lexical_weights"][0])

    def _encode(self, texts: list[str], *, sparse: bool) -> dict[str, Any]:
        output: dict[str, Any] = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=sparse,
            return_colbert_vecs=False,
        )
        return output

    def _dense_rows(self, dense_vecs: Any) -> list[list[float]]:
        rows = dense_vecs.tolist() if hasattr(dense_vecs, "tolist") else list(dense_vecs)
        out: list[list[float]] = []
        for row in rows:
            vector = [float(x) for x in row]
            if len(vector) != BGE_M3_DIMS:
                raise ValueError(
                    f"embedding dims mismatch: got {len(vector)}, expected {BGE_M3_DIMS}"
                )
            out.append(vector)
        return out

    def _require_sparse(self) -> None:
        if not self._enable_sparse:
            raise RuntimeError("sparse output is disabled; construct with enable_sparse=True")
