"""Gated root-expansion sparse field (separate from surface BM25).

Maps surface tokens through the root dictionary, then BM25-encodes those roots as their own
sparse vector. Disabled by default; when enabled it is a low-weight extra arm so morphological
variants can match without polluting the primary surface field.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from shamela_rag.data.root_dictionary import RootDictionary
from shamela_rag.embeddings.bm25 import Bm25Encoder, SparseVector, tokenize
from shamela_rag.text.normalization import normalize_for_index

_STATE_VERSION = 1

DEFAULT_ROOT_EXPANSION_WEIGHT = 0.25


def _folded_key(token: str) -> str:
    parts = tokenize(token)
    if len(parts) == 1:
        return parts[0]
    return normalize_for_index(token)


class RootExpansionEncoder:
    """BM25 over dictionary roots; values are scaled by ``weight``."""

    def __init__(
        self,
        dictionary: RootDictionary,
        *,
        weight: float = DEFAULT_ROOT_EXPANSION_WEIGHT,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if weight < 0:
            raise ValueError(f"weight must be >= 0, got {weight}")
        self._weight = weight
        self._bm25 = Bm25Encoder(k1=k1, b=b)
        self._by_folded = _index_by_folded(dictionary)

    @property
    def is_fitted(self) -> bool:
        return self._bm25.is_fitted

    def root_terms(self, text: str) -> list[str]:
        terms: list[str] = []
        for token in tokenize(text):
            terms.extend(self._by_folded.get(token, ()))
        return terms

    def fit(self, corpus: Iterable[str]) -> RootExpansionEncoder:
        self._bm25.fit(self._expanded(text) for text in corpus)
        return self

    def encode_document(self, text: str) -> SparseVector:
        return self._scale(self._bm25.encode_document(self._expanded(text)))

    def encode_query(self, text: str) -> SparseVector:
        return self._scale(self._bm25.encode_query(self._expanded(text)))

    def _expanded(self, text: str) -> str:
        return " ".join(self.root_terms(text))

    def _scale(self, vector: SparseVector) -> SparseVector:
        if self._weight == 1.0:
            return vector
        return SparseVector(
            indices=vector.indices, values=[value * self._weight for value in vector.values]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "weight": self._weight,
            "bm25": self._bm25.to_dict(),
        }

    @classmethod
    def from_dict(cls, dictionary: RootDictionary, data: Mapping[str, Any]) -> RootExpansionEncoder:
        encoder = cls(dictionary, weight=float(data["weight"]))
        encoder._bm25 = Bm25Encoder.from_dict(data["bm25"])
        return encoder

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, dictionary: RootDictionary) -> RootExpansionEncoder:
        return cls.from_dict(dictionary, json.loads(path.read_text(encoding="utf-8")))


def _index_by_folded(dictionary: RootDictionary) -> dict[str, tuple[str, ...]]:
    folded: dict[str, list[str]] = {}
    for token, roots in dictionary.items():
        key = _folded_key(token)
        if not key:
            continue
        bucket = folded.setdefault(key, [])
        for root in roots:
            if root not in bucket:
                bucket.append(root)
    return {key: tuple(roots) for key, roots in folded.items()}
