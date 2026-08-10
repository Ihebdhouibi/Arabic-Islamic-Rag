"""Surface-form BM25 sparse encoder (the primary sparse arm).

Fits IDF and average document length on a corpus of lightly-normalized surface tokens (no root
expansion — names, sects, and titles stay precise), then encodes documents and queries as sparse
vectors for Qdrant. The full BM25 term weight (including IDF) is placed on the **document** side and
query terms are binary, so a sparse dot product reproduces the BM25 score.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from shamela_rag.text.normalization import normalize_for_index

_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class SparseVector:
    """A sparse vector as parallel term-id / weight arrays (Qdrant sparse format)."""

    indices: list[int]
    values: list[float]


def tokenize(text: str) -> list[str]:
    """Lightly-normalized surface tokens (index normalization, no root expansion)."""
    return _TOKEN.findall(normalize_for_index(text))


class Bm25Encoder:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")
        self._k1 = k1
        self._b = b
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocab)

    def term_id(self, term: str) -> int | None:
        """Sparse index assigned to a surface ``term``, or ``None`` if out of vocabulary."""
        return self._vocab.get(term)

    def fit(self, corpus: Iterable[str]) -> Bm25Encoder:
        doc_freq: Counter[str] = Counter()
        total_len = 0
        n_docs = 0
        for text in corpus:
            tokens = tokenize(text)
            n_docs += 1
            total_len += len(tokens)
            doc_freq.update(set(tokens))
        if n_docs == 0:
            raise ValueError("cannot fit on an empty corpus")
        self._avgdl = total_len / n_docs
        self._vocab = {term: idx for idx, term in enumerate(sorted(doc_freq))}
        self._idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }
        self._fitted = True
        return self

    def encode_document(self, text: str) -> SparseVector:
        self._require_fitted()
        counts = Counter(tokenize(text))
        doc_len = sum(counts.values())
        indices: list[int] = []
        values: list[float] = []
        for term, tf in counts.items():
            idx = self._vocab.get(term)
            if idx is None:
                continue
            denom = tf + self._k1 * (1 - self._b + self._b * doc_len / self._avgdl)
            weight = self._idf[term] * (tf * (self._k1 + 1)) / denom
            if weight == 0.0:
                continue
            indices.append(idx)
            values.append(weight)
        return SparseVector(indices=indices, values=values)

    def encode_query(self, text: str) -> SparseVector:
        self._require_fitted()
        indices: list[int] = []
        values: list[float] = []
        seen: set[int] = set()
        for term in tokenize(text):
            idx = self._vocab.get(term)
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)
            values.append(1.0)
        return SparseVector(indices=indices, values=values)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Bm25Encoder must be fit() before encoding")
