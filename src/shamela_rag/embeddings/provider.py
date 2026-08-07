"""Dense embedding provider interface and an in-memory test double.

Shared contract for later Qwen3-8B / BGE-M3 backends (``embed_documents``, ``embed_query``,
``dims``, ``tokenizer``, optional ``query_instruction`` recorded for eval). The in-memory
provider is deterministic and offline so unit tests need no model weights.
"""

from __future__ import annotations

import hashlib
import math
import struct
from abc import ABC, abstractmethod
from collections.abc import Sequence

from shamela_rag.chunking.tokens import HeuristicTokenCounter, TokenCounter


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dims(self) -> int: ...

    @property
    @abstractmethod
    def tokenizer(self) -> TokenCounter: ...

    @property
    def query_instruction(self) -> str | None:
        return None

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class InMemoryEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        dims: int = 8,
        tokenizer: TokenCounter | None = None,
        query_instruction: str | None = None,
    ) -> None:
        if dims <= 0:
            raise ValueError(f"dims must be positive, got {dims}")
        self._dims = dims
        self._tokenizer: TokenCounter = (
            tokenizer if tokenizer is not None else HeuristicTokenCounter()
        )
        self._query_instruction = query_instruction

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def tokenizer(self) -> TokenCounter:
        return self._tokenizer

    @property
    def query_instruction(self) -> str | None:
        return self._query_instruction

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if self._query_instruction is not None:
            return self._vector(f"{self._query_instruction}\n{text}")
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw: list[float] = []
        block = digest
        while len(raw) < self._dims:
            for i in range(0, len(block), 4):
                if len(raw) >= self._dims:
                    break
                (unsigned,) = struct.unpack_from(">I", block, i)
                raw.append((unsigned / 0xFFFFFFFF) * 2.0 - 1.0)
            block = hashlib.sha256(block).digest()

        norm = math.sqrt(sum(v * v for v in raw))
        if norm == 0.0:
            return [0.0] * self._dims
        return [v / norm for v in raw]
