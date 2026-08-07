"""Qwen3-Embedding-8B backend for ``EmbeddingProvider``.

Uses sentence-transformers. Query text is formatted with Qwen's official
``Instruct: …\\nQuery:…`` template (documents are embedded unchanged). Optional
``[qwen]`` extra installs the heavy deps; integration tests skip when weights
are absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from shamela_rag.chunking.tokens import TokenCounter
from shamela_rag.embeddings.provider import EmbeddingProvider

QWEN3_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
DEFAULT_EMBEDDING_DIMS = 4096
DEFAULT_TASK_DESCRIPTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def format_qwen_query(task_description: str, query: str) -> str:
    """Official Qwen3 embedding query formatting (instruction on queries only)."""
    return f"Instruct: {task_description}\nQuery:{query}"


def _load_sentence_transformer(
    model_id: str,
    *,
    device: str | None,
    truncate_dim: int | None,
) -> Any:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            'Qwen3EmbeddingProvider requires optional deps: pip install "shamela-rag[qwen]"'
        ) from exc

    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    if truncate_dim is not None:
        kwargs["truncate_dim"] = truncate_dim
    return SentenceTransformer(model_id, **kwargs)


class _HFTokenizerCounter:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


class Qwen3EmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        model_id: str = QWEN3_EMBEDDING_MODEL_ID,
        device: str | None = None,
        batch_size: int = 32,
        task_description: str = DEFAULT_TASK_DESCRIPTION,
        dims: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if dims is not None and dims <= 0:
            raise ValueError(f"dims must be positive, got {dims}")

        self._model = _load_sentence_transformer(model_id, device=device, truncate_dim=dims)
        self._batch_size = batch_size
        self._task_description = task_description
        self._tokenizer_counter: TokenCounter = _HFTokenizerCounter(self._model.tokenizer)
        reported = self._model.get_sentence_embedding_dimension()
        self._dims = int(reported) if reported is not None else DEFAULT_EMBEDDING_DIMS

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def tokenizer(self) -> TokenCounter:
        return self._tokenizer_counter

    @property
    def query_instruction(self) -> str | None:
        return f"Instruct: {self._task_description}\nQuery:"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._as_list_vectors(vectors)

    def embed_query(self, text: str) -> list[float]:
        formatted = format_qwen_query(self._task_description, text)
        vectors = self._model.encode(
            [formatted],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._as_list_vectors(vectors)[0]

    def _as_list_vectors(self, vectors: Any) -> list[list[float]]:
        rows = vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)
        out: list[list[float]] = []
        for row in rows:
            vector = [float(x) for x in row]
            if len(vector) != self._dims:
                raise ValueError(
                    f"embedding dims mismatch: got {len(vector)}, expected {self._dims}"
                )
            out.append(vector)
        return out
