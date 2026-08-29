"""OpenRouter OpenAI-compatible remote embedding backend."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from shamela_rag.chunking.tokens import HeuristicTokenCounter, TokenCounter
from shamela_rag.embeddings.provider import EmbeddingProvider
from shamela_rag.embeddings.qwen import DEFAULT_TASK_DESCRIPTION, format_qwen_query

OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_QWEN3_EMBEDDING_8B = "qwen/qwen3-embedding-8b"
OPENROUTER_BGE_M3 = "baai/bge-m3"

# Known output dimensions for supported OpenRouter embedding models.
_MODEL_DIMS: dict[str, int] = {
    OPENROUTER_QWEN3_EMBEDDING_8B: 4096,
    OPENROUTER_BGE_M3: 1024,
    "qwen/qwen3-embedding-4b": 2560,
}


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """Dense embeddings via OpenRouter's OpenAI-compatible embeddings API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = OPENROUTER_API_BASE_URL,
        dims: int | None = None,
        batch_size: int = 32,
        timeout_seconds: float = 120.0,
        max_retries: int = 8,
        retry_backoff: float = 2.0,
        apply_qwen_query_instruction: bool | None = None,
        task_description: str = DEFAULT_TASK_DESCRIPTION,
        opener: Any | None = None,
    ) -> None:
        name = model.strip()
        if not name:
            raise ValueError("model name must be non-empty")
        key = api_key.strip()
        if not key:
            raise ValueError("api_key must be non-empty")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        if max_retries <= 0:
            raise ValueError(f"max_retries must be positive, got {max_retries}")
        if retry_backoff <= 0:
            raise ValueError(f"retry_backoff must be positive, got {retry_backoff}")

        resolved_dims = dims if dims is not None else _MODEL_DIMS.get(name)
        if resolved_dims is None or resolved_dims <= 0:
            raise ValueError(
                f"dims required for model {name!r} (pass dims=... or use a known model id)"
            )

        self._model = name
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._dims = resolved_dims
        self._batch_size = batch_size
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._task_description = task_description
        self._apply_qwen_query = (
            apply_qwen_query_instruction
            if apply_qwen_query_instruction is not None
            else name.startswith("qwen/qwen3-embedding")
        )
        self._tokenizer: TokenCounter = HeuristicTokenCounter()
        self._opener = opener or urllib.request.urlopen

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def tokenizer(self) -> TokenCounter:
        return self._tokenizer

    @property
    def query_instruction(self) -> str | None:
        if not self._apply_qwen_query:
            return None
        return f"Instruct: {self._task_description}\nQuery:"

    @property
    def model(self) -> str:
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        batch: list[str] = []
        for text in texts:
            batch.append(text)
            if len(batch) >= self._batch_size:
                out.extend(self._embed_batch(batch))
                batch = []
        if batch:
            out.extend(self._embed_batch(batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        payload = text
        if self._apply_qwen_query:
            payload = format_qwen_query(self._task_description, text)
        return self._embed_batch([payload])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps(
            {"model": self._model, "input": texts, "encoding_format": "float"}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/embeddings",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Ihebdhouibi/Arabic-Islamic-Rag",
                "X-Title": "shamela-rag",
            },
        )
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                with self._opener(request, timeout=self._timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                return self._parse_embeddings_payload(raw, expected=len(texts))
            except urllib.error.HTTPError as exc:
                message = self._api_error_message(exc)
                last_error = ConnectionError(message)
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < self._max_retries:
                    delay = min(60.0, self._retry_backoff * (2 ** (attempt - 1)))
                    time.sleep(delay)
                    continue
                raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = ConnectionError(f"{self._base_url} request failed: {exc}")
                if attempt < self._max_retries:
                    delay = min(60.0, self._retry_backoff * (2 ** (attempt - 1)))
                    time.sleep(delay)
                    continue
                raise last_error from exc
        assert last_error is not None
        raise last_error

    def _parse_embeddings_payload(self, raw: str, *, expected: int) -> list[list[float]]:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("embeddings response must be a JSON object")
        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) != expected:
            raise ValueError(
                f"expected {expected} embedding rows, got "
                f"{len(rows) if isinstance(rows, list) else type(rows).__name__}"
            )
        ordered = sorted(
            rows,
            key=lambda row: int(row.get("index", 0)) if isinstance(row, dict) else 0,
        )
        vectors: list[list[float]] = []
        for row in ordered:
            if not isinstance(row, dict):
                raise ValueError("embedding row must be an object")
            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("embedding row missing embedding list")
            vector = [float(x) for x in embedding]
            if len(vector) != self._dims:
                raise ValueError(
                    f"embedding dims mismatch: got {len(vector)}, expected {self._dims}"
                )
            vectors.append(vector)
        return vectors

    def _api_error_message(self, exc: urllib.error.HTTPError) -> str:
        detail = ""
        try:
            body = exc.read().decode("utf-8")
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                err = parsed.get("error")
                if isinstance(err, dict):
                    detail = str(err.get("message") or err)
                elif err is not None:
                    detail = str(err)
                else:
                    detail = body[:300]
            else:
                detail = body[:300]
        except Exception:  # noqa: BLE001
            detail = ""
        if detail:
            return f"OpenRouter embeddings HTTP {exc.code}: {detail}"
        return f"OpenRouter embeddings HTTP {exc.code}"
