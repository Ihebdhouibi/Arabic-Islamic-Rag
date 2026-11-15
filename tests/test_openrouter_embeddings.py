"""Unit tests for OpenRouter embedding provider (no live network)."""

from __future__ import annotations

import functools
import io
import json
from typing import Any

import pytest

from shamela_rag.embeddings.openrouter import (
    OPENROUTER_BGE_M3,
    OPENROUTER_QWEN3_EMBEDDING_8B,
    OpenRouterEmbeddingProvider,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], *, status: int = 200) -> None:
        self._raw = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_a: object) -> None:
        return None


def test_openrouter_embed_documents_posts_batch_and_orders_by_index() -> None:
    seen: dict[str, Any] = {}

    def opener(request: object, timeout: float = 0) -> _FakeResponse:
        seen["timeout"] = timeout
        assert hasattr(request, "full_url")
        assert request.full_url.endswith("/embeddings")  # type: ignore[attr-defined]
        headers = dict(request.header_items())  # type: ignore[attr-defined]
        assert headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        seen["body"] = body
        return _FakeResponse(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }
        )

    provider = OpenRouterEmbeddingProvider(
        OPENROUTER_BGE_M3,
        api_key="test-key",
        dims=2,
        batch_size=8,
        opener=opener,
    )
    vectors = provider.embed_documents(["a", "b"])
    assert seen["body"]["model"] == OPENROUTER_BGE_M3
    assert seen["body"]["input"] == ["a", "b"]
    assert seen["body"]["provider"] == {
        "order": ["nebius", "deepinfra"],
        "allow_fallbacks": False,
    }
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert provider.dims == 2
    assert provider.query_instruction is None


def test_openrouter_embed_query_applies_qwen_instruction() -> None:
    seen: dict[str, Any] = {}

    def opener(request: object, timeout: float = 0) -> _FakeResponse:
        seen["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        return _FakeResponse({"data": [{"index": 0, "embedding": [1.0] * 4}]})

    provider = OpenRouterEmbeddingProvider(
        OPENROUTER_QWEN3_EMBEDDING_8B,
        api_key="k",
        dims=4,
        opener=opener,
    )
    vec = provider.embed_query("hello")
    assert len(vec) == 4
    assert seen["body"]["input"][0].startswith("Instruct:")
    assert "Query:hello" in seen["body"]["input"][0]
    assert provider.query_instruction is not None
    assert seen["body"]["dimensions"] == 4


def test_openrouter_bge_does_not_send_dimensions() -> None:
    seen: dict[str, Any] = {}

    def opener(request: object, timeout: float = 0) -> _FakeResponse:
        seen["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        return _FakeResponse({"data": [{"index": 0, "embedding": [0.0, 1.0]}]})

    provider = OpenRouterEmbeddingProvider(OPENROUTER_BGE_M3, api_key="k", dims=2, opener=opener)
    provider.embed_documents(["x"])
    assert "dimensions" not in seen["body"]
    with pytest.raises(ValueError, match="api_key"):
        OpenRouterEmbeddingProvider(OPENROUTER_BGE_M3, api_key=" ")
    with pytest.raises(ValueError, match="model"):
        OpenRouterEmbeddingProvider("  ", api_key="k", dims=2)
    with pytest.raises(ValueError, match="dims"):
        OpenRouterEmbeddingProvider("unknown/model", api_key="k")


def test_openrouter_surfaces_http_error() -> None:
    import urllib.error

    def opener(request: object, timeout: float = 0) -> Any:
        raise urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/embeddings",
            code=401,
            msg="Unauthorized",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(json.dumps({"error": {"message": "bad key"}}).encode()),
        )

    provider = OpenRouterEmbeddingProvider(OPENROUTER_BGE_M3, api_key="k", dims=2, opener=opener)
    with pytest.raises(ConnectionError, match="bad key"):
        provider.embed_documents(["x"])


def test_openrouter_splits_batch_when_context_length_is_exceeded() -> None:
    import urllib.error

    calls: list[list[str]] = []

    def opener(request: object, timeout: float = 0) -> _FakeResponse:
        body = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        texts = list(body["input"])
        calls.append(texts)
        if len(texts) > 1:
            message = (
                "This model's maximum context length is 40960 tokens. input_tokens value=40961"
            )
            raise urllib.error.HTTPError(
                url="https://openrouter.ai/api/v1/embeddings",
                code=400,
                msg="Bad Request",
                hdrs=None,  # type: ignore[arg-type]
                fp=io.BytesIO(json.dumps({"error": {"message": message}}).encode()),
            )
        return _FakeResponse({"data": [{"index": 0, "embedding": [float(len(texts[0])), 0.0]}]})

    provider = OpenRouterEmbeddingProvider(
        OPENROUTER_QWEN3_EMBEDDING_8B,
        api_key="k",
        dims=2,
        batch_size=2,
        opener=opener,
    )
    assert provider.embed_documents(["aa", "bbb"]) == [[2.0, 0.0], [3.0, 0.0]]
    assert calls[0] == ["aa", "bbb"]
    assert calls[1:] == [["aa"], ["bbb"]]


def test_openrouter_truncates_single_input_on_http_422_char_limit() -> None:
    import urllib.error

    calls: list[list[str]] = []

    def opener(request: object, timeout: float = 0) -> _FakeResponse:
        body = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        texts = list(body["input"])
        calls.append(texts)
        if any(len(t) > 8 for t in texts):
            message = (
                "Value error, The input sequence should have less than 131072 "
                f"characters. Input length: {len(texts[0])}"
            )
            raise urllib.error.HTTPError(
                url="https://openrouter.ai/api/v1/embeddings",
                code=422,
                msg="Unprocessable Entity",
                hdrs=None,  # type: ignore[arg-type]
                fp=io.BytesIO(json.dumps({"error": {"message": message}}).encode()),
            )
        return _FakeResponse({"data": [{"index": 0, "embedding": [float(len(texts[0])), 0.0]}]})

    provider = OpenRouterEmbeddingProvider(
        OPENROUTER_QWEN3_EMBEDDING_8B,
        api_key="k",
        dims=2,
        batch_size=1,
        opener=opener,
    )
    assert provider.embed_documents(["abcdefghijklmnop"]) == [[8.0, 0.0]]
    assert calls[0] == ["abcdefghijklmnop"]
    assert all(len(t) <= 8 for batch in calls[1:] for t in batch)


def test_build_embedder_openrouter_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from shamela_rag import factory
    from shamela_rag.config import Settings

    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: Settings(embedding_backend="openrouter", embedding_api_key=""),
    )
    with pytest.raises(ValueError, match="SHAMELA_EMBEDDING_API_KEY"):
        factory.build_embedder("bge-m3")


def test_build_embedder_openrouter_maps_model_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from shamela_rag import factory
    from shamela_rag.config import Settings
    from shamela_rag.embeddings.openrouter import OpenRouterEmbeddingProvider

    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: Settings(
            embedding_backend="openrouter",
            embedding_api_key="secret",
            embedding_api_base_url="https://openrouter.ai/api/v1",
            qdrant_dense_dim=1024,
        ),
    )
    qwen = factory.build_embedder("qwen3")
    bge = factory.build_embedder("bge-m3")
    assert isinstance(qwen, OpenRouterEmbeddingProvider)
    assert isinstance(bge, OpenRouterEmbeddingProvider)
    assert qwen.model == OPENROUTER_QWEN3_EMBEDDING_8B
    assert bge.model == OPENROUTER_BGE_M3
    assert qwen.dims == 1024
    assert bge.dims == 1024


def test_cli_build_embedder_routes_through_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: ingest/build-bm25 must respect SHAMELA_EMBEDDING_BACKEND, not just /ask."""
    from shamela_rag import cli, factory
    from shamela_rag.config import Settings
    from shamela_rag.embeddings.openrouter import OpenRouterEmbeddingProvider

    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: Settings(embedding_backend="openrouter", embedding_api_key="secret"),
    )
    embedder = cli._build_embedder("qwen3")
    assert isinstance(embedder, OpenRouterEmbeddingProvider)
    assert embedder.model == OPENROUTER_QWEN3_EMBEDDING_8B


def test_openrouter_rejects_non_positive_max_concurrency() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        OpenRouterEmbeddingProvider(OPENROUTER_BGE_M3, api_key="k", dims=2, max_concurrency=0)


def test_openrouter_concurrent_batches_preserve_input_order() -> None:
    import threading
    import time

    started = threading.Barrier(2)
    call_order: list[str] = []
    lock = threading.Lock()

    def opener(request: object, timeout: float = 0) -> _FakeResponse:
        body = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        first = body["input"][0]
        with lock:
            call_order.append(first)
        started.wait(timeout=5)
        if first == "a":
            time.sleep(0.05)
        payload = {
            "data": [
                {"index": i, "embedding": [float(ord(text[0])), 0.0]}
                for i, text in enumerate(body["input"])
            ]
        }
        return _FakeResponse(payload)

    provider = OpenRouterEmbeddingProvider(
        OPENROUTER_BGE_M3,
        api_key="k",
        dims=2,
        batch_size=2,
        max_concurrency=2,
        opener=opener,
    )
    vectors = provider.embed_documents(["a", "b", "c", "d"])
    assert vectors == [
        [float(ord("a")), 0.0],
        [float(ord("b")), 0.0],
        [float(ord("c")), 0.0],
        [float(ord("d")), 0.0],
    ]
    assert set(call_order) == {"a", "c"}


def test_openrouter_shared_backoff_under_concurrent_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    import urllib.error

    mono = [0.0]
    attempt_by_batch: dict[str, int] = {}
    attempt_lock = threading.Lock()
    barrier = threading.Barrier(2)
    second_wave_started_at: list[float] = []

    def fake_monotonic() -> float:
        return mono[0]

    def fake_sleep(seconds: float) -> None:
        mono[0] += seconds

    monkeypatch.setattr("shamela_rag.embeddings.openrouter.time.monotonic", fake_monotonic)
    monkeypatch.setattr("shamela_rag.embeddings.openrouter.time.sleep", fake_sleep)

    def opener(request: object, timeout: float = 0) -> _FakeResponse:
        body = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        first = body["input"][0]
        with attempt_lock:
            attempt_by_batch[first] = attempt_by_batch.get(first, 0) + 1
            attempt = attempt_by_batch[first]
        if attempt == 1:
            barrier.wait(timeout=5)
            raise urllib.error.HTTPError(
                url="https://openrouter.ai/api/v1/embeddings",
                code=429,
                msg="Too Many Requests",
                hdrs=None,  # type: ignore[arg-type]
                fp=io.BytesIO(json.dumps({"error": {"message": "rate limited"}}).encode()),
            )
        second_wave_started_at.append(mono[0])
        return _FakeResponse(
            {"data": [{"index": i, "embedding": [1.0, 0.0]} for i, _ in enumerate(body["input"])]}
        )

    provider = OpenRouterEmbeddingProvider(
        OPENROUTER_BGE_M3,
        api_key="k",
        dims=2,
        batch_size=1,
        max_concurrency=2,
        retry_backoff=1.0,
        opener=opener,
    )
    vectors = provider.embed_documents(["a", "b"])
    assert vectors == [[1.0, 0.0], [1.0, 0.0]]
    assert attempt_by_batch == {"a": 2, "b": 2}
    assert second_wave_started_at
    assert min(second_wave_started_at) >= 1.0 - 1e-9


def _capture_body(seen: dict[str, Any]) -> Any:
    def opener(request: object, timeout: float = 0) -> _FakeResponse:
        seen["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        return _FakeResponse({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    return opener


def test_provider_routing_is_sent_only_to_openrouter() -> None:
    openrouter: dict[str, Any] = {}
    other: dict[str, Any] = {}
    OpenRouterEmbeddingProvider(
        OPENROUTER_QWEN3_EMBEDDING_8B, api_key="k", dims=2, opener=_capture_body(openrouter)
    ).embed_documents(["x"])
    OpenRouterEmbeddingProvider(
        OPENROUTER_QWEN3_EMBEDDING_8B,
        api_key="k",
        dims=2,
        base_url="https://10.0.0.1/v2/infer/abc/v1",
        opener=_capture_body(other),
    ).embed_documents(["x"])
    assert "provider" in openrouter["body"]
    assert "provider" not in other["body"]
    assert other["body"]["dimensions"] == 2


@pytest.mark.parametrize(
    "error",
    [ConnectionResetError("reset by peer"), TimeoutError("The read operation timed out")],
    ids=["connection_reset", "read_timeout"],
)
def test_connection_reset_and_read_timeout_are_retried(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr("shamela_rag.embeddings.openrouter.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def opener(request: object, timeout: float = 0) -> _FakeResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            raise error
        return _FakeResponse({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    provider = OpenRouterEmbeddingProvider(
        OPENROUTER_BGE_M3, api_key="k", dims=2, retry_backoff=0.01, opener=opener
    )
    assert provider.embed_documents(["x"]) == [[1.0, 0.0]]
    assert calls["n"] == 2


def test_retries_stop_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shamela_rag.embeddings.openrouter.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def opener(request: object, timeout: float = 0) -> _FakeResponse:
        calls["n"] += 1
        raise ConnectionResetError("reset by peer")

    provider = OpenRouterEmbeddingProvider(
        OPENROUTER_BGE_M3, api_key="k", dims=2, max_retries=6, retry_backoff=0.01, opener=opener
    )
    with pytest.raises(ConnectionError, match="reset by peer"):
        provider.embed_documents(["x"])
    assert calls["n"] == 6


def test_verify_tls_false_passes_an_unverified_context(monkeypatch: pytest.MonkeyPatch) -> None:
    import ssl

    seen: dict[str, Any] = {}

    def fake_urlopen(request: object, timeout: float = 0, context: Any = None) -> _FakeResponse:
        seen["context"] = context
        seen["timeout"] = timeout
        return _FakeResponse({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    monkeypatch.setattr("shamela_rag.embeddings.openrouter.urllib.request.urlopen", fake_urlopen)
    OpenRouterEmbeddingProvider(
        OPENROUTER_BGE_M3, api_key="k", dims=2, timeout_seconds=60, verify_tls=False
    ).embed_documents(["x"])
    assert seen["context"].verify_mode == ssl.CERT_NONE
    assert seen["context"].check_hostname is False
    assert seen["timeout"] == 60

    seen.clear()
    OpenRouterEmbeddingProvider(OPENROUTER_BGE_M3, api_key="k", dims=2).embed_documents(["x"])
    assert seen["context"] is None


def test_build_embedder_passes_timeout_retry_and_tls_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shamela_rag import factory
    from shamela_rag.config import Settings

    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: Settings(
            embedding_backend="openrouter",
            embedding_api_key="secret",
            embedding_api_timeout_seconds=60,
            embedding_api_max_retries=6,
            embedding_api_retry_backoff_seconds=1,
            embedding_api_verify_tls=False,
        ),
    )
    embedder = factory.build_embedder("qwen3")
    assert embedder._timeout_seconds == 60
    assert embedder._max_retries == 6
    assert embedder._retry_backoff == 1
    assert isinstance(embedder._opener, functools.partial)


@pytest.mark.parametrize(
    "model,configured,allowed",
    [
        ("qwen3", "qwen/qwen3-embedding-8b", True),
        ("bge-m3", "BAAI/bge-m3", True),
        ("bge-m3", "qwen/qwen3-embedding-8b", False),
        ("qwen3", "BAAI/bge-m3", False),
    ],
)
def test_ingest_refuses_a_model_other_than_the_configured_one(
    model: str, configured: str, allowed: bool
) -> None:
    from shamela_rag.cli import require_configured_model

    if allowed:
        require_configured_model(model, configured)
    else:
        with pytest.raises(ValueError, match="SHAMELA_DENSE_EMBEDDING_MODEL"):
            require_configured_model(model, configured)
