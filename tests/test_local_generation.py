from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from shamela_rag.generation.local import (
    LlamaCppGenerationProvider,
    OllamaGenerationProvider,
    OpenAICompatibleGenerationProvider,
)
from shamela_rag.generation.provider import InMemoryGenerationProvider


class _FakeLlama:
    def __init__(self, text: str = "local answer") -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
            }
        )
        if stream:
            return iter(
                [
                    {"choices": [{"delta": {"content": self.text[:5]}}]},
                    {"choices": [{"delta": {"content": self.text[5:]}}]},
                ]
            )
        return {"choices": [{"message": {"content": self.text}}]}


class _FakeOllamaResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self._joined = b"".join(chunks)

    def __enter__(self) -> _FakeOllamaResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._joined

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._chunks)


def test_llama_cpp_generate_uses_chat_completion(tmp_path: Path) -> None:
    fake = _FakeLlama("cited answer")
    provider = LlamaCppGenerationProvider(tmp_path / "x.gguf", llm=fake, max_tokens=64)
    assert provider.generate("prompt text") == "cited answer"
    assert fake.calls[0]["messages"] == [{"role": "user", "content": "prompt text"}]
    assert fake.calls[0]["max_tokens"] == 64
    assert fake.calls[0]["stream"] is False


def test_llama_cpp_generate_stream_yields_deltas(tmp_path: Path) -> None:
    fake = _FakeLlama("abcdefghij")
    provider = LlamaCppGenerationProvider(tmp_path / "x.gguf", llm=fake)
    assert list(provider.generate_stream("p")) == ["abcde", "fghij"]
    assert fake.calls[0]["stream"] is True


def test_llama_cpp_respects_per_call_max_tokens(tmp_path: Path) -> None:
    fake = _FakeLlama("ok")
    provider = LlamaCppGenerationProvider(tmp_path / "x.gguf", llm=fake, max_tokens=512)
    provider.generate("p", max_tokens=12)
    assert fake.calls[0]["max_tokens"] == 12


def test_llama_cpp_missing_file_without_injected_llm(tmp_path: Path) -> None:
    missing = tmp_path / "absent.gguf"
    with pytest.raises(FileNotFoundError, match="GGUF"):
        LlamaCppGenerationProvider(missing)


def test_llama_cpp_import_error_mentions_optional_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise ImportError(
            'LlamaCppGenerationProvider requires optional deps: pip install "shamela-rag[llm]"'
        )

    monkeypatch.setattr("shamela_rag.generation.local._load_llama", _boom)
    with pytest.raises(ImportError, match=r"shamela-rag\[llm\]"):
        LlamaCppGenerationProvider(gguf)


def test_llama_cpp_rejects_bad_params(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        LlamaCppGenerationProvider(tmp_path / "x.gguf", llm=_FakeLlama(), max_tokens=0)
    with pytest.raises(ValueError, match="temperature"):
        LlamaCppGenerationProvider(tmp_path / "x.gguf", llm=_FakeLlama(), temperature=-0.1)


def test_ollama_generate_posts_chat_payload() -> None:
    captured: dict[str, Any] = {}

    def opener(request: urllib.request.Request, timeout: float) -> _FakeOllamaResponse:
        captured["timeout"] = timeout
        captured["url"] = request.full_url
        assert request.data is not None
        captured["body"] = json.loads(request.data.decode("utf-8"))
        body = json.dumps({"message": {"content": "from ollama"}}).encode()
        return _FakeOllamaResponse([body])

    provider = OllamaGenerationProvider("qwen2.5:3b", opener=opener, max_tokens=40)
    assert provider.generate("ask me") == "from ollama"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["body"]["model"] == "qwen2.5:3b"
    assert captured["body"]["stream"] is False
    assert captured["body"]["options"]["num_predict"] == 40
    assert captured["body"]["messages"][0]["content"] == "ask me"


def test_ollama_generate_stream_reads_ndjson() -> None:
    lines = [
        json.dumps({"message": {"content": "hel"}}) + "\n",
        json.dumps({"message": {"content": "lo"}, "done": True}) + "\n",
    ]

    def opener(request: urllib.request.Request, timeout: float) -> _FakeOllamaResponse:
        payload = json.loads(request.data.decode("utf-8")) if request.data else {}
        assert payload["stream"] is True
        return _FakeOllamaResponse([line.encode() for line in lines])

    provider = OllamaGenerationProvider("qwen2.5:3b", opener=opener)
    assert "".join(provider.generate_stream("p")) == "hello"


def test_ollama_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model name"):
        OllamaGenerationProvider("  ")


def test_openai_compatible_generate_posts_bearer_auth_and_body() -> None:
    captured: dict[str, Any] = {}

    def opener(request: urllib.request.Request, timeout: float) -> _FakeOllamaResponse:
        captured["timeout"] = timeout
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        body = json.dumps({"choices": [{"message": {"content": "from together"}}]}).encode()
        return _FakeOllamaResponse([body])

    provider = OpenAICompatibleGenerationProvider(
        "Qwen/Qwen3.5-9B",
        api_key="secret-key",
        base_url="https://api.together.xyz/v1",
        opener=opener,
        max_tokens=40,
    )
    assert provider.generate("ask me") == "from together"
    assert captured["url"] == "https://api.together.xyz/v1/chat/completions"
    assert captured["auth"] == "Bearer secret-key"
    assert captured["body"]["model"] == "Qwen/Qwen3.5-9B"
    assert captured["body"]["stream"] is False
    assert captured["body"]["max_tokens"] == 40
    assert captured["body"]["messages"][0]["content"] == "ask me"


def test_openai_compatible_generate_stream_reads_sse() -> None:
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": "hel"}}]}) + "\n",
        "\n",
        "data: " + json.dumps({"choices": [{"delta": {"content": "lo"}}]}) + "\n",
        "data: [DONE]\n",
    ]

    def opener(request: urllib.request.Request, timeout: float) -> _FakeOllamaResponse:
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["stream"] is True
        return _FakeOllamaResponse([line.encode() for line in lines])

    provider = OpenAICompatibleGenerationProvider("Qwen/Qwen3.5-9B", api_key="k", opener=opener)
    assert "".join(provider.generate_stream("p")) == "hello"


def test_openai_compatible_surfaces_api_error_message() -> None:
    error_body = json.dumps({"error": {"message": "model_not_available"}}).encode()

    def opener(request: urllib.request.Request, timeout: float) -> Any:
        raise urllib.error.HTTPError(
            request.full_url, 400, "Bad Request", None, io.BytesIO(error_body)
        )

    provider = OpenAICompatibleGenerationProvider("Qwen/Qwen3-8B", api_key="k", opener=opener)
    with pytest.raises(ConnectionError, match="model_not_available"):
        provider.generate("ask me")


def test_openai_compatible_rejects_empty_model_or_key() -> None:
    with pytest.raises(ValueError, match="model name"):
        OpenAICompatibleGenerationProvider("  ", api_key="k")
    with pytest.raises(ValueError, match="api_key"):
        OpenAICompatibleGenerationProvider("m", api_key=" ")
    with pytest.raises(ValueError, match="timeout_seconds"):
        OpenAICompatibleGenerationProvider("m", api_key="k", timeout_seconds=0)


def test_openai_compatible_error_message_falls_back_on_non_json_body() -> None:
    def opener(request: urllib.request.Request, timeout: float) -> Any:
        raise urllib.error.HTTPError(
            request.full_url, 500, "Internal Error", None, io.BytesIO(b"not json")
        )

    provider = OpenAICompatibleGenerationProvider("m", api_key="k", opener=opener)
    with pytest.raises(ConnectionError, match="HTTP 500"):
        provider.generate("ask me")


def test_openai_compatible_wraps_connection_errors() -> None:
    def opener(request: urllib.request.Request, timeout: float) -> Any:
        raise urllib.error.URLError("connection refused")

    provider = OpenAICompatibleGenerationProvider("m", api_key="k", opener=opener)
    with pytest.raises(ConnectionError, match="request failed"):
        provider.generate("ask me")
    with pytest.raises(ConnectionError, match="request failed"):
        list(provider.generate_stream("ask me"))


def test_build_generation_provider_memory_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from shamela_rag.config import Settings
    from shamela_rag.factory import build_generation_provider

    monkeypatch.setattr(
        "shamela_rag.factory.get_settings",
        lambda: Settings(_env_file=None, llm_backend="memory"),
    )
    provider = build_generation_provider()
    assert isinstance(provider, InMemoryGenerationProvider)


def test_build_generation_provider_llamacpp_requires_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shamela_rag.config import Settings
    from shamela_rag.factory import build_generation_provider

    monkeypatch.setattr(
        "shamela_rag.factory.get_settings",
        lambda: Settings(_env_file=None, llm_backend="llamacpp", llm_gguf_path=None),
    )
    with pytest.raises(ValueError, match="SHAMELA_LLM_GGUF_PATH"):
        build_generation_provider()


def test_build_generation_provider_ollama_requires_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shamela_rag.config import Settings
    from shamela_rag.factory import build_generation_provider

    monkeypatch.setattr(
        "shamela_rag.factory.get_settings",
        lambda: Settings(_env_file=None, llm_backend="ollama", llm_ollama_model=""),
    )
    with pytest.raises(ValueError, match="SHAMELA_LLM_OLLAMA_MODEL"):
        build_generation_provider()


def test_build_generation_provider_ollama_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shamela_rag.config import Settings
    from shamela_rag.factory import build_generation_provider

    monkeypatch.setattr(
        "shamela_rag.factory.get_settings",
        lambda: Settings(_env_file=None, llm_backend="ollama", llm_ollama_model="qwen2.5:3b"),
    )
    provider = build_generation_provider()
    assert isinstance(provider, OllamaGenerationProvider)


def test_build_generation_provider_openai_compatible_requires_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shamela_rag.config import Settings
    from shamela_rag.factory import build_generation_provider

    monkeypatch.setattr(
        "shamela_rag.factory.get_settings",
        lambda: Settings(
            _env_file=None,
            llm_backend="openai_compatible",
            llm_api_key="",
            llm_api_model="Qwen/Qwen3.5-9B",
        ),
    )
    with pytest.raises(ValueError, match="SHAMELA_LLM_API_KEY"):
        build_generation_provider()


def test_build_generation_provider_openai_compatible_requires_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shamela_rag.config import Settings
    from shamela_rag.factory import build_generation_provider

    monkeypatch.setattr(
        "shamela_rag.factory.get_settings",
        lambda: Settings(
            _env_file=None,
            llm_backend="openai_compatible",
            llm_api_key="secret",
            llm_api_model="",
        ),
    )
    with pytest.raises(ValueError, match="SHAMELA_LLM_API_MODEL"):
        build_generation_provider()


def test_build_generation_provider_openai_compatible_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shamela_rag.config import Settings
    from shamela_rag.factory import build_generation_provider

    monkeypatch.setattr(
        "shamela_rag.factory.get_settings",
        lambda: Settings(
            _env_file=None,
            llm_backend="openai_compatible",
            llm_api_key="secret",
            llm_api_model="Qwen/Qwen3.5-9B",
        ),
    )
    provider = build_generation_provider()
    assert isinstance(provider, OpenAICompatibleGenerationProvider)


def test_build_general_qa_service_uses_configured_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shamela_rag.embeddings.bm25 import Bm25Encoder
    from shamela_rag.embeddings.provider import InMemoryEmbeddingProvider
    from shamela_rag.factory import build_general_qa_service
    from shamela_rag.retrieval.rerank import LexicalOverlapReranker
    from shamela_rag.retrieval.translate import InMemoryTranslator

    sentinel = InMemoryGenerationProvider(prefix="LOCAL:")
    called: list[bool] = []

    def _provider() -> InMemoryGenerationProvider:
        called.append(True)
        return sentinel

    monkeypatch.setattr("shamela_rag.factory.build_generation_provider", _provider)
    encoder = Bm25Encoder().fit(["نص أول للاختبار", "نص ثانٍ مختلف"])
    build_general_qa_service(
        embedder=InMemoryEmbeddingProvider(dims=8),
        reranker=LexicalOverlapReranker(),
        sparse_encoder=encoder,
        translator=InMemoryTranslator(),
    )
    assert called == [True]
