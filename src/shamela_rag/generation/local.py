"""Local ``GenerationProvider`` backends: llama.cpp GGUF and Ollama on localhost."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from shamela_rag.generation.provider import GenerationProvider

_MISSING_LLAMA_CPP = (
    'LlamaCppGenerationProvider requires optional deps: pip install "shamela-rag[llm]"'
)


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_non_negative_float(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _chat_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _resolve_max_tokens(requested: int | None, default: int) -> int:
    if requested is None:
        return default
    _require_positive("max_tokens", requested)
    return requested


def _load_llama(
    model_path: Path,
    *,
    n_ctx: int,
    n_threads: int | None,
    n_gpu_layers: int,
) -> Any:
    try:
        from llama_cpp import Llama  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_MISSING_LLAMA_CPP) from exc

    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "verbose": False,
    }
    if n_threads is not None:
        kwargs["n_threads"] = n_threads
    return Llama(**kwargs)


def _completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = choice.get("text")
    return text if isinstance(text, str) else ""


def _delta_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
    text = choice.get("text")
    return text if isinstance(text, str) else ""


class LlamaCppGenerationProvider(GenerationProvider):
    """GGUF weights via ``llama-cpp-python`` (optional ``[llm]`` extra)."""

    def __init__(
        self,
        model_path: Path,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        llm: Any | None = None,
    ) -> None:
        _require_positive("max_tokens", max_tokens)
        _require_non_negative_float("temperature", temperature)
        _require_positive("n_ctx", n_ctx)
        if n_gpu_layers < 0:
            raise ValueError(f"n_gpu_layers must be >= 0, got {n_gpu_layers}")
        if n_threads is not None and n_threads <= 0:
            raise ValueError(f"n_threads must be positive, got {n_threads}")

        path = Path(model_path)
        if llm is None and not path.is_file():
            raise FileNotFoundError(f"GGUF model not found: {path}")

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._llm = (
            llm
            if llm is not None
            else _load_llama(path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers)
        )

    def generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
        payload = self._llm.create_chat_completion(
            messages=_chat_messages(prompt),
            max_tokens=_resolve_max_tokens(max_tokens, self._max_tokens),
            temperature=self._temperature,
            stream=False,
        )
        if not isinstance(payload, dict):
            return ""
        return _completion_text(payload)

    def generate_stream(self, prompt: str, *, max_tokens: int | None = None) -> Iterator[str]:
        stream = self._llm.create_chat_completion(
            messages=_chat_messages(prompt),
            max_tokens=_resolve_max_tokens(max_tokens, self._max_tokens),
            temperature=self._temperature,
            stream=True,
        )
        for chunk in stream:
            if not isinstance(chunk, dict):
                continue
            piece = _delta_text(chunk)
            if piece:
                yield piece


class OllamaGenerationProvider(GenerationProvider):
    """Local Ollama HTTP daemon (stdlib only, no API keys)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        max_tokens: int = 512,
        temperature: float = 0.1,
        timeout_seconds: float = 120.0,
        opener: Any | None = None,
    ) -> None:
        name = model.strip()
        if not name:
            raise ValueError("Ollama model name must be non-empty")
        _require_positive("max_tokens", max_tokens)
        _require_non_negative_float("temperature", temperature)
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")

        self._model = name
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds
        self._opener = opener or urllib.request.urlopen

    def generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
        payload = self._post(
            {
                "model": self._model,
                "messages": _chat_messages(prompt),
                "stream": False,
                "options": self._options(max_tokens),
            }
        )
        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        return ""

    def generate_stream(self, prompt: str, *, max_tokens: int | None = None) -> Iterator[str]:
        request = self._chat_request(
            {
                "model": self._model,
                "messages": _chat_messages(prompt),
                "stream": True,
                "options": self._options(max_tokens),
            }
        )
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                for raw in response:
                    line = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if not isinstance(chunk, dict):
                        continue
                    message = chunk.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str) and content:
                            yield content
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Ollama request failed at {self._base_url}: {exc}") from exc

    def _options(self, max_tokens: int | None) -> dict[str, float | int]:
        return {
            "temperature": self._temperature,
            "num_predict": _resolve_max_tokens(max_tokens, self._max_tokens),
        }

    def _chat_request(self, payload: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with self._opener(
                self._chat_request(payload), timeout=self._timeout_seconds
            ) as response:
                raw = response.read()
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Ollama request failed at {self._base_url}: {exc}") from exc
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        parsed: Any = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Ollama returned a non-object JSON payload")
        return parsed
