"""Hosted EN→AR translator for the retrieval path (issue #159).

Uses the same OpenAI-compatible chat-completions HTTP shape as generation providers.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from shamela_rag.retrieval.translate import QueryLanguage, Translator

_SYSTEM_PROMPT = (
    "You translate English questions into formal Arabic for searching classical "
    "Islamic Arabic texts. Reply with only the Arabic translation—no explanation, "
    "quotes, or English."
)


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


class OpenAICompatibleTranslator(Translator):
    """Translate via an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "https://api.together.xyz/v1",
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        opener: Any | None = None,
    ) -> None:
        name = model.strip()
        if not name:
            raise ValueError("model name must be non-empty")
        key = api_key.strip()
        if not key:
            raise ValueError("api_key must be non-empty")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")

        self._model = name
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds
        self._opener = opener or urllib.request.urlopen

    def translate(
        self,
        text: str,
        *,
        source: QueryLanguage,
        target: QueryLanguage,
    ) -> str:
        if source == target:
            return text
        if source is not QueryLanguage.ENGLISH or target is not QueryLanguage.ARABIC:
            raise ValueError(
                f"OpenAICompatibleTranslator only supports EN→AR, got {source.value}→{target.value}"
            )
        payload = self._post(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text.strip()},
                ],
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "stream": False,
            }
        )
        translated = _completion_text(payload).strip()
        if not translated:
            raise ValueError("translation API returned empty text")
        return translated

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _chat_request(self, payload: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

    def _api_error_message(self, exc: urllib.error.HTTPError) -> str:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return f"{self._base_url} request failed: HTTP {exc.code}"
        message = body.get("error", {}).get("message") if isinstance(body, dict) else None
        return f"{self._base_url} request failed: HTTP {exc.code} - {message or body}"

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with self._opener(
                self._chat_request(payload), timeout=self._timeout_seconds
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise ConnectionError(self._api_error_message(exc)) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"{self._base_url} request failed: {exc}") from exc
        parsed: Any = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("translation API returned a non-object JSON payload")
        return parsed
