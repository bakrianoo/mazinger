"""LLM client factory with native Ollama support.

When the base URL points to an Ollama server, requests are routed through
the native ``/api/chat`` endpoint so that parameters like ``think`` are
handled correctly.  For all other providers the standard OpenAI SDK is used.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger(__name__)

_OLLAMA_DEFAULT_PORT = 11434


def _is_ollama_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return (
        host in ("localhost", "127.0.0.1", "0.0.0.0")
        and (port == _OLLAMA_DEFAULT_PORT or path == "")
        and "ollama" in url.lower()
    ) or port == _OLLAMA_DEFAULT_PORT


def _ollama_base(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "localhost"
    port = parsed.port or _OLLAMA_DEFAULT_PORT
    return f"{scheme}://{host}:{port}"


# -- Lightweight response objects that match the OpenAI SDK shape ----------

class _Usage:
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")

    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class _Message:
    __slots__ = ("role", "content")

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


class _Choice:
    __slots__ = ("message",)

    def __init__(self, message: _Message) -> None:
        self.message = message


class _ChatCompletion:
    __slots__ = ("choices", "usage")

    def __init__(self, choices: list[_Choice], usage: _Usage) -> None:
        self.choices = choices
        self.usage = usage


# -- Ollama native chat ----------------------------------------------------

class _OllamaChatCompletions:
    def __init__(self, base_url: str, think: bool | None) -> None:
        self._url = f"{base_url}/api/chat"
        self._think = think

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style multimodal messages to Ollama format.

        Ollama expects ``images`` as a list of raw base64 strings on the
        message dict, not nested ``image_url`` content blocks.
        """
        converted = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                converted.append(msg)
                continue
            text_parts: list[str] = []
            images: list[str] = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part["text"])
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    # Strip the data URI prefix to get raw base64
                    if url.startswith("data:"):
                        url = url.split(",", 1)[-1]
                    images.append(url)
            out: dict[str, Any] = {"role": msg["role"], "content": "\n".join(text_parts)}
            if images:
                out["images"] = images
            converted.append(out)
        return converted

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 1.0,
        **_kwargs: Any,
    ) -> _ChatCompletion:
        body: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "stream": False,
            "options": {"temperature": temperature},
        }
        # Per-call ``think`` overrides the client-level default.
        think = _kwargs.get("think", self._think)
        if think is not None:
            body["think"] = think

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self._url, data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())

        content = result.get("message", {}).get("content", "")
        prompt_tokens = result.get("prompt_eval_count", 0) or 0
        eval_tokens = result.get("eval_count", 0) or 0

        return _ChatCompletion(
            choices=[_Choice(_Message("assistant", content))],
            usage=_Usage(prompt_tokens, eval_tokens),
        )


class _OllamaChat:
    __slots__ = ("completions",)

    def __init__(self, completions: _OllamaChatCompletions) -> None:
        self.completions = completions


class _OllamaClient:
    """Drop-in replacement for ``openai.OpenAI`` that talks native Ollama."""

    def __init__(self, base_url: str, think: bool | None) -> None:
        self.chat = _OllamaChat(_OllamaChatCompletions(base_url, think))


# -- Factory ---------------------------------------------------------------

def build_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    think: bool | None = None,
) -> Any:
    """Return an LLM client appropriate for the given backend.

    For Ollama endpoints, returns a lightweight native client that honours
    the ``think`` parameter.  For everything else, returns a standard
    ``openai.OpenAI`` instance.
    """
    if _is_ollama_url(base_url):
        ollama_base = _ollama_base(base_url)
        log.debug("Using native Ollama client → %s", ollama_base)
        return _OllamaClient(ollama_base, think)

    from openai import OpenAI

    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)
