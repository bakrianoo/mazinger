"""LLM client factory with native Ollama support.

When the base URL points to an Ollama server, requests are routed through
the native ``/api/chat`` endpoint so that parameters like ``think`` are
handled correctly.  For all other providers the standard OpenAI SDK is used.

Streaming
---------
Call :func:`set_stream_callback` with a ``callback(token: str)`` function
before running pipeline stages.  When set, every LLM completion will stream
tokens through the callback *and* still return the full response object as
usual — callers do not need any changes.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Any, Callable
from urllib.parse import urlparse

log = logging.getLogger(__name__)


class OllamaRequestError(RuntimeError):
    """An Ollama HTTP call failed, carrying the server's own explanation."""


def _urlopen(req: urllib.request.Request, timeout: float | None = None):
    """``urlopen`` that preserves Ollama's error message.

    Ollama explains every failure in the response body as ``{"error": "..."}``
    — the model is not pulled, it needs more memory than is available, the
    context is too long, the runner died.  ``urlopen`` raises ``HTTPError``
    *without* reading that body, so the caller is left with a bare
    "HTTP Error 500: Internal Server Error" and no way to tell those apart.
    A pipeline that dies after transcription has already run deserves better
    than that, so unpack the body here.
    """
    try:
        if timeout is None:
            return urllib.request.urlopen(req)
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            raw = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:  # body already consumed or unreadable
            raw = ""
        if raw:
            try:
                parsed = json.loads(raw)
                detail = parsed.get("error") if isinstance(parsed, dict) else ""
                detail = detail or raw
            except json.JSONDecodeError:
                detail = raw
        message = f"Ollama returned HTTP {exc.code} for {req.full_url}"
        if detail:
            message += f": {detail[:500]}"
        else:
            message += " (no detail in the response body)"
        raise OllamaRequestError(message) from exc
    except urllib.error.URLError as exc:
        raise OllamaRequestError(
            f"Could not reach the Ollama server at {req.full_url} ({exc.reason}). "
            "Start it with `ollama serve`, or point OLLAMA_HOST at the right host."
        ) from exc


# -- Global stream callback ------------------------------------------------

_stream_lock = threading.Lock()
_stream_callback: Callable[[str], Any] | None = None


def set_stream_callback(callback: Callable[[str], Any] | None) -> None:
    """Set a global callback that receives each streamed token.

    Pass ``None`` to disable streaming.  The callback signature is
    ``callback(token: str)``.
    """
    global _stream_callback
    with _stream_lock:
        _stream_callback = callback


def get_stream_callback() -> Callable[[str], Any] | None:
    """Return the current stream callback (or ``None``)."""
    with _stream_lock:
        return _stream_callback


def clear_stream_callback() -> None:
    """Convenience alias for ``set_stream_callback(None)``."""
    set_stream_callback(None)

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
        callback = get_stream_callback()

        options: dict[str, Any] = {"temperature": temperature}
        # Forward Ollama-specific sampling options when provided.
        for opt_key in (
            "repeat_penalty", "top_p", "top_k", "num_predict",
            "frequency_penalty", "presence_penalty", "seed",
        ):
            if opt_key in _kwargs:
                options[opt_key] = _kwargs[opt_key]

        body: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "stream": bool(callback),
            "options": options,
        }
        # Per-call ``think`` overrides the client-level default.
        # Default to *disabled* so thinking models don't burn tokens
        # unless explicitly opted-in via ``build_client(think=True)``
        # or a per-call ``think=True``.
        think = _kwargs.get("think", self._think)
        body["think"] = bool(think)

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self._url, data=data,
            headers={"Content-Type": "application/json"},
        )

        if not callback:
            # Non-streaming path (original behaviour)
            with _urlopen(req) as resp:
                result = json.loads(resp.read())

            content = result.get("message", {}).get("content", "")
            prompt_tokens = result.get("prompt_eval_count", 0) or 0
            eval_tokens = result.get("eval_count", 0) or 0
        else:
            # Streaming path — accumulate tokens, forward to callback
            content_parts: list[str] = []
            prompt_tokens = 0
            eval_tokens = 0
            with _urlopen(req) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        content_parts.append(token)
                        try:
                            callback(token)
                        except Exception:
                            pass
                    # Last chunk carries the usage counters
                    if chunk.get("done"):
                        prompt_tokens = chunk.get("prompt_eval_count", 0) or 0
                        eval_tokens = chunk.get("eval_count", 0) or 0
            content = "".join(content_parts)

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
        self._base_url = base_url
        self.chat = _OllamaChat(_OllamaChatCompletions(base_url, think))

    def unload_model(self, model: str) -> None:
        """Tell Ollama to unload *model* from GPU memory."""
        body = json.dumps({
            "model": model, "keep_alive": 0,
        }).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/generate", body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _urlopen(req, timeout=10) as resp:
                resp.read()
            log.info("Ollama model %s unloaded from GPU", model)
        except Exception:
            log.debug("Ollama unload request failed (non-critical)", exc_info=True)


# -- Factory ---------------------------------------------------------------


class _StreamingOpenAIChatCompletions:
    """Proxy that intercepts ``create()`` to stream tokens via callback."""

    # Keys that are Ollama-specific and not understood by the OpenAI SDK.
    _OLLAMA_ONLY_KEYS = frozenset({
        "repeat_penalty", "top_k", "num_predict", "think",
    })

    # Map portable kwarg names to their OpenAI equivalents.
    _KWARG_MAP = {"num_predict": "max_tokens"}

    def __init__(self, inner) -> None:
        self._inner = inner

    def _normalise_kwargs(self, kwargs: dict) -> dict:
        """Translate portable kwargs to OpenAI names, drop unsupported ones."""
        # Apply mappings first (e.g. num_predict → max_tokens)
        for src, dst in self._KWARG_MAP.items():
            if src in kwargs and dst not in kwargs:
                kwargs[dst] = kwargs.pop(src)

        # Strip remaining Ollama-only keys
        for key in self._OLLAMA_ONLY_KEYS:
            kwargs.pop(key, None)
        return kwargs

    def create(self, **kwargs):
        kwargs = self._normalise_kwargs(kwargs)

        callback = get_stream_callback()
        if not callback:
            return self._inner.create(**kwargs)

        # Force streaming on, collect full response for caller
        kwargs["stream"] = True
        stream_resp = self._inner.create(**kwargs)

        content_parts: list[str] = []
        role = "assistant"
        prompt_tokens = 0
        completion_tokens = 0

        for chunk in stream_resp:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                content_parts.append(delta.content)
                try:
                    callback(delta.content)
                except Exception:
                    pass
            if delta and delta.role:
                role = delta.role
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0

        content = "".join(content_parts)
        return _ChatCompletion(
            choices=[_Choice(_Message(role, content))],
            usage=_Usage(prompt_tokens, completion_tokens),
        )


class _StreamingOpenAIChat:
    """Proxy for ``client.chat`` that wraps ``completions``."""

    def __init__(self, inner_chat) -> None:
        self.completions = _StreamingOpenAIChatCompletions(inner_chat.completions)


class _StreamingOpenAIClient:
    """Thin wrapper around ``openai.OpenAI`` that adds stream-callback support."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.chat = _StreamingOpenAIChat(inner.chat)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def build_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    think: bool | None = None,
) -> Any:
    """Return an LLM client appropriate for the given backend.

    For Ollama endpoints, returns a lightweight native client that honours
    the ``think`` parameter.  For everything else, returns a standard
    ``openai.OpenAI`` instance (wrapped for stream-callback support).
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
    return _StreamingOpenAIClient(OpenAI(**kwargs))
