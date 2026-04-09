"""LLM client factory with native Ollama and Gemini support.

When the base URL points to an Ollama server, requests are routed through
the native ``/api/chat`` endpoint so that parameters like ``think`` are
handled correctly.  When a Gemini API key is provided, the Google GenAI
SDK is used.  For all other providers the standard OpenAI SDK is used.

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
import urllib.request
from typing import Any, Callable
from urllib.parse import urlparse

log = logging.getLogger(__name__)

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
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())

            content = result.get("message", {}).get("content", "")
            prompt_tokens = result.get("prompt_eval_count", 0) or 0
            eval_tokens = result.get("eval_count", 0) or 0
        else:
            # Streaming path — accumulate tokens, forward to callback
            content_parts: list[str] = []
            prompt_tokens = 0
            eval_tokens = 0
            with urllib.request.urlopen(req) as resp:
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
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
            log.info("Ollama model %s unloaded from GPU", model)
        except Exception:
            log.debug("Ollama unload request failed (non-critical)", exc_info=True)


# -- Gemini (Google GenAI) native chat ------------------------------------

class _GeminiChatCompletions:
    """Translate OpenAI-style ``create()`` calls to Google GenAI SDK."""

    _UNSUPPORTED_KEYS = frozenset({
        "repeat_penalty", "top_k", "num_predict", "think",
        "compression_ratio_threshold", "log_prob_threshold",
        "no_speech_threshold",
    })

    def __init__(self, api_key: str) -> None:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai not installed. Install with: "
                "pip install 'mazinger[llm-gemini]'\n"
                "Or use an OpenAI-compatible provider."
            ) from e
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Split OpenAI messages into Gemini system_instruction + contents."""
        import base64
        from google.genai import types

        system_text: list[str] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]
            raw_content = msg.get("content", "")

            if role == "system":
                if isinstance(raw_content, str):
                    system_text.append(raw_content)
                elif isinstance(raw_content, list):
                    system_text.extend(
                        p["text"] for p in raw_content if p.get("type") == "text"
                    )
                continue

            gemini_role = "model" if role == "assistant" else "user"
            parts: list[types.Part] = []

            if isinstance(raw_content, str):
                parts.append(types.Part.from_text(text=raw_content))
            elif isinstance(raw_content, list):
                for block in raw_content:
                    if block.get("type") == "text":
                        parts.append(types.Part.from_text(text=block["text"]))
                    elif block.get("type") == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            header, b64_data = url.split(",", 1)
                            mime = header.split(":")[1].split(";")[0]
                            parts.append(types.Part.from_bytes(
                                data=base64.b64decode(b64_data),
                                mime_type=mime,
                            ))

            if parts:
                contents.append(types.Content(role=gemini_role, parts=parts))

        system_instruction = "\n".join(system_text) if system_text else None
        return system_instruction, contents

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> _ChatCompletion:
        from google.genai import types

        system_instruction, contents = self._convert_messages(messages)

        config_kw: dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            config_kw["system_instruction"] = system_instruction
        if "top_p" in kwargs:
            config_kw["top_p"] = kwargs["top_p"]
        if "seed" in kwargs:
            config_kw["seed"] = kwargs["seed"]
        max_tokens = kwargs.get("max_tokens") or kwargs.get("num_predict")
        if max_tokens:
            config_kw["max_output_tokens"] = max_tokens
        if "frequency_penalty" in kwargs:
            config_kw["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            config_kw["presence_penalty"] = kwargs["presence_penalty"]

        callback = get_stream_callback()

        def _call(kw: dict[str, Any]) -> _ChatCompletion:
            config = types.GenerateContentConfig(**kw)

            if not callback:
                response = self._client.models.generate_content(
                    model=model, contents=contents, config=config,
                )
                text = response.text or ""
                usage = response.usage_metadata
                p_tok = getattr(usage, "prompt_token_count", 0) or 0
                c_tok = getattr(usage, "response_token_count", 0) or 0
            else:
                parts: list[str] = []
                p_tok = c_tok = 0
                for chunk in self._client.models.generate_content_stream(
                    model=model, contents=contents, config=config,
                ):
                    token = chunk.text or ""
                    if token:
                        parts.append(token)
                        try:
                            callback(token)
                        except Exception:
                            pass
                    usage = chunk.usage_metadata
                    if usage:
                        p_tok = getattr(usage, "prompt_token_count", 0) or 0
                        c_tok = getattr(usage, "response_token_count", 0) or 0
                text = "".join(parts)

            return _ChatCompletion(
                choices=[_Choice(_Message("assistant", text))],
                usage=_Usage(p_tok, c_tok),
            )

        try:
            return _call(config_kw)
        except Exception as exc:
            err_msg = str(exc).lower()
            if "penalty" in err_msg and (
                "frequency_penalty" in config_kw or "presence_penalty" in config_kw
            ):
                log.debug("Gemini rejected penalty params, retrying without them")
                config_kw.pop("frequency_penalty", None)
                config_kw.pop("presence_penalty", None)
                return _call(config_kw)
            raise


class _GeminiChat:
    __slots__ = ("completions",)

    def __init__(self, completions: _GeminiChatCompletions) -> None:
        self.completions = completions


class _GeminiClient:
    """Drop-in replacement for ``openai.OpenAI`` that talks to Google Gemini."""

    def __init__(self, api_key: str) -> None:
        self.chat = _GeminiChat(_GeminiChatCompletions(api_key))


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
    gemini_api_key: str | None = None,
) -> Any:
    """Return an LLM client appropriate for the given backend.

    For Ollama endpoints, returns a lightweight native client that honours
    the ``think`` parameter.  When *gemini_api_key* is provided, returns a
    Gemini client using the Google GenAI SDK.  For everything else, returns
    a standard ``openai.OpenAI`` instance (wrapped for stream-callback
    support).
    """
    if gemini_api_key:
        log.debug("Using Gemini client (google-genai)")
        return _GeminiClient(gemini_api_key)

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
