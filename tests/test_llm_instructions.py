"""Extra user instructions reach the system prompt of every LLM call."""

from __future__ import annotations

import io
import json

import pytest

from mazinger import llm
from mazinger.llm import (
    _INSTRUCTIONS_HEADER,
    _OllamaChatCompletions,
    _StreamingOpenAIChatCompletions,
    inject_instructions,
)


def test_appends_to_first_system_message():
    msgs = [{"role": "system", "content": "Translate."}, {"role": "user", "content": "hi"}]
    out = inject_instructions(msgs, "  Keep names in English.  ")
    assert out[0]["content"] == f"Translate.\n\n{_INSTRUCTIONS_HEADER}\nKeep names in English."
    assert out[1] == msgs[1]
    assert msgs[0]["content"] == "Translate."  # caller's list is untouched


def test_adds_system_message_when_missing():
    msgs = [{"role": "user", "content": "hi"}]
    out = inject_instructions(msgs, "Be brief.")
    assert out[0] == {"role": "system", "content": f"{_INSTRUCTIONS_HEADER}\nBe brief."}
    assert out[1:] == msgs


def test_multimodal_system_content_gets_a_text_part():
    msgs = [{"role": "system", "content": [{"type": "text", "text": "Describe."}]}]
    out = inject_instructions(msgs, "Be brief.")
    assert out[0]["content"][-1] == {"type": "text", "text": f"{_INSTRUCTIONS_HEADER}\nBe brief."}


@pytest.mark.parametrize("empty", [None, "", "   \n"])
def test_empty_instructions_change_nothing(empty):
    msgs = [{"role": "system", "content": "Translate."}]
    assert inject_instructions(msgs, empty) is msgs


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return "ok"


def test_openai_client_injects_into_every_call():
    inner = _FakeCompletions()
    completions = _StreamingOpenAIChatCompletions(inner, "Use formal tone.")
    completions.create(model="m", messages=[{"role": "system", "content": "Task."}])
    completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    assert inner.calls[0]["messages"][0]["content"].endswith("Use formal tone.")
    assert inner.calls[1]["messages"][0]["role"] == "system"
    assert inner.calls[1]["messages"][0]["content"].endswith("Use formal tone.")


def test_ollama_client_injects_into_request_body(monkeypatch):
    seen = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data)
        seen["timeout"] = timeout
        return _Resp(json.dumps({"message": {"content": "OK"}}).encode())

    monkeypatch.setattr(llm, "_urlopen", fake_urlopen)
    monkeypatch.setattr(llm, "get_stream_callback", lambda: None)
    completions = _OllamaChatCompletions("http://localhost:11434", False, "Be brief.", timeout=5)
    resp = completions.create(model="m", messages=[{"role": "system", "content": "Task."}])
    assert resp.choices[0].message.content == "OK"
    assert seen["body"]["messages"][0]["content"].endswith("Be brief.")
    assert seen["timeout"] == 5


def test_health_check_reports_unreachable_endpoint():
    from mazinger.studio.pipeline import check_llm_connection

    msg = check_llm_connection(
        "OpenAI (Cloud)", "", "sk-test", "http://127.0.0.1:9/v1", "gpt-4.1",
    )
    assert msg.startswith("❌ Could not reach the API Base URL")


def test_health_check_requires_api_key():
    from mazinger.studio.pipeline import check_llm_connection

    assert check_llm_connection("OpenAI (Cloud)", "", "  ", "", "gpt-4.1").startswith("❌")
