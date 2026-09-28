"""OpenAI models that reject sampling kwargs still get a completion."""

from __future__ import annotations

import httpx
import openai
import pytest

from mazinger.llm import _StreamingOpenAIChatCompletions


def _unsupported(param: str, code: str = "unsupported_parameter") -> openai.BadRequestError:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"message": f"'{param}' is not supported", "type": "invalid_request_error",
            "param": param, "code": code}
    return openai.BadRequestError(
        "Error code: 400", response=httpx.Response(400, request=req), body=body,
    )


class _FakeCompletions:
    """Rejects each kwarg in *rejects* the way a reasoning model does."""

    def __init__(self, rejects: dict[str, str]) -> None:
        self.rejects = rejects
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        for param, code in self.rejects.items():
            if param in kwargs:
                raise _unsupported(param, code)
        return "ok"


@pytest.fixture(autouse=True)
def _clear_cache():
    _StreamingOpenAIChatCompletions._rejected.clear()
    yield
    _StreamingOpenAIChatCompletions._rejected.clear()


def test_drops_rejected_params_and_remembers_them():
    inner = _FakeCompletions({"max_tokens": "unsupported_parameter",
                              "temperature": "unsupported_value"})
    comp = _StreamingOpenAIChatCompletions(inner)

    out = comp.create(model="gpt-5", messages=[], temperature=0.2,
                      num_predict=2048, top_k=40)

    assert out == "ok"
    assert "max_tokens" not in inner.calls[-1]
    assert "temperature" not in inner.calls[-1]
    assert len(inner.calls) == 3

    # A second call for the same model skips the failing round-trips.
    inner.calls.clear()
    assert comp.create(model="gpt-5", messages=[], temperature=0.3, num_predict=128) == "ok"
    assert len(inner.calls) == 1


def test_other_models_keep_their_params():
    comp = _StreamingOpenAIChatCompletions(_FakeCompletions({"max_tokens": "unsupported_parameter"}))
    comp.create(model="gpt-5", messages=[], num_predict=10)

    inner = _FakeCompletions({})
    _StreamingOpenAIChatCompletions(inner).create(model="gpt-4.1", messages=[], num_predict=10)
    assert inner.calls[-1]["max_tokens"] == 10


def test_unrelated_errors_propagate():
    inner = _FakeCompletions({"messages": "unsupported_parameter"})
    with pytest.raises(openai.BadRequestError):
        _StreamingOpenAIChatCompletions(inner).create(model="gpt-5", messages=[])
