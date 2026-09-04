"""Ollama failures must arrive with the server's own explanation attached.

``urlopen`` raises ``HTTPError`` without reading the response body, but that
body is where Ollama says *why* — model not pulled, not enough memory, context
too long, runner crashed. Losing it turns every distinct failure into the same
useless "HTTP Error 500: Internal Server Error", which is especially bad here:
the LLM stages run after download and transcription, so the run has already
spent real time by the point it fails.
"""

import io
import json
import urllib.error

import pytest

from mazinger.llm import OllamaRequestError, _OllamaClient, _urlopen


def _http_error(code, body, url="http://localhost:11434/api/chat"):
    return urllib.error.HTTPError(
        url, code, "Internal Server Error", {}, io.BytesIO(body.encode()),
    )


def _request():
    import urllib.request

    return urllib.request.Request(
        "http://localhost:11434/api/chat", data=b"{}",
        headers={"Content-Type": "application/json"},
    )


@pytest.fixture
def raising(monkeypatch):
    """Make ``urlopen`` raise whatever the test supplies."""
    def _install(exc):
        def _boom(*_args, **_kwargs):
            raise exc
        monkeypatch.setattr("urllib.request.urlopen", _boom)
    return _install


class TestErrorDetail:
    def test_json_error_body_is_surfaced(self, raising):
        raising(_http_error(500, json.dumps({"error": "model requires more system memory"})))
        with pytest.raises(OllamaRequestError, match="model requires more system memory"):
            _urlopen(_request())

    def test_status_code_and_url_are_kept(self, raising):
        raising(_http_error(500, json.dumps({"error": "boom"})))
        with pytest.raises(OllamaRequestError) as exc:
            _urlopen(_request())
        assert "500" in str(exc.value)
        assert "/api/chat" in str(exc.value)

    def test_non_json_body_is_passed_through(self, raising):
        raising(_http_error(502, "<html>bad gateway</html>"))
        with pytest.raises(OllamaRequestError, match="bad gateway"):
            _urlopen(_request())

    def test_empty_body_still_reports_the_status(self, raising):
        raising(_http_error(500, ""))
        with pytest.raises(OllamaRequestError, match="no detail in the response body"):
            _urlopen(_request())

    def test_json_body_without_an_error_key_falls_back_to_raw(self, raising):
        raising(_http_error(500, '{"unexpected": "shape"}'))
        with pytest.raises(OllamaRequestError, match="unexpected"):
            _urlopen(_request())

    def test_the_original_exception_is_chained(self, raising):
        original = _http_error(500, '{"error": "boom"}')
        raising(original)
        with pytest.raises(OllamaRequestError) as exc:
            _urlopen(_request())
        assert exc.value.__cause__ is original

    def test_overlong_detail_is_truncated(self, raising):
        raising(_http_error(500, json.dumps({"error": "x" * 5000})))
        with pytest.raises(OllamaRequestError) as exc:
            _urlopen(_request())
        assert len(str(exc.value)) < 700


class TestUnreachableServer:
    def test_connection_refused_is_actionable(self, raising):
        raising(urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")))
        with pytest.raises(OllamaRequestError) as exc:
            _urlopen(_request())
        message = str(exc.value)
        assert "Could not reach the Ollama server" in message
        assert "ollama serve" in message
        assert "OLLAMA_HOST" in message


class TestThroughTheClient:
    """The detail must survive the path the pipeline stages actually take."""

    def test_chat_completions_create_reports_the_reason(self, raising):
        raising(_http_error(500, json.dumps({"error": "runner process has terminated"})))
        client = _OllamaClient("http://localhost:11434", None)
        with pytest.raises(OllamaRequestError, match="runner process has terminated"):
            client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "hi"}],
            )

    def test_streaming_path_reports_the_reason(self, raising, monkeypatch):
        from mazinger import llm

        monkeypatch.setattr(llm, "get_stream_callback", lambda: (lambda _tok: None))
        raising(_http_error(500, json.dumps({"error": "context length exceeded"})))
        client = _OllamaClient("http://localhost:11434", None)
        with pytest.raises(OllamaRequestError, match="context length exceeded"):
            client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "hi"}],
            )

    def test_unload_model_never_raises(self, raising):
        """Unloading is best-effort cleanup; a failure must not break a run."""
        raising(_http_error(500, json.dumps({"error": "boom"})))
        _OllamaClient("http://localhost:11434", None).unload_model("m")


def test_server_log_path_is_configurable(monkeypatch):
    """The server log is the only record of a runner-side crash."""
    import importlib

    from mazinger import ollama_setup

    monkeypatch.setenv("OLLAMA_SERVER_LOG", "/tmp/custom-ollama.log")
    importlib.reload(ollama_setup)
    try:
        assert ollama_setup.SERVER_LOG == "/tmp/custom-ollama.log"
    finally:
        monkeypatch.delenv("OLLAMA_SERVER_LOG", raising=False)
        importlib.reload(ollama_setup)
