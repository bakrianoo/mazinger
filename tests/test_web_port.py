"""`mazinger web` must not fail when port 7860 is taken.

Gradio searches for a free port from 7860 (or $GRADIO_SERVER_PORT) upward
only when ``server_port`` is ``None``; an explicit port is used as-is.
"""

import socket
import sys
import types

import pytest

from mazinger.cli import _build_parser, main


def _web_launch_kwargs(monkeypatch, argv):
    calls = []
    fake_app = types.ModuleType("mazinger.studio.app")
    fake_app.launch = lambda **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, "mazinger.studio.app", fake_app)
    main(["web", "--no-share", *argv])
    return calls[0]


def test_default_port_lets_gradio_pick_a_free_one(monkeypatch):
    assert _build_parser().parse_args(["web"]).server_port is None
    assert _web_launch_kwargs(monkeypatch, [])["server_port"] is None


def test_explicit_port_is_passed_through(monkeypatch):
    assert _web_launch_kwargs(monkeypatch, ["--server-port", "8123"])["server_port"] == 8123


def test_gradio_falls_back_when_7860_is_busy():
    gr = pytest.importorskip("gradio")
    blocker = socket.socket()
    try:
        blocker.bind(("127.0.0.1", 7860))
        blocker.listen()
    except OSError:
        pass  # already taken by something else — the case being tested
    try:
        with gr.Blocks() as demo:
            gr.Markdown("x")
        demo.launch(prevent_thread_lock=True, server_name="127.0.0.1", quiet=True)
        try:
            assert demo.server_port != 7860
        finally:
            demo.close()
    finally:
        blocker.close()
