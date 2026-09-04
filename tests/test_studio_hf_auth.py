"""Tests for the Studio's HuggingFace sign-in flow.

``huggingface_hub.notebook_login()`` cannot drive a Gradio UI — it renders
through ``IPython.display`` into a notebook cell and otherwise falls back to a
blocking terminal prompt.  The Studio therefore drives the same OAuth
device-code flow directly.  These tests pin that behaviour.
"""

import os
import sys
import types
from unittest.mock import patch

import pytest

hub = pytest.importorskip("huggingface_hub")

from mazinger.studio import helpers as H  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_hub_env():
    """Materialise lazy exports and isolate token state between tests."""
    # huggingface_hub resolves `login` etc. lazily through huggingface_hub._login,
    # which these tests replace — force them into the module dict first.
    for name in ("login", "whoami", "get_token", "logout"):
        setattr(hub, name, getattr(hub, name))

    saved = os.environ.pop("HF_TOKEN", None)
    import mazinger.profiles as profiles
    saved_profile_token = profiles.HF_TOKEN
    yield
    profiles.HF_TOKEN = saved_profile_token
    os.environ.pop("HF_TOKEN", None)
    if saved is not None:
        os.environ["HF_TOKEN"] = saved


DEVICE_INFO = {
    "verification_uri": "https://huggingface.co/device",
    "verification_uri_complete": "https://huggingface.co/device?code=ABCD-1234",
    "user_code": "ABCD-1234",
    "device_code": "dev-code",
    "interval": 1,
    "expires_in": 30,
}


def fake_login_module(*, request=None, poll=None):
    mod = types.ModuleType("huggingface_hub._login")
    mod.request_device_code = request or (lambda: DEVICE_INFO)
    mod.poll_device_token = poll or (lambda info, **kw: {"access_token": "hf_granted"})
    return mod


def run_flow(login_module):
    """Collect every Markdown frame the generator yields."""
    with patch.dict(sys.modules, {"huggingface_hub._login": login_module}), \
         patch("huggingface_hub.get_token", return_value=None), \
         patch("huggingface_hub.login") as login_fn, \
         patch("huggingface_hub.whoami", return_value={"name": "tester"}):
        frames = list(H.hf_login_flow())
    return frames, login_fn


# ── Status ──────────────────────────────────────────────────────────────────

def test_status_signed_out():
    with patch("huggingface_hub.get_token", return_value=None):
        assert "Not signed in" in H.hf_status()


def test_status_signed_in_propagates_token():
    import mazinger.profiles as profiles
    with patch("huggingface_hub.get_token", return_value="hf_abc"), \
         patch("huggingface_hub.whoami", return_value={"name": "tester"}):
        assert "tester" in H.hf_status()
    # mazinger.profiles caches HF_TOKEN at import time, so setting the
    # environment variable alone would not reach gated profile downloads.
    assert os.environ["HF_TOKEN"] == "hf_abc"
    assert profiles.HF_TOKEN == "hf_abc"


def test_status_reports_unverifiable_token():
    with patch("huggingface_hub.get_token", return_value="hf_stale"), \
         patch("huggingface_hub.whoami", side_effect=Exception("401")):
        assert "could not be verified" in H.hf_status()


# ── Device-code flow ────────────────────────────────────────────────────────

def test_device_flow_shows_code_before_polling():
    """A fast authorisation must not skip past the link and code."""
    frames, login_fn = run_flow(fake_login_module())
    assert len(frames) >= 2
    assert "ABCD-1234" in frames[0]
    assert "device?code=ABCD-1234" in frames[0]
    assert frames[-1].startswith("✅")
    login_fn.assert_called_once_with(token="hf_granted", add_to_git_credential=False)


def test_device_flow_keeps_code_visible_while_waiting():
    import time

    def slow_poll(info, **kw):
        time.sleep(5)
        return {"access_token": "hf_granted"}

    frames, _ = run_flow(fake_login_module(poll=slow_poll))
    assert len(frames) > 2, "expected countdown frames while waiting"
    assert all("ABCD-1234" in f for f in frames[:-1])


def test_device_flow_applies_token_everywhere():
    import mazinger.profiles as profiles
    run_flow(fake_login_module())
    assert os.environ["HF_TOKEN"] == "hf_granted"
    assert profiles.HF_TOKEN == "hf_granted"


def test_device_flow_surfaces_denial():
    def denied(info, **kw):
        raise RuntimeError("access denied by user")

    frames, _ = run_flow(fake_login_module(poll=denied))
    assert "Login failed" in frames[-1]
    assert "access denied by user" in frames[-1]


def test_device_flow_handles_unreachable_endpoint():
    def boom():
        raise RuntimeError("connection refused")

    frames, _ = run_flow(fake_login_module(request=boom))
    assert "Could not start HuggingFace login" in frames[0]
    assert "connection refused" in frames[0]


def test_device_flow_falls_back_when_unavailable():
    """Older huggingface_hub has no device flow — offer the token box instead."""
    with patch.dict(sys.modules, {"huggingface_hub._login": None}), \
         patch("huggingface_hub.get_token", return_value=None):
        frames = list(H.hf_login_flow())
    assert "no device-code login" in frames[0]
    assert "settings/tokens" in frames[0]


def test_already_signed_in_skips_device_code():
    called = []
    mod = fake_login_module(request=lambda: called.append(1) or DEVICE_INFO)
    with patch.dict(sys.modules, {"huggingface_hub._login": mod}), \
         patch("huggingface_hub.get_token", return_value="hf_existing"), \
         patch("huggingface_hub.whoami", return_value={"name": "tester"}):
        frames = list(H.hf_login_flow())
    assert "Already signed in" in frames[0]
    assert not called, "should not request a device code when already signed in"


def test_gated_model_links_are_listed():
    """Signing in is not enough — users must accept terms per model."""
    frames, _ = run_flow(fake_login_module())
    from mazinger.studio.constants import GATED_MODELS
    for _label, repo in GATED_MODELS:
        assert repo in frames[-1]


# ── Token paste fallback ────────────────────────────────────────────────────

def test_token_paste_accepts_valid_token():
    with patch("huggingface_hub.whoami", return_value={"name": "tester"}), \
         patch("huggingface_hub.login") as login_fn:
        out = H.hf_login_with_token("  hf_good  ")
    assert "tester" in out
    login_fn.assert_called_once_with(token="hf_good", add_to_git_credential=False)
    assert os.environ["HF_TOKEN"] == "hf_good"


def test_token_paste_rejects_invalid_token():
    with patch("huggingface_hub.whoami", side_effect=Exception("401 Unauthorized")):
        out = H.hf_login_with_token("hf_bad")
    assert "rejected" in out
    assert "HF_TOKEN" not in os.environ


def test_token_paste_requires_input():
    assert "Paste an access token" in H.hf_login_with_token("   ")


# ── Logout ──────────────────────────────────────────────────────────────────

def test_logout_clears_every_token_location():
    import mazinger.profiles as profiles
    with patch("huggingface_hub.whoami", return_value={"name": "tester"}), \
         patch("huggingface_hub.login"):
        H.hf_login_with_token("hf_good")
    with patch("huggingface_hub.logout"):
        assert "Signed out" in H.hf_logout()
    assert "HF_TOKEN" not in os.environ
    assert profiles.HF_TOKEN is None


def test_logout_survives_missing_stored_token():
    with patch("huggingface_hub.logout", side_effect=Exception("no token stored")):
        assert "Signed out" in H.hf_logout()


# ── Studio wiring ───────────────────────────────────────────────────────────

def test_studio_copies_stay_in_sync():
    """The Colab copy under docs/notebooks must expose the same helpers."""
    import pathlib
    packaged = pathlib.Path("mazinger/studio/helpers.py").read_text(encoding="utf-8")
    notebook = pathlib.Path("docs/notebooks/studio/helpers.py").read_text(encoding="utf-8")
    assert notebook == packaged.replace(
        "from mazinger.studio.constants import", "from constants import"
    )


@pytest.mark.parametrize("path", [
    "mazinger/studio/app.py",
    "docs/notebooks/studio/app.py",
])
def test_app_wires_the_login_buttons(path):
    import pathlib
    src = pathlib.Path(path).read_text(encoding="utf-8")
    assert "hf_login_btn.click(hf_login_flow, None, hf_status_md)" in src
    assert "hf_logout_btn.click(hf_logout, None, hf_status_md)" in src
    assert "hf_token_btn.click(hf_login_with_token, hf_token_box, hf_status_md)" in src
    # The token box must never render in plain text.
    assert 'type="password"' in src
