"""Tests for the Moonshine ASR backend."""

import re

import pytest


# ── Default-model selection by language ────────────────────────────────────


def test_moonshine_default_model_arabic():
    from mazinger.transcribe import _moonshine_default_model

    assert _moonshine_default_model("ar") == "UsefulSensors/moonshine-tiny-ar"
    # BCP-47 forms with a region tag should still resolve to Arabic.
    assert _moonshine_default_model("ar-EG") == "UsefulSensors/moonshine-tiny-ar"
    assert _moonshine_default_model("AR") == "UsefulSensors/moonshine-tiny-ar"


def test_moonshine_default_model_english_or_unknown():
    from mazinger.transcribe import _moonshine_default_model

    assert _moonshine_default_model("en") == "UsefulSensors/moonshine-base"
    assert _moonshine_default_model("en-US") == "UsefulSensors/moonshine-base"
    assert _moonshine_default_model(None) == "UsefulSensors/moonshine-base"
    # Unknown languages fall back to the English/general checkpoint.
    assert _moonshine_default_model("fr") == "UsefulSensors/moonshine-base"
    assert _moonshine_default_model("xx") == "UsefulSensors/moonshine-base"


# ── Hallucination cleanup (loop collapsing) ────────────────────────────────


def test_clean_text_collapses_arabic_phrase_loop():
    from mazinger.transcribe import _clean_text

    looped = "كما قلت، كما قلت، كما قلت، كما قلت، كما قلت"
    assert _clean_text(looped) == "كما قلت"


def test_clean_text_collapses_latin_phrase_loop():
    from mazinger.transcribe import _clean_text

    assert _clean_text("I think, I think, I think, I think") == "I think"


def test_clean_text_collapses_single_word_punctuation_loop():
    from mazinger.transcribe import _clean_text

    assert _clean_text("يعني. يعني. يعني. يعني") == "يعني"


def test_clean_text_preserves_normal_speech():
    """The phrase-loop regex must NOT collapse legitimate sentences."""
    from mazinger.transcribe import _clean_text

    text = "I love programming and testing and debugging"
    assert _clean_text(text) == text


def test_clean_text_collapses_loop_after_real_text():
    from mazinger.transcribe import _clean_text

    text = "Welcome everyone. كما قلت، كما قلت، كما قلت، كما قلت"
    assert _clean_text(text) == "Welcome everyone. كما قلت"


# ── Method dispatch ────────────────────────────────────────────────────────


def test_moonshine_method_in_literal():
    """Guard against accidentally dropping moonshine from TranscribeMethod."""
    import typing
    from mazinger.transcribe import TranscribeMethod

    members = typing.get_args(TranscribeMethod)
    assert "moonshine" in members


def test_transcribe_rejects_unknown_method():
    """Unknown methods should fail with a clear list of supported backends."""
    import tempfile, os
    from mazinger.transcribe import transcribe

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"RIFF")  # not a valid WAV but we want the dispatch check
        path = f.name
    try:
        with pytest.raises(ValueError) as excinfo:
            transcribe(path, "/tmp/out.srt", method="not-a-real-backend")
        msg = str(excinfo.value)
        # The error should mention all supported methods.
        for name in ("openai", "faster-whisper", "whisperx",
                     "mlx-whisper", "deepgram", "moonshine"):
            assert name in msg
    finally:
        os.unlink(path)
