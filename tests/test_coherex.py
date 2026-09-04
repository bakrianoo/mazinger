"""Tests for the CohereX transcription backend.

The real backend needs a CUDA GPU and HuggingFace access to two gated
repositories, so these tests stub the ``coherex`` module with fakes that mirror
its published API (``coherex/schema.py``) and assert that Mazinger drives it
correctly and post-processes its output unchanged.
"""

import os
import sys
import types
from unittest.mock import patch

import pytest

import mazinger.transcribe as T


# ── Fake coherex module ─────────────────────────────────────────────────────

def _words(text, t0, t1):
    """Word records in coherex's SingleWordSegment shape."""
    toks = text.split()
    step = (t1 - t0) / max(len(toks), 1)
    return [
        {"word": w, "start": round(t0 + i * step, 3),
         "end": round(t0 + (i + 1) * step, 3), "score": 0.9}
        for i, w in enumerate(toks)
    ]


SAMPLE = [
    (0.5, 6.1, "So the first thing you need to understand is really quite simple."),
    (6.1, 12.3, "We take the input signal and pass it through the encoder."),
    (12.3, 20.0, "And that gives us the representation we want, which we decode "
                 "back out into the target domain."),
]


class _FakePipeline:
    """Stand-in for coherex.asr.CohereAsrPipeline."""

    def __init__(self, supported_languages, calls):
        self.supported_languages = list(supported_languages)
        self.calls = calls
        self.shutdown_called = False

    def transcribe(self, audio, language=None, batch_size=None, chunk_size=30):
        self.calls["transcribe"] = {
            "audio": audio, "language": language,
            "batch_size": batch_size, "chunk_size": chunk_size,
        }
        return {
            "segments": [{"start": s, "end": e, "text": t} for s, e, t in SAMPLE],
            "language": language,
        }

    def shutdown(self):
        self.shutdown_called = True


def make_fake_coherex(calls, supported=None, align_raises=False):
    mod = types.ModuleType("coherex")
    supported = supported or list(T.COHEREX_LANGUAGES)

    def load_model(model_name, **kw):
        calls["load_model"] = {"model_name": model_name, **kw}
        return _FakePipeline(supported, calls)

    def detect_language(pipeline, audio, candidates=None):
        calls["detect_language"] = {"audio": audio, "candidates": candidates}
        return "fr"

    def load_align_model(language_code, device, **kw):
        calls["load_align_model"] = {"language_code": language_code, "device": device}
        if align_raises:
            raise ValueError(f"No default align-model for language: {language_code}")
        return object(), {"language": language_code}

    def align(segments, model, metadata, audio, device, **kw):
        calls["align"] = {"n": len(segments), "device": device, **kw}
        return {
            "segments": [
                {**s, "words": _words(s["text"], s["start"], s["end"])} for s in segments
            ],
            "word_segments": [],
        }

    mod.load_model = load_model
    mod.detect_language = detect_language
    mod.load_align_model = load_align_model
    mod.align = align
    return mod


@pytest.fixture
def fake_env():
    """Patch coherex + torch + torchaudio into sys.modules."""
    calls = {}

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None,
    )
    fake_torchaudio = types.ModuleType("torchaudio")

    def _install(mod):
        return patch.dict(sys.modules, {
            "coherex": mod, "torch": fake_torch, "torchaudio": fake_torchaudio,
        })

    return calls, _install, fake_torchaudio


# ── Model selection ─────────────────────────────────────────────────────────

def test_arabic_auto_selects_arabic_model():
    assert T.resolve_coherex_model(None, "ar") == T.COHEREX_ARABIC_MODEL


def test_non_arabic_uses_base_model():
    assert T.resolve_coherex_model(None, "en") == T.DEFAULT_COHEREX_MODEL
    assert T.resolve_coherex_model(None, None) == T.DEFAULT_COHEREX_MODEL


def test_explicit_model_overrides_arabic_default():
    assert T.resolve_coherex_model("my/model", "ar") == "my/model"


# ── compute_type mapping ────────────────────────────────────────────────────

@pytest.mark.parametrize("given,device,expected", [
    ("bfloat16", "cuda", "bfloat16"),
    ("float16", "cuda", "float16"),
    ("int8_float16", "cuda", "default"),   # CTranslate2 type CohereX rejects
    ("int8", "cuda", "default"),
    ("float16", "cpu", "float32"),         # fp16 is useless on CPU
])
def test_compute_type_mapping(given, device, expected):
    assert T._coherex_compute_type(given, device) == expected


# ── Transcription ───────────────────────────────────────────────────────────

def test_transcribe_returns_aligned_segments(fake_env):
    calls, install, _ = fake_env
    with install(make_fake_coherex(calls)):
        segments, lang = T._transcribe_coherex("a.mp3", language="en", device="cpu")

    assert lang == "en"
    assert len(segments) == len(SAMPLE)
    assert all({"start", "end", "text", "words"} <= set(s) for s in segments)
    assert {"word", "start", "end"} <= set(segments[0]["words"][0])
    assert calls["load_model"]["model_name"] == T.DEFAULT_COHEREX_MODEL
    assert calls["transcribe"]["language"] == "en"


def test_arabic_routes_to_arabic_model(fake_env):
    calls, install, _ = fake_env
    with install(make_fake_coherex(calls)):
        T._transcribe_coherex("a.mp3", language="ar", device="cpu")
    assert calls["load_model"]["model_name"] == T.COHEREX_ARABIC_MODEL


def test_unsupported_language_fails_fast(fake_env):
    calls, install, _ = fake_env
    with install(make_fake_coherex(calls)):
        with pytest.raises(ValueError, match="does not support language 'ru'"):
            T._transcribe_coherex("a.mp3", language="ru", device="cpu")
    # Must fail before spending a single forward pass.
    assert "transcribe" not in calls


def test_missing_language_triggers_probe_detection(fake_env):
    calls, install, _ = fake_env
    with install(make_fake_coherex(calls)):
        segments, lang = T._transcribe_coherex("a.mp3", language=None, device="cpu")
    assert lang == "fr"
    assert calls["detect_language"]["candidates"] == list(T.COHEREX_LANGUAGES)
    assert calls["transcribe"]["language"] == "fr"


def test_auto_language_is_treated_as_detection(fake_env):
    calls, install, _ = fake_env
    with install(make_fake_coherex(calls)):
        _, lang = T._transcribe_coherex("a.mp3", language="auto", device="cpu")
    assert lang == "fr"
    assert "detect_language" in calls


def test_alignment_failure_degrades_to_segment_timings(fake_env):
    calls, install, _ = fake_env
    with install(make_fake_coherex(calls, align_raises=True)):
        segments, lang = T._transcribe_coherex("a.mp3", language="en", device="cpu")
    # Transcription survives; only word-level timing is lost.
    assert len(segments) == len(SAMPLE)
    assert "words" not in segments[0]
    assert lang == "en"


def test_device_index_is_split_out(fake_env):
    calls, install, _ = fake_env
    with install(make_fake_coherex(calls)):
        T._transcribe_coherex("a.mp3", language="en", device="cuda:1")
    assert calls["load_model"]["device"] == "cuda"
    assert calls["load_model"]["device_index"] == 1


def test_hf_token_falls_back_to_env(fake_env):
    calls, install, _ = fake_env
    with install(make_fake_coherex(calls)), patch.dict(os.environ, {"HF_TOKEN": "hf_xyz"}):
        T._transcribe_coherex("a.mp3", language="en", device="cpu")
    assert calls["load_model"]["use_auth_token"] == "hf_xyz"


def test_vad_method_is_forwarded(fake_env):
    calls, install, _ = fake_env
    with install(make_fake_coherex(calls)):
        T._transcribe_coherex("a.mp3", language="en", device="cpu", vad_method="silero")
    assert calls["load_model"]["vad_method"] == "silero"


def test_torchaudio_shims_are_installed(fake_env):
    calls, install, fake_torchaudio = fake_env
    with install(make_fake_coherex(calls)):
        T._transcribe_coherex("a.mp3", language="en", device="cpu")
    # pyannote pulls in speechbrain, which still calls these torchaudio<2.11 APIs.
    assert fake_torchaudio.list_audio_backends() == ["soundfile"]
    assert fake_torchaudio.set_audio_backend("soundfile") is None


def test_missing_package_raises_actionable_error():
    fake_torch = types.ModuleType("torch")
    fake_torchaudio = types.ModuleType("torchaudio")
    with patch.dict(sys.modules, {"torch": fake_torch, "torchaudio": fake_torchaudio,
                                  "coherex": None}):
        with pytest.raises(ImportError, match="transcribe-coherex"):
            T._transcribe_coherex("a.mp3", language="en", device="cpu")


# ── Integration with Mazinger's post-processing ─────────────────────────────

def test_output_survives_mazinger_postprocessing(fake_env):
    """CohereX segments must flow through clean -> resegment -> SRT -> parse."""
    calls, install, _ = fake_env
    with install(make_fake_coherex(calls)):
        segments, _ = T._transcribe_coherex("a.mp3", language="en", device="cpu")

    cleaned = T._clean_segments(segments)
    assert all("words" in s for s in cleaned), "word timings lost in _clean_segments"

    reseg = T.resegment(cleaned, max_chars=84, max_duration=5.0)
    assert len(reseg) > len(cleaned), "word-level splitting did not engage"

    srt = T._segments_to_srt(reseg)

    from mazinger.srt import parse_blocks
    blocks = parse_blocks(srt)
    assert len(blocks) == len(reseg)

    # assemble.py places segments on a timeline and assumes no overlap.
    for (_, _, end, _), (_, nxt_start, _, _) in zip(blocks, blocks[1:]):
        assert end <= nxt_start + 1e-6


def test_dispatch_routes_coherex_and_writes_srt(tmp_path, fake_env):
    """The public transcribe() entry point wires the backend end to end."""
    calls, install, _ = fake_env
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"\x00")
    out = tmp_path / "source.srt"

    with install(make_fake_coherex(calls)):
        T.transcribe(str(audio), str(out), method="coherex",
                     language="en", device="cpu", beam_size=None)

    assert out.exists() and out.read_text(encoding="utf-8").strip()
    assert (tmp_path / "source.raw.srt").exists()
    # Detected language sidecar drives the translation stage.
    assert (tmp_path / "source.lang.txt").read_text(encoding="utf-8") == "en"


def test_initial_prompt_is_ignored_with_warning(tmp_path, fake_env, caplog):
    """Cohere Transcribe has no initial_prompt — say so instead of pretending."""
    calls, install, _ = fake_env
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"\x00")

    with install(make_fake_coherex(calls)):
        with caplog.at_level("WARNING"):
            T.transcribe(str(audio), str(tmp_path / "o.srt"), method="coherex",
                         language="en", device="cpu", beam_size=None,
                         initial_prompt="Kubernetes, Docker, gRPC")

    assert "initial_prompt is not supported" in caplog.text


# ── Source-language routing through the dub pipeline ────────────────────────

def test_dub_forwards_source_language_to_asr():
    """CohereX cannot detect language, so dub() must forward --source-language."""
    import inspect
    from mazinger.pipeline import MazingerDubber

    src = inspect.getsource(MazingerDubber.dub)
    call = src[src.index("transcribe.transcribe("):]
    call = call[:call.index("\n            )")]
    assert "language=_asr_language" in call
    assert "vad_method=vad_method" in call


def test_unknown_source_language_does_not_force_english():
    """lang_code_from_name() falls back to 'en'; that must not reach the ASR."""
    import inspect
    from mazinger.pipeline import MazingerDubber

    src = inspect.getsource(MazingerDubber.dub)
    assert "lang_name_from_code(_asr_language) != source_language" in src


# ── Opt-in integration test (real model, real audio) ────────────────────────

FIXTURE = os.path.join(os.path.dirname(__file__), "youtube-short.mp4")


@pytest.mark.skipif(
    not os.environ.get("MAZINGER_TEST_COHEREX"),
    reason="Set MAZINGER_TEST_COHEREX=1 (plus a CUDA GPU and HF_TOKEN) to run "
           "the real CohereX backend against tests/youtube-short.mp4.",
)
def test_real_coherex_end_to_end(tmp_path):
    """Full stage-2 run: extract audio -> CohereX -> SRT, on the real fixture.

    Requires a CUDA GPU and HuggingFace access to the gated Cohere Transcribe
    and pyannote repositories.
    """
    import subprocess

    from mazinger.srt import parse_blocks

    audio = tmp_path / "audio.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", FIXTURE, "-vn", str(audio)],
        check=True,
    )

    out = tmp_path / "source.srt"
    T.transcribe(
        str(audio), str(out),
        method="coherex",
        language=os.environ.get("MAZINGER_TEST_COHEREX_LANG", "en"),
        device="cuda",
        beam_size=None,
    )

    blocks = parse_blocks(out.read_text(encoding="utf-8"))
    assert blocks, "CohereX produced no subtitle blocks"

    # Timings must be sane: inside the clip, ordered, non-overlapping.
    duration = 33.3
    for _, start, end, text in blocks:
        assert 0 <= start < end <= duration + 1.0
        assert text.strip()
    for (_, _, end, _), (_, nxt, _, _) in zip(blocks, blocks[1:]):
        assert end <= nxt + 1e-6


# ── Language coverage ───────────────────────────────────────────────────────

def test_every_coherex_language_maps_to_a_mazinger_language():
    """Codes must round-trip, or the pipeline cannot name the source language."""
    from mazinger.translate import lang_name_from_code, lang_code_from_name
    for code in T.COHEREX_LANGUAGES:
        name = lang_name_from_code(code)
        assert name, f"no Mazinger language name for CohereX code {code!r}"
        assert lang_code_from_name(name) == code
