"""Guards for how ``beam_size`` is resolved per transcription backend.

The CLI default is ``None`` because mlx-whisper decodes by sampling and rejects
any beam size. The beam-search backends need a real integer, though:
faster-whisper forwards the value straight into CTranslate2, whose
``generate()`` raises ``TypeError`` on ``None``. That combination silently broke
the *default* backend for every CLI run that did not pass ``--beam-size``.
"""

import pytest

import mazinger.transcribe as T


class _Sentinel(Exception):
    """Raised by the stub to stop ``transcribe()`` once the call is captured."""


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "audio.mp3"
    path.write_bytes(b"not really audio; the backend is stubbed out")
    return str(path)


@pytest.fixture
def captured(monkeypatch):
    """Capture the kwargs the faster-whisper backend is called with."""
    seen = {}

    def stub(_audio_path, **kwargs):
        seen.update(kwargs)
        raise _Sentinel

    monkeypatch.setattr(T, "_transcribe_faster_whisper", stub)
    return seen


def _run(audio, tmp_path, **kwargs):
    with pytest.raises(_Sentinel):
        T.transcribe(audio, str(tmp_path / "out.srt"), method="faster-whisper", **kwargs)


def test_unset_beam_size_resolves_to_an_int_for_faster_whisper(audio, tmp_path, captured):
    """``None`` must not reach CTranslate2 — it only accepts an int."""
    _run(audio, tmp_path, beam_size=None)

    assert captured["beam_size"] is not None
    assert isinstance(captured["beam_size"], int)
    assert captured["beam_size"] == T.DEFAULT_BEAM_SIZE


def test_explicit_beam_size_is_passed_through_untouched(audio, tmp_path, captured):
    _run(audio, tmp_path, beam_size=3)
    assert captured["beam_size"] == 3


def test_default_beam_size_is_a_positive_int():
    assert isinstance(T.DEFAULT_BEAM_SIZE, int)
    assert T.DEFAULT_BEAM_SIZE >= 1


def test_mlx_whisper_still_rejects_an_explicit_beam_size(audio, tmp_path):
    """Resolving the default must not weaken mlx-whisper's guard."""
    with pytest.raises(ValueError, match="beam_size not supported with mlx-whisper"):
        T.transcribe(
            audio, str(tmp_path / "out.srt"), method="mlx-whisper", beam_size=5,
        )


def test_mlx_whisper_accepts_the_unset_default(audio, tmp_path, monkeypatch):
    """``beam_size=None`` is the value mlx-whisper needs; it must survive."""
    def stub(_audio_path, **kwargs):
        assert "beam_size" not in kwargs, "mlx-whisper takes no beam size"
        raise _Sentinel

    monkeypatch.setattr(T, "_transcribe_mlx_whisper", stub)
    with pytest.raises(_Sentinel):
        T.transcribe(
            audio, str(tmp_path / "out.srt"), method="mlx-whisper", beam_size=None,
        )


def test_cli_leaves_beam_size_unset_so_every_backend_can_resolve_it():
    """The CLI default must stay ``None``; the resolution lives in transcribe()."""
    from mazinger.cli import _build_parser

    args = _build_parser().parse_args(["dub", "video.mp4"])
    assert args.beam_size is None
