"""Studio's "Show CLI command" gives a command that dubs exactly as Studio does.

Each case runs the Studio's dub path and the generated ``mazinger dub``
command against a stand-in ``MazingerDubber`` and compares every argument
both hand to it, defaults included.
"""

from __future__ import annotations

import inspect
import os
import subprocess
from types import SimpleNamespace

import pytest

import mazinger
import mazinger.pipeline
import mazinger.profiles
from mazinger.pipeline import MazingerDubber
from mazinger.studio import cli_command as CC
from mazinger.studio import pipeline as SP

KEY = "sk-test-key"

#: Studio's defaults, as the form opens.
DEFAULTS = dict(
    source_type="YouTube URL", url="https://youtu.be/abc", uploaded_file=None, local_path=None,
    cookies_text="",
    target_language="English", voice_type="Auto-Clone", voice_theme_label="Narrator — Male",
    voice_preset="abubakr", voice_file=None, voice_script_text="",
    llm_provider="Ollama (Local — Free)", ollama_model="qwen3.5:2b-q8_0", openai_key="",
    api_base_url="https://api.openai.com/v1", llm_model="gpt-4.1",
    quality="High (best)", start_time="", end_time="",
    transcribe_method="CohereX (local GPU, 14 languages)", whisper_model="",
    source_language="Auto-detect", words_per_second=0.0, duration_budget=0.85,
    translate_technical=False, use_translation_model=False,
    tts_engine="Qwen3-TTS", tts_dtype="bfloat16",
    tempo_mode="Sync", max_tempo=1.5, segment_mode="Long (default)",
    loudness_match=True, mix_background=True, background_volume=1.0,
    output_type="Dubbed Audio", force_reset=False, stream_llm=False,
    youtube_subs=False, user_instructions="", llm_instructions="", fit_check=True,
)

OPENAI = dict(llm_provider="OpenAI (Cloud)", openai_key=KEY)

CASES = {
    "defaults": {},
    "voice theme on omnivoice": dict(voice_type="Voice Theme", voice_theme_label="Warm — Female",
                                     tts_engine="OmniVoice", target_language="Hindi"),
    "voice theme on qwen": dict(voice_type="Voice Theme", voice_theme_label="Kid — Boy"),
    "preset voice": dict(voice_type="Preset Voice", voice_preset="daheeh-v1"),
    "custom voice": dict(voice_type="Custom Voice", voice_file="/tmp/gradio/voice.wav",
                         voice_script_text="  Hello there, it's me.\nSecond line.  "),
    "openai everything changed": dict(
        **OPENAI, api_base_url="https://example.test/v1", llm_model="gpt-5",
        quality="Low (360p)", start_time="00:01:30", end_time="300",
        transcribe_method="OpenAI Whisper (cloud)", whisper_model="whisper-1",
        source_language="Arabic", target_language="Spanish",
        words_per_second=2.7, duration_budget=0.7, translate_technical=True,
        tts_dtype="float16", tempo_mode="Dynamic", max_tempo=1.25,
        segment_mode="Short", loudness_match=False, mix_background=False,
        background_volume=0.35, force_reset=True, youtube_subs=True, fit_check=False,
        user_instructions="Keep culinary terms in Italian.\nPrefer 'tu'.",
        llm_instructions="Never use dialect; it's formal.",
    ),
    "openai with empty overrides": dict(**OPENAI, api_base_url="", llm_model=""),
    "ollama drops cloud whisper": dict(transcribe_method="OpenAI Whisper (cloud)"),
    "translation model": dict(use_translation_model=True),
    "tempo off, auto segments": dict(tempo_mode="Off", segment_mode="Auto", background_volume=0.15),
    "tempo fixed": dict(tempo_mode="Fixed", mix_background=False),
    "chinese": dict(target_language="Chinese", source_language="Cantonese",
                    transcribe_method="Faster Whisper (local GPU)"),
    "local path": dict(source_type="Local Path", local_path="{video}"),
    "upload": dict(source_type="Upload File", uploaded_file=["{video}"]),
    "cookies": dict(cookies_text="# Netscape HTTP Cookie File\n.youtube.com\tTRUE"),
}


class _Recorder:
    """Stands in for ``MazingerDubber``; records what it was built and run with."""

    calls: list[tuple[dict, dict]] = []

    def __init__(self, **init):
        self._init = init

    def dub(self, **kw):
        type(self).calls.append((self._init, kw))
        return SimpleNamespace(summary=lambda: "")


def _bound(fn, kw: dict) -> dict:
    """Every parameter of *fn* as it would be called with *kw*, defaults filled in."""
    sig = inspect.signature(fn)
    ba = sig.bind(None, **kw)
    ba.apply_defaults()
    args = dict(ba.arguments)
    args.pop(next(iter(sig.parameters)))  # self
    return args


def _normalise(init: dict, dub: dict) -> tuple[dict, dict]:
    init = _bound(MazingerDubber.__init__, init)
    dub = _bound(MazingerDubber.dub, dub)
    init["base_dir"] = os.path.abspath(init["base_dir"])
    # Studio passes no model for an empty box, the CLI its default; both end
    # up with MazingerDubber's own fallback.
    init["llm_model"] = init["llm_model"] or os.environ.get("OPENAI_MODEL") or "gpt-4.1"
    # The CLI leaves the beam size unset; transcription then uses 5.
    if dub["beam_size"] is None:
        dub["beam_size"] = 5
    # Studio writes the cookies box to a temp file, the command names one.
    if dub["cookies"]:
        dub["cookies"] = "<cookies file>"
    return init, dub


@pytest.fixture
def env(monkeypatch, tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"x")
    _Recorder.calls = []
    monkeypatch.setattr(mazinger, "MazingerDubber", _Recorder)  # Studio's import
    monkeypatch.setattr(mazinger.pipeline, "MazingerDubber", _Recorder)  # the CLI's
    monkeypatch.setattr(mazinger.profiles, "fetch_profile",
                        lambda name, *a, **k: (f"/profiles/{name}/voice.wav",
                                               f"/profiles/{name}/script.txt"))
    monkeypatch.setattr("mazinger.gpu.release_idle", lambda: None)
    _sleep = SP.time.sleep
    monkeypatch.setattr(SP.time, "sleep", lambda s: _sleep(0.01))  # Studio polls every 2 s
    monkeypatch.setattr(SP, "ensure_ollama", lambda *a, **k: None)
    monkeypatch.setattr(SP, "check_ollama_health", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("MAZINGER_TRANSLATION_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    return str(video)


def _settings(case: dict, video: str) -> dict:
    s = {**DEFAULTS, **case}
    if s["local_path"]:
        s["local_path"] = s["local_path"].format(video=video)
    if s["uploaded_file"]:
        s["uploaded_file"] = [p.format(video=video) for p in s["uploaded_file"]]
    return s


def _run_studio(settings: dict) -> tuple[dict, dict]:
    for _ in SP.run_dubbing(**settings):
        pass
    assert len(_Recorder.calls) == 1, "Studio did not reach MazingerDubber.dub"
    return _Recorder.calls.pop()


def shell_argvs(script: str) -> list[list[str]]:
    """Run *script* in bash with ``mazinger`` stubbed; the argv of each call."""
    out = subprocess.run(
        ["bash", "-c", 'mazinger() { printf "%s\\0" "$@"; printf "\\1"; }\n' + script],
        capture_output=True, text=True, check=True,
    ).stdout
    return [call.split("\0")[:-1] for call in out.split("\1")[:-1]]


def _run_cli(script: str, monkeypatch) -> tuple[dict, dict]:
    from mazinger.cli import _build_parser, _dub

    monkeypatch.setenv("OPENAI_API_KEY", KEY)  # the script's `export`
    (argv,) = shell_argvs(script)
    args = _build_parser().parse_args(argv)
    assert args.command == "dub"
    _dub.handler(args)
    assert len(_Recorder.calls) == 1, "the command did not reach MazingerDubber.dub"
    return _Recorder.calls.pop()


@pytest.mark.parametrize("name", list(CASES))
def test_command_dubs_with_the_studio_arguments(name, env, monkeypatch):
    settings = _settings(CASES[name], env)
    studio = _normalise(*_run_studio(settings))

    script, notes = CC.build_cli_command(**settings)
    assert script, notes
    assert KEY not in script
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cli = _normalise(*_run_cli(script, monkeypatch))

    assert cli[0] == studio[0], "MazingerDubber() arguments differ"
    assert cli[1] == studio[1], "dub() arguments differ"


class TestScript:
    def test_takes_the_run_dubbing_inputs(self):
        """app.py wires one input list to both Start and Show CLI command."""
        def params(fn):
            return [n for n, p in inspect.signature(fn).parameters.items()
                    if p.kind is p.POSITIONAL_OR_KEYWORD]
        assert params(CC.build_cli_command) == params(SP.run_dubbing)

    def test_several_sources_give_one_command_each(self):
        script, notes = CC.build_cli_command(**{**DEFAULTS, "url": "https://a\nhttps://b"})
        assert [argv[1] for argv in shell_argvs(script)] == ["https://a", "https://b"]
        assert any("2 sources" in n for n in notes)

    def test_the_openai_key_is_a_placeholder(self):
        script, notes = CC.build_cli_command(**{**DEFAULTS, **OPENAI})
        assert KEY not in script
        assert f'# Needs your OpenAI key: export OPENAI_API_KEY="{CC.KEY_PLACEHOLDER}"' in script
        # Pasting the script leaves a key the shell already has alone.
        assert "\nexport " not in "\n" + script
        assert any("never copied" in n for n in notes)

    def test_ollama_needs_no_key_but_a_running_server(self):
        script, notes = CC.build_cli_command(**{**DEFAULTS, "use_translation_model": True})
        assert "export OPENAI_API_KEY" not in script
        assert any("ollama pull qwen3.5:2b-q8_0" in n and "ollama pull translategemma" in n
                   for n in notes)

    def test_fixed_tempo_is_explained(self):
        _, notes = CC.build_cli_command(**{**DEFAULTS, "tempo_mode": "Fixed"})
        assert any("--fixed-tempo" in n for n in notes)

    def test_a_missing_source_gives_the_studio_error(self):
        script, notes = CC.build_cli_command(**{**DEFAULTS, "url": ""})
        assert script == "" and "URL" in notes[0]

    def test_a_custom_voice_needs_its_transcript(self):
        script, notes = CC.build_cli_command(**{
            **DEFAULTS, "voice_type": "Custom Voice", "voice_file": "/v.wav",
            "voice_script_text": " ",
        })
        assert script == "" and "transcript" in notes[0]

    def test_every_line_but_the_last_continues(self):
        script, _ = CC.build_cli_command(**{**DEFAULTS, "llm_instructions": "a\nb"})
        assert len(script.split(" \\\n")) > 5
        (argv,) = shell_argvs(script)
        assert argv[argv.index("--llm-instructions") + 1] == "a\nb"


class TestSubtitleOutputs:
    def _parse(self, script):
        from mazinger.cli import _build_parser
        (argv,) = shell_argvs(script)
        return _build_parser().parse_args(argv)

    def test_transcription_subtitles(self):
        script, notes = CC.build_cli_command(**{
            **DEFAULTS, "output_type": "Transcription Subtitles",
            "source_language": "Arabic", "whisper_model": "large-v3",
            "transcribe_method": "Faster Whisper (local GPU)",
        })
        args = self._parse(script)
        assert (args.command, args.method, args.model, args.language, args.asr_review) == (
            "transcribe", "faster-whisper", "large-v3", "ar", True)
        assert args.openai_base_url == "http://localhost:11434/v1"
        assert "Approximate" in script and "Approximate" in notes[0]

    def test_translated_subtitles(self):
        script, notes = CC.build_cli_command(**{
            **DEFAULTS, **OPENAI, "output_type": "Translated Subtitles",
            "target_language": "French", "duration_budget": 0.7, "start_time": "10",
        })
        args = self._parse(script)
        assert (args.command, args.target_language, args.duration_budget) == (
            "translate", "French", 0.7)
        assert args.llm_model == "gpt-4.1"
        assert "Approximate" in notes[0]
        assert any("start/end time" in n for n in notes)
