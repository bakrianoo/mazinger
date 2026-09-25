"""Shared fixtures: a small finished project on disk and fake models.

The Editor and the run-record tests need a project that looks like the output
of a completed dub, plus stand-ins for the TTS model, the ASR backend and the
LLM client so nothing heavy is loaded and no network is touched.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest
import soundfile as sf

from mazinger.paths import ProjectPaths
from mazinger.srt import build
from mazinger.tts import TTSWrapper

SR = 24_000


# ---------------------------------------------------------------------------
#  File helpers
# ---------------------------------------------------------------------------

def write_tone(path: str, seconds: float, *, sr: int = SR, freq: float = 220.0,
               amp: float = 0.3) -> str:
    """Write a mono sine tone.  The format is WAV whatever the extension."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (amp * np.sin(2 * np.pi * freq * t)).astype("float32"), sr, format="WAV")
    return path


def write_srt(path: str, entries: list[tuple[float, float, str]]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build(entries))
    return path


# ---------------------------------------------------------------------------
#  A finished project
# ---------------------------------------------------------------------------

# Source transcription: four entries.
SOURCE_ENTRIES = [
    (0.0, 2.0, "Hello and welcome."),
    (2.0, 4.5, "Today we talk about"),
    (4.5, 7.0, "neural networks."),
    (7.5, 10.0, "Let's begin."),
]
# Translation, 1:1 with the source.
TRANSLATED_RAW_ENTRIES = [
    (0.0, 2.0, "Hola y bienvenidos."),
    (2.0, 4.5, "Hoy hablamos de"),
    (4.5, 7.0, "redes neuronales."),
    (7.5, 10.0, "Empecemos."),
]
# Re-segmented: the middle two entries were merged into one chunk.
FINAL_ENTRIES = [
    (0.0, 2.0, "Hola y bienvenidos."),
    (2.0, 7.0, "Hoy hablamos de redes neuronales."),
    (7.5, 10.0, "Empecemos."),
]


@pytest.fixture
def mini_project(tmp_path) -> ProjectPaths:
    """A completed Spanish dub: source audio, SRTs, segment WAVs, output."""
    proj = ProjectPaths("demo", base_dir=str(tmp_path), target_language="Spanish").ensure_dirs()

    write_tone(proj.audio, 10.0)
    write_srt(proj.source_raw_srt, SOURCE_ENTRIES)
    write_srt(proj.source_srt, SOURCE_ENTRIES)
    write_srt(proj.translated_raw_srt, TRANSLATED_RAW_ENTRIES)
    write_srt(proj.final_srt, FINAL_ENTRIES)
    for i, (start, end, _) in enumerate(FINAL_ENTRIES, 1):
        write_tone(os.path.join(proj.tts_segments_dir, f"seg_{i:04d}.wav"), (end - start) * 0.9)
    write_tone(proj.final_audio, 10.0)
    return proj


# ---------------------------------------------------------------------------
#  Fake models
# ---------------------------------------------------------------------------

class FakeVoicePrompt(TTSWrapper):
    """A ``TTSWrapper`` that speaks 0.1 s of tone per word."""

    engine = "fake"

    def __init__(self, sr: int = SR) -> None:
        self.sr = sr
        self.calls: list[tuple[str, str]] = []

    def synthesize(self, text: str, language: str = "English"):
        self.calls.append((text, language))
        seconds = max(0.2, 0.1 * len(text.split()))
        t = np.arange(int(seconds * self.sr)) / self.sr
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), self.sr

    def unload(self) -> None:
        pass


class FakeLLMClient:
    """OpenAI-shaped client whose replies come from *responder(messages)*."""

    def __init__(self, responder: Callable[[list[dict]], str] | None = None) -> None:
        self.responder = responder or (lambda messages: "{}")
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        content = self.responder(kwargs.get("messages", []))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class FakeTranscriber:
    """Replaces ``transcribe.transcribe``.

    Writes the same files the real one does: ``<out>``, ``<base>.raw.srt`` and
    the ``<base>.lang.txt`` detected-language sidecar.
    """

    def __init__(self, entries: list[tuple[float, float, str]] | None = None,
                 detected_lang: str = "en") -> None:
        self.entries = entries or SOURCE_ENTRIES
        self.detected_lang = detected_lang
        self.calls: list[dict] = []

    def __call__(self, audio_path: str, output_path: str, **kwargs) -> str:
        self.calls.append({"audio_path": audio_path, **kwargs})
        base, ext = os.path.splitext(output_path)
        write_srt(f"{base}.raw{ext}", self.entries)
        with open(f"{base}.lang.txt", "w", encoding="utf-8") as fh:
            fh.write(self.detected_lang)
        write_srt(output_path, self.entries)
        return output_path


@pytest.fixture
def fake_voice_prompt() -> FakeVoicePrompt:
    return FakeVoicePrompt()


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def fake_transcriber() -> FakeTranscriber:
    return FakeTranscriber()
