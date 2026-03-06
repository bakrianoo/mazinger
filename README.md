# Mazinger Dubber

End-to-end video dubbing pipeline.
Download a video, transcribe it, translate the subtitles, and generate a voice-cloned dubbed audio track — all in one command.

---

## Features

| Stage          | What it does                                          |
|----------------|-------------------------------------------------------|
| **Download**   | Fetch video + extract audio (yt-dlp)                  |
| **Transcribe** | Speech-to-text (OpenAI API / faster-whisper / WhisperX) |
| **Thumbnails** | LLM-selected key frames for visual context            |
| **Describe**   | Structured content analysis (title, summary, keywords)|
| **Translate**  | Context-aware SRT translation with visual grounding   |
| **Re-segment** | Split long subtitles into readable caption blocks     |
| **TTS**        | Voice-cloned speech (Qwen3-TTS or Chatterbox)         |
| **Assemble**   | Time-aligned final audio matching original duration   |

Every stage works **independently** or chained through the `MazingerDubber` class / `mazinger-dubber` CLI.

---

## Prerequisites

- **Python 3.10+**
- **ffmpeg** — `apt install ffmpeg` or `brew install ffmpeg`
- **OpenAI API key** — set `OPENAI_API_KEY` env var (or pass via `--openai-api-key`)
- **CUDA GPU** — needed for local transcription and TTS (optional if using OpenAI transcription only)

---

## Installation

Pick **one** of the bundles below based on the TTS engine you want:

```bash
# Chatterbox TTS + faster-whisper (recommended for voice cloning without transcript)
pip install ".[all-chatterbox]"

# Qwen TTS + WhisperX (recommended for multilingual + best alignment)
pip install ".[all-qwen]"

# Core only (no local TTS, no local transcription — uses OpenAI APIs)
pip install .
```

> **Qwen and Chatterbox cannot coexist** in the same environment (conflicting `transformers` versions).
> See [DOCS.md](DOCS.md#installation-options) for advanced install methods (venv, Colab, extras).

---

## Quick Start

### One command — dub a video

```bash
mazinger-dubber dub "https://youtube.com/watch?v=VIDEO_ID" \
    --voice-sample reference.m4a \
    --voice-script reference_transcript.txt \
    --base-dir ./output
```

Add `--tts-engine chatterbox` to use Chatterbox instead of Qwen.

### Python

```python
from mazinger_dubber import MazingerDubber

dubber = MazingerDubber(openai_api_key="sk-...", base_dir="./output")
proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    voice_sample="reference.m4a",
    voice_script="reference_transcript.txt",
)
print(proj.final_audio)  # ./output/projects/<slug>/tts/dubbed.wav
```

### Run individual stages

Each pipeline stage has its own sub-command:

```bash
mazinger-dubber download   ...
mazinger-dubber transcribe ...
mazinger-dubber thumbnails ...
mazinger-dubber describe   ...
mazinger-dubber translate  ...
mazinger-dubber resegment  ...
mazinger-dubber tts        ...
```

Run any command with `--help` for all options. Full step-by-step guide in [DOCS.md](DOCS.md#step-by-step-usage).

---

## FAQ

**Which TTS engine should I use?**
Use **Chatterbox** if you only have a voice sample (no transcript needed) or want emotion/pacing control.
Use **Qwen** for multilingual support with a reference transcript.

**Do I need a GPU?**
Only for local transcription (`faster-whisper`, `whisperx`) and TTS. If you use `--transcribe-method openai` and an external TTS, a CPU-only machine works fine.

**Can I use Qwen and Chatterbox together?**
No. They require different `transformers` versions. Use separate virtual environments if you need both.

**Which transcription method works with Chatterbox?**
`openai` (cloud) and `faster-whisper` (local). WhisperX has a dependency conflict with Chatterbox.

**Where do the output files go?**
Under `<base-dir>/projects/<slug>/` — organised into `source/`, `transcription/`, `subtitles/`, `thumbnails/`, `analysis/`, and `tts/` folders.

**How do I pass YouTube cookies for age-restricted videos?**
Use `--cookies-from-browser chrome` or `--cookies path/to/cookies.txt` with any command.

**flash-attn won't install — is that a problem?**
No. It's an optional acceleration for TTS. Chatterbox and Qwen both fall back to standard attention automatically.

---

## Documentation

Full installation guides, step-by-step API reference, project structure details, and configuration options:

**[DOCS.md](DOCS.md)**

---

## License

[MIT](LICENSE)
