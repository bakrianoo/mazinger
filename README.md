<p align="center">
  <img src="https://raw.githubusercontent.com/bakrianoo/mazinger/refs/heads/master/docs/assets/main-logo-refined.png" alt="Mazinger Dubber" width="320" height="320" />
</p>

<h1 align="center">Mazinger Dubber</h1>

<p align="center">
  End-to-end video dubbing pipeline. Download a video, transcribe it, translate the subtitles, clone a voice, and produce a fully dubbed audio or video file — in one command.
</p>

## What It Does

Mazinger chains nine stages into a single pipeline:

1. **Download** — fetch a video from a URL or ingest a local file, extract the audio track
2. **Transcribe** — convert speech to SRT subtitles (OpenAI Whisper API, faster-whisper, or WhisperX)
3. **Thumbnails** — use an LLM to pick key frames from the video for visual context
4. **Describe** — analyze the transcript and thumbnails to produce a structured summary (title, key points, keywords)
5. **Translate** — translate the SRT into another language with duration-aware word budgets
6. **Re-segment** — merge fragments and split oversized subtitles for readability
7. **Speak** — synthesize voice-cloned speech for every subtitle entry (Qwen3-TTS or Chatterbox)
8. **Assemble** — place each audio segment on the original timeline with optional tempo adjustment, loudness matching, and background audio mixing
9. **Subtitle** — burn styled subtitles into the video and/or mux the new audio track

Every stage can run independently or as part of the full pipeline. Interrupted runs resume automatically — completed stages and individual TTS segments are cached and skipped.

## Prerequisites

- Python 3.10 or later
- ffmpeg installed and on `PATH` (`apt install ffmpeg` / `brew install ffmpeg`)
- An OpenAI API key for LLM-powered stages (transcription, translation, thumbnails, description)
- A CUDA GPU for local transcription and TTS (not needed for cloud-only workflows)

## Installation

The base install covers download, transcription (cloud), thumbnails, description, translation, re-segmentation, and subtitle embedding. No GPU needed.

```bash
pip install mazinger
```

Add local transcription or TTS as optional extras:

```bash
# Local transcription
pip install "mazinger[transcribe-faster]"      # faster-whisper (Chatterbox-compatible)
pip install "mazinger[transcribe-whisperx]"    # WhisperX (best word-level alignment)

# Voice synthesis
pip install "mazinger[tts]"                    # Qwen3-TTS (voice sample + transcript)
pip install "mazinger[tts-chatterbox]"         # Chatterbox (voice sample only, emotion control)

# Full bundles
pip install "mazinger[all-qwen]"              # WhisperX + Qwen3-TTS
pip install "mazinger[all-chatterbox]"        # faster-whisper + Chatterbox
```

> Qwen and Chatterbox require different `transformers` versions and cannot share an environment.
> WhisperX also conflicts with Chatterbox — pair it with Qwen, or use faster-whisper with Chatterbox.

See the [Installation Guide](docs/installation.md) for venv recipes, Colab setup, and uv overrides.

## Quick Start

### Dub a video in one command

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --voice-sample speaker.m4a \
    --voice-script speaker_transcript.txt \
    --target-language Spanish \
    --base-dir ./output
```

### Use a voice profile instead of local files

Voice profiles are hosted on HuggingFace and downloaded automatically. Several ready-made profiles are available out of the box:

`abubakr` · `daheeh-v1` · `3b1b` · `italian-v1` · `morgan-freeman` · `trump-v1`

See the full list with descriptions in the [Available Voice Profiles](docs/voice-profiles.md#available-profiles) doc.

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --target-language Arabic
```

### Produce a video with burned subtitles

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --output-type video \
    --embed-subtitles \
    --subtitle-google-font "Noto Sans Arabic" \
    --subtitle-font-size 24
```

### Run a single stage

Every stage has its own sub-command:

```bash
mazinger download   "https://youtube.com/watch?v=VIDEO_ID" --base-dir ./output
mazinger slice      "https://youtube.com/watch?v=VIDEO_ID" --start 00:01:00 --end 00:04:00
mazinger transcribe ./output/projects/my-video/source/audio.mp3 -o subs.srt
mazinger translate  --srt subs.srt --target-language French -o translated.srt
mazinger subtitle   video.mp4 --srt translated.srt -o output.mp4
```

### Python API

```python
from mazinger import MazingerDubber

dubber = MazingerDubber(openai_api_key="sk-...", base_dir="./output")

proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    voice_sample="speaker.m4a",
    voice_script="speaker_transcript.txt",
    target_language="Spanish",
    output_type="video",
    embed_subtitles=True,
)

print(proj.final_video)   # ./output/projects/<slug>/tts/dubbed.mp4
```

## Documentation

Full documentation lives in the [`docs/`](docs/) directory:

| Chapter | Contents |
|---------|----------|
| [Installation](docs/installation.md) | All install methods, extras, compatibility matrix, Colab and venv recipes |
| [Quick Start](docs/quick-start.md) | Common workflows with copy-paste examples |
| [Pipeline Overview](docs/pipeline.md) | How the nine stages connect, data flow, and resume behavior |
| [CLI Reference](docs/cli-reference.md) | Every command, flag, and default value |
| [Python API](docs/python-api.md) | Classes, functions, and parameters for programmatic use |
| [Voice Profiles](docs/voice-profiles.md) | Using, creating, and uploading voice profiles |
| [Subtitle Styling](docs/subtitle-styling.md) | Fonts, colors, positioning, RTL support, Google Fonts |
| [Configuration](docs/configuration.md) | Environment variables, caching, tempo control, LLM usage tracking |
| [Project Structure](docs/project-structure.md) | Output directory layout and file naming conventions |

## License

MIT
