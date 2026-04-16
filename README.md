<p align="center">
  <img src="https://raw.githubusercontent.com/bakrianoo/mazinger/refs/heads/master/docs/assets/main-logo-refined.png" alt="Mazinger Dubber" width="320" height="320" />
</p>

<h1 align="center">Mazinger Dubber</h1>

<p align="center">
  End-to-end video dubbing pipeline. Download a video, transcribe it, translate the subtitles, clone a voice, and produce a fully dubbed audio or video file — in one command.
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/bakrianoo/mazinger-dubber-profiles/blob/main/promo-demo/mazinger-promo.mp4">
    <img src="https://raw.githubusercontent.com/bakrianoo/mazinger/refs/heads/master/docs/assets/thumbnail-demo.png" alt="Watch demo video" width="720" /><br/>
    ▶️ Watch Demo Video (with audio)
  </a>
</p>

## What It Does

Mazinger chains ten stages into a single pipeline:

1. **Download** — fetch a video from a URL or ingest a local file, extract the audio track
2. **Transcribe** — convert speech to SRT subtitles (OpenAI Whisper API, faster-whisper, WhisperX, MLX Whisper, or Deepgram Nova 3)
3. **Thumbnails** — use an LLM to pick key frames from the video for visual context
4. **Describe** — analyze the transcript and thumbnails to produce a structured summary (title, key points, keywords)
5. **Review** — optionally refine ASR output: fix typos, reshape punctuation, and convert technical terms to English
6. **Translate** — translate the SRT into another language with duration-aware word budgets
7. **Re-segment** — merge fragments and split oversized subtitles for readability
8. **Speak** — synthesize voice-cloned speech for every subtitle entry (Qwen3-TTS, Chatterbox, OmniVoice, or MLX), with 16 pre-defined voice themes or your own voice sample
9. **Assemble** — place each audio segment on the original timeline with optional tempo adjustment, loudness matching, and background audio mixing
10. **Subtitle** — burn styled subtitles into the video and/or mux the new audio track

Every stage can run independently or as part of the full pipeline. Interrupted runs resume automatically — completed stages and individual TTS segments are cached and skipped.

## Prerequisites

- Python 3.10 or later
- ffmpeg installed and on `PATH` (`apt install ffmpeg` / `brew install ffmpeg`)
- An OpenAI API key for LLM-powered stages (transcription, translation, thumbnails, description)
- A CUDA GPU for local transcription and TTS (not needed for cloud-only workflows)
- Apple Silicon (M1/M2/M3/M4/M5) for MLX-accelerated TTS and transcription (optional)

## Installation

The base install covers download, transcription (cloud), thumbnails, description, translation, re-segmentation, and subtitle embedding. No GPU needed.

```bash
pip install mazinger
```

Add local transcription or TTS as optional extras:

```bash
# Local transcription
pip install "mazinger[transcribe-faster]"      # faster-whisper (default, recommended)
pip install "mazinger[transcribe-whisperx]"    # WhisperX (optional, word-level alignment)

# Cloud transcription (no GPU needed)
pip install "mazinger[transcribe-deepgram]"    # Deepgram Nova 3 (cloud, free $200 credit)

# Voice synthesis
pip install "mazinger[tts]"                    # Qwen3-TTS (voice sample + transcript)
pip install "mazinger[tts-chatterbox]"         # Chatterbox (voice sample only, emotion control)
pip install "mazinger[tts-omnivoice]"          # OmniVoice (24 languages, zero-shot cloning)
pip install "mazinger[tts-mlx]"                # MLX Qwen3-TTS (Apple Silicon)

# MLX transcription (Apple Silicon)
pip install "mazinger[transcribe-mlx]"         # MLX Whisper (Apple Silicon)

# Full bundles
pip install "mazinger[all-qwen]"              # faster-whisper + Qwen3-TTS
pip install "mazinger[all-chatterbox]"        # faster-whisper + Chatterbox
pip install "mazinger[all-omnivoice]"         # faster-whisper + OmniVoice
pip install "mazinger[all-mlx]"               # MLX Whisper + MLX Qwen3-TTS
```

> Qwen and Chatterbox require different `transformers` versions and cannot share an environment.
> WhisperX is available as an optional extra but is not installed by default due to complex dependencies.

See the [Installation Guide](docs/installation.md) for venv recipes, Colab setup, and uv overrides.

## Quick Start

### Launch the Web UI

The easiest way to use Mazinger is through the built-in Gradio web interface. This installs Ollama (local LLM) and downloads the Faster Whisper model automatically:

```bash
pip install "mazinger[all-omnivoice,web]"
mazinger web --with-ollama --with-faster-whisper
```

A local and public URL will appear — open it in your browser, paste a video URL, pick a voice, and go.

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

### Use a voice theme (no files needed)

Choose from 16 pre-defined voice themes — no voice sample or profile download required:

`narrator-m/f` · `young-m/f` · `deep-m/f` · `warm-m/f` · `news-m/f` · `storyteller-m/f` · `kid-m/f` · `teen-m/f`

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --voice-theme narrator-m \
    --target-language Spanish
```

List all themes with `mazinger profile list`. Generate a reusable profile with `mazinger profile generate`. See [Voice Profiles](docs/voice-profiles.md) for details.

### Auto-clone the original speaker's voice

When no voice option is provided, Mazinger automatically clones the speaker directly from the source audio. The pipeline picks the best 20–60 s segment from the transcription and uses it as the cloning reference — no files or configuration needed.

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --target-language Spanish
```

```python
proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    target_language="Spanish",
)
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

### Cloud transcription with Deepgram (no GPU required)

Deepgram Nova 3 offers strong multilingual quality (including Arabic) and gives you **$200
in free credits on sign-up with no credit card required** — enough to transcribe many hours
of audio for free. Get your key at [deepgram.com](https://deepgram.com), then:

```bash
export DEEPGRAM_API_KEY=your_key_here

# Standalone transcription
mazinger transcribe "https://youtube.com/watch?v=VIDEO_ID" \
    --method deepgram --language ar -o subs.srt

# Full dubbing pipeline with Deepgram for STT
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --transcribe-method deepgram \
    --voice-theme narrator-m \
    --target-language English
```

### Python API

```python
from mazinger import MazingerDubber

dubber = MazingerDubber(openai_api_key="sk-...", base_dir="./output")

# With a voice theme (simplest)
proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    voice_theme="narrator-m",
    target_language="Spanish",
    output_type="video",
)

# Auto-clone the speaker's voice (no voice option needed)
proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    target_language="Spanish",
)

# Or with explicit voice files
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
| [YouTube Cookies](docs/youtube-cookies.md) | How to export and pass cookies for age-restricted or region-locked videos |

## License

MIT
