<p align="center">
  <img src="https://raw.githubusercontent.com/bakrianoo/mazinger/refs/heads/master/docs/assets/main-logo-refined.png" alt="Mazinger Dubber" width="240" height="240" />
</p>

<h1 align="center">Mazinger Dubber</h1>

<p align="center">
  End-to-end video dubbing pipeline. Download a video, transcribe it, translate the subtitles, clone a voice, and produce a fully dubbed audio or video — in one command.
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/bakrianoo/mazinger-dubber-profiles/blob/main/promo-demo/mazinger-promo.mp4">
    <img src="https://raw.githubusercontent.com/bakrianoo/mazinger/refs/heads/master/docs/assets/thumbnail-demo.png" alt="Watch demo video" width="640" /><br/>
    ▶️ Watch Demo Video (with audio)
  </a>
</p>

---

## 🚀 Get Started in 2 Steps

**Prerequisites:** Python 3.10+ and `ffmpeg` on your `PATH` (`apt install ffmpeg` / `brew install ffmpeg`).

### 1. Install

```bash
pip install "mazinger[all]"
```

### 2. Launch the Web UI

```bash
mazinger web --with-ollama --with-faster-whisper
```

A local URL opens in your browser. Paste a video link, pick a voice, and click **Start**. The flags install a free local LLM (Ollama) and download the speech-recognition model on first run — no API keys required.

> 💡 **No GPU?** Run it on a free Colab T4 in two clicks:
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bakrianoo/mazinger/blob/master/notebooks/mazinger_colab.ipynb)

> Prefer the command line or Python? Skip ahead to [Common Tasks](#-common-tasks) or the [Python API](#-python-api).

---

## ✨ What You Get

`mazinger[all]` is a single, GPU-friendly install that includes everything needed for the full pipeline and the Studio web UI:

| Capability | Engine |
|---|---|
| Local transcription (default) | Faster Whisper |
| Cloud transcription (no GPU) | Deepgram Nova 3 — $200 free credit |
| Voice-cloned TTS | Qwen3-TTS, OmniVoice (24 languages) |
| Background-audio separation | Demucs |
| Web UI | Gradio (Mazinger Studio) |
| Local LLM (optional) | Ollama (auto-installed by `mazinger web`) |

Need Chatterbox, MLX (Apple Silicon), or a minimal install? See the [Installation Guide](docs/installation.md).

---

## 🛠️ Common Tasks

### Dub a video — auto-clone the original speaker

No voice files needed. Mazinger picks the best 20–60 s of the source as the cloning reference.

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --target-language Spanish
```

### Dub with a ready-made voice theme

16 built-in themes — no files, no profile downloads.

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --voice-theme narrator-m \
    --target-language Spanish
```

Themes: `narrator-m/f` · `young-m/f` · `deep-m/f` · `warm-m/f` · `news-m/f` · `storyteller-m/f` · `kid-m/f` · `teen-m/f`

### Dub with a HuggingFace voice profile

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --target-language Arabic
```

Available out-of-the-box: `abubakr` · `daheeh-v1` · `3b1b` · `italian-v1` · `morgan-freeman` · `trump-v1` — full list in [Voice Profiles](docs/voice-profiles.md#available-profiles).

### Dub with your own voice sample

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --voice-sample speaker.m4a \
    --voice-script speaker_transcript.txt \
    --target-language Spanish
```

### Output a video with burned-in subtitles

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --voice-theme narrator-m \
    --target-language Arabic \
    --output-type video \
    --embed-subtitles \
    --subtitle-google-font "Noto Sans Arabic"
```

See [Subtitle Styling](docs/subtitle-styling.md) for fonts, colors, positioning, and RTL options.

### Use Deepgram instead of a local GPU

```bash
export DEEPGRAM_API_KEY=your_key_here
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --transcribe-method deepgram \
    --voice-theme narrator-m \
    --target-language English
```

### Run a single stage

```bash
mazinger download   "https://youtube.com/watch?v=VIDEO_ID"
mazinger transcribe ./output/projects/my-video/source/audio.mp3 -o subs.srt
mazinger translate  --srt subs.srt --target-language French -o translated.srt
mazinger subtitle   video.mp4 --srt translated.srt -o output.mp4
```

Every stage caches its output. Re-running resumes where it stopped. Full command list in the [CLI Reference](docs/cli-reference.md).

---

## 🐍 Python API

```python
from mazinger import MazingerDubber

dubber = MazingerDubber(openai_api_key="sk-...", base_dir="./output")

proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    voice_theme="narrator-m",
    target_language="Spanish",
    output_type="video",
)

print(proj.final_video)   # ./output/projects/<slug>/tts/dubbed.mp4
```

Full reference: [Python API](docs/python-api.md).

---

## 🔧 How It Works

Mazinger chains ten resumable stages: **Download → Transcribe → Thumbnails → Describe → Review → Translate → Re-segment → Speak → Assemble → Subtitle**. Every stage runs standalone or as part of the full pipeline; completed stages and individual TTS segments are cached and skipped on re-runs.

See the [Pipeline Overview](docs/pipeline.md) for a diagram and the data flow between stages.

---

## 📚 Documentation

| Topic | What's inside |
|---|---|
| [Installation](docs/installation.md) | All install options, advanced extras, Apple Silicon, Colab, uv overrides |
| [Quick Start](docs/quick-start.md) | More copy-paste workflows |
| [Pipeline Overview](docs/pipeline.md) | The ten stages, data flow, resume behavior |
| [CLI Reference](docs/cli-reference.md) | Every command, flag, and default |
| [Python API](docs/python-api.md) | Classes, functions, parameters |
| [Voice Profiles](docs/voice-profiles.md) | Using, creating, uploading profiles |
| [Subtitle Styling](docs/subtitle-styling.md) | Fonts, colors, positioning, RTL, Google Fonts |
| [Configuration](docs/configuration.md) | Env vars, caching, tempo, LLM usage tracking |
| [Project Structure](docs/project-structure.md) | Output directory layout |
| [YouTube Cookies](docs/youtube-cookies.md) | Cookies for age-restricted / region-locked videos |

---

## License

MIT
