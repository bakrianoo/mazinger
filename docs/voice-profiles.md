# Voice Profiles & Custom Voice Cloning

Mazinger can clone any voice to produce dubbed audio. There are four ways to provide a voice:

1. **Auto-clone** — omit all voice flags and Mazinger clones the speaker directly from the source audio (nothing to configure)
2. **Use a voice theme** — pass `--voice-theme <name>` to generate a voice from 16 pre-defined themes (no files needed)
3. **Use a built-in profile** — pass `--clone-profile <name>` and Mazinger downloads the voice sample automatically from HuggingFace (or point it to a local directory)
4. **Use your own voice files** — pass `--voice-sample` and `--voice-script` with local files you recorded yourself

All four approaches work with the full `dub` pipeline and the standalone `speak` sub-command.

---

## Option 1: Auto-Clone (Simplest)

When no voice option is provided, Mazinger automatically extracts a 20–60 second segment from the source audio with the highest word density and uses it — along with the corresponding transcript — as the voice cloning reference. The cloned profile is saved to the project's `voice_profile/` directory and reused on subsequent runs.

### CLI

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --target-language Spanish
```

### Python

```python
from mazinger import MazingerDubber

dubber = MazingerDubber(openai_api_key="sk-...", base_dir="./output")

proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    target_language="Spanish",
)
```

### Requirements

- The source audio must be at least 20 seconds long
- The selected segment must contain at least 20 words
- If the source is too short or too sparse, provide a voice explicitly with one of the options below

---

## Option 2: Use a Voice Theme (Easiest)

Pre-defined voice themes are the simplest way to get started. No recording or file download needed — Mazinger generates a reference voice automatically using the Qwen3-TTS VoiceDesign model.

### Available Themes

| Theme | Gender | Style |
|-------|--------|-------|
| `narrator-m` | Male | Professional narrator |
| `narrator-f` | Female | Professional narrator |
| `young-m` | Male | Youthful, dynamic |
| `young-f` | Female | Youthful, dynamic |
| `deep-m` | Male | Deep, resonant |
| `deep-f` | Female | Deep, resonant |
| `warm-m` | Male | Warm, friendly |
| `warm-f` | Female | Warm, friendly |
| `news-m` | Male | News reader |
| `news-f` | Female | News reader |
| `storyteller-m` | Male | Engaged storytelling |
| `storyteller-f` | Female | Engaged storytelling |
| `kid-m` | Male | Child (~8 years old) |
| `kid-f` | Female | Child (~8 years old) |
| `teen-m` | Male | Teenager (~16 years old) |
| `teen-f` | Female | Teenager (~16 years old) |

All themes support 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian.

### CLI

```bash
# Dub with a voice theme
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --voice-theme narrator-m \
    --target-language Spanish

# Speak sub-command with a theme
mazinger speak --srt translated.srt --original-audio audio.mp3 \
    --voice-theme warm-f -o dubbed.wav
```

### Python

```python
from mazinger import MazingerDubber

dubber = MazingerDubber(openai_api_key="sk-...", base_dir="./output")

proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    voice_theme="narrator-m",
    target_language="Spanish",
)
```

### List All Themes

```bash
mazinger profile list
```

In Python:

```python
from mazinger.profiles import list_themes

for theme in list_themes():
    print(f"{theme['name']:20s} {theme['gender']:8s} langs={', '.join(theme['languages'])}")
```

### Generate and Reuse a Theme Profile

You can pre-generate a profile from a theme and save it for repeated use:

```bash
mazinger profile generate narrator-f English -o ./my-narrator

# Use it with --clone-profile
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile ./my-narrator \
    --target-language English
```

### Profile Persistence

When you use `--voice-theme` with `dub` or `speak`, the generated profile is automatically saved to the project's `voice_profile/` directory (`voice.wav` + `script.txt`). On subsequent runs of the same project, the saved profile is reused — no regeneration needed.

---

## Option 3: Use a Built-in Profile

Profiles are hosted on HuggingFace at [`bakrianoo/mazinger-dubber-profiles`](https://huggingface.co/datasets/bakrianoo/mazinger-dubber-profiles) and downloaded on first use. Files are cached in `/tmp/mazinger-dubber-profiles/` and reused on subsequent calls. Non-WAV voice files are automatically converted to 16-kHz mono WAV for TTS compatibility.

### CLI

```bash
# Full pipeline — dub an entire video with the abubakr voice
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --target-language Arabic

# Speak sub-command — synthesize from an existing SRT
mazinger speak --srt translated.srt --original-audio audio.mp3 \
    --clone-profile morgan-freeman -o dubbed.wav
```

You can override the voice script while keeping the profile's audio sample:

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr --voice-script custom_transcript.txt
```

### Python

```python
from mazinger.profiles import fetch_profile

voice_path, script_path = fetch_profile("abubakr")
# voice_path  → /tmp/mazinger-dubber-profiles/abubakr/voice.wav
# script_path → /tmp/mazinger-dubber-profiles/abubakr/script.txt
```

### Local Directory Profiles

`--clone-profile` also accepts a local directory path. The directory must contain a voice sample file (`voice.wav`, `voice.m4a`, or `voice.mp3`) and a `script.txt` transcript.

```bash
# Use a local profile directory
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile ./my-profile \
    --target-language Spanish
```

This is useful for profiles generated via `mazinger profile generate` or your own custom voice directories.

```python
from mazinger.profiles import fetch_profile

# Also works with local paths
voice_path, script_path = fetch_profile("./my-profile")
```

### Available Profiles

| Profile | Language | Description |
|---------|----------|-------------|
| `abubakr` | English | Abu Bakr Soliman — male voice |
| `daheeh-v1` | English | Ahmed El Ghandour (The Daheeh) — male Arabic-accented English voice |
| `3b1b` | English | 3Blue1Brown (Grant Sanderson) — male voice, math/science narration style |
| `italian-v1` | Italian | Italian male narrator voice |
| `morgan-freeman` | English | Morgan Freeman — male narration voice |
| `trump-v1` | English | Donald Trump — male political speech style |

---

## Option 4: Use Your Own Custom Voice

If you have your own voice recording, you can use it directly — no profile upload required.

### What You Need

| File | Purpose | Required by |
|------|---------|-------------|
| **Voice sample** (`.wav`, `.m4a`, `.mp3`) | 10–30 seconds of clear speech | Both engines |
| **Transcript** (`script.txt`) | Exact word-for-word text of what is spoken in the sample | Qwen TTS only (Chatterbox ignores it) |

### CLI Examples

**Minimal — dub with your own voice using Qwen TTS (default engine):**

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --voice-sample ./my-voice.wav \
    --voice-script ./my-transcript.txt \
    --target-language Spanish \
    --base-dir ./output
```

**With Chatterbox (no transcript needed):**

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --voice-sample ./my-voice.wav \
    --voice-script "" \
    --tts-engine chatterbox \
    --target-language French \
    --base-dir ./output
```

**Inline transcript text (instead of a file path):**

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --voice-sample ./my-voice.wav \
    --voice-script "Hello, this is the exact text I spoke in my recording." \
    --target-language German
```

> Mazinger auto-detects whether `--voice-script` is a file path or inline text. If the value points to an existing file, it reads the file; otherwise it treats the string as literal transcript text.

**Standalone speak sub-command with a custom voice:**

```bash
mazinger speak \
    --srt translated.srt \
    --original-audio source-audio.mp3 \
    --voice-sample ./my-voice.wav \
    --voice-script ./my-transcript.txt \
    -o dubbed_audio.wav
```

### Python API

```python
from mazinger import MazingerDubber

dubber = MazingerDubber(openai_api_key="sk-...", base_dir="./output")

# Qwen TTS — requires voice sample + matching transcript
proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    voice_sample="./my-voice.wav",
    voice_script="./my-transcript.txt",       # file path or inline text
    target_language="Spanish",
    tts_engine="qwen",
    output_type="video",
)

# Chatterbox — requires only the voice sample
proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    voice_sample="./my-voice.wav",
    voice_script="",                           # not used by Chatterbox
    target_language="French",
    tts_engine="chatterbox",
    chatterbox_exaggeration=0.6,               # emotion intensity (0.0–1.0)
    chatterbox_cfg=0.5,                        # pacing control (0.0–1.0)
    output_type="audio",
)
```

---

## TTS Engine Comparison: Qwen vs Chatterbox

| Feature | Qwen3-TTS (default) | Chatterbox | MLX |
|---------|---------------------|------------|-----|
| **Voice sample** | Required | Required | Required |
| **Transcript of sample** | **Required** — must match the audio word-for-word | Not used | **Required** — must match the audio word-for-word |
| **Install extra** | `pip install "mazinger[tts]"` | `pip install "mazinger[tts-chatterbox]"` | `pip install "mazinger[tts-mlx]"` |
| **CLI flag** | `--tts-engine qwen` | `--tts-engine chatterbox` | `--tts-engine mlx` |
| **Model flag** | `--tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `--chatterbox-model ResembleAI/chatterbox` | `--mlx-tts-model mlx-community/...` |
| **Emotion control** | — | `--chatterbox-exaggeration` (0.0–1.0) | — |
| **Pacing control** | — | `--chatterbox-cfg` (0.0–1.0) | — |
| **Supported languages** | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | English (primary) | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |
| **Hardware** | CUDA GPU | CUDA GPU | Apple Silicon |

> **Important:** Qwen, Chatterbox, and MLX require different dependencies and **cannot share the same Python environment**. Create separate virtual environments for each.

### When to Choose Which

- **Qwen** — Best for multilingual dubbing and when you have an accurate transcript of the voice sample.
- **Chatterbox** — Best when you only have an audio clip with no transcript, or when you want fine-grained control over emotion and pacing.
- **MLX** — Best for Apple Silicon users who want GPU-accelerated TTS without CUDA.

---

## Recording Tips for Custom Voices

- **Environment** — Record in a quiet room with minimal echo and background noise
- **Duration** — 10–30 seconds of speech is ideal; longer samples don't significantly improve quality
- **Speaking style** — Speak naturally at your normal pace and tone
- **Audio format** — WAV at 16 kHz mono is optimal; Mazinger auto-converts other formats (`.m4a`, `.mp3`)
- **Transcript accuracy** — For Qwen, the transcript must match the recording **exactly**, word for word — mismatches degrade clone quality

> **Tip:** If you don't want to record your own voice, use voice themes (`--voice-theme`) instead. Themes are pre-trained and ready to use immediately.

### Example: Recording Your Voice from Scratch

```bash
# 1. Record 20 seconds of speech (requires sox / arecord / any recorder)
#    Read a paragraph from a book or article in a clear, natural voice.

# 2. Save the recording
#    → my-voice.wav

# 3. Write the transcript — exactly what you said, word for word
cat > my-transcript.txt << 'EOF'
The quick brown fox jumps over the lazy dog. Pack my box with five
dozen liquor jugs. How vexingly quick daft zebras jump.
EOF

# 4. Test it on a short video
mazinger dub ./test-video.mp4 \
    --voice-sample my-voice.wav \
    --voice-script my-transcript.txt \
    --target-language English \
    --base-dir ./test-output
```

---

## Creating and Sharing a Reusable Profile

Once you've tested your voice and are happy with the results, you can package it as a profile on HuggingFace so others (or your CI/CD pipeline) can use `--clone-profile your-name`.

### Profile Structure

Each profile is a folder containing two files:

```
profiles/your-name/
├── voice.wav          # or voice.m4a, voice.mp3
└── script.txt         # plain-text transcript of the voice sample
```

### Step-by-Step Upload

**1. Prepare the folder:**

```bash
mkdir -p profiles/my-name
cp my-voice.wav profiles/my-name/voice.wav
cp my-transcript.txt profiles/my-name/script.txt
```

**2. Log in to HuggingFace:**

```bash
pip install huggingface_hub
huggingface-cli login
```

**3. Upload to the shared dataset:**

```bash
huggingface-cli upload bakrianoo/mazinger-dubber-profiles \
    ./profiles/my-name my-name \
    --repo-type=dataset \
    --commit-message "Add profile: my-name"
```

**4. Use it immediately:**

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" --clone-profile my-name
```

---

## Using a Private HuggingFace Dataset

You can use your own private HuggingFace dataset instead of the default public one. This is useful for storing custom voice profiles securely.

### Setup

**1. Set environment variables:**

```bash
# Your HuggingFace token (from https://huggingface.co/settings/tokens)
export HF_TOKEN="hf_..."

# Your private dataset URL
export MAZINGER_PROFILES_REPO_URL="https://huggingface.co/datasets/YOUR_USERNAME/your-dataset/resolve/main/profiles"
```

Add these to your shell profile (`~/.zshrc`) for persistence.

**2. Get your token:**

```bash
pip install huggingface_hub
hf auth login  # or go to https://huggingface.co/settings/tokens
```

### Upload a Profile

**1. Prepare the folder structure:**

```bash
mkdir -p profiles/your-voice-name
cp your-voice.wav profiles/your-voice-name/voice.wav
cp your-transcript.txt profiles/your-voice-name/script.txt
```

**2. Upload to your private dataset:**

```bash
# Create dataset first if needed: https://huggingface.co/datasets/new

hf upload YOUR_USERNAME/mazinger-dubber-profiles \
    ./profiles/your-voice-name \
    profiles/your-voice-name \
    --repo-type dataset
```

### Use Your Private Profile

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile your-voice-name \
    --target-language Spanish
```

### Priority Order

When you set `MAZINGER_PROFILES_REPO_URL`:

1. Custom repo is checked first
2. If profile exists in both custom and default → uses custom repo
3. Falls back to default repo if not found in custom

### Python API

```python
import os

os.environ["HF_TOKEN"] = "hf_..."
os.environ["MAZINGER_PROFILES_REPO_URL"] = "https://huggingface.co/datasets/..."

from mazinger.profiles import fetch_profile
voice, script = fetch_profile("your-voice-name")
```

---

## Python API for Themes and Profiles

### List available themes

```python
from mazinger.profiles import list_themes

themes = list_themes()
# Returns: [{'name': 'narrator-m', 'gender': 'male', 'languages': [...]}, ...]
```

### Generate a profile from a theme

```python
from mazinger.profiles import generate_profile

voice_path, script_path = generate_profile(
    "narrator-m", "Spanish", "./my-profile",
    device="cuda:0", dtype="bfloat16",
)
# Generates voice.wav and script.txt in ./my-profile/
```

### Use themes in dub()

```python
from mazinger import MazingerDubber

dubber = MazingerDubber(openai_api_key="sk-...")

proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    voice_theme="narrator-f",     # instead of voice_sample + voice_script
    target_language="French",
    output_type="video",
)
```

---

## CLI Reference: All Voice-Related Flags

| Flag | Description | Default |
|------|-------------|--------|
| `--voice-theme NAME` | Use a pre-defined voice theme (see `mazinger profile list`) | — |
| `--clone-profile NAME_OR_PATH` | Voice profile from HuggingFace or local directory path | — |
| `--voice-sample PATH` | Path to a local voice reference audio file | — |
| `--voice-script PATH_OR_TEXT` | Path to transcript file, or inline transcript text | — |
| `--tts-engine {qwen,chatterbox}` | TTS engine to use | `qwen` |
| `--tts-model NAME` | HuggingFace model ID for Qwen | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| `--chatterbox-model NAME` | HuggingFace model ID for Chatterbox | `ResembleAI/chatterbox` |
| `--tts-language LANG` | Language hint for TTS pronunciation | same as `--target-language` |
| `--chatterbox-exaggeration FLOAT` | Emotion intensity (Chatterbox only) | `0.5` |
| `--chatterbox-cfg FLOAT` | Pacing/fluency control (Chatterbox only) | `0.5` |
