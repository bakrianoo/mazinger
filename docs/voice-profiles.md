# Voice Profiles & Custom Voice Cloning

Mazinger can clone any voice to produce dubbed audio. There are two ways to provide a voice:

1. **Use a built-in profile** — pass `--clone-profile <name>` and Mazinger downloads the voice sample automatically
2. **Use your own voice files** — pass `--voice-sample` and `--voice-script` with local files you recorded yourself

Both approaches work with the full `dub` pipeline and the standalone `speak` sub-command.

---

## Option 1: Use a Built-in Profile

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

## Option 2: Use Your Own Custom Voice

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

| Feature | Qwen3-TTS (default) | Chatterbox |
|---------|---------------------|------------|
| **Voice sample** | Required | Required |
| **Transcript of sample** | **Required** — must match the audio word-for-word | Not used |
| **Install extra** | `pip install "mazinger[tts]"` | `pip install "mazinger[tts-chatterbox]"` |
| **CLI flag** | `--tts-engine qwen` | `--tts-engine chatterbox` |
| **Model flag** | `--tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `--chatterbox-model ResembleAI/chatterbox` |
| **Emotion control** | — | `--chatterbox-exaggeration` (0.0–1.0) |
| **Pacing control** | — | `--chatterbox-cfg` (0.0–1.0) |
| **Supported languages** | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | English (primary) |

> **Important:** Qwen and Chatterbox require different `transformers` versions and **cannot share the same Python environment**. Create separate virtual environments for each.

### When to Choose Which

- **Qwen** — Best for multilingual dubbing and when you have an accurate transcript of the voice sample.
- **Chatterbox** — Best when you only have an audio clip with no transcript, or when you want fine-grained control over emotion and pacing.

---

## Recording Tips for Custom Voices

- **Environment** — Record in a quiet room with minimal echo and background noise
- **Duration** — 10–30 seconds of speech is ideal; longer samples don't significantly improve quality
- **Speaking style** — Speak naturally at your normal pace and tone
- **Audio format** — WAV at 16 kHz mono is optimal; Mazinger auto-converts other formats (`.m4a`, `.mp3`)
- **Transcript accuracy** — For Qwen, the transcript must match the recording **exactly**, word for word — mismatches degrade clone quality

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

## CLI Reference: All Voice-Related Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--clone-profile NAME` | Download a pre-built voice profile from HuggingFace | — |
| `--voice-sample PATH` | Path to a local voice reference audio file | — |
| `--voice-script PATH_OR_TEXT` | Path to transcript file, or inline transcript text | — |
| `--tts-engine {qwen,chatterbox}` | TTS engine to use | `qwen` |
| `--tts-model NAME` | HuggingFace model ID for Qwen | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| `--chatterbox-model NAME` | HuggingFace model ID for Chatterbox | `ResembleAI/chatterbox` |
| `--tts-language LANG` | Language hint for TTS pronunciation | same as `--target-language` |
| `--chatterbox-exaggeration FLOAT` | Emotion intensity (Chatterbox only) | `0.5` |
| `--chatterbox-cfg FLOAT` | Pacing/fluency control (Chatterbox only) | `0.5` |
