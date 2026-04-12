# Python API

## MazingerDubber

The main pipeline class. Orchestrates all ten stages.

```python
from mazinger import MazingerDubber
```

### Constructor

```python
MazingerDubber(
    openai_api_key=None,      # str — or set OPENAI_API_KEY env var
    openai_base_url=None,     # str — or set OPENAI_BASE_URL env var
    llm_model=None,           # str — default: OPENAI_MODEL env var or "gpt-4.1"
    base_dir="./mazinger_output",
    llm_think=None,           # bool | None — True/False to control LLM thinking mode;
                              #   None (default) sends no preference.
                              #   Set to False for Ollama models like Qwen3 that
                              #   have thinking enabled by default.
)
```

### dub()

Runs the full pipeline and returns a `ProjectPaths` object.

```python
proj = dubber.dub(
    source,                           # str — URL or local file path (required)
    voice_sample=None,                # str | None — path to reference voice audio
    voice_script=None,                # str | None — path to reference transcript
    *,
    voice_theme=None,                 # str | None — pre-defined theme (alternative to sample+script)
    slug=None,                        # str — project directory name (auto-generated if omitted)
    device="cuda",                    # str — "cuda", "cpu", or "auto"
    transcribe_method="faster-whisper",  # str — "openai", "faster-whisper", "whisperx", "mlx-whisper"
    whisper_model=None,               # str — model name override
    mlx_whisper_model="mlx-community/whisper-large-v3-turbo",  # str — MLX Whisper model
    tts_model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    tts_dtype="bfloat16",             # str — "bfloat16", "float16", "float32"
    tts_language=None,                # str — TTS language hint (defaults to target_language)
    tts_engine="qwen",               # str — "qwen", "chatterbox", or "mlx"
    mlx_model="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",  # str — MLX TTS model
    source_language="auto",           # str — source language or "auto"
    target_language="English",        # str — one of 33 supported languages
    chatterbox_model="ResembleAI/chatterbox",
    chatterbox_exaggeration=0.5,      # float — 0.0–1.0, emotion intensity
    chatterbox_cfg=0.5,               # float — 0.0–1.0, pacing control
    cookies_from_browser=None,        # str — browser name for yt-dlp cookie extraction
    cookies=None,                     # str — path to cookies.txt
    quality=None,                     # str — "low", "medium", "high", or numeric
    start=None,                       # str — start timestamp for slicing (e.g. "00:01:30" or "90")
    end=None,                         # str — end timestamp for slicing (e.g. "00:05:00" or "300")
    skip_existing=True,               # bool — skip stages with existing output
    force_reset=False,                # bool — delete cache and re-run everything
    use_resegmented=False,            # bool — use resegmented SRT for TTS input
    tempo_mode="auto",                # str — "auto", "dynamic", "fixed", "off"
    fixed_tempo=None,                 # float — constant speed multiplier
    max_tempo=1.5,                    # float — speed-up cap for auto/dynamic
    words_per_second=None,            # float — speech rate for word budgets (auto-estimated)
    duration_budget=None,             # float — fraction of time for speech (default: 0.85)
    translate_technical_terms=False,   # bool — translate tech terms vs. keep in English
    asr_review=False,                  # bool — review ASR transcript (fix typos, punctuation)
    keep_technical_english=False,      # bool — convert technical terms to English (requires asr_review)
    use_youtube_subs=False,            # bool — download YouTube captions and compare with ASR
    loudness_match=True,              # bool — normalise dubbed loudness to original
    mix_background=True,              # bool — mix original background audio under dub
    background_volume=0.15,           # float — background layer gain (0.0–1.0)
    output_type="audio",              # str — "audio" (WAV) or "video" (MP4)
    subtitle_style=None,              # SubtitleStyle — styling for burned subtitles
    subtitle_source="translated",     # str — "translated", "original", or file path
)
```

Returns a `ProjectPaths` instance with all output paths populated.

> When neither `voice_sample`/`voice_script` nor `voice_theme` is provided, the pipeline automatically clones the speaker's voice from the source audio (auto-clone). A 20–60 s segment with the best word coverage is selected from the transcript.
>
> When `voice_theme` is provided, `voice_sample` and `voice_script` are optional. The theme generates a voice profile automatically and saves it to the project's `voice_profile/` directory.

### Full workflow example

```python
from mazinger import MazingerDubber

# Auto-clone the speaker's voice (simplest — no voice configuration)
dubber = MazingerDubber(openai_api_key="sk-...", base_dir="./output")

proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    target_language="Spanish",
    output_type="video",
)

print(proj.final_video)
```

```python
# Using a voice theme
dubber = MazingerDubber(openai_api_key="sk-...", base_dir="./output")

proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    voice_theme="narrator-m",
    target_language="Spanish",
    output_type="video",
)

print(proj.final_video)
```

```python
# Using a voice profile
from mazinger import MazingerDubber
from mazinger.profiles import fetch_profile
from mazinger.subtitle import SubtitleStyle

dubber = MazingerDubber(
    openai_api_key="sk-...",
    llm_model="gpt-4.1",
    base_dir="./output",
)

voice, script = fetch_profile("abubakr")

style = SubtitleStyle(
    font_size=24,
    font_color="yellow",
    bg_color="black",
    bg_alpha=0.8,
    bold=True,
)

proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    voice_sample=voice,
    voice_script=script,
    target_language="Arabic",
    output_type="video",
    subtitle_style=style,
    subtitle_source="translated",
    tts_engine="qwen",
    transcribe_method="faster-whisper",
)

print(proj.summary())
```

---

## ProjectPaths

Manages all file paths for a single project.

```python
from mazinger import ProjectPaths
```

### Constructor

```python
ProjectPaths(slug, base_dir="./mazinger_output")
```

### Properties

| Property | Path |
|----------|------|
| `root` | `<base_dir>/projects/<slug>/` |
| `video` | `source/video.mp4` |
| `audio` | `source/audio.mp3` |
| `source_srt` | `transcription/source.srt` |
| `source_raw_srt` | `transcription/source.raw.srt` |
| `translated_raw_srt` | `transcription/translated.raw.srt` |
| `final_srt` | `subtitles/translated.srt` |
| `thumbs_meta` | `thumbnails/meta.json` |
| `description` | `analysis/description.json` |
| `final_audio` | `tts/dubbed.wav` |
| `final_video` | `tts/dubbed.mp4` |
| `tts_segments_dir` | `tts/segments/` |
| `voice_profile_dir` | `voice_profile/` |

### Methods

```python
proj.ensure_dirs()   # Create all subdirectories (idempotent). Returns self.
proj.summary()       # Human-readable overview of which files exist.
```

---

## Individual Stage Functions

Each pipeline stage is available as a standalone function.

### download

```python
from mazinger.download import resolve_slug, download_video, extract_audio

slug, info = resolve_slug("https://youtube.com/watch?v=VIDEO_ID")
download_video("https://youtube.com/watch?v=VIDEO_ID", "video.mp4")
extract_audio("video.mp4", "audio.mp3")
```

Slice a downloaded file to a time range:

```python
from mazinger.download import slice_media, slice_project

# Slice a media file directly
slice_media("video.mp4", "clip.mp4", start="00:01:30", end="00:05:00")

# Slice a project’s video/audio in-place
slice_project(proj, start="90", end="300")
```

### transcribe

```python
from mazinger.transcribe import transcribe

# Cloud
transcribe("audio.mp3", "subtitles.srt")

# Local
transcribe("audio.mp3", "subtitles.srt", method="faster-whisper", device="cuda")
transcribe("audio.mp3", "subtitles.srt", method="whisperx", device="cuda")

# MLX (Apple Silicon)
transcribe("audio.mp3", "subtitles.srt", method="mlx-whisper")
```

### thumbnails

```python
from openai import OpenAI
from mazinger.thumbnails import select_timestamps, extract_frames

client = OpenAI()
with open("subtitles.srt") as f:
    srt_text = f.read()

timestamps = select_timestamps(srt_text, client)
frames = extract_frames("video.mp4", timestamps, "./thumbs")
```

### describe

```python
from mazinger.describe import describe_content
from mazinger.utils import load_json

thumbs = load_json("./thumbs/meta.json")
desc = describe_content(srt_text, thumbs, client)
# Returns: {"title": "...", "summary": "...", "keypoints": [...], "keywords": [...]}
```

### review

```python
from mazinger.review import review_srt

reviewed = review_srt(
    srt_text,
    description=desc,
    client=client,
    source_language="auto",
    keep_technical_english=False,
)
```

Full signature:

```python
review_srt(
    srt_text,              # str — source SRT content
    description,           # dict — from describe_content
    client,                # OpenAI client
    *,
    llm_model="gpt-4.1",
    source_language="auto",
    keep_technical_english=False,
    blocks_per_batch=30,
    overlap_size=6,
    usage_tracker=None,    # LLMUsageTracker
)
```

### translate

```python
from mazinger.translate import translate_srt

translated = translate_srt(
    srt_text,
    description=desc,
    thumb_paths=thumbs,
    client=client,
    target_language="Spanish",
)

with open("translated.srt", "w") as f:
    f.write(translated)
```

Full signature:

```python
translate_srt(
    srt_text,              # str — source SRT content
    description,           # dict — from describe_content
    thumb_paths,           # list[dict] — from thumbnails meta.json
    client,                # OpenAI client
    *,
    llm_model="gpt-4.1",
    source_language="auto",
    target_language="English",
    blocks_per_batch=24,
    overlap_size=8,
    words_per_second=None,       # auto-estimated from source + target language
    duration_budget=0.85,
    translate_technical_terms=False,
    usage_tracker=None,    # LLMUsageTracker
)
```

### resegment

```python
from mazinger.resegment import resegment_srt

final = resegment_srt(srt_text, client=client, max_chars=84, max_dur=4.0)
```

The `client` parameter is optional. Without it, the function uses rule-based merging instead of LLM-powered merging.

### tts

```python
from mazinger import tts
from mazinger.srt import parse_file

# Load model
model = tts.load_model(engine="qwen", device="cuda:0")

# Create voice prompt from reference recording
wrapper = tts.create_voice_prompt(
    model,
    ref_audio="speaker.m4a",
    ref_text="Transcript of the voice sample.",
    engine="qwen",
)

# Synthesize all segments
entries = parse_file("translated.srt")
segments = tts.synthesize_segments(model, wrapper, entries, "./segments")

# Clean up
tts.unload_model(model)
```

For Chatterbox:

```python
model = tts.load_model(engine="chatterbox")
wrapper = tts.create_voice_prompt(
    model,
    ref_audio="speaker.m4a",
    ref_text="",              # not needed for Chatterbox
    engine="chatterbox",
    chatterbox_exaggeration=0.5,
    chatterbox_cfg=0.5,
)
segments = tts.synthesize_segments(model, wrapper, entries, "./segments", language="Spanish")
```

To re-synthesize all segments instead of resuming:

```python
segments = tts.synthesize_segments(model, wrapper, entries, "./segments", force_reset=True)
```

### assemble

```python
from mazinger.assemble import assemble_timeline, mux_video
from mazinger.utils import get_audio_duration

duration = get_audio_duration("audio.mp3")

# No tempo adjustment
assemble_timeline(segments, duration, "dubbed.wav")

# Fixed tempo
assemble_timeline(segments, duration, "dubbed.wav", tempo_mode="fixed", fixed_tempo=1.1)

# Dynamic tempo
assemble_timeline(segments, duration, "dubbed.wav", tempo_mode="dynamic", max_tempo=1.3)

# Mux audio into video
mux_video("video.mp4", "dubbed.wav", "dubbed.mp4")

# Post-process: loudness matching + background mixing
from mazinger.assemble import post_process
post_process("dubbed.wav", "audio.mp3", "dubbed_final.wav")

# Skip background mixing, only normalise loudness
post_process("dubbed.wav", "audio.mp3", "dubbed_final.wav", mix_background=False)
```

### subtitle

```python
from mazinger.subtitle import SubtitleStyle, burn_subtitles, download_google_font

style = SubtitleStyle(
    font="DejaVu Sans",
    font_size=28,
    font_color="yellow",
    position="bottom",
    bold=True,
)

# Burn subtitles (keep original audio)
burn_subtitles("video.mp4", "output.mp4", "translated.srt", style)

# Burn subtitles and replace audio
burn_subtitles("video.mp4", "output.mp4", "translated.srt", style, audio_path="dubbed.wav")

# Use a Google Font
font_path = download_google_font("Noto Sans Arabic")
style = SubtitleStyle(font_file=font_path, font_size=24)
burn_subtitles("video.mp4", "output.mp4", "translated.srt", style)
```

### profiles

```python
from mazinger.profiles import fetch_profile

# From HuggingFace
voice_path, script_path = fetch_profile("abubakr")
# voice_path  → /tmp/mazinger-dubber-profiles/abubakr/voice.wav
# script_path → /tmp/mazinger-dubber-profiles/abubakr/script.txt

# From a local directory
voice_path, script_path = fetch_profile("./my-profile")
```

Files are cached in the system temp directory. Non-WAV voice files are converted to 16-kHz mono WAV automatically.

### Voice themes and profile generation

```python
from mazinger.profiles import list_themes, generate_profile

# List all 16 pre-defined themes
themes = list_themes()
for t in themes:
    print(f"{t['name']:20s} {t['gender']:8s} {', '.join(t['languages'])}")

# Generate a reusable profile from a theme
voice_path, script_path = generate_profile(
    "narrator-m", "Spanish", "./my-profile",
    device="cuda:0", dtype="bfloat16",
)
# Creates: ./my-profile/voice.wav and ./my-profile/script.txt
# Use with: --clone-profile ./my-profile
```

---

## LLMUsageTracker

Tracks token usage across LLM calls.

```python
from mazinger import LLMUsageTracker

tracker = LLMUsageTracker()

# Pass to any LLM-calling function
translated = translate_srt(srt_text, desc, thumbs, client, usage_tracker=tracker)

# Inspect results
print(tracker.report())             # Formatted usage report
print(tracker.total_input)          # Total input tokens
print(tracker.total_output)         # Total output tokens
print(tracker.total_tokens)         # Combined total
print(tracker.summary_by_stage())   # Dict grouped by stage
```

Records are also saved to `<project>/llm_usage.json` when running through the pipeline.

---

## SRT Parsing

```python
from mazinger.srt import parse_file, parse_string, format_srt

entries = parse_file("subtitles.srt")
# Returns: [{"idx": "1", "start": 5.0, "end": 10.5, "text": "Hello world"}, ...]

entries = parse_string(srt_content)

srt_text = format_srt(entries)
```

---

## Utility Functions

```python
from mazinger.utils import (
    sanitize_filename,
    get_audio_duration,
    save_json,
    load_json,
    image_to_base64,
    make_image_content,
)

slug = sanitize_filename("My Video Title! (2024)")  # "my-video-title-2024"
duration = get_audio_duration("audio.mp3")           # seconds as float
save_json(data, "output.json")
data = load_json("input.json")
b64 = image_to_base64("thumb.jpg")
content = make_image_content("thumb.jpg", detail="low")  # OpenAI vision block
```
