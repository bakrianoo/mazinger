# Python API

## MazingerDubber

The main pipeline class. Orchestrates all nine stages.

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
)
```

### dub()

Runs the full pipeline and returns a `ProjectPaths` object.

```python
proj = dubber.dub(
    source,                           # str — URL or local file path (required)
    voice_sample,                     # str — path to reference voice audio (required)
    voice_script,                     # str — path to reference transcript (required)
    *,
    slug=None,                        # str — project directory name (auto-generated if omitted)
    device="cuda",                    # str — "cuda", "cpu", or "auto"
    transcribe_method="openai",       # str — "openai", "faster-whisper", "whisperx"
    whisper_model=None,               # str — model name override
    tts_model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    tts_dtype="bfloat16",             # str — "bfloat16", "float16", "float32"
    tts_language=None,                # str — TTS language hint (defaults to target_language)
    tts_engine="qwen",               # str — "qwen" or "chatterbox"
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
    max_tempo=1.3,                    # float — speed-up cap for auto/dynamic
    words_per_second=None,            # float — speech rate for word budgets (default: 2.0)
    duration_budget=None,             # float — fraction of time for speech (default: 0.80)
    translate_technical_terms=False,   # bool — translate tech terms vs. keep in English
    output_type="audio",              # str — "audio" (WAV) or "video" (MP4)
    subtitle_style=None,              # SubtitleStyle — styling for burned subtitles
    subtitle_source="translated",     # str — "translated", "original", or file path
)
```

Returns a `ProjectPaths` instance with all output paths populated.

### Full workflow example

```python
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

### translate

```python
from mazinger.translate import translate_srt

translated = translate_srt(
    srt_text,
    description=desc,
    thumb_paths=thumbs,
    client=client,
    target_language="Spanish",
    words_per_second=2.0,
    duration_budget=0.80,
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
    words_per_second=2.0,
    duration_budget=0.80,
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

voice_path, script_path = fetch_profile("abubakr")
# voice_path  → /tmp/mazinger-dubber-profiles/abubakr/voice.wav
# script_path → /tmp/mazinger-dubber-profiles/abubakr/script.txt
```

Files are cached in the system temp directory. Non-WAV voice files are converted to 16-kHz mono WAV automatically.

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
