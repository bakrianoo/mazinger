# Project Structure

## Output Directory Layout

All output files are organized under a single root directory. Each video gets its own project folder named by a slug (auto-generated from the video title or a custom `--slug` value). Files that do not depend on the target language (source media, transcription, thumbnails, analysis) are shared; everything produced for one target language lives under `lang/<language>/`, so one video can be dubbed into several languages side by side.

```
<base_dir>/
└── projects/
    └── <slug>/
        ├── source/                         # shared
        │   ├── video.mp4                   # Downloaded or copied video
        │   ├── audio.mp3                   # Extracted audio track
        │   ├── video_meta.json             # Title, description, channel (downloads)
        │   ├── background.24000.wav        # Cached background (non-vocal) stem
        │   └── loudness.json               # Cached loudness of audio.mp3
        ├── transcription/                  # shared
        │   ├── source.raw.srt              # Raw transcription output
        │   ├── source.srt                  # Cleaned and resegmented transcription
        │   └── source.reviewed.srt         # ASR-reviewed transcript (when --asr-review is used)
        ├── thumbnails/                     # shared
        │   ├── thumb_000_12.5s.jpg         # Extracted key frames
        │   └── meta.json                   # Thumbnail metadata (timestamps, reasons, paths)
        ├── analysis/                       # shared
        │   └── description.json            # Content analysis (title, summary, keypoints, keywords)
        ├── lang/
        │   └── <language>/                 # one folder per target language
        │       ├── run.json                # Settings the dub used (no API keys)
        │       ├── transcription/
        │       │   └── translated.raw.srt  # Translation, 1:1 with source.srt
        │       ├── subtitles/
        │       │   └── translated.srt      # Final translated and resegmented subtitles
        │       ├── tts/
        │       │   ├── segments/
        │       │   │   ├── seg_0001.wav    # One TTS clip per translated.srt entry
        │       │   │   └── ...
        │       │   ├── dubbed.wav          # Assembled dubbed audio
        │       │   └── dubbed.mp4          # Final video with dubbed audio and subtitles
        │       ├── voice_profile/
        │       │   ├── voice.wav           # Theme or auto-clone voice reference
        │       │   ├── script.txt          # Its transcript
        │       │   ├── instruct.txt        # OmniVoice voice-design instruction
        │       │   ├── reference/          # Kept copy of an uploaded sample or profile
        │       │   └── omnivoice_auto/     # Segment kept as the OmniVoice auto-voice reference
        │       └── editor/                 # Editor session (see below)
        └── llm_usage.json                  # Token usage records for all LLM calls
```

The default `<base_dir>` is `./mazinger_output`. Change it with `--base-dir` or the `base_dir` constructor parameter.

## File Descriptions

### source/

| File | Created by | Description |
|------|-----------|-------------|
| `video.mp4` | download | Original video (from URL or local copy) |
| `audio.mp3` | download | Audio track extracted with ffmpeg |
| `video_meta.json` | download | Video title, description, channel and tags, used as translation context |
| `background.<sr>.wav` | assemble | Background stem separated by Demucs. Shared by every language and reused while it is newer than `audio.mp3` |
| `loudness.json` | assemble | Integrated loudness of `audio.mp3`, reused while `audio.mp3` keeps its size and modification time |

### transcription/

| File | Created by | Description |
|------|-----------|-------------|
| `source.raw.srt` | transcribe | Direct output from the speech recognition engine |
| `source.srt` | transcribe | Cleaned version with basic resegmentation applied |
| `source.reviewed.srt` | review | Transcript corrected by the LLM review (`--asr-review`) |

### lang/&lt;language&gt;/

| File | Created by | Description |
|------|-----------|-------------|
| `run.json` | dub | Settings of the last completed dub: ASR, LLM (model and base URL only), translation, TTS, voice, assembly and output options. API keys, tokens and cookies are never written. The [Editor](editor.md) redoes steps with these settings |
| `transcription/translated.raw.srt` | translate | Translation output before resegmentation, one entry per `source.srt` entry |
| `subtitles/translated.srt` | resegment | Final subtitles — translated, merged, and split for readability. The Editor rewrites it on assemble |

### thumbnails/

| File | Created by | Description |
|------|-----------|-------------|
| `thumb_NNN_Xs.jpg` | thumbnails | JPEG frames at LLM-selected timestamps |
| `meta.json` | thumbnails | Array of objects with `timestamp`, `seconds`, `reason`, `path` |

Example `meta.json`:

```json
[
    {
        "timestamp": "02:00",
        "seconds": 120.5,
        "reason": "Speaker opens the configuration dashboard",
        "path": "/absolute/path/to/thumb_034_120.5s.jpg"
    }
]
```

### analysis/

| File | Created by | Description |
|------|-----------|-------------|
| `description.json` | describe | Structured content analysis |

Example `description.json`:

```json
{
    "title": "Building REST APIs with FastAPI",
    "summary": "A walkthrough of creating REST endpoints using FastAPI...",
    "keypoints": [
        "FastAPI uses type hints for validation",
        "Automatic OpenAPI documentation generation"
    ],
    "keywords": ["FastAPI", "REST", "Pydantic", "OpenAPI"]
}
```

### tts/

| File | Created by | Description |
|------|-----------|-------------|
| `segments/seg_NNNN.wav` | speak | One WAV file per `translated.srt` entry |
| `dubbed.wav` | assemble | All segments placed on a timeline matching the original duration |
| `dubbed.mp4` | subtitle / mux | Final video with dubbed audio and optional burned subtitles |
| `dubbed.prev.wav`, `dubbed.prev.mp4` | Editor | The output before the last Editor assemble |

### voice_profile/

| File | Created by | Description |
|------|-----------|-------------|
| `voice.wav`, `script.txt` | dub (theme or auto-clone) | Generated or extracted voice reference and its transcript, reused on later runs |
| `instruct.txt` | dub (OmniVoice theme) | The voice-design instruction used |
| `reference/voice.*`, `reference/script.txt` | dub | A copy of the uploaded voice sample or profile, so single segments can be re-dubbed after the upload is gone |
| `omnivoice_auto/`, `omnivoice_design/` | dub / Editor | A dubbed segment kept as the voice reference for OmniVoice auto-voice and voice-design dubs, which have no reference clip of their own |

### editor/

Created by the [Editor](editor.md#files-and-backups).

| File | Description |
|------|-------------|
| `session.json` | Snapshot of the session: every chunk, its stale flags and undo history |
| `changes.jsonl` | Edits since the snapshot, one line each |
| `segments/<id>_v<n>.wav` | Re-dubbed chunks; every version is kept for undo |
| `cache/orig_<start>_<end>.ogg` | Original-audio clips for the player (size-capped) |
| `source.edited.srt` | The edited transcription for this language |
| `subtitles.display.srt`, `source.display.srt` | On-screen subtitle lines built on assemble |

### llm_usage.json

| File | Created by | Description |
|------|-----------|-------------|
| `llm_usage.json` | pipeline | Token usage for every LLM call across all stages |

## Slug Generation

When downloading from a URL, the slug is derived from the video title:

1. The title is lowercased
2. Special characters are removed
3. Spaces become hyphens
4. Consecutive hyphens are collapsed

For example, "Building REST APIs with FastAPI (2024)" becomes `building-rest-apis-with-fastapi-2024`.

Override with `--slug my-custom-name` to use a fixed name.

## ProjectPaths in Python

The `ProjectPaths` class provides typed access to every path:

```python
from mazinger import ProjectPaths

proj = ProjectPaths("my-video", base_dir="./output", target_language="Spanish")
proj.ensure_dirs()  # create all subdirectories

print(proj.root)               # ./output/projects/my-video
print(proj.video)              # ./output/projects/my-video/source/video.mp4
print(proj.audio)              # ./output/projects/my-video/source/audio.mp3
print(proj.background_audio()) # ./output/projects/my-video/source/background.24000.wav
print(proj.source_srt)         # ./output/projects/my-video/transcription/source.srt
print(proj.translated_raw_srt) # ./output/projects/my-video/lang/Spanish/transcription/translated.raw.srt
print(proj.final_srt)          # ./output/projects/my-video/lang/Spanish/subtitles/translated.srt
print(proj.final_audio)        # ./output/projects/my-video/lang/Spanish/tts/dubbed.wav
print(proj.final_video)        # ./output/projects/my-video/lang/Spanish/tts/dubbed.mp4
print(proj.tts_segments_dir)   # ./output/projects/my-video/lang/Spanish/tts/segments
print(proj.run_info)           # ./output/projects/my-video/lang/Spanish/run.json
print(proj.editor_dir)         # ./output/projects/my-video/lang/Spanish/editor

print(proj.summary())          # human-readable overview of which files exist
```

Without `target_language`, the per-language paths fall back to the project root (the layout of projects made before languages had their own folders).
