# Project Structure

## Output Directory Layout

All output files are organized under a single root directory. Each video gets its own project folder named by a slug (auto-generated from the video title or a custom `--slug` value).

```
<base_dir>/
└── projects/
    └── <slug>/
        ├── source/
        │   ├── video.mp4           # Downloaded or copied video
        │   └── audio.mp3           # Extracted audio track
        ├── transcription/
        │   ├── source.raw.srt      # Raw transcription output
        │   ├── source.srt          # Cleaned and resegmented transcription
        │   └── source.reviewed.srt # ASR-reviewed transcript (when --asr-review is used)
        ├── subtitles/
        │   └── translated.srt      # Final translated and resegmented subtitles
        ├── thumbnails/
        │   ├── thumb_000_12.5s.jpg # Extracted key frames
        │   ├── thumb_001_45.0s.jpg
        │   └── meta.json           # Thumbnail metadata (timestamps, reasons, paths)
        ├── analysis/
        │   └── description.json    # Content analysis (title, summary, keypoints, keywords)
        ├── tts/
        │   ├── segments/
        │   │   ├── seg_0001.wav    # Individual TTS segment audio
        │   │   ├── seg_0002.wav
        │   │   └── ...
        │   ├── dubbed.wav          # Assembled dubbed audio
        │   └── dubbed.mp4          # Final video with dubbed audio and subtitles
        ├── voice_profile/              # Only when using --voice-theme
        │   ├── voice.wav           # Generated voice reference audio
        │   └── script.txt          # Transcript of the generated voice
        └── llm_usage.json          # Token usage records for all LLM calls
```

The default `<base_dir>` is `./mazinger_output`. Change it with `--base-dir` or the `base_dir` constructor parameter.

## File Descriptions

### source/

| File | Created by | Description |
|------|-----------|-------------|
| `video.mp4` | download | Original video (from URL or local copy) |
| `audio.mp3` | download | Audio track extracted with ffmpeg |

### transcription/

| File | Created by | Description |
|------|-----------|-------------|
| `source.raw.srt` | transcribe | Direct output from the speech recognition engine |
| `source.srt` | transcribe | Cleaned version with basic resegmentation applied |
| `translated.raw.srt` | translate | Translation output before resegmentation |

### subtitles/

| File | Created by | Description |
|------|-----------|-------------|
| `translated.srt` | resegment | Final subtitles — translated, merged, and split for readability |

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
| `segments/seg_NNNN.wav` | speak | One WAV file per subtitle entry |
| `dubbed.wav` | assemble | All segments placed on a timeline matching the original duration |
| `dubbed.mp4` | subtitle / mux | Final video with dubbed audio and optional burned subtitles |

### voice_profile/

Created only when using `--voice-theme`. The generated profile is saved here and reused on subsequent runs of the same project.

| File | Created by | Description |
|------|-----------|-------------|
| `voice.wav` | speak / dub (via VoiceDesign) | Generated voice reference audio from the selected theme |
| `script.txt` | speak / dub (via VoiceDesign) | Transcript matching the generated voice audio |

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

proj = ProjectPaths("my-video", base_dir="./output")
proj.ensure_dirs()  # create all subdirectories

print(proj.root)              # ./output/projects/my-video/
print(proj.video)             # ./output/projects/my-video/source/video.mp4
print(proj.audio)             # ./output/projects/my-video/source/audio.mp3
print(proj.source_srt)        # ./output/projects/my-video/transcription/source.srt
print(proj.source_raw_srt)    # ./output/projects/my-video/transcription/source.raw.srt
print(proj.translated_raw_srt) # ./output/projects/my-video/transcription/translated.raw.srt
print(proj.final_srt)         # ./output/projects/my-video/subtitles/translated.srt
print(proj.thumbs_meta)       # ./output/projects/my-video/thumbnails/meta.json
print(proj.description)       # ./output/projects/my-video/analysis/description.json
print(proj.final_audio)       # ./output/projects/my-video/tts/dubbed.wav
print(proj.final_video)       # ./output/projects/my-video/tts/dubbed.mp4
print(proj.tts_segments_dir)  # ./output/projects/my-video/tts/segments/

print(proj.summary())         # human-readable overview of which files exist
```
