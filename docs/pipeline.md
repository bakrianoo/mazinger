# Pipeline Overview

Mazinger processes video through ten stages. Each stage reads its inputs from the project directory and writes its outputs back, so stages can run independently or chain together.

## Stages

```
 Source        ┌──────────┐   ┌─────────────┐   ┌─────────────┐
 URL/file  ──▶ │ Download │──▶│ Transcribe  │──▶│ Thumbnails  │
               └──────────┘   └─────────────┘   └──────┬──────┘
                                                        │
                                                 ┌──────▼──────┐
                                                 │  Describe    │
                                                 └──────┬──────┘
                                                        │
               ┌───────────┐   ┌───────────┐   ┌───────▼──────┐
               │ Re-segment│◀──│ Translate  │◀──│   Review     │
               └─────┬─────┘   └───────────┘   └──────────────┘
                     │
               ┌─────▼─────┐   ┌───────────┐   ┌──────────────┐
               │   Speak   │──▶│ Assemble   │──▶│  Subtitle    │
               └───────────┘   └───────────┘   └──────────────┘
```

### 1. Download

Fetches a video from a URL using yt-dlp (supports YouTube, Vimeo, and hundreds of other sites) or copies a local file. Then extracts the audio track using ffmpeg.

**Inputs:** URL or local file path
**Outputs:** `source/video.mp4`, `source/audio.mp3`

Supports quality selection (`low`, `medium`, `high`, or a numeric height like `1080`) and browser cookie files for authenticated downloads.

Optional `--start` / `--end` flags (or `start`/`end` in the Python API) slice the downloaded media to a time range before any further processing. The standalone `slice` command provides this as an independent operation.

### 2. Transcribe

Converts the audio track into SRT subtitles. Three backends are available:

| Backend | How it runs | What you need |
|---------|-------------|---------------|
| OpenAI Whisper API | Cloud | API key |
| faster-whisper | Local, CTranslate2 | `transcribe-faster` extra, CUDA GPU |
| WhisperX | Local, PyTorch + wav2vec2 | `transcribe-whisperx` extra, CUDA GPU |

**Audio preprocessing:** Before transcription, the audio is automatically converted to 16 kHz mono WAV — the native format Whisper was trained on. This avoids lossy-codec artefacts and redundant internal resampling.

**Metadata-driven prompting:** When video metadata is available (e.g. YouTube title, description, tags), the pipeline automatically builds a Whisper `initial_prompt` from it. This anchors the decoder on expected vocabulary and reduces misheard domain-specific terms.

The raw transcription is saved as `source.raw.srt`. A cleaned-up version with basic re-segmentation (merging short fragments, splitting long entries) is saved as `source.srt`.

**Inputs:** `source/audio.mp3`
**Outputs:** `transcription/source.raw.srt`, `transcription/source.srt`

### 3. Thumbnails

Sends the full SRT transcript to an LLM and asks it to select timestamps where the visual content is most relevant to the spoken content. Then uses ffmpeg to extract JPEG frames at those timestamps.

The thumbnails serve as visual context for the translation and description stages — they help the LLM understand what is on screen when translating technical terms or ambiguous phrases.

**Inputs:** `source/video.mp4`, `transcription/source.srt`
**Outputs:** `thumbnails/thumb_NNN_Xs.jpg`, `thumbnails/meta.json`

### 4. Describe

Sends the transcript and thumbnail images to an LLM to produce a structured content analysis:

- **title** — a concise title for the video
- **summary** — 2–4 sentence description
- **keypoints** — list of main concepts covered
- **keywords** — technical terms, library names, proper nouns

This description is passed to the translation stage as context, so the LLM can produce coherent, domain-aware translations.

**Inputs:** `transcription/source.srt`, `thumbnails/meta.json`
**Outputs:** `analysis/description.json`

### 5. Review (optional)

Uses an LLM to review and refine the ASR-generated transcript. Enabled with `--asr-review` in the CLI or `asr_review=True` in the Python API.

The review step:

- Fixes obvious typos and misspellings (high confidence only)
- Adds missing punctuation and reshapes existing punctuation
- Corrects ASR word-boundary errors (words incorrectly split or merged)
- Optionally converts technical terms to their standard English spelling (`--keep-technical-english`)

The LLM receives the content description for context and processes subtitles in batches of 30 with 6-entry overlap. Each batch response is validated — if parsing fails, the original text is preserved.

**Inputs:** `transcription/source.srt` (or `source.raw.srt`), `analysis/description.json`
**Outputs:** `transcription/source.reviewed.srt`

### 6. Translate

Translates the SRT into the target language. Subtitles are processed in batches of 24 entries with an 8-entry overlap window for context continuity.

Each entry gets a word-count target calculated as:

```
max_words = duration_seconds × words_per_second × duration_budget
```

The defaults are 2.0 words/second and 0.80 budget (80% of available time). This prevents the translated text from being longer than what TTS can speak within the original timing.

Thumbnails and the content description are included in the LLM prompt so translations stay grounded in what is visually on screen.

**Inputs:** `transcription/source.srt`, `analysis/description.json`, `thumbnails/meta.json`
**Outputs:** `transcription/translated.raw.srt`

### 7. Re-segment

Reorganizes subtitle entries for readability and timing. Runs in two phases:

1. **Merge** — combines short, fragmented entries into complete sentences. Uses an LLM when available, falls back to rule-based merging otherwise.
2. **Split** — breaks entries that exceed the character or duration limit into multiple entries.

Default limits: 84 characters per entry, 4.0 seconds per entry (configurable).

**Inputs:** `transcription/translated.raw.srt`
**Outputs:** `subtitles/translated.srt`

### 8. Speak

Generates a WAV file for each subtitle entry using voice-cloned TTS.

**Qwen3-TTS** requires both a voice sample (audio file) and a transcript of that sample. It supports 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian.

**Chatterbox** requires only a voice sample. It supports 23 languages and has emotion control parameters (`exaggeration` and `cfg`).

**Voice themes** offer a third option: instead of providing a voice sample, pass `--voice-theme` to select from 16 pre-defined themes (e.g. `narrator-m`, `warm-f`, `kid-m`). The theme generates a reference voice via Qwen3-TTS VoiceDesign. When used with `dub` or `speak`, the generated profile is saved to the project's `voice_profile/` directory and reused on subsequent runs.

Each segment is saved individually (`seg_0001.wav`, `seg_0002.wav`, ...) so interrupted runs can resume without re-synthesizing completed segments.

**Inputs:** `subtitles/translated.srt`, voice sample + optional transcript (or `--voice-theme`)
**Outputs:** `tts/segments/seg_NNNN.wav`, optionally `voice_profile/voice.wav` and `voice_profile/script.txt`

### 9. Assemble

Places all segment WAVs onto a silence-filled timeline that matches the original audio duration. Optionally applies tempo adjustment using ffmpeg to make segments fit their allocated time windows.

Tempo modes:

| Mode | Behavior |
|------|----------|
| `auto` | Speed up segments that overflow (never slow down) |
| `dynamic` | Per-segment speed matching, both faster and slower |
| `fixed` | Apply a constant multiplier to all segments (e.g., 1.1×) |
| `off` | No tempo change — place segments as-is |

After placement, two post-processing steps run by default:

- **Loudness matching** — measures the original audio's integrated loudness (LUFS) and normalises the dubbed track to the same level via ffmpeg `loudnorm`.
- **Background mixing** — extracts non-vocal audio (music, ambience) from the original and mixes it beneath the dubbed voice. Uses `demucs` when available, otherwise a centre-cancel technique.

Both can be disabled with `--no-loudness-match` and `--no-mix-background`. The background level is tuneable with `--background-volume` (default 0.15).

**Inputs:** `tts/segments/seg_NNNN.wav`, original audio duration, `source/audio.mp3`
**Outputs:** `tts/dubbed.wav`

### 10. Subtitle (optional)

Burns styled subtitles into the video using the ffmpeg `subtitles` filter. Can also replace the audio track with the dubbed version in the same encoding pass.

Supports font selection (system fonts, local files, or Google Fonts), color and opacity control, text positioning, outline, bold, and line spacing. Automatically detects RTL scripts (Arabic, Farsi, Hebrew) and inserts Unicode directional markers.

**Inputs:** `source/video.mp4`, `subtitles/translated.srt`, optionally `tts/dubbed.wav`
**Outputs:** `tts/dubbed.mp4`

## Data Flow

When you run the full `dub` command, data flows through the stages in order. Each stage reads from and writes to a well-defined set of paths inside the project directory. The `ProjectPaths` class manages all paths — see [Project Structure](project-structure.md).

Stages that perform LLM calls (thumbnails, describe, review, translate, resegment-merge) record their token usage. After the pipeline finishes, a summary report is logged and the raw records are saved to `llm_usage.json`. See [Configuration](configuration.md#llm-usage-tracking) for details.

## Resume and Caching

Every stage checks whether its output files already exist before running. If they do, the stage is skipped entirely. TTS synthesis is more granular — individual segment WAVs are checked, so a run interrupted at segment 150 of 300 resumes from segment 151.

This behavior is controlled by two flags:

- **`skip_existing`** (default: `True`) — skip stages whose outputs exist
- **`force_reset`** — delete all cached outputs and re-run everything from scratch

In the CLI, caching is always on. Pass `--force-reset` when you need a clean run.

## Supported Languages

### Translation (33 languages)

Arabic, Bengali, Chinese (Simplified), Chinese (Traditional), Czech, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Malay, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian, Urdu, Vietnamese.

### Qwen3-TTS (10 languages)

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.

### Chatterbox (23 languages)

Broader multilingual support — see Chatterbox documentation for the full list.

### Transcription

All backends support automatic language detection. You can also force a specific language code (e.g., `--language en`) for better accuracy.
