# CLI Reference

All commands follow the pattern `mazinger <command> [options]`. Run `mazinger <command> --help` for the built-in help text.

## Global Options

These options are available on every command:

| Flag | Default | Description |
|------|---------|-------------|
| `--base-dir` | `./mazinger_output` | Root directory for project output |
| `--verbose` | off | Enable debug logging |

## dub

Run the full pipeline: download, transcribe, translate, synthesize, assemble.

```bash
mazinger dub <source> [options]
```

**Positional arguments:**

| Argument | Description |
|----------|-------------|
| `source` | Video URL, local video file, or local audio file (required) |

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--slug` | auto-generated | Project slug (directory name) |
| `--quality` | best available | Video quality: `low`, `medium`, `high`, or numeric height (e.g., `1080`) |
| `--cookies-from-browser` | — | Browser name for yt-dlp cookie extraction |
| `--cookies` | — | Path to a Netscape cookies.txt file |
| `--clone-profile` | — | Voice profile name from HuggingFace or local directory path |
| `--voice-theme` | — | Pre-defined voice theme (e.g. `narrator-m`, `warm-f`). See `mazinger profile list` |
| `--voice-sample` | — | Path to reference voice audio file |
| `--voice-script` | — | Path to transcript of the voice sample (or inline text) |
| `--transcribe-method` | `openai` | `openai`, `faster-whisper`, or `whisperx` |
| `--whisper-model` | varies by method | Whisper model name |
| `--beam-size` | `5` | Beam size for decoding (faster-whisper/whisperx) |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--source-language` | `auto` | Source language for translation (or `auto` to detect) |
| `--target-language` | `English` | Target language for translation |
| `--words-per-second` | `2.0` | Speech rate used for duration-aware word budgets |
| `--duration-budget` | `0.80` | Fraction of available time for dubbed speech |
| `--translate-technical-terms` | off | Translate technical terms instead of keeping them in English |
| `--asr-review` | off | Review ASR transcript with LLM to fix typos and punctuation |
| `--keep-technical-english` | off | Convert technical terms to English in the source transcript (requires `--asr-review`) |
| `--youtube-subs` | off | Download YouTube subtitles and compare with ASR to pick the best source |
| `--tts-engine` | `qwen` | `qwen` or `chatterbox` |
| `--tts-model` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Qwen model ID |
| `--chatterbox-model` | `ResembleAI/chatterbox` | Chatterbox model ID |
| `--tts-language` | same as `--target-language` | Language hint for TTS |
| `--chatterbox-exaggeration` | `0.5` | Emotion intensity (0.0–1.0) |
| `--chatterbox-cfg` | `0.5` | Pacing control (0.0–1.0) |
| `--use-resegmented` | off | Use resegmented SRT instead of raw transcription |
| `--output-type` | `audio` | `audio` (WAV only) or `video` (muxed MP4) |
| `--embed-subtitles` | off | Burn subtitles into output video (implies `--output-type video`) |
| `--subtitle-source` | `translated` | `translated`, `original`, or path to a custom SRT file |
| `--dynamic-tempo` | off | Per-segment speed matching |
| `--fixed-tempo` | — | Constant speed multiplier (e.g., `1.1`) |
| `--max-tempo` | `1.3` | Maximum speed-up for dynamic/auto tempo |
| `--no-loudness-match` | off | Skip loudness normalisation against the original audio |
| `--no-mix-background` | off | Skip mixing background audio from the original |
| `--background-volume` | `0.15` | Background audio mix level (0.0–1.0) |
| `--start` | — | Start timestamp for slicing (e.g. `00:01:30` or `90`) |
| `--end` | — | End timestamp for slicing (e.g. `00:05:00` or `300`) |
| `--force-reset` | off | Discard all cached outputs and re-run from scratch |
| `--openai-api-key` | `$OPENAI_API_KEY` | OpenAI API key |
| `--openai-base-url` | `$OPENAI_BASE_URL` | Custom API base URL |
| `--llm-model` | `gpt-4.1` | LLM model for translation/analysis |
| `--llm-think` / `--no-llm-think` | — | Enable/disable LLM thinking mode (use `--no-llm-think` for Ollama Qwen3) |

All `--subtitle-*` styling flags are also accepted. See [Subtitle Styling](subtitle-styling.md).

**Examples:**

```bash
# Auto-clone the speaker's voice (simplest — no voice flags needed)
mazinger dub "https://youtube.com/watch?v=abc123" \
    --target-language Spanish

# Dub with a voice theme (easiest)
mazinger dub "https://youtube.com/watch?v=abc123" \
    --voice-theme narrator-m --target-language Spanish

# Basic dub with a profile
mazinger dub "https://youtube.com/watch?v=abc123" \
    --clone-profile abubakr --target-language Arabic

# Dub with Chatterbox, video output, subtitles
mazinger dub ./lecture.mp4 \
    --voice-sample speaker.m4a \
    --tts-engine chatterbox \
    --output-type video \
    --embed-subtitles \
    --target-language Spanish

# Dub only a portion of the video
mazinger dub "https://youtube.com/watch?v=abc123" \
    --clone-profile abubakr --target-language Arabic \
    --start 00:01:30 --end 00:05:00

# Local transcription, dynamic tempo
mazinger dub "https://youtube.com/watch?v=abc123" \
    --clone-profile abubakr \
    --transcribe-method faster-whisper \
    --dynamic-tempo --max-tempo 1.3
```

---

## download

Download a video and extract its audio track.

```bash
mazinger download <source> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--slug` | auto-generated | Project slug |
| `--quality` | best | Video quality |
| `--cookies-from-browser` | — | Browser name for yt-dlp cookies |
| `--cookies` | — | Path to cookies.txt |
| `--start` | — | Start timestamp for slicing (e.g. `00:01:30` or `90`) |
| `--end` | — | End timestamp for slicing (e.g. `00:05:00` or `300`) |

**Example:**

```bash
mazinger download "https://youtube.com/watch?v=abc123" --base-dir ./output --quality 720

# Download and extract only a segment
mazinger download "https://youtube.com/watch?v=abc123" --start 00:02:00 --end 00:04:00
```

---

## slice

Extract a time range from a video or audio file.

```bash
mazinger slice <source> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--slug` | auto-generated | Project slug |
| `--quality` | best | Video quality |
| `--cookies-from-browser` | — | Browser name for yt-dlp cookies |
| `--cookies` | — | Path to cookies.txt |
| `--start` | — | Start timestamp (e.g. `00:01:30` or `90`) |
| `--end` | — | End timestamp (e.g. `00:05:00` or `300`) |

**Examples:**

```bash
# Extract a 3-minute clip from a YouTube video
mazinger slice "https://youtube.com/watch?v=abc123" --start 00:01:00 --end 00:04:00

# Extract from a local file
mazinger slice ./lecture.mp4 --start 90 --end 300
```

---

## transcribe

Convert audio to SRT subtitles.

```bash
mazinger transcribe [source] [options]
```

If `source` is provided, the video is downloaded first and its audio is transcribed. Otherwise, use `--audio` to point to an existing audio file.

| Flag | Default | Description |
|------|---------|-------------|
| `--audio` | — | Path to audio file (overrides source) |
| `-o`, `--output` | — | Output SRT path |
| `--method` | `openai` | `openai`, `faster-whisper`, or `whisperx` |
| `--model` | varies | Whisper model name (`whisper-1` for OpenAI, `large-v3` for local) |
| `--device` | `auto` | `auto`, `cuda`, `cpu` |
| `--batch-size` | `16` | Batch size for local transcription |
| `--compute-type` | `float16` | Weight precision: `float16`, `int8`, `int8_float16` |
| `--beam-size` | `5` | Beam size for decoding (default: 5) |
| `--language` | auto-detect | Force a language code (e.g., `en`, `ar`, `fr`) |
| `--initial-prompt` | — | Initial text to condition Whisper (e.g., domain terms, video title) |
| `--no-condition-on-previous-text` | off | Disable conditioning on previous segment text |
| `--max-chars` | `84` | Max characters per subtitle entry |
| `--max-duration` | `5.0` | Max seconds per subtitle entry |
| `--no-resegment` | off | Skip the post-transcription resegmentation step |
| `--refine` | off | Use LLM to add punctuation and fix misheard words |
| `--asr-review` | off | Review transcript with LLM: fix typos, punctuation, and optionally normalise technical terms |
| `--keep-technical-english` | off | Convert technical terms to English (requires `--asr-review`) |
| `--llm-model` | `gpt-4.1` | LLM model for refinement |
| `--llm-think` / `--no-llm-think` | — | Enable/disable LLM thinking mode |
| `--openai-api-key` | `$OPENAI_API_KEY` | OpenAI API key (for cloud method) |

**Examples:**

```bash
# Cloud transcription
mazinger transcribe --audio recording.mp3 -o subs.srt

# Local with faster-whisper on GPU
mazinger transcribe --audio recording.mp3 -o subs.srt \
    --method faster-whisper --device cuda

# From a URL, auto-download first
mazinger transcribe "https://youtube.com/watch?v=abc123" --base-dir ./output
```

---

## thumbnails

Extract LLM-selected key frames from a video.

```bash
mazinger thumbnails [source] [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | — | Path to video file |
| `--srt` | — | Path to SRT file |
| `--output-dir` | — | Output directory for thumbnails |
| `--meta` | — | Path to save metadata JSON |
| `--openai-api-key` | `$OPENAI_API_KEY` | OpenAI API key |
| `--llm-model` | `gpt-4.1` | LLM model |
| `--llm-think` / `--no-llm-think` | — | Enable/disable LLM thinking mode |
| `--transcribe-method` | `openai` | Transcription method (if SRT not provided) |
| `--whisper-model` | varies | Whisper model |

**Example:**

```bash
mazinger thumbnails --video video.mp4 --srt subs.srt --output-dir ./thumbs
```

---

## describe

Generate a structured content analysis (title, summary, key points, keywords).

```bash
mazinger describe [source] [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--srt` | — | Path to SRT file |
| `--thumbnails-meta` | — | Path to thumbnails meta.json |
| `-o`, `--output` | — | Output JSON path |
| `--openai-api-key` | `$OPENAI_API_KEY` | OpenAI API key |
| `--llm-model` | `gpt-4.1` | LLM model |
| `--llm-think` / `--no-llm-think` | — | Enable/disable LLM thinking mode |

**Example:**

```bash
mazinger describe --srt subs.srt --thumbnails-meta ./thumbs/meta.json -o description.json
```

---

## translate

Translate SRT subtitles into another language.

```bash
mazinger translate [source] [options]
```

If `source` is provided, the video is downloaded, transcribed, and translated automatically.

| Flag | Default | Description |
|------|---------|-------------|
| `--srt` | — | Path to source SRT (overrides auto-transcription) |
| `--description` | — | Path to description JSON |
| `--thumbnails-meta` | — | Path to thumbnails meta.json |
| `-o`, `--output` | — | Output SRT path |
| `--source-language` | `auto` | Source language |
| `--target-language` | `English` | Target language |
| `--words-per-second` | `2.0` | Speech rate for word budget calculation |
| `--duration-budget` | `0.80` | Fraction of time allocated for dubbed speech |
| `--translate-technical-terms` | off | Translate technical terms |
| `--video` | — | Source video for subtitle embedding |
| `--video-output` | — | Output video path (when embedding subtitles) |
| `--embed-subtitles` | off | Burn translated subtitles into video |
| `--openai-api-key` | `$OPENAI_API_KEY` | OpenAI API key |
| `--llm-model` | `gpt-4.1` | LLM model |
| `--llm-think` / `--no-llm-think` | — | Enable/disable LLM thinking mode |
| `--transcribe-method` | `openai` | Transcription method (if SRT not provided) |

All `--subtitle-*` styling flags are accepted when `--embed-subtitles` is set.

**Examples:**

```bash
# Translate an existing SRT
mazinger translate --srt subs.srt --target-language French -o translated.srt

# Full auto: download, transcribe, translate, burn subtitles
mazinger translate "https://youtube.com/watch?v=abc123" \
    --target-language Arabic \
    --embed-subtitles \
    --subtitle-google-font "Noto Sans Arabic"
```

---

## resegment

Re-segment subtitles for readability by merging fragments and splitting long entries.

```bash
mazinger resegment [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--srt` | — | Path to input SRT (required) |
| `-o`, `--output` | — | Output SRT path (required) |
| `--max-chars` | `84` | Max characters per subtitle entry |
| `--max-dur` | `4.0` | Max seconds per subtitle entry |
| `--openai-api-key` | `$OPENAI_API_KEY` | OpenAI API key (optional — falls back to rules) |
| `--llm-model` | `gpt-4.1` | LLM model |
| `--llm-think` / `--no-llm-think` | — | Enable/disable LLM thinking mode |

**Example:**

```bash
mazinger resegment --srt translated.srt -o final.srt --max-chars 80 --max-dur 5.0
```

---

## speak

Synthesize dubbed audio from an SRT file using voice-cloned TTS.

```bash
mazinger speak [source] [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--srt` | — | Path to translated SRT |
| `--original-audio` | — | Original audio file (for duration matching) |
| `--clone-profile` | — | Voice profile name from HuggingFace or local directory path |
| `--voice-theme` | — | Pre-defined voice theme (e.g. `narrator-m`, `warm-f`) |
| `--voice-sample` | — | Path to reference voice audio |
| `--voice-script` | — | Path to transcript of voice sample |
| `-o`, `--output` | — | Output WAV path |
| `--segments-dir` | — | Directory for individual segment WAVs |
| `--tts-engine` | `qwen` | `qwen` or `chatterbox` |
| `--tts-model` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Qwen model ID |
| `--chatterbox-model` | `ResembleAI/chatterbox` | Chatterbox model ID |
| `--tts-language` | — | Language hint for TTS |
| `--chatterbox-exaggeration` | `0.5` | Emotion intensity (0.0–1.0) |
| `--chatterbox-cfg` | `0.5` | Pacing control (0.0–1.0) |
| `--device` | `auto` | `auto`, `cuda`, `cpu` |
| `--dtype` | `bfloat16` | Weight dtype for Qwen: `bfloat16`, `float16`, `float32` |
| `--dynamic-tempo` | off | Per-segment tempo matching |
| `--fixed-tempo` | — | Constant speed multiplier |
| `--max-tempo` | `1.3` | Maximum speed-up ratio |
| `--force-reset` | off | Re-synthesize all segments from scratch |

**Examples:**

```bash
# With a voice theme
mazinger speak --srt translated.srt --original-audio audio.mp3 \
    --voice-theme warm-f -o dubbed.wav

# Qwen with a profile
mazinger speak --srt translated.srt --original-audio audio.mp3 \
    --clone-profile abubakr -o dubbed.wav

# Chatterbox with emotion
mazinger speak --srt translated.srt --original-audio audio.mp3 \
    --voice-sample speaker.m4a \
    --tts-engine chatterbox \
    --chatterbox-exaggeration 0.7 --chatterbox-cfg 0.3 \
    -o dubbed.wav

# Fixed tempo speed-up
mazinger speak --srt translated.srt --original-audio audio.mp3 \
    --clone-profile abubakr --fixed-tempo 1.1 -o dubbed.wav
```

---

## profile

List available voice themes or generate a reusable voice profile from a theme.

### profile list

List all available voice themes.

```bash
mazinger profile list
```

Displays all 16 pre-defined themes with name, gender, and supported languages.

### profile generate

Generate a voice profile directory from a theme.

```bash
mazinger profile generate <theme> <language> -o <output-dir> [options]
```

| Argument / Flag | Default | Description |
|-----------------|---------|-------------|
| `theme` | (required) | Theme name (from `profile list`) |
| `language` | (required) | Target language (e.g. `English`, `Spanish`) |
| `-o`, `--output` | (required) | Output directory for the profile |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--dtype` | `bfloat16` | Weight dtype for VoiceDesign model |

The output directory will contain `voice.wav` and `script.txt`, suitable for use with `--clone-profile <path>`.

**Examples:**

```bash
# Generate a narrator profile for Spanish
mazinger profile generate narrator-m Spanish -o ./my-narrator

# Use the generated profile
mazinger dub "https://youtube.com/watch?v=abc123" \
    --clone-profile ./my-narrator --target-language Spanish
```

---

## subtitle

Burn subtitles into a video file, optionally replacing the audio track.

```bash
mazinger subtitle [source] [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | — | Source video path |
| `--srt` | — | SRT file path |
| `--audio` | — | Replacement audio track |
| `-o`, `--output` | — | Output video path |
| `--openai-api-key` | `$OPENAI_API_KEY` | OpenAI API key (if auto-translating) |
| `--llm-model` | `gpt-4.1` | LLM model |
| `--llm-think` / `--no-llm-think` | — | Enable/disable LLM thinking mode |
| `--transcribe-method` | `openai` | Transcription method |
| `--source-language` | `auto` | Source language |
| `--target-language` | `English` | Target language |

All `--subtitle-*` styling flags are accepted. See [Subtitle Styling](subtitle-styling.md).

**Examples:**

```bash
# Burn subtitles, keep original audio
mazinger subtitle video.mp4 --srt translated.srt -o output.mp4

# Burn subtitles and replace audio
mazinger subtitle video.mp4 --srt translated.srt --audio dubbed.wav -o output.mp4

# With custom styling
mazinger subtitle video.mp4 --srt translated.srt -o output.mp4 \
    --subtitle-font-size 28 \
    --subtitle-font-color yellow \
    --subtitle-bg-color black \
    --subtitle-bg-alpha 0.8 \
    --subtitle-position bottom \
    --subtitle-bold
```
