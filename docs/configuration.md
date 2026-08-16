# Configuration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key for transcription and LLM tasks |
| `OPENAI_BASE_URL` | — | Custom base URL for OpenAI-compatible API providers |
| `OPENAI_MODEL` | `gpt-4.1` | Default LLM model for translation, description, etc. |
| `HF_TOKEN` | — | HuggingFace token for accessing private voice profile datasets |
| `MAZINGER_PROFILES_REPO_URL` | — | Custom HuggingFace dataset URL for voice profiles |

CLI flags take precedence over environment variables. If neither is set, `OPENAI_MODEL` defaults to `gpt-4.1`.

```bash
# Set via environment
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf_..."           # For private datasets
export MAZINGER_PROFILES_REPO_URL="https://huggingface.co/datasets/YOUR_NAME/dataset/resolve/main/profiles"

mazinger dub "https://youtube.com/watch?v=VIDEO_ID" --clone-profile abubakr

# Or pass directly
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --openai-api-key "sk-..."
```

## Using a Custom LLM Provider

Any OpenAI-compatible API works. Set the base URL to point at your provider:

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --openai-base-url "https://api.your-provider.com/v1" \
    --openai-api-key "your-key" \
    --llm-model "your-model-name"
```

### Ollama (Local LLM)

To use Ollama as a local LLM provider, point the base URL at the Ollama
OpenAI-compatible endpoint and disable thinking mode for models like Qwen3
that enable it by default:

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --openai-base-url "http://localhost:11434/v1" \
    --openai-api-key "ollama" \
    --llm-model "qwen3.5:2b-q8_0" \
    --no-llm-think
```

Or in Python:

```python
dubber = MazingerDubber(
    openai_api_key="ollama",
    openai_base_url="http://localhost:11434/v1",
    llm_model="qwen3.5:2b-q8_0",
    llm_think=False,
)
```

## Caching and Resume

Every pipeline stage checks whether its output files exist before running. If they do, the stage is skipped. This makes runs idempotent and resumable.

TTS synthesis has finer granularity — individual segment WAVs (`seg_0001.wav`, `seg_0002.wav`, ...) are checked, so a run interrupted at segment 150 of 300 resumes from segment 151.

| Behavior | How to get it |
|----------|--------------|
| Resume from where you left off | Re-run the same command (default) |
| Start completely fresh | Add `--force-reset` |

`--force-reset` works with both `dub` and `speak`.

## Transcription Methods

| Method | Flag value | Model default | GPU | Cost |
|--------|-----------|---------------|-----|------|
| OpenAI Whisper API | `openai` | `whisper-1` | No | Pay per audio minute |
| faster-whisper | `faster-whisper` | `large-v3` | Yes (or CPU) | Free |
| WhisperX | `whisperx` | `large-v3` | Yes | Free |
| CohereX | `coherex` | `CohereLabs/cohere-transcribe-03-2026` | Yes | Free |
| MLX Whisper | `mlx-whisper` | `mlx-community/whisper-large-v3-turbo` | Apple Silicon | Free |

**Choosing a method:**

- Using MLX (Apple Silicon) → pick `mlx-whisper` (no CUDA needed)
- Using Chatterbox TTS → pick `openai` or `faster-whisper` (WhisperX conflicts)
- Need offline processing → pick `faster-whisper` (default)
- Need word-level alignment → pick `whisperx` with Qwen TTS (requires `transcribe-whisperx` extra)
- Transcribing Arabic → pick `coherex`, which auto-selects the dedicated Cohere Arabic model
- Transcribing one of CohereX's 14 languages → pick `coherex` for word-level alignment without the Chatterbox conflict (requires `transcribe-coherex` extra)

### CohereX

CohereX pairs Cohere Transcribe with wav2vec2 forced alignment, following the
same design as WhisperX. It covers 14 languages: `en`, `fr`, `de`, `es`, `it`,
`pt`, `nl`, `pl`, `el`, `ar`, `ja`, `zh`, `vi`, `ko`.

Two behaviours differ from the Whisper backends:

- **The source language is required.** Cohere Transcribe performs no language
  detection, and transcribes confidently in whatever language it is given — a
  wrong language yields fluent nonsense rather than an error. Always pass
  `--source-language`. When you omit it, Mazinger falls back to CohereX's
  probe-based detector, which costs one short generation per candidate
  language.
- **`--initial-prompt` is ignored.** The Cohere processor accepts only a
  language and a punctuation flag, so the metadata-derived prompt that
  Mazinger builds for Whisper does not apply. A warning is logged.

The Cohere models are gated on HuggingFace: sign in and accept the terms on
each model page once. CohereX's pyannote VAD weights ship inside the package,
so the VAD needs no authorisation — `--vad-method silero` is available as an
alternative detector, not as a way around a gate.

Authenticate in whichever way suits you:

- **Mazinger Studio** — the *Hugging Face* panel has a **Sign in with Hugging
  Face** button that runs the Hub's device-code flow in the browser.
- **CLI** — set `HF_TOKEN`, or pass `--hf-token`.

Arabic sources automatically use `CohereLabs/cohere-transcribe-arabic-07-2026`
instead of the multilingual base model. Override with `--whisper-model`.

## TTS Engines

| Feature | Qwen3-TTS | Chatterbox | MLX |
|---------|-----------|------------|-----|
| Voice cloning requires | Audio + transcript | Audio only | Audio + transcript |
| Emotion control | No | Yes (`exaggeration` param) | No |
| Pacing control | No | Yes (`cfg` param) | No |
| Languages | 10 | 23 | 10+ |
| Default model | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `ResembleAI/chatterbox` | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` |
| Hardware | CUDA GPU | CUDA GPU | Apple Silicon |

### Chatterbox Parameter Guide

| Use case | exaggeration | cfg |
|----------|:------------:|:---:|
| General use | 0.5 | 0.5 |
| Fast speakers | 0.5 | 0.3 |
| Expressive speech | 0.7 | 0.3 |

## Tempo Control

Controls how dubbed audio segments fit into the original timeline.

| Mode | CLI flags | Behavior |
|------|-----------|----------|
| Default (auto) | *(none)* | Speed up segments that overflow; never slow down |
| Dynamic | `--dynamic-tempo` | Per-segment speed matching (both faster and slower) |
| Fixed | `--fixed-tempo 1.1` | Constant multiplier applied to all segments |
| Off | neither flag, set `tempo_mode="off"` in Python | No speed adjustment — segments placed as-is |

`--max-tempo` (default: `1.3`) caps the speed-up ratio for both auto and dynamic modes.

If both `--fixed-tempo` and `--dynamic-tempo` are given, fixed tempo takes precedence.

## Translation Tuning

### Duration-Aware Word Budgets

The translator calculates a maximum word count for each subtitle entry:

```
max_words = duration_seconds × words_per_second × duration_budget
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--words-per-second` | `2.0` | Assumed speech rate in the target language |
| `--duration-budget` | `0.80` | Fraction of time allocated for dubbed speech |

Lower `duration_budget` leaves more silence between entries. Higher `words_per_second` allows more words per entry (useful for fast-paced languages).

### Technical Terms

By default, technical terms (library names, API calls, acronyms) are kept in English. To translate them:

```bash
mazinger translate --srt subs.srt --target-language Arabic \
    --translate-technical-terms
```

### Batching

Translation processes 24 subtitle entries per LLM call with an 8-entry overlap for context continuity. These defaults are not exposed as CLI flags but can be changed in the Python API:

```python
translated = translate_srt(
    srt_text, desc, thumbs, client,
    blocks_per_batch=24,
    overlap_size=8,
)
```

## LLM Usage Tracking

Every LLM call (thumbnails, describe, translate, resegment-merge) is recorded. After the pipeline completes, a summary is logged:

```
═══ LLM Usage Report ═══
  thumbnails        model=gpt-4.1                     calls=1  in=   3,420  out=    812
  describe          model=gpt-4.1                     calls=1  in=  12,105  out=    534
  translate         model=gpt-4.1                     calls=4  in=  45,230  out=  6,102
  resegment-merge   model=gpt-4.1                     calls=2  in=   8,400  out=  1,230
  ────────────────────────────────────────────────────────────────────────────
  TOTAL                                               calls=8  in=  69,155  out=  8,678
══════════════════════════
```

Raw records are saved to `<project>/llm_usage.json`:

```json
[
    {"stage": "translate", "model": "gpt-4.1", "input_tokens": 5000, "output_tokens": 2000},
    {"stage": "describe", "model": "gpt-4.1", "input_tokens": 3000, "output_tokens": 500}
]
```

### Using the tracker in Python

```python
from mazinger import LLMUsageTracker

tracker = LLMUsageTracker()
translated = translate_srt(srt_text, desc, thumbs, client, usage_tracker=tracker)
resegmented = resegment_srt(translated, client=client, usage_tracker=tracker)

print(tracker.report())
print(f"Total tokens: {tracker.total_tokens}")
```
