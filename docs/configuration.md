# Configuration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key for transcription and LLM tasks |
| `OPENAI_BASE_URL` | — | Custom base URL for OpenAI-compatible API providers |
| `OPENAI_MODEL` | `gpt-4.1` | Default LLM model for translation, description, etc. |
| `HF_TOKEN` | — | HuggingFace token for accessing private voice profile datasets |
| `MAZINGER_PROFILES_REPO_URL` | — | Custom HuggingFace dataset URL for voice profiles |
| `MAZINGER_YTDLP_PLAYER_CLIENT` | — | Override the YouTube player clients yt-dlp uses (see below) |

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

## YouTube Download Failures

YouTube serves video formats through internal "player clients", and which of
them work depends on the network your request comes from. When none of them
work, the download stage fails before listing a single format:

```
ERROR: [youtube] VIDEO_ID: The page needs to be reloaded.
ERROR: [youtube] VIDEO_ID: Requested format is not available.
ERROR: [youtube] VIDEO_ID: no video formats found
```

Despite how they read, all three mean the same thing, and none of them are
about the video — YouTube challenges requests from shared and datacenter IPs,
which is exactly what **Google Colab, a VPS, and CI runners** look like. A
telling symptom: the metadata step succeeds (you see `Resolved slug: ...`) and
only the download fails, because metadata needs no formats.

Mazinger asks yt-dlp for a client set chosen to survive this, and retries once
with yt-dlp's own defaults if it fails:

| Client | Why it is in the list |
|---|---|
| `web_safari` | Best format coverage; needs a JS runtime and a PO token on a challenged network |
| `web_embedded` | No PO token needed, but only serves embeddable videos |
| `visionos` | The only client needing no JS runtime — covers boxes without Node |
| `android_vr` | No PO token, no JS — the one that still works from a datacenter IP. Low bitrate, so it only wins when nothing else is left |

The `tv` clients are excluded: `tv` is what answers *"The page needs to be
reloaded"*, and yt-dlp puts its `tv_downgraded` variant in the default set for
**authenticated** sessions — the path taken as soon as you supply cookies.

### If it still fails

Work down this list; each step is stronger than the last.

**1. Upgrade yt-dlp.** YouTube changes often and this is usually the whole fix:

```bash
uv pip install --upgrade yt-dlp
```

**2. Supply cookies.** In the Studio, use the **🍪 YouTube Cookies** panel; from
the CLI, `--cookies cookies.txt` or `--cookies-from-browser chrome`. See
[YouTube Cookies](youtube-cookies.md). Note that cookies used from a datacenter
IP can get the account flagged — prefer a throwaway account.

**3. Install a PO-token provider.** This is the reliable fix on Colab or a VPS.
A "Proof of Origin" token is what YouTube uses to tell a genuine client from a
scraper; the provider plugin mints them and yt-dlp picks it up automatically
once installed — no configuration:

```bash
uv pip install bgutil-ytdlp-pot-provider
```

It needs Node on the `PATH`, which Mazinger already expects.

**4. Override the client list** with `MAZINGER_YTDLP_PLAYER_CLIENT`:

```bash
# Force the client that needs neither a PO token nor a JS runtime
export MAZINGER_YTDLP_PLAYER_CLIENT="android_vr"

# Hand control back to yt-dlp entirely
export MAZINGER_YTDLP_PLAYER_CLIENT="default"
```

The value is a comma-separated list matching yt-dlp's
`--extractor-args "youtube:player_client=..."`; prefix a client with `-` to
exclude it. Setting this replaces the whole ladder — one attempt, no fallback.

To see which clients still work from your machine:

```bash
yt-dlp --extractor-args "youtube:player_client=android_vr" -F "VIDEO_URL"
```

If the error instead mentions signing in or a bot check, that is authentication
rather than a blocked network — go straight to
[YouTube Cookies](youtube-cookies.md).

---

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
- Transcribing one of CohereX's 14 languages → pick `coherex` for word-level alignment without the Chatterbox conflict (in `mazinger[all]`; the Studio already defaults to it)

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
| Default (auto) | *(none)* | Per-segment matching in both directions — speed up segments that overflow their window, slow down ones that fall well short |
| Dynamic | `--dynamic-tempo` | Currently identical to the default; both resolve to the same code path |
| Fixed | `--fixed-tempo 1.1` | Constant multiplier applied to all segments |
| Off | neither flag, set `tempo_mode="off"` in Python | No speed adjustment — segments placed as-is |

`--max-tempo` (default: `1.5`) caps the speed-up ratio. Slow-down is capped
separately so speech never drags, and is skipped when the correction would be
negligible.

> `--dynamic-tempo` is currently a no-op: `auto` already does per-segment
> matching in both directions. The flag is kept for backwards compatibility.

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
