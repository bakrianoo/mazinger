# Installation

## Recommended Install (Default)

One command installs everything needed to run the full Mazinger pipeline and the Studio web UI:

```bash
pip install "mazinger[all]"
```

This bundles a single, mutually-compatible set of engines:

| Component | Purpose |
|-----------|---------|
| `faster-whisper` | Local GPU transcription (default) |
| `deepgram-sdk` | Cloud transcription via Deepgram Nova 3 ($200 free credit, no GPU) |
| Qwen3-TTS | Voice-cloned TTS (reference audio + transcript) — vendored, no separate package |
| `omnivoice` | 24-language zero-shot voice cloning TTS |
| `demucs` | Background-audio separation for clean voice mixing |
| `gradio` | Mazinger Studio web UI |

After installing, launch the Studio:

```bash
mazinger web --with-ollama --with-faster-whisper
```

`--with-ollama` installs Ollama, starts its server and pulls the LLM up front. It is an optimisation, not a requirement: with plain `mazinger web`, Studio performs the same setup automatically the first time you start a mission with the **Ollama (Local — Free)** provider, showing progress in the status box. Point `OLLAMA_HOST` at another machine to use a remote Ollama server instead (nothing is installed locally in that case).

> **Compatibility:** Chatterbox and MLX TTS pull conflicting `transformers` / platform requirements and are **not** included in `[all]`. Use the dedicated bundles below if you need them.

## Core-Only Install

If you only need download, cloud transcription (OpenAI Whisper API), thumbnails, description, translation, re-segmentation, and subtitle embedding — and want the smallest possible footprint — install the core package without extras:

```bash
pip install mazinger
```

Core dependencies: `yt-dlp`, `openai`, `json-repair`, `Pillow`, `soundfile`, `numpy`, `tqdm`.

## Advanced / Alternative Extras

The extras below are for users who need an engine **not** included in `[all]`, or who want to install a smaller subset.

### Local Transcription

```bash
pip install "mazinger[transcribe-faster]"      # faster-whisper — CTranslate2, ~4× faster than Whisper
pip install "mazinger[transcribe-whisperx]"    # WhisperX — best word-level alignment via wav2vec2
pip install "mazinger[transcribe-coherex]"     # CohereX — Cohere Transcribe + wav2vec2 (14 languages, Arabic model)
```

Both require a CUDA GPU (or can fall back to CPU at reduced speed).

### Cloud Transcription (no GPU required)

```bash
pip install "mazinger[transcribe-deepgram]"    # Deepgram Nova 3 — 47+ languages, free $200 credit
```

Deepgram offers strong multilingual quality (including Arabic) and gives new accounts $200
in free credits without a credit card. Set `DEEPGRAM_API_KEY` and use `--method deepgram`.

### Voice Synthesis (TTS)

```bash
pip install "mazinger[tts]"                    # Qwen3-TTS — needs a voice sample + transcript
pip install "mazinger[tts-chatterbox]"         # Chatterbox — needs only a voice sample, has emotion control
pip install "mazinger[tts-omnivoice]"          # OmniVoice — 24 languages, zero-shot voice cloning
pip install "mazinger[tts-mlx]"                # MLX Qwen3-TTS — Apple Silicon (M1/M2/M3/M4/M5)
```

### MLX Transcription (Apple Silicon)

```bash
pip install "mazinger[transcribe-mlx]"         # MLX Whisper — Apple Silicon
```

### Alternative Full Bundles

Use these only if `[all]` doesn't fit your environment (e.g. you need Chatterbox, or you're on Apple Silicon).

```bash
pip install "mazinger[all-qwen]"              # faster-whisper + Qwen3-TTS
pip install "mazinger[all-chatterbox]"        # faster-whisper + Chatterbox (separate env required)
pip install "mazinger[all-omnivoice]"         # faster-whisper + OmniVoice (24 languages)
pip install "mazinger[all-mlx]"               # MLX Whisper + MLX Qwen3-TTS (Apple Silicon only)
```

## Compatibility Matrix

Qwen, Chatterbox, and MLX pull different versions of `transformers` and cannot coexist in one environment. Pick one per virtual environment.

| Extra | transformers | Compatible with |
|-------|-------------|-----------------|
| `tts` (Qwen) | ≥ 4.48 | `transcribe-faster`, `transcribe-whisperx`, `transcribe-coherex` |
| `tts-chatterbox` | == 4.46.3 | `transcribe-faster`, OpenAI transcription || `tts-omnivoice` | (omnivoice) | `transcribe-faster`, `transcribe-whisperx`, `transcribe-coherex` || `tts-mlx` | (mlx-audio) | `transcribe-mlx` |
| `transcribe-coherex` | ≥ 5.4 | `tts`, `tts-omnivoice`, `transcribe-faster` (not `tts-chatterbox`) |
| `all-mlx` | (mlx-audio + mlx-whisper) | Apple Silicon only |

WhisperX requires `transformers>=4.48`, so it conflicts with Chatterbox. When using Chatterbox, choose `transcribe-faster` or the cloud-based OpenAI transcription.

CohereX requires `transformers>=5.4`, which matches what Qwen TTS and OmniVoice
already resolve to. It adds roughly 50 packages (the pyannote/lightning stack),
so it is kept out of `mazinger[all]` — but it resolves cleanly alongside it:

```bash
uv pip install "mazinger[all,transcribe-coherex]"   # or: mazinger[all-coherex]
```
> **Note:** `faster-whisper` is the recommended default for local transcription. It is lightweight, easy to install, and compatible with all TTS engines. WhisperX is still available as an optional extra (`transcribe-whisperx`) for users who need word-level alignment via wav2vec2.
## What Each Task Requires

| Task | Command | Core install | Extra needed |
|------|---------|:------------:|--------------|
| Download | `mazinger download` | yes | — |
| Transcribe (cloud, OpenAI) | `mazinger transcribe --method openai` | yes | — (OpenAI API) |
| Transcribe (cloud, Deepgram) | `mazinger transcribe --method deepgram` | no | `transcribe-deepgram` + Deepgram API |
| Transcribe (local) | `mazinger transcribe --method faster-whisper` | no | `transcribe-faster` + CUDA |
| Transcribe (local) | `mazinger transcribe --method whisperx` | no | `transcribe-whisperx` + CUDA |
| Transcribe (local) | `mazinger transcribe --method coherex` | no | `transcribe-coherex` + CUDA + HuggingFace sign-in |
| Transcribe (MLX) | `mazinger transcribe --method mlx-whisper` | no | `transcribe-mlx` + Apple Silicon |
| Thumbnails | `mazinger thumbnails` | yes | — |
| Describe | `mazinger describe` | yes | — |
| Translate | `mazinger translate` | yes | — |
| Re-segment | `mazinger resegment` | yes | — |
| Speak (Qwen) | `mazinger speak` | no | `tts` + CUDA |
| Speak (Chatterbox) | `mazinger speak --tts-engine chatterbox` | no | `tts-chatterbox` + CUDA |
| Speak (OmniVoice) | `mazinger speak --tts-engine omnivoice` | no | `tts-omnivoice` + CUDA |
| Speak (MLX) | `mazinger speak --tts-engine mlx` | no | `tts-mlx` + Apple Silicon |
| Subtitle embed | `mazinger subtitle` | yes | ffmpeg only |
| Full dub (Qwen) | `mazinger dub` | no | `all-qwen` + CUDA |
| Full dub (Chatterbox) | `mazinger dub --tts-engine chatterbox` | no | `all-chatterbox` + CUDA |
| Full dub (OmniVoice) | `mazinger dub --tts-engine omnivoice` | no | `all-omnivoice` + CUDA |
| Full dub (MLX) | `mazinger dub --tts-engine mlx` | no | `all-mlx` + Apple Silicon |

## System Dependencies

| Tool | Used by | Install |
|------|---------|---------|
| ffmpeg | download, thumbnails, assemble, subtitle | `apt install ffmpeg` / `brew install ffmpeg` |
| ffprobe | assemble (duration detection) | Bundled with ffmpeg |
| CUDA GPU + drivers | local transcription, TTS | NVIDIA driver + CUDA toolkit |

## Environment Recipes

### Fresh venv with Chatterbox (Python 3.12)

```bash
uv venv .venv --python 3.12
source .venv/bin/activate

# numpy must exist before the pkuseg C extension compiles
uv pip install "numpy>=1.26"

# CUDA-enabled PyTorch — adjust cu128 to match your driver
uv pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

uv pip install --no-build-isolation "mazinger[all-chatterbox]"
```

### Fresh venv with Qwen (Python 3.12)

```bash
uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

cat > /tmp/qwen_overrides.txt << 'EOF'
torch>=2.0
torchaudio>=2.0
EOF

uv pip install --override /tmp/qwen_overrides.txt "mazinger[all-qwen]"
```

### Fresh venv with OmniVoice (Python 3.12)

```bash
uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

uv pip install "mazinger[all-omnivoice]"
```

### Google Colab — Chatterbox

```bash
cat > /tmp/cb_overrides.txt << 'EOF'
torch>=2.0
torchaudio>=2.0
numpy>=1.26
pandas>=2.2
gradio>=5.0
safetensors>=0.3
EOF

uv pip install --system --no-build-isolation \
    --override /tmp/cb_overrides.txt \
    "mazinger[all-chatterbox]"
```

### Google Colab — Qwen

```bash
cat > /tmp/qwen_overrides.txt << 'EOF'
torch>=2.0
torchaudio>=2.0
EOF

uv pip install --system \
    --override /tmp/qwen_overrides.txt \
    "mazinger[all-qwen]"
```

The overrides prevent `chatterbox-tts` from downgrading PyTorch and other packages that Colab ships with pre-configured CUDA support.

### Why Qwen3-TTS is vendored

Mazinger ships its own copy of Qwen3-TTS under `mazinger/_vendor/qwen_tts`
instead of installing the `qwen-tts` package. The published release hard-pins
`transformers==4.57.3`, which cannot coexist with CohereX (`>=5.4`) or
OmniVoice (`>=5.3`) in one environment — and a pin cannot be relaxed from
outside the package. The vendored copy carries the (small) set of changes
needed to run on transformers 5.x; they are catalogued in
[`mazinger/_vendor/qwen_tts/NOTICE.md`](../mazinger/_vendor/qwen_tts/NOTICE.md).

Nothing about this is visible at install time — `mazinger[tts]` and
`mazinger[all]` behave exactly as before, minus the conflicting pin.

## Flash Attention (Optional)

Speeds up TTS inference on supported GPUs. Not required — Chatterbox falls back to standard attention automatically.

```bash
uv pip install --no-build-isolation flash-attn
```

Or include it as an extra:

```bash
pip install "mazinger[flash-attn]"
```
