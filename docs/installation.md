# Installation

## Core Install

The base package covers video download, cloud transcription (OpenAI Whisper API), thumbnail extraction, content description, translation, re-segmentation, and subtitle embedding. No GPU required.

```bash
pip install mazinger
```

Core dependencies: `yt-dlp`, `openai`, `json-repair`, `Pillow`, `soundfile`, `numpy`, `tqdm`.

## Optional Extras

### Local Transcription

```bash
pip install "mazinger[transcribe-faster]"      # faster-whisper — CTranslate2, ~4× faster than Whisper
pip install "mazinger[transcribe-whisperx]"    # WhisperX — best word-level alignment via wav2vec2
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
pip install "mazinger[tts-mlx]"                # MLX Qwen3-TTS — Apple Silicon (M1/M2/M3/M4/M5)
pip install "mazinger[tts-pocket]"             # Pocket TTS — CPU-only, English only, 8 predefined voices
```

Pocket TTS is the only TTS backend that runs without a GPU. It's useful for CI,
low-resource machines, or when you just want to prototype without configuring CUDA.

### MLX Transcription (Apple Silicon)

```bash
pip install "mazinger[transcribe-mlx]"         # MLX Whisper — Apple Silicon
```

### Full Bundles

```bash
pip install "mazinger[all-qwen]"              # faster-whisper + Qwen3-TTS
pip install "mazinger[all-chatterbox]"        # faster-whisper + Chatterbox
pip install "mazinger[all-mlx]"               # MLX Whisper + MLX Qwen3-TTS (Apple Silicon only)
```

## Compatibility Matrix

Qwen, Chatterbox, and MLX pull different versions of `transformers` and cannot coexist in one environment. Pick one per virtual environment.

| Extra | transformers | Compatible with |
|-------|-------------|-----------------|
| `tts` (Qwen) | ≥ 4.48 | `transcribe-faster`, `transcribe-whisperx` |
| `tts-chatterbox` | == 4.46.3 | `transcribe-faster`, OpenAI transcription |
| `tts-mlx` | (mlx-audio) | `transcribe-mlx` |
| `tts-pocket` | (not required) | any transcription backend (no GPU, CPU-only) |
| `all-mlx` | (mlx-audio + mlx-whisper) | Apple Silicon only |

WhisperX requires `transformers>=4.48`, so it conflicts with Chatterbox. When using Chatterbox, choose `transcribe-faster` or the cloud-based OpenAI transcription.
> **Note:** `faster-whisper` is the recommended default for local transcription. It is lightweight, easy to install, and compatible with all TTS engines. WhisperX is still available as an optional extra (`transcribe-whisperx`) for users who need word-level alignment via wav2vec2.
## What Each Task Requires

| Task | Command | Core install | Extra needed |
|------|---------|:------------:|--------------|
| Download | `mazinger download` | yes | — |
| Transcribe (cloud, OpenAI) | `mazinger transcribe --method openai` | yes | — (OpenAI API) |
| Transcribe (cloud, Deepgram) | `mazinger transcribe --method deepgram` | no | `transcribe-deepgram` + Deepgram API |
| Transcribe (local) | `mazinger transcribe --method faster-whisper` | no | `transcribe-faster` + CUDA |
| Transcribe (local) | `mazinger transcribe --method whisperx` | no | `transcribe-whisperx` + CUDA |
| Transcribe (MLX) | `mazinger transcribe --method mlx-whisper` | no | `transcribe-mlx` + Apple Silicon |
| Thumbnails | `mazinger thumbnails` | yes | — |
| Describe | `mazinger describe` | yes | — |
| Translate | `mazinger translate` | yes | — |
| Re-segment | `mazinger resegment` | yes | — |
| Speak (Qwen) | `mazinger speak` | no | `tts` + CUDA |
| Speak (Chatterbox) | `mazinger speak --tts-engine chatterbox` | no | `tts-chatterbox` + CUDA |
| Speak (MLX) | `mazinger speak --tts-engine mlx` | no | `tts-mlx` + Apple Silicon |
| Speak (Pocket) | `mazinger speak --tts-engine pocket` | no | `tts-pocket` (CPU-only, English only) |
| Subtitle embed | `mazinger subtitle` | yes | ffmpeg only |
| Full dub (Qwen) | `mazinger dub` | no | `all-qwen` + CUDA |
| Full dub (Chatterbox) | `mazinger dub --tts-engine chatterbox` | no | `all-chatterbox` + CUDA |
| Full dub (MLX) | `mazinger dub --tts-engine mlx` | no | `all-mlx` + Apple Silicon |
| Full dub (Pocket) | `mazinger dub --tts-engine pocket` | no | `tts-pocket` (CPU-only, English only) |

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

The overrides prevent `chatterbox-tts` and `qwen-tts` from downgrading PyTorch and other packages that Colab ships with pre-configured CUDA support.

## Flash Attention (Optional)

Speeds up TTS inference on supported GPUs. Not required — Chatterbox falls back to standard attention automatically.

```bash
uv pip install --no-build-isolation flash-attn
```

Or include it as an extra:

```bash
pip install "mazinger[flash-attn]"
```
