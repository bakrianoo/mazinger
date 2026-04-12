"""Transcribe audio to SRT using OpenAI Whisper, faster-whisper, WhisperX, MLX Whisper, or Deepgram.

Supports five transcription backends:
- **openai** (default): Uses OpenAI's Whisper API. No local GPU required,
  simple setup, works with any transformers version. Requires OPENAI_API_KEY.
- **faster-whisper**: Fast local transcription using CTranslate2. 4x faster
  than original Whisper with lower memory usage. No transformers dependency,
  compatible with Chatterbox TTS. Requires [transcribe-faster] extra.
- **whisperx**: Local transcription with word-level alignment via wav2vec2.
  Requires PyTorch + CUDA and the [transcribe-whisperx] extra. Not compatible
  with chatterbox-tts due to conflicting transformers requirements.
- **mlx-whisper**: MLX-accelerated Whisper for Apple Silicon. Runs natively
  on M-series GPUs. Requires [transcribe-mlx] extra.
- **deepgram**: Uses Deepgram's Nova API for fast, accurate cloud
  transcription with word-level timestamps. Supports 47+ languages.
  Requires DEEPGRAM_API_KEY. Requires [transcribe-deepgram] extra.
"""

from __future__ import annotations

import gc
import logging
import os
import re
from typing import Any, Literal

log = logging.getLogger(__name__)

# Supported transcription methods
TranscribeMethod = Literal["openai", "faster-whisper", "whisperx", "mlx-whisper", "deepgram"]

# Module-level cache for faster-whisper models — avoids reloading across runs
_whisper_cache: dict[str, Any] = {}

# Default MLX Whisper model
DEFAULT_MLX_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


def build_initial_prompt(video_meta: dict | None) -> str | None:
    """Build a Whisper initial prompt from video metadata.

    Whisper uses the initial prompt to condition its decoder on expected
    vocabulary — domain-specific terms, channel names, and topic keywords
    that appear in the title/description are far less likely to be misheard.

    Returns ``None`` when there is nothing useful to include.
    """
    if not video_meta:
        return None

    parts: list[str] = []
    title = video_meta.get("title", "").strip()
    if title:
        parts.append(title)

    # Add the first 200 chars of description (enough for topic context,
    # avoids bloating the prompt with links/timestamps/etc.)
    desc = video_meta.get("description", "").strip()
    if desc:
        # Take first paragraph or first 200 chars, whichever is shorter
        first_para = desc.split("\n\n")[0].strip()
        parts.append(first_para[:200])

    tags = video_meta.get("tags")
    if tags and isinstance(tags, list):
        parts.append(", ".join(tags[:15]))

    if not parts:
        return None

    prompt = ". ".join(parts)
    # Whisper initial_prompt is limited to 224 tokens (~800 chars);
    # truncate to stay safely within that window.
    if len(prompt) > 800:
        prompt = prompt[:800]
    return prompt


def clear_cache() -> None:
    """Remove all cached Whisper models and free GPU memory."""
    if not _whisper_cache:
        return
    import torch

    _whisper_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Whisper model cache cleared, GPU memory freed.")


# ── SRT formatting (self-contained so this module has no intra-package deps) ──

def _fmt_srt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int(round((s % 1) * 1000))
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _segments_to_srt(segments: list[dict]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


# ── Post-transcription text cleanup ──────────────────────────────────────────

# Phantom subtitle that some Whisper models hallucinate on silence
_PHANTOM_PATTERNS = [
    re.compile(r"ترجمة\s+نانسي\s+قنقر"),
]

# Character repeated 3+ times consecutively → collapse to single occurrence
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")

# Same word repeated 3+ times consecutively → keep one
_REPEATED_WORD_RE = re.compile(r"\b(\S+)(?:\s+\1){2,}\b")


def _clean_text(text: str) -> str:
    """Fix common Whisper transcription artifacts in a single subtitle line."""
    # Remove phantom/hallucinated subtitles
    for pat in _PHANTOM_PATTERNS:
        text = pat.sub("", text)

    # Collapse characters repeated 3+ times (e.g. هنااااااك → هناك)
    text = _REPEATED_CHAR_RE.sub(r"\1", text)

    # Collapse words repeated 3+ times (e.g. كذا كذا كذا كذا → كذا)
    text = _REPEATED_WORD_RE.sub(r"\1", text)

    # Normalize whitespace
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _clean_segments(segments: list[dict]) -> list[dict]:
    """Apply text cleanup to all segments and drop empty ones."""
    cleaned: list[dict] = []
    for seg in segments:
        seg = dict(seg)
        seg["text"] = _clean_text(seg["text"])
        if seg["text"]:
            cleaned.append(seg)
    dropped = len(segments) - len(cleaned)
    if dropped:
        log.info("Dropped %d empty segments after text cleanup", dropped)
    return cleaned


# ── Resegmentation helpers ────────────────────────────────────────────────────

def _split_by_words(
    words: list[dict],
    max_chars: int,
    max_dur: float,
) -> list[dict]:
    """Accumulate words into chunks, preferring splits at natural pauses.

    Instead of chopping blindly when a limit is hit, we track the best
    split point seen so far — defined as the word boundary with the longest
    silence gap.  When a limit *would* be exceeded we flush at that best
    point, keeping phrases like "ان شاء الله" intact.

    Falls back to a hard split only when no pause-based candidate exists
    (i.e. the very first word already exceeds a limit).
    """
    chunks: list[dict] = []

    buf_words: list[dict] = []      # words accumulated in current chunk
    buf_text = ""
    buf_start: float | None = None

    # Track the best split *inside* the current buffer:
    #   best_idx   – index in buf_words *after* which we'd flush
    #   best_gap   – silence duration at that boundary
    best_idx: int | None = None
    best_gap: float = -1.0

    def _flush(up_to: int | None = None) -> None:
        """Flush buf_words[0:up_to] as a chunk; keep the rest."""
        nonlocal buf_words, buf_text, buf_start, best_idx, best_gap

        if up_to is None:
            up_to = len(buf_words)

        flush_words = buf_words[:up_to]
        keep_words = buf_words[up_to:]

        if flush_words:
            flush_text = " ".join(w.get("word", "") for w in flush_words).strip()
            flush_end = flush_words[-1].get("end", buf_start)
            chunks.append({"start": buf_start, "end": flush_end, "text": flush_text})

        # Reset buffer to the remaining words
        buf_words = keep_words
        if keep_words:
            buf_text = " ".join(w.get("word", "") for w in keep_words).strip()
            buf_start = next(
                (w.get("start") for w in keep_words if w.get("start") is not None),
                None,
            )
        else:
            buf_text = ""
            buf_start = None

        best_idx = None
        best_gap = -1.0

    for w in words:
        word_text = w.get("word", "")
        w_start = w.get("start")
        w_end = w.get("end")

        # Words without timestamps — just attach to buffer
        if w_start is None or w_end is None:
            candidate = (buf_text + " " + word_text).strip() if buf_text else word_text
            buf_text = candidate
            buf_words.append(w)
            continue

        candidate = (buf_text + " " + word_text).strip() if buf_text else word_text
        candidate_dur = w_end - (buf_start if buf_start is not None else w_start)

        would_exceed = buf_text and (
            len(candidate) > max_chars or candidate_dur > max_dur
        )

        if would_exceed:
            if best_idx is not None and best_idx > 0:
                # Flush up to the best pause point we found
                _flush(up_to=best_idx + 1)
                # Re-evaluate: add current word to the new (shorter) buffer
                candidate = (buf_text + " " + word_text).strip() if buf_text else word_text
                buf_words.append(w)
                buf_text = candidate
                if buf_start is None:
                    buf_start = w_start
            else:
                # No good pause point — hard flush everything in buffer
                _flush()
                buf_words = [w]
                buf_text = word_text
                buf_start = w_start
        else:
            # Track the silence gap before this word as a potential split point
            if buf_words:
                prev_end = buf_words[-1].get("end")
                if prev_end is not None and w_start is not None:
                    gap = w_start - prev_end
                    if gap > best_gap:
                        best_gap = gap
                        best_idx = len(buf_words) - 1  # split *after* this index

            buf_words.append(w)
            buf_text = candidate
            if buf_start is None:
                buf_start = w_start

    # Flush remaining
    if buf_text.strip():
        _flush()

    return chunks


def _split_proportional(
    text: str,
    start: float,
    end: float,
    max_chars: int,
) -> list[dict]:
    """Split text at word boundaries with proportional timestamp distribution."""
    words = text.split()
    text_chunks: list[str] = []
    buf = ""

    for word in words:
        candidate = (buf + " " + word).strip() if buf else word
        if buf and len(candidate) > max_chars:
            text_chunks.append(buf.strip())
            buf = word
        else:
            buf = candidate
    if buf.strip():
        text_chunks.append(buf.strip())

    total_chars = len(text) or 1
    total_dur = end - start
    result: list[dict] = []
    t = start
    for chunk in text_chunks:
        proportion = len(chunk) / total_chars
        chunk_end = min(t + total_dur * proportion, end)
        result.append({"start": t, "end": chunk_end, "text": chunk})
        t = chunk_end
    if result:
        result[-1]["end"] = end
    return result


def resegment(
    segments: list[dict],
    max_chars: int = 84,
    max_duration: float = 5.0,
) -> list[dict]:
    """Split long WhisperX segments into subtitle-friendly chunks.

    Uses word-level timestamps when available; falls back to proportional
    time splitting otherwise.
    """
    result: list[dict] = []
    for seg in segments:
        text = seg.get("text", "").strip()
        start, end = seg["start"], seg["end"]
        dur = end - start

        if len(text) <= max_chars and dur <= max_duration:
            result.append({"start": start, "end": end, "text": text})
            continue

        words = seg.get("words", [])
        if words and all("start" in w and "end" in w for w in words):
            result.extend(_split_by_words(words, max_chars, max_duration))
        else:
            result.extend(_split_proportional(text, start, end, max_chars))

    return result


# ── OpenAI Whisper API backend ────────────────────────────────────────────────

def _transcribe_openai(
    audio_path: str,
    *,
    model: str = "whisper-1",
    language: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[list[dict], str]:
    """Transcribe using OpenAI's Whisper API.

    Returns:
        A tuple of (segments, detected_language).
        Each segment has 'start', 'end', and 'text' keys.
    """
    from openai import OpenAI

    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    with open(audio_path, "rb") as audio_file:
        # Request segment-level timestamps (more cost-effective than word-level)
        kwargs: dict[str, Any] = {
            "file": audio_file,
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language:
            kwargs["language"] = language

        log.info("Calling OpenAI Whisper API (model=%s)...", model)
        response = client.audio.transcriptions.create(**kwargs)

    # Parse response - OpenAI returns segments with start/end/text
    detected_lang = getattr(response, "language", "unknown")
    raw_segments = []

    for seg in response.segments or []:
        raw_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })

    log.info("OpenAI transcription complete: %d segments, language=%s",
             len(raw_segments), detected_lang)
    return raw_segments, detected_lang

# ── Audio preprocessing ───────────────────────────────────────────────────────

def _preprocess_audio(audio_path: str) -> str:
    """Convert audio to 16 kHz mono WAV — the native Whisper input format.

    Whisper was trained on 16 kHz mono audio.  Feeding it an MP3 or a stereo
    file forces an internal resample that can introduce artefacts (especially
    with lossy codecs).  Doing the conversion once up-front with ffmpeg
    produces a clean, lossless input and avoids redundant work inside every
    backend.

    Returns the path to the preprocessed WAV (placed next to the original).
    If the file is already 16 kHz mono WAV, it is returned unchanged.
    """
    import subprocess

    # Quick probe: skip conversion if file is already 16 kHz mono WAV
    if audio_path.lower().endswith(".wav"):
        try:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=sample_rate,channels",
                    "-of", "csv=p=0",
                    audio_path,
                ],
                capture_output=True, text=True, check=True,
            )
            parts = probe.stdout.strip().split(",")
            if len(parts) == 2 and parts[0].strip() == "16000" and parts[1].strip() == "1":
                log.debug("Audio already 16 kHz mono WAV — skipping preprocessing")
                return audio_path
        except (subprocess.CalledProcessError, Exception):
            pass  # fall through to conversion

    base, _ = os.path.splitext(audio_path)
    wav_path = f"{base}_16k.wav"
    if os.path.exists(wav_path):
        log.debug("Preprocessed audio already exists: %s", wav_path)
        return wav_path

    log.info("Preprocessing audio → 16 kHz mono WAV: %s", wav_path)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", audio_path,
            "-ac", "1",               # mono
            "-ar", "16000",            # 16 kHz
            "-sample_fmt", "s16",      # 16-bit PCM
            "-c:a", "pcm_s16le",       # WAV codec
            wav_path,
        ],
        check=True,
        capture_output=True,
    )
    return wav_path


# ── faster-whisper local backend ──────────────────────────────────────────────────

def _transcribe_faster_whisper(
    audio_path: str,
    *,
    model: str = "large-v3",
    device: str = "cuda",
    batch_size: int = 16,
    compute_type: str = "float16",
    language: str | None = None,
    beam_size: int = 5,
    initial_prompt: str | None = None,
    condition_on_previous_text: bool = True,
    vad_options: dict | None = None,
) -> tuple[list[dict], str]:
    """Transcribe using faster-whisper (CTranslate2).

    faster-whisper is 4x faster than original Whisper with lower memory usage.
    Supports word-level timestamps and VAD filtering.

    Requires: pip install "mazinger-dubber[transcribe-faster]"

    Returns:
        A tuple of (segments, detected_language).
        Each segment has 'start', 'end', 'text', and 'words' keys.
    """
    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
    except ImportError as e:
        raise ImportError(
            "faster-whisper not installed. Install with: pip install 'mazinger-dubber[transcribe-faster]'\n"
            "Or use method='openai' for cloud-based transcription."
        ) from e

    # ── Audio preprocessing: convert to 16 kHz mono WAV ───────────
    audio_path = _preprocess_audio(audio_path)

    log.info(
        "Transcribing with faster-whisper (model=%s, device=%s, batch=%d, compute=%s)",
        model, device, batch_size, compute_type,
    )

    # Load model (reuse cached instance when available)
    cache_key = f"{model}|{device}|{compute_type}"
    if cache_key in _whisper_cache:
        log.info("Reusing cached faster-whisper model: %s", cache_key)
        whisper_model = _whisper_cache[cache_key]
    else:
        try:
            whisper_model = WhisperModel(model, device=device, compute_type=compute_type)
        except ValueError:
            fallback = "int8" if device == "cpu" else "int8_float16"
            log.warning(
                "Compute type %s not supported on %s — falling back to %s",
                compute_type, device, fallback,
            )
            compute_type = fallback
            cache_key = f"{model}|{device}|{compute_type}"
            if cache_key in _whisper_cache:
                whisper_model = _whisper_cache[cache_key]
            else:
                whisper_model = WhisperModel(model, device=device, compute_type=compute_type)
        _whisper_cache[cache_key] = whisper_model

    # ── Pre-compute VAD clips and apply head/tail guards ──────────
    # We run Silero VAD ourselves so we can insert extra clips for any
    # audio the VAD dropped at the start or end of the file (e.g.
    # when background music lowers speech probability).
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import get_speech_timestamps, VadOptions

    sampling_rate = whisper_model.feature_extractor.sampling_rate
    audio = decode_audio(audio_path, sampling_rate=sampling_rate)
    audio_samples = audio.shape[0]
    duration_s = audio_samples / sampling_rate

    chunk_length = whisper_model.feature_extractor.chunk_length
    # Start with defaults tuned for transcription quality over speed.
    # - min_silence_duration_ms=500 avoids splitting mid-word on
    #   short pauses (the original 160ms was too aggressive).
    # - speech_pad_ms=200 adds a small buffer around each detected
    #   speech region so word onsets/offsets aren't clipped.
    # - threshold=0.35 (lower than default 0.5) makes VAD more
    #   sensitive, reducing missed speech at the cost of a few more
    #   false positives (which Whisper will harmlessly decode as silence).
    vad_kw: dict = {
        "max_speech_duration_s": chunk_length,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
        "threshold": 0.35,
    }
    if vad_options:
        vad_kw.update(vad_options)
    vad_params = VadOptions(**vad_kw)

    speech_clips = get_speech_timestamps(audio, vad_params)

    # Guard: add extra clips for head/tail gaps the VAD missed.
    # We add *separate* clips (not extend existing ones) so each
    # clip stays under the 30s chunk_length limit.
    _GUARD_THRESHOLD_S = 1.0
    guard_samples = int(_GUARD_THRESHOLD_S * sampling_rate)

    if speech_clips:
        # Head guard — add a clip covering [0, first_clip_start)
        head_gap = speech_clips[0]["start"]
        if head_gap > guard_samples:
            log.info(
                "Head guard: adding %.1fs clip before first VAD segment",
                head_gap / sampling_rate,
            )
            speech_clips.insert(0, {"start": 0, "end": speech_clips[0]["start"]})

        # Tail guard — add a clip covering (last_clip_end, audio_end]
        tail_gap = audio_samples - speech_clips[-1]["end"]
        if tail_gap > guard_samples:
            log.info(
                "Tail guard: adding %.1fs clip after last VAD segment",
                tail_gap / sampling_rate,
            )
            speech_clips.append({"start": speech_clips[-1]["end"], "end": audio_samples})
    else:
        # No speech detected at all — transcribe the full audio
        log.info("VAD detected no speech — transcribing full audio")
        speech_clips = [{"start": 0, "end": audio_samples}]

    # Convert from sample indices to seconds (the batched pipeline
    # expects seconds when clip_timestamps is provided externally).
    speech_clips_sec = [
        {"start": c["start"] / sampling_rate, "end": c["end"] / sampling_rate}
        for c in speech_clips
    ]

    # Use batched inference for better performance
    batched_model = BatchedInferencePipeline(model=whisper_model)

    # ── Transcription parameters matching OpenAI Whisper behaviour ─
    # OpenAI's Whisper uses temperature fallback: it starts at 0.0
    # and retries with higher temperatures if decoding quality is poor
    # (high compression ratio or low avg log-prob).  We replicate the
    # same strategy here.
    transcribe_kw: dict[str, Any] = {
        "batch_size": batch_size,
        "language": language,
        "beam_size": beam_size,
        "word_timestamps": True,
        "clip_timestamps": speech_clips_sec,
        "condition_on_previous_text": condition_on_previous_text,
        # Temperature fallback: try 0.0 first, then escalate.
        # Matches openai/whisper default behaviour.
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        # Thresholds that trigger a temperature retry
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }
    if initial_prompt:
        transcribe_kw["initial_prompt"] = initial_prompt

    segments_gen, info = batched_model.transcribe(audio, **transcribe_kw)

    detected_lang = info.language or "unknown"
    log.info("Detected language: %s (probability: %.2f)", detected_lang, info.language_probability)

    # Convert generator to list of segment dicts
    raw_segments = []
    for seg in segments_gen:
        segment_dict = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        }
        # Include word-level timestamps if available
        if seg.words:
            segment_dict["words"] = [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in seg.words
            ]
        raw_segments.append(segment_dict)

    log.info("faster-whisper transcription complete: %d segments", len(raw_segments))

    # Keep model in cache for reuse; only clean up the batched pipeline
    del batched_model

    return raw_segments, detected_lang


def _transcribe_gap(
    audio_path: str,
    start: float,
    end: float,
    *,
    language: str | None = None,
    beam_size: int = 5,
    initial_prompt: str | None = None,
) -> list[dict]:
    """Re-transcribe a specific time range using the cached faster-whisper model.

    Extracts the gap audio to a temp WAV, runs inference, then offsets
    timestamps back to the original timeline.
    """
    import subprocess
    import tempfile

    if not _whisper_cache:
        return []

    whisper_model = next(iter(_whisper_cache.values()))
    tmp_path = None

    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(start), "-to", str(end),
                "-i", audio_path,
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                tmp_path,
            ],
            check=True, capture_output=True,
        )

        from faster_whisper import BatchedInferencePipeline

        batched = BatchedInferencePipeline(model=whisper_model)
        kw: dict = {
            "beam_size": beam_size,
            "word_timestamps": True,
            "condition_on_previous_text": False,
        }
        if language:
            kw["language"] = language
        if initial_prompt:
            kw["initial_prompt"] = initial_prompt

        segments_gen, _ = batched.transcribe(tmp_path, **kw)

        result: list[dict] = []
        for seg in segments_gen:
            text = _clean_text(seg.text.strip())
            if not text:
                continue
            entry = {
                "start": round(seg.start + start, 3),
                "end": round(seg.end + start, 3),
                "text": text,
            }
            if seg.words:
                entry["words"] = [
                    {
                        "word": w.word,
                        "start": round(w.start + start, 3),
                        "end": round(w.end + start, 3),
                    }
                    for w in seg.words
                ]
            result.append(entry)

        del batched
        return result

    except Exception as exc:
        log.warning("Gap transcription failed (%.1f-%.1fs): %s", start, end, exc)
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── WhisperX local backend ────────────────────────────────────────────────────

def _transcribe_whisperx(
    audio_path: str,
    *,
    model: str = "large-v3",
    device: str = "cuda",
    batch_size: int = 4,
    compute_type: str = "float16",
    language: str | None = None,
) -> tuple[list[dict], str]:
    """Transcribe using local WhisperX with word-level alignment.

    Requires: pip install "mazinger-dubber[transcribe]"

    Returns:
        A tuple of (segments, detected_language).
        Each segment has 'start', 'end', 'text', and 'words' keys.
    """
    try:
        import torch

        # torchaudio >=2.11 removed list_audio_backends() / set_audio_backend();
        # speechbrain (used by pyannote/whisperx) still references them.
        # Provide no-op shims so the import chain doesn't break.
        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda backend: None

        import whisperx
    except ImportError as e:
        raise ImportError(
            "WhisperX not installed. Install with: pip install 'mazinger-dubber[transcribe-whisperx]'\n"
            "Or use method='openai' or method='faster-whisper' instead."
        ) from e

    log.info(
        "Transcribing with WhisperX (model=%s, device=%s, batch=%d, compute=%s)",
        model, device, batch_size, compute_type,
    )

    # Step 1 -- transcribe
    try:
        whisper_model = whisperx.load_model(model, device, compute_type=compute_type)
    except ValueError:
        fallback = "int8" if device == "cpu" else "int8_float16"
        log.warning(
            "Compute type %s not supported on %s — falling back to %s",
            compute_type, device, fallback,
        )
        compute_type = fallback
        whisper_model = whisperx.load_model(model, device, compute_type=compute_type)
    result = whisper_model.transcribe(audio_path, batch_size=batch_size, language=language)
    detected_lang = result.get("language", "unknown")
    log.info("Detected language: %s  (%d raw segments)", detected_lang, len(result["segments"]))

    del whisper_model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2 -- word-level alignment
    log.info("Aligning word-level timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
    result = whisperx.align(
        result["segments"], model_a, metadata, audio_path, device,
        return_char_alignments=False,
    )

    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    raw_segments = result["segments"]
    log.info("WhisperX aligned segments: %d", len(raw_segments))
    return raw_segments, detected_lang


# ── MLX Whisper backend (Apple Silicon) ───────────────────────────────────────

def _transcribe_mlx_whisper(
    audio_path: str,
    *,
    model: str = DEFAULT_MLX_WHISPER_MODEL,
    language: str | None = None,
    initial_prompt: str | None = None,
    condition_on_previous_text: bool = True,
    vad_options: dict | None = None,
    word_timestamps: bool = True,
) -> tuple[list[dict], str]:
    """Transcribe using MLX Whisper for Apple Silicon acceleration.

    MLX Whisper runs natively on Apple Silicon GPUs without requiring PyTorch.
    Uses sampling-based decoding (no beam search support).

    Requires: pip install "mazinger[transcribe-mlx]"

    Returns:
        A tuple of (segments, detected_language).
        Each segment has 'start', 'end', 'text', and 'words' keys.
    """
    import platform
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Darwin" or machine not in {"arm64", "aarch64"}:
        raise RuntimeError(
            "MLX Whisper requires Apple Silicon (M1/M2/M3/M4/M5). "
            f"Current platform: {system} ({platform.machine()}). "
            "Use method='faster-whisper' or method='openai' instead."
        )
    try:
        import mlx_whisper
    except ImportError as e:
        raise ImportError(
            "mlx-whisper not installed. Install with: pip install 'mazinger[transcribe-mlx]'\n"
            "Or use method='openai' for cloud-based transcription."
        ) from None

    audio_path = _preprocess_audio(audio_path)

    log.info("Transcribing with MLX Whisper (model=%s)", model)

    transcribe_kw: dict[str, Any] = {
        "path_or_hf_repo": model,
        "word_timestamps": word_timestamps,
        "condition_on_previous_text": condition_on_previous_text,
    }
    if language:
        transcribe_kw["language"] = language
    if initial_prompt:
        transcribe_kw["initial_prompt"] = initial_prompt
    if vad_options and "no_speech_threshold" in vad_options:
        transcribe_kw["no_speech_threshold"] = vad_options["no_speech_threshold"]

    result = mlx_whisper.transcribe(audio_path, **transcribe_kw)

    detected_lang = result.get("language", "unknown")
    log.info("Detected language: %s", detected_lang)

    raw_segments = []
    for seg in result.get("segments", []):
        segment_dict = {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        }
        if "words" in seg and seg["words"]:
            segment_dict["words"] = [
                {"word": w["word"], "start": w["start"], "end": w["end"]}
                for w in seg["words"]
            ]
        raw_segments.append(segment_dict)

    log.info("MLX Whisper transcription complete: %d segments", len(raw_segments))

    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass

    return raw_segments, detected_lang


# ── Deepgram Nova API backend ───────────────────────────────────────────────

def _transcribe_deepgram(
    audio_path: str,
    *,
    model: str = "nova-3",
    language: str | None = None,
    api_key: str | None = None,
    keyterms: list[str] | None = None,
) -> tuple[list[dict], str]:
    """Transcribe using Deepgram's Nova STT API.

    Deepgram Nova-3 is a fast, accurate cloud transcription service
    supporting 47+ languages with word-level timestamps.

    Requires: pip install "mazinger[transcribe-deepgram]"

    Returns:
        A tuple of (segments, detected_language).
        Each segment has 'start', 'end', 'text', and 'words' keys.
    """
    try:
        from deepgram import DeepgramClient
    except ImportError as e:
        raise ImportError(
            "deepgram-sdk not installed. Install with: pip install 'mazinger[transcribe-deepgram]'\n"
            "Or use method='openai' for cloud-based transcription."
        ) from e

    client = DeepgramClient(api_key=api_key) if api_key else DeepgramClient()

    file_size = os.path.getsize(audio_path)
    _WARN_THRESHOLD = 100 * 1024 * 1024  # 100 MiB
    if file_size > _WARN_THRESHOLD:
        log.warning(
            "Audio file is %.0f MiB — reading into memory for Deepgram upload. "
            "For very large files consider splitting or compressing first.",
            file_size / (1024 * 1024),
        )

    with open(audio_path, "rb") as f:
        buffer_data = f.read()

    options_kw: dict[str, Any] = {
        "model": model,
        "smart_format": True,
        "punctuate": True,
        "utterances": True,
    }
    if language:
        options_kw["language"] = language
    else:
        options_kw["detect_language"] = True

    # Nova-3 keyterm boosting: improve recognition of domain-specific terms
    # extracted from video metadata (title, tags, description).
    if keyterms and "nova-3" in model:
        options_kw["keyterm"] = keyterms[:100]

    log.info("Calling Deepgram API (model=%s)...", model)
    response = client.listen.v1.media.transcribe_file(
        request=buffer_data, **options_kw,
    )

    # Extract detected language and confidence
    detected_lang = "unknown"
    channels = response.results.channels
    if channels:
        detected_lang = getattr(channels[0], "detected_language", None) or "unknown"
        lang_confidence = getattr(channels[0], "language_confidence", None)
        if lang_confidence is not None:
            log.info("Detected language: %s (confidence: %.2f)", detected_lang, lang_confidence)
        else:
            log.info("Detected language: %s", detected_lang)

    # Build segments from utterances (natural speech boundaries)
    raw_segments: list[dict] = []
    utterances = getattr(response.results, "utterances", None)
    if utterances:
        for utt in utterances:
            segment: dict[str, Any] = {
                "start": utt.start,
                "end": utt.end,
                "text": utt.transcript.strip(),
            }
            if utt.words:
                segment["words"] = [
                    {
                        "word": getattr(w, "punctuated_word", None) or w.word,
                        "start": w.start,
                        "end": w.end,
                    }
                    for w in utt.words
                ]
            raw_segments.append(segment)
    elif channels:
        # Fallback: build a single segment from the full transcript
        alt = channels[0].alternatives[0]
        if alt.transcript.strip():
            segment = {
                "start": alt.words[0].start if alt.words else 0.0,
                "end": alt.words[-1].end if alt.words else 0.0,
                "text": alt.transcript.strip(),
            }
            if alt.words:
                segment["words"] = [
                    {
                        "word": getattr(w, "punctuated_word", None) or w.word,
                        "start": w.start,
                        "end": w.end,
                    }
                    for w in alt.words
                ]
            raw_segments.append(segment)

    log.info(
        "Deepgram transcription complete: %d segments, language=%s",
        len(raw_segments), detected_lang,
    )
    return raw_segments, detected_lang


# ── LLM-based text refinement ────────────────────────────────────────────────

def _refine_segments_llm(
    segments: list[dict],
    detected_lang: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    llm_model: str = "gpt-4.1",
) -> list[dict]:
    """Use an LLM to add punctuation and fix misheard words in transcribed text.

    Preserves timestamps — only the text of each segment is modified.
    """
    from mazinger.llm import build_client

    client = build_client(api_key=api_key, base_url=base_url)

    # Build the text block with indices so we can map back
    numbered = []
    for i, seg in enumerate(segments):
        numbered.append(f"[{i}] {seg['text']}")
    text_block = "\n".join(numbered)

    resp = client.chat.completions.create(
        model=llm_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": (
                f"You are a transcript editor for {detected_lang} speech-to-text output. "
                "Fix misheard words, add punctuation (commas, periods, question marks), "
                "and correct obvious spelling errors. Keep the original meaning and word order intact. "
                "Do NOT add, remove, or reorder content. Do NOT translate. "
                "Return each line in the same [index] format."
            )},
            {"role": "user", "content": text_block},
        ],
    )

    refined_text = resp.choices[0].message.content.strip()

    # Parse refined lines back by index
    refined_map: dict[int, str] = {}
    for line in refined_text.splitlines():
        line = line.strip()
        if line.startswith("["):
            bracket_end = line.find("]")
            if bracket_end > 0:
                try:
                    idx = int(line[1:bracket_end])
                    refined_map[idx] = line[bracket_end + 1:].strip()
                except ValueError:
                    pass

    result = []
    for i, seg in enumerate(segments):
        seg = dict(seg)
        if i in refined_map and refined_map[i]:
            seg["text"] = refined_map[i]
        result.append(seg)

    log.info("LLM refinement applied to %d/%d segments", len(refined_map), len(segments))
    return result


# ── Main transcription entry point ────────────────────────────────────────────

def transcribe(
    audio_path: str,
    output_path: str,
    *,
    method: TranscribeMethod = "faster-whisper",
    model: str | None = None,
    mlx_whisper_model: str = DEFAULT_MLX_WHISPER_MODEL,
    device: str = "cuda",
    batch_size: int = 4,
    compute_type: str = "float16",
    language: str | None = None,
    beam_size: int | None = 5,
    max_chars: int = 84,
    max_duration: float = 5.0,
    skip_resegment: bool = False,
    refine: bool = False,
    llm_model: str = "gpt-4.1",
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    deepgram_api_key: str | None = None,
    initial_prompt: str | None = None,
    condition_on_previous_text: bool = True,
    vad_options: dict | None = None,
    word_timestamps: bool = True,
) -> str:
    """Transcribe audio to SRT using OpenAI Whisper, faster-whisper, WhisperX, MLX Whisper, or Deepgram.

    Parameters:
        audio_path:     Path to the input audio file.
        output_path:    Where to save the final SRT.
        method:         Transcription backend: ``faster-whisper`` (default),
                        ``openai``, ``whisperx``, ``mlx-whisper``, or
                        ``deepgram``.
        model:          Model name. Defaults to ``whisper-1`` for OpenAI,
                        ``large-v3`` for faster-whisper/WhisperX,
                        ``nova-3`` for Deepgram.
        device:         ``cuda`` or ``cpu`` (local methods only).
        batch_size:     Inference batch size (local methods only).
        compute_type:   ``float16``, ``int8``, or ``int8_float16`` (local methods).
        language:       Force a language code (e.g., ``en``, ``ar``) or ``None``
                        for auto-detection.
        max_chars:      Maximum characters per subtitle segment.
        max_duration:   Maximum seconds per subtitle segment.
        skip_resegment: When ``True``, keep original segments as-is.
        openai_api_key: OpenAI API key (OpenAI method only). Falls back to
                        ``OPENAI_API_KEY`` environment variable.
        deepgram_api_key: Deepgram API key (Deepgram method only). Falls back
                        to ``DEEPGRAM_API_KEY`` environment variable.

    Returns:
        The path to the saved SRT file.

    Examples:
        # Using faster-whisper (default, fast local transcription)
        transcribe("audio.mp3", "output.srt")

        # Using OpenAI (cloud, no local GPU needed)
        transcribe("audio.mp3", "output.srt", method="openai")

        # Using faster-whisper explicitly
        transcribe("audio.mp3", "output.srt", method="faster-whisper", device="cuda")

        # Using WhisperX (requires [transcribe-whisperx] extra)
        transcribe("audio.mp3", "output.srt", method="whisperx", device="cuda")

        # Using Deepgram Nova-3 (cloud, fast and accurate)
        transcribe("audio.mp3", "output.srt", method="deepgram")

        # Using MLX Whisper (Apple Silicon, requires [transcribe-mlx] extra)
        transcribe("audio.mp3", "output.srt", method="mlx-whisper")
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Select backend and transcribe
    if method == "openai":
        default_model = "whisper-1"
        raw_segments, detected_lang = _transcribe_openai(
            audio_path,
            model=model or default_model,
            language=language,
            api_key=openai_api_key,
            base_url=openai_base_url,
        )
    elif method == "faster-whisper":
        default_model = "large-v3"
        raw_segments, detected_lang = _transcribe_faster_whisper(
            audio_path,
            model=model or default_model,
            device=device,
            batch_size=batch_size,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
            vad_options=vad_options,
        )
    elif method == "whisperx":
        default_model = "large-v3"
        raw_segments, detected_lang = _transcribe_whisperx(
            audio_path,
            model=model or default_model,
            device=device,
            batch_size=batch_size,
            compute_type=compute_type,
            language=language,
        )
    elif method == "mlx-whisper":
        if beam_size is not None:
            log.info("MLX Whisper uses sampling-based decoding (beam_size is ignored)")
        else:
            log.info("MLX Whisper uses sampling-based decoding")
        raw_segments, detected_lang = _transcribe_mlx_whisper(
            audio_path,
            model=mlx_whisper_model,
            language=language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
            vad_options=vad_options,
            word_timestamps=word_timestamps,
        )
    elif method == "deepgram":
        default_model = "nova-3"
        # Extract keyterms from initial_prompt for Nova-3 term boosting
        keyterms = None
        if initial_prompt:
            keyterms = [
                t.strip() for t in re.split(r"[.,;|]+", initial_prompt) if t.strip()
            ]
        raw_segments, detected_lang = _transcribe_deepgram(
            audio_path,
            model=model or default_model,
            language=language,
            api_key=deepgram_api_key,
            keyterms=keyterms,
        )
    else:
        raise ValueError(
            f"Unknown transcription method: {method!r}. "
            "Use 'openai', 'faster-whisper', 'whisperx', 'mlx-whisper', or 'deepgram'."
        )

    log.info("Transcription complete: %d segments, language=%s", len(raw_segments), detected_lang)

    # Clean up common transcription artifacts
    raw_segments = _clean_segments(raw_segments)

    # Validate: recover speech gaps that faster-whisper skipped
    if method == "faster-whisper":
        from mazinger.validate import validate_transcription
        from mazinger.utils import get_audio_duration

        audio_dur = get_audio_duration(audio_path)
        pre_validation = list(raw_segments)

        def _gap_fn(_path, _start, _end):
            return _transcribe_gap(
                _path, _start, _end,
                language=detected_lang,
                beam_size=beam_size,
                initial_prompt=initial_prompt,
            )

        raw_segments, _was_modified = validate_transcription(
            raw_segments, audio_path, audio_dur,
            transcribe_gap_fn=_gap_fn,
        )
        if _was_modified:
            _base, _ext = os.path.splitext(output_path)
            _pre_path = f"{_base}.PRE_VALIDATION{_ext}"
            with open(_pre_path, "w", encoding="utf-8") as fh:
                fh.write(_segments_to_srt(pre_validation))
            log.info("Pre-validation SRT saved: %s", _pre_path)

    # LLM refinement: punctuation and misheard-word correction
    if refine:
        raw_segments = _refine_segments_llm(
            raw_segments, detected_lang,
            api_key=openai_api_key,
            base_url=openai_base_url,
            llm_model=llm_model,
        )
        # Drop word-level timestamps — they no longer match the refined text
        for seg in raw_segments:
            seg.pop("words", None)

    # Save raw SRT
    base, ext = os.path.splitext(output_path)
    raw_srt_path = f"{base}.raw{ext}"
    raw_srt_content = _segments_to_srt(raw_segments)
    with open(raw_srt_path, "w", encoding="utf-8") as fh:
        fh.write(raw_srt_content)
    log.info("Raw SRT saved: %s (%d segments)", raw_srt_path, len(raw_segments))

    # Resegment for readability
    if skip_resegment:
        final_segments = raw_segments
    else:
        final_segments = resegment(raw_segments, max_chars, max_duration)
        log.info("Resegmented: %d -> %d segments", len(raw_segments), len(final_segments))

    # Save final SRT
    srt_content = _segments_to_srt(final_segments)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(srt_content)
    log.info("SRT saved: %s (%d segments)", output_path, len(final_segments))

    return output_path
