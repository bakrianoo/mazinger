"""Transcribe audio to SRT using OpenAI Whisper API, faster-whisper, or WhisperX.

Supports three transcription backends:
- **openai** (default): Uses OpenAI's Whisper API. No local GPU required,
  simple setup, works with any transformers version. Requires OPENAI_API_KEY.
- **faster-whisper**: Fast local transcription using CTranslate2. 4x faster
  than original Whisper with lower memory usage. No transformers dependency,
  compatible with Chatterbox TTS. Requires [transcribe-faster] extra.
- **whisperx**: Local transcription with word-level alignment via wav2vec2.
  Requires PyTorch + CUDA and the [transcribe-whisperx] extra. Not compatible
  with chatterbox-tts due to conflicting transformers requirements.
"""

from __future__ import annotations

import gc
import logging
import os
import re
from typing import Any, Literal

log = logging.getLogger(__name__)

# Supported transcription methods
TranscribeMethod = Literal["openai", "faster-whisper", "whisperx"]

# Module-level cache for faster-whisper models — avoids reloading across runs
_whisper_cache: dict[str, Any] = {}


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

    # Use batched inference for better performance
    batched_model = BatchedInferencePipeline(model=whisper_model)

    # Transcribe with word-level timestamps and VAD filtering
    segments_gen, info = batched_model.transcribe(
        audio_path,
        batch_size=batch_size,
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,
    )

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
    method: TranscribeMethod = "openai",
    model: str | None = None,
    device: str = "cuda",
    batch_size: int = 4,
    compute_type: str = "float16",
    language: str | None = None,
    beam_size: int = 5,
    max_chars: int = 84,
    max_duration: float = 5.0,
    skip_resegment: bool = False,
    refine: bool = False,
    llm_model: str = "gpt-4.1",
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> str:
    """Transcribe audio to SRT using OpenAI Whisper API, faster-whisper, or WhisperX.

    Parameters:
        audio_path:     Path to the input audio file.
        output_path:    Where to save the final SRT.
        method:         Transcription backend: ``openai`` (default),
                        ``faster-whisper``, or ``whisperx``.
        model:          Model name. Defaults to ``whisper-1`` for OpenAI,
                        ``large-v3`` for faster-whisper and WhisperX.
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

    Returns:
        The path to the saved SRT file.

    Examples:
        # Using OpenAI (default, no local GPU needed)
        transcribe("audio.mp3", "output.srt")

        # Using faster-whisper (fast local, compatible with Chatterbox)
        transcribe("audio.mp3", "output.srt", method="faster-whisper", device="cuda")

        # Using WhisperX (requires [transcribe-whisperx] extra)
        transcribe("audio.mp3", "output.srt", method="whisperx", device="cuda")
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
    else:
        raise ValueError(f"Unknown transcription method: {method!r}. Use 'openai', 'faster-whisper', or 'whisperx'.")

    log.info("Transcription complete: %d segments, language=%s", len(raw_segments), detected_lang)

    # Clean up common transcription artifacts
    raw_segments = _clean_segments(raw_segments)

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
