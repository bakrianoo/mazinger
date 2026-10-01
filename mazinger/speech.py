"""Speech map: where the original speaker actually talks.

Subtitle timestamps are a poor timing reference for dubbing.  Transcription
segments carry lead-in and trailing silence, and re-segmentation merges and
splits them for readability.  The speech map is built from the audio itself
(voice-activity detection, on the Demucs vocals stem when one exists) and
gives every dubbed line the *speech* span it replaces:

* ``onset`` — where the original speech of the line starts; the dub is
  placed here.
* ``offset`` — where it ends; the dub is fitted to ``offset - onset``.

Vocal activity that no line covers (laughs, breaths, a missed phrase) is
reported by :func:`uncovered_regions`, so assembly can keep it.

The map is cached as JSON next to the source audio and rebuilt only when
the audio it was built from changes.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

SPEECH_MAP_VERSION = 1
VAD_SR = 16_000

# Search this far outside a subtitle entry for the edges of its speech:
# ASR timestamps are often a few hundred milliseconds early or late.
EDGE_TOLERANCE = 0.3
# Shorter spans than this are not trusted; the entry's own span is used.
MIN_SPAN = 0.15
# Speech this close outside an entry still counts as the entry's own.
NEAR = 0.1

_SILERO_OPTIONS = dict(
    threshold=0.45,
    min_speech_duration_ms=100,
    min_silence_duration_ms=150,
    speech_pad_ms=30,
)


@dataclass
class SpeechMap:
    """Voiced regions of a source track, in seconds."""

    duration: float
    regions: list[tuple[float, float]] = field(default_factory=list)
    method: str = "silero"
    source: str = "mix"

    def to_dict(self) -> dict:
        return {
            "version": SPEECH_MAP_VERSION,
            "duration": self.duration,
            "method": self.method,
            "source": self.source,
            "regions": [[round(s, 3), round(e, 3)] for s, e in self.regions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> SpeechMap:
        return cls(
            duration=float(d["duration"]),
            regions=[(float(s), float(e)) for s, e in d.get("regions", [])],
            method=d.get("method", "silero"),
            source=d.get("source", "mix"),
        )

    def speech_seconds(self) -> float:
        return sum(e - s for s, e in self.regions)


@dataclass
class Span:
    """The speech a dubbed line replaces."""

    onset: float
    offset: float
    voiced: bool = True   # False: no speech found, the entry's own span is used

    @property
    def duration(self) -> float:
        return self.offset - self.onset


# ═══════════════════════════════════════════════════════════════════════════════
#  Detection
# ═══════════════════════════════════════════════════════════════════════════════

def decode_mono(path: str, sr: int) -> np.ndarray:
    """Decode *path* to mono float32 at *sr* with ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
         "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def _frame_db(audio: np.ndarray, frame: int) -> np.ndarray:
    n = len(audio) // frame
    if n == 0:
        return np.full(1, -120.0, dtype=np.float32)
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.einsum("ij,ij->i", frames, frames) / frame + 1e-12)
    return 20.0 * np.log10(rms + 1e-9)


def _energy_regions(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Fallback VAD: frames well above the track's noise floor."""
    hop = int(sr * 0.02)
    db = _frame_db(audio, hop)
    floor = float(np.percentile(db, 10))
    peak = float(np.percentile(db, 99.5))
    thresh = max(floor + 12.0, peak - 40.0, -60.0)
    active = db > thresh
    regions: list[tuple[float, float]] = []
    start = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            regions.append((start * hop / sr, i * hop / sr))
            start = None
    if start is not None:
        regions.append((start * hop / sr, len(active) * hop / sr))
    return _tidy(regions, min_gap=0.15, min_len=0.1)


def _silero_regions(audio: np.ndarray) -> list[tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    stamps = get_speech_timestamps(audio, VadOptions(**_SILERO_OPTIONS), sampling_rate=VAD_SR)
    return [(s["start"] / VAD_SR, s["end"] / VAD_SR) for s in stamps]


def _tidy(
    regions: list[tuple[float, float]], *, min_gap: float = 0.0, min_len: float = 0.0,
) -> list[tuple[float, float]]:
    """Sort, merge regions closer than *min_gap* and drop ones under *min_len*."""
    out: list[list[float]] = []
    for s, e in sorted(regions):
        if e <= s:
            continue
        if out and s - out[-1][1] <= min_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if e - s >= min_len]


def detect_speech(audio: np.ndarray, sr: int = VAD_SR) -> tuple[list[tuple[float, float]], str]:
    """Voiced regions of mono *audio*; returns ``(regions, method)``.

    Uses Silero VAD (shipped with faster-whisper) and falls back to an
    energy detector when it is not installed.
    """
    if sr != VAD_SR:
        raise ValueError(f"detect_speech expects {VAD_SR} Hz audio, got {sr}")
    try:
        return _tidy(_silero_regions(audio), min_gap=0.0), "silero"
    except ImportError:
        log.info("Silero VAD unavailable (faster-whisper not installed) — using energy VAD")
    except Exception as exc:  # noqa: BLE001 — a broken VAD must not stop a dub
        log.warning("Silero VAD failed (%s) — using energy VAD", exc)
    return _energy_regions(audio, sr), "energy"


def _stamp(path: str) -> list[int]:
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]


def build_speech_map(
    audio_path: str,
    cache_path: str | None = None,
    *,
    vocals_path: str | None = None,
) -> SpeechMap:
    """Return the speech map of *audio_path*, from cache when still valid.

    When *vocals_path* (a separated vocals stem) is given, detection runs on
    it: music and effects under the voice then no longer read as speech.
    The duration always comes from *audio_path*, decoded to samples, so it
    is exact rather than a container estimate.
    """
    detect_on = vocals_path if vocals_path and os.path.isfile(vocals_path) else audio_path
    stamp = {"audio": _stamp(audio_path), "input": _stamp(detect_on)}

    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as fh:
                cached = json.load(fh)
            if cached.get("version") == SPEECH_MAP_VERSION and cached.get("stamp") == stamp:
                log.info("Reusing speech map: %s", cache_path)
                return SpeechMap.from_dict(cached)
        except (OSError, ValueError, KeyError):
            pass

    source = decode_mono(audio_path, VAD_SR)
    duration = len(source) / VAD_SR
    audio = source if detect_on == audio_path else decode_mono(detect_on, VAD_SR)
    del source
    regions, method = detect_speech(audio)
    regions = [(s, min(e, duration)) for s, e in regions if s < duration]
    smap = SpeechMap(
        duration=duration, regions=regions, method=method,
        source="vocals" if detect_on != audio_path else "mix",
    )
    log.info(
        "Speech map: %d regions, %.1fs of speech in %.1fs (%s on %s)",
        len(regions), smap.speech_seconds(), duration, method, smap.source,
    )
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({**smap.to_dict(), "stamp": stamp}, fh)
        os.replace(tmp, cache_path)
    return smap


def load_speech_map(cache_path: str) -> SpeechMap | None:
    """Load a cached speech map without validating it; ``None`` if absent."""
    try:
        with open(cache_path, encoding="utf-8") as fh:
            return SpeechMap.from_dict(json.load(fh))
    except (OSError, ValueError, KeyError):
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Lines ↔ speech
# ═══════════════════════════════════════════════════════════════════════════════

def speech_spans(
    entries: list[tuple[float, float]],
    smap: SpeechMap | None,
    *,
    tolerance: float = EDGE_TOLERANCE,
) -> list[Span]:
    """The speech span of each ``(start, end)`` entry, in the given order.

    A region belongs to an entry when it overlaps the entry itself (or
    starts or ends within :data:`NEAR` of it); the span then reaches out to
    that speech's real edges, up to *tolerance* beyond the entry but never
    into a neighbouring entry.  Entries with no speech — typically ASR
    hallucinations over music or silence — keep their own span
    (``voiced=False``).  Spans never overlap.
    """
    order = sorted(range(len(entries)), key=lambda i: entries[i][0])
    spans: list[Span | None] = [None] * len(entries)
    regions = smap.regions if smap else []
    starts = np.array([s for s, _ in regions]) if regions else np.zeros(0)
    prev_offset = 0.0

    for pos, i in enumerate(order):
        s, e = entries[i]
        prev_end = entries[order[pos - 1]][1] if pos > 0 else 0.0
        next_start = entries[order[pos + 1]][0] if pos + 1 < len(order) else None
        lower = max(s - tolerance, min(prev_end, s), prev_offset)
        upper = e + tolerance
        if next_start is not None:
            upper = min(upper, max(next_start, e))
        if smap is not None:
            upper = min(upper, smap.duration)

        onset = offset = None
        if len(starts):
            # Regions that overlap the entry's own [s, e], widened by NEAR.
            core_s, core_e = max(s - NEAR, lower), min(e + NEAR, upper)
            k = int(np.searchsorted(starts, core_e, side="left"))
            j = k - 1
            while j >= 0 and regions[j][1] > core_s:
                j -= 1
            hit = regions[j + 1:k]
            if hit:
                onset = max(hit[0][0], lower)
                offset = min(hit[-1][1], upper)

        if onset is None or offset - onset < MIN_SPAN:
            span = Span(max(s, prev_offset), max(e, max(s, prev_offset)), voiced=False)
        else:
            span = Span(onset, offset)
        spans[i] = span
        prev_offset = span.offset

    return spans  # type: ignore[return-value]


def uncovered_regions(
    smap: SpeechMap,
    spans: list[Span],
    *,
    margin: float = 0.1,
    min_len: float = 0.12,
) -> list[tuple[float, float]]:
    """Vocal activity that no dubbed line covers.

    Each span is widened by *margin* before subtraction so the edges of a
    line's own speech are not reported; leftovers shorter than *min_len*
    are dropped.
    """
    covered = _tidy([(sp.onset - margin, sp.offset + margin) for sp in spans])
    out: list[tuple[float, float]] = []
    c = 0
    for s, e in smap.regions:
        cur = s
        while c < len(covered) and covered[c][1] <= cur:
            c += 1
        k = c
        while k < len(covered) and covered[k][0] < e:
            cs, ce = covered[k]
            if cs > cur:
                out.append((cur, cs))
            cur = max(cur, ce)
            if cur >= e:
                break
            k += 1
        if cur < e:
            out.append((cur, e))
    return [(s, e) for s, e in out if e - s >= min_len]


# Lines with no speech under them are skipped only while they are few: when
# many are, the speech map itself is unreliable (VAD failure, unusual audio)
# and every line is dubbed.
UNVOICED_MAX_SHARE = 0.25


def trust_unvoiced(spans: list[Span]) -> bool:
    """Whether the ``voiced=False`` spans can be taken as "no speech here"."""
    n = sum(not sp.voiced for sp in spans)
    if n and n > max(2, UNVOICED_MAX_SHARE * len(spans)):
        log.warning(
            "%d of %d lines have no detected speech under them — the speech map looks "
            "unreliable, so every line is dubbed", n, len(spans),
        )
        return False
    return True


def line_targets(
    entries: list[dict], smap: SpeechMap | None,
) -> tuple[dict[str, float], set[str]]:
    """Per-line speech targets for parsed SRT *entries*.

    Returns ``({idx: seconds}, unvoiced)`` where *unvoiced* holds the lines
    that have text but no speech under them in *smap* (typically ASR
    hallucinations); it is empty without a map.
    """
    spans = speech_spans([(e["start"], e["end"]) for e in entries], smap)
    targets = {e["idx"]: sp.duration for e, sp in zip(entries, spans)}
    unvoiced = set()
    if smap is not None:
        texted = [(e, sp) for e, sp in zip(entries, spans) if e["text"].strip()]
        if trust_unvoiced([sp for _, sp in texted]):
            unvoiced = {e["idx"] for e, sp in texted if not sp.voiced}
    return targets, unvoiced


def project_speech_map(proj) -> tuple[SpeechMap | None, str | None, str | None]:
    """``(speech_map, vocals_path, stems_method)`` for a project's source audio.

    Separates the source with Demucs when it is installed — cached in the
    project and reused by the background mix — and detects speech on the
    vocals stem; otherwise detects on the mix.  *vocals_path* and
    *stems_method* are ``None`` without Demucs, and the map is ``None`` if
    detection itself failed.
    """
    import importlib.util

    from mazinger import assemble

    vocals, method = None, None
    if importlib.util.find_spec("demucs") is not None:
        try:
            _, vocals, method = assemble.extract_stems_cached(
                proj.audio,
                proj.background_audio(assemble.TARGET_SR),
                proj.vocals_audio(assemble.TARGET_SR),
            )
        except Exception as exc:  # noqa: BLE001 — detect on the mix instead
            log.warning("Source separation failed (%s); detecting speech on the mix", exc)
            vocals, method = None, None
    if method != "demucs":
        vocals = None
    try:
        smap = build_speech_map(proj.audio, proj.speech_map, vocals_path=vocals)
    except Exception as exc:  # noqa: BLE001 — sync then falls back to subtitle spans
        log.warning("Speech map failed (%s); lines are fitted to their subtitle spans", exc)
        smap = None
    return smap, vocals, method


# ═══════════════════════════════════════════════════════════════════════════════
#  Clip edges
# ═══════════════════════════════════════════════════════════════════════════════

def speech_extent(
    audio: np.ndarray, sr: int, *, rel_db: float = -40.0, abs_db: float = -60.0,
    pad: float = 0.02,
) -> tuple[int, int]:
    """Sample range ``[i0, i1)`` of *audio* that holds sound.

    Leading and trailing frames quieter than *rel_db* below the clip's peak
    (and below *abs_db* absolute) are treated as silence; *pad* seconds are
    kept on either side.  An all-silent clip returns ``(0, len(audio))``.
    """
    n = len(audio)
    frame = max(1, int(sr * 0.01))
    db = _frame_db(np.asarray(audio, dtype=np.float32), frame)
    thresh = max(float(db.max()) + rel_db, abs_db)
    loud = np.nonzero(db > thresh)[0]
    if not len(loud):
        return 0, n
    p = int(pad * sr)
    i0 = max(0, int(loud[0]) * frame - p)
    i1 = min(n, (int(loud[-1]) + 1) * frame + p)
    return i0, i1


def clip_regions(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Speech regions of a TTS clip in seconds, found with the speech map's VAD.

    Bounded by the clip's sound (:func:`speech_extent`); falls back to that
    single region when the VAD is unavailable or finds nothing.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    e0, e1 = speech_extent(audio, sr)
    fallback = [(e0 / sr, e1 / sr)]
    try:
        if sr == VAD_SR:
            a16 = audio
        else:
            from math import gcd

            from scipy.signal import resample_poly
            g = gcd(VAD_SR, sr)
            a16 = resample_poly(audio, VAD_SR // g, sr // g).astype(np.float32)
        regions = _silero_regions(a16)
    except Exception:  # noqa: BLE001 — no VAD: energy edges
        return fallback
    lo, hi = e0 / sr, e1 / sr
    regions = [(max(a, lo), min(b, hi)) for a, b in regions if b > lo and a < hi]
    return regions or fallback


def clip_extent(audio: np.ndarray, sr: int) -> tuple[int, int]:
    """Sample range ``[i0, i1)`` of the speech in a TTS clip.

    Uses the same VAD as the speech map, so a dub line and the original
    speech it replaces are measured alike: a breath or a soft click before
    the first word is not counted as speech on either side.  Falls back to
    :func:`speech_extent` when the VAD is unavailable or finds nothing.
    """
    regions = clip_regions(audio, sr)
    i0 = max(0, int(regions[0][0] * sr))
    i1 = min(len(audio), int(np.ceil(regions[-1][1] * sr)))
    return (i0, i1) if i1 > i0 else (0, len(audio))


# ═══════════════════════════════════════════════════════════════════════════════
#  Phrases (pause-aware placement)
# ═══════════════════════════════════════════════════════════════════════════════

# A pause at least this long splits a line's original speech into phrases.
SOURCE_PAUSE = 0.3
# TTS pauses at commas and full stops run a little shorter than a speaker's.
CLIP_PAUSE = 0.2
# Matched pauses must sit at about the same point of their line (as a share
# of its speech); farther apart they are not the same pause.
PAUSE_MATCH = 0.2


def phrases(regions: list[tuple[float, float]], min_pause: float) -> list[tuple[float, float]]:
    """*regions* merged into phrases separated by pauses of at least *min_pause*."""
    return _tidy(regions, min_gap=min_pause - 1e-9)


def span_phrases(smap: SpeechMap, span: Span, min_pause: float = SOURCE_PAUSE) -> list[tuple[float, float]]:
    """The original phrases inside *span*, clipped to it."""
    inside = [(max(s, span.onset), min(e, span.offset)) for s, e in smap.regions
              if e > span.onset and s < span.offset]
    return phrases(inside, min_pause)


def _pause_points(ph: list[tuple[float, float]]) -> list[float]:
    """Where each pause between *ph* falls, as a share of their speech."""
    total = sum(e - s for s, e in ph) or 1.0
    out, acc = [], 0.0
    for s, e in ph[:-1]:
        acc += e - s
        out.append(acc / total)
    return out


def match_pauses(
    src: list[tuple[float, float]], clip: list[tuple[float, float]], *, tolerance: float = PAUSE_MATCH,
) -> list[tuple[int, int]]:
    """Pair pauses of the original (*src* phrases) with pauses of the clip.

    Pause ``i`` lies after phrase ``i``.  Pairs are monotonic and one to one,
    chosen to minimise the gap between their relative positions, and only
    kept when within *tolerance*.  Returns ``[(src_pause, clip_pause), ...]``.
    """
    a, b = _pause_points(src), _pause_points(clip)
    if not a or not b:
        return []
    n, m = len(a), len(b)
    # dp[i][j]: best (pairs, -cost) using a[:i], b[:j] — most pairs first, then least cost.
    dp = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
    back: dict[tuple[int, int], tuple[int, int, bool]] = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = [(dp[i - 1][j], (i - 1, j, False)), (dp[i][j - 1], (i, j - 1, False))]
            d = abs(a[i - 1] - b[j - 1])
            if d <= tolerance:
                p, c = dp[i - 1][j - 1]
                opts.append(((p + 1, c - d), (i - 1, j - 1, True)))
            best = max(opts, key=lambda o: o[0])
            dp[i][j] = best[0]
            back[(i, j)] = best[1]
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        pi, pj, took = back[(i, j)]
        if took:
            pairs.append((i - 1, j - 1))
        i, j = pi, pj
    return pairs[::-1]


def trim_silence(audio: np.ndarray, sr: int, **kw) -> np.ndarray:
    """*audio* without its leading and trailing silence (see :func:`speech_extent`)."""
    i0, i1 = speech_extent(audio, sr, **kw)
    return audio[i0:i1]


def trim_to_speech(audio: np.ndarray, sr: int) -> np.ndarray:
    """*audio* cut to its speech (see :func:`clip_extent`)."""
    i0, i1 = clip_extent(audio, sr)
    return audio[i0:i1]


__all__ = [
    "SpeechMap", "Span", "build_speech_map", "load_speech_map", "detect_speech",
    "line_targets", "project_speech_map", "trust_unvoiced",
    "decode_mono", "speech_spans", "uncovered_regions", "speech_extent", "clip_extent",
    "clip_regions", "phrases", "span_phrases", "match_pauses",
    "trim_silence", "trim_to_speech",
]
