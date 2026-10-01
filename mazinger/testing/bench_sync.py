"""Measure how closely a dub follows the timing of its source.

Compares the voice track of a dub (before background mixing) with the
speech map of the source:

* ``length_delta_ms`` — output length minus source length.
* ``onset_error_ms`` — per line, where the dub's speech starts versus the
  original speech it replaces (median / p90 / max).
* ``length_ratio`` — per line, dub speech length over original speech
  length (median and the share of lines within ±5% and ±10%).
* ``speech_iou`` — overlap of voiced time between dub and source.

Usage::

    python -m mazinger.testing.bench_sync SOURCE_AUDIO DUB_VOICE SRT [--vocals VOCALS]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mazinger.speech import VAD_SR, SpeechMap, decode_mono, detect_speech, speech_spans


def _voiced_mask(regions: list[tuple[float, float]], n: int, hop: float) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for s, e in regions:
        mask[int(s / hop):int(np.ceil(e / hop))] = True
    return mask


def measure(
    source: SpeechMap,
    dub_audio: np.ndarray,
    entries: list[tuple[float, float]],
    *,
    source_samples: int | None = None,
    sr: int = VAD_SR,
) -> dict:
    """Timing report of *dub_audio* (mono, at *sr*) against *source*."""
    dub_regions, _ = detect_speech(dub_audio, sr)
    dub = SpeechMap(duration=len(dub_audio) / sr, regions=dub_regions)
    src_spans = [sp for sp in speech_spans(entries, source) if sp.voiced]
    src_spans.sort(key=lambda sp: sp.onset)

    # Each dub region belongs to the last line whose onset it follows
    # (with a little slack), so an overrun counts against its own line.
    onset_err, ratios = [], []
    onsets = np.array([sp.onset for sp in src_spans])
    owned: dict[int, list[tuple[float, float]]] = {}
    for s, e in dub_regions:
        if not len(onsets) or e - s < 0.08:
            continue
        k = int(np.searchsorted(onsets, s + 0.25, side="right")) - 1
        if k >= 0:
            owned.setdefault(k, []).append((s, e))
    for k, sp in enumerate(src_spans):
        if k not in owned:
            continue
        d_on, d_off = owned[k][0][0], owned[k][-1][1]
        onset_err.append(abs(d_on - sp.onset) * 1000)
        ratios.append((d_off - d_on) / max(sp.duration, 1e-3))

    hop = 0.01
    n = int(max(source.duration, dub.duration) / hop) + 1
    a = _voiced_mask(source.regions, n, hop)
    b = _voiced_mask(dub_regions, n, hop)
    iou = float((a & b).sum() / max((a | b).sum(), 1))

    ratios_a = np.array(ratios) if ratios else np.zeros(1)
    onset_a = np.array(onset_err) if onset_err else np.zeros(1)
    src_n = source_samples if source_samples is not None else int(round(source.duration * sr))
    return {
        "lines": len(src_spans),
        "matched": len(ratios),
        "length_delta_ms": round((len(dub_audio) - src_n) / sr * 1000, 2),
        "onset_error_ms": {
            "median": round(float(np.median(onset_a)), 1),
            "p90": round(float(np.percentile(onset_a, 90)), 1),
            "max": round(float(onset_a.max()), 1),
        },
        "length_ratio": {
            "median": round(float(np.median(ratios_a)), 3),
            "within_5pct": round(float(np.mean(np.abs(ratios_a - 1) <= 0.05)), 3),
            "within_10pct": round(float(np.mean(np.abs(ratios_a - 1) <= 0.10)), 3),
        },
        "speech_iou": round(iou, 3),
    }


def main(argv: list[str] | None = None) -> None:
    from mazinger.speech import build_speech_map
    from mazinger.srt import parse_file

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="Source audio")
    ap.add_argument("dub", help="Dub voice track (before background mixing)")
    ap.add_argument("srt", help="SRT the dub was made from")
    ap.add_argument("--vocals", help="Vocals stem of the source")
    args = ap.parse_args(argv)

    smap = build_speech_map(args.source, vocals_path=args.vocals)
    entries = [(e["start"], e["end"]) for e in parse_file(args.srt)]
    report = measure(smap, decode_mono(args.dub, VAD_SR), entries)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
