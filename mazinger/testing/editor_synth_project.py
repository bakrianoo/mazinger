"""Generate a synthetic finished dub for Editor performance checks.

Usage:
    python -m mazinger.testing.editor_synth_project [--out DIR] [--hours H]
           [--chunks N] [--language LANG] [--video] [--seed N]

Builds ``<out>/projects/<slug>/`` with everything the Editor opens: a source
``audio.mp3`` of the requested length, the source and translated SRTs, one
placeholder WAV per chunk, ``run.json``, an initial ``dubbed.wav``, and the
cached background stem and source loudness a dub leaves behind (so assembly
never runs Demucs).  No model is loaded.

The chunks look like a real dub: slot lengths vary, some dubs overrun their
slot (they need a tempo change), and most chunks map to one or two source
entries, some of which straddle a chunk boundary.

Defaults: 2 h, 2,500 chunks — the Editor's scale target.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import time

import numpy as np
import soundfile as sf

from mazinger.assemble import measure_loudness_cached
from mazinger.paths import ProjectPaths
from mazinger.runinfo import project_relpath, save_run_info
from mazinger.srt import build

SR = 24_000
DEFAULT_SLUG = "editor-synthetic"

_SOURCE_WORDS = (
    "the model learns a simple rule from data and then we check how well it works "
    "on examples it has never seen before which is the whole point of training"
).split()
_TARGET_WORDS = (
    "el modelo aprende una regla simple de los datos y luego comprobamos si funciona "
    "con ejemplos que nunca ha visto antes que es el objetivo del entrenamiento"
).split()


def _sentence(rng: random.Random, words: list[str], n: int) -> str:
    start = rng.randrange(len(words))
    text = " ".join(words[(start + k) % len(words)] for k in range(n))
    return text[0].upper() + text[1:] + "."


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
                   check=True, capture_output=True)


def plan_chunks(rng: random.Random, hours: float, n_chunks: int) -> list[tuple[float, float]]:
    """``n_chunks`` non-overlapping ``(start, end)`` slots covering ~*hours*."""
    total = hours * 3600
    # Random slot and gap weights, scaled so the chunks fill the duration.
    slots = [rng.uniform(1.2, 5.0) for _ in range(n_chunks)]
    gaps = [rng.choice((0.0, 0.0, 0.1, 0.3, 0.8)) for _ in range(n_chunks)]
    scale = (total - 1.0) / (sum(slots) + sum(gaps))
    out, t = [], 0.5
    for slot, gap in zip(slots, gaps):
        start = round(t, 3)
        end = round(t + slot * scale, 3)
        out.append((start, end))
        t = end + gap * scale
    return out


def generate(
    out_dir: str,
    *,
    hours: float = 2.0,
    n_chunks: int = 2500,
    language: str = "Spanish",
    slug: str = DEFAULT_SLUG,
    video: bool = False,
    seed: int = 0,
    log=print,
) -> ProjectPaths:
    """Write the synthetic project and return its paths."""
    rng = random.Random(seed)
    proj = ProjectPaths(slug, base_dir=out_dir, target_language=language).ensure_dirs()
    duration = hours * 3600
    t0 = time.perf_counter()

    # -- source media --------------------------------------------------------
    # A tone plus noise, so loudness measurement and clip cutting do real work.
    log(f"Source audio ({hours:g} h)…")
    _ffmpeg("-f", "lavfi", "-i", f"sine=frequency=180:sample_rate=44100:duration={duration}",
            "-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude=0.05:sample_rate=44100:duration={duration}",
            "-filter_complex", "[0:a]volume=0.2[a];[a][1:a]amix=inputs=2:duration=first",
            "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", proj.audio)
    if video:
        log("Source video…")
        _ffmpeg("-f", "lavfi", "-i", f"color=c=0x202830:s=320x180:r=5:d={duration}",
                "-i", proj.audio, "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage",
                "-c:a", "aac", "-b:a", "64k", "-shortest", proj.video)

    # What a dub leaves in source/ for later assemblies: the source loudness
    # (really measured) and the background stem, which must be newer than
    # audio.mp3 to be reused.
    log("Source loudness…")
    measure_loudness_cached(proj.audio, proj.source_loudness)
    log("Cached background stem…")
    _ffmpeg("-f", "lavfi", "-i", f"anoisesrc=color=brown:amplitude=0.1:sample_rate={SR}:duration={duration}",
            "-ac", "1", proj.background_audio(SR))

    # -- chunks, SRTs and segments ------------------------------------------
    log(f"{n_chunks:,} chunks…")
    slots = plan_chunks(rng, hours, n_chunks)
    sources, raw, final = [], [], []
    for start, end in slots:
        tgt = _sentence(rng, _TARGET_WORDS, max(2, round((end - start) * 2.6)))
        final.append((start, end, tgt))
        # One or two source entries per chunk; a split point sometimes lands
        # past the chunk end so the entry straddles the boundary.
        if rng.random() < 0.35 and end - start > 2.0:
            mid = round(rng.uniform(start + 0.6, end - 0.6), 3)
            parts = [(start, mid), (mid, round(end + rng.choice((0.0, 0.0, 0.4)), 3))]
        else:
            parts = [(start, end)]
        for s, e in parts:
            text = _sentence(rng, _SOURCE_WORDS, max(2, round((e - s) * 2.5)))
            sources.append((s, e, text))
            raw.append((s, e, _sentence(rng, _TARGET_WORDS, max(2, round((e - s) * 2.6)))))
    # Straddling entries may overlap the next one; keep the SRT ordered and valid.
    for k in range(1, len(sources)):
        if sources[k][0] < sources[k - 1][1]:
            s, e, t = sources[k - 1]
            sources[k - 1] = (s, sources[k][0], t)
            s, e, t = raw[k - 1]
            raw[k - 1] = (s, sources[k][0], t)

    for path, entries in ((proj.source_raw_srt, sources), (proj.source_srt, sources),
                          (proj.translated_raw_srt, raw), (proj.final_srt, final)):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(build(entries))

    # Placeholder dubs: a tone of 0.7–1.3× the slot, so ~1 in 6 overruns
    # its slot by more than 15% and many need a tempo change.
    for i, (start, end) in enumerate(slots, 1):
        seconds = (end - start) * rng.uniform(0.7, 1.3)
        n = int(seconds * SR)
        t = np.arange(n) / SR
        tone = 0.25 * np.sin(2 * np.pi * rng.uniform(150, 300) * t)
        env = np.minimum(1.0, np.minimum(t, seconds - t) / 0.05)
        sf.write(os.path.join(proj.tts_segments_dir, f"seg_{i:04d}.wav"),
                 (tone * env).astype(np.float32), SR, subtype="PCM_16")

    # An earlier output, so assembly exercises its backups.
    _ffmpeg("-f", "lavfi", "-i", f"anullsrc=r={SR}:cl=mono", "-t", str(duration), proj.final_audio)

    # -- run.json --------------------------------------------------------------
    voice_wav = os.path.join(proj.voice_reference_dir, "voice.wav")
    os.makedirs(proj.voice_reference_dir, exist_ok=True)
    sf.write(voice_wav, (0.2 * np.sin(np.arange(6 * SR) / 20)).astype(np.float32), SR)
    with open(os.path.join(proj.voice_reference_dir, "script.txt"), "w", encoding="utf-8") as fh:
        fh.write("A synthetic reference voice.")
    save_run_info(proj, dict(
        slug=slug, target_language=language, source_language="English",
        detected_source_language="en",
        transcription=dict(method="faster-whisper", model="large-v3", beam_size=5),
        translation_source_srt=project_relpath(proj, proj.source_srt),
        llm=dict(model="gpt-4.1", base_url=None, think=None),
        translation=dict(words_per_second=None, duration_budget=None,
                         translate_technical_terms=False, user_instructions="",
                         translation_model=None),
        tts=dict(engine="qwen", model=None, dtype="bfloat16", language=language),
        voice=dict(kind="sample", theme=None, sample=project_relpath(proj, voice_wav),
                   script=project_relpath(proj, os.path.join(proj.voice_reference_dir, "script.txt")),
                   instruct=None),
        segmentation=dict(mode="resegment"),
        assembly=dict(tempo_mode="auto", fixed_tempo=None, max_tempo=1.5,
                      loudness_match=True, mix_background=True, background_volume=0.15),
        output=dict(output_type="video" if video else "audio",
                    subtitle_style=None, subtitle_source="translated"),
        synthetic=True,
    ))
    log(f"Done in {time.perf_counter() - t0:.0f} s: {proj.root}")
    return proj


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="mazinger/testing/output/editor",
                    help="Base directory (the project goes under <out>/projects/<slug>/)")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--chunks", type=int, default=2500)
    ap.add_argument("--language", default="Spanish")
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    ap.add_argument("--video", action="store_true", help="Also write a source video (slower)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    generate(args.out, hours=args.hours, n_chunks=args.chunks, language=args.language,
             slug=args.slug, video=args.video, seed=args.seed)


if __name__ == "__main__":
    main()
