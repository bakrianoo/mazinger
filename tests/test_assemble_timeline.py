"""``assemble_timeline`` speed-ups keep its output unchanged.

Segments are read without ffmpeg when they are lossless and already at the
target rate, and prepared (loaded, stretched, trimmed) on a thread pool while
placement stays in order.  Both must be invisible in the result.
"""

from __future__ import annotations

import logging
import subprocess

import numpy as np
import pytest
import soundfile as sf

from mazinger import assemble

SR = assemble.TARGET_SR


def _ffmpeg_load(path: str, sr: int) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", str(sr), "-ac", "1", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def _noise(seconds: float, channels: int = 1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (int(seconds * SR),) if channels == 1 else (int(seconds * SR), channels)
    return (0.3 * rng.standard_normal(shape)).clip(-1, 1).astype(np.float32)


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
def test_direct_read_matches_ffmpeg(tmp_path, subtype):
    path = str(tmp_path / "seg.wav")
    sf.write(path, _noise(0.5), SR, subtype=subtype)
    direct = assemble._load_and_resample(path, SR)
    assert direct.dtype == np.float32 and direct.ndim == 1
    np.testing.assert_array_equal(direct, _ffmpeg_load(path, SR))


def test_other_files_go_through_ffmpeg(tmp_path, monkeypatch):
    """Other rates, stereo (ffmpeg's downmix is not a plain average) and
    lossy files (decoders differ on encoder padding)."""
    wav16k = str(tmp_path / "a.wav")
    sf.write(wav16k, _noise(0.5)[:8000], 16_000)
    stereo = str(tmp_path / "s.wav")
    sf.write(stereo, _noise(0.5, channels=2), SR)
    mp3 = str(tmp_path / "b.mp3")
    subprocess.run(["ffmpeg", "-y", "-i", wav16k, "-ar", str(SR), mp3], capture_output=True, check=True)

    calls = []
    real_run = subprocess.run
    monkeypatch.setattr(assemble.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or real_run(cmd, **kw))
    assert len(assemble._load_and_resample(wav16k, SR)) == pytest.approx(0.5 * SR, abs=64)
    assert assemble._load_and_resample(stereo, SR).ndim == 1
    assemble._load_and_resample(mp3, SR)
    assert len(calls) == 3


def _segments(tmp_path, n: int = 24) -> tuple[list[dict], float]:
    """Segments that need every treatment: speed-up, slow-down, fits, overruns."""
    rng = np.random.default_rng(1)
    segs, t = [], 0.3
    for i in range(n):
        slot = float(rng.uniform(0.8, 2.0))
        ratio = [0.6, 0.85, 1.0, 1.3, 2.2][i % 5]  # 2.2 overruns even at max tempo
        path = str(tmp_path / f"seg_{i:04d}.wav")
        sf.write(path, _noise(slot * ratio, seed=i), SR, subtype="PCM_16")
        segs.append(dict(idx=str(i + 1), start=round(t, 3), end=round(t + slot, 3),
                         target_dur=slot, wav_path=path))
        t += slot + float(rng.choice([0.0, 0.2]))
    return segs, t + 0.5


@pytest.mark.parametrize("tempo_mode", ["auto", "off", "fixed"])
def test_parallel_output_is_identical_to_sequential(tmp_path, monkeypatch, tempo_mode):
    segs, duration = _segments(tmp_path)
    kw = dict(tempo_mode=tempo_mode, fixed_tempo=1.1 if tempo_mode == "fixed" else None)

    monkeypatch.setattr(assemble, "ASSEMBLE_WORKERS", 1)
    one = str(tmp_path / "one.wav")
    assemble.assemble_timeline(segs, duration, one, **kw)

    monkeypatch.setattr(assemble, "ASSEMBLE_WORKERS", 8)
    many = str(tmp_path / "many.wav")
    assemble.assemble_timeline(segs, duration, many, **kw)

    a, _ = sf.read(one, dtype="float32")
    b, _ = sf.read(many, dtype="float32")
    assert np.array_equal(a, b)
    assert np.abs(a).max() > 0


def test_stats_and_trim_warnings_are_reported(tmp_path, caplog):
    segs, duration = _segments(tmp_path)
    with caplog.at_level(logging.INFO, logger="mazinger.assemble"):
        assemble.assemble_timeline(segs, duration, str(tmp_path / "out.wav"))
    summary = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Timeline assembled"))
    counts = dict(part.split("=") for part in summary.split("|")[1].split())
    assert int(counts["sped_up"]) > 0 and int(counts["slowed"]) > 0 and int(counts["trimmed"]) > 0
    assert int(counts["sped_up"]) + int(counts["slowed"]) + int(counts["ok"]) == len(segs)
    assert any("trimmed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_empty_segment_is_skipped(tmp_path):
    segs, duration = _segments(tmp_path, n=3)
    sf.write(segs[1]["wav_path"], np.zeros(0, dtype=np.float32), SR)
    out = assemble.assemble_timeline(segs, duration, str(tmp_path / "out.wav"))
    assert sf.info(out).duration == pytest.approx(duration, abs=0.2)


def test_a_failing_segment_raises(tmp_path):
    segs, duration = _segments(tmp_path, n=40)
    segs[20]["wav_path"] = str(tmp_path / "missing.wav")
    with pytest.raises(subprocess.CalledProcessError):
        assemble.assemble_timeline(segs, duration, str(tmp_path / "out.wav"))


def test_ordered_map_bounds_work_in_flight():
    from concurrent.futures import ThreadPoolExecutor
    import threading

    running = peak = 0
    lock = threading.Lock()

    def work(i):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        with lock:
            running -= 1
        return i * i

    with ThreadPoolExecutor(4) as pool:
        submitted = []
        orig_submit = pool.submit
        pool.submit = lambda fn, x: submitted.append(x) or orig_submit(fn, x)
        gen = assemble._ordered_map(pool, work, range(100), window=5)
        first = next(gen)
        assert first == 0 and len(submitted) <= 5
        assert [first] + list(gen) == [i * i for i in range(100)]
