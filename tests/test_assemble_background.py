"""Assembly speed-ups: cached background stem, block-wise Demucs, and bounded
assemble_timeline memory (EDITOR_PLAN 2.4–2.6)."""

from __future__ import annotations

import os
import sys
import time
import tracemalloc
import types

import numpy as np
import pytest
import soundfile as sf

from mazinger import assemble
from mazinger.paths import ProjectPaths

from .conftest import SR, write_tone


# ---------------------------------------------------------------------------
#  Background cache (2.4)
# ---------------------------------------------------------------------------

def _files(d: str) -> list[str]:
    return sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))


@pytest.fixture
def fake_extract(monkeypatch):
    """Replace the Demucs/librosa extraction with a quick tone writer."""
    calls: list[str] = []

    def extract(audio_path, out_path, sr=assemble.TARGET_SR):
        calls.append(out_path)
        write_tone(out_path, sf.info(audio_path).duration, sr=sr, freq=110.0, amp=0.1)
        return out_path

    monkeypatch.setattr(assemble, "_extract_background", extract)
    return calls


def test_project_background_path():
    proj = ProjectPaths("demo", base_dir="/b", target_language="Spanish")
    assert proj.background_audio() == "/b/projects/demo/source/background.24000.wav"
    assert assemble.background_cache_path(proj.audio) == proj.background_audio(assemble.TARGET_SR)


def test_background_is_extracted_once_then_reused(mini_project, fake_extract):
    cache = mini_project.background_audio()
    assert assemble.extract_background_cached(mini_project.audio, cache) == cache
    assert assemble.extract_background_cached(mini_project.audio, cache) == cache
    assert len(fake_extract) == 1
    assert sf.info(cache).samplerate == assemble.TARGET_SR
    assert _files(mini_project.source_dir) == ["audio.mp3", "background.24000.wav"]


def test_background_is_re_extracted_when_source_is_newer(mini_project, fake_extract):
    cache = assemble.extract_background_cached(mini_project.audio)
    later = time.time() + 10
    os.utime(mini_project.audio, (later, later))
    assemble.extract_background_cached(mini_project.audio)
    assert len(fake_extract) == 2
    assert os.path.getmtime(cache) >= os.path.getmtime(mini_project.audio) - 10


def test_empty_cache_file_is_not_reused(mini_project, fake_extract):
    cache = mini_project.background_audio()
    open(cache, "wb").close()
    assemble.extract_background_cached(mini_project.audio, cache)
    assert len(fake_extract) == 1 and os.path.getsize(cache) > 0


def test_failed_extraction_leaves_no_partial_cache(mini_project, monkeypatch):
    def broken(audio_path, out_path, sr=assemble.TARGET_SR):
        with open(out_path, "wb") as fh:
            fh.write(b"partial")
        raise RuntimeError("out of memory")

    monkeypatch.setattr(assemble, "_extract_background", broken)
    with pytest.raises(RuntimeError):
        assemble.extract_background_cached(mini_project.audio)
    assert _files(mini_project.source_dir) == ["audio.mp3"]


def test_post_process_uses_the_cache(mini_project, fake_extract):
    cache = mini_project.background_audio()
    for _ in range(2):
        assemble.post_process(
            mini_project.final_audio, mini_project.audio, mini_project.final_audio,
            loudness_match=False, mix_background=True, background_cache=cache,
        )
    assert fake_extract == [fake_extract[0]] and fake_extract[0] != cache  # temp, then renamed
    assert os.path.isfile(cache)
    assert not os.path.exists(os.path.join(mini_project.tts_dir, "background.wav"))
    assert sf.info(mini_project.final_audio).duration == pytest.approx(10.0, abs=0.05)


def test_post_process_without_cache_keeps_old_behaviour(mini_project, fake_extract):
    for _ in range(2):
        assemble.post_process(
            mini_project.final_audio, mini_project.audio, mini_project.final_audio,
            loudness_match=False, mix_background=True,
        )
    bg = os.path.join(mini_project.tts_dir, "background.wav")
    assert fake_extract == [bg, bg]


# ---------------------------------------------------------------------------
#  Source loudness cache (Phase 6: the measurement is ~1 min per hour of audio)
# ---------------------------------------------------------------------------

@pytest.fixture
def count_loudness(monkeypatch):
    calls: list[str] = []
    real = assemble._loudness_or_none

    def measure(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(assemble, "_loudness_or_none", measure)
    return calls


def test_loudness_is_measured_once_then_reused(mini_project, count_loudness):
    cache = mini_project.source_loudness
    first = assemble.measure_loudness_cached(mini_project.audio, cache)
    assert assemble.measure_loudness_cached(mini_project.audio, cache) == first
    assert count_loudness == [mini_project.audio]
    assert first == pytest.approx(assemble._measure_loudness(mini_project.audio))
    assert os.path.dirname(cache) == mini_project.source_dir


def test_loudness_is_measured_again_when_the_source_changes(mini_project, count_loudness):
    cache = mini_project.source_loudness
    before = assemble.measure_loudness_cached(mini_project.audio, cache)
    write_tone(mini_project.audio, 10.0, amp=0.05)
    after = assemble.measure_loudness_cached(mini_project.audio, cache)
    assert len(count_loudness) == 2 and after < before - 10


@pytest.mark.parametrize("junk", ["", "{", '{"source": 1}', "[]"])
def test_unreadable_loudness_cache_is_remeasured(mini_project, count_loudness, junk):
    with open(mini_project.source_loudness, "w") as fh:
        fh.write(junk)
    assert assemble.measure_loudness_cached(mini_project.audio, mini_project.source_loudness) < 0
    assert len(count_loudness) == 1


def test_failed_loudness_measurement_is_not_cached(mini_project, monkeypatch):
    monkeypatch.setattr(assemble, "_loudness_or_none", lambda path: None)
    assert assemble.measure_loudness_cached(mini_project.audio, mini_project.source_loudness) == -24.0
    assert not os.path.exists(mini_project.source_loudness)


def test_post_process_uses_the_loudness_cache(mini_project, count_loudness):
    for _ in range(2):
        assemble.post_process(
            mini_project.final_audio, mini_project.audio, mini_project.final_audio,
            loudness_match=True, mix_background=False,
            loudness_cache=mini_project.source_loudness,
        )
    assert count_loudness == [mini_project.audio]


# ---------------------------------------------------------------------------
#  Block-wise Demucs (2.5)
# ---------------------------------------------------------------------------

class FakeDemucs:
    samplerate = SR
    audio_channels = 2
    sources = ["drums", "bass", "other", "vocals"]


def test_demucs_blocks_join_without_seams(tmp_path, monkeypatch):
    """Stitched block output equals processing the whole file at once."""
    audio = str(tmp_path / "audio.wav")
    rng = np.random.default_rng(0)
    sf.write(audio, (rng.standard_normal(int(3.37 * SR)) * 0.1).astype("float32"), SR)

    blocks: list[int] = []

    def separate(model, block, sr, *, device, segment, overlap):
        blocks.append(len(block))
        assert block.shape[1] == 2 and sr == SR and segment is None
        return block.mean(axis=1)

    monkeypatch.setattr(assemble, "_load_demucs", lambda device: (FakeDemucs(), "cpu"))
    monkeypatch.setattr(assemble, "_demucs_background_block", separate)

    out = str(tmp_path / "bg.wav")
    assemble._extract_background_demucs(audio, out, SR, block_sec=1.0, context_sec=0.25)

    assert len(blocks) == 4
    assert max(blocks) <= int(1.5 * SR) + 1  # a block plus its context, never the file
    got, sr = sf.read(out, dtype="float32")
    ref = assemble._decode_audio(audio, SR, 2).mean(axis=1)
    assert sr == SR and len(got) == len(ref)
    np.testing.assert_allclose(got, ref, atol=1e-4)  # PCM_16 rounding


class FakeTensor:
    def __init__(self, a):
        self.a = np.asarray(a)

    def __getitem__(self, key):
        return FakeTensor(self.a[key])

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def _fake_torch_modules(monkeypatch, apply_calls, resample_calls):
    import contextlib

    torch = types.ModuleType("torch")
    torch.from_numpy = FakeTensor
    torch.no_grad = contextlib.nullcontext

    def resample(t, orig, new):
        resample_calls.append((orig, new))
        return FakeTensor(t.a[:: orig // new])

    torchaudio = types.ModuleType("torchaudio")
    torchaudio.functional = types.SimpleNamespace(resample=resample)

    def apply_model(model, mix, **kwargs):
        apply_calls.append(kwargs)
        chans = mix.a[0]  # (channels, samples)
        # Stems: drums=1, bass=2, other=3, vocals=100 (× the mix).
        return FakeTensor(np.stack([chans * k for k in (1, 2, 3, 100)])[None])

    demucs = types.ModuleType("demucs")
    demucs_apply = types.ModuleType("demucs.apply")
    demucs_apply.apply_model = apply_model
    for name, mod in {"torch": torch, "torchaudio": torchaudio,
                      "demucs": demucs, "demucs.apply": demucs_apply}.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_demucs_block_runs_split_on_device_and_drops_vocals(monkeypatch):
    apply_calls, resample_calls = [], []
    _fake_torch_modules(monkeypatch, apply_calls, resample_calls)

    model = FakeDemucs()
    model.samplerate = 48_000
    block = np.stack([np.full(96, 0.01), np.full(96, 0.03)], axis=1).astype("float32")
    bg = assemble._demucs_background_block(
        model, block, 24_000, device="cuda", segment=7.0, overlap=0.25,
    )
    assert apply_calls == [dict(device="cuda", split=True, segment=7.0,
                                overlap=0.25, progress=False)]
    assert resample_calls == [(48_000, 24_000)]
    # (1 + 2 + 3) × mean of the two channels; vocals excluded.
    np.testing.assert_allclose(bg, np.full(48, 6 * 0.02), rtol=1e-5)


def test_extract_background_falls_back_when_demucs_fails(tmp_path, monkeypatch):
    pytest.importorskip("librosa")
    audio = write_tone(str(tmp_path / "a.wav"), 1.0)

    def missing(*a, **k):
        raise ImportError("No module named 'demucs'")

    monkeypatch.setattr(assemble, "_extract_background_demucs", missing)
    out = assemble._extract_background(audio, str(tmp_path / "bg.wav"))
    assert sf.info(out).duration == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
#  assemble_timeline memory (2.6)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 7, 20, 23])
def test_blockwise_scans_match_numpy(monkeypatch, n):
    monkeypatch.setattr(assemble, "_SCAN_BLOCK", 7)
    rng = np.random.default_rng(n)
    a = np.zeros(n, dtype=np.float32)
    if n:
        a[: n // 2] = rng.standard_normal(n // 2)
    nz = np.nonzero(a)[0]
    assert assemble._last_nonzero(a) == (int(nz[-1]) if len(nz) else -1)
    assert assemble._peak_abs(a) == (float(np.max(np.abs(a))) if n else 0.0)


def test_assemble_timeline_peak_memory_stays_near_timeline_size(tmp_path, monkeypatch):
    """A long timeline must not need several times its own size in RAM."""
    duration = 600.0
    n = 200
    slot = duration / n
    seg = (0.5 * np.sin(np.arange(int(slot * 0.9 * SR)) / 10)).astype(np.float32)
    monkeypatch.setattr(assemble, "_load_and_resample", lambda path, sr: seg)
    segs = [dict(idx=str(i + 1), start=i * slot, end=(i + 1) * slot,
                 target_dur=slot, wav_path="x.wav") for i in range(n)]
    out = str(tmp_path / "out.wav")

    tracemalloc.start()
    try:
        assemble.assemble_timeline(segs, duration, out, tempo_mode="off")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    timeline_bytes = duration * SR * 4
    assert peak < 1.3 * timeline_bytes, f"peak {peak / timeline_bytes:.2f}× the timeline"
    assert sf.info(out).duration == pytest.approx(duration, abs=0.2)
