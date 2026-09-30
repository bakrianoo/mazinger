"""The "sync" tempo mode: speech map, per-line targets and exact assembly."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import soundfile as sf

from mazinger import assemble, speech, tts
from mazinger.speech import SpeechMap, Span, speech_spans, uncovered_regions
from tests.conftest import SR, FakeVoicePrompt, write_tone


def smap(regions, duration=20.0):
    return SpeechMap(duration=duration, regions=regions)


def tone(seconds, sr=SR, freq=220.0, amp=0.3, lead=0.0, tail=0.0):
    t = np.arange(int(seconds * sr)) / sr
    a = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.concatenate([np.zeros(int(lead * sr), np.float32), a,
                           np.zeros(int(tail * sr), np.float32)])


# ---------------------------------------------------------------------------
#  Lines ↔ speech
# ---------------------------------------------------------------------------

class TestSpeechSpans:
    def test_span_reaches_the_real_edges_of_the_speech(self):
        # ASR says 1.0-3.0; the speaker really talks 1.2-3.2.
        [sp] = speech_spans([(1.0, 3.0)], smap([(1.2, 3.2)]))
        assert (sp.onset, sp.offset, sp.voiced) == (1.2, 3.2, True)

    def test_leading_and_trailing_silence_in_the_entry_is_dropped(self):
        [sp] = speech_spans([(0.0, 5.0)], smap([(1.5, 2.0), (2.3, 3.4)]))
        assert (sp.onset, sp.offset) == (1.5, 3.4)

    def test_edges_stop_at_the_tolerance(self):
        [sp] = speech_spans([(2.0, 3.0)], smap([(1.0, 4.5)]), tolerance=0.3)
        assert sp.onset == pytest.approx(1.7) and sp.offset == pytest.approx(3.3)

    def test_edges_never_reach_into_a_neighbour(self):
        spans = speech_spans([(0.0, 2.0), (2.1, 4.0)], smap([(0.1, 2.05), (2.05, 3.9)]))
        assert spans[0].offset <= 2.1 <= spans[1].onset
        assert spans[0].offset <= spans[1].onset

    def test_entry_without_speech_keeps_its_span_and_is_unvoiced(self):
        [sp] = speech_spans([(5.0, 6.0)], smap([(1.0, 2.0)]))
        assert (sp.onset, sp.offset, sp.voiced) == (5.0, 6.0, False)

    def test_speech_just_outside_the_entry_still_counts(self):
        [sp] = speech_spans([(1.0, 2.0)], smap([(2.05, 2.8)]))
        assert sp.voiced and sp.onset == pytest.approx(2.05)

    def test_without_a_map_every_span_is_the_entry(self):
        spans = speech_spans([(1.0, 2.0), (3.0, 4.0)], None)
        assert [(s.onset, s.offset, s.voiced) for s in spans] == [(1.0, 2.0, False), (3.0, 4.0, False)]

    def test_results_follow_the_input_order(self):
        spans = speech_spans([(3.0, 4.0), (1.0, 2.0)], smap([(1.1, 1.9), (3.1, 3.9)]))
        assert [round(s.onset, 1) for s in spans] == [3.1, 1.1]


def test_uncovered_regions_are_vocals_no_line_covers():
    m = smap([(0.5, 2.0), (2.6, 3.0), (5.0, 7.0)])
    spans = [Span(0.5, 2.0), Span(5.2, 7.0)]
    got = uncovered_regions(m, spans, margin=0.1, min_len=0.12)
    assert got == [(2.6, 3.0)]


def test_line_targets_skip_hallucinations_only_while_few():
    entries = [{"idx": str(i), "start": float(2 * i), "end": 2 * i + 1.5, "text": "x"} for i in range(8)]
    regions = [(2 * i + 0.1, 2 * i + 1.4) for i in range(7)]  # line 7 has no speech
    targets, unvoiced = speech.line_targets(entries, smap(regions))
    assert unvoiced == {"7"}
    assert targets["0"] == pytest.approx(1.3)

    # A map that finds no speech at all is not trusted: nothing is skipped.
    _, unvoiced = speech.line_targets(entries, smap([]))
    assert unvoiced == set()


# ---------------------------------------------------------------------------
#  Detection and the cached map
# ---------------------------------------------------------------------------

def test_speech_extent_finds_the_sound():
    a = tone(1.0, lead=0.5, tail=0.7)
    i0, i1 = speech.speech_extent(a, SR, pad=0.0)
    assert abs(i0 / SR - 0.5) < 0.011 and abs(i1 / SR - 1.5) < 0.011


def test_clip_extent_falls_back_to_energy_without_vad(monkeypatch):
    def no_vad(audio):
        raise ImportError("no faster-whisper")
    monkeypatch.setattr(speech, "_silero_regions", no_vad)
    a = tone(1.0, lead=0.3, tail=0.3)
    assert speech.clip_extent(a, SR) == speech.speech_extent(a, SR)


def test_build_speech_map_is_cached_until_the_audio_changes(tmp_path, monkeypatch):
    calls = []

    def fake_detect(audio, sr=speech.VAD_SR):
        calls.append(len(audio))
        return [(0.5, 1.0)], "fake"

    monkeypatch.setattr(speech, "detect_speech", fake_detect)
    audio = write_tone(str(tmp_path / "a.wav"), 2.0)
    cache = str(tmp_path / "speech_map.json")

    m1 = speech.build_speech_map(audio, cache)
    m2 = speech.build_speech_map(audio, cache)
    assert len(calls) == 1 and m1.regions == m2.regions == [(0.5, 1.0)]
    assert m1.duration == pytest.approx(2.0, abs=1e-3)

    write_tone(audio, 3.0)
    os.utime(audio, ns=(1, os.stat(cache).st_mtime_ns + 10**9))
    m3 = speech.build_speech_map(audio, cache)
    assert len(calls) == 2 and m3.duration == pytest.approx(3.0, abs=1e-3)


def test_energy_vad_finds_tone_bursts():
    sr = speech.VAD_SR
    a = np.concatenate([np.zeros(sr, np.float32), tone(1.0, sr=sr), np.zeros(sr, np.float32),
                        tone(0.5, sr=sr), np.zeros(sr // 2, np.float32)])
    regions = speech._energy_regions(a, sr)
    assert len(regions) == 2
    assert regions[0][0] == pytest.approx(1.0, abs=0.03)
    assert regions[1][1] == pytest.approx(3.5, abs=0.03)


# ---------------------------------------------------------------------------
#  Fitting one clip
# ---------------------------------------------------------------------------

class TestSyncPlan:
    def test_within_limits_the_clip_takes_the_target_length(self):
        assert assemble.sync_plan(2.4, 2.0, 5.0) == pytest.approx((1.2, 2.0, "sped_up"))
        rate, length, outcome = assemble.sync_plan(1.8, 2.0, 5.0)
        assert outcome == "slowed_down" and length == 2.0 and rate == pytest.approx(0.9)

    def test_near_exact_clips_are_not_stretched(self):
        assert assemble.sync_plan(2.01, 2.0, 5.0) == (1.0, 2.0, "exact")

    def test_too_long_uses_the_following_silence_up_to_the_room(self):
        rate, length, outcome = assemble.sync_plan(4.0, 2.0, 2.5, max_tempo=1.5)
        assert (rate, outcome) == (1.5, "too_long") and length == pytest.approx(2.5)
        rate, length, outcome = assemble.sync_plan(3.3, 2.0, 5.0, max_tempo=1.5)
        assert length == pytest.approx(2.2)

    def test_too_short_ends_early(self):
        rate, length, outcome = assemble.sync_plan(1.0, 2.0, 5.0, min_tempo=0.8)
        assert (rate, outcome) == (0.8, "too_short") and length == pytest.approx(1.25)


def test_stretch_audio_changes_length_by_the_rate():
    a = tone(2.0)
    out = assemble.stretch_audio(a, SR, 1.25)
    assert abs(len(out) / SR - 1.6) < 0.05
    assert assemble.fit_length(out, 1000, SR).shape == (1000,)
    assert len(assemble.fit_length(a[:500], 1000, SR)) == 1000


# ---------------------------------------------------------------------------
#  Assembly
# ---------------------------------------------------------------------------

def _segments(tmp_path, specs):
    segs = []
    for i, (start, end, secs) in enumerate(specs, 1):
        path = str(tmp_path / f"seg_{i:04d}.wav")
        sf.write(path, tone(secs, lead=0.2, tail=0.3), SR)
        segs.append({"idx": str(i), "start": start, "end": end, "wav_path": path,
                     "target_dur": end - start, "actual_dur": secs + 0.5})
    return segs


def _no_vad(monkeypatch):
    def no_vad(audio):
        raise ImportError
    monkeypatch.setattr(speech, "_silero_regions", no_vad)


def _onsets(audio, sr=SR):
    db = speech._frame_db(audio, int(sr * 0.005))
    on = db > -50
    edges = np.nonzero(on[1:] & ~on[:-1])[0] + 1
    return [e * 0.005 for e in edges]


def test_assembly_places_each_line_on_its_onset_at_its_length(tmp_path, monkeypatch):
    _no_vad(monkeypatch)
    m = smap([(1.2, 3.2), (4.5, 6.0)], duration=8.0)
    segs = _segments(tmp_path, [(1.0, 3.0, 2.4), (4.4, 6.2, 1.2)])
    out = str(tmp_path / "dub.wav")
    assemble.assemble_timeline(segs, 8.0, out, tempo_mode="sync", speech_map=m)

    audio, sr = sf.read(out, dtype="float32")
    assert len(audio) == 8 * SR                          # exactly the source length
    # The energy fallback keeps 20 ms before the first sound (speech_extent pad).
    onsets = _onsets(audio)
    assert onsets[0] == pytest.approx(1.22, abs=0.01)
    assert onsets[1] == pytest.approx(4.52, abs=0.01)

    report = json.load(open(str(tmp_path / "dub.sync.json")))
    first, second = report["segments"]
    assert first["placed"] == pytest.approx(2.0, abs=1e-3) and first["outcome"] == "sped_up"
    assert second["placed"] == pytest.approx(1.5, abs=1e-3) and second["outcome"] == "slowed_down"
    assert report["summary"]["samples"] == 8 * SR


def test_assembly_skips_lines_over_no_speech(tmp_path, monkeypatch):
    _no_vad(monkeypatch)
    m = smap([(0.5, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)], duration=12.0)
    segs = _segments(tmp_path, [(0.5, 2.0, 1.5), (3.0, 4.0, 1.0), (5.0, 6.0, 1.0),
                                (7.0, 8.0, 1.0), (9.5, 10.5, 1.0)])
    out = str(tmp_path / "dub.wav")
    assemble.assemble_synced(segs, 12.0, out, speech_map=m)
    audio, _ = sf.read(out, dtype="float32")
    assert np.abs(audio[int(9.4 * SR):]).max() == 0.0
    assert json.load(open(str(tmp_path / "dub.sync.json")))["summary"]["unvoiced"] == 1


def test_assembly_keeps_short_untranscribed_vocals(tmp_path, monkeypatch):
    _no_vad(monkeypatch)
    vocals = str(tmp_path / "vocals.wav")
    v = np.zeros(6 * SR, np.float32)
    v[int(3.0 * SR):int(3.5 * SR)] = tone(0.5, freq=440.0, amp=0.2)   # a laugh nobody transcribed
    v[int(4.0 * SR):int(6.0 * SR)] = tone(2.0, freq=330.0, amp=0.2)   # too long to keep
    sf.write(vocals, v, SR)
    m = smap([(0.5, 2.0), (3.0, 3.5), (4.0, 6.0)], duration=6.0)
    segs = _segments(tmp_path, [(0.5, 2.0, 1.5)])
    out = str(tmp_path / "dub.wav")
    assemble.assemble_synced(segs, 6.0, out, speech_map=m, vocals_path=vocals, passthrough_max=1.0)
    audio, _ = sf.read(out, dtype="float32")
    assert np.abs(audio[int(3.1 * SR):int(3.4 * SR)]).max() > 0.1
    assert np.abs(audio[int(4.2 * SR):]).max() == 0.0


def test_post_process_exact_keeps_length_and_timing(tmp_path):
    dub = str(tmp_path / "dub.wav")
    a = np.zeros(5 * SR, np.float32)
    a[SR:2 * SR] = tone(1.0)
    sf.write(dub, a, SR)
    source = write_tone(str(tmp_path / "src.wav"), 5.0, amp=0.05)
    bg = write_tone(str(tmp_path / "bg.wav"), 5.0, freq=60.0, amp=0.05)
    out = str(tmp_path / "out.wav")
    assemble.post_process(dub, source, out, loudness_match=True, mix_background=True,
                          background_volume=1.0, background_cache=bg,
                          loudness_cache=str(tmp_path / "l.json"), exact=True)
    got, _ = sf.read(out, dtype="float32")
    assert len(got) == len(a)
    ref = np.abs(a) > 0.1
    got_on = np.abs(got - np.mean(got)) > 0.1
    first = int(np.argmax(got_on))
    assert abs(first - int(np.argmax(ref))) < SR * 0.005


# ---------------------------------------------------------------------------
#  TTS: batching and targets
# ---------------------------------------------------------------------------

class BatchingVoice(FakeVoicePrompt):
    def __init__(self):
        super().__init__()
        self.batches: list[list] = []

    def synthesize_batch_to(self, items):
        self.batches.append(list(items))
        return [(tone(1.0 + 0.1 * len(t), lead=0.1, tail=0.2), SR) for t, _, _ in items]


def _entries(texts):
    return [{"idx": str(i), "start": float(2 * i), "end": 2 * i + 1.5, "text": t}
            for i, t in enumerate(texts, 1)]


def test_segments_are_synthesised_in_length_sorted_batches(tmp_path):
    voice = BatchingVoice()
    entries = _entries(["a long line here", "hi", "a middle line", "", "x"])
    targets = {e["idx"]: 1.25 for e in entries}
    info = tts.synthesize_segments(None, voice, entries, str(tmp_path), targets=targets,
                                   batch_size=2)
    assert [len(b) for b in voice.batches] == [2, 2]
    assert [t for t, _, _ in voice.batches[0]] == ["x", "hi"]   # shortest first
    assert all(target == 1.25 for b in voice.batches for _, _, target in b)
    assert info[3]["wav_path"] is None
    for rec in (r for r in info if r["wav_path"]):
        assert os.path.isfile(rec["wav_path"])
        assert rec["target_dur"] == 1.25
        assert rec["speech_dur"] < rec["actual_dur"]


def test_a_failed_batch_is_retried_line_by_line(tmp_path):
    class Flaky(BatchingVoice):
        def synthesize_batch_to(self, items):
            raise RuntimeError("CUDA out of memory")

    voice = Flaky()
    info = tts.synthesize_segments(None, voice, _entries(["one", "two"]), str(tmp_path))
    assert all(os.path.isfile(r["wav_path"]) for r in info)


def test_non_batching_engines_get_their_target(tmp_path):
    seen = []

    class Targeted(FakeVoicePrompt):
        def synthesize_to(self, text, language="English", target_dur=None):
            seen.append(target_dur)
            return tone(1.0), SR

    tts.synthesize_segments(None, Targeted(), _entries(["one", "two"]), str(tmp_path),
                            targets={"1": 0.9, "2": 1.1})
    assert seen == [0.9, 1.1]


def test_qwen_length_processor_pushes_toward_the_target_and_stops_runaways():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from types import SimpleNamespace

    model = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
        talker_config=SimpleNamespace(codec_eos_token_id=3))))
    procs, max_new = tts._qwen_length_processor(model, [0.8, None])   # 10 frames; no target
    assert max_new is None          # a row without a target keeps the model's own limit
    steer = procs[0]
    limit = 10 * tts.QWEN_RUNAWAY_FACTOR + tts.QWEN_RUNAWAY_SLACK / tts.QWEN_FRAME_SECONDS

    def eos_after(steps):
        ids = torch.zeros((2, steps), dtype=torch.long)
        return steer(ids, torch.zeros((2, 5)))

    steer.first_len = 0
    assert eos_after(5)[0, 3] == 0.0                      # before the target: untouched
    assert eos_after(12)[0, 3] == pytest.approx(tts.QWEN_EOS_PRESSURE * 2.0)
    stopped = eos_after(int(limit) + 1)
    assert stopped[0, 3] == 0.0 and torch.isinf(stopped[0, 0])
    assert stopped[1, 3] == 0.0 and not torch.isinf(stopped[1, 0])   # untargeted row runs on

    _, max_new = tts._qwen_length_processor(model, [0.8, 1.6])
    assert max_new == int(20 * tts.QWEN_RUNAWAY_FACTOR + tts.QWEN_RUNAWAY_SLACK / tts.QWEN_FRAME_SECONDS) + 2


# ---------------------------------------------------------------------------
#  Fit check and translation budget
# ---------------------------------------------------------------------------

def test_fit_ratios_in_sync_mode_use_speech_over_target():
    from mazinger.fit import fit_ratios
    segs = [{"idx": "1", "start": 0.0, "end": 3.0, "target_dur": 2.0, "wav_path": "x",
             "actual_dur": 3.0, "speech_dur": 2.5}]
    assert fit_ratios(segs, 10.0, sync=True) == [pytest.approx(1.25)]


def test_translation_budget_uses_measured_speech():
    from mazinger.translate import _blocks_to_json_entries
    blocks = [("1", 0.0, 6.0, "hello there"), ("2", 6.0, 12.0, "bye")]
    got = json.loads(_blocks_to_json_entries(blocks, 3.0, 1.0, {"1": 2.0}))
    assert got[0]["target_words"] == 6        # 2 s of real speech
    assert got[1]["target_words"] == 18       # no measurement: the 6 s span


# ---------------------------------------------------------------------------
#  Editor and CLI
# ---------------------------------------------------------------------------

def test_editor_fit_uses_the_speech_target_in_sync_mode(mini_project):
    from mazinger.editor.session import Session
    from mazinger.runinfo import load_run_info, save_run_info

    with open(mini_project.speech_map, "w") as fh:
        json.dump({"version": speech.SPEECH_MAP_VERSION, "duration": 10.0,
                   "regions": [[0.3, 1.5], [2.2, 4.0], [4.6, 6.5], [7.6, 9.5]]}, fh)
    session = Session.import_project(mini_project, save=False)
    assert all(c.target is None for c in session.chunks)   # run.json says "auto"

    info = load_run_info(mini_project) or {}
    info.setdefault("assembly", {})["tempo_mode"] = "sync"
    save_run_info(mini_project, info)
    session = Session.import_project(mini_project, save=False)
    c = session.chunks[0]
    assert c.target == pytest.approx(1.2)
    assert c.fit_ratio == pytest.approx(c.dub_dur / 1.2)


def test_cli_defaults_to_sync():
    import argparse
    from mazinger.cli._groups import add_tempo, tempo_mode_from_args
    p = argparse.ArgumentParser()
    add_tempo(p)
    assert tempo_mode_from_args(p.parse_args([])) == "sync"
    assert tempo_mode_from_args(p.parse_args(["--dynamic-tempo"])) == "dynamic"
    assert tempo_mode_from_args(p.parse_args(["--tempo-mode", "auto"])) == "auto"


# ---------------------------------------------------------------------------
#  Pause-aware placement
# ---------------------------------------------------------------------------

def test_pauses_are_matched_by_relative_position():
    src = [(0.0, 2.0), (2.5, 4.0), (4.6, 6.0)]
    clip = [(0.0, 1.8), (2.0, 3.5), (3.7, 4.1), (4.3, 5.6)]   # one extra comma pause
    assert speech.match_pauses(src, clip) == [(0, 0), (1, 2)]
    assert speech.match_pauses(src, [(0.0, 5.0)]) == []
    # A pause far from any of the original's is not matched.
    assert speech.match_pauses([(0, 1), (1.5, 5)], [(0, 3.5), (4, 4.5)]) == []


def test_a_line_over_two_phrases_pauses_where_the_speaker_paused(tmp_path, monkeypatch):
    _no_vad(monkeypatch)
    sr = SR
    # The clip: 1.0 s, 0.25 s pause, 1.2 s.  The original: 1.1 s, 0.6 s pause, 1.3 s.
    clip = np.concatenate([tone(1.0), np.zeros(int(0.25 * sr), np.float32), tone(1.2)])
    regions = [(0.0, 1.0), (1.25, 2.45)]
    out, pieces = assemble._place_phrases(
        clip, regions, [(5.0, 6.1), (6.7, 8.0)], 5.0, 8.0, sr, max_tempo=1.5, min_tempo=0.8)
    assert pieces == 2 and len(out) == 3 * sr
    loud = np.abs(out) > 0.05
    assert not loud[int(1.2 * sr):int(1.6 * sr)].any()            # the original pause is kept
    assert loud[int(0.05 * sr):int(0.1 * sr)].any() and loud[int(1.8 * sr):int(1.85 * sr)].any()

    # A piece needing more than max_tempo falls back to whole-line fitting.
    assert assemble._place_phrases(
        clip, regions, [(5.0, 5.5), (6.7, 8.0)], 5.0, 8.0, sr, max_tempo=1.5, min_tempo=0.8) is None
