"""Editor operations with mocked models (EDITOR_PLAN phase 4).

Chunks of the ``mini_project`` fixture: c1 0–2 s, c2 2–7 s, c3 7.5–10 s.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from mazinger import gpu, tts
from mazinger import transcribe as transcribe_mod
from mazinger.editor import media, ops
from mazinger.editor.session import STALE_CHECK, STALE_DUB, STALE_TRANSLATION, Session
from mazinger.runinfo import save_run_info

from .conftest import SR, FakeLLMClient, FakeTranscriber, FakeVoicePrompt, write_tone


def _run_info(proj, **over):
    info = dict(
        target_language="Spanish", source_language="English", detected_source_language="en",
        transcription=dict(method="faster-whisper", model="large-v3", beam_size=5, vad_method="silero"),
        llm=dict(model="test-llm", base_url="http://llm.invalid/v1", think=None),
        translation=dict(translation_model=None, words_per_second=3.0, duration_budget=0.85,
                         translate_technical_terms=False, user_instructions="Keep it formal."),
        tts=dict(engine="qwen", model="Qwen/Qwen3-TTS-12Hz-1.7B-Base", dtype="bfloat16", language="Spanish"),
        voice=dict(kind="sample", sample="lang/Spanish/voice_profile/reference/voice.wav",
                   script="lang/Spanish/voice_profile/reference/script.txt"),
        assembly=dict(tempo_mode="auto", fixed_tempo=None, max_tempo=1.5,
                      loudness_match=False, mix_background=False, background_volume=0.15),
        output=dict(output_type="audio", subtitle_style=None, subtitle_source="translated"),
    )
    for k, v in over.items():
        info[k] = {**info[k], **v} if isinstance(v, dict) and isinstance(info.get(k), dict) else v
    save_run_info(proj, info)


@pytest.fixture
def project(mini_project):
    ref = os.path.join(mini_project.voice_reference_dir, "voice.wav")
    write_tone(ref, 5.0)
    with open(os.path.join(mini_project.voice_reference_dir, "script.txt"), "w") as fh:
        fh.write("Reference words.")
    _run_info(mini_project)
    return mini_project


@pytest.fixture
def session(project) -> Session:
    return Session.import_project(project)


@pytest.fixture
def voice() -> FakeVoicePrompt:
    return FakeVoicePrompt()


@pytest.fixture
def res(session, voice, fake_llm) -> ops.Resources:
    return ops.Resources(session, device="cpu", llm_client=fake_llm, voice_prompt=voice)


def _drain(gen) -> list[ops.Progress]:
    return list(gen)


# ═══════════════════════════════════════════════════════════════════════════════
#  Resources
# ═══════════════════════════════════════════════════════════════════════════════

def test_asr_settings_come_from_run_json(res):
    kw = res.asr_settings()
    assert kw["method"] == "faster-whisper" and kw["model"] == "large-v3"
    assert kw["language"] == "en" and kw["beam_size"] == 5 and kw["vad_method"] == "silero"
    assert kw["device"] == "cpu"
    assert "openai_api_key" not in kw


def test_missing_run_json_needs_settings(mini_project):
    s = Session.import_project(mini_project)
    res = ops.Resources(s, device="cpu", voice_prompt=FakeVoicePrompt())
    with pytest.raises(ops.SettingsMissing):
        res.asr_settings()
    with pytest.raises(ops.SettingsMissing):
        _drain(ops.retranscribe(s, ["c1"], res))
    # Settings entered in the UI (same shape as run.json) are accepted.
    res = ops.Resources(s, device="cpu", settings={"transcription": {"method": "openai"}})
    assert res.asr_settings()["method"] == "openai"


def test_voice_for_a_kept_sample(res, project):
    audio, text = res.resolve_voice()
    assert audio == os.path.join(project.voice_reference_dir, "voice.wav")
    assert text == "Reference words."


def test_voice_design_run_promotes_a_segment(project, session):
    _run_info(project, voice=dict(kind="theme-instruct", sample=None, script=None,
                                  instruct="female, low pitch"),
              tts=dict(engine="omnivoice"))
    res = ops.Resources(Session.load(project), device="cpu")
    audio, text = res.resolve_voice()
    assert audio == os.path.join(project.voice_profile_dir, "omnivoice_design", "voice.wav")
    assert text == "Hoy hablamos de redes neuronales."   # the 4.5 s segment is in range
    assert res.notices and "reference" in res.notices[0]
    res2 = ops.Resources(Session.load(project), device="cpu")
    res2.resolve_voice()
    assert not res2.notices                              # reused, not promoted again


def test_voice_unavailable(project, session):
    os.remove(os.path.join(project.voice_reference_dir, "voice.wav"))
    res = ops.Resources(session, device="cpu")
    with pytest.raises(ops.VoiceUnavailable):
        res.resolve_voice()


def test_voice_is_loaded_once_with_run_settings(res, monkeypatch, project):
    res._voice = None
    calls = []
    monkeypatch.setattr(tts, "load_model", lambda **kw: calls.append(kw) or "model")
    monkeypatch.setattr(tts, "create_voice_prompt",
                        lambda model, ref, text, **kw: calls.append((model, ref, text, kw)) or FakeVoicePrompt())
    v = res.voice()
    assert res.voice() is v and len(calls) == 2
    assert calls[0] == dict(device="cpu", dtype="bfloat16", engine="qwen",
                            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    assert calls[1][2] == "Reference words." and calls[1][3]["engine"] == "qwen"


class CountingVoice(FakeVoicePrompt):
    def __init__(self):
        super().__init__()
        self.unloaded = 0

    def unload(self):
        self.unloaded += 1


def test_resource_manager_frees_on_switch_and_for_a_full_dub(project, mini_project):
    mgr = ops.ResourceManager()
    try:
        s1 = Session.import_project(project)
        r1 = mgr.get(s1)
        assert mgr.get(s1) is r1
        v = CountingVoice()
        r1._voice = v
        s2 = Session.load(project)            # another session object → new resources
        r2 = mgr.get(s2)
        assert r2 is not r1 and v.unloaded == 1
        v2 = CountingVoice()
        r2._voice = v2
        gpu.release_idle()                    # what the Dub tab calls before a dub
        assert v2.unloaded == 1
    finally:
        gpu.unregister_releaser(mgr.free)


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU lock
# ═══════════════════════════════════════════════════════════════════════════════

def test_editor_operations_fail_fast_while_a_dub_runs(session, res):
    with gpu.gpu_lock.hold("a full dub"):
        with pytest.raises(gpu.GPUBusy, match="a full dub"):
            _drain(ops.redub(session, ["c1"], res))
    assert not gpu.gpu_lock.busy()


def test_a_dub_waits_for_an_editor_operation():
    order = []
    release = threading.Event()

    def editor():
        with gpu.gpu_lock.hold("an Editor re-dub"):
            order.append("editor")
            release.wait(2)

    t = threading.Thread(target=editor)
    t.start()
    time.sleep(0.05)
    threading.Timer(0.1, release.set).start()
    with gpu.gpu_lock.hold("a full dub", wait=True):
        order.append("dub")
    t.join()
    assert order == ["editor", "dub"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Re-transcribe / re-translate / re-dub
# ═══════════════════════════════════════════════════════════════════════════════

def test_retranscribe_clears_only_the_check_flag(session, res, monkeypatch):
    fake = FakeTranscriber([(0.3, 1.5, "Hello, welcome!")])
    monkeypatch.setattr(transcribe_mod, "transcribe", fake)
    session.set_timing("c1", end=1.8)
    steps = _drain(ops.retranscribe(session, ["c1"], res))
    assert [p.ok for p in steps] == [True] and steps[-1].changed == ["c1"]
    c1 = session.chunk("c1")
    assert c1.source_text == "Hello, welcome!"
    assert c1.stale == {STALE_TRANSLATION, STALE_DUB}
    assert fake.calls[0]["language"] == "en" and fake.calls[0]["skip_resegment"] is True


def test_retranscribe_keeps_going_after_a_failure(session, res, monkeypatch):
    def transcribe(audio_path, output_path, **kw):
        if sf.info(audio_path).duration > 4:     # c2's clip
            raise RuntimeError("CUDA error")
        return FakeTranscriber([(0.3, 1.0, "ok")])(audio_path, output_path, **kw)

    monkeypatch.setattr(transcribe_mod, "transcribe", transcribe)
    for cid in ("c1", "c2", "c3"):
        session.set_timing(cid, start=session.chunk(cid).start + 0.05)
    steps = _drain(ops.retranscribe(session, ["c1", "c2", "c3"], res))
    assert [p.chunk_id for p in steps] == ["c1", "c2", "c3"]
    assert [p.ok for p in steps] == [True, False, True]
    assert "CUDA error" in steps[1].error
    assert STALE_CHECK in session.chunk("c2").stale
    assert STALE_CHECK not in session.chunk("c3").stale
    assert steps[-1].finished and "2/3 done, 1 failed" in steps[-1].message


def test_retranscribe_does_not_blank_a_chunk(session, res, monkeypatch):
    monkeypatch.setattr(transcribe_mod, "transcribe", FakeTranscriber([(0.0, 0.1, "")]))
    (step,) = _drain(ops.retranscribe(session, ["c1"], res))
    assert not step.ok and session.chunk("c1").source_text == "Hello and welcome."


def test_retranslate_uses_context_and_clears_the_flag(session, res, fake_llm):
    fake_llm.responder = lambda m: '[{"index": "2", "text": "Hoy: redes."}]'
    session.set_source_text("c2", "Today: networks.")
    steps = _drain(ops.retranslate(session, ["c2"], res))
    assert steps[-1].ok
    c2 = session.chunk("c2")
    assert c2.target_text == "Hoy: redes." and c2.stale == {STALE_DUB}
    req = fake_llm.requests[0]
    assert req["model"] == "test-llm"
    user = "\n".join(p["text"] for p in req["messages"][1]["content"] if p["type"] == "text")
    assert '1: "Hello and welcome."' in user and '3: "Let\'s begin."' in user
    assert "Keep it formal." in req["messages"][0]["content"]


def test_retranslate_with_a_template_model(project, session, fake_llm):
    _run_info(project, translation=dict(translation_model="translategemma"))
    s = Session.load(project)
    res = ops.Resources(s, device="cpu", llm_client=fake_llm)
    fake_llm.responder = lambda m: "Hola y bienvenidos a todos."
    s.set_source_text("c1", "Hello and welcome, everyone.")
    assert _drain(ops.retranslate(s, ["c1"], res))[-1].ok
    assert fake_llm.requests[0]["model"] == "translategemma"
    assert "English (en) to Spanish (es)" in fake_llm.requests[0]["messages"][0]["content"]
    assert s.chunk("c1").target_text == "Hola y bienvenidos a todos."


def test_retranslate_failure_keeps_the_old_text(session, res, fake_llm):
    fake_llm.responder = lambda m: "not json"
    session.set_source_text("c1", "Changed.")
    (step,) = _drain(ops.retranslate(session, ["c1"], res))
    assert not step.ok
    c1 = session.chunk("c1")
    assert c1.target_text == "Hola y bienvenidos." and STALE_TRANSLATION in c1.stale


def test_redub_writes_versioned_segments(session, res, voice, project):
    session.set_target_text("c1", "uno dos tres")
    old = session.chunk("c1").dub_wav
    steps = _drain(ops.redub(session, ["c1"], res))
    assert steps[-1].ok and steps[-1].changed == ["c1"]
    c1 = session.chunk("c1")
    assert c1.dub_wav == os.path.join("lang", "Spanish", "editor", "segments", "c1_v1.wav")
    assert c1.dub_dur == pytest.approx(0.3) and not c1.stale
    assert voice.calls == [("uno dos tres", "Spanish")]
    assert os.path.isfile(session.abs_path(old))          # the original segment is untouched
    _drain(ops.redub(session, ["c1"], res))
    assert session.chunk("c1").dub_wav.endswith("c1_v2.wav")


def test_redub_partial_failure(session, res, voice):
    orig = voice.synthesize

    def flaky(text, language="English"):
        if "neuronales" in text:
            raise RuntimeError("OOM")
        return orig(text, language)

    voice.synthesize = flaky
    for cid in session.ids():
        session.set_target_text(cid, session.chunk(cid).target_text + " otra")
    steps = _drain(ops.redub(session, session.ids(), res))
    assert [p.ok for p in steps] == [True, False, True]
    assert session.ids(STALE_DUB) == ["c2"]


class BatchVoice(FakeVoicePrompt):
    def __init__(self, fail_batches=False):
        super().__init__()
        self.batches: list[int] = []
        self.fail_batches = fail_batches

    def synthesize_batch(self, items):
        self.batches.append(len(items))
        if self.fail_batches:
            raise RuntimeError("batch too big")
        return [self.synthesize(t, lang) for t, lang in items]


def test_redub_uses_engine_batching(session, fake_llm):
    voice = BatchVoice()
    res = ops.Resources(session, device="cpu", voice_prompt=voice)
    for cid in session.ids():
        session.set_target_text(cid, "a b c")
    steps = _drain(ops.redub(session, session.ids(), res, batch_size=2))
    assert voice.batches == [2] and len(voice.calls) == 3    # 2 batched + 1 alone
    assert all(p.ok for p in steps) and session.ids(STALE_DUB) == []


def test_redub_falls_back_when_a_batch_fails(session):
    voice = BatchVoice(fail_batches=True)
    res = ops.Resources(session, device="cpu", voice_prompt=voice)
    for cid in session.ids():
        session.set_target_text(cid, "a b")
    steps = _drain(ops.redub(session, session.ids(), res, batch_size=3))
    assert voice.batches == [3] and all(p.ok for p in steps)


def test_qwen_wrapper_batches_one_call(monkeypatch):
    class Model:
        def __init__(self):
            self.calls = []

        def generate_voice_clone(self, text, language, voice_clone_prompt):
            self.calls.append((text, language, voice_clone_prompt))
            return [np.zeros(SR * (i + 1), dtype="float32") for i in range(len(text))], SR

    model = Model()
    wrapper = tts._QwenTTSWrapper(model, ["prompt-item"])
    out = wrapper.synthesize_batch([("a", "Spanish"), ("b", "Spanish")])
    assert model.calls == [(["a", "b"], ["Spanish", "Spanish"], ["prompt-item"])]
    assert [len(a) / sr for a, sr in out] == [1.0, 2.0]
    with pytest.raises(ValueError):
        wrapper.synthesize_batch([("a", "Klingon")])


def test_redo_stale_counts_then_runs(session, res):
    session.set_target_text("c1", "x y")
    session.set_target_text("c3", "z")
    assert ops.stale_ids(session, STALE_DUB) == ["c1", "c3"]
    with pytest.raises(ValueError, match="confirm again"):
        ops.redo_stale(session, STALE_DUB, res, expected=5)
    steps = _drain(ops.redo_stale(session, STALE_DUB, res, expected=2))
    assert [p.chunk_id for p in steps] == ["c1", "c3"]
    assert ops.stale_ids(session, STALE_DUB) == []
    assert _drain(ops.redo_stale(session, STALE_DUB, res))[0].message == "Nothing to do"


# ═══════════════════════════════════════════════════════════════════════════════
#  Display subtitles
# ═══════════════════════════════════════════════════════════════════════════════

def test_display_entries_split_long_chunks():
    text = ("Hoy hablamos de redes neuronales, que son modelos inspirados en el cerebro. "
            "Empecemos por lo básico.")
    lines = ops.display_entries([(10.0, 20.0, text)])
    assert len(lines) > 1 and all(len(t) <= 42 for _, _, t in lines)
    assert " ".join(t for _, _, t in lines) == text
    assert lines[0][0] == 10.0 and lines[-1][1] == 20.0
    for (_, e, _), (s, _, _) in zip(lines, lines[1:]):
        assert e == s                                      # contiguous
    words = [len(t.split()) for _, _, t in lines]
    for (s, e, _), w in zip(lines, words):
        assert e - s == pytest.approx(10.0 * w / sum(words), abs=0.01)


def test_display_entries_short_and_unspaced():
    assert ops.display_entries([(0, 1, "Hola.")]) == [(0, 1, "Hola.")]
    assert ops.display_entries([(0, 1, "  ")]) == []
    zh = "我们今天讨论神经网络" * 6
    lines = ops.display_entries([(0.0, 6.0, zh)])
    assert all(len(t) <= 42 for _, _, t in lines) and "".join(t for _, _, t in lines) == zh


# ═══════════════════════════════════════════════════════════════════════════════
#  Assemble
# ═══════════════════════════════════════════════════════════════════════════════

def _read_srt(path):
    from mazinger.srt import parse_file
    return [(e["start"], e["end"], e["text"]) for e in parse_file(path)]


def test_assemble_rebuilds_outputs_and_keeps_backups(session, project):
    with open(project.final_srt, encoding="utf-8") as fh:
        original_srt = fh.read()
    old_audio = sf.read(project.final_audio)[0]
    session.set_target_text("c1", "Hola a todos.")
    session.set_source_text("c3", "Let us begin.")
    assert session.output_stale

    steps = _drain(ops.assemble(session))
    assert steps[-1].finished and steps[-1].message == "Output rebuilt"
    out = steps[-1].outputs
    assert out["audio"] == project.final_audio
    np.testing.assert_array_equal(sf.read(out["previous_audio"])[0], old_audio)
    assert sf.info(project.final_audio).duration == pytest.approx(10.0, abs=0.2)

    assert _read_srt(project.final_srt)[0] == (0.0, 2.0, "Hola a todos.")
    with open(ops.prev_path(project.final_srt), encoding="utf-8") as fh:
        assert fh.read() == original_srt
    assert _read_srt(out["source_srt"])[2] == (7.5, 10.0, "Let us begin.")
    with open(project.source_srt, encoding="utf-8") as fh:
        assert "Let us begin." not in fh.read()          # shared transcription untouched

    assert not session.output_stale and not session.out_of_sync()
    assert not Session.load(project).output_stale
    assert not [n for n in os.listdir(project.tts_dir) if ".tmp" in n]
    assert "2 with an out-of-date dub" in steps[0].message   # c1 target and c3 source edited
    assert ops.previous_outputs(session)["audio"] == ops.prev_path(project.final_audio)


def test_assemble_reports_stale_and_silent_chunks(session, project):
    session.set_target_text("c1", "Cambiado.")
    os.remove(os.path.join(project.tts_segments_dir, "seg_0003.wav"))
    s = Session.load(project)
    first = next(iter(ops.assemble(s)))
    assert "1 with an out-of-date dub" in first.message and "1 without a dub" in first.message


def test_failed_assembly_leaves_outputs_untouched(session, project, monkeypatch):
    from mazinger import assemble as asm
    before = sf.read(project.final_audio)[0]
    session.set_target_text("c1", "Cambiado.")

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(asm, "post_process", boom)
    _run_info(project, assembly=dict(loudness_match=True))
    s = Session.load(project)
    with pytest.raises(RuntimeError, match="disk full"):
        _drain(ops.assemble(s))
    np.testing.assert_array_equal(sf.read(project.final_audio)[0], before)
    assert not os.path.exists(ops.prev_path(project.final_audio))
    assert not [n for n in os.listdir(project.tts_dir) if ".tmp" in n]
    assert s.output_stale and not gpu.gpu_lock.busy()


def test_assemble_mixes_with_the_cached_background(session, project, monkeypatch):
    from mazinger import assemble as asm
    calls = []

    def extract(audio_path, out_path, sr=asm.TARGET_SR):
        calls.append(out_path)
        return write_tone(out_path, 10.0, sr=sr, freq=110, amp=0.05)

    monkeypatch.setattr(asm, "_extract_background", extract)
    _run_info(project, assembly=dict(mix_background=True))
    for _ in range(2):
        _drain(ops.assemble(Session.load(project)))
    assert len(calls) == 1 and os.path.isfile(project.background_audio())


def test_assemble_muxes_video_when_the_run_made_one(session, project):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=10",
         "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "10", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", project.video],
        check=True, capture_output=True,
    )
    _run_info(project, output=dict(output_type="video"))
    steps = _drain(ops.assemble(Session.load(project)))
    assert steps[-1].outputs["video"] == project.final_video
    assert os.path.getsize(project.final_video) > 0
    _drain(ops.assemble(Session.load(project)))
    assert os.path.isfile(ops.prev_path(project.final_video))


# ═══════════════════════════════════════════════════════════════════════════════
#  Original clips
# ═══════════════════════════════════════════════════════════════════════════════

def test_clip_is_cut_once_and_reused(session, project):
    cache = media.clip_cache_for(session)
    path = cache.clip(2.0, 7.0)
    assert path.endswith(os.path.join("editor", "cache", "orig_2000_7000.ogg"))
    info = sf.info(path)
    assert info.channels == 1 and info.duration == pytest.approx(5.0, abs=0.05)
    mtime = os.path.getmtime(path)
    os.utime(path, (mtime - 100, mtime - 100))
    assert cache.clip(2.0, 7.0) == path
    assert os.path.getmtime(path) > mtime - 100            # touched as recently used
    with pytest.raises(ValueError):
        cache.clip(3.0, 3.0)


def test_clip_cache_evicts_least_recently_used(session, project):
    cache = media.clip_cache_for(session)
    paths = [cache.clip(s, s + 1.0) for s in (0.0, 1.0, 2.0)]
    for n, p in enumerate(paths):
        os.utime(p, (1000 + n, 1000 + n))
    os.utime(paths[0], (5000, 5000))                        # recently played
    cache.max_bytes = sum(os.path.getsize(p) for p in paths[:2])
    new = cache.clip(3.0, 4.0)
    left = set(os.listdir(cache.cache_dir))
    assert os.path.basename(new) in left and os.path.basename(paths[0]) in left
    assert os.path.basename(paths[1]) not in left            # oldest went first


def test_neighbours_are_precut(session):
    cache = media.clip_cache_for(session)
    ranges = media.neighbour_ranges(session, "c2")
    assert ranges == [(7.5, 10.0), (0.0, 2.0)]
    cache.prefetch(ranges).result(timeout=30)
    assert all(os.path.isfile(cache.path_for(s, e)) for s, e in ranges)
    cache.close()
