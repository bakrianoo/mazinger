"""A completed dub records its settings and keeps its voice.

The Editor re-does single stages for single segments long after the dub ran.
That only works if the project folder itself holds (a) the settings the run
used — ``lang/<language>/run.json`` — and (b) a voice reference that still
exists and still sounds like the rest of the dub.  These tests run the real
``MazingerDubber.dub`` with every heavy stage replaced by a fake, one test per
way of choosing a voice.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import soundfile as sf

from mazinger import MazingerDubber, profiles
from mazinger.paths import ProjectPaths
from mazinger.runinfo import (
    RUN_INFO_VERSION, load_run_info, resolve_project_path, save_run_info,
)
from mazinger.srt import build

from tests.conftest import (
    FINAL_ENTRIES, TRANSLATED_RAW_ENTRIES, FakeLLMClient, FakeTranscriber,
    FakeVoicePrompt, write_tone,
)

SECRETS = {
    "openai": "sk-TEST-SECRET-openai",
    "hf": "hf_TEST_SECRET_token",
    "deepgram": "dg-TEST-SECRET",
}


# ---------------------------------------------------------------------------
#  Harness: MazingerDubber.dub with fake stages
# ---------------------------------------------------------------------------

@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Patch every heavy stage and return a function that runs a dub."""
    from mazinger import assemble, download, resegment, transcribe, translate, tts

    record: dict = {"voice_prompts": []}

    monkeypatch.setattr(download, "ingest_local_audio",
                        lambda src, dest: write_tone(dest, 10.0))
    monkeypatch.setattr(transcribe, "transcribe", FakeTranscriber())
    monkeypatch.setattr(transcribe, "clear_cache", lambda: None)
    monkeypatch.setattr(translate, "translate_srt",
                        lambda *a, **k: build(TRANSLATED_RAW_ENTRIES))
    monkeypatch.setattr(resegment, "resegment_srt",
                        lambda *a, **k: build(FINAL_ENTRIES))

    monkeypatch.setattr(tts, "load_model", lambda *a, **k: object())

    def _create_voice_prompt(model, ref_audio, ref_text=None, **kwargs):
        record["voice_prompts"].append(
            {"ref_audio": ref_audio, "ref_text": ref_text, **kwargs})
        return FakeVoicePrompt()

    monkeypatch.setattr(tts, "create_voice_prompt", _create_voice_prompt)
    monkeypatch.setattr(tts, "unload_model", lambda *a, **k: None)

    def _assemble(segment_info, duration, out, **kw):
        write_tone(out, duration)
        return out

    monkeypatch.setattr(assemble, "assemble_timeline", _assemble)
    monkeypatch.setattr(assemble, "post_process", lambda src, orig, out, **kw: out)
    monkeypatch.setattr("mazinger.pipeline.get_audio_duration", lambda p: 10.0)
    monkeypatch.setattr(MazingerDubber, "_llm_client", lambda self: FakeLLMClient())

    def _generate_profile(theme, language, output_dir, **kw):
        wav = write_tone(os.path.join(output_dir, "voice.wav"), 5.0)
        script = os.path.join(output_dir, "script.txt")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write("theme reference text")
        return wav, script

    monkeypatch.setattr(profiles, "generate_profile", _generate_profile)
    monkeypatch.setattr(
        profiles, "create_auto_clone_profile",
        lambda audio, srt, out: write_tone(os.path.join(out, "voice.wav"), 25.0))

    source = write_tone(str(tmp_path / "input" / "talk.wav"), 10.0)

    def run(**kwargs) -> ProjectPaths:
        dubber = MazingerDubber(
            openai_api_key=SECRETS["openai"],
            base_dir=str(tmp_path / "out"),
        )
        kwargs.setdefault("target_language", "Spanish")
        return dubber.dub(source, device="cpu", **kwargs)

    run.record = record
    run.tmp_path = tmp_path
    return run


def _run_json(proj: ProjectPaths) -> dict:
    with open(proj.run_info, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
#  run.json contents
# ---------------------------------------------------------------------------

class TestRunRecord:
    def test_written_next_to_the_language_outputs(self, harness):
        proj = harness(voice_theme="narrator-m")
        assert proj.run_info == os.path.join(proj.root, "lang", "Spanish", "run.json")
        assert os.path.isfile(proj.run_info)

    def test_records_the_settings_a_redo_needs(self, harness):
        proj = harness(
            voice_theme="narrator-m",
            transcribe_method="faster-whisper",
            whisper_model="large-v3",
            tts_engine="qwen",
            tempo_mode="fixed", fixed_tempo=1.1, max_tempo=1.3,
            background_volume=0.2,
            words_per_second=2.5,
            translate_technical_terms=True,
            user_instructions="Keep brand names in English.",
        )
        info = _run_json(proj)

        assert info["version"] == RUN_INFO_VERSION
        assert info["target_language"] == "Spanish"
        assert info["transcription"]["method"] == "faster-whisper"
        assert info["transcription"]["model"] == "large-v3"
        assert info["tts"]["engine"] == "qwen"
        assert info["tts"]["model"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        assert info["tts"]["language"] == "Spanish"
        assert info["assembly"] == {
            "tempo_mode": "fixed", "fixed_tempo": 1.1, "max_tempo": 1.3,
            "loudness_match": True, "mix_background": True, "background_volume": 0.2,
        }
        assert info["translation"]["words_per_second"] == 2.5
        assert info["translation"]["translate_technical_terms"] is True
        assert info["translation"]["user_instructions"] == "Keep brand names in English."
        assert info["segmentation"]["segment_mode"] == "short"
        assert info["llm"]["model"] == "gpt-4.1"
        assert info["detected_source_language"] == "en"

    def test_records_the_srt_that_was_translated(self, harness):
        proj = harness(voice_theme="narrator-m")
        info = _run_json(proj)
        # Default pipeline translates the raw ASR output.
        assert info["translation_source_srt"] == os.path.join("transcription", "source.raw.srt")

    def test_engine_specific_model_is_recorded(self, harness):
        proj = harness(voice_theme="narrator-m", tts_engine="omnivoice")
        from mazinger.tts import DEFAULT_OMNIVOICE_MODEL
        assert _run_json(proj)["tts"]["model"] == DEFAULT_OMNIVOICE_MODEL

    def test_paths_are_relative_to_the_project(self, harness):
        proj = harness(voice_theme="narrator-m")
        info = _run_json(proj)
        for path in (info["translation_source_srt"], info["voice"]["sample"]):
            assert not os.path.isabs(path)
            assert os.path.isfile(resolve_project_path(proj, path))

    def test_never_contains_credentials(self, harness, tmp_path):
        cookies = tmp_path / "cookies.txt"
        cookies.write_text("# Netscape HTTP Cookie File\n")
        proj = harness(
            voice_theme="narrator-m",
            hf_token=SECRETS["hf"],
            deepgram_api_key=SECRETS["deepgram"],
            cookies=str(cookies),
        )
        text = open(proj.run_info, encoding="utf-8").read()
        for secret in SECRETS.values():
            assert secret not in text
        assert "cookies.txt" not in text

    def test_not_written_when_the_dub_fails(self, harness, monkeypatch):
        from mazinger import assemble

        def _boom(*a, **k):
            raise RuntimeError("assembly failed")

        monkeypatch.setattr(assemble, "assemble_timeline", _boom)
        with pytest.raises(RuntimeError):
            harness(voice_theme="narrator-m")
        proj = ProjectPaths("talk", base_dir=str(harness.tmp_path / "out"),
                            target_language="Spanish")
        assert os.path.isdir(proj.tts_segments_dir), "dub did not reach TTS — wrong project?"
        assert os.listdir(proj.tts_segments_dir)
        assert not os.path.exists(proj.run_info)


# ---------------------------------------------------------------------------
#  Voice kinds
# ---------------------------------------------------------------------------

class TestVoiceIsKept:
    def test_supplied_sample_is_copied_into_the_project(self, harness):
        upload_dir = harness.tmp_path / "gradio-tmp"
        sample = write_tone(str(upload_dir / "my voice.m4a"), 6.0)

        proj = harness(voice_sample=sample, voice_script="  Hello, this is me.  ")
        voice = _run_json(proj)["voice"]

        assert voice["kind"] == "sample"
        assert voice["sample"] == os.path.join("lang", "Spanish", "voice_profile", "reference", "voice.m4a")
        kept = resolve_project_path(proj, voice["sample"])
        assert open(resolve_project_path(proj, voice["script"]), encoding="utf-8").read() == "Hello, this is me."

        # The dub itself was voiced from the kept copy, so the Editor's
        # later re-dubs use exactly the same reference.
        assert harness.record["voice_prompts"][0]["ref_audio"] == kept

        # The upload can disappear without breaking anything.
        os.remove(sample)
        assert os.path.isfile(kept)

    def test_kept_sample_does_not_shadow_the_theme_cache(self, harness):
        """voice_profile/voice.wav is reused by themes and auto-clone; a
        supplied sample must never be written there."""
        sample = write_tone(str(harness.tmp_path / "s.wav"), 6.0)
        proj = harness(voice_sample=sample, voice_script="hi")
        assert not os.path.exists(os.path.join(proj.voice_profile_dir, "voice.wav"))

    def test_sample_given_as_a_script_file(self, harness):
        sample = write_tone(str(harness.tmp_path / "s.wav"), 6.0)
        script = harness.tmp_path / "s.txt"
        script.write_text("From a file.\n", encoding="utf-8")
        proj = harness(voice_sample=sample, voice_script=str(script))
        voice = _run_json(proj)["voice"]
        assert open(resolve_project_path(proj, voice["script"]), encoding="utf-8").read() == "From a file."

    def test_qwen_theme(self, harness):
        proj = harness(voice_theme="narrator-m")
        voice = _run_json(proj)["voice"]
        assert voice["kind"] == "theme"
        assert voice["theme"] == "narrator-m"
        assert voice["sample"] == os.path.join("lang", "Spanish", "voice_profile", "voice.wav")
        assert voice["script"] == os.path.join("lang", "Spanish", "voice_profile", "script.txt")
        assert voice["instruct"] is None

    def test_omnivoice_theme_keeps_its_instruct(self, harness):
        proj = harness(voice_theme="narrator-m", tts_engine="omnivoice")
        voice = _run_json(proj)["voice"]
        assert voice["kind"] == "theme-instruct"
        assert voice["instruct"]
        assert voice["sample"] is None
        with open(proj.voice_instruct, encoding="utf-8") as fh:
            assert fh.read() == voice["instruct"]
        assert harness.record["voice_prompts"][0]["voice_design_instruct"] == voice["instruct"]

    def test_qwen_auto_clone(self, harness):
        proj = harness()
        voice = _run_json(proj)["voice"]
        assert voice["kind"] == "auto-clone"
        assert voice["sample"] == os.path.join("lang", "Spanish", "voice_profile", "voice.wav")
        assert voice["script"] is None  # Qwen auto-clone runs x-vector only

    def test_omnivoice_auto_voice_promotes_a_segment(self, harness):
        proj = harness(tts_engine="omnivoice")
        voice = _run_json(proj)["voice"]

        assert voice["kind"] == "omnivoice-auto"
        assert voice["sample"] == os.path.join("lang", "Spanish", "voice_profile", "omnivoice_auto", "voice.wav")
        script = open(resolve_project_path(proj, voice["script"]), encoding="utf-8").read()
        assert script in {text for _, _, text in FINAL_ENTRIES}
        # Must not land where a later Qwen auto-clone would pick it up as
        # the *original speaker's* voice.
        assert not os.path.exists(os.path.join(proj.voice_profile_dir, "voice.wav"))


# ---------------------------------------------------------------------------
#  runinfo module
# ---------------------------------------------------------------------------

class TestRunInfoModule:
    def test_round_trip(self, mini_project):
        save_run_info(mini_project, {"tts": {"engine": "qwen"}})
        info = load_run_info(mini_project)
        assert info["tts"] == {"engine": "qwen"}
        assert info["version"] == RUN_INFO_VERSION
        assert "created_at" in info and "mazinger_version" in info

    def test_missing_record_is_none(self, mini_project):
        assert load_run_info(mini_project) is None

    @pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
    def test_unreadable_record_is_none(self, mini_project, content):
        with open(mini_project.run_info, "w", encoding="utf-8") as fh:
            fh.write(content)
        assert load_run_info(mini_project) is None

    def test_newer_record_is_still_returned(self, mini_project, caplog):
        with open(mini_project.run_info, "w", encoding="utf-8") as fh:
            json.dump({"version": RUN_INFO_VERSION + 1, "tts": {}}, fh)
        assert load_run_info(mini_project)["tts"] == {}
        assert "understands up to" in caplog.text

    @pytest.mark.parametrize("key", ["api_key", "hf_token", "openai_api_key", "cookies"])
    def test_refuses_credential_keys(self, mini_project, key):
        with pytest.raises(ValueError, match=key):
            save_run_info(mini_project, {"llm": {key: "x"}})
        assert not os.path.exists(mini_project.run_info)

    def test_write_is_atomic(self, mini_project):
        save_run_info(mini_project, {"a": 1})
        assert not os.path.exists(mini_project.run_info + ".tmp")


# ---------------------------------------------------------------------------
#  Voice helpers
# ---------------------------------------------------------------------------

class TestKeepVoiceReference:
    def test_replaces_a_previous_reference(self, tmp_path):
        out = str(tmp_path / "ref")
        profiles.keep_voice_reference(write_tone(str(tmp_path / "a.mp3"), 1.0), "one", out)
        profiles.keep_voice_reference(write_tone(str(tmp_path / "b.wav"), 1.0), None, out)
        assert sorted(os.listdir(out)) == ["voice.wav"]

    def test_keeping_the_kept_copy_is_a_no_op(self, tmp_path):
        out = str(tmp_path / "ref")
        kept, _ = profiles.keep_voice_reference(write_tone(str(tmp_path / "a.wav"), 1.0), "x", out)
        again, _ = profiles.keep_voice_reference(kept, "x", out)
        assert again == kept and os.path.isfile(kept)


class TestSelectReferenceSegment:
    def _seg(self, tmp_path, idx, seconds, amp=0.3):
        path = write_tone(str(tmp_path / f"seg_{idx}.wav"), seconds, amp=amp)
        return {"idx": str(idx), "wav_path": path, "actual_dur": seconds}

    def _entries(self, n):
        return [{"idx": str(i), "text": f"text {i}"} for i in range(1, n + 1)]

    def test_picks_the_loudest_in_range(self, tmp_path):
        segs = [
            self._seg(tmp_path, 1, 1.0, amp=0.9),   # too short
            self._seg(tmp_path, 2, 5.0, amp=0.2),
            self._seg(tmp_path, 3, 6.0, amp=0.5),   # winner
        ]
        out = tmp_path / "out"
        voice, script = profiles.select_reference_segment(segs, self._entries(3), str(out))
        assert open(script, encoding="utf-8").read() == "text 3"
        assert sf.info(voice).duration == pytest.approx(6.0)

    def test_falls_back_to_the_closest_duration(self, tmp_path):
        segs = [self._seg(tmp_path, 1, 1.0), self._seg(tmp_path, 2, 2.5)]
        _, script = profiles.select_reference_segment(segs, self._entries(2), str(tmp_path / "o"))
        assert open(script, encoding="utf-8").read() == "text 2"

    def test_ignores_silence(self, tmp_path):
        silent = str(tmp_path / "silent.wav")
        sf.write(silent, np.zeros(24_000 * 5, dtype="float32"), 24_000)
        segs = [{"idx": "1", "wav_path": silent, "actual_dur": 5.0}]
        assert profiles.select_reference_segment(segs, self._entries(1), str(tmp_path / "o")) is None

    def test_ignores_segments_without_audio_or_text(self, tmp_path):
        segs = [
            {"idx": "1", "wav_path": None, "actual_dur": 0},
            self._seg(tmp_path, 2, 5.0),
        ]
        entries = [{"idx": "1", "text": "a"}, {"idx": "2", "text": "  "}]
        assert profiles.select_reference_segment(segs, entries, str(tmp_path / "o")) is None
