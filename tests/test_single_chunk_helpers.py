"""Single-chunk helpers used by the Editor: translate_chunk, synthesize_one,
transcribe_clip (EDITOR_PLAN 2.1–2.3)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import soundfile as sf

from mazinger import transcribe as transcribe_mod
from mazinger import translate, tts

from .conftest import SR, FakeLLMClient, FakeTranscriber, FakeVoicePrompt, write_tone


# ---------------------------------------------------------------------------
#  translate_chunk
# ---------------------------------------------------------------------------

def _user_text(request: dict) -> str:
    parts = request["messages"][1]["content"]
    return "\n".join(p["text"] for p in parts if p.get("type") == "text")


def _main_block(request: dict) -> list[dict]:
    text = _user_text(request)
    main = text.split("== MAIN BLOCK (translate these entries) ==\n", 1)[1]
    main = main.split("\n\n== CONTEXT AFTER", 1)[0]
    return json.loads(main)


def _chunk(client, text="Today we talk about neural networks.", **kw):
    args = dict(
        prev_ctx=["Hello and welcome."],
        next_ctx=["Let's begin.", "First, the basics."],
        duration=5.0,
        description={"keywords": ["neural networks"], "keypoints": ["intro to ML"]},
        client=client,
        llm_model="test-llm",
        source_language="English",
        target_language="Spanish",
        words_per_second=3.0,
    )
    args.update(kw)
    return translate.translate_chunk(text, **args)


def test_translate_chunk_returns_translation_for_the_main_entry():
    client = FakeLLMClient(lambda m: '[{"index": "2", "text": "Hoy hablamos de redes neuronales."}]')
    assert _chunk(client) == "Hoy hablamos de redes neuronales."
    assert client.requests[0]["model"] == "test-llm"


def test_translate_chunk_sends_neighbours_as_context_only():
    client = FakeLLMClient(lambda m: '[{"index": "2", "text": "Hoy hablamos."}]')
    _chunk(client)
    text = _user_text(client.requests[0])
    before = text.split("== CONTEXT BEFORE (do NOT translate, for reference only) ==\n", 1)[1]
    assert before.startswith('1: "Hello and welcome."')
    after = text.split("== CONTEXT AFTER (do NOT translate, for reference only) ==\n", 1)[1]
    assert after == '3: "Let\'s begin."\n4: "First, the basics."'
    assert [e["index"] for e in _main_block(client.requests[0])] == ["2"]


def test_translate_chunk_uses_the_translate_srt_word_budget():
    client = FakeLLMClient(lambda m: '[{"index": "2", "text": "x"}]')
    _chunk(client, duration=5.0, words_per_second=3.0, duration_budget=0.8)
    assert _main_block(client.requests[0])[0]["target_words"] == max(
        translate.MIN_TARGET_WORDS, round(5.0 * 3.0 * 0.8))

    # Short chunks get the same minimum as a full translation.
    client = FakeLLMClient(lambda m: '[{"index": "2", "text": "x"}]')
    _chunk(client, duration=0.5, words_per_second=3.0)
    assert _main_block(client.requests[0])[0]["target_words"] == translate.MIN_TARGET_WORDS


def test_translate_chunk_estimates_wps_when_unset():
    client = FakeLLMClient(lambda m: '[{"index": "1", "text": "x"}]')
    text = "one two three four five six seven eight nine ten"
    translate.translate_chunk(
        text, duration=5.0, client=client, target_language="English",
    )
    wps = translate.estimate_wps([("1", 0.0, 5.0, text)], "English")
    expected = max(translate.MIN_TARGET_WORDS, round(5.0 * wps * translate.DURATION_BUDGET))
    assert _main_block(client.requests[0])[0]["target_words"] == expected


def test_translate_chunk_carries_prompt_settings():
    client = FakeLLMClient(lambda m: '[{"index": "2", "text": "x"}]')
    _chunk(client, user_instructions="Use formal register.", translate_technical_terms=True)
    system = client.requests[0]["messages"][0]["content"]
    assert "Use formal register." in system
    assert "professional, widely-accepted Spanish equivalents" in system
    assert "The source subtitles are in English." in system


def test_translate_chunk_accepts_a_renumbered_single_answer():
    client = FakeLLMClient(lambda m: '```json\n[{"index": "1", "text": "Hoy hablamos."}]\n```')
    assert _chunk(client) == "Hoy hablamos."


def test_translate_chunk_accepts_a_merge_with_context():
    client = FakeLLMClient(lambda m: '[{"index": "2-3", "text": "Hoy hablamos. Empecemos."}]')
    assert _chunk(client) == "Hoy hablamos. Empecemos."


@pytest.mark.parametrize("reply", ["", "[]", '[{"index": "2", "text": ""}]', "I cannot help."])
def test_translate_chunk_never_passes_off_the_source_as_a_translation(reply):
    client = FakeLLMClient(lambda m: reply)
    with pytest.raises(ValueError, match="Could not parse"):
        _chunk(client)


def test_translate_chunk_rejects_empty_source():
    with pytest.raises(ValueError, match="empty"):
        _chunk(FakeLLMClient(), text="   ")


def test_translate_chunk_attaches_thumbnails_in_range(tmp_path):
    from PIL import Image

    thumbs = []
    for sec in (1.0, 12.0, 30.0):
        path = str(tmp_path / f"t{sec}.jpg")
        Image.new("RGB", (4, 4)).save(path)
        thumbs.append({"seconds": sec, "timestamp": f"00:{int(sec):02d}", "reason": "r", "path": path})
    client = FakeLLMClient(lambda m: '[{"index": "2", "text": "x"}]')
    _chunk(client, thumb_paths=thumbs, start=10.0, duration=5.0)
    parts = client.requests[0]["messages"][1]["content"]
    assert [p for p in parts if p["type"] == "image_url"].__len__() == 1
    assert any("00:12" in p.get("text", "") for p in parts)


def test_translate_text_simple_cleans_the_reply():
    client = FakeLLMClient(lambda m: "Translation: Hola mundo")
    assert translate.translate_text_simple(
        "Hello world", client, source_language="English", target_language="Spanish",
    ) == "Hola mundo"
    assert "Hello world" in client.requests[0]["messages"][0]["content"]


def test_translate_srt_simple_still_keeps_the_original_on_failure():
    def boom(_):
        raise RuntimeError("down")

    srt = "1\n00:00:00,000 --> 00:00:02,000\nHello.\n"
    out = translate.translate_srt_simple(srt, FakeLLMClient(boom), target_language="Spanish")
    assert "Hello." in out


# ---------------------------------------------------------------------------
#  synthesize_one / synthesize_segments
# ---------------------------------------------------------------------------

def test_synthesize_one_writes_and_reports_duration(tmp_path):
    prompt = FakeVoicePrompt()
    out = str(tmp_path / "new" / "dir" / "c1_v1.wav")
    path, dur = tts.synthesize_one(prompt, "  one two three  ", out, "Spanish")
    assert path == out
    assert dur == pytest.approx(0.3)
    assert sf.info(out).duration == pytest.approx(dur)
    assert prompt.calls == [("one two three", "Spanish")]
    assert os.listdir(os.path.dirname(out)) == ["c1_v1.wav"]


def test_synthesize_one_overwrites_an_existing_file(tmp_path):
    out = write_tone(str(tmp_path / "seg.wav"), 3.0)
    _, dur = tts.synthesize_one(FakeVoicePrompt(), "two words", out)
    assert sf.info(out).duration == pytest.approx(dur) == pytest.approx(0.2)


def test_synthesize_one_rejects_empty_text(tmp_path):
    with pytest.raises(ValueError):
        tts.synthesize_one(FakeVoicePrompt(), "  ", str(tmp_path / "x.wav"))
    assert not os.listdir(tmp_path)


def test_synthesize_one_supports_legacy_qwen_prompts(tmp_path):
    class LegacyModel:
        def generate_voice_clone(self, text, language, voice_clone_prompt):
            assert voice_clone_prompt == "legacy-prompt"
            return [np.zeros(SR // 2, dtype="float32")], SR

    out = str(tmp_path / "x.wav")
    assert tts.synthesize_one("legacy-prompt", "hi", out, model=LegacyModel())[1] == 0.5
    with pytest.raises(ValueError, match="model"):
        tts.synthesize_one("legacy-prompt", "hi", out)


def test_synthesize_segments_behaviour_is_unchanged(tmp_path):
    seg_dir = str(tmp_path / "segments")
    cached = write_tone(os.path.join(seg_dir, "seg_0001.wav"), 1.5)
    entries = [
        {"idx": "1", "start": 0.0, "end": 2.0, "text": "cached already"},
        {"idx": "2", "start": 2.0, "end": 4.0, "text": "one two three four"},
        {"idx": "3", "start": 4.0, "end": 5.0, "text": "   "},
    ]
    prompt = FakeVoicePrompt()
    info = tts.synthesize_segments(None, prompt, entries, seg_dir, language="Spanish")

    assert prompt.calls == [("one two three four", "Spanish")]
    assert info[0] == {"idx": "1", "start": 0.0, "end": 2.0, "target_dur": 2.0,
                       "wav_path": cached, "actual_dur": pytest.approx(1.5), "_skipped": True}
    assert info[1]["wav_path"] == os.path.join(seg_dir, "seg_0002.wav")
    assert info[1]["actual_dur"] == pytest.approx(0.4)
    assert info[2]["wav_path"] is None and info[2]["actual_dur"] == 0
    assert sorted(os.listdir(seg_dir)) == ["seg_0001.wav", "seg_0002.wav"]

    # force_reset re-synthesises everything.
    prompt = FakeVoicePrompt()
    tts.synthesize_segments(None, prompt, entries, seg_dir, force_reset=True)
    assert len(prompt.calls) == 2


# ---------------------------------------------------------------------------
#  transcribe_clip
# ---------------------------------------------------------------------------

class ClipTranscriber(FakeTranscriber):
    """Records the clip it was given before answering."""

    def __call__(self, audio_path, output_path, **kwargs):
        self.clip_duration = sf.info(audio_path).duration
        return super().__call__(audio_path, output_path, **kwargs)


def test_transcribe_clip_cuts_padded_range_and_joins_text(tmp_path, monkeypatch):
    audio = write_tone(str(tmp_path / "audio.mp3"), 10.0)
    # Clip-relative times: the clip for [4, 6] with pad 0.3 spans 3.7–6.3.
    fake = ClipTranscriber([
        (0.0, 0.25, "tail of previous"),   # midpoint inside the leading pad
        (0.3, 1.2, "Hello"),
        (1.2, 2.3, "world."),
        (2.35, 2.6, "next one"),           # midpoint inside the trailing pad
    ])
    monkeypatch.setattr(transcribe_mod, "transcribe", fake)

    text = transcribe_mod.transcribe_clip(
        audio, 4.0, 6.0, method="faster-whisper", language="en", skip_resegment=False,
    )
    assert text == "Hello world."
    assert fake.clip_duration == pytest.approx(2.6, abs=0.01)
    call = fake.calls[0]
    assert call["skip_resegment"] is True
    assert call["method"] == "faster-whisper" and call["language"] == "en"
    assert not os.path.exists(call["audio_path"])  # temp files cleaned up


def test_transcribe_clip_clamps_padding_at_the_start(tmp_path, monkeypatch):
    audio = write_tone(str(tmp_path / "audio.wav"), 5.0)
    fake = ClipTranscriber([(0.0, 1.0, "Start.")])
    monkeypatch.setattr(transcribe_mod, "transcribe", fake)
    assert transcribe_mod.transcribe_clip(audio, 0.1, 1.0) == "Start."
    assert fake.clip_duration == pytest.approx(1.3, abs=0.01)


def test_transcribe_clip_returns_empty_for_silence(tmp_path, monkeypatch):
    audio = write_tone(str(tmp_path / "audio.wav"), 5.0)
    monkeypatch.setattr(transcribe_mod, "transcribe", ClipTranscriber([(0.0, 0.1, "")]))
    assert transcribe_mod.transcribe_clip(audio, 1.0, 2.0) == ""


def test_transcribe_clip_validates_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        transcribe_mod.transcribe_clip(str(tmp_path / "missing.mp3"), 0, 1)
    audio = write_tone(str(tmp_path / "audio.wav"), 1.0)
    with pytest.raises(ValueError):
        transcribe_mod.transcribe_clip(audio, 2.0, 2.0)
