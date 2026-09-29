"""The fit check shortens only overflowing lines, and never makes one worse."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from mazinger import fit, tts
from mazinger.srt import parse_blocks

WPS = 3.0  # fake TTS speaks 3 words per second


class _Voice(tts.TTSWrapper):
    def __init__(self):
        pass

    def unload(self):
        pass

    def synthesize(self, text, language):
        return np.zeros(int(len(text.split()) / WPS * 1000), dtype=np.float32), 1000


class _LLM:
    """Replies with a scripted rewrite per index."""

    def __init__(self, rewrites):
        self.rewrites = rewrites
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        items = json.loads(kwargs["messages"][1]["content"])
        rows = [{"index": it["index"], "text": self.rewrites[it["index"]]}
                for it in items if it["index"] in self.rewrites]
        msg = SimpleNamespace(content=json.dumps(rows))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def _setup(tmp_path, texts, slots):
    entries, t = [], 0.0
    for i, (text, slot) in enumerate(zip(texts, slots), 1):
        entries.append({"idx": str(i), "start": t, "end": t + slot, "text": text})
        t += slot + 0.05
    segs = tts.synthesize_segments(None, _Voice(), entries, str(tmp_path / "segs"))
    return entries, segs, entries[-1]["end"]


def _run(tmp_path, entries, segs, total, llm, **kw):
    srt = tmp_path / "final.srt"
    stats = fit.fit_segments(
        segs, entries, client=llm, llm_model="m", voice_prompt=_Voice(),
        tts_language="English", target_language="English",
        original_duration=total, srt_path=str(srt), **kw,
    )
    return stats, srt


def test_only_overflowing_lines_are_rewritten(tmp_path):
    long = "one two three four five six seven eight nine ten eleven twelve"   # 12 words = 4 s
    entries, segs, total = _setup(tmp_path, ["fits in its slot fine", long], [3.0, 3.0])
    llm = _LLM({"2": "one two three four five six seven eight nine"})         # 9 words = 3 s
    stats, srt = _run(tmp_path, entries, segs, total, llm)

    assert stats["rewritten"] == 1
    sent = json.loads(llm.requests[0]["messages"][1]["content"])
    assert [s["index"] for s in sent] == ["2"]
    assert sent[0]["max_words"] == 9           # 12 words at 1.33x -> 12 * 1.10 / 1.33
    assert segs[1]["actual_dur"] == 3.0
    assert sf.info(segs[1]["wav_path"]).duration == 3.0
    assert parse_blocks(srt.read_text())[1][3] == "one two three four five six seven eight nine"
    assert stats["over_max_fit"] == 0


def test_rewrite_that_drops_too_much_is_rejected(tmp_path):
    long = "one two three four five six seven eight nine ten eleven twelve"
    entries, segs, total = _setup(tmp_path, ["short line here", long], [3.0, 3.0])
    stats, srt = _run(tmp_path, entries, segs, total, _LLM({"2": "one two three"}))
    assert stats["rewritten"] == 0
    assert entries[1]["text"] == long
    assert not srt.exists()                    # nothing changed, nothing written


def test_last_line_uses_time_to_end_of_audio(tmp_path):
    long = "one two three four five six seven eight nine ten eleven twelve"   # 4 s
    entries, segs, _ = _setup(tmp_path, [long], [3.0])
    # The source audio runs 1.5 s past the subtitle: 4 s fits in 4.5 s.
    stats, _ = _run(tmp_path, entries, segs, 4.5, _LLM({}))
    assert stats["rounds"] == 0 and stats["rewritten"] == 0


def test_reports_lines_that_will_still_be_trimmed(tmp_path):
    long = " ".join(["word"] * 30)                                             # 10 s
    entries, segs, total = _setup(tmp_path, ["a short line", long], [3.0, 3.0])
    stats, _ = _run(tmp_path, entries, segs, total, _LLM({}), rounds=1)
    assert stats["over_max_tempo"] == 1 and stats["max_ratio"] > 3
