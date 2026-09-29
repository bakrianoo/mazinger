"""Review and translation must not shrink, drop or duplicate content."""

from __future__ import annotations

import json
from types import SimpleNamespace

from mazinger import translate
from mazinger.review import _is_safe_edit
from mazinger.translate import (
    _blocks_to_json_entries,
    _clean_llm_text,
    _parse_translation_response,
    _validate_word_counts,
    estimate_wps,
)

CORE = [("1", 0.0, 2.0, "a"), ("2", 2.0, 4.0, "b"), ("3", 4.0, 6.0, "c")]


# -- Word budget ---------------------------------------------------------------

def _target(blocks, lang):
    wps = estimate_wps(blocks, lang)
    return json.loads(_blocks_to_json_entries(blocks[:1], wps, 0.85))[0]["target_words"]


def test_budget_ignores_source_word_count():
    # Arabic has about half the words of its English translation, and Chinese
    # has no spaces: neither may shrink the English budget of a 5 s line.
    ar = [(str(i), i * 5.0, i * 5.0 + 5, "سنتحدث اليوم عن أهم مفاهيم التعلم الآلي وطرقه") for i in range(1, 11)]
    zh = [(str(i), i * 5.0, i * 5.0 + 5, "今天我们来讨论一下机器学习中最重要的几个概念") for i in range(1, 11)]
    en = round(5 * translate._TTS_WPS["English"] * 0.85)
    assert _target(ar, "English") == en
    assert _target(zh, "English") == en


def test_cjk_targets_are_counted_in_characters():
    block = [("1", 0.0, 5.0, "今天我们来讨论机器学习")]  # 11 characters, no spaces
    over = _validate_word_counts(block, words_per_second=1.0, duration_budget=1.0,
                                 target_language="Chinese (Simplified)")
    assert over and over[0][4] == 11
    assert not _validate_word_counts(block, 4.0, 0.85, target_language="Chinese (Simplified)")


def test_prompt_does_not_ask_to_drop_content():
    prompt = translate._build_system_prompt(["Python"], ["setup"], "English")
    assert "drop minor asides" not in prompt
    assert "HARD MAXIMUM" not in prompt
    assert "keep every point" in prompt
    zh_prompt = translate._build_system_prompt([], [], "Japanese")
    assert "counts characters" in zh_prompt


# -- Parsing -------------------------------------------------------------------

def test_merge_replaces_entries_already_returned_alone():
    raw = '[{"index":"2","text":"B"},{"index":"1-2","text":"AB"},{"index":"3","text":"C"}]'
    assert _parse_translation_response(raw, CORE) == [("1", 0.0, 4.0, "AB"), ("3", 4.0, 6.0, "C")]


def test_overlapping_merge_is_ignored_and_reported():
    missing: list[str] = []
    raw = '[{"index":"1-2","text":"AB"},{"index":"2-3","text":"BC"}]'
    out = _parse_translation_response(raw, CORE, missing)
    assert out[0] == ("1", 0.0, 4.0, "AB")
    assert out[1] == ("3", 4.0, 6.0, "c")  # kept original …
    assert missing == ["3"]               # … and flagged for retry


def test_missing_and_failed_entries_are_reported():
    missing: list[str] = []
    _parse_translation_response('[{"index":"1","text":"A"},{"index":"2","text":""}]', CORE, missing)
    assert sorted(missing) == ["2", "3"]
    missing.clear()
    _parse_translation_response("not json at all", CORE, missing)
    assert missing == ["1", "2", "3"]


def test_leading_number_is_kept_unless_it_echoes_the_index():
    assert _clean_llm_text("3. Install the package.", "7") == "3. Install the package."
    assert _clean_llm_text("2024: the year AI changed") == "2024: the year AI changed"
    assert _clean_llm_text("7. Install the package.", "7") == "Install the package."
    assert _clean_llm_text("2-3: Merged text", "2-3") == "Merged text"


class _Client:
    """Answers each translate call with the next scripted reply."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        msg = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def _srt(blocks):
    from mazinger.srt import blocks_to_text
    return blocks_to_text(blocks)


def test_untranslated_entries_are_retried_once():
    client = _Client([
        '[{"index":"1","text":"A"},{"index":"2","text":"B"}]',  # drops entry 3
        '[{"index":"3","text":"C"}]',
    ])
    out = translate.translate_srt(_srt(CORE), {}, [], client, target_language="French")
    assert [b[3] for b in translate.parse_blocks(out)] == ["A", "B", "C"]
    retry_entries = client.requests[1]["messages"][1]["content"][-1]["text"]
    assert '"index": "3"' in retry_entries and '"index": "1"' not in retry_entries.split("MAIN BLOCK")[-1]


def test_no_repetition_penalties_on_translation():
    client = _Client(['[{"index":"1","text":"A"},{"index":"2","text":"B"},{"index":"3","text":"C"}]'])
    translate.translate_srt(_srt(CORE), {}, [], client, target_language="French")
    assert "repeat_penalty" not in client.requests[0]
    assert "frequency_penalty" not in client.requests[0]


# -- Review --------------------------------------------------------------------

def test_review_accepts_real_corrections():
    assert _is_safe_edit("so lets talk about the importent featurs",
                         "So let's talk about the important features.")
    assert _is_safe_edit("first  we need to instal the dependancies",
                         "First, we need to install the dependencies.")
    assert _is_safe_edit("this is im portant ofthe day", "This is important of the day.")
    assert _is_safe_edit("نثبت المكتبة باستخدام بايثون وهذا الفريم وورك يدعم ريأكت",
                         "نثبت المكتبة باستخدام Python وهذا الـ Framework يدعم React.")
    assert _is_safe_edit("ok", "OK.")


def test_review_rejects_dropped_or_invented_content():
    original = "today we are going to talk about the most important features of this library"
    assert not _is_safe_edit(original, "Today we talk about features.")
    assert not _is_safe_edit(original, original + " and many other great things you will love to learn")
    assert not _is_safe_edit(original, "")
