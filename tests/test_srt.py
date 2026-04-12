"""Tests for mazinger.srt — parsing, formatting, and LLM output cleanup."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mazinger.srt import (
    blocks_to_text,
    build,
    format_time,
    parse_blocks,
    parse_file,
    sanitize,
    time_to_seconds,
)


def test_time_to_seconds_and_format_time_roundtrip():
    samples = [
        "00:00:00,000",
        "00:00:01,500",
        "00:01:02,345",
        "01:00:00,000",
        "10:15:30,999",
    ]
    for ts in samples:
        sec = time_to_seconds(ts)
        assert format_time(sec) == ts


def test_format_time_handles_sub_second_rounding():
    assert format_time(1.2345) == "00:00:01,235"


def test_parse_blocks_parses_multi_line_text():
    srt = """1
00:00:00,000 --> 00:00:02,000
Line one
Line two

2
00:00:02,000 --> 00:00:04,000
Second cue
"""
    blocks = parse_blocks(srt)
    assert len(blocks) == 2
    assert blocks[0] == ("1", 0.0, 2.0, "Line one\nLine two")
    assert blocks[1] == ("2", 2.0, 4.0, "Second cue")


def test_blocks_to_text_roundtrip():
    original = """1
00:00:01,000 --> 00:00:03,500
Hello world
"""
    blocks = parse_blocks(original)
    out = blocks_to_text(blocks)
    assert parse_blocks(out) == blocks


def test_sanitize_strips_llm_tags_and_fences():
    messy = """```srt
1
00:00:00,000 --> 00:00:02,000
<subtitle>Hello</subtitle> there
2
00:00:02,000 --> 00:00:04,000
<TranslatedText>Second line</TranslatedText>
```
"""
    clean = sanitize(messy)
    assert "<subtitle>" not in clean and "</subtitle>" not in clean
    assert "<TranslatedText>" not in clean
    assert "```" not in clean
    assert "Hello" in clean and "Second line" in clean
    blocks = parse_blocks(clean)
    assert len(blocks) == 2


def test_sanitize_empty_returns_newline():
    assert sanitize("") == "\n"


def test_build_no_wrap_short_lines():
    entries = [(0.0, 2.0, "Short"), (2.0, 4.0, "Also short")]
    srt = build(entries, wrap_at=0)
    assert "Short" in srt
    assert parse_blocks(srt)[0][3] == "Short"


def test_build_wraps_long_line():
    long_text = "word " * 20  # well over 42 chars
    srt = build([(0.0, 5.0, long_text)], wrap_at=42)
    lines = [ln for ln in srt.splitlines() if ln and not ln[0].isdigit() and "-->" not in ln]
    assert len(lines) >= 2


def test_parse_file_reads_utf8(tmp_path: Path):
    p = tmp_path / "subs.srt"
    content = """1
00:00:00,000 --> 00:00:01,000
مرحبا بالعالم
"""
    p.write_text(content, encoding="utf-8")
    entries = parse_file(str(p))
    assert len(entries) == 1
    assert entries[0]["idx"] == "1"
    assert entries[0]["start"] == 0.0
    assert entries[0]["end"] == 1.0
    assert "مرحبا" in entries[0]["text"]
