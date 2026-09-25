"""Editor session model and persistence (EDITOR_PLAN phase 3).

The ``mini_project`` fixture is a finished Spanish dub with three chunks::

    c1  0.0– 2.0  "Hello and welcome."                    "Hola y bienvenidos."
    c2  2.0– 7.0  "Today we talk about neural networks."  "Hoy hablamos de redes neuronales."
    c3  7.5–10.0  "Let's begin."                          "Empecemos."

c2 was merged from two source entries during re-segmentation.
"""

from __future__ import annotations

import json
import os

import pytest

from mazinger.editor import store as store_mod
from mazinger.editor.session import (
    HISTORY_LIMIT, MIN_CHUNK_SEC, STALE_CHECK, STALE_DUB, STALE_TRANSLATION,
    Session, UndoError, default_split_index, join_texts, map_source_to_chunks,
    split_text, text_units,
)
from mazinger.editor.store import SessionStore
from mazinger.runinfo import save_run_info

from .conftest import FINAL_ENTRIES, write_srt, write_tone


@pytest.fixture
def session(mini_project) -> Session:
    return Session.import_project(mini_project)


def _log_lines(proj) -> list[dict]:
    path = os.path.join(proj.editor_dir, "changes.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _state(s: Session) -> dict:
    return s.snapshot()


# ═══════════════════════════════════════════════════════════════════════════════
#  Import
# ═══════════════════════════════════════════════════════════════════════════════

def test_import_builds_chunks_from_the_final_srt(session, mini_project):
    assert session.ids() == ["c1", "c2", "c3"]
    c1, c2, c3 = session.chunks
    assert (c2.start, c2.end) == (2.0, 7.0)
    assert c1.source_text == "Hello and welcome."
    assert c2.source_text == "Today we talk about neural networks."
    assert c2.target_text == "Hoy hablamos de redes neuronales."
    assert c3.dub_wav == os.path.join("lang", "Spanish", "tts", "segments", "seg_0003.wav")
    assert c2.dub_dur == pytest.approx(4.5)
    assert c2.fit_ratio == pytest.approx(0.9)
    assert all(not c.stale for c in session.chunks)
    assert not session.output_stale
    assert session.duration == pytest.approx(10.0, abs=0.05)
    assert [c.status for c in session.chunks] == ["✓ ok"] * 3


def test_import_saves_a_snapshot_and_references_segments_without_copying(session, mini_project):
    assert os.path.isfile(os.path.join(mini_project.editor_dir, "session.json"))
    assert not os.path.exists(os.path.join(mini_project.editor_dir, "segments"))
    assert _log_lines(mini_project) == []


def test_missing_segment_is_marked_for_redub(mini_project):
    os.remove(os.path.join(mini_project.tts_segments_dir, "seg_0002.wav"))
    s = Session.import_project(mini_project)
    c2 = s.chunk("c2")
    assert c2.dub_wav is None and STALE_DUB in c2.stale
    assert c2.status == "⚠ no dub"
    assert s.ids(STALE_DUB) == ["c2"]


def test_import_prefers_the_srt_that_was_translated(mini_project):
    write_srt(mini_project.reviewed_srt, [(0.0, 2.0, "Hello, and welcome!"), (2.0, 7.0, "Reviewed.")])
    assert Session.import_project(mini_project, save=False).chunk("c1").source_text == "Hello, and welcome!"

    other = os.path.join(mini_project.root, "transcription", "custom.srt")
    write_srt(other, [(0.0, 2.0, "From run.json.")])
    save_run_info(mini_project, {"translation_source_srt": "transcription/custom.srt"})
    assert Session.import_project(mini_project, save=False).chunk("c1").source_text == "From run.json."


def test_import_requires_a_finished_dub(mini_project):
    os.remove(mini_project.final_srt)
    with pytest.raises(FileNotFoundError):
        Session.import_project(mini_project)


@pytest.mark.parametrize("sources, chunks, expected", [
    # 1:1
    ([(0, 1, "a"), (1, 2, "b")], [(0, 1), (1, 2)], [["a"], ["b"]]),
    # Several entries per chunk, in order
    ([(0, 1, "a"), (1, 2, "b"), (2, 3, "c")], [(0, 2), (2, 3)], [["a", "b"], ["c"]]),
    # Spanning a boundary: goes to the larger overlap
    ([(0.5, 2.5, "x")], [(0, 1), (1, 3)], [[], ["x"]]),
    ([(0.0, 1.6, "x")], [(0, 1.5), (1.5, 3)], [["x"], []]),
    # In a gap: nearest chunk
    ([(2.1, 2.3, "g")], [(0, 2), (3, 4)], [["g"], []]),
    ([(2.7, 2.9, "g")], [(0, 2), (3, 4)], [[], ["g"]]),
    # Before the first / after the last chunk
    ([(0, 0.5, "pre"), (9, 9.5, "post")], [(1, 2), (3, 4)], [["pre"], ["post"]]),
    # Zero-length entry inside a chunk
    ([(1.5, 1.5, "z")], [(0, 1), (1, 2)], [[], ["z"]]),
    # No chunks
    ([(0, 1, "a")], [], []),
])
def test_map_source_to_chunks(sources, chunks, expected):
    assert map_source_to_chunks(sources, chunks) == expected


# ═══════════════════════════════════════════════════════════════════════════════
#  Text helpers
# ═══════════════════════════════════════════════════════════════════════════════

def test_text_helpers():
    assert split_text("one two three four", 1) == ("one", "two three four")
    assert split_text("one two", 9) == ("one two", "")
    assert split_text("你好世界", 2) == ("你好", "世界")
    assert text_units("สวัสดี") == ["ส", "วั", "ส", "ดี"]  # marks stay on their base
    assert default_split_index("a b c d", 0.5) == 2
    assert join_texts("你好", "世界") == "你好世界"
    assert join_texts("안녕", "세계") == "안녕 세계"
    assert join_texts("Hi", "") == "Hi"


# ═══════════════════════════════════════════════════════════════════════════════
#  Edits and stale rules
# ═══════════════════════════════════════════════════════════════════════════════

def test_source_text_edit_stales_translation_and_dub(session):
    assert session.set_source_text("c2", "  Today: networks.  ") == ["c2"]
    c2 = session.chunk("c2")
    assert c2.source_text == "Today: networks."
    assert c2.stale == {STALE_TRANSLATION, STALE_DUB}
    assert c2.dub_wav is not None  # kept until re-dubbed
    assert c2.badges() == ["⚠ translation", "⚠ dub"]
    assert session.output_stale
    assert session.set_source_text("c2", "Today: networks.") == []  # no-op


def test_target_text_edit_stales_dub(session):
    assert session.set_target_text("c1", "¡Hola!") == ["c1"]
    assert session.chunk("c1").stale == {STALE_DUB}
    assert session.ids(STALE_DUB) == ["c1"]


def test_emptying_the_translation_drops_the_dub(session):
    session.set_target_text("c3", "")
    c3 = session.chunk("c3")
    assert c3.dub_wav is None and not c3.stale and c3.status == "empty"


def test_timing_edit_flags_transcription_and_keeps_dub(session):
    assert session.set_timing("c3", start=7.0) == ["c3"]
    c3 = session.chunk("c3")
    assert (c3.start, c3.end) == (7.0, 10.0)
    assert c3.stale == {STALE_CHECK}
    assert c3.dub_wav is not None
    assert c3.fit_ratio == pytest.approx(2.25 / 3.0)
    assert session.output_stale


def test_timing_is_clamped_to_neighbours_and_minimum(session):
    session.set_timing("c3", start=1.0, end=99.0)       # prev ends 7.0; media ends ~10
    c3 = session.chunk("c3")
    assert c3.start == 7.0 and c3.end == pytest.approx(10.0, abs=0.05)

    session.set_timing("c1", end=0.05)                  # would be shorter than the minimum
    c1 = session.chunk("c1")
    assert (c1.start, c1.end) == (0.0, MIN_CHUNK_SEC)

    session.set_timing("c1", start=5.0)                 # start past the end: end gives room
    c1 = session.chunk("c1")
    assert c1.end - c1.start >= MIN_CHUNK_SEC - 1e-9 and c1.end <= 2.0

    assert session.set_timing("c2") == []               # unchanged


def test_timing_needs_room(session):
    session.set_timing("c3", start=7.0)
    session.set_timing("c2", end=7.0)
    # c2 fills 2.0–7.0 exactly; shrinking c3 to 7.05 leaves c2 no room past 7.05.
    session.set_timing("c3", start=7.1)
    with pytest.raises(KeyError):
        session.set_timing("nope", start=1)


def test_split_creates_two_new_chunks(session):
    new = session.split("c2", 4.5)
    assert len(new) == 2 and "c2" not in session.ids()
    assert session.ids() == ["c1", *new, "c3"]
    a, b = (session.chunk(i) for i in new)
    assert (a.start, a.end, b.start, b.end) == (2.0, 4.5, 4.5, 7.0)
    # Default text split follows the time split: half the words (rounded) each.
    assert a.source_text == "Today we talk" and b.source_text == "about neural networks."
    assert a.target_text == "Hoy hablamos" and b.target_text == "de redes neuronales."
    for p in (a, b):
        assert p.stale == {STALE_DUB} and p.dub_wav is None and p.status == "⚠ no dub"
    assert session.output_stale


def test_split_with_explicit_points(session):
    a, b = session.split("c2", 3.0, src_split_idx=5, tgt_split_idx=2)
    assert session.chunk(a).source_text == "Today we talk about neural"
    assert session.chunk(b).target_text == "de redes neuronales."


@pytest.mark.parametrize("at", [2.0, 2.1, 6.9, 7.0, 8.0])
def test_split_point_must_leave_room(session, at):
    with pytest.raises(ValueError):
        session.split("c2", at)


def test_merge_with_next(session):
    (m,) = session.merge_with_next("c1")
    assert session.ids() == [m, "c3"]
    merged = session.chunk(m)
    assert (merged.start, merged.end) == (0.0, 7.0)
    assert merged.source_text == "Hello and welcome. Today we talk about neural networks."
    assert merged.target_text == "Hola y bienvenidos. Hoy hablamos de redes neuronales."
    assert merged.stale == {STALE_DUB} and merged.dub_wav is None
    with pytest.raises(ValueError):
        session.merge_with_next("c3")


def test_merge_carries_stale_flags(session):
    session.set_source_text("c2", "Changed.")
    (m,) = session.merge_with_next("c2")
    assert session.chunk(m).stale == {STALE_TRANSLATION, STALE_DUB}


def test_ids_are_stable_and_never_reused(session, mini_project):
    a, b = session.split("c2", 4.5)
    (m,) = session.merge_with_next("c1")
    assert "c3" in session.ids()
    assert len({"c1", "c2", "c3", a, b, m}) == 6
    reloaded = Session.load(mini_project)
    new_a, new_b = reloaded.split("c3", 8.5)
    assert not {new_a, new_b} & {"c1", "c2", "c3", a, b, m}


def test_dismiss_clears_a_flag_without_staling_output(session):
    session.set_timing("c3", start=7.2)
    session.mark_output_current()
    assert session.dismiss("c3", STALE_CHECK) == ["c3"]
    assert not session.chunk("c3").stale and not session.output_stale
    assert session.dismiss("c3", STALE_CHECK) == []
    with pytest.raises(ValueError):
        session.dismiss("c3", "bogus")


# ═══════════════════════════════════════════════════════════════════════════════
#  Results of operations
# ═══════════════════════════════════════════════════════════════════════════════

def test_apply_transcription(session):
    session.set_timing("c1", end=1.8)
    session.apply_transcription("c1", "Hello and welcome.")          # same text
    assert session.chunk("c1").stale == set()
    session.set_timing("c1", end=1.6)
    session.apply_transcription("c1", "Hello, welcome.")             # new text
    assert session.chunk("c1").stale == {STALE_TRANSLATION, STALE_DUB}


def test_apply_translation_and_dub(session, mini_project):
    session.set_source_text("c1", "Hi there.")
    session.apply_translation("c1", "Hola.")
    assert session.chunk("c1").stale == {STALE_DUB}
    assert session.ids(STALE_TRANSLATION) == []

    path = session.new_dub_path("c1")
    assert path.endswith(os.path.join("editor", "segments", "c1_v1.wav"))
    write_tone(path, 1.0)
    assert session.new_dub_path("c1").endswith("c1_v2.wav")
    session.apply_dub("c1", path, 1.0)
    c1 = session.chunk("c1")
    assert c1.stale == set() and c1.dub_dur == 1.0
    assert c1.dub_wav == os.path.join("lang", "Spanish", "editor", "segments", "c1_v1.wav")
    assert session.abs_path(c1.dub_wav) == path


def test_counts_and_badges(session):
    session.set_source_text("c1", "Changed.")
    session.set_timing("c3", start=7.0, end=8.0)   # dub 2.25 s in a 1 s slot
    counts = session.counts()
    assert counts == {"chunks": 3, STALE_TRANSLATION: 1, STALE_DUB: 1,
                      STALE_CHECK: 1, "long": 1, "empty": 0}
    assert session.chunk("c3").badges() == ["🔍 check", "⏱ long"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Undo
# ═══════════════════════════════════════════════════════════════════════════════

def test_undo_text_and_timing(session):
    before = session.chunk("c2").to_dict()
    session.set_target_text("c2", "Uno.")
    session.set_timing("c2", end=6.0)
    assert session.undo("c2") == ["c2"]
    assert session.chunk("c2").end == 7.0 and session.chunk("c2").target_text == "Uno."
    session.undo("c2")
    c2 = session.chunk("c2")
    assert c2.state() == {k: before[k] for k in c2.state()}
    assert session.undo("c2") == []


def test_undo_restores_the_previous_dub(session, mini_project):
    old = session.chunk("c1").dub_wav
    path = write_tone(session.new_dub_path("c1"), 1.0)
    session.apply_dub("c1", path, 1.0)
    session.undo("c1")
    assert session.chunk("c1").dub_wav == old


def test_history_is_capped(session):
    for n in range(HISTORY_LIMIT + 5):
        session.set_target_text("c1", f"v{n}")
    assert len(session.chunk("c1").history) == HISTORY_LIMIT
    for _ in range(HISTORY_LIMIT):
        session.undo("c1")
    assert session.chunk("c1").target_text == "v4"
    assert session.undo("c1") == []


def test_undo_split(session):
    original = session.chunk("c2").to_dict()
    a, b = session.split("c2", 4.5)
    assert session.undo(b) == ["c2"]
    assert session.ids() == ["c1", "c2", "c3"]
    assert session.chunk("c2").to_dict() == original


def test_undo_split_is_blocked_while_a_part_has_later_edits(session):
    a, b = session.split("c2", 4.5)
    session.set_target_text(b, "Otra cosa.")
    with pytest.raises(UndoError):
        session.undo(a)
    session.undo(b)            # undo the text edit…
    session.undo(a)            # …then the split goes through
    assert session.ids() == ["c1", "c2", "c3"]


def test_undo_merge(session):
    c1, c2 = session.chunk("c1").to_dict(), session.chunk("c2").to_dict()
    (m,) = session.merge_with_next("c1")
    assert session.undo(m) == ["c1", "c2"]
    assert session.chunk("c1").to_dict() == c1 and session.chunk("c2").to_dict() == c2


def test_undo_after_split_of_a_merge_steps_back_one_at_a_time(session):
    (m,) = session.merge_with_next("c1")
    a, b = session.split(m, 3.0)
    session.undo(a)
    assert session.ids() == [m, "c3"]
    session.undo(m)
    assert session.ids() == ["c1", "c2", "c3"]


def test_repeated_split_merge_keeps_undo_entries_bounded(session):
    target = "c2"
    for _ in range(30):
        a, b = session.split(target, 4.5)
        (target,) = session.merge_with_next(a)
    size = len(json.dumps(session.chunk(target).to_dict()))
    assert size < 20_000
    session.undo(target)                                  # the last merge…
    assert len(session) == 4
    session.undo(session.ids()[1])                        # …and the split before it
    assert len(session) == 3


def test_undo_timing_refuses_to_overlap_a_changed_neighbour(session):
    session.set_timing("c3", start=7.0)
    session.set_timing("c2", end=6.5)
    session.set_timing("c3", start=6.5)
    with pytest.raises(UndoError):
        session.undo("c2")      # c2 would end at 7.0 again, inside c3


# ═══════════════════════════════════════════════════════════════════════════════
#  Persistence
# ═══════════════════════════════════════════════════════════════════════════════

def test_each_edit_appends_one_log_line(session, mini_project):
    session.set_target_text("c1", "A.")
    session.split("c2", 4.5)
    lines = _log_lines(mini_project)
    assert [r["op"] for r in lines] == ["set_target_text", "split"]
    assert [r["rev"] for r in lines] == [1, 2]
    assert lines[1]["remove"] == ["c2"] and len(lines[1]["chunks"]) == 2


def test_reload_replays_the_log_then_merges_it(session, mini_project):
    session.set_source_text("c1", "Hi.")
    session.set_timing("c3", start=7.2)
    a, b = session.split("c2", 4.5)
    session.merge_with_next("c1")
    session.undo("c3")
    expected = _state(session)

    reloaded = Session.load(mini_project)
    assert _state(reloaded) == expected
    assert _log_lines(mini_project) == []           # merged on load
    assert _state(Session.load(mini_project)) == expected


def test_log_is_merged_every_n_changes(session, mini_project, monkeypatch):
    from mazinger.editor import session as session_mod
    monkeypatch.setattr(session_mod, "MERGE_EVERY", 3)
    for n in range(4):
        session.set_target_text("c1", f"v{n}")
    assert len(_log_lines(mini_project)) == 1
    with open(os.path.join(mini_project.editor_dir, "session.json"), encoding="utf-8") as fh:
        assert json.load(fh)["rev"] == 3


def test_torn_last_line_is_ignored(session, mini_project):
    session.set_target_text("c1", "Kept.")
    with open(os.path.join(mini_project.editor_dir, "changes.jsonl"), "a", encoding="utf-8") as fh:
        fh.write('{"rev": 2, "op": "set_tar')
    reloaded = Session.load(mini_project)
    assert reloaded.chunk("c1").target_text == "Kept." and reloaded.rev == 1


def test_crash_between_snapshot_and_log_truncation(session, mini_project):
    session.set_target_text("c1", "One.")
    session.set_target_text("c1", "Two.")
    log_path = os.path.join(mini_project.editor_dir, "changes.jsonl")
    with open(log_path, encoding="utf-8") as fh:
        stale_log = fh.read()
    session.save()
    session.set_target_text("c1", "Three.")
    with open(log_path, encoding="utf-8") as fh:
        new_line = fh.read()
    with open(log_path, "w", encoding="utf-8") as fh:   # log lines already in the snapshot
        fh.write(stale_log + new_line)
    reloaded = Session.load(mini_project)
    assert reloaded.chunk("c1").target_text == "Three." and reloaded.rev == 3
    assert [h["state"]["target_text"] for h in reloaded.chunk("c1").history] == [
        "Hola y bienvenidos.", "One.", "Two."]


def test_mismatched_record_stops_replay(session, mini_project):
    session.set_target_text("c1", "Good.")
    store_mod.SessionStore(mini_project.editor_dir).append(
        {"rev": 2, "op": "split", "at": 0, "remove": ["zz"], "chunks": []})
    reloaded = Session.load(mini_project)
    assert reloaded.ids() == ["c1", "c2", "c3"] and reloaded.chunk("c1").target_text == "Good."


def test_missing_dub_on_load_marks_redub(session, mini_project):
    path = write_tone(session.new_dub_path("c1"), 1.0)
    session.apply_dub("c1", path, 1.0)
    os.remove(path)
    c1 = Session.load(mini_project).chunk("c1")
    assert c1.dub_wav is None and STALE_DUB in c1.stale


def test_newer_session_version_is_refused(session, mini_project):
    path = os.path.join(mini_project.editor_dir, "session.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    data["version"] = store_mod.SESSION_VERSION + 1
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    with pytest.raises(ValueError, match="version"):
        Session.load(mini_project)


def test_open_imports_once_then_loads(mini_project):
    s = Session.open(mini_project)
    s.set_target_text("c1", "Persisted.")
    assert Session.open(mini_project).chunk("c1").target_text == "Persisted."


def test_output_state_survives_reload(session, mini_project):
    session.set_target_text("c1", "X.")
    assert Session.load(mini_project).output_stale
    session = Session.load(mini_project)
    session.mark_output_current()
    assert not Session.load(mini_project).output_stale


def test_out_of_sync_after_a_new_dub(session, mini_project):
    assert not session.out_of_sync()
    write_srt(mini_project.final_srt, FINAL_ENTRIES[:2])
    assert Session.load(mini_project).out_of_sync()


def test_store_without_snapshot(tmp_path):
    store = SessionStore(str(tmp_path / "editor"))
    assert not store.exists()
    with pytest.raises(FileNotFoundError):
        store.load()
