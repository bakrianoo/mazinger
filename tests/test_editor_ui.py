"""Editor tab logic (EDITOR_PLAN phase 5), driven through its handlers.

Handlers return ``{component_name: value}`` dicts, so they run here without
a browser.  Chunks of ``mini_project``: c1 0–2 s, c2 2–7 s, c3 7.5–10 s.
"""

from __future__ import annotations

import json
import os

import pytest

gr = pytest.importorskip("gradio")
pd = pytest.importorskip("pandas")

from mazinger.editor import ops  # noqa: E402
from mazinger.editor.session import STALE_CHECK, STALE_DUB, STALE_TRANSLATION  # noqa: E402
from mazinger.runinfo import load_run_info  # noqa: E402
from mazinger.studio import editor_ui as ui  # noqa: E402

from .conftest import FakeLLMClient, FakeVoicePrompt, write_tone  # noqa: E402
from .test_editor_ops import _run_info  # noqa: E402


@pytest.fixture(autouse=True)
def clean_registry():
    ui._SESSIONS.clear()
    ui._ERRORS.clear()
    ui._CLIPS.clear()
    yield
    ui._RESOURCES.free()
    for cache in ui._CLIPS.values():
        cache.close()


@pytest.fixture
def project(mini_project):
    write_tone(os.path.join(mini_project.voice_reference_dir, "voice.wav"), 5.0)
    _run_info(mini_project)
    return mini_project


@pytest.fixture
def key(project):
    return ui.key_for(project)


@pytest.fixture
def view(key):
    return ui.h_open(key, ui._empty_view())["view"]


@pytest.fixture
def fakes(monkeypatch):
    voice, llm = FakeVoicePrompt(), FakeLLMClient()

    def resources(session, api_key, dub_api_key):
        res = ui._RESOURCES.get(session)
        res._voice, res._llm, res.device = voice, llm, "cpu"
        res.set_api_key = lambda k: None
        return res

    monkeypatch.setattr(ui, "_resources", resources)
    return voice, llm


def _df(styler) -> pd.DataFrame:
    return styler.data


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure helpers
# ═══════════════════════════════════════════════════════════════════════════════

def test_project_keys(project, key):
    assert key == os.path.abspath(os.path.join(project.root, "lang", "Spanish"))
    p = ui.paths_for(key)
    assert (p.slug, p.target_language, os.path.abspath(p.root)) == ("demo", "Spanish", os.path.abspath(project.root))
    assert ui.key_from_output(project.final_audio) == key
    assert ui.key_from_output(project.video) is None


def test_scan_projects(project, tmp_path):
    (found,) = ui.scan_projects(str(tmp_path))
    assert found["key"] == ui.key_for(project) and found["chunks"] == 3
    assert not found["has_session"] and found["has_run_info"]
    ui.get_session(found["key"])
    label = ui.project_label(ui.scan_projects(str(tmp_path))[0])
    assert label.startswith("demo · Spanish · ") and "3 chunks" in label and "edited" in label


@pytest.mark.parametrize("text, seconds", [
    ("75", 75.0), ("1:15", 75.0), ("1:15.5", 75.5), ("1:02:05.25", 3725.25), ("0,5", 0.5), (12.5, 12.5),
])
def test_parse_time(text, seconds):
    assert ui.parse_time(text) == pytest.approx(seconds)


def test_parse_time_rejects_junk():
    for bad in ("", "abc", "1:2:3:4", "-5"):
        with pytest.raises(ValueError):
            ui.parse_time(bad)


def test_fmt_time_round_trips():
    for t in (0.0, 5.25, 75.5, 3725.25):
        assert ui.parse_time(ui.fmt_time(t)) == pytest.approx(t, abs=0.01)
    assert ui.fmt_time(75.25) == "1:15.25" and ui.fmt_time(3725.5) == "1:02:05.50"


def test_filter_and_pages(key, view):
    session = ui.get_session(key)
    assert ui.filter_indices(session, ui.FILTER_ALL, "", {}) == [0, 1, 2]
    session.set_target_text("c2", "Cambiado.")
    assert ui.filter_indices(session, ui.FILTER_NEEDS_WORK, "", {}) == [1]
    assert ui.filter_indices(session, ui.FILTER_ALL, "NEURAL", {}) == [1]
    assert ui.filter_indices(session, ui.FILTER_ERRORS, "", {"c3": "boom"}) == [2]
    session.set_timing("c3", 7.5, 8.0)                   # 2.25 s dub in 0.5 s
    assert ui.filter_indices(session, ui.FILTER_LONG, "", {}) == [2]
    assert ui.page_count(0) == 1 and ui.page_count(50) == 1 and ui.page_count(51) == 2
    assert ui.page_of_position([0, 4, 9], 4) == 0 and ui.page_of_position([0, 4], 3) is None


def test_table_page_shows_only_its_rows(monkeypatch, key, view):
    monkeypatch.setattr(ui, "PAGE_SIZE", 2)
    session = ui.get_session(key)
    session.set_target_text("c3", "Cambiado.")
    styler, ids, page = ui.table_page(session, [0, 1, 2], 1, {}, selected="c3")
    assert ids == ["c3"] and page == 1
    df = _df(styler)
    assert list(df.columns) == ui.COLUMNS and len(df) == 1
    assert df.iloc[0]["#"] == 3 and df.iloc[0]["Status"] == "⚠ dub"
    assert df.iloc[0]["Fit"] == "90%"
    html = styler.to_html()
    assert "rgba(244, 162, 97" in html and "outline" in html      # stale + selected
    _, ids, page = ui.table_page(session, [0, 1, 2], 99, {})
    assert page == 1                                              # clamped


def test_summary_and_bulk_labels(key, view):
    session = ui.get_session(key)
    assert ui.summary_text(session) == "**3** chunks · output up to date"
    session.set_source_text("c1", "Hi.")
    s = ui.summary_text(session)
    assert "⚠ 1 to re-translate" in s and "⚠ 1 to re-dub" in s and "**output out of date**" in s
    assert ui.bulk_label(session, STALE_DUB) == ("Re-dub 1 stale", True)
    assert ui.bulk_label(session, STALE_CHECK) == ("Re-transcribe stale", False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Handlers
# ═══════════════════════════════════════════════════════════════════════════════

def test_open_selects_the_first_row(key, project):
    out = ui.h_open(key, ui._empty_view())
    assert out["view"]["key"] == key and out["view"]["selected"] == "c1"
    assert out["view"]["page_ids"] == ["c1", "c2", "c3"]
    assert out["src"] == "Hello and welcome." and out["start"] == "0:00.00"
    assert out["orig_audio"].endswith("orig_0_2000.ogg") and os.path.isfile(out["orig_audio"])
    assert out["dub_audio"].endswith("seg_0001.wav")
    assert out["settings_group"]["visible"] is False and out["status"] == ""
    assert "3 rows" in out["page_md"]


def test_open_without_run_json_shows_the_settings_form(mini_project):
    out = ui.h_open(ui.key_for(mini_project), ui._empty_view())
    assert out["settings_group"]["visible"] is True
    assert "no record of its dub settings" in out["status"]


def test_open_reports_a_broken_project(tmp_path):
    out = ui.h_open(str(tmp_path / "projects" / "x" / "lang" / "Spanish"), ui._empty_view())
    assert out["status"].startswith("❌ Could not open")


def test_select_step_and_jump(view):
    out = ui.h_select_row(view, 2)
    assert out["view"]["selected"] == "c3" and out["tgt"] == "Empecemos."
    assert ui.h_select_row(view, 9) == {}
    v = out["view"]
    assert ui.h_step(v, -1)["view"]["selected"] == "c2"
    assert ui.h_step(v, +1) == {}                                 # already last
    assert ui.h_jump(view, "0:05")["view"]["selected"] == "c2"
    assert ui.h_jump(view, "0:07.2")["view"]["selected"] == "c3"  # in the gap → next chunk
    assert ui.h_jump(view, "nope")["status"].startswith("❌")


def test_selecting_outside_the_filter_clears_it(key, view):
    v = ui.h_filter(view, ui.FILTER_ERRORS, "")["view"]
    out = ui._select(v, "c2")
    assert out["filter"] == ui.FILTER_ALL and out["view"]["selected"] == "c2"


def test_save_applies_text_and_timing(key, view):
    out = ui.h_save(view, "Hello there.", "Hola y bienvenidos.", "0:00.00", "0:01.50")
    c1 = ui.get_session(key).chunk("c1")
    assert c1.source_text == "Hello there." and c1.end == 1.5
    assert c1.stale == {STALE_TRANSLATION, STALE_DUB, STALE_CHECK}
    assert out["dirty"] == "" and "⚠ translation" in out["detail_md"]
    assert "⚠ 1 to re-translate" in out["summary"]
    assert ui.h_save(view, "x", "y", "zz", "1")["row_error"].startswith("❌")


def test_save_without_changes_does_nothing(key, view):
    ui.h_save(view, "Hello and welcome.", "Hola y bienvenidos.", "0:00.00", "0:02.00")
    assert ui.get_session(key).rev == 0


def test_undo_merge_and_split(key, view):
    session = ui.get_session(key)
    ui.h_save(view, "Changed.", "Hola y bienvenidos.", "0:00", "0:02")
    ui.h_undo(view)
    assert session.chunk("c1").source_text == "Hello and welcome."
    assert "Nothing to undo" in ui.h_undo(view)["row_error"]

    out = ui.h_merge(view)
    merged = out["view"]["selected"]
    assert session.ids() == [merged, "c3"]
    ui.h_undo(out["view"])
    assert session.ids() == ["c1", "c2", "c3"]
    assert ui.h_merge(ui._select(view, "c3")["view"])["row_error"].startswith("❌")

    v = ui._select(view, "c2")["view"]
    opened = ui.h_split_open(v)
    assert opened["split_group"]["visible"] and opened["split_time"] == "0:04.50"
    assert opened["src_split"]["maximum"] == 6 and opened["src_split"]["value"] == 3
    assert "**1st** 0:02.00–0:04.50: Today we talk" in opened["split_preview"]
    moved = ui.h_split_time(v, "0:03")
    assert moved["src_split"] == 1                                # 1/5 of 6 words
    done = ui.h_split_apply(v, "0:03", 2, 1)
    a = done["view"]["selected"]
    assert session.chunk(a).source_text == "Today we" and session.chunk(a).end == 3.0
    assert ui.h_split_apply(done["view"], "0:02.05", 1, 1)["split_preview"].startswith("❌")


def test_dismiss_check(key, view):
    ui.h_save(view, "Hello and welcome.", "Hola y bienvenidos.", "0:00", "0:01.8")
    ui.h_dismiss_check(view)
    assert ui.get_session(key).chunk("c1").stale == set()


def test_bulk_actions_ask_before_running(key, view, fakes):
    session = ui.get_session(key)
    assert ui.h_ask(view, STALE_DUB)["confirm_group"]["visible"] is False   # nothing stale
    session.set_target_text("c1", "uno")
    session.set_target_text("c2", "dos")
    asked = ui.h_ask(view, STALE_DUB)
    assert asked["confirm_md"].startswith("**Re-dub 2 chunks?**")
    assert asked["view"]["pending"] == {"kind": STALE_DUB, "count": 2}
    assert ui.h_cancel(asked["view"])["view"]["pending"] is None

    steps = list(ui.h_run(asked["view"], STALE_DUB, "stale"))
    assert steps[0]["busy"] is True and steps[-1]["busy"] is False
    assert all("bulk_dub" not in s for s in steps[1:-1])             # stay disabled mid-run
    assert "✓ #1 (1/2)" in steps[-1]["log"] and "Re-dub: 2/2 done" in steps[-1]["log"]
    assert session.ids(STALE_DUB) == []
    assert steps[-1]["bulk_dub"]["interactive"] is False


def test_bulk_run_refuses_a_changed_count(key, view, fakes):
    session = ui.get_session(key)
    session.set_target_text("c1", "uno")
    asked = ui.h_ask(view, STALE_DUB)
    session.set_target_text("c2", "dos")                          # changed after the ask
    last = list(ui.h_run(asked["view"], STALE_DUB, "stale"))[-1]
    assert "confirm again" in last["status"] and len(session.ids(STALE_DUB)) == 2


def test_errors_show_inline_on_the_row(key, view, fakes):
    voice, llm = fakes
    llm.responder = lambda m: "garbage"
    session = ui.get_session(key)
    session.set_source_text("c1", "Hi.")
    last = list(ui.h_run(view, STALE_TRANSLATION, "one"))[-1]
    assert "❌ #1" in last["log"]
    assert ui.errors_for(key)["c1"]
    table = _df(ui.refresh(view)["table"])
    assert table.iloc[0]["Status"] == "❌ error"
    assert ui.refresh(view)["row_error"].startswith("❌ **Last action failed:**")
    assert ui.filter_indices(session, ui.FILTER_ERRORS, "", ui.errors_for(key)) == [0]

    llm.responder = lambda m: '[{"index": "1", "text": "Hola."}]'
    list(ui.h_run(view, STALE_TRANSLATION, "one"))
    assert "c1" not in ui.errors_for(key)                          # cleared on success


def test_operations_report_a_busy_gpu(view, fakes):
    from mazinger.gpu import gpu_lock
    with gpu_lock.hold("a full dub"):
        last = list(ui.h_run(view, STALE_DUB, "one"))[-1]
    assert "a full dub is running" in last["status"] and last["busy"] is False


def test_assemble_shows_outputs_and_previous_version(key, view, project):
    ui.get_session(key).set_target_text("c1", "Hola a todos.")
    steps = list(ui.h_assemble(view))
    last = steps[-1]
    assert last["status"].startswith("✅ Output rebuilt.") and "Previous version kept" in last["status"]
    assert last["out_audio"]["value"] == project.final_audio
    assert last["out_video"]["visible"] is False                   # audio-only run
    files = last["out_files"]["value"]
    assert project.final_audio in files and project.final_srt in files
    assert ops.prev_path(project.final_audio) in last["prev_files"]["value"]
    assert "output up to date" in last["summary"]


def test_settings_form_writes_run_json(mini_project, tmp_path):
    key = ui.key_for(mini_project)
    v = ui.h_open(key, ui._empty_view())["view"]
    sample = write_tone(str(tmp_path / "upload.wav"), 3.0)
    form = ["English", "faster-whisper", "", "gpt-4.1", "", "", "qwen", "", "bfloat16",
            ui.VOICE_FROM_FILE, sample, "Reference words.", "auto", 1.4, True, False, 0.2, "audio"]
    out = ui.h_save_settings(v, *form)
    assert out["settings_msg"].startswith("✅") and out["settings_group"]["visible"] is False
    info = load_run_info(mini_project)
    assert info["voice"]["kind"] == "sample" and info["assembly"]["max_tempo"] == 1.4
    assert info["source_language"] == "English" and info["entered_in_editor"] is True
    res = ops.Resources(ui.get_session(key))
    assert res.resolve_voice() == (os.path.join(mini_project.voice_reference_dir, "voice.wav"),
                                   "Reference words.")

    form[9], form[10] = ui.VOICE_FROM_FILE, None
    assert ui.h_save_settings(v, *form)["settings_msg"].startswith("❌")


def test_settings_form_can_reuse_a_dubbed_segment(mini_project):
    key = ui.key_for(mini_project)
    v = ui.h_open(key, ui._empty_view())["view"]
    form = ["auto", "faster-whisper", "", "", "", "", "omnivoice", "", "float16",
            ui.VOICE_FROM_SEGMENT, None, "", "auto", 1.5, True, True, 0.15, "audio"]
    ui.h_save_settings(v, *form)
    res = ops.Resources(ui.get_session(key))
    audio, text = res.resolve_voice()
    assert audio == os.path.join(mini_project.voice_profile_dir, "dub_segment", "voice.wav")
    assert text and res.notices


def test_reimport_after_a_new_dub(key, view, project):
    from .conftest import FINAL_ENTRIES, write_srt
    ui.get_session(key).set_target_text("c1", "Editado.")
    write_srt(project.final_srt, FINAL_ENTRIES[:2])
    assert "dubbed again" in ui.banner_for(ui.get_session(key))
    out = ui.h_reimport(view)
    assert out["view"]["page_ids"] == ["c1", "c2"] and out["status"] == ""
    assert ui.get_session(key).chunk("c1").target_text == "Hola y bienvenidos."


def test_free_gpu(view):
    assert ui.h_free_gpu()["status"].startswith("✅")


# ═══════════════════════════════════════════════════════════════════════════════
#  Layout
# ═══════════════════════════════════════════════════════════════════════════════

def test_studio_app_has_dub_and_editor_tabs():
    from mazinger.studio import app as studio
    tabs = [b for b in studio.app.blocks.values() if isinstance(b, gr.Tab)]
    ids = {t.id for t in tabs}
    assert {"dub", "editor"} <= ids
    assert ui.BASE_DIR in studio.ALLOWED_PATHS


def test_every_name_a_handler_returns_is_a_component(key, view, fakes, project):
    """to_updates() drops unknown names silently; catch typos here."""
    with gr.Blocks():
        tab = ui.build()
    known = set(tab.components) | {"busy"}
    session = ui.get_session(key)
    session.set_target_text("c1", "uno")
    results = [
        ui.h_open(key, ui._empty_view()), ui.refresh(view), ui.h_filter(view, ui.FILTER_ALL, ""),
        ui.h_page(view, 1), ui.h_select_row(view, 1), ui.h_step(view, 1), ui.h_jump(view, "0:03"),
        ui.h_save(view, "a", "b", "0:00", "0:02"), ui.h_undo(view), ui.h_split_open(view),
        ui.h_split_time(view, "0:01"), ui.h_split_preview(view, "0:01", 1, 1),
        ui.h_ask(view, STALE_DUB), ui.h_cancel(view), ui.h_free_gpu(), ui.h_dismiss_check(view),
        *ui.h_run(view, STALE_DUB, "one"), *ui.h_assemble(view), ui.h_reimport(view),
        ui.h_merge(view),
    ]
    for r in results:
        assert set(r) <= known, set(r) - known
    assert tab.view in tab.outputs and tab.project in tab.outputs


def test_open_from_dub(project, key):
    upd = ui.open_from_dub({"video": project.video, "final_audio": project.final_audio})
    assert upd["value"] == key and any(v == key for _, v in upd["choices"])
    assert ui.open_from_dub(None)["value"] is None
