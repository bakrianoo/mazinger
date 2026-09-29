"""Studio runs several sources in one go: one URL or path per line, or several uploads.

Every source is validated before the first one starts, a failed source does
not stop the ones after it, and the progress panel tracks each source.
"""

from mazinger.studio import pipeline as P


def _ok(source):
    yield "⏳ Working…", f"log {source}", "", None, None, None
    yield "✅ Dubbing complete!", f"log {source}", "", f"/out/{source}.mp3", None, {"final_srt": source}


def _fails(source):
    yield "❌ Pipeline failed: boom", f"log {source}", "", None, None, None


class TestParseLines:
    def test_blank_lines_comments_and_repeats_are_dropped(self):
        text = "  https://a  \n\n# later\nhttps://b\nhttps://a\n"
        assert P._parse_lines(text) == ["https://a", "https://b"]

    def test_empty_input(self):
        assert P._parse_lines(None) == []
        assert P._parse_lines("  \n ") == []


class TestResolveSources:
    def test_several_urls(self):
        sources, err = P._resolve_sources("YouTube URL", "https://a\nhttps://b", None)
        assert err is None and sources == ["https://a", "https://b"]

    def test_no_url(self):
        sources, err = P._resolve_sources("YouTube URL", "\n", None)
        assert sources is None and "URL" in err

    def test_every_bad_local_path_is_reported_before_running(self, tmp_path):
        good = tmp_path / "a.mp4"
        good.write_bytes(b"x")
        bad_ext = tmp_path / "notes.txt"
        bad_ext.write_text("x")
        text = f"{good}\n{tmp_path / 'missing.mp4'}\n{bad_ext}"
        sources, err = P._resolve_sources("Local Path", None, None, text)
        assert sources is None
        assert "missing.mp4" in err and "notes.txt" in err

    def test_valid_local_paths(self, tmp_path):
        a, b = tmp_path / "a.mp4", tmp_path / "b.wav"
        a.write_bytes(b"x")
        b.write_bytes(b"x")
        sources, err = P._resolve_sources("Local Path", None, None, f"{a}\n{b}")
        assert err is None and sources == [str(a), str(b)]

    def test_uploads_accept_a_list_or_a_single_file(self):
        assert P._resolve_sources("Upload File", None, ["/t/a.mp4", "/t/b.mp4"]) == (
            ["/t/a.mp4", "/t/b.mp4"], None,
        )
        assert P._resolve_sources("Upload File", None, "/t/a.mp4") == (["/t/a.mp4"], None)
        assert P._resolve_sources("Upload File", None, [])[0] is None


class TestRunBatch:
    def test_every_source_runs_and_the_last_success_is_shown(self):
        outs = list(P._run_batch(["one", "two"], _ok))
        status, logs, _, audio, _, render_paths, html = outs[-1]
        assert status.startswith("✅") and "2" in status
        assert "log one" in logs and "log two" in logs
        assert audio == "/out/two.mp3" and render_paths == {"final_srt": "two"}
        assert html.count("bp-done") == 2 and "100%" in html

    def test_a_failure_does_not_stop_the_rest(self):
        runs = iter([_fails, _ok])
        outs = list(P._run_batch(["bad", "good"], lambda s: next(runs)(s)))
        status, _, _, audio, _, _, html = outs[-1]
        assert status.startswith("⚠️") and "1 of 2" in status
        assert audio == "/out/good.mp3"
        assert "bp-failed" in html and "bp-done" in html and "boom" in html

    def test_a_crashing_runner_is_recorded_as_failed(self):
        def crash(source):
            raise RuntimeError("kaput")
            yield  # pragma: no cover

        status, *_, html = list(P._run_batch(["x"], crash))[-1]
        assert status.startswith("❌") and "bp-failed" in html

    def test_progress_marks_the_running_source(self):
        first = next(P._run_batch(["one", "two"], _ok))
        assert first[0].startswith("[1/2] one")
        assert "bp-running" in first[6] and "bp-pending" in first[6] and "0%" in first[6]

    def test_names_are_escaped(self):
        html = P._batch_progress_html([{"name": "<b>x</b>", "state": "pending", "detail": ""}])
        assert "<b>x</b>" not in html and "&lt;b&gt;" in html


class TestSingleSource:
    def test_single_source_leaves_the_batch_panel_empty(self, monkeypatch):
        monkeypatch.setattr(P, "_run_full_dub", lambda source, *a, **k: _ok(source))
        outs = list(P.run_dubbing(
            "YouTube URL", "https://a", None, None, "",
            "English", "Auto-Clone", None, None, None, None,
            "OpenAI", None, "sk-test", None, None,
            None, None, None, None, None,
            None, 0, 0.85, False, False, "Qwen3-TTS", None,
            "Auto", 1.5, None, True, True, 0.15,
            "Dubbed Audio", False, False,
        ))
        assert all(len(o) == 7 and o[6] == "" for o in outs)
        assert outs[-1][3] == "/out/https://a.mp3"
