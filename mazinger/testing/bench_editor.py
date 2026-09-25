"""Check the Editor's performance targets on a long synthetic project.

Usage:
    python -m mazinger.testing.bench_editor [--out DIR] [--generate]
           [--hours H] [--chunks N] [--skip-assemble] [--skip-payload]
           [--json PATH]

Runs against the project made by :mod:`mazinger.testing.editor_synth_project`
(generated first with ``--generate``, or when missing) and measures, through
the same code the Editor tab calls:

====================================  ===========
Measure                               Target
====================================  ===========
Session import                        < 3 s
Load existing session (with log)      < 1 s
Page change / filter (handler)        < 300 ms
Save an edit (one log line)           < 50 ms
Open a row (cold clip cut)            < 500 ms
Assemble (background cached)          < 60 s
Browser payload per page change       < 100 KB
Assembly peak memory                  < 1.3× the timeline
====================================  ===========

The payload is measured on the wire: the Editor tab is served by a local
Gradio server and driven over its queue API, and the bytes of each event's
result message are counted.

The project's ``translated.srt`` is restored afterwards and the Editor
session is removed, so the benchmark can be run again on the same project.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import threading
import time

from mazinger.testing.editor_synth_project import DEFAULT_SLUG, generate

TARGETS = {
    "import_s": 3.0,
    "load_s": 1.0,
    "page_ms": 300.0,
    "edit_ms": 50.0,
    "open_row_ms": 500.0,
    "assemble_s": 60.0,
    "payload_kb": 100.0,
    "assemble_mem_ratio": 1.3,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

class Timer:
    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {}

    def time(self, name: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        self.samples.setdefault(name, []).append(time.perf_counter() - t0)
        return out

    def stats(self, name: str, scale: float = 1.0) -> dict:
        s = self.samples.get(name, [])
        if not s:
            return {}
        return {"n": len(s), "median": statistics.median(s) * scale, "max": max(s) * scale}


class PeakRSS:
    """Peak resident memory of this process while the context is open (Linux)."""

    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self.base = self.peak = 0
        self._stop = threading.Event()

    @staticmethod
    def rss() -> int:
        try:
            with open("/proc/self/statm") as fh:
                return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            return 0

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self.rss())

    def __enter__(self):
        self.base = self.peak = self.rss()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.rss())

    @property
    def available(self) -> bool:
        return self.base > 0

    @property
    def growth(self) -> int:
        return self.peak - self.base


def _line(label: str, value: float | None, target: float, unit: str, extra: str = "") -> tuple[str, bool]:
    if value is None:
        return f"  {label:<38} {'—':>10}   (target < {target:g} {unit}) SKIPPED", True
    ok = value < target
    return (f"  {label:<38} {value:>10.3f} {unit:<3} (target < {target:g}) "
            f"{'PASS' if ok else 'FAIL'}{extra}"), ok


# ═══════════════════════════════════════════════════════════════════════════════
#  Measurements
# ═══════════════════════════════════════════════════════════════════════════════

def bench_session(proj, timer: Timer, rng: random.Random) -> dict:
    """Import, edits (one log line each), and load with a pending change log."""
    from mazinger.editor.session import Session

    results: dict = {}
    for _ in range(3):
        shutil.rmtree(proj.editor_dir, ignore_errors=True)
        session = timer.time("import", Session.import_project, proj)
    results["chunks"] = len(session)

    # ~150 edits of every kind — below the merge threshold, so the next load
    # has a real change log to replay.
    for _ in range(3):
        shutil.rmtree(proj.editor_dir, ignore_errors=True)
        session = Session.import_project(proj)
        n = len(session)
        for k in range(150):
            c = session.chunks[rng.randrange(n - 1)]
            kind = k % 10
            if kind < 6:
                timer.time("edit:set_target_text", session.set_target_text, c.id, c.target_text + " editado")
            elif kind == 6:
                timer.time("edit:set_source_text", session.set_source_text, c.id, c.source_text + " edited")
            elif kind == 7:
                timer.time("edit:set_timing", session.set_timing, c.id, c.start + 0.05, c.end - 0.05)
            elif kind == 8 and c.duration > 1.0:
                timer.time("edit:split", session.split, c.id, (c.start + c.end) / 2)
            else:
                timer.time("edit:merge_with_next", session.merge_with_next, c.id)
            n = len(session)
        cid = session.chunks[rng.randrange(n)].id
        session.set_target_text(cid, "texto de prueba")
        timer.time("edit:undo", session.undo, cid)
        pending = session.store.pending
        timer.time("load", Session.load, proj)
    results["log_records_replayed"] = pending
    # And once more with an empty log (the common case).
    timer.time("load:no_log", Session.load, proj)

    # A full snapshot (every MERGE_EVERY edits, on assemble and on load).
    session = Session.load(proj)
    for _ in range(3):
        timer.time("snapshot", session.save)
    return results


def bench_clips(proj, session, timer: Timer, rng: random.Random) -> None:
    """Cold clip cuts across the whole source."""
    from mazinger.editor import media

    cache = media.ClipCache(proj.audio, os.path.join(proj.editor_dir, "cache-bench"))
    try:
        for c in rng.sample(session.chunks, 20):
            timer.time("clip:cold", cache.clip, c.start, c.end)
            timer.time("clip:hit", cache.clip, c.start, c.end)
    finally:
        cache.close()
        shutil.rmtree(cache.cache_dir, ignore_errors=True)


def bench_handlers(key: str, table, timer: Timer, rng: random.Random) -> None:
    """The Editor tab's handlers, including the Dataframe's own postprocess."""
    from mazinger.studio import editor_ui as ui

    def run(name, fn, *args):
        def call():
            out = fn(*args)
            if "table" in out:
                table.postprocess(out["table"])
            return out
        return timer.time(name, call)

    ui.get_session(key, reload=True)
    shutil.rmtree(os.path.join(ui.paths_for(key).editor_dir, "cache"), ignore_errors=True)
    view = run("ui:open", ui.h_open, key, ui._empty_view())["view"]
    for _ in range(20):
        view = run("ui:page", ui.h_page, view, +1)["view"]
    for flt in ui.FILTERS * 3:
        view = run("ui:filter", ui.h_filter, view, flt, "")["view"]
    for q in ("modelo", "entrenamiento", "zzz-no-match") * 2:
        view = run("ui:search", ui.h_filter, view, ui.FILTER_ALL, q)["view"]
    view = ui.h_filter(view, ui.FILTER_ALL, "")["view"]
    session = ui.get_session(key)
    for _ in range(10):
        t = rng.uniform(0, session.chunks[-1].end)
        view = run("ui:jump", ui.h_jump, view, ui.fmt_time(t))["view"]

    # Opening rows: a cold clip cut each time (the prefetch of the rows around
    # the previous selection is not what is being measured).
    for _ in range(20):
        view = ui.h_page(view, rng.randrange(-5, 6))["view"]
        ui._CLIPS[key].close()
        shutil.rmtree(os.path.join(session.proj.editor_dir, "cache"), ignore_errors=True)
        view = run("ui:open_row", ui.h_select_row, view, rng.randrange(len(view["page_ids"])))["view"]
    for _ in range(10):
        view = run("ui:next_row", ui.h_step, view, +1)["view"]

    for k in range(10):
        c = session.chunk(view["selected"])
        run("ui:save", ui.h_save, view, c.source_text, c.target_text + f" ({k})",
            ui.fmt_time(c.start), ui.fmt_time(c.end))
    ui._CLIPS[key].close()


def bench_payload(key: str, base_dir: str) -> dict:
    """Bytes sent to the browser per event, measured on the wire."""
    import gradio as gr
    import httpx

    from mazinger.studio import editor_ui as ui

    with gr.Blocks() as demo:
        tab = ui.build()
    demo.queue()
    demo.launch(prevent_thread_lock=True, server_name="127.0.0.1", quiet=True,
                allowed_paths=[base_dir], show_error=True)
    try:
        root = demo.local_url.rstrip("/") + "/gradio_api"
        c = tab.components
        by_label = {getattr(b, "value", None): b for b in demo.blocks.values()
                    if isinstance(b, gr.Button)}

        def fn_for(block, event):
            for fid, fn in demo.fns.items():
                if any(t[0] == block._id and t[1] == event for t in fn.targets):
                    return fid, fn
            raise LookupError(f"No {event} handler on {block}")

        session_hash = "bench" + str(random.randrange(10**9))
        client = httpx.Client(timeout=120)

        def call(block, event, values: dict, event_data=None) -> int:
            fid, fn = fn_for(block, event)
            data = [values.get(inp._id) for inp in fn.inputs]
            r = client.post(f"{root}/queue/join", json={
                "data": data, "event_data": event_data, "fn_index": fid,
                "trigger_id": block._id, "session_hash": session_hash,
            })
            r.raise_for_status()
            event_id = r.json()["event_id"]
            size = 0
            with client.stream("GET", f"{root}/queue/data", params={"session_hash": session_hash}) as s:
                for line in s.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    msg = json.loads(line[5:])
                    if msg.get("event_id") != event_id:
                        continue
                    if msg.get("msg") in ("process_generating", "process_completed"):
                        size += len(line.encode("utf-8"))
                    if msg.get("msg") == "process_completed":
                        if not msg.get("success", True):
                            raise RuntimeError(f"{event} failed: {msg.get('output')}")
                        return size
            raise RuntimeError("Stream ended early")

        out = {
            "open_project": call(c["project"], "change", {c["project"]._id: key}),
            "page_change": call(by_label["Page ▶"], "click", {}),
            "filter": call(c["filter"], "change", {c["filter"]._id: ui.FILTER_NEEDS_WORK,
                                                   c["search"]._id: ""}),
            "select_row": call(c["table"], "select", {},
                               event_data={"index": [3, 1], "value": "", "row_value": []}),
            "next_row": call(c["next_row"], "click", {}),
        }
        client.close()
        return out
    finally:
        demo.close()


def bench_assemble(session) -> dict:
    from mazinger.editor import ops

    # Each progress message announces the step that follows it, so the time
    # until the next message belongs to that step.
    steps = []
    with PeakRSS() as mem:
        t0 = last = time.perf_counter()
        step = "Starting"
        for p in ops.assemble(session):
            now = time.perf_counter()
            steps.append((step, now - last))
            step, last = p.message, now
        total = time.perf_counter() - t0
    timeline_bytes = session.duration * 24_000 * 4
    return {
        "total_s": total,
        "steps": steps,
        "peak_growth_mb": mem.growth / 2**20 if mem.available else None,
        "timeline_mb": timeline_bytes / 2**20,
        "mem_ratio": mem.growth / timeline_bytes if mem.available else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="mazinger/testing/output/editor",
                    help="Base directory of the synthetic project")
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    ap.add_argument("--language", default="Spanish")
    ap.add_argument("--generate", action="store_true", help="(Re)generate the project first")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--chunks", type=int, default=2500)
    ap.add_argument("--skip-assemble", action="store_true")
    ap.add_argument("--skip-payload", action="store_true")
    ap.add_argument("--json", help="Also write the results here")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    from mazinger.paths import ProjectPaths

    base = os.path.abspath(args.out)
    # The Editor tab lists (and its dropdown accepts) projects under this base.
    os.environ["MAZINGER_OUTPUT_DIR"] = base
    proj = ProjectPaths(args.slug, base_dir=base, target_language=args.language)
    if args.generate or not os.path.isfile(proj.final_srt):
        generate(base, hours=args.hours, n_chunks=args.chunks, language=args.language, slug=args.slug)
    key = os.path.dirname(proj.run_info)
    rng = random.Random(args.seed)
    timer = Timer()

    with open(proj.final_srt, encoding="utf-8") as fh:
        original_srt = fh.read()
    results: dict = {"project": proj.root}
    try:
        print("Session: import, edits, load…", flush=True)
        results.update(bench_session(proj, timer, rng))

        from mazinger.editor.session import Session
        session = Session.load(proj)
        print("Clips…", flush=True)
        bench_clips(proj, session, timer, rng)

        try:
            import gradio as gr
            from mazinger.studio import editor_ui as ui
            print("Editor tab handlers…", flush=True)
            with gr.Blocks():
                table = ui.build().components["table"]
            bench_handlers(key, table, timer, rng)
        except ImportError as exc:
            print(f"  (skipped: {exc})")
            gr = None

        if gr is not None and not args.skip_payload:
            print("Wire payload…", flush=True)
            results["payload_bytes"] = bench_payload(key, base)

        if not args.skip_assemble:
            print("Assemble…", flush=True)
            results["assemble"] = bench_assemble(Session.load(proj))
    finally:
        # Leave the project as it was generated.
        with open(proj.final_srt, "w", encoding="utf-8") as fh:
            fh.write(original_srt)
        shutil.rmtree(proj.editor_dir, ignore_errors=True)
        try:
            from mazinger.studio import editor_ui as ui
            ui._SESSIONS.pop(key, None)
        except ImportError:
            pass

    results["timings"] = {name: timer.stats(name) for name in timer.samples}

    # ── report ────────────────────────────────────────────────────────────────
    ms = lambda name: (timer.stats(name, 1000) or {}).get("max")  # noqa: E731
    sec = lambda name: (timer.stats(name) or {}).get("max")  # noqa: E731
    edit_max = max((v for k in timer.samples if k.startswith("edit:") for v in [ms(k)]), default=None)
    page_max = max((v for k in ("ui:page", "ui:filter", "ui:search", "ui:jump")
                    if (v := ms(k)) is not None), default=None)
    payload = results.get("payload_bytes") or {}
    page_kb = max((payload[k] for k in ("page_change", "filter") if k in payload), default=None)
    asm = results.get("assemble") or {}

    print(f"\n{results['chunks']:,} chunks · {proj.root}\n\nTargets (worst case of every run):")
    checks = [
        _line("Session import", sec("import"), TARGETS["import_s"], "s"),
        _line("Load existing session", sec("load"), TARGETS["load_s"], "s",
              f"  ({results['log_records_replayed']} log records replayed)"),
        _line("Page change / filter / search / jump", page_max, TARGETS["page_ms"], "ms"),
        _line("Save an edit (session)", edit_max, TARGETS["edit_ms"], "ms"),
        _line("Open a row (cold clip cut)", ms("ui:open_row"), TARGETS["open_row_ms"], "ms"),
        _line("Assemble (background cached)", asm.get("total_s"), TARGETS["assemble_s"], "s"),
        _line("Payload per page change", page_kb / 1024 if page_kb else None,
              TARGETS["payload_kb"], "KB"),
        _line("Assembly peak memory / timeline", asm.get("mem_ratio"), TARGETS["assemble_mem_ratio"], "×",
              f"  ({asm['peak_growth_mb']:.0f} MB for a {asm['timeline_mb']:.0f} MB timeline)"
              if asm.get("mem_ratio") else ""),
    ]
    for text, _ in checks:
        print(text)

    print("\nAll timings (median / max):")
    for name in sorted(timer.samples):
        st = timer.stats(name, 1000)
        print(f"  {name:<26} {st['median']:>9.1f} / {st['max']:>9.1f} ms   (n={st['n']})")
    if payload:
        print("\nPayload per event:")
        for name, size in payload.items():
            print(f"  {name:<26} {size / 1024:>9.1f} KB")
    if asm.get("steps"):
        print("\nAssemble steps:")
        for msg, dt in asm["steps"]:
            print(f"  {dt:>7.1f} s  {msg}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
