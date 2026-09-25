# Mazinger Testing Scripts

Benchmark and smoke-test scripts for the Mazinger dubbing pipeline.

Run any script from the repo root with the virtual environment activated:

```bash
source /workspace/.venv/bin/activate
python -m mazinger.testing.<script_name>
```

## Scripts

| Script | Purpose | GPU required |
|--------|---------|:------------:|
| `bench_qwen_tts` | Benchmark Qwen3-TTS inference speed (latency, RTF, throughput) on the current GPU | yes |
| `editor_synth_project` | Generate a synthetic finished dub (default 2 h, 2,500 chunks) for Editor checks | no |
| `bench_editor` | Check the Editor's performance targets on that project | no |

### bench_qwen_tts

Benchmark native Qwen3-TTS models. Uses a real mazinger voice theme for reference audio.

```bash
# Default: 1.7B model, narrator-m theme
python -m mazinger.testing.bench_qwen_tts --warmup 1 --runs 2

# 0.6B model
python -m mazinger.testing.bench_qwen_tts --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --warmup 1 --runs 2

# Custom voice theme
python -m mazinger.testing.bench_qwen_tts --voice-theme warm-f --warmup 1 --runs 2
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | HuggingFace model ID |
| `--device` | `cuda:0` | Device |
| `--dtype` | `bfloat16` | Weight dtype (`bfloat16`, `float16`, `float32`) |
| `--warmup` | `2` | Warmup iterations |
| `--runs` | `3` | Benchmark runs per sentence |
| `--voice-theme` | `narrator-m` | Mazinger voice theme for reference audio |
| `--output-dir` | `mazinger/testing/output/` | Directory for output WAVs and results JSON |

**Output:** WAV files and `results.json` saved to `output/<model-short-name>/` (git-ignored).

### editor_synth_project

Writes a project that looks like a finished dub without loading any model:
source `audio.mp3` (tone + noise), source and translated SRTs, one placeholder
WAV per chunk (0.7–1.3× its slot, so many need a tempo change), `run.json`, an
earlier `dubbed.wav`, and the cached background stem and source loudness a dub
leaves in `source/`.

```bash
python -m mazinger.testing.editor_synth_project                 # 2 h, 2,500 chunks
python -m mazinger.testing.editor_synth_project --video         # also a source video
python -m mazinger.testing.editor_synth_project --hours 0.5 --chunks 600
```

| Flag | Default | Description |
|------|---------|-------------|
| `--out` | `mazinger/testing/output/editor` | Base directory; the project goes under `projects/<slug>/` |
| `--hours` | `2` | Source length |
| `--chunks` | `2500` | Number of dubbed chunks |
| `--language` | `Spanish` | Target language folder |
| `--slug` | `editor-synthetic` | Project name |
| `--video` | off | Also write a 320×180 source video (the output is then a video too) |
| `--seed` | `0` | Random seed |

Generating the default project takes about 3 minutes (about 1 GB).

### bench_editor

Measures, through the code the Editor tab runs: session import and load
(with a replayed change log), edits, page changes, filters, search and
jump-to-time (including the Dataframe's own serialization), opening a row
(a cold clip cut), a full assembly with its peak memory, and the bytes sent to
the browser per event. For that last one, it serves the Editor tab on a local
Gradio server and drives it over the queue API. It prints PASS/FAIL against
the Editor's targets and exits non-zero if any target fails.

```bash
python -m mazinger.testing.bench_editor --generate      # make the project first
python -m mazinger.testing.bench_editor                 # reuse it
python -m mazinger.testing.bench_editor --skip-assemble --json results.json
```

| Target | Limit |
|--------|-------|
| Session import | < 3 s |
| Load an existing session | < 1 s |
| Page change / filter / search / jump | < 300 ms |
| Save an edit | < 50 ms |
| Open a row (cold clip cut) | < 500 ms |
| Assemble (background cached) | < 60 s |
| Payload per page change | < 100 KB |
| Assembly peak memory | < 1.3× the timeline (24 kHz float32) |

The project's `translated.srt` is restored and the Editor session removed
afterwards, so the benchmark can be run again. Needs `gradio` for the UI and
payload checks (skipped otherwise).
