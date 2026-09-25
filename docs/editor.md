# Editor

The **Editor** is a tab in Mazinger Studio for fixing a finished dub one chunk
at a time. You can correct a mistranscribed word, reword a translation that
runs too long, move a chunk's start or end, split or merge chunks, re-dub
only the chunks you changed, and then rebuild the final audio or video. The
rest of the dub is not run again.

Editing does not run anything. An edit only **marks** the steps that depend
on it as out of date (*stale*). You decide what to redo and when to rebuild
the output.

- [Open a project](#open-a-project)
- [The chunk table](#the-chunk-table)
- [Editing a chunk](#editing-a-chunk)
- [Stale rules](#stale-rules)
- [Redoing steps](#redoing-steps)
- [Assemble](#assemble)
- [Files and backups](#files-and-backups)
- [Long videos](#long-videos)
- [Python API](#python-api)

---

## Open a project

Start Studio (`mazinger web`) and open the **✏️ Editor** tab. There are two
ways to load a dub:

- **After a dub finishes** in the **🎬 Dub** tab, click **✏️ Open in
  Editor**. The Editor tab opens on that project and language.
- **Pick an older dub** from the **Finished dub** list. It shows every
  project language under `./mazinger_output/projects/` that has final
  subtitles and dubbed segments, newest first. Each entry shows the date, the
  chunk count, and `✏️ edited` if an Editor session already exists. To list
  another output folder, set `MAZINGER_OUTPUT_DIR` before you start Studio.
  Click **🔄 Refresh list** after a new dub.

The first time you open a dub, the Editor imports it and creates a
*session*. The next time, it loads that session with your edits.

### Chunks

A chunk is **one dubbed segment**: one entry of the final
`subtitles/translated.srt`, which the dub synthesized as one TTS clip. These
entries are made after translation, by merging and splitting subtitles, so
they don't always line up with the transcription's entries. When a dub is
imported, each transcription entry is assigned to the chunk it overlaps most
in time. No transcription text is lost, even for an entry that falls in a
gap between chunks.

### Projects dubbed before the Editor existed

Dubs record their settings in `lang/<language>/run.json`: transcription
method, LLM, TTS engine and model, voice, and assembly options. Re-doing a
single step needs these settings. Dubs made with older versions of Mazinger
have no `run.json`. For those, the Editor shows a **⚙️ Dub settings** form.
Fill it in once and it is saved to `run.json`. For **Voice for re-dubs**,
upload the original voice sample or reuse one of the dub's own segments.

API keys are never saved. The Editor uses the key in its **🔑 LLM API key**
field. If that field is empty, it falls back to the Dub tab's key, then to
`OPENAI_API_KEY`.

---

## The chunk table

The table shows 50 rows per page:

| Column | Meaning |
|---|---|
| **#** | Position in the dub |
| **Start**, **End** | Time slot (`m:ss.ss` or `h:mm:ss.ss`) |
| **Transcription** / **Translation** | Text, shortened to fit the table |
| **Status** | Every flag on the chunk (see below) |
| **Fit** | Dub length ÷ slot length. Above 115% the dub must be sped up or trimmed to fit |

Status flags:

| Flag | Meaning |
|---|---|
| `✓ ok` | Nothing to do |
| `⚠ translation` | The transcription changed since the chunk was translated |
| `⚠ dub` | The text or timing changed since the chunk was dubbed |
| `⚠ no dub` | The chunk has text but no dubbed audio |
| `🔍 check` | The timing moved, so the transcription may no longer match the audio |
| `⏱ long` | Fit above 115% |
| `empty` | No translation; the chunk stays silent |
| `❌ error` | The last operation on this chunk failed. Details are in the detail panel |

Rows that need work are tinted, and the selected row is outlined.

- **Show**: *All*, *Needs work* (any stale flag or missing dub), *Fit > 115%*,
  or *Errors*.
- **Search text**: matches the transcription or the translation (press
  Enter).
- **Jump to time**: type `m:ss` or `h:mm:ss` to select the chunk playing at
  that time.
- **◀ Page** / **Page ▶**: move between pages.

If you select a row that the current filter hides, for example with
**Jump to time**, the filter is cleared so the row is visible.

---

## Editing a chunk

Click a row to open it in the detail panel:

- **Original** plays the source audio under the chunk. **Dubbed** plays the
  chunk's current dub.
- Edit **Transcription**, **Translation**, **Start** and **End**, then click
  **💾 Save**. `● Unsaved changes` appears until you save. Timings accept
  `ss`, `m:ss` or `h:mm:ss`, with decimals.
- **◀ Prev** / **Next ▶** step through the rows in the current filter.
  Unsaved text is discarded when you move to another row.

Timing edits are clamped. A chunk can't overlap its neighbours or become
shorter than 0.2 s. If the new time would do either, the edge you moved
stops at the limit.

### Split

**✂️ Split…** opens the split controls:

- **Split at**: a time inside the chunk. It defaults to the middle. Each
  part must be at least 0.2 s long.
- **Words in the 1st part** for the transcription and for the translation.
  These default to the same proportion as the time split. Adjust them until
  the preview shows each half with the right words.

For languages written without spaces between words (Chinese, Japanese,
Thai …), the split counts characters instead of words.

Splitting makes two new chunks. Neither has a dub yet.

### Merge with next

**🔗 Merge with next** joins the chunk with the one after it. The times and
both texts are combined. The merged chunk has no dub until you re-dub it.

### Undo

**↶ Undo** reverts the selected chunk's last change. Each chunk keeps its
last 10 changes, including re-transcriptions, re-translations and re-dubs.
Old dub files are kept, so undoing a re-dub brings back the previous audio.

- Undoing a **split** restores the original chunk. Both parts must first be
  back as they were right after the split. If you edited a part, undo that
  edit first.
- Undoing a **merge** restores both original chunks.
- Undo goes back through at most two splits or merges in a row.
- A timing undo is refused if the old timing would now overlap a neighbour
  that moved.

### Keep the timing

**✓ Timing is fine** clears `🔍 check` without re-transcribing. Use it when
you moved a boundary and the transcription is still correct.

---

## Stale rules

| You change | It marks |
|---|---|
| Transcription | `⚠ translation` and `⚠ dub` |
| Translation | `⚠ dub` (an empty translation removes the dub, so the chunk is silent) |
| Start / end | `🔍 check`. The dub is kept, and its fit is recalculated |
| Split / merge | `⚠ dub` on the new chunk(s) |
| Anything | **output out of date** in the summary bar |

Each redo action clears only the flag it fixes. Its result can mark later
steps stale in turn. For example, a re-transcription that changes the text
marks the translation and the dub stale, just as editing the text by hand
does. A re-transcription that finds the same text only clears `🔍 check`.

---

## Redoing steps

### One chunk

In the detail panel:

- **🎙 Re-transcribe** transcribes the audio under the chunk again with the
  dub's transcription settings. Clears `🔍 check`. If no speech is
  recognized, the current text is kept.
- **🌐 Re-translate** translates the chunk's transcription again. The prompt
  includes two chunks of context on each side, the video's content
  description and screenshots, and a word budget based on the chunk's
  length. Clears `⚠ translation`.
- **🔊 Re-dub** synthesizes the chunk's translation in the dub's voice.
  Clears `⚠ dub`.

### Every stale chunk

The summary bar has **Re-translate N stale**, **Re-dub N stale** and
**Re-transcribe N stale**. Each one asks for confirmation with the count
first. If the count changes before you confirm, for example because another
browser tab edited the session, the Editor asks you to confirm again.

Bulk operations keep going when a chunk fails. The failed chunk is marked
`❌ error`, and the log lists every result. Engines that support batching
(Qwen3-TTS) synthesize several chunks per call. A result is discarded if the
chunk was edited while it was running.

### Models and the GPU

The Editor loads the TTS model, the voice, the ASR model and the LLM client
the first time they are needed. It keeps them loaded, so single-chunk
actions after the first are quick.

- **🧹 Free GPU** unloads them. Opening another project also unloads them.
- Only one GPU job runs at a time. If a full dub is running in the Dub tab,
  Editor operations fail at once with *"The GPU is busy"*. When you start a
  full dub, it waits for any running Editor operation to finish and then
  frees the Editor's models.

### The voice

Re-dubs use the voice the dub used:

| The dub used | Re-dubs use |
|---|---|
| An uploaded voice sample or a voice profile | The copy kept in `voice_profile/reference/` |
| A voice theme | The generated `voice_profile/voice.wav` |
| Auto-clone | The clip taken from the source audio (`voice_profile/voice.wav`) |
| OmniVoice auto voice | A segment of the dub, promoted to `voice_profile/omnivoice_auto/` |
| OmniVoice voice design (instruct) | A segment of the dub, promoted to `voice_profile/omnivoice_design/` |

OmniVoice's auto-voice and voice-design modes have no reference clip, so
they would produce a different voice each time. The Editor uses one of the
dub's own segments as the reference instead: the loudest one that is 3–10 s
long. It shows a notice the first time it does this. Auto-voice dubs made
with this version of Mazinger already keep that segment.

---

## Assemble

**🎬 Assemble** rebuilds the output from the session:

1. Every chunk's current dub is placed at its start time. Tempo is adjusted
   with the dub's settings: overruns are sped up (up to the max tempo) and
   short dubs slowed slightly. Anything that still overruns is trimmed at a
   quiet point.
2. Loudness is matched to the source, and the background is mixed in. The
   background is separated once and cached, and so is the source's loudness,
   so later assemblies skip both.
3. Subtitles are written from the chunks. On-screen lines are rebuilt at
   42 characters or fewer, and each line gets a share of the chunk's time
   in proportion to its word count.
4. If the dub produced a video, the video is rebuilt the same way: audio
   muxed in, or subtitles burned in with the same style.

Assembling never redoes a step. A chunk marked `⚠ dub` is assembled with its
current (out-of-date) dub, and a chunk with no dub is left silent. The first
line of the log counts both, so you can re-dub first if you want to.

The new output is built next to the old one and only replaces it once every
step has succeeded. If assembly fails, the previous output is untouched.

---

## Files and backups

The Editor writes only inside the project language's folder, plus the
shared caches in `source/`:

```
<base_dir>/projects/<slug>/
├── source/
│   ├── background.24000.wav      # cached background stem (shared by all languages)
│   └── loudness.json             # cached loudness of audio.mp3
└── lang/<language>/
    ├── run.json                  # settings the dub used
    ├── subtitles/
    │   ├── translated.srt        # rewritten from the session on assemble
    │   └── translated.prev.srt   # the version before the last assemble
    ├── tts/
    │   ├── segments/seg_NNNN.wav # the original dub's segments (never modified)
    │   ├── dubbed.wav            # rebuilt output
    │   ├── dubbed.prev.wav       # the output before the last assemble
    │   ├── dubbed.mp4
    │   └── dubbed.prev.mp4
    └── editor/
        ├── session.json          # the session: every chunk, its flags and undo history
        ├── changes.jsonl         # edits since session.json was last written
        ├── segments/<id>_v<n>.wav  # re-dubbed chunks; every version is kept
        ├── cache/orig_*.ogg      # original-audio clips for the player (capped at 200 MB)
        ├── source.edited.srt     # the edited transcription
        ├── subtitles.display.srt # on-screen subtitle lines (translated)
        └── source.display.srt    # on-screen subtitle lines (original)
```

- **The shared transcription is never modified.** Transcription edits are
  per language and saved to `editor/source.edited.srt`, so other languages
  of the same project are unaffected.
- **Backups:** each assemble moves the current `dubbed.wav`, `dubbed.mp4` and
  `translated.srt` to `*.prev.*` first. Only one previous version is kept.
  The output area links to it.
- **Crash safety:** each edit is one line appended to `changes.jsonl`. The
  log is merged into `session.json` every 200 edits, on assemble, and when
  the session is opened. Both files are written atomically, so a crash loses
  at most the edit being written.
- **Deleting `editor/`** discards every edit. The next open imports the
  dub again.

### Several browser tabs

All browser tabs on the same Studio server share one session per project
language, so their edits never conflict. Each tab keeps its own page,
filter and selection.

### The project was dubbed again

If the project language is dubbed again in the Dub tab after you started
editing, the Editor shows a warning. **Start over from the latest dub**
drops the session and imports the new dub.

---

## Long videos

The Editor is built for long videos. Only the current page and the selected
chunk are sent to the browser, and original-audio clips are cut only when a
row is opened. The rows around it are cut in advance. Measured on a
synthetic 2-hour project with 2,500 chunks
([`bench_editor`](../mazinger/testing/README.md#bench_editor)):

| Action | Time |
|---|---|
| First open (import) | 0.4 s |
| Open again (replaying 150 logged edits) | 0.1 s |
| Change page, filter, search or jump | ≤ 0.19 s (data per page change: ≤ 30 KB) |
| Save an edit | < 1 ms (plus ≈ 10 ms to refresh the page) |
| Open a row (cut its original clip) | ≤ 0.27 s |
| Assemble, with video | ≈ 3 min: timeline 17 s, loudness and background mix 2 min, video 40 s |

Most of the assembly time goes to ffmpeg's loudness normalization of the
whole dub and to encoding the video's audio track, which grow with the
length of the video, not with the number of edits. The first assembly of a
dub made before Mazinger 2.3 also measures the source's loudness once (about
a minute per hour of audio). If that dub has no cached background, it also
runs Demucs once.

---

## Python API

The Editor's logic lives in `mazinger.editor` and does not depend on
Gradio, so you can script it:

```python
from mazinger.paths import ProjectPaths
from mazinger.editor import Session, STALE_DUB
from mazinger.editor import ops

proj = ProjectPaths("my-video", base_dir="./mazinger_output", target_language="Spanish")
session = Session.open(proj)          # load the session, or import the dub

c = session.chunks[12]
session.set_target_text(c.id, "Una traducción más corta.")   # → ⚠ dub
a, b = session.split(session.chunks[40].id, at_time=183.5)  # two new chunk ids
session.merge_with_next(session.chunks[7].id)

res = ops.Resources(session)          # settings from run.json; api_key= for the LLM
for p in ops.redo_stale(session, STALE_DUB, res):
    print(p.done, p.total, p.chunk_id, p.error or "ok")

for p in ops.assemble(session):
    print(p.message)
print(p.outputs)                      # {"audio": ..., "video": ..., "srt": ...}
res.free()                            # unload the models
```

Every edit returns the ids of the chunks it changed and marks what is
stale. `ops.retranscribe`, `ops.retranslate` and `ops.redub` take a list of
chunk ids and yield one `ops.Progress` per chunk. The single-chunk helpers
they use, `translate_chunk`, `tts.synthesize_one` and
`transcribe.transcribe_clip`, are covered in the [Python API](python-api.md#single-chunk-helpers).
