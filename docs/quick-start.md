# Quick Start

## Dub a Video End-to-End

The `dub` command runs the full pipeline: download, transcribe, translate, synthesize, and assemble.

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --voice-sample speaker.m4a \
    --voice-script speaker_transcript.txt \
    --target-language Spanish \
    --base-dir ./output
```

This produces a dubbed WAV file at `./output/projects/<slug>/tts/dubbed.wav`.

To get a video file instead of just audio:

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --voice-sample speaker.m4a \
    --voice-script speaker_transcript.txt \
    --target-language Spanish \
    --output-type video
```

## Use a Voice Profile

Voice profiles let you skip `--voice-sample` and `--voice-script`. They are hosted on HuggingFace and downloaded automatically.

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --target-language Arabic
```

Profiles also work with local files:

```bash
mazinger dub ./lecture.mp4 --clone-profile abubakr
mazinger dub ./podcast.mp3 --clone-profile abubakr
```

See [Voice Profiles](voice-profiles.md) for the full list and how to upload your own.

## Add Subtitles to the Output

### During dubbing

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --embed-subtitles \
    --subtitle-font-size 24 \
    --subtitle-font-color yellow
```

`--embed-subtitles` implies `--output-type video` — no need to set both.

### Standalone subtitle burn

```bash
# Subtitles only (keeps original audio)
mazinger subtitle video.mp4 --srt translated.srt -o output.mp4

# Subtitles + replacement audio
mazinger subtitle video.mp4 --srt translated.srt --audio dubbed.wav -o output.mp4
```

For detailed styling options, see [Subtitle Styling](subtitle-styling.md).

## Translate Without Dubbing

If you only need a translated SRT file and no audio synthesis:

```bash
mazinger translate "https://youtube.com/watch?v=VIDEO_ID" \
    --target-language French \
    --base-dir ./output
```

This downloads the video, transcribes it, extracts thumbnails and a description, then translates the subtitles. The result is saved at `./output/projects/<slug>/subtitles/translated.srt`.

### Translate and burn subtitles in one step

```bash
mazinger translate "https://youtube.com/watch?v=VIDEO_ID" \
    --target-language Arabic \
    --embed-subtitles \
    --subtitle-google-font "Noto Sans Arabic" \
    --subtitle-font-size 24 \
    --base-dir ./output
```

## Run Individual Stages

Each pipeline stage has its own sub-command. This is useful when you want to inspect or modify intermediate output before continuing.

### Download and extract audio

```bash
mazinger download "https://youtube.com/watch?v=VIDEO_ID" --base-dir ./output
```

### Slice a video to a time range

```bash
# Extract a clip
mazinger slice "https://youtube.com/watch?v=VIDEO_ID" --start 00:01:30 --end 00:05:00

# Dub only a portion of a video
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr --target-language Arabic \
    --start 00:01:30 --end 00:05:00
```

### Transcribe

```bash
# Cloud (OpenAI Whisper API)
mazinger transcribe ./output/projects/my-video/source/audio.mp3 -o subs.srt

# Local with faster-whisper
mazinger transcribe audio.mp3 -o subs.srt --method faster-whisper --device cuda

# Local with WhisperX
mazinger transcribe audio.mp3 -o subs.srt --method whisperx --device cuda
```

### Extract thumbnails

```bash
mazinger thumbnails \
    --video video.mp4 \
    --srt subs.srt \
    --output-dir ./thumbs
```

### Describe content

```bash
mazinger describe \
    --srt subs.srt \
    --thumbnails-meta ./thumbs/meta.json \
    -o description.json
```

### Translate subtitles

```bash
mazinger translate \
    --srt subs.srt \
    --description description.json \
    --thumbnails-meta ./thumbs/meta.json \
    --target-language German \
    -o translated.srt
```

### Re-segment for readability

```bash
mazinger resegment --srt translated.srt -o final.srt
```

### Synthesize dubbed audio

```bash
mazinger speak \
    --srt translated.srt \
    --original-audio audio.mp3 \
    --voice-sample speaker.m4a \
    --voice-script speaker_transcript.txt \
    -o dubbed.wav
```

Or with a profile:

```bash
mazinger speak \
    --srt translated.srt \
    --original-audio audio.mp3 \
    --clone-profile abubakr \
    -o dubbed.wav
```

## Resume an Interrupted Run

Every stage caches its output files. If a run is interrupted — for example, during TTS synthesis on segment 47 of 200 — re-running the same command picks up where it left off. Already-completed stages and individual segment WAVs are skipped.

```bash
# Same command, just run it again
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --target-language Arabic
```

To discard all cached outputs and start fresh:

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --force-reset
```

`--force-reset` also works with the `speak` sub-command to re-synthesize all TTS segments.

## Use Chatterbox Instead of Qwen

Add `--tts-engine chatterbox` to any command that involves voice synthesis:

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --tts-engine chatterbox

mazinger speak --srt translated.srt --original-audio audio.mp3 \
    --voice-sample speaker.m4a \
    --tts-engine chatterbox \
    -o dubbed.wav
```

Chatterbox does not require a voice transcript (`--voice-script`), only the audio sample. It also supports emotion control:

```bash
mazinger speak --srt translated.srt --original-audio audio.mp3 \
    --voice-sample speaker.m4a \
    --tts-engine chatterbox \
    --chatterbox-exaggeration 0.7 \
    --chatterbox-cfg 0.3 \
    -o dubbed.wav
```

## Control Playback Speed

By default, dubbed segments are placed at their original timestamps without speed adjustment. Use tempo flags to control pacing:

```bash
# Fixed speed-up: all segments 10% faster
mazinger speak --srt translated.srt --original-audio audio.mp3 \
    --clone-profile abubakr \
    --fixed-tempo 1.1 \
    -o dubbed.wav

# Dynamic: per-segment speed to match original timing, max 1.3×
mazinger speak --srt translated.srt --original-audio audio.mp3 \
    --clone-profile abubakr \
    --dynamic-tempo --max-tempo 1.3 \
    -o dubbed.wav
```

## Python API

```python
from mazinger import MazingerDubber
from mazinger.profiles import fetch_profile

dubber = MazingerDubber(openai_api_key="sk-...", base_dir="./output")

# Option A: explicit voice files
proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    voice_sample="speaker.m4a",
    voice_script="speaker_transcript.txt",
    target_language="Spanish",
)

# Option B: use a profile
voice, script = fetch_profile("abubakr")
proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    voice_sample=voice,
    voice_script=script,
    target_language="Arabic",
    output_type="video",
    embed_subtitles=True,
)

print(proj.final_audio)   # path to dubbed.wav
print(proj.final_video)   # path to dubbed.mp4 (when output_type="video")
print(proj.summary())     # human-readable overview of all output files
```

See [Python API](python-api.md) for the full reference.
