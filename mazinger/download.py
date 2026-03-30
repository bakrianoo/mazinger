"""Download a video from a URL, or ingest a local video/audio file."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import yt_dlp

from mazinger.utils import sanitize_filename, save_json

log = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".ts", ".m2ts"}
_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus"}

# -- Video quality helpers ------------------------------------------------

_QUALITY_PRESETS: dict[str, int | None] = {
    "low": 360,
    "medium": 720,
    "high": None,  # best available
}


def resolve_quality(quality: str | None) -> int | None:
    """Map a quality specifier to a maximum video height (pixels).

    Accepts named presets (``low``, ``medium``, ``high``) or a numeric
    resolution (e.g. ``"1080"``).  Returns ``None`` for *best available*.
    Defaults to ``medium`` when *quality* is ``None``.
    """
    if quality is None:
        quality = "medium"
    quality = quality.strip().lower()
    if quality in _QUALITY_PRESETS:
        return _QUALITY_PRESETS[quality]
    try:
        return int(quality)
    except ValueError:
        raise ValueError(
            f"Invalid quality: {quality!r}. "
            "Use low/medium/high or a resolution like 144, 720, 1080."
        )


def _build_format_string(max_height: int | None) -> str:
    """Return a yt-dlp ``format`` string constrained to *max_height*.

    Falls back to the best available stream when the requested height is
    unavailable, ensuring a download always succeeds.
    """
    if max_height is None:
        return "bestvideo*+bestaudio/best"
    return (
        f"bestvideo[height<={max_height}]+bestaudio"
        f"/best[height<={max_height}]"
        f"/bestvideo+bestaudio/best"
    )


def _probe_video_height(path: str) -> int | None:
    """Return the pixel height of the first video stream, or ``None``."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=height",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def is_url(source: str) -> bool:
    """Return ``True`` if *source* looks like a URL rather than a local path."""
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https", "ftp", "ftps")


def _strip_playlist_params(url: str) -> str:
    """Remove playlist-related query parameters from a YouTube URL.

    yt-dlp's ``noplaylist`` option is not always sufficient — some playlist
    parameters can still cause it to resolve the playlist instead of the
    single video.  Stripping ``list``, ``index``, and ``start_radio`` from
    the query string ensures only the target video is downloaded.

    Non-YouTube URLs are returned unchanged.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not any(h in host for h in ("youtube.com", "youtu.be", "youtube-nocookie.com")):
        return url
    qs = parse_qs(parsed.query, keep_blank_values=True)
    _PLAYLIST_KEYS = {"list", "index", "start_radio"}
    stripped = {k: v for k, v in qs.items() if k not in _PLAYLIST_KEYS}
    new_query = urlencode(stripped, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def is_audio_file(path: str) -> bool:
    """Return ``True`` if *path* has a recognised audio extension."""
    return Path(path).suffix.lower() in _AUDIO_EXTS


def is_video_file(path: str) -> bool:
    """Return ``True`` if *path* has a recognised video extension."""
    return Path(path).suffix.lower() in _VIDEO_EXTS


def slug_from_path(path: str) -> str:
    """Derive a filesystem-safe slug from a local file path."""
    return sanitize_filename(Path(path).stem)


def _copy_file(src: str, dst: str) -> str:
    """Copy *src* to *dst* unless *dst* already exists. Returns *dst*."""
    if os.path.exists(dst):
        log.info("File already exists: %s", dst)
        return dst
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copy2(src, dst)
    log.info("Copied %s -> %s", src, dst)
    return dst


def ingest_local_video(video_path: str, proj_video: str, proj_audio: str) -> str:
    """Copy a local video into the project and extract its audio.

    Returns:
        The project audio path.
    """
    _copy_file(video_path, proj_video)
    return extract_audio(proj_video, proj_audio)


def ingest_local_audio(audio_path: str, proj_audio: str) -> str:
    """Copy a local audio file into the project.

    Returns:
        The project audio path.
    """
    return _copy_file(audio_path, proj_audio)


def _yt_dlp_auth_opts(
    *,
    cookies_from_browser: str | None = None,
    cookies: str | None = None,
) -> dict[str, Any]:
    """Build yt-dlp auth options from cookie-related inputs."""
    opts: dict[str, Any] = {}
    if cookies_from_browser:
        # yt-dlp expects a tuple-like spec for cookies-from-browser.
        parts = [part or None for part in cookies_from_browser.split(":")]
        opts["cookiesfrombrowser"] = tuple(parts)
    if cookies:
        opts["cookiefile"] = cookies
    return opts


def _yt_dlp_common_opts() -> dict[str, Any]:
    """Return yt-dlp options that keep behavior predictable across environments."""
    return {
        # Avoid picking up a user's global yt-dlp config that may force
        # custom formats and break metadata-only calls.
        "ignoreconfig": True,
        "noplaylist": True,
        # Enable the Node.js runtime for YouTube JS challenge solving.
        "js_runtimes": {"node": {}},
        "remote_components": ["ejs:github"],
    }


def resolve_slug(
    url: str,
    *,
    cookies_from_browser: str | None = None,
    cookies: str | None = None,
) -> tuple[str, dict]:
    """Fetch video metadata and derive a filesystem-safe slug from the title.

    Returns:
        A ``(slug, info_dict)`` tuple where *info_dict* is the full metadata
        dictionary returned by yt-dlp.
    """
    url = _strip_playlist_params(url)
    opts = {
        "skip_download": True,
        "quiet": True,
        **_yt_dlp_common_opts(),
        **_yt_dlp_auth_opts(
            cookies_from_browser=cookies_from_browser,
            cookies=cookies,
        ),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        # ``process=False`` avoids format selection during metadata lookup.
        info = ydl.extract_info(url, download=False, process=False)
    title = info.get("title") or info.get("id") or "video"
    slug = sanitize_filename(title)
    log.info("Resolved slug: %s", slug)
    return slug, info


# Keys to extract from the yt-dlp info dict for LLM context.
_META_KEYS = (
    "title", "description", "uploader", "channel", "upload_date",
    "duration", "categories", "tags", "language",
)

# Map YouTube auto-caption language codes (with -orig suffix stripped)
# to the language names used by Qwen TTS.
_YT_CODE_TO_LANG: dict[str, str] = {
    "zh-Hans": "Chinese", "zh-Hant": "Chinese",
    "zh": "Chinese", "en": "English", "ja": "Japanese",
    "ko": "Korean", "de": "German", "fr": "French",
    "ru": "Russian", "pt": "Portuguese", "es": "Spanish",
    "it": "Italian",
    # Broader translate-supported languages
    "ar": "Arabic", "bn": "Bengali", "cs": "Czech",
    "da": "Danish", "nl": "Dutch", "fi": "Finnish",
    "el": "Greek", "he": "Hebrew", "hi": "Hindi",
    "hu": "Hungarian", "id": "Indonesian", "ms": "Malay",
    "nb": "Norwegian", "no": "Norwegian", "fa": "Persian",
    "pl": "Polish", "ro": "Romanian", "sv": "Swedish",
    "th": "Thai", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "vi": "Vietnamese",
}

# Reverse: language name → preferred YouTube code (first wins).
_LANG_TO_YT_CODE: dict[str, str] = {}
for _c, _l in _YT_CODE_TO_LANG.items():
    _LANG_TO_YT_CODE.setdefault(_l, _c)


def _detect_original_language(info: dict) -> str | None:
    """Detect the original spoken language from yt-dlp *info*.

    Checks the ``*-orig`` key in ``automatic_captions``, then falls back to
    the ``language`` field.
    """
    auto = info.get("automatic_captions") or {}
    for code in auto:
        if code.endswith("-orig"):
            base = code[: -len("-orig")]
            return _YT_CODE_TO_LANG.get(base, base)
    return info.get("language")


def save_video_meta(info: dict, output_path: str) -> str:
    """Extract useful video metadata from a yt-dlp *info* dict and save as JSON.

    Stores a curated subset of fields — enough to give LLMs useful context
    without leaking private or overly verbose data.  Also records the
    detected original language, available manual subtitles, and auto-caption
    language codes.

    Returns *output_path*.
    """
    meta = {k: info[k] for k in _META_KEYS if info.get(k) is not None}
    if not meta:
        log.debug("No video metadata to save")
        return output_path

    # Derived fields
    orig_lang = _detect_original_language(info)
    if orig_lang:
        meta["original_language"] = orig_lang

    manual_subs = list((info.get("subtitles") or {}).keys())
    if manual_subs:
        meta["available_subtitles"] = manual_subs

    auto_codes = list((info.get("automatic_captions") or {}).keys())
    if auto_codes:
        meta["available_auto_captions"] = auto_codes

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_json(meta, output_path)
    log.info("Video metadata saved: %s", output_path)
    return output_path


# ── YouTube subtitle download ───────────────────────────────────────────────


def _subtitle_url(info: dict, lang_code: str, fmt: str = "srt") -> str | None:
    """Return the download URL for *lang_code* in *fmt*, or ``None``.

    Checks manual subtitles first, then automatic captions.
    """
    for bucket in ("subtitles", "automatic_captions"):
        tracks = (info.get(bucket) or {}).get(lang_code, [])
        for entry in tracks:
            if entry.get("ext") == fmt:
                return entry["url"]
    return None


def _download_subtitle_url(url: str, dest: str) -> str:
    """Download a subtitle file from *url* to *dest*."""
    import urllib.request

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        out.write(resp.read())
    return dest


def download_youtube_subtitles(
    info: dict,
    output_dir: str,
    *,
    target_languages: list[str] | None = None,
) -> dict[str, str]:
    """Download available YouTube subtitles into *output_dir*.

    Always downloads the **original-language** track (the ``*-orig``
    auto-caption) when available.  Additionally downloads any tracks
    whose language matches *target_languages* (language names like
    ``"English"``, ``"Arabic"``).  When *target_languages* is ``None``,
    downloads tracks for all languages in ``_YT_CODE_TO_LANG``.

    Manual (creator-uploaded) subtitles are preferred over auto-generated
    ones.  Files are saved as ``<lang_code>.srt`` (auto-generated) or
    ``<lang_code>.manual.srt`` (creator-uploaded).

    Returns:
        A dict mapping ``"<lang_code>"`` → absolute file path for each
        successfully downloaded subtitle.
    """
    manual_tracks = info.get("subtitles") or {}
    auto_tracks = info.get("automatic_captions") or {}
    all_codes: set[str] = set()

    # 1. Always include the original-language track.
    for code in auto_tracks:
        if code.endswith("-orig"):
            all_codes.add(code)
            # Also include the base code (e.g. "ar" alongside "ar-orig").
            all_codes.add(code[: -len("-orig")])
            break

    # 2. Determine which target language codes to download.
    if target_languages is None:
        wanted_codes = set(_YT_CODE_TO_LANG.keys())
    else:
        wanted_codes = set()
        for lang_name in target_languages:
            code = _LANG_TO_YT_CODE.get(lang_name)
            if code:
                wanted_codes.add(code)

    # Only keep codes that actually exist in the info dict.
    for code in wanted_codes:
        if code in manual_tracks or code in auto_tracks:
            all_codes.add(code)

    if not all_codes:
        log.info("No matching YouTube subtitles to download")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    downloaded: dict[str, str] = {}

    for code in sorted(all_codes):
        is_manual = code in manual_tracks
        url = _subtitle_url(info, code)
        if not url:
            continue
        suffix = ".manual.srt" if is_manual else ".srt"
        dest = os.path.join(output_dir, f"{code}{suffix}")
        if os.path.exists(dest):
            log.debug("YouTube subtitle already cached: %s", dest)
            downloaded[code] = dest
            continue
        try:
            _download_subtitle_url(url, dest)
            tag = "manual" if is_manual else "auto"
            log.info("Downloaded YouTube subtitle [%s] (%s): %s", code, tag, dest)
            downloaded[code] = dest
        except Exception as exc:
            log.warning("Failed to download subtitle [%s]: %s", code, exc)

    return downloaded


def download_video(
    url: str,
    output_path: str,
    *,
    quality: str | None = None,
    cookies_from_browser: str | None = None,
    cookies: str | None = None,
) -> str:
    """Download a video from *url* into *output_path*.

    Parameters:
        quality: Target quality — ``low``, ``medium`` (default), ``high``,
                 or a numeric resolution like ``"1080"``.  When the exact
                 resolution is unavailable the next-best option is used and
                 a warning is logged.

    Skips the download when *output_path* already exists.

    Returns:
        The resolved *output_path*.
    """
    if os.path.exists(output_path):
        log.info("Video already exists: %s", output_path)
        return output_path

    max_height = resolve_quality(quality)
    fmt = _build_format_string(max_height)
    log.info(
        "Requesting quality=%s (max_height=%s, format=%s)",
        quality or "medium", max_height, fmt,
    )

    url = _strip_playlist_params(url)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    opts = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": output_path,
        **_yt_dlp_common_opts(),
        **_yt_dlp_auth_opts(
            cookies_from_browser=cookies_from_browser,
            cookies=cookies,
        ),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    # -- Warn when actual resolution differs from requested ---------------
    if max_height is not None:
        actual = _probe_video_height(output_path)
        if actual is not None and actual != max_height:
            if actual > max_height:
                log.warning(
                    "Requested %dp but downloaded %dp (closest available).",
                    max_height, actual,
                )
            else:
                log.warning(
                    "Requested %dp not available — downloaded %dp instead.",
                    max_height, actual,
                )

    log.info("Video saved: %s", output_path)
    return output_path


def extract_audio(video_path: str, audio_path: str) -> str:
    """Extract the audio track from *video_path* as an MP3 file.

    Skips extraction when *audio_path* already exists.

    Returns:
        The resolved *audio_path*.
    """
    if os.path.exists(audio_path):
        log.info("Audio already exists: %s", audio_path)
        return audio_path

    os.makedirs(os.path.dirname(audio_path) or ".", exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "libmp3lame", "-q:a", "2",
            audio_path,
        ],
        check=True,
        capture_output=True,
    )
    log.info("Audio extracted: %s", audio_path)
    return audio_path


def _parse_timestamp(ts: str) -> float:
    """Convert a timestamp string to seconds.

    Accepts ``HH:MM:SS``, ``HH:MM:SS.mmm``, ``MM:SS``, ``MM:SS.mmm``,
    or plain seconds (``"90"``, ``"90.5"``).
    """
    ts = ts.strip()
    if ":" not in ts:
        return float(ts)
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    raise ValueError(f"Unrecognised timestamp format: {ts!r}")


def _has_video_stream(path: str) -> bool:
    """Return ``True`` if *path* contains at least one video stream."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip().lower() == "video"
    except subprocess.CalledProcessError:
        return False


def slice_media(
    input_path: str,
    output_path: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> str:
    """Extract a time range from a media file using ffmpeg.

    Produces frame-accurate cuts with no frozen frames by re-encoding
    video at the cut boundaries.  For audio-only files the audio is
    re-encoded to avoid partial-frame artifacts.

    The strategy:

    * ``-ss`` is placed **before** ``-i`` for fast input seeking (jumps
      to the nearest keyframe before the target).
    * The output is **re-encoded** so that the first frame is a clean
      I-frame at exactly the requested timestamp — no frozen/black
      leading frames.
    * ``-t`` (duration) is used instead of ``-to`` so the length is
      always relative to the seek point, regardless of ``-ss`` position.

    Parameters:
        input_path:  Source video or audio file.
        output_path: Destination path for the sliced file.
        start:       Start timestamp (e.g. ``"00:01:30"`` or ``"90"``).
        end:         End timestamp (e.g. ``"00:05:00"`` or ``"300"``).

    Returns:
        The *output_path*.
    """
    if not start and not end:
        return input_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Compute duration from start/end so we can use -t (relative)
    start_sec = _parse_timestamp(start) if start else 0.0
    end_sec = _parse_timestamp(end) if end else None
    duration = (end_sec - start_sec) if end_sec is not None else None

    if duration is not None and duration <= 0:
        raise ValueError(
            f"End ({end}) must be after start ({start}): "
            f"computed duration={duration:.3f}s"
        )

    has_video = _has_video_stream(input_path)

    # -- Build ffmpeg command ---------------------------------------------
    # -ss before -i = fast seek to the nearest preceding keyframe.
    cmd: list[str] = ["ffmpeg", "-y"]
    if start:
        cmd += ["-ss", start]
    cmd += ["-i", input_path]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]

    if has_video:
        # Re-encode video for frame-accurate start (no frozen frames).
        from mazinger.subtitle import _has_nvenc
        if _has_nvenc():
            cmd += ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "18"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"]
        cmd += [
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
        ]
    else:
        # Audio-only: re-encode to ensure clean boundaries.
        ext = Path(output_path).suffix.lower()
        if ext == ".mp3":
            cmd += ["-c:a", "libmp3lame", "-q:a", "2"]
        elif ext in (".m4a", ".aac"):
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        elif ext == ".wav":
            cmd += ["-c:a", "pcm_s16le"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k"]

    # Reset timestamps so the output starts at 0.
    cmd += ["-avoid_negative_ts", "make_zero", "-map_metadata", "-1"]
    cmd.append(output_path)

    subprocess.run(cmd, check=True, capture_output=True)
    log.info(
        "Sliced %s -> %s (start=%s, end=%s, duration=%.3fs, re-encoded=%s)",
        input_path, output_path, start, end,
        duration if duration is not None else -1, has_video,
    )
    return output_path


def slice_project(proj, *, start: str | None = None, end: str | None = None) -> None:
    """Slice the video and/or audio of a project in-place.

    After slicing the video, the old audio is discarded and a fresh
    audio track is extracted from the sliced video to guarantee sync.
    """
    if not start and not end:
        return

    if os.path.exists(proj.video):
        orig = proj.video + ".orig"
        os.rename(proj.video, orig)
        try:
            slice_media(orig, proj.video, start=start, end=end)
        except Exception:
            # Restore original on failure.
            if not os.path.exists(proj.video):
                os.rename(orig, proj.video)
            raise
        else:
            os.remove(orig)
        # Always re-extract audio from the sliced video for consistency.
        if os.path.exists(proj.audio):
            os.remove(proj.audio)
        extract_audio(proj.video, proj.audio)
    elif os.path.exists(proj.audio):
        orig = proj.audio + ".orig"
        os.rename(proj.audio, orig)
        try:
            slice_media(orig, proj.audio, start=start, end=end)
        except Exception:
            if not os.path.exists(proj.audio):
                os.rename(orig, proj.audio)
            raise
        else:
            os.remove(orig)
