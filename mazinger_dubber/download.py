"""Download a video from a URL, or ingest a local video/audio file."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yt_dlp

from mazinger_dubber.utils import sanitize_filename

log = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".ts", ".m2ts"}
_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus"}


def is_url(source: str) -> bool:
    """Return ``True`` if *source* looks like a URL rather than a local path."""
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https", "ftp", "ftps")


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


def download_video(
    url: str,
    output_path: str,
    *,
    cookies_from_browser: str | None = None,
    cookies: str | None = None,
) -> str:
    """Download the best-quality MP4 from *url* into *output_path*.

    Skips the download when *output_path* already exists.

    Returns:
        The resolved *output_path*.
    """
    if os.path.exists(output_path):
        log.info("Video already exists: %s", output_path)
        return output_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    opts = {
        # Prefer separate streams when available; fall back to a single best
        # stream if muxable formats are restricted/unavailable.
        "format": "bestvideo*+bestaudio/best",
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
