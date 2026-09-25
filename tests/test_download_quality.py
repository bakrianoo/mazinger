"""Guards for ``--quality`` selection in ``mazinger.download``.

"720p" names the short side of the frame.  The old ``[height<=720]`` filter
read it as the height, so a vertical Short (720x1280 available) came down at
360x640.  These tests run yt-dlp's real format selector over a synthetic
format list — no network — so they pin the behaviour, not just the options.
"""

import subprocess

import pytest
import yt_dlp

import mazinger.download as D


def _video(fmt_id, width, height, tbr):
    return {
        "format_id": fmt_id, "url": f"https://example.invalid/{fmt_id}",
        "ext": "mp4", "protocol": "https", "width": width, "height": height,
        "vcodec": "avc1", "acodec": "none", "tbr": tbr,
    }


_AUDIO = {
    "format_id": "a", "url": "https://example.invalid/a", "ext": "m4a",
    "protocol": "https", "vcodec": "none", "acodec": "mp4a", "tbr": 128,
}


def _select(formats, quality):
    """Return (width, height) of the video yt-dlp picks for *quality*."""
    opts = {**D._build_format_opts(D.resolve_quality(quality)), "quiet": True}
    info = {
        "id": "x", "title": "x", "extractor": "generic",
        "extractor_key": "Generic", "webpage_url": "https://example.invalid",
        "formats": [dict(f) for f in formats] + [dict(_AUDIO)],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        result = ydl.process_ie_result(info, download=False)
    video = next(f for f in result["requested_formats"] if f["vcodec"] != "none")
    return video["width"], video["height"]


LANDSCAPE = [
    _video("360", 640, 360, 300),
    _video("720", 1280, 720, 1000),
    _video("1080", 1920, 1080, 2000),
]

VERTICAL = [
    _video("360", 360, 640, 300),
    _video("720", 720, 1280, 1100),
    _video("1080", 1080, 1920, 3700),
]


@pytest.mark.parametrize("quality, expected", [
    ("low", (640, 360)),
    ("medium", (1280, 720)),
    ("1080", (1920, 1080)),
    ("high", (1920, 1080)),
])
def test_landscape_quality_caps_on_height(quality, expected):
    assert _select(LANDSCAPE, quality) == expected


@pytest.mark.parametrize("quality, expected", [
    ("low", (360, 640)),
    ("medium", (720, 1280)),   # was 360x640 under the height filter
    ("1080", (1080, 1920)),
    ("high", (1080, 1920)),
])
def test_vertical_quality_caps_on_short_side(quality, expected):
    assert _select(VERTICAL, quality) == expected


def test_cap_below_every_rendition_falls_back_to_the_smallest():
    # A download must always succeed: with nothing at or under the cap,
    # take the closest rendition above it rather than failing.
    assert _select(LANDSCAPE[1:], "low") == (1280, 720)


def test_high_quality_adds_no_sort_cap():
    assert D._build_format_opts(None) == {"format": "bestvideo*+bestaudio/best"}


# -- Resolution probe used for the "requested vs downloaded" warning -------


@pytest.mark.parametrize("stdout, expected", [
    ("1280,720\n", 720),
    ("720,1280\n", 720),
])
def test_probe_reports_short_side(monkeypatch, stdout, expected):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(D.subprocess, "run", fake_run)
    assert D._probe_video_resolution("video.mp4") == expected


def test_probe_returns_none_on_unparseable_output(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="\n", stderr="")

    monkeypatch.setattr(D.subprocess, "run", fake_run)
    assert D._probe_video_resolution("video.mp4") is None
