"""Structured directory and file-path management for a single video project."""

from __future__ import annotations

import os


class ProjectPaths:
    """Manages every file path for one video project.

    Directory layout under ``<base_dir>/projects/<slug>/``::

        source/          downloaded video + extracted audio
        transcription/   raw and processed SRT files
        subtitles/       final production-ready SRT
        thumbnails/      extracted frames + metadata
        analysis/        LLM-generated content description
        tts/             synthesised audio segments + final output
    """

    def __init__(self, slug: str, base_dir: str = ".") -> None:
        self.slug = slug
        self.base_dir = base_dir
        self.root = os.path.join(base_dir, "projects", slug)

        # Directories
        self.source_dir = os.path.join(self.root, "source")
        self.transcription_dir = os.path.join(self.root, "transcription")
        self.subtitles_dir = os.path.join(self.root, "subtitles")
        self.thumbnails_dir = os.path.join(self.root, "thumbnails")
        self.analysis_dir = os.path.join(self.root, "analysis")
        self.tts_dir = os.path.join(self.root, "tts")
        self.tts_segments_dir = os.path.join(self.tts_dir, "segments")

        # Common file paths
        self.video = os.path.join(self.source_dir, "video.mp4")
        self.audio = os.path.join(self.source_dir, "audio.mp3")
        self.source_srt = os.path.join(self.transcription_dir, "source.srt")
        self.source_raw_srt = os.path.join(self.transcription_dir, "source.raw.srt")
        self.translated_raw_srt = os.path.join(self.transcription_dir, "translated.raw.srt")
        self.final_srt = os.path.join(self.subtitles_dir, "translated.srt")
        self.thumbs_meta = os.path.join(self.thumbnails_dir, "meta.json")
        self.description = os.path.join(self.analysis_dir, "description.json")
        self.final_audio = os.path.join(self.tts_dir, "dubbed.wav")

    # ------------------------------------------------------------------

    def ensure_dirs(self) -> ProjectPaths:
        """Create all project sub-directories (idempotent)."""
        for d in (
            self.source_dir,
            self.transcription_dir,
            self.subtitles_dir,
            self.thumbnails_dir,
            self.analysis_dir,
            self.tts_dir,
            self.tts_segments_dir,
        ):
            os.makedirs(d, exist_ok=True)
        return self

    def summary(self) -> str:
        """Return a human-readable overview of which project files exist."""
        labels = {
            "source/video.mp4": self.video,
            "source/audio.mp3": self.audio,
            "transcription/source.srt": self.source_srt,
            "transcription/source.raw.srt": self.source_raw_srt,
            "transcription/translated.raw.srt": self.translated_raw_srt,
            "subtitles/translated.srt": self.final_srt,
            "thumbnails/meta.json": self.thumbs_meta,
            "analysis/description.json": self.description,
            "tts/dubbed.wav": self.final_audio,
        }
        lines = [f"Project: {self.slug}", f"  Root: {self.root}"]
        for label, path in labels.items():
            marker = "[x]" if os.path.exists(path) else "[ ]"
            lines.append(f"  {marker} {label}")
        return "\n".join(lines)
