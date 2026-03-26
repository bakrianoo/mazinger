"""Structured directory and file-path management for a single video project."""

from __future__ import annotations

import os


class ProjectPaths:
    """Manages every file path for one video project.

    Directory layout under ``<base_dir>/projects/<slug>/``::

        source/              downloaded video + extracted audio   (shared)
        transcription/       raw and processed SRT files          (shared)
        thumbnails/          extracted frames + metadata          (shared)
        analysis/            LLM-generated content description    (shared)
        lang/<language>/     per-language dubbing results
            transcription/   translated SRT (raw)
            subtitles/       final production-ready SRT
            tts/             synthesised audio segments + final output
            voice_profile/   cloned / generated voice reference

    When *target_language* is ``None`` the language-scoped paths fall back
    to the project root (legacy flat layout).
    """

    def __init__(
        self,
        slug: str,
        base_dir: str = "./mazinger_output",
        target_language: str | None = None,
    ) -> None:
        self.slug = slug
        self.base_dir = base_dir
        self.target_language = target_language
        self.root = os.path.join(base_dir, "projects", slug)

        _lang = (
            os.path.join(self.root, "lang", target_language)
            if target_language else self.root
        )

        # Shared directories
        self.source_dir = os.path.join(self.root, "source")
        self.transcription_dir = os.path.join(self.root, "transcription")
        self.thumbnails_dir = os.path.join(self.root, "thumbnails")
        self.analysis_dir = os.path.join(self.root, "analysis")

        # Language-scoped directories
        self.subtitles_dir = os.path.join(_lang, "subtitles")
        self.tts_dir = os.path.join(_lang, "tts")
        self.tts_segments_dir = os.path.join(self.tts_dir, "segments")
        self.voice_profile_dir = os.path.join(_lang, "voice_profile")

        # Shared file paths
        self.video = os.path.join(self.source_dir, "video.mp4")
        self.audio = os.path.join(self.source_dir, "audio.mp3")
        self.source_srt = os.path.join(self.transcription_dir, "source.srt")
        self.source_raw_srt = os.path.join(self.transcription_dir, "source.raw.srt")
        self.thumbs_meta = os.path.join(self.thumbnails_dir, "meta.json")
        self.description = os.path.join(self.analysis_dir, "description.json")

        # Shared file paths (review is language-independent)
        self.reviewed_srt = os.path.join(self.transcription_dir, "source.reviewed.srt")

        # Language-scoped file paths
        self.translated_raw_srt = os.path.join(_lang, "transcription", "translated.raw.srt")
        self.final_srt = os.path.join(self.subtitles_dir, "translated.srt")
        self.final_audio = os.path.join(self.tts_dir, "dubbed.wav")
        self.final_video = os.path.join(self.tts_dir, "dubbed.mp4")

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
            os.path.dirname(self.translated_raw_srt),
        ):
            os.makedirs(d, exist_ok=True)
        return self

    def summary(self) -> str:
        """Return a human-readable overview of which project files exist."""
        def _rel(path: str) -> str:
            return os.path.relpath(path, self.root)

        labels = {
            "source/video.mp4": self.video,
            "source/audio.mp3": self.audio,
            "transcription/source.srt": self.source_srt,
            "transcription/source.raw.srt": self.source_raw_srt,
            _rel(self.translated_raw_srt): self.translated_raw_srt,
            _rel(self.final_srt): self.final_srt,
            "thumbnails/meta.json": self.thumbs_meta,
            "analysis/description.json": self.description,
            _rel(self.final_audio): self.final_audio,
            _rel(self.final_video): self.final_video,
            _rel(os.path.join(self.voice_profile_dir, "voice.wav")): os.path.join(self.voice_profile_dir, "voice.wav"),
            _rel(os.path.join(self.voice_profile_dir, "script.txt")): os.path.join(self.voice_profile_dir, "script.txt"),
        }
        lines = [f"Project: {self.slug}", f"  Root: {self.root}"]
        if self.target_language:
            lines.append(f"  Language: {self.target_language}")
        for label, path in labels.items():
            marker = "[x]" if os.path.exists(path) else "[ ]"
            lines.append(f"  {marker} {label}")
        return "\n".join(lines)
