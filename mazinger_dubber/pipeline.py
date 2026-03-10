"""High-level pipeline that chains every stage into a single ``dub()`` call."""

from __future__ import annotations

import logging
import os
from typing import Any

from mazinger_dubber.paths import ProjectPaths
from mazinger_dubber.utils import save_json, load_json, get_audio_duration, LLMUsageTracker

log = logging.getLogger(__name__)


class MazingerDubber:
    """Orchestrates the full dubbing pipeline.

    Each step can also be called individually through the underlying modules
    (``mazinger_dubber.download``, ``mazinger_dubber.transcribe``, etc.).  This class
    provides a convenient wrapper that chains them together with sensible
    defaults and shared state.

    Parameters:
        openai_api_key: API key for OpenAI (or set ``OPENAI_API_KEY`` env var).
        llm_model:      Model identifier used for translation and analysis tasks.
        base_dir:       Root directory under which project folders are created.
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        llm_model: str | None = None,
        base_dir: str = "./mazinger_dubber_output",
    ) -> None:
        self.llm_model = llm_model or os.environ.get("OPENAI_MODEL") or "gpt-4.1"
        self.base_dir = base_dir
        self._api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = openai_base_url or os.environ.get("OPENAI_BASE_URL")

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _openai_client(self) -> Any:
        from openai import OpenAI

        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return OpenAI(**kwargs)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def dub(
        self,
        source: str,
        voice_sample: str,
        voice_script: str,
        *,
        slug: str | None = None,
        device: str = "cuda",
        transcribe_method: str = "openai",
        whisper_model: str | None = None,
        tts_model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        tts_dtype: str = "bfloat16",
        tts_language: str = "English",
        tts_engine: str = "qwen",
        chatterbox_model: str = "ResembleAI/chatterbox",
        chatterbox_exaggeration: float = 0.5,
        chatterbox_cfg: float = 0.5,
        cookies_from_browser: str | None = None,
        cookies: str | None = None,
        skip_existing: bool = True,
        force_reset: bool = False,
        use_resegmented: bool = False,
        tempo_mode: str = "off",
        fixed_tempo: float | None = None,
        max_tempo: float = 1.3,
        words_per_second: float | None = None,
        duration_budget: float | None = None,
        output_type: str = "audio",
    ) -> ProjectPaths:
        """Run the full pipeline: download/ingest, transcribe, translate, and dub.

        Parameters:
            source:         Video URL, local video path, or local audio path.
            voice_sample:   Path to the voice-cloning reference audio.
            voice_script:   Path to a text file containing the reference transcript,
                            **or** the transcript string itself.
            slug:           Project slug override. Derived from the video title
                            or filename when ``None``.
            device:         Accelerator device (``cuda`` or ``cpu``).
            transcribe_method: Transcription backend: ``openai`` (default, cloud API)
                            or ``whisperx`` (local GPU, requires [transcribe] extra).
            whisper_model:  Whisper model name. Defaults to ``whisper-1`` for OpenAI,
                            ``large-v3`` for WhisperX.
            tts_model_name: HuggingFace model identifier for TTS.
            tts_dtype:      Weight dtype for the TTS model.
            tts_language:   Language name passed to the TTS model.
            tts_engine:     TTS engine to use: ``qwen`` or ``chatterbox``.
            chatterbox_model: HuggingFace model identifier for Chatterbox.
            chatterbox_exaggeration: Exaggeration level for Chatterbox (0.0-1.0).
            chatterbox_cfg: CFG weight for Chatterbox (0.0-1.0).
            cookies_from_browser: yt-dlp ``--cookies-from-browser`` value.
            cookies:       Path to a Netscape cookie file for yt-dlp ``--cookies``.
            skip_existing:  When ``True``, skip stages whose outputs already exist.
            force_reset:    When ``True``, discard all cached/intermediate outputs
                            and re-run every stage from scratch.  Overrides
                            *skip_existing*.
            use_resegmented: When ``True``, translate and dub from the resegmented
                            SRT (``source.srt``) instead of the raw output
                            (``source.raw.srt``).  The resegmented SRT has
                            shorter, more readable segments.
            output_type:    ``audio`` (default) — produce dubbed WAV only;
                            ``video`` — also mux the dubbed audio into the
                            source video (requires a video source).

        Returns:
            The :class:`ProjectPaths` instance with all output paths populated.
        """
        from mazinger_dubber import download, transcribe, thumbnails, describe
        from mazinger_dubber import translate, resegment, tts, assemble
        from mazinger_dubber.srt import parse_file

        # -- Resolve project paths ----------------------------------------
        is_remote = download.is_url(source)
        if slug is None:
            if is_remote:
                slug, _ = download.resolve_slug(
                    source,
                    cookies_from_browser=cookies_from_browser,
                    cookies=cookies,
                )
            else:
                slug = download.slug_from_path(source)
        proj = ProjectPaths(slug, base_dir=self.base_dir).ensure_dirs()
        log.info("Project: %s", proj.root)

        if force_reset:
            skip_existing = False
            log.info("Force-reset enabled — all stages will re-run from scratch")

        client = self._openai_client()
        usage_tracker = LLMUsageTracker()

        # -- Read voice script -------------------------------------------
        if os.path.isfile(voice_script):
            with open(voice_script, encoding="utf-8") as fh:
                ref_text = fh.read().strip()
        else:
            ref_text = voice_script.strip()

        # 1. Acquire source audio ----------------------------------------
        is_local_audio = not is_remote and download.is_audio_file(source)

        if is_local_audio:
            # Local audio file — copy into project, no video to process.
            if not (skip_existing and os.path.exists(proj.audio)):
                download.ingest_local_audio(source, proj.audio)
        elif is_remote:
            # URL — download video then extract audio.
            if skip_existing and os.path.exists(proj.video):
                log.info("Skipping download (video exists)")
            else:
                download.download_video(
                    source,
                    proj.video,
                    cookies_from_browser=cookies_from_browser,
                    cookies=cookies,
                )
            download.extract_audio(proj.video, proj.audio)
        else:
            # Local video file — copy into project and extract audio.
            if not (skip_existing and os.path.exists(proj.video)):
                download.ingest_local_video(source, proj.video, proj.audio)
            else:
                download.extract_audio(proj.video, proj.audio)

        # 2. Transcribe --------------------------------------------------
        if skip_existing and os.path.exists(proj.source_srt):
            log.info("Skipping transcription (SRT exists)")
        else:
            transcribe.transcribe(
                proj.audio, proj.source_srt,
                method=transcribe_method,
                model=whisper_model,
                device=device,
                openai_api_key=self._api_key,
                openai_base_url=self._base_url,
                skip_resegment=not use_resegmented,
            )

        # 3. Extract thumbnails ------------------------------------------
        source_srt_for_pipeline = proj.source_srt if use_resegmented else proj.source_raw_srt
        log.info("Using %s SRT for translation/dubbing: %s",
                 "resegmented" if use_resegmented else "raw",
                 source_srt_for_pipeline)
        with open(source_srt_for_pipeline, encoding="utf-8") as fh:
            source_srt_text = fh.read()

        has_video = os.path.exists(proj.video)

        if not has_video:
            log.info("No video available — skipping thumbnail extraction")
            thumb_paths = []
        elif skip_existing and os.path.exists(proj.thumbs_meta):
            log.info("Skipping thumbnails (metadata exists)")
            thumb_paths = load_json(proj.thumbs_meta)
        else:
            ts = thumbnails.select_timestamps(
                source_srt_text, client, llm_model=self.llm_model,
                usage_tracker=usage_tracker,
            )
            thumb_paths = thumbnails.extract_frames(
                proj.video, ts, proj.thumbnails_dir,
            )
            save_json(thumb_paths, proj.thumbs_meta)

        # 4. Describe content --------------------------------------------
        if skip_existing and os.path.exists(proj.description):
            log.info("Skipping description (file exists)")
            description = load_json(proj.description)
        elif not has_video:
            log.info("Skipping description (no video — thumbnails unavailable)")
            description = {"title": "", "summary": "", "keypoints": [], "keywords": []}
        else:
            description = describe.describe_content(
                source_srt_text, thumb_paths, client,
                llm_model=self.llm_model,
                usage_tracker=usage_tracker,
            )
            save_json(description, proj.description)

        # 5. Translate ---------------------------------------------------
        if skip_existing and os.path.exists(proj.translated_raw_srt):
            log.info("Skipping translation (file exists)")
            with open(proj.translated_raw_srt, encoding="utf-8") as fh:
                translated_srt = fh.read()
        else:
            translated_srt = translate.translate_srt(
                source_srt_text, description, thumb_paths, client,
                llm_model=self.llm_model,
                target_language=tts_language,
                usage_tracker=usage_tracker,
                **(dict(words_per_second=words_per_second) if words_per_second is not None else {}),
                **(dict(duration_budget=duration_budget) if duration_budget is not None else {}),
            )
            with open(proj.translated_raw_srt, "w", encoding="utf-8") as fh:
                fh.write(translated_srt)

        # 6. Re-segment --------------------------------------------------
        if skip_existing and os.path.exists(proj.final_srt):
            log.info("Skipping re-segmentation (file exists)")
        else:
            resegmented = resegment.resegment_srt(
                translated_srt, client=client, llm_model=self.llm_model,
                usage_tracker=usage_tracker,
            )
            with open(proj.final_srt, "w", encoding="utf-8") as fh:
                fh.write(resegmented)

        # 7. TTS ---------------------------------------------------------
        # Use the resegmented SRT (merged phrases) so TTS doesn't produce
        # awkward gaps in the middle of a sentence.
        srt_entries = parse_file(proj.final_srt)
        original_duration = get_audio_duration(proj.audio)

        device_for_tts = device.split(":")[0] + ":0" if ":" not in device else device
        tts_model = tts.load_model(
            tts_model_name, device=device_for_tts,
            dtype=tts_dtype, engine=tts_engine,
            chatterbox_model=chatterbox_model,
        )
        voice_prompt = tts.create_voice_prompt(
            tts_model, voice_sample, ref_text,
            engine=tts_engine,
            chatterbox_exaggeration=chatterbox_exaggeration,
            chatterbox_cfg=chatterbox_cfg,
        )
        segment_info = tts.synthesize_segments(
            tts_model, voice_prompt, srt_entries, proj.tts_segments_dir,
            language=tts_language,
            force_reset=force_reset,
        )
        tts.unload_model(voice_prompt)

        # 8. Assemble final audio ----------------------------------------
        assemble.assemble_timeline(
            segment_info, original_duration, proj.final_audio,
            tempo_mode=tempo_mode,
            fixed_tempo=fixed_tempo,
            max_tempo=max_tempo,
        )

        drift = abs(get_audio_duration(proj.final_audio) - original_duration)
        log.info("Done. Final audio: %s (drift: %.3fs)", proj.final_audio, drift)

        # 9. Mux video (optional) ----------------------------------------
        if output_type == "video":
            if os.path.exists(proj.video):
                assemble.mux_video(proj.video, proj.final_audio, proj.final_video)
            else:
                log.warning("No source video available — skipping video muxing")

        # 10. LLM usage report -------------------------------------------
        if usage_tracker.records:
            log.info(usage_tracker.report())
            save_json(usage_tracker.records, os.path.join(proj.root, "llm_usage.json"))

        return proj
