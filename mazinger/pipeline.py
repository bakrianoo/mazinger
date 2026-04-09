"""High-level pipeline that chains every stage into a single ``dub()`` call."""

from __future__ import annotations

import logging
import os
from typing import Any

from mazinger.paths import ProjectPaths
from mazinger.tts import DEFAULT_MLX_MODEL
from mazinger.utils import (
    save_json, load_json, get_audio_duration, LLMUsageTracker,
    is_valid_media_file, is_valid_srt_file, is_valid_json_file,
    is_valid_thumbs_meta,
)

log = logging.getLogger(__name__)


class MazingerDubber:
    """Orchestrates the full dubbing pipeline.

    Each step can also be called individually through the underlying modules
    (``mazinger.download``, ``mazinger.transcribe``, etc.).  This class
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
        base_dir: str = "./mazinger_output",
        llm_think: bool | None = None,
    ) -> None:
        self.llm_model = llm_model or os.environ.get("OPENAI_MODEL") or "gpt-4.1"
        self.base_dir = base_dir
        self._api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = openai_base_url or os.environ.get("OPENAI_BASE_URL")
        self._llm_think = llm_think

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _llm_client(self) -> Any:
        from mazinger.llm import build_client

        return build_client(
            api_key=self._api_key,
            base_url=self._base_url,
            think=self._llm_think,
        )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def dub(
        self,
        source: str,
        voice_sample: str | None = None,
        voice_script: str | None = None,
        *,
        voice_theme: str | None = None,
        slug: str | None = None,
        device: str = "cuda",
        transcribe_method: str = "faster-whisper",
        whisper_model: str | None = None,
        mlx_whisper_model: str = "mlx-community/whisper-large-v3-turbo",
        beam_size: int = 5,
        tts_model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        tts_dtype: str = "bfloat16",
        tts_language: str | None = None,
        tts_engine: str = "qwen",
        mlx_model: str = DEFAULT_MLX_MODEL,
        source_language: str = "auto",
        target_language: str = "English",
        chatterbox_model: str = "ResembleAI/chatterbox",
        chatterbox_exaggeration: float = 0.5,
        chatterbox_cfg: float = 0.5,
        pocket_voice: str | None = None,
        loudness_match: bool = True,
        mix_background: bool = True,
        background_volume: float = 0.15,
        cookies_from_browser: str | None = None,
        cookies: str | None = None,
        quality: str | None = None,
        start: str | None = None,
        end: str | None = None,
        skip_existing: bool = True,
        force_reset: bool = False,
        use_resegmented: bool = False,
        segment_mode: str = "short",
        min_segment_duration: float = 8.0,
        max_segment_duration: float = 30.0,
        tempo_mode: str = "auto",
        fixed_tempo: float | None = None,
        max_tempo: float = 1.5,
        words_per_second: float | None = None,
        duration_budget: float | None = None,
        translate_technical_terms: bool = False,
        asr_review: bool = False,
        keep_technical_english: bool = False,
        use_youtube_subs: bool = False,
        output_type: str = "audio",
        subtitle_style=None,
        subtitle_source: str = "translated",
    ) -> ProjectPaths:
        """Run the full pipeline: download/ingest, transcribe, translate, and dub.

        Parameters:
            source:         Video URL, local video path, or local audio path.
            voice_sample:   Path to the voice-cloning reference audio.  Can be
                            omitted when *voice_theme* is provided.
            voice_script:   Path to a text file containing the reference transcript,
                            **or** the transcript string itself.  Can be omitted
                            when *voice_theme* is provided.
            voice_theme:    Pre-defined voice theme name (e.g. ``narrator-m``).
                            Generates a reference voice via Qwen VoiceDesign in
                            the target language.  Mutually exclusive with
                            *voice_sample* / *voice_script*.
            slug:           Project slug override. Derived from the video title
                            or filename when ``None``.
            device:         Accelerator device (``cuda`` or ``cpu``).
            transcribe_method: Transcription backend: ``faster-whisper`` (default,
                            local GPU), ``openai`` (cloud API), or ``whisperx``
                            (requires [transcribe-whisperx] extra).
            whisper_model:  Whisper model name. Defaults to ``whisper-1`` for OpenAI,
                            ``large-v3`` for faster-whisper/WhisperX.
            tts_model_name: HuggingFace model identifier for TTS.
            tts_dtype:      Weight dtype for the TTS model.
            tts_language:   Language name passed to the TTS model.
            tts_engine:     TTS engine to use: ``qwen`` or ``chatterbox``.
            chatterbox_model: HuggingFace model identifier for Chatterbox.
            chatterbox_exaggeration: Exaggeration level for Chatterbox (0.0-1.0).
            chatterbox_cfg: CFG weight for Chatterbox (0.0-1.0).
            cookies_from_browser: yt-dlp ``--cookies-from-browser`` value.
            cookies:       Path to a Netscape cookie file for yt-dlp ``--cookies``.
            quality:       Video download quality: ``low`` (360p), ``medium``
                            (720p, default), ``high`` (best), or a numeric
                            resolution like ``"1080"``.
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
            subtitle_style: Optional :class:`~mazinger.subtitle.SubtitleStyle`
                            for burning subtitles into the video.  Implies
                            video output.
            subtitle_source: SRT to burn — ``'translated'`` (default),
                            ``'original'``, or a file path.

        Returns:
            The :class:`ProjectPaths` instance with all output paths populated.
        """
        from mazinger import download, transcribe, thumbnails, describe
        from mazinger import translate, resegment, tts, assemble
        from mazinger.srt import parse_file, parse_blocks

        if tts_language is None:
            tts_language = target_language

        device_for_tts = device.split(":")[0] + ":0" if ":" not in device else device

        # -- Resolve project paths ----------------------------------------
        is_remote = download.is_url(source)
        _yt_info: dict | None = None
        if slug is None:
            if is_remote:
                slug, _yt_info = download.resolve_slug(
                    source,
                    cookies_from_browser=cookies_from_browser,
                    cookies=cookies,
                )
            else:
                slug = download.slug_from_path(source)
        proj = ProjectPaths(slug, base_dir=self.base_dir, target_language=target_language).ensure_dirs()
        log.info("Project: %s  Language: %s", proj.root, target_language)

        # -- Save video metadata when available ---------------------------
        if _yt_info and not os.path.exists(proj.video_meta):
            download.save_video_meta(_yt_info, proj.video_meta)
        video_meta: dict | None = None
        if os.path.exists(proj.video_meta):
            video_meta = load_json(proj.video_meta)

        # -- Download YouTube subtitles (original + target language) ------
        if _yt_info and use_youtube_subs:
            download.download_youtube_subtitles(
                _yt_info,
                proj.youtube_subs_dir,
                target_languages=[target_language],
            )
        elif _yt_info:
            log.debug("YouTube subtitle download disabled (use_youtube_subs=False)")

        # -- Resolve voice (theme / profile / explicit sample+script) -----
        if voice_theme and not (voice_sample and voice_script):
            from mazinger.profiles import generate_profile
            profile_dir = proj.voice_profile_dir
            profile_wav = os.path.join(profile_dir, "voice.wav")
            if os.path.isfile(profile_wav):
                log.info("Reusing saved voice profile: %s", profile_dir)
                from mazinger.profiles import _load_local_profile
                voice_sample, voice_script = _load_local_profile(profile_dir)
            else:
                voice_sample, voice_script = generate_profile(
                    voice_theme, tts_language, profile_dir,
                    device=device_for_tts, dtype=tts_dtype,
                )

        auto_clone = not voice_sample and not voice_script

        if force_reset:
            skip_existing = False
            log.info("Force-reset enabled — all stages will re-run from scratch")

        client = self._llm_client()
        usage_tracker = LLMUsageTracker()

        # -- Read voice script (deferred when auto-cloning) ---------------
        ref_text = None
        if not auto_clone:
            if os.path.isfile(voice_script):
                with open(voice_script, encoding="utf-8") as fh:
                    ref_text = fh.read().strip()
            else:
                ref_text = voice_script.strip()

        # 1. Acquire source audio ----------------------------------------
        is_local_audio = not is_remote and download.is_audio_file(source)

        if is_local_audio:
            # Local audio file — copy into project, no video to process.
            if not (skip_existing and is_valid_media_file(proj.audio)):
                download.ingest_local_audio(source, proj.audio)
        elif is_remote:
            # URL — download video then extract audio.
            if skip_existing and is_valid_media_file(proj.video):
                log.info("Skipping download (video exists)")
            else:
                download.download_video(
                    source,
                    proj.video,
                    quality=quality,
                    cookies_from_browser=cookies_from_browser,
                    cookies=cookies,
                )
            download.extract_audio(proj.video, proj.audio)
        else:
            # Local video file — copy into project and extract audio.
            if not (skip_existing and is_valid_media_file(proj.video)):
                download.ingest_local_video(source, proj.video, proj.audio)
            else:
                download.extract_audio(proj.video, proj.audio)

        # 1b. Slice to time range ----------------------------------------
        if start or end:
            download.slice_project(proj, start=start, end=end)

        # 2. Transcribe --------------------------------------------------
        if skip_existing and is_valid_srt_file(proj.source_srt):
            log.info("Skipping transcription (SRT exists)")
        else:
            # Build an initial prompt from video metadata (title, tags, etc.)
            # to anchor Whisper's decoder on the expected vocabulary.
            _initial_prompt = transcribe.build_initial_prompt(video_meta)
            if _initial_prompt:
                log.info("Whisper initial prompt (from metadata): %.120s…", _initial_prompt)

            transcribe.transcribe(
                proj.audio, proj.source_srt,
                method=transcribe_method,
                model=whisper_model,
                mlx_whisper_model=mlx_whisper_model,
                device=device,
                beam_size=None if transcribe_method == "mlx-whisper" else beam_size,
                openai_api_key=self._api_key,
                openai_base_url=self._base_url,
                skip_resegment=not use_resegmented,
                initial_prompt=_initial_prompt,
            )

        transcribe.clear_cache()

        # 2b. Select best SRT source (ASR vs YouTube) --------------------
        source_srt_for_pipeline = proj.source_srt if use_resegmented else proj.source_raw_srt

        if use_youtube_subs:
            yt_orig_srt = None
            for fname in os.listdir(proj.youtube_subs_dir) if os.path.isdir(proj.youtube_subs_dir) else []:
                if fname.endswith("-orig.srt") or fname.endswith("-orig.manual.srt"):
                    yt_orig_srt = os.path.join(proj.youtube_subs_dir, fname)
                    break

            if yt_orig_srt and os.path.exists(source_srt_for_pipeline):
                from mazinger.review import select_srt
                with open(source_srt_for_pipeline, encoding="utf-8") as fh:
                    asr_text = fh.read()
                with open(yt_orig_srt, encoding="utf-8") as fh:
                    yt_text = fh.read()
                choice = select_srt(
                    asr_text, yt_text, client,
                    llm_model=self.llm_model,
                    video_meta=video_meta,
                    usage_tracker=usage_tracker,
                )
                if choice == "B":
                    source_srt_for_pipeline = yt_orig_srt
                    log.info("Using YouTube SRT as primary source: %s", yt_orig_srt)

        log.info("Using %s SRT for translation/dubbing: %s",
                 "resegmented" if use_resegmented else "raw",
                 source_srt_for_pipeline)
        with open(source_srt_for_pipeline, encoding="utf-8") as fh:
            source_srt_text = fh.read()

        # 3. Extract thumbnails ------------------------------------------
        has_video = os.path.exists(proj.video)

        if not has_video:
            log.info("No video available — skipping thumbnail extraction")
            thumb_paths = []
        elif skip_existing and is_valid_thumbs_meta(proj.thumbs_meta):
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
        if skip_existing and is_valid_json_file(proj.description, required_keys=("title", "summary")):
            log.info("Skipping description (file exists)")
            description = load_json(proj.description)
        elif not has_video:
            log.info("Skipping description (no video — thumbnails unavailable)")
            description = {"title": "", "summary": "", "keypoints": [], "keywords": []}
        else:
            description = describe.describe_content(
                source_srt_text, thumb_paths, client,
                llm_model=self.llm_model,
                video_meta=video_meta,
                usage_tracker=usage_tracker,
            )
            save_json(description, proj.description)

        # 4b. ASR review (optional) --------------------------------------
        if asr_review:
            if skip_existing and is_valid_srt_file(proj.reviewed_srt):
                log.info("Skipping ASR review (file exists)")
                with open(proj.reviewed_srt, encoding="utf-8") as fh:
                    source_srt_text = fh.read()
            else:
                from mazinger import review
                source_srt_text = review.review_srt(
                    source_srt_text, description, client,
                    llm_model=self.llm_model,
                    source_language=source_language,
                    keep_technical_english=keep_technical_english,
                    video_meta=video_meta,
                    usage_tracker=usage_tracker,
                )
                with open(proj.reviewed_srt, "w", encoding="utf-8") as fh:
                    fh.write(source_srt_text)

        # -- Auto-clone voice from source audio ---------------------------
        if auto_clone:
            from mazinger.profiles import create_auto_clone_profile
            profile_dir = proj.voice_profile_dir
            profile_wav = os.path.join(profile_dir, "voice.wav")
            if skip_existing and is_valid_media_file(profile_wav):
                log.info("Reusing auto-cloned voice profile: %s", profile_dir)
                voice_sample = profile_wav
            else:
                clone_srt = (
                    proj.reviewed_srt
                    if os.path.exists(proj.reviewed_srt)
                    else source_srt_for_pipeline
                )
                voice_sample = create_auto_clone_profile(
                    proj.audio, clone_srt, profile_dir,
                )

        # 5. Translate ---------------------------------------------------
        if skip_existing and is_valid_srt_file(proj.translated_raw_srt):
            log.info("Skipping translation (file exists)")
            with open(proj.translated_raw_srt, encoding="utf-8") as fh:
                translated_srt = fh.read()
        else:
            translated_srt = translate.translate_srt(
                source_srt_text, description, thumb_paths, client,
                llm_model=self.llm_model,
                source_language=source_language,
                target_language=target_language,
                translate_technical_terms=translate_technical_terms,
                video_meta=video_meta,
                usage_tracker=usage_tracker,
                **(dict(words_per_second=words_per_second) if words_per_second is not None else {}),
                **(dict(duration_budget=duration_budget) if duration_budget is not None else {}),
            )
            with open(proj.translated_raw_srt, "w", encoding="utf-8") as fh:
                fh.write(translated_srt)

        # 6. Re-segment --------------------------------------------------
        effective_mode = segment_mode
        if segment_mode == "auto":
            src_blocks = parse_blocks(translated_srt)
            durs = sorted(e - s for _, s, e, _ in src_blocks) if src_blocks else [5.0]
            median_dur = durs[len(durs) // 2]
            effective_mode = "long" if median_dur < 3.0 else "short"
            log.info("Auto segment mode: median=%.1fs -> %s", median_dur, effective_mode)

        if skip_existing and is_valid_srt_file(proj.final_srt):
            log.info("Skipping re-segmentation (file exists)")
        elif effective_mode == "long":
            resegmented = resegment.merge_long_segments(
                translated_srt,
                source_audio=proj.audio,
                min_duration=min_segment_duration,
                max_duration=max_segment_duration,
            )
            with open(proj.final_srt, "w", encoding="utf-8") as fh:
                fh.write(resegmented)
        else:
            resegmented = resegment.resegment_srt(
                translated_srt, client=client, llm_model=self.llm_model,
                usage_tracker=usage_tracker,
            )
            with open(proj.final_srt, "w", encoding="utf-8") as fh:
                fh.write(resegmented)

        if hasattr(client, 'unload_model'):
            client.unload_model(self.llm_model)

        # 7. TTS ---------------------------------------------------------
        # Use the resegmented SRT (merged phrases) so TTS doesn't produce
        # awkward gaps in the middle of a sentence.
        srt_entries = parse_file(proj.final_srt)
        original_duration = get_audio_duration(proj.audio)

        tts_model = tts.load_model(
            tts_model_name, device=device_for_tts,
            dtype=tts_dtype, engine=tts_engine,
            chatterbox_model=chatterbox_model,
            mlx_model=mlx_model,
        )
        # Pocket TTS: allow predefined voice name to override ref audio
        _ref_audio = voice_sample
        if tts_engine == "pocket" and pocket_voice:
            _ref_audio = pocket_voice
        voice_prompt = tts.create_voice_prompt(
            tts_model, _ref_audio, ref_text,
            engine=tts_engine,
            chatterbox_exaggeration=chatterbox_exaggeration,
            chatterbox_cfg=chatterbox_cfg,
            mlx_model=mlx_model,
        )
        segment_info = tts.synthesize_segments(
            tts_model, voice_prompt, srt_entries, proj.tts_segments_dir,
            language=tts_language,
            force_reset=force_reset,
        )
        tts.unload_model(voice_prompt, force=True)

        # 8. Assemble final audio ----------------------------------------
        assemble.assemble_timeline(
            segment_info, original_duration, proj.final_audio,
            tempo_mode=tempo_mode,
            fixed_tempo=fixed_tempo,
            max_tempo=max_tempo,
        )

        # 8b. Post-process: loudness + background -------------------------
        if loudness_match or mix_background:
            assemble.post_process(
                proj.final_audio, proj.audio, proj.final_audio,
                loudness_match=loudness_match,
                mix_background=mix_background,
                background_volume=background_volume,
            )

        drift = abs(get_audio_duration(proj.final_audio) - original_duration)
        log.info("Done. Final audio: %s (drift: %.3fs)", proj.final_audio, drift)

        # 9. Mux video / burn subtitles (optional) ----------------------
        produce_video = output_type == "video" or subtitle_style is not None
        if produce_video:
            if not os.path.exists(proj.video):
                log.warning("No source video available — skipping video output")
            elif subtitle_style:
                from mazinger.subtitle import burn_subtitles
                if subtitle_source == "translated":
                    # Prefer the pre-merged SRT for on-screen display;
                    # final_srt may contain long merged chunks unsuitable
                    # for readable subtitles.
                    srt_path = (
                        proj.translated_raw_srt
                        if os.path.exists(proj.translated_raw_srt)
                        else proj.final_srt
                    )
                elif subtitle_source == "original":
                    srt_path = proj.source_srt
                else:
                    srt_path = subtitle_source
                burn_subtitles(
                    proj.video, proj.final_video, srt_path,
                    subtitle_style, audio_path=proj.final_audio,
                )
            else:
                assemble.mux_video(proj.video, proj.final_audio, proj.final_video)

        # 10. LLM usage report -------------------------------------------
        if usage_tracker.records:
            log.info(usage_tracker.report())
            lang_root = os.path.dirname(proj.tts_dir)
            save_json(usage_tracker.records, os.path.join(lang_root, "llm_usage.json"))

        return proj
