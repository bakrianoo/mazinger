"""Pipeline runner for Mazinger Studio."""

import logging
import os
import shutil
import subprocess as sp
import threading
import time
import traceback

from mazinger.ollama_setup import DEFAULT_TRANSLATION_MODEL, OllamaSetupError
from mazinger.studio.constants import OLLAMA_DEFAULT_MODEL, QUALITY_MAP, METHOD_MAP, THEME_KEY_MAP
from mazinger.studio.helpers import LogCollector, LLMStreamCollector, ensure_ollama, detect_phase, check_ollama_health


# ═══════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def _setup_logging(collector):
    collector.setFormatter(logging.Formatter(
        "%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    ))
    maz_log = logging.getLogger("mazinger")
    maz_log.setLevel(logging.INFO)
    maz_log.addHandler(collector)
    return maz_log


def _prepare_ollama(ollama_model, extra_models, empty):
    """Install/start Ollama and pull the models, streaming live status.

    Runs in a worker thread so the UI keeps updating during a first-run
    install or a multi-minute model download. Returns ``True`` on success;
    on failure it yields the (actionable) error and returns ``False``.
    """
    messages = ["⏳ Checking Ollama server and model…"]
    error_box = {}
    done = threading.Event()

    def _worker():
        try:
            ensure_ollama(
                ollama_model.strip() if ollama_model else None,
                extra_models=extra_models,
                progress=lambda msg: messages.append(f"⏳ {msg}"),
            )
        except Exception as exc:  # noqa: BLE001 — surfaced in the UI below
            error_box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=_worker, daemon=True).start()

    yield messages[-1], *empty[1:]
    while not done.wait(1.5):
        yield messages[-1], *empty[1:]

    exc = error_box.get("error")
    if exc is None:
        return True

    logging.getLogger("mazinger").error("Ollama setup failed: %s", exc)
    prefix = "❌ " if isinstance(exc, OllamaSetupError) else "❌ Ollama setup failed: "
    yield f"{prefix}{exc}", *empty[1:]
    return False


def _resolve_source(source_type, url, uploaded_file, local_path=None):
    if source_type == "YouTube URL":
        if not url or not url.strip():
            return None, "❌ Please enter a video URL."
        return url.strip(), None
    if source_type == "Local Path":
        path = (local_path or "").strip()
        if not path:
            return None, "❌ Please enter a local file path."
        if not os.path.isfile(path):
            return None, f"❌ File not found: {path}"
        from mazinger.download import is_audio_file, is_video_file
        if not is_audio_file(path) and not is_video_file(path):
            return None, (
                f"❌ Unsupported file type: {os.path.splitext(path)[1] or '(no extension)'}. "
                "Supported: mp4 mkv avi mov webm flv wmv ts m2ts mp3 wav flac aac ogg m4a wma opus"
            )
        return path, None
    if not uploaded_file:
        return None, "❌ Please upload a video or audio file."
    return uploaded_file, None


def _resolve_llm(is_ollama, ollama_model, openai_key, api_base_url, llm_model):
    if is_ollama:
        _api_key = "ollama"
        _base_url = "http://localhost:11434/v1"
        _llm = (ollama_model.strip()
                if ollama_model and ollama_model.strip()
                else OLLAMA_DEFAULT_MODEL)
    else:
        _api_key = openai_key.strip()
        _base_url = (api_base_url.strip()
                     if api_base_url and api_base_url.strip() else None)
        _llm = (llm_model.strip()
                if llm_model and llm_model.strip() else None)
    os.environ["OPENAI_API_KEY"] = _api_key
    return _api_key, _base_url, _llm


def _write_cookies(cookies_text):
    if cookies_text and cookies_text.strip():
        import tempfile
        path = os.path.join(tempfile.gettempdir(), "mazinger_cookies.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(cookies_text.strip())
        return path
    return None


# Friendly message shown when yt-dlp asks for cookies. Points the user to
# the in-app "🍪 YouTube Cookies" accordion and the step-by-step guide.
_COOKIES_HELP_URL = (
    "https://github.com/bakrianoo/mazinger/blob/master/docs/youtube-cookies.md"
)
_COOKIES_FRIENDLY_MSG = (
    "🍪 YouTube needs your cookies to download this video.\n"
    "\n"
    "This usually means the video is age-restricted, region-locked, "
    "members-only, or YouTube is asking the downloader to prove it isn't a bot.\n"
    "\n"
    "How to fix it:\n"
    "  1. Open the “🍪 YouTube Cookies” panel above the language selector.\n"
    "  2. Click “📖 How to get cookies” and follow the 3 steps "
    "(install the Chrome extension → export cookies from youtube.com → paste them).\n"
    "  3. Click Start again.\n"
    "\n"
    f"Detailed guide: {_COOKIES_HELP_URL}"
)


def _format_pipeline_error(exc: BaseException, prefix: str = "Pipeline failed") -> str:
    """Render *exc* as a user-friendly status message.

    Detects yt-dlp's "cookies required" family of errors and replaces the
    raw stack-trace text with a clear instruction pointing the user to
    the YouTube Cookies panel.
    """
    try:
        from mazinger.download import is_cookies_required_error
    except Exception:  # noqa: BLE001 — fail open: keep the raw error
        is_cookies_required_error = lambda _e: False  # noqa: E731

    if is_cookies_required_error(exc):
        return "❌ " + _COOKIES_FRIENDLY_MSG
    return f"❌ {prefix}: {exc}"


def run_dubbing(
    source_type, url, uploaded_file, local_path,
    cookies_text,
    target_language, voice_type, voice_theme_label, voice_preset,
    voice_file, voice_script_text,
    llm_provider, ollama_model, openai_key,
    api_base_url, llm_model,
    quality, start_time, end_time,
    transcribe_method, whisper_model,
    source_language, words_per_second, duration_budget, translate_technical,
    use_translation_model,
    tts_engine,
    tts_dtype,
    tempo_mode, max_tempo, segment_mode, loudness_match, mix_background, background_volume,
    output_type, force_reset,
    stream_llm,
    youtube_subs=False,
    user_instructions="",
):
    """Generator → yields (status, logs, llm_stream, audio, srt_file, render_paths) tuples."""

    _empty = "", "", "", None, None, None

    is_ollama = (llm_provider == "Ollama (Local — Free)")

    if not is_ollama and (not openai_key or not openai_key.strip()):
        yield "❌ Please enter your OpenAI API key.", *_empty[1:]
        return

    source, err = _resolve_source(source_type, url, uploaded_file, local_path)
    if err:
        yield err, *_empty[1:]
        return

    if output_type == "Dubbed Audio":
        # Voice validation only needed for dubbing (skip for Auto-Clone)
        if voice_type == "Preset Voice" and not voice_preset:
            yield "❌ Please select a voice preset.", *_empty[1:]
            return
        if voice_type == "Custom Voice":
            if not voice_file:
                yield "❌ Please upload a voice sample (10-30 sec audio clip).", *_empty[1:]
                return
            if not voice_script_text or not voice_script_text.strip():
                yield "❌ Please enter the transcript of your voice sample.", *_empty[1:]
                return

    # Ensure Ollama is installed, running, and holds every model we need
    if is_ollama:
        _extra_models = []
        if use_translation_model:
            _extra_models.append(
                os.environ.get("MAZINGER_TRANSLATION_MODEL")
                or DEFAULT_TRANSLATION_MODEL
            )
        if not (yield from _prepare_ollama(ollama_model, _extra_models, _empty)):
            return

    if output_type != "Dubbed Audio":
        yield from _run_subtitles(
            source, source_type, cookies_text,
            target_language, is_ollama, ollama_model, openai_key,
            api_base_url, llm_model,
            quality, start_time, end_time,
            transcribe_method, whisper_model,
            source_language, words_per_second, duration_budget, translate_technical,
            use_translation_model,
            output_type, force_reset,
            stream_llm,
            youtube_subs,
            user_instructions=user_instructions,
        )
        return

    yield from _run_full_dub(
        source, source_type, cookies_text,
        target_language, voice_type, voice_theme_label, voice_preset,
        voice_file, voice_script_text,
        is_ollama, ollama_model, openai_key,
        api_base_url, llm_model,
        quality, start_time, end_time,
        transcribe_method, whisper_model,
        source_language, words_per_second, duration_budget, translate_technical,
        use_translation_model,
        tts_engine,
        tts_dtype,
        tempo_mode, max_tempo, segment_mode, loudness_match, mix_background, background_volume,
        force_reset,
        stream_llm,
        youtube_subs,
        user_instructions=user_instructions,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Subtitle-only pipeline (transcription or translation)
# ═══════════════════════════════════════════════════════════════════════

def _run_subtitles(
    source, source_type, cookies_text,
    target_language, is_ollama, ollama_model, openai_key,
    api_base_url, llm_model,
    quality, start_time, end_time,
    transcribe_method, whisper_model,
    source_language, words_per_second, duration_budget, translate_technical,
    use_translation_model,
    output_type, force_reset,
    stream_llm,
    youtube_subs=False,
    user_instructions="",
):
    """Generator → yields (status, logs, llm_stream, audio, srt_file, render_paths) tuples."""

    want_translation = (output_type == "Translated Subtitles")
    collector = LogCollector()
    maz_log = _setup_logging(collector)
    stream_collector = LLMStreamCollector() if stream_llm else None

    yield "⏳ Starting…", "", "", None, None, None

    result = {}
    error_box = {}
    done = threading.Event()

    def _worker():
        if stream_collector:
            from mazinger.llm import set_stream_callback
            set_stream_callback(stream_collector)
        try:
            from mazinger import ProjectPaths
            from mazinger import download as dl
            from mazinger.transcribe import transcribe as do_transcribe

            _api_key, _base_url, _llm = _resolve_llm(
                is_ollama, ollama_model, openai_key, api_base_url, llm_model,
            )
            _cookies_path = _write_cookies(cookies_text)

            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

            # Resolve slug
            is_remote = source_type == "YouTube URL"
            _yt_info = None
            if is_remote:
                slug, _yt_info = dl.resolve_slug(
                    source, **({"cookies": _cookies_path} if _cookies_path else {}),
                )
            else:
                slug = dl.slug_from_path(source)

            proj = ProjectPaths(
                slug, target_language=target_language,
            ).ensure_dirs()

            # Save video metadata when available
            if _yt_info and not os.path.exists(proj.video_meta):
                dl.save_video_meta(_yt_info, proj.video_meta)

            skip = not force_reset

            # 1. Download / ingest
            is_audio = not is_remote and dl.is_audio_file(source)
            if is_audio:
                if not (skip and os.path.exists(proj.audio)):
                    dl.ingest_local_audio(source, proj.audio)
            elif is_remote:
                if not (skip and os.path.exists(proj.video)):
                    q = QUALITY_MAP.get(quality)
                    dl.download_video(
                        source, proj.video,
                        **({"quality": q} if q else {}),
                        **({"cookies": _cookies_path} if _cookies_path else {}),
                    )
                dl.extract_audio(proj.video, proj.audio)
            else:
                if not (skip and os.path.exists(proj.video)):
                    dl.ingest_local_video(source, proj.video, proj.audio)
                else:
                    dl.extract_audio(proj.video, proj.audio)

            if start_time and start_time.strip() or end_time and end_time.strip():
                dl.slice_project(
                    proj,
                    start=start_time.strip() if start_time else None,
                    end=end_time.strip() if end_time else None,
                )

            # 2. Transcribe
            if not (skip and os.path.exists(proj.source_srt)):
                m = METHOD_MAP.get(transcribe_method, "faster-whisper")
                if is_ollama and m == "openai":
                    m = "faster-whisper"

                # Build initial prompt from video metadata (title, tags…)
                from mazinger.transcribe import build_initial_prompt
                from mazinger.utils import load_json as _load_json
                _video_meta = _load_json(proj.video_meta) if os.path.exists(proj.video_meta) else None
                _initial_prompt = build_initial_prompt(_video_meta)

                # Honour an explicit source language at the ASR stage — CohereX
                # cannot detect it, and the Whisper backends benefit too.
                _asr_lang = None
                if source_language and source_language != "Auto-detect":
                    from mazinger.translate import lang_code_from_name
                    _asr_lang = lang_code_from_name(source_language)

                do_transcribe(
                    proj.audio, proj.source_srt,
                    method=m,
                    model=whisper_model if whisper_model and whisper_model.strip() else None,
                    device=device,
                    language=_asr_lang,
                    openai_api_key=_api_key,
                    openai_base_url=_base_url,
                    initial_prompt=_initial_prompt,
                )

            # Build LLM client (needed for ASR review and translation)
            from mazinger.llm import build_client
            from mazinger.describe import describe_content
            from mazinger.utils import load_json, save_json

            init_kw = {"api_key": _api_key}
            if _base_url:
                init_kw["base_url"] = _base_url
            if is_ollama:
                init_kw["think"] = False
            client = build_client(**init_kw)

            if not want_translation:
                # ASR review: describe content then refine transcript
                from mazinger.review import review_srt

                with open(proj.source_srt, encoding="utf-8") as f:
                    srt_text = f.read()

                if skip and os.path.exists(proj.description):
                    description = load_json(proj.description)
                else:
                    description = describe_content(
                        srt_text, [], client, llm_model=_llm,
                    )
                    save_json(description, proj.description)

                if not (skip and os.path.exists(proj.reviewed_srt)):
                    reviewed = review_srt(
                        srt_text, description, client,
                        llm_model=_llm,
                        source_language=source_language if source_language != "Auto-detect" else "auto",
                    )
                    with open(proj.reviewed_srt, "w", encoding="utf-8") as f:
                        f.write(reviewed)

                result["srt"] = proj.reviewed_srt
                result["paths"] = proj
                return

            # 3-6. Thumbnails → Describe → Translate → Resegment
            from mazinger.thumbnails import select_timestamps, extract_frames
            from mazinger.translate import translate_srt
            from mazinger.resegment import resegment_srt
            from mazinger.review import review_srt

            with open(proj.source_raw_srt, encoding="utf-8") as f:
                srt_text = f.read()

            # 3. Thumbnails
            has_video = os.path.exists(proj.video)
            thumb_paths = []
            if has_video:
                if skip and os.path.exists(proj.thumbs_meta):
                    thumb_paths = load_json(proj.thumbs_meta)
                else:
                    ts = select_timestamps(srt_text, client, llm_model=_llm)
                    thumb_paths = extract_frames(
                        proj.video, ts, proj.thumbnails_dir,
                    )
                    save_json(thumb_paths, proj.thumbs_meta)

            # 4. Describe
            if skip and os.path.exists(proj.description):
                description = load_json(proj.description)
            elif not has_video:
                description = {"title": "", "summary": "", "keypoints": [], "keywords": []}
            else:
                description = describe_content(
                    srt_text, thumb_paths, client, llm_model=_llm,
                    user_instructions=user_instructions,
                )
                save_json(description, proj.description)

            # 4b. ASR review
            if skip and os.path.exists(proj.reviewed_srt):
                with open(proj.reviewed_srt, encoding="utf-8") as f:
                    srt_text = f.read()
            else:
                srt_text = review_srt(
                    srt_text, description, client,
                    llm_model=_llm,
                    source_language=source_language if source_language != "Auto-detect" else "auto",
                )
                with open(proj.reviewed_srt, "w", encoding="utf-8") as f:
                    f.write(srt_text)

            # 5. Translate
            if not (skip and os.path.exists(proj.translated_raw_srt)):
                if use_translation_model:
                    from mazinger.translate import (
                        translate_srt_simple, lang_name_from_code,
                    )
                    detected_src = (
                        source_language if source_language and source_language != "Auto-detect"
                        else "auto"
                    )
                    if detected_src == "auto" and os.path.exists(proj.source_lang):
                        try:
                            with open(proj.source_lang, encoding="utf-8") as f:
                                code = f.read().strip()
                            name = lang_name_from_code(code)
                            if name:
                                detected_src = name
                        except OSError:
                            pass
                    translated = translate_srt_simple(
                        srt_text, client,
                        llm_model=os.environ.get("MAZINGER_TRANSLATION_MODEL")
                        or "translategemma",
                        source_language=detected_src,
                        target_language=target_language,
                    )
                else:
                    translated = translate_srt(
                        srt_text, description, thumb_paths, client,
                        llm_model=_llm,
                        source_language=source_language if source_language != "Auto-detect" else "auto",
                        target_language=target_language,
                        translate_technical_terms=translate_technical,
                        user_instructions=user_instructions,
                        **({"words_per_second": words_per_second} if words_per_second > 0 else {}),
                        **({"duration_budget": duration_budget} if duration_budget != 0.85 else {}),
                    )
                with open(proj.translated_raw_srt, "w", encoding="utf-8") as f:
                    f.write(translated)
            else:
                with open(proj.translated_raw_srt, encoding="utf-8") as f:
                    translated = f.read()

            # 6. Resegment
            if not (skip and os.path.exists(proj.final_srt)):
                final = resegment_srt(translated, client=client, llm_model=_llm)
                with open(proj.final_srt, "w", encoding="utf-8") as f:
                    f.write(final)

            result["srt"] = proj.final_srt
            result["paths"] = proj

        except Exception as exc:
            error_box["error"] = exc
            logging.getLogger("mazinger").error(
                "Pipeline failed: %s\n%s", exc, traceback.format_exc(),
            )
        finally:
            if stream_collector:
                from mazinger.llm import clear_stream_callback
                clear_stream_callback()
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    _llm_text = lambda: stream_collector.read() if stream_collector else ""

    while not done.is_set():
        time.sleep(2)
        yield detect_phase(collector.read()), collector.read(), _llm_text(), None, None, None

    maz_log.removeHandler(collector)

    if "error" in error_box:
        yield _format_pipeline_error(error_box["error"]), collector.read(), _llm_text(), None, None, None
        return

    srt_out = result.get("srt")
    if not srt_out or not os.path.isfile(srt_out):
        yield "❌ No subtitle file produced.", collector.read(), _llm_text(), None, None, None
        return

    render_paths = {}
    proj = result.get("paths")
    if proj:
        for attr in ("video", "final_srt", "source_srt",
                      "translated_raw_srt"):
            p = getattr(proj, attr, None)
            if p and os.path.isfile(p):
                render_paths[attr] = p

    label = "Transcription" if not want_translation else "Translation"
    yield f"✅ {label} complete!\nSRT → {srt_out}", collector.read(), _llm_text(), None, srt_out, render_paths


# ═══════════════════════════════════════════════════════════════════════
#  Full dubbing pipeline
# ═══════════════════════════════════════════════════════════════════════

def _run_full_dub(
    source, source_type, cookies_text,
    target_language, voice_type, voice_theme_label, voice_preset,
    voice_file, voice_script_text,
    is_ollama, ollama_model, openai_key,
    api_base_url, llm_model,
    quality, start_time, end_time,
    transcribe_method, whisper_model,
    source_language, words_per_second, duration_budget, translate_technical,
    use_translation_model,
    tts_engine,
    tts_dtype,
    tempo_mode, max_tempo, segment_mode, loudness_match, mix_background, background_volume,
    force_reset,
    stream_llm,
    youtube_subs=False,
    user_instructions="",
):
    """Generator → yields (status, logs, llm_stream, audio, srt_file, render_paths) tuples."""

    _engine_map = {
        "Qwen3-TTS": "qwen",
        "OmniVoice": "omnivoice",
    }
    _tts_engine_key = _engine_map.get(tts_engine, "qwen")

    collector = LogCollector()
    maz_log = _setup_logging(collector)
    stream_collector = LLMStreamCollector() if stream_llm else None

    yield ("⏳ Preparing voice profile…" if voice_type not in ("Voice Theme", "Auto-Clone")
           else "⏳ Voice theme selected — will generate on first run…" if voice_type == "Voice Theme"
           else "⏳ Auto-clone — voice will be extracted from source…"), "", "", None, None, None

    voice_sample_path = None
    voice_script_path = None
    voice_theme_key = None

    try:
        if voice_type == "Auto-Clone":
            pass  # pipeline handles it when both sample and script are None
        elif voice_type == "Voice Theme":
            voice_theme_key = THEME_KEY_MAP.get(voice_theme_label)
            if not voice_theme_key:
                yield "❌ Unknown voice theme selected.", "", "", None, None, None
                return
        elif voice_type == "Preset Voice":
            from mazinger.profiles import fetch_profile
            voice_sample_path, voice_script_path = fetch_profile(voice_preset)
        else:
            voice_sample_path = voice_file
            voice_script_path = voice_script_text.strip()
    except Exception as exc:
        maz_log.removeHandler(collector)
        yield _format_pipeline_error(exc, prefix="Voice profile error"), collector.read(), "", None, None, None
        return

    result = {}
    error_box = {}
    done = threading.Event()

    def _worker():
        if stream_collector:
            from mazinger.llm import set_stream_callback
            set_stream_callback(stream_collector)
        try:
            from mazinger import MazingerDubber

            _api_key, _base_url, _llm = _resolve_llm(
                is_ollama, ollama_model, openai_key, api_base_url, llm_model,
            )
            _cookies_path = _write_cookies(cookies_text)

            init_kw = dict(openai_api_key=_api_key)
            if _base_url:
                init_kw["openai_base_url"] = _base_url
            if _llm:
                init_kw["llm_model"] = _llm
            if is_ollama:
                init_kw["llm_think"] = False

            dubber = MazingerDubber(**init_kw)

            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

            dub_kw = dict(
                source=source,
                voice_sample=voice_sample_path,
                voice_script=voice_script_path,
                voice_theme=voice_theme_key,
                device=device,
                target_language=target_language,
                output_type="audio",
                force_reset=force_reset,
                tts_engine=_tts_engine_key,
                tts_dtype=tts_dtype,
                tempo_mode=tempo_mode.lower(),
                max_tempo=max_tempo,
                loudness_match=loudness_match,
                mix_background=mix_background,
                background_volume=background_volume,
                translate_technical_terms=translate_technical,
                asr_review=True,
                use_youtube_subs=youtube_subs,
                **(dict(cookies=_cookies_path) if _cookies_path else {}),
                **(dict(user_instructions=user_instructions) if user_instructions and user_instructions.strip() else {}),
            )

            if source_language and source_language != "Auto-detect":
                dub_kw["source_language"] = source_language
            q = QUALITY_MAP.get(quality)
            if q:
                dub_kw["quality"] = q
            if start_time and start_time.strip():
                dub_kw["start"] = start_time.strip()
            if end_time and end_time.strip():
                dub_kw["end"] = end_time.strip()
            m = METHOD_MAP.get(transcribe_method)
            if is_ollama and m == "openai":
                m = "faster-whisper"
            if m:
                dub_kw["transcribe_method"] = m
            if whisper_model and whisper_model.strip():
                dub_kw["whisper_model"] = whisper_model.strip()
            if words_per_second > 0:
                dub_kw["words_per_second"] = words_per_second
            if duration_budget != 0.85:
                dub_kw["duration_budget"] = duration_budget

            from mazinger.studio.constants import SEGMENT_MODE_MAP
            _seg_mode = SEGMENT_MODE_MAP.get(segment_mode, "short")
            if _seg_mode != "short":
                dub_kw["segment_mode"] = _seg_mode

            if use_translation_model:
                dub_kw["translation_model"] = (
                    os.environ.get("MAZINGER_TRANSLATION_MODEL")
                    or "translategemma"
                )

            paths = dubber.dub(**dub_kw)
            result["paths"] = paths

        except Exception as exc:
            error_box["error"] = exc
            logging.getLogger("mazinger").error(
                "Pipeline failed: %s\n%s", exc, traceback.format_exc()
            )
        finally:
            if stream_collector:
                from mazinger.llm import clear_stream_callback
                clear_stream_callback()
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    _poll_count = 0
    _llm_text = lambda: stream_collector.read() if stream_collector else ""

    while not done.is_set():
        time.sleep(2)
        _poll_count += 1
        _log_snapshot = collector.read()
        _phase = detect_phase(_log_snapshot)

        if _poll_count % 5 == 0 and "LLM" in _phase:
            _ollama_warn = check_ollama_health()
            if _ollama_warn:
                _phase += _ollama_warn

        yield _phase, _log_snapshot, _llm_text(), None, None, None

    maz_log.removeHandler(collector)

    if "error" in error_box:
        yield _format_pipeline_error(error_box["error"]), collector.read(), _llm_text(), None, None, None
        return

    paths = result.get("paths")
    audio_out = None

    if paths:
        if hasattr(paths, "final_audio") and os.path.isfile(paths.final_audio):
            audio_out = paths.final_audio
            mp3_preview = paths.final_audio.rsplit(".", 1)[0] + ".mp3"
            try:
                sp.run(
                    ["ffmpeg", "-y", "-i", paths.final_audio,
                     "-codec:a", "libmp3lame", "-b:a", "192k", mp3_preview],
                    capture_output=True, check=True,
                )
                audio_out = mp3_preview
            except Exception:
                pass

    render_paths = {}
    if paths:
        for attr in ("video", "final_audio", "final_srt", "source_srt",
                      "translated_raw_srt"):
            p = getattr(paths, attr, None)
            if p and os.path.isfile(p):
                render_paths[attr] = p

    status_parts = ["✅ Dubbing complete!"]
    if audio_out:
        status_parts.append(f"Audio → {audio_out}")

    yield "\n".join(status_parts), collector.read(), _llm_text(), audio_out, None, render_paths


def render_video(
    render_paths,
    use_dubbed_audio, use_original_subs, use_translated_subs,
    sub_font_size, sub_position, sub_color, sub_bg_alpha,
):
    """Generator → yields (status, log, video_file) tuples."""

    if not render_paths:
        yield "❌ No dubbing result available. Run dubbing first.", "", None
        return

    video_path = render_paths.get("video")
    if not video_path or not os.path.isfile(video_path):
        yield "❌ Source video not found. Was the source audio-only?", "", None
        return

    audio_path = render_paths.get("final_audio") if use_dubbed_audio else None
    if use_dubbed_audio and (not audio_path or not os.path.isfile(audio_path)):
        yield "❌ Dubbed audio not found.", "", None
        return

    srt_path = None
    if use_translated_subs:
        # Prefer pre-merged SRT for readable on-screen subtitles;
        # final_srt may have long merged chunks from long-segment mode.
        srt_path = render_paths.get("translated_raw_srt") or render_paths.get("final_srt")
    elif use_original_subs:
        srt_path = render_paths.get("source_srt")

    if (use_translated_subs or use_original_subs) and (not srt_path or not os.path.isfile(srt_path)):
        yield "❌ Subtitle file not found.", "", None
        return

    if not use_dubbed_audio and not srt_path:
        yield "❌ Select at least one option (audio or subtitles).", "", None
        return

    yield "⏳ Rendering video…", "", None

    collector = LogCollector()
    collector.setFormatter(logging.Formatter(
        "%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    ))
    maz_log = logging.getLogger("mazinger")
    maz_log.setLevel(logging.INFO)
    maz_log.addHandler(collector)

    suffix_parts = []
    if use_dubbed_audio:
        suffix_parts.append("dubbed")
    if use_original_subs:
        suffix_parts.append("orig-subs")
    elif use_translated_subs:
        suffix_parts.append("trans-subs")
    suffix = "-".join(suffix_parts)

    out_dir = os.path.dirname(render_paths.get("final_audio", video_path))
    output_path = os.path.join(out_dir, f"render-{suffix}.mp4")

    error = None
    try:
        if srt_path:
            from mazinger.subtitle import SubtitleStyle, burn_subtitles

            _POSITION_MAP = {"Bottom": "bottom", "Top": "top", "Center": "center"}
            style = SubtitleStyle(
                font_size=int(sub_font_size),
                position=_POSITION_MAP.get(sub_position, "bottom"),
                font_color=sub_color.lower(),
                bg_alpha=sub_bg_alpha,
            )
            burn_subtitles(video_path, output_path, srt_path,
                           style=style, audio_path=audio_path)
        else:
            from mazinger.assemble import mux_video
            mux_video(video_path, audio_path, output_path)
    except Exception as exc:
        error = exc
        logging.getLogger("mazinger").error(
            "Render failed: %s\n%s", exc, traceback.format_exc()
        )
    finally:
        maz_log.removeHandler(collector)

    if error:
        yield f"❌ Render failed: {error}", collector.read(), None
        return

    if not os.path.isfile(output_path):
        yield "❌ Render produced no output.", collector.read(), None
        return

    yield "✅ Video ready!", collector.read(), output_path
