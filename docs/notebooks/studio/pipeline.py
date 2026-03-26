"""Pipeline runner for Mazinger Studio."""

import logging
import os
import shutil
import subprocess as sp
import threading
import time
import traceback

from constants import OLLAMA_DEFAULT_MODEL, QUALITY_MAP, METHOD_MAP, THEME_KEY_MAP
from helpers import LogCollector, ensure_ollama, detect_phase, check_ollama_health


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


def _resolve_source(source_type, url, uploaded_file):
    if source_type == "YouTube URL":
        if not url or not url.strip():
            return None, "❌ Please enter a video URL."
        return url.strip(), None
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


def run_dubbing(
    source_type, url, uploaded_file,
    cookies_text,
    target_language, voice_type, voice_theme_label, voice_preset,
    voice_file, voice_script_text,
    llm_provider, ollama_model, openai_key,
    api_base_url, llm_model,
    quality, start_time, end_time,
    transcribe_method, whisper_model,
    source_language, words_per_second, duration_budget, translate_technical,
    tts_dtype,
    tempo_mode, max_tempo, loudness_match, mix_background, background_volume,
    output_type, force_reset,
):
    """Generator → yields (status, logs, audio, srt_file, render_paths) tuples."""

    _empty = "", "", None, None, None

    is_ollama = (llm_provider == "Ollama (Local — Free)")

    if not is_ollama and (not openai_key or not openai_key.strip()):
        yield "❌ Please enter your OpenAI API key.", *_empty[1:]
        return

    source, err = _resolve_source(source_type, url, uploaded_file)
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

    # Ensure Ollama server + model are ready
    if is_ollama:
        yield "⏳ Checking Ollama server and model…", *_empty[1:]
        try:
            ensure_ollama(ollama_model.strip() if ollama_model else None)
        except Exception as exc:
            yield f"❌ Ollama setup failed: {exc}", *_empty[1:]
            return

    if output_type != "Dubbed Audio":
        yield from _run_subtitles(
            source, source_type, cookies_text,
            target_language, is_ollama, ollama_model, openai_key,
            api_base_url, llm_model,
            quality, start_time, end_time,
            transcribe_method, whisper_model,
            source_language, words_per_second, duration_budget, translate_technical,
            output_type, force_reset,
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
        tts_dtype,
        tempo_mode, max_tempo, loudness_match, mix_background, background_volume,
        force_reset,
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
    output_type, force_reset,
):
    """Generator → yields (status, logs, audio, srt_file, render_paths) tuples."""

    want_translation = (output_type == "Translated Subtitles")
    collector = LogCollector()
    maz_log = _setup_logging(collector)

    yield "⏳ Starting…", "", None, None, None

    result = {}
    error_box = {}
    done = threading.Event()

    def _worker():
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
            if is_remote:
                slug, _ = dl.resolve_slug(
                    source, **({"cookies": _cookies_path} if _cookies_path else {}),
                )
            else:
                slug = dl.slug_from_path(source)

            proj = ProjectPaths(
                slug, target_language=target_language,
            ).ensure_dirs()

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
                do_transcribe(
                    proj.audio, proj.source_srt,
                    method=m,
                    model=whisper_model if whisper_model and whisper_model.strip() else None,
                    device=device,
                    openai_api_key=_api_key,
                    openai_base_url=_base_url,
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
                translated = translate_srt(
                    srt_text, description, thumb_paths, client,
                    llm_model=_llm,
                    source_language=source_language if source_language != "Auto-detect" else "auto",
                    target_language=target_language,
                    translate_technical_terms=translate_technical,
                    **({"words_per_second": words_per_second} if words_per_second != 2.0 else {}),
                    **({"duration_budget": duration_budget} if duration_budget != 0.80 else {}),
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
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while not done.is_set():
        time.sleep(2)
        yield detect_phase(collector.read()), collector.read(), None, None, None

    maz_log.removeHandler(collector)

    if "error" in error_box:
        yield f"❌ Pipeline failed: {error_box['error']}", collector.read(), None, None, None
        return

    srt_out = result.get("srt")
    if not srt_out or not os.path.isfile(srt_out):
        yield "❌ No subtitle file produced.", collector.read(), None, None, None
        return

    render_paths = {}
    proj = result.get("paths")
    if proj:
        for attr in ("video", "final_srt", "source_srt"):
            p = getattr(proj, attr, None)
            if p and os.path.isfile(p):
                render_paths[attr] = p

    label = "Transcription" if not want_translation else "Translation"
    yield f"✅ {label} complete!\nSRT → {srt_out}", collector.read(), None, srt_out, render_paths


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
    tts_dtype,
    tempo_mode, max_tempo, loudness_match, mix_background, background_volume,
    force_reset,
):
    """Generator → yields (status, logs, audio, srt_file, render_paths) tuples."""

    collector = LogCollector()
    maz_log = _setup_logging(collector)

    yield ("⏳ Preparing voice profile…" if voice_type not in ("Voice Theme", "Auto-Clone")
           else "⏳ Voice theme selected — will generate on first run…" if voice_type == "Voice Theme"
           else "⏳ Auto-clone — voice will be extracted from source…"), "", None, None, None

    voice_sample_path = None
    voice_script_path = None
    voice_theme_key = None

    try:
        if voice_type == "Auto-Clone":
            pass  # pipeline handles it when both sample and script are None
        elif voice_type == "Voice Theme":
            voice_theme_key = THEME_KEY_MAP.get(voice_theme_label)
            if not voice_theme_key:
                yield "❌ Unknown voice theme selected.", "", None, None, None
                return
        elif voice_type == "Preset Voice":
            from mazinger.profiles import fetch_profile
            voice_sample_path, voice_script_path = fetch_profile(voice_preset)
        else:
            voice_sample_path = voice_file
            voice_script_path = voice_script_text.strip()
    except Exception as exc:
        maz_log.removeHandler(collector)
        yield f"❌ Voice profile error: {exc}", collector.read(), None, None, None
        return

    result = {}
    error_box = {}
    done = threading.Event()

    def _worker():
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
                tts_engine="qwen",
                tts_dtype=tts_dtype,
                tempo_mode=tempo_mode.lower(),
                max_tempo=max_tempo,
                loudness_match=loudness_match,
                mix_background=mix_background,
                background_volume=background_volume,
                translate_technical_terms=translate_technical,
                asr_review=True,
                **(dict(cookies=_cookies_path) if _cookies_path else {}),
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
            if words_per_second != 2.0:
                dub_kw["words_per_second"] = words_per_second
            if duration_budget != 0.80:
                dub_kw["duration_budget"] = duration_budget
            paths = dubber.dub(**dub_kw)
            result["paths"] = paths

        except Exception as exc:
            error_box["error"] = exc
            logging.getLogger("mazinger").error(
                "Pipeline failed: %s\n%s", exc, traceback.format_exc()
            )
        finally:
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    _poll_count = 0
    while not done.is_set():
        time.sleep(2)
        _poll_count += 1
        _log_snapshot = collector.read()
        _phase = detect_phase(_log_snapshot)

        if _poll_count % 5 == 0 and "LLM" in _phase:
            _ollama_warn = check_ollama_health()
            if _ollama_warn:
                _phase += _ollama_warn

        yield _phase, _log_snapshot, None, None, None

    maz_log.removeHandler(collector)

    if "error" in error_box:
        yield f"❌ Pipeline failed: {error_box['error']}", collector.read(), None, None, None
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
        for attr in ("video", "final_audio", "final_srt", "source_srt"):
            p = getattr(paths, attr, None)
            if p and os.path.isfile(p):
                render_paths[attr] = p

    status_parts = ["✅ Dubbing complete!"]
    if audio_out:
        status_parts.append(f"Audio → {audio_out}")

    yield "\n".join(status_parts), collector.read(), audio_out, None, render_paths


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
        srt_path = render_paths.get("final_srt")
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
