"""Pipeline runner for Mazinger Studio."""

import logging
import os
import threading
import time
import traceback

from constants import OLLAMA_DEFAULT_MODEL, QUALITY_MAP, METHOD_MAP
from helpers import LogCollector, ensure_ollama, detect_phase, check_ollama_health


def run_dubbing(
    source_type, url, uploaded_file,
    cookies_text,
    target_language, voice_type, voice_preset, voice_file, voice_script_text,
    llm_provider, ollama_model, openai_key,
    api_base_url, llm_model,
    quality, start_time, end_time,
    transcribe_method, whisper_model,
    source_language, words_per_second, duration_budget, translate_technical,
    tts_engine, tts_dtype, chatterbox_exaggeration, chatterbox_cfg,
    tempo_mode, max_tempo, loudness_match, mix_background, background_volume,
    output_type, force_reset,
):
    """Generator → yields (status, logs, audio, video) tuples."""

    is_ollama = (llm_provider == "Ollama (Local — Free)")

    if not is_ollama and (not openai_key or not openai_key.strip()):
        yield "❌ Please enter your OpenAI API key.", "", None, None
        return

    # Ensure Ollama server + model are ready
    if is_ollama:
        yield "⏳ Checking Ollama server and model…", "", None, None
        try:
            ensure_ollama(ollama_model.strip() if ollama_model else None)
        except Exception as exc:
            yield f"❌ Ollama setup failed: {exc}", "", None, None
            return

    source = None
    if source_type == "YouTube URL":
        if not url or not url.strip():
            yield "❌ Please enter a video URL.", "", None, None
            return
        source = url.strip()
    else:
        if not uploaded_file:
            yield "❌ Please upload a video or audio file.", "", None, None
            return
        source = uploaded_file

    if voice_type == "Preset Voice":
        if not voice_preset:
            yield "❌ Please select a voice preset.", "", None, None
            return
    else:
        if not voice_file:
            yield "❌ Please upload a voice sample (10-30 sec audio clip).", "", None, None
            return
        if not voice_script_text or not voice_script_text.strip():
            yield "❌ Please enter the transcript of your voice sample.", "", None, None
            return

    collector = LogCollector()
    collector.setFormatter(logging.Formatter(
        "%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    ))
    maz_log = logging.getLogger("mazinger")
    maz_log.setLevel(logging.INFO)
    maz_log.addHandler(collector)

    yield "⏳ Preparing voice profile…", "", None, None

    try:
        if voice_type == "Preset Voice":
            from mazinger.profiles import fetch_profile
            voice_sample_path, voice_script_path = fetch_profile(voice_preset)
        else:
            voice_sample_path = voice_file
            voice_script_path = voice_script_text.strip()
    except Exception as exc:
        maz_log.removeHandler(collector)
        yield f"❌ Voice profile error: {exc}", collector.read(), None, None
        return

    result = {}
    error_box = {}
    done = threading.Event()

    def _worker():
        try:
            from mazinger import MazingerDubber

            # Resolve API credentials based on LLM provider choice
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

            # Write cookies to a temp file if provided
            _cookies_path = None
            if cookies_text and cookies_text.strip():
                import tempfile
                _cookies_path = os.path.join(tempfile.gettempdir(), "mazinger_cookies.txt")
                with open(_cookies_path, "w", encoding="utf-8") as _cf:
                    _cf.write(cookies_text.strip())

            dub_kw = dict(
                source=source,
                voice_sample=voice_sample_path,
                voice_script=voice_script_path,
                device=device,
                target_language=target_language,
                output_type=output_type.lower(),
                force_reset=force_reset,
                tts_engine=tts_engine.lower(),
                tts_dtype=tts_dtype,
                tempo_mode=tempo_mode.lower(),
                max_tempo=max_tempo,
                loudness_match=loudness_match,
                mix_background=mix_background,
                background_volume=background_volume,
                translate_technical_terms=translate_technical,
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
            # Ollama doesn't support OpenAI Whisper — force local transcription
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
            if tts_engine.lower() == "chatterbox":
                dub_kw["chatterbox_exaggeration"] = chatterbox_exaggeration
                dub_kw["chatterbox_cfg"] = chatterbox_cfg

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

        # Every 5th poll (~10 sec), check Ollama health during LLM stages
        if _poll_count % 5 == 0 and "LLM" in _phase:
            _ollama_warn = check_ollama_health()
            if _ollama_warn:
                _phase += _ollama_warn

        yield (
            _phase,
            _log_snapshot,
            None,
            None,
        )

    maz_log.removeHandler(collector)

    if "error" in error_box:
        yield (
            f"❌ Pipeline failed: {error_box['error']}",
            collector.read(),
            None,
            None,
        )
        return

    paths = result.get("paths")
    audio_out = None
    video_out = None

    if paths:
        if hasattr(paths, "final_audio") and os.path.isfile(paths.final_audio):
            audio_out = paths.final_audio
        if hasattr(paths, "final_video") and os.path.isfile(paths.final_video):
            video_out = paths.final_video

    status_parts = ["✅ Dubbing complete!"]
    if audio_out:
        status_parts.append(f"Audio → {audio_out}")
    if video_out:
        status_parts.append(f"Video → {video_out}")

    yield "\n".join(status_parts), collector.read(), audio_out, video_out
