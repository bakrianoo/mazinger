"""Turn the Studio's settings into the equivalent ``mazinger`` CLI command.

:func:`build_cli_command` takes the same inputs, in the same order, as
:func:`mazinger.studio.pipeline.run_dubbing`, and returns a shell script plus
notes on what the command cannot carry over.  Nothing is run or written.

For a dub, the command reaches :meth:`MazingerDubber.dub` with the same
arguments the Studio passes (see ``tests/test_studio_cli_command.py``).  The
subtitle-only outputs have no single CLI equivalent: Studio chains the stages
itself, so those commands are marked approximate.
"""

from __future__ import annotations

import os
import shlex

from mazinger.studio.constants import (
    METHOD_MAP, QUALITY_MAP, SEGMENT_MODE_MAP, THEME_KEY_MAP,
)
from mazinger.studio.pipeline import _llm_settings, _resolve_sources

#: Where the Studio's dubs land: ``MazingerDubber``'s default, relative to the
#: directory Studio was started from.
STUDIO_BASE_DIR = "./mazinger_output"

#: Written in place of the real key, which never appears in the command.
KEY_PLACEHOLDER = "sk-..."
COOKIES_PLACEHOLDER = "cookies.txt"

_TTS_ENGINES = {"Qwen3-TTS": "qwen", "OmniVoice": "omnivoice"}
_OLLAMA_LABEL = "Ollama (Local — Free)"


def _translation_model() -> str:
    return os.environ.get("MAZINGER_TRANSLATION_MODEL") or "translategemma"


def format_command(argv: list[str]) -> str:
    """Quote *argv* for a POSIX shell, one flag and its value per line."""
    parts = [shlex.quote(a) for a in argv]
    lines, i = [" ".join(parts[:3])], 3  # mazinger <command> <source>
    while i < len(parts):
        takes_value = i + 1 < len(parts) and not argv[i + 1].startswith("--")
        lines.append(" ".join(parts[i:i + 2]) if takes_value else parts[i])
        i += 2 if takes_value else 1
    return " \\\n    ".join(lines)


def _llm_args(is_ollama, ollama_model, openai_key, api_base_url, llm_model) -> list[str]:
    _key, base_url, model = _llm_settings(
        is_ollama, ollama_model, openai_key or "", api_base_url, llm_model,
    )
    args = []
    if is_ollama:
        args += ["--openai-api-key", _key]  # "ollama" — not a secret
    if base_url:
        args += ["--openai-base-url", base_url]
    if model:
        args += ["--llm-model", model]
    return args


def build_cli_command(
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
    llm_instructions="",
    fit_check=True,
    *,
    base_dir: str = STUDIO_BASE_DIR,
) -> tuple[str, list[str]]:
    """Return ``(script, notes)`` for the Studio settings.

    *script* is empty when the settings would not start a run; *notes* then
    holds the reason.
    """
    sources, err = _resolve_sources(source_type, url, uploaded_file, local_path)
    if err:
        return "", [err]

    is_dub = output_type == "Dubbed Audio"
    if is_dub and voice_type == "Preset Voice" and not voice_preset:
        return "", ["❌ Please select a voice preset."]
    if is_dub and voice_type == "Custom Voice":
        if not voice_file:
            return "", ["❌ Please upload a voice sample (10-30 sec audio clip)."]
        if not voice_script_text or not voice_script_text.strip():
            return "", ["❌ Please enter the transcript of your voice sample."]

    is_ollama = llm_provider == _OLLAMA_LABEL
    method = METHOD_MAP.get(transcribe_method, "faster-whisper")
    if is_ollama and method == "openai":
        method = "faster-whisper"
    has_source_language = bool(source_language) and source_language != "Auto-detect"
    whisper = (whisper_model or "").strip()
    start, end = (start_time or "").strip(), (end_time or "").strip()

    common = ["--base-dir", os.path.abspath(base_dir)]
    q = QUALITY_MAP.get(quality)
    if q:
        common += ["--quality", q]
    if cookies_text and cookies_text.strip():
        common += ["--cookies", COOKIES_PLACEHOLDER]
    llm = _llm_args(is_ollama, ollama_model, openai_key, api_base_url, llm_model)

    notes: list[str] = []
    if is_dub:
        command = "dub"
        flags = _dub_flags(
            target_language, voice_type, voice_theme_label, voice_preset,
            voice_file, voice_script_text, is_ollama, method, whisper,
            source_language if has_source_language else None,
            start, end, words_per_second, duration_budget, translate_technical,
            use_translation_model, tts_engine, tts_dtype,
            tempo_mode, max_tempo, segment_mode, loudness_match, mix_background,
            background_volume, force_reset, youtube_subs,
            user_instructions, llm_instructions, fit_check,
        )
        flags = common + llm + flags
        if tempo_mode == "Fixed":
            notes.append(
                "**Fixed** tempo has no rate in Studio, so no tempo change is "
                "applied, the same as **Off**. Add `--fixed-tempo 1.1` (or "
                "another rate) to the command for a real fixed tempo."
            )
    else:
        command, flags = _subtitle_flags(
            output_type, method, whisper,
            source_language if has_source_language else None,
            target_language, words_per_second, duration_budget, translate_technical,
        )
        flags = common + llm + flags
        notes += _subtitle_notes(
            output_type, start, end, force_reset, use_translation_model,
            user_instructions, llm_instructions, is_ollama, has_source_language,
        )

    # Comments, not exports: pasting the script must not overwrite a key the
    # shell already has.
    header = []
    if not is_ollama:
        header.append(f'# Needs your OpenAI key: export OPENAI_API_KEY="{KEY_PLACEHOLDER}"')
    if method == "deepgram":
        header.append('# Needs your Deepgram key: export DEEPGRAM_API_KEY="..."')
    if command != "dub":
        header.append("# Approximate: Studio runs more stages than this command (see the notes).")
    blocks = ["\n".join(header)] if header else []
    blocks += [format_command(["mazinger", command, s, *flags]) for s in sources]
    script = "\n\n".join(blocks)

    notes += _general_notes(
        sources, source_type, voice_type if is_dub else None,
        cookies_text, is_ollama, ollama_model, use_translation_model and is_dub,
        method, base_dir,
    )
    return script, notes


def _dub_flags(
    target_language, voice_type, voice_theme_label, voice_preset,
    voice_file, voice_script_text, is_ollama, method, whisper,
    source_language, start, end, words_per_second, duration_budget,
    translate_technical, use_translation_model, tts_engine, tts_dtype,
    tempo_mode, max_tempo, segment_mode, loudness_match, mix_background,
    background_volume, force_reset, youtube_subs,
    user_instructions, llm_instructions, fit_check,
) -> list[str]:
    f = ["--target-language", target_language]
    if source_language:
        f += ["--source-language", source_language]

    if voice_type == "Voice Theme":
        f += ["--voice-theme", THEME_KEY_MAP.get(voice_theme_label, voice_theme_label)]
    elif voice_type == "Preset Voice":
        f += ["--clone-profile", voice_preset]
    elif voice_type == "Custom Voice":
        f += ["--voice-sample", voice_file, "--voice-script", voice_script_text.strip()]

    if is_ollama:
        f.append("--no-llm-think")
    if llm_instructions and llm_instructions.strip():
        f += ["--llm-instructions", llm_instructions]
    if user_instructions and user_instructions.strip():
        f += ["--user-instructions", user_instructions]

    if start:
        f += ["--start", start]
    if end:
        f += ["--end", end]

    f += ["--transcribe-method", method]
    if whisper:
        f += ["--whisper-model", whisper]
    if youtube_subs:
        f.append("--youtube-subs")
    f.append("--asr-review")

    if words_per_second > 0:
        f += ["--words-per-second", f"{words_per_second:g}"]
    if duration_budget != 0.85:
        f += ["--duration-budget", f"{duration_budget:g}"]
    if translate_technical:
        f.append("--translate-technical-terms")
    if use_translation_model:
        f += ["--translation-model", _translation_model()]

    seg = SEGMENT_MODE_MAP.get(segment_mode, "short")
    if seg != "short":
        f += ["--segment-mode", seg]

    f += ["--tts-engine", _TTS_ENGINES.get(tts_engine, "qwen")]
    if tts_dtype and tts_dtype != "bfloat16":
        f += ["--dtype", tts_dtype]

    f += ["--tempo-mode", tempo_mode.lower()]
    if max_tempo != 1.5:
        f += ["--max-tempo", f"{max_tempo:g}"]
    if not fit_check:
        f.append("--no-fit-check")
    if not loudness_match:
        f.append("--no-loudness-match")
    # The CLI's defaults for both depend on the tempo mode, so always spell
    # out what Studio chose.
    f.append("--mix-background" if mix_background else "--no-mix-background")
    f += ["--background-volume", f"{background_volume:g}"]

    if force_reset:
        f.append("--force-reset")
    return f


def _subtitle_flags(
    output_type, method, whisper, source_language,
    target_language, words_per_second, duration_budget, translate_technical,
) -> tuple[str, list[str]]:
    if output_type == "Transcription Subtitles":
        f = ["--method", method]
        if whisper:
            f += ["--model", whisper]
        if source_language:
            from mazinger.translate import lang_code_from_name
            code = lang_code_from_name(source_language)
            if code:
                f += ["--language", code]
        f.append("--asr-review")
        return "transcribe", f

    f = ["--target-language", target_language]
    if source_language:
        f += ["--source-language", source_language]
    f += ["--transcribe-method", method]
    if whisper:
        f += ["--whisper-model", whisper]
    if words_per_second > 0:
        f += ["--words-per-second", f"{words_per_second:g}"]
    if duration_budget != 0.85:
        f += ["--duration-budget", f"{duration_budget:g}"]
    if translate_technical:
        f.append("--translate-technical-terms")
    return "translate", f


def _subtitle_notes(
    output_type, start, end, force_reset, use_translation_model,
    user_instructions, llm_instructions, is_ollama, has_source_language,
) -> list[str]:
    if output_type == "Transcription Subtitles":
        notes = [
            "⚠️ **Approximate.** `mazinger transcribe` does not use the video's "
            "title and tags as a transcription hint, and it writes the reviewed "
            "transcript over `transcription/source.srt` (Studio keeps it in "
            "`source.reviewed.srt`)."
        ]
    else:
        notes = [
            "⚠️ **Approximate.** `mazinger translate` translates the transcript "
            "only. Studio also picks thumbnails, analyses the content, reviews "
            "the transcript and re-segments the translation, so its wording and "
            "line breaks will differ."
        ]
        if use_translation_model:
            notes.append("The dedicated translation model is not available "
                         "here; the main LLM translates.")
    missing = []
    if start or end:
        missing.append("start/end time")
    if force_reset:
        missing.append("force reset (delete the project folder instead)")
    if user_instructions and user_instructions.strip():
        missing.append("content & translation instructions")
    if llm_instructions and llm_instructions.strip():
        missing.append("extra LLM instructions")
    if is_ollama:
        missing.append("Ollama thinking off")
    if has_source_language and output_type != "Transcription Subtitles":
        missing.append("the source language as a transcription hint")
    if missing:
        notes.append("Not available for this command: " + ", ".join(missing) + ".")
    return notes


def _general_notes(
    sources, source_type, voice_type, cookies_text, is_ollama, ollama_model,
    translation_model, method, base_dir,
) -> list[str]:
    notes = []
    if len(sources) > 1:
        notes.append(f"{len(sources)} sources: one command each. Studio runs "
                     "them one after another; so does pasting the script.")
    if source_type == "Upload File":
        notes.append("Uploads are Gradio's temporary copies and are deleted "
                     "later; for a lasting command, point it at your own copy.")
    if voice_type == "Custom Voice":
        notes.append("The voice sample path is Gradio's temporary copy of your "
                     "upload; point `--voice-sample` at your own copy to keep it.")
    if cookies_text and cookies_text.strip():
        notes.append(f"Save the text of the **YouTube Cookies** box as "
                     f"`{COOKIES_PLACEHOLDER}` next to where you run the command.")
    if is_ollama:
        model = (ollama_model or "").strip() or "the model"
        pulls = f"`ollama pull {model}`"
        if translation_model:
            pulls += f" and `ollama pull {_translation_model()}`"
        notes.append("Studio installs and starts Ollama for you; the CLI "
                     f"expects the server to be running. Run {pulls} first.")
    else:
        notes.append("Set `OPENAI_API_KEY` to your key. The key is never "
                     "copied into the command.")
    if method == "coherex":
        notes.append("CohereX uses the Hugging Face sign-in saved on this "
                     "machine, or `HF_TOKEN`.")
    if method == "deepgram":
        notes.append("Deepgram reads `DEEPGRAM_API_KEY`.")
    notes.append(f"Output goes to `{os.path.abspath(base_dir)}`, the Studio's "
                 "folder, so stages the Studio already finished are reused. "
                 "LLM and TTS output varies from run to run, so the result "
                 "matches in settings, not byte for byte.")
    return notes
