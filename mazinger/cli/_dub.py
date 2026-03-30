"""mazinger dub — full dubbing pipeline."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import (
    add_common, add_llm, add_segment_mode, add_slice, add_source,
    add_subtitles, add_tempo, add_tts_engine, add_transcription,
    add_translation, add_voice,
    resolve_voice, subtitle_style_from_args, tempo_mode_from_args,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("dub", help="Run the full dubbing pipeline.")
    add_source(p, required=True)
    add_voice(p)
    add_transcription(p)
    add_tts_engine(p)
    add_llm(p)
    add_slice(p)
    p.add_argument("--device", default="auto", help="Device: auto (default), cuda, or cpu.")
    p.add_argument("--use-resegmented", action="store_true",
                   help="Translate from the resegmented SRT instead of the raw transcript.")
    p.add_argument("--output-type", choices=["audio", "video"], default="audio",
                   help="Output type: 'audio' (default) or 'video'.")
    add_subtitles(p)
    add_tempo(p)
    add_segment_mode(p)
    add_translation(p)
    p.add_argument("--asr-review", action="store_true", default=False,
                   help="Review ASR transcript with LLM to fix typos and punctuation.")
    p.add_argument("--keep-technical-english", action="store_true", default=False,
                   help="Convert technical terms to English in the source transcript (requires --asr-review).")
    p.add_argument("--youtube-subs", action="store_true", default=False,
                   help="Download YouTube subtitles and compare with ASR to pick the best source.")
    p.add_argument("--no-loudness-match", action="store_true",
                   help="Skip loudness normalisation against the original audio.")
    p.add_argument("--mix-background", action="store_true", default=False,
                   help="Extract and mix background audio from the original.")
    p.add_argument("--background-volume", type=float, default=0.15,
                   help="Background audio mix level, 0.0-1.0 (default: 0.15).")
    p.add_argument("--force-reset", action="store_true",
                   help="Discard all cached outputs and re-run every stage.")
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    from mazinger.pipeline import MazingerDubber
    from mazinger.cli._groups import resolve_device

    args.device = resolve_device(args.device)
    voice_sample, voice_script = resolve_voice(args)
    voice_theme = getattr(args, "voice_theme", None)
    subtitle_style = subtitle_style_from_args(args) if args.embed_subtitles else None

    dubber = MazingerDubber(
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        llm_model=args.llm_model,
        base_dir=args.base_dir,
        llm_think=args.llm_think,
    )
    proj = dubber.dub(
        source=args.source,
        voice_sample=voice_sample,
        voice_script=voice_script,
        voice_theme=voice_theme,
        slug=args.slug,
        device=args.device,
        transcribe_method=args.transcribe_method,
        whisper_model=args.whisper_model,
        beam_size=args.beam_size,
        tts_model_name=args.tts_model,
        tts_language=args.tts_language,
        tts_engine=args.tts_engine,
        source_language=args.source_language,
        target_language=args.target_language,
        deepgram_api_key=args.deepgram_api_key,
        chatterbox_model=args.chatterbox_model,
        chatterbox_exaggeration=args.chatterbox_exaggeration,
        chatterbox_cfg=args.chatterbox_cfg,
        cookies_from_browser=args.cookies_from_browser,
        cookies=args.cookies,
        quality=args.quality,
        start=args.start,
        end=args.end,
        force_reset=args.force_reset,
        use_resegmented=args.use_resegmented,
        segment_mode=args.segment_mode,
        min_segment_duration=args.min_segment_duration,
        max_segment_duration=args.max_segment_duration,
        output_type=args.output_type,
        tempo_mode=tempo_mode_from_args(args),
        fixed_tempo=args.fixed_tempo,
        max_tempo=args.max_tempo,
        words_per_second=args.words_per_second,
        duration_budget=args.duration_budget,
        translate_technical_terms=args.translate_technical_terms,
        asr_review=args.asr_review,
        keep_technical_english=args.keep_technical_english,
        use_youtube_subs=args.youtube_subs,
        subtitle_style=subtitle_style,
        subtitle_source=args.subtitle_source,
        loudness_match=not args.no_loudness_match,
        mix_background=args.mix_background,
        background_volume=args.background_volume,
    )
    print(proj.summary())
