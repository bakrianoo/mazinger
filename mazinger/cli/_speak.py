"""mazinger speak — synthesise dubbed audio from SRT (voice cloning + assembly)."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import (
    DEFAULT_MLX_MODEL,
    add_common, add_llm, add_source, add_tempo, add_transcription,
    add_translation, add_tts_engine, add_voice, ensure_transcription,
    make_llm_client, require_voice, resolve_project,
    tempo_mode_from_args,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("speak", help="Synthesise dubbed audio from SRT.")
    add_source(p)
    p.add_argument("--srt", default=None, help="Path to translated SRT (overrides auto-translate).")
    p.add_argument("--original-audio", default=None, help="Original audio for duration matching.")
    add_voice(p)
    p.add_argument("-o", "--output", default=None, help="Output dubbed WAV path.")
    p.add_argument("--segments-dir", default=None, help="Directory for segment WAVs.")
    add_tts_engine(p)
    p.add_argument("--device", default="auto", help="Device: auto (default), cuda, or cpu.")
    p.add_argument("--dtype", default="bfloat16", help="Weight dtype (for Qwen engine).")
    add_tempo(p)
    add_llm(p)
    add_transcription(p)
    add_translation(p)
    p.add_argument("--force-reset", action="store_true",
                   help="Delete existing TTS segment files and re-synthesise from scratch.")
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    import os
    import sys
    from mazinger import tts, assemble
    from mazinger.srt import parse_file
    from mazinger.utils import get_audio_duration
    from mazinger.cli._groups import resolve_device

    args.device = resolve_device(args.device)
    proj = resolve_project(args)

    srt_path = args.srt
    if not srt_path and proj:
        # Auto-transcribe + translate if needed
        ensure_transcription(proj, args)
        if not os.path.exists(proj.final_srt):
            from mazinger.translate import translate_srt
            from mazinger.utils import load_json
            client = make_llm_client(args)
            with open(proj.source_srt, encoding="utf-8") as fh:
                srt_text = fh.read()
            description = load_json(proj.description) if os.path.exists(proj.description) else {}
            thumb_paths = load_json(proj.thumbs_meta) if os.path.exists(proj.thumbs_meta) else []
            translated = translate_srt(
                srt_text, description, thumb_paths, client, llm_model=args.llm_model,
                source_language=args.source_language,
                target_language=args.target_language,
                **(dict(words_per_second=args.words_per_second) if args.words_per_second is not None else {}),
                **(dict(duration_budget=args.duration_budget) if args.duration_budget is not None else {}),
            )
            os.makedirs(os.path.dirname(proj.final_srt) or ".", exist_ok=True)
            with open(proj.final_srt, "w", encoding="utf-8") as fh:
                fh.write(translated)
        srt_path = proj.final_srt
    if not srt_path:
        sys.exit("Error: provide a source (positional) or --srt.")

    original_audio = args.original_audio or (proj.audio if proj else None)
    if not original_audio:
        sys.exit("Error: provide a source (positional) or --original-audio.")

    output = args.output or (proj.final_audio if proj else None)
    if not output:
        sys.exit("Error: provide a source (positional) or -o/--output.")

    segments_dir = args.segments_dir or (proj.tts_segments_dir if proj else "./tts_segments")

    srt_entries = parse_file(srt_path)
    original_duration = get_audio_duration(original_audio)

    voice_theme = getattr(args, "voice_theme", None)
    if voice_theme and proj:
        from mazinger.profiles import generate_profile, _load_local_profile
        from mazinger.tts import validate_language
        profile_dir = proj.voice_profile_dir
        profile_wav = os.path.join(profile_dir, "voice.wav")
        language = getattr(args, "tts_language", None) or getattr(args, "target_language", "English")
        validate_language(language)
        device = args.device.split(":")[0] + ":0" if ":" not in args.device else args.device
        if os.path.isfile(profile_wav):
            voice_sample, voice_script = _load_local_profile(profile_dir)
        else:
            voice_sample, voice_script = generate_profile(
                voice_theme, language, profile_dir,
                device=device, dtype=args.dtype,
            )
    else:
        voice_sample, voice_script = require_voice(args)

    if os.path.isfile(voice_script):
        with open(voice_script, encoding="utf-8") as fh:
            ref_text = fh.read().strip()
    else:
        ref_text = voice_script.strip()

    engine = args.tts_engine
    model = tts.load_model(
        args.tts_model, device=args.device, dtype=args.dtype, engine=engine,
        chatterbox_model=args.chatterbox_model,
        mlx_model=args.mlx_tts_model,
        omnivoice_model=args.omnivoice_model,
    )
    voice_prompt = tts.create_voice_prompt(
        model, voice_sample, ref_text,
        engine=engine,
        chatterbox_exaggeration=args.chatterbox_exaggeration,
        chatterbox_cfg=args.chatterbox_cfg,
        mlx_model=args.mlx_tts_model,
        omnivoice_model=args.omnivoice_model,
    )
    segment_info = tts.synthesize_segments(
        model, voice_prompt, srt_entries, segments_dir,
        language=args.tts_language or "English",
        force_reset=args.force_reset,
    )
    tts.unload_model(voice_prompt)

    assemble.assemble_timeline(
        segment_info, original_duration, output,
        tempo_mode=tempo_mode_from_args(args),
        fixed_tempo=args.fixed_tempo,
        max_tempo=args.max_tempo,
    )
    print(f"Dubbed audio saved: {output}")
