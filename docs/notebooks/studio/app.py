"""Mazinger Studio — Gradio application entry point."""

import gradio as gr

from constants import (
    LANGUAGES, VOICE_PRESETS, METHOD_MAP, OLLAMA_DEFAULT_MODEL,
    THEME_CHOICES, VOICE_THEMES,
)
from theme import theme, CSS
from helpers import free_gpu_and_restart_ollama
from pipeline import run_dubbing, render_video


# ═══════════════════════════════════════════════════════════════════════
#  Build the Gradio interface
# ═══════════════════════════════════════════════════════════════════════

with gr.Blocks(theme=theme, title="Mazinger Studio", css=CSS) as app:

    # ── Header ────────────────────────────────────────────────────
    gr.Markdown(
        "# 🎬 Mazinger Studio\n"
        "Dub any video into another language with AI — paste a URL, pick a voice, and go.",
        elem_classes="app-header",
    )

    # ── LLM Provider ──────────────────────────────────────────────
    gr.Markdown("#### 🤖  LLM PROVIDER", elem_classes="section-title")
    with gr.Group(elem_classes="card-highlight"):
        llm_provider = gr.Radio(
            ["Ollama (Local — Free)", "OpenAI (Cloud)"],
            value="Ollama (Local — Free)",
            label="Translation engine",
            container=False,
        )

        with gr.Group(visible=True) as ollama_group:
            ollama_model = gr.Textbox(
                label="Ollama Model",
                value=OLLAMA_DEFAULT_MODEL,
                placeholder="e.g. qwen3.5:2b-q8_0, llama3.1:8b, …",
                info="Model will be pulled automatically on first run",
            )
            gr.Markdown(
                "✅ **No API key needed.** Runs 100% locally.  \n"
                "Transcription uses local Faster Whisper on your GPU.",
                elem_classes="ollama-info",
            )

        with gr.Group(visible=False) as openai_group:
            openai_key = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="sk-…",
                info="Required for transcription (Whisper) and translation (GPT)",
            )
            gr.Markdown(
                "Uses OpenAI Whisper for transcription and GPT for translation.",
                elem_classes="openai-info",
            )

    with gr.Row():
        gpu_btn = gr.Button(
            "🧹 Free GPU & Restart Ollama",
            variant="secondary",
            size="sm",
        )
        gpu_status = gr.Textbox(
            label="GPU Status",
            interactive=False,
            scale=3,
        )
    gpu_btn.click(fn=free_gpu_and_restart_ollama, inputs=[], outputs=[gpu_status])

    gr.HTML('<hr class="divider">')

    # ── Source ─────────────────────────────────────────────────────
    gr.Markdown("#### 📹  SOURCE", elem_classes="section-title")
    with gr.Group(elem_classes="card"):
        source_type = gr.Radio(
            ["YouTube URL", "Upload File"],
            value="YouTube URL",
            label="Source type",
            container=False,
        )
        url_input = gr.Textbox(
            label="Video URL",
            placeholder="https://www.youtube.com/watch?v=…",
            visible=True,
        )
        file_input = gr.File(
            label="Upload a video or audio file",
            file_types=[".mp4", ".mkv", ".webm", ".mov", ".mp3", ".wav", ".m4a"],
            visible=False,
        )

    def _toggle_source(choice):
        return (
            gr.update(visible=(choice == "YouTube URL")),
            gr.update(visible=(choice == "Upload File")),
        )
    source_type.change(_toggle_source, source_type, [url_input, file_input])

    # ── YouTube Cookies (collapsed) ───────────────────────────────
    _IMG_BASE = "https://raw.githubusercontent.com/bakrianoo/mazinger/refs/heads/master/docs/assets/yt-cache"
    _EXT_URL = "https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc"

    with gr.Accordion(
        "🍪  YouTube Cookies (only if downloads fail)",
        open=False,
    ):
        gr.Markdown(
            "Some YouTube videos require authentication to download. "
            "If you see a download error, paste your YouTube cookies below.\n\n"
            "*Don't know how? Click **How to get cookies** below.*",
            elem_classes="openai-info",
        )
        cookies_text = gr.Textbox(
            label="Cookies (Netscape format)",
            placeholder="# Netscape HTTP Cookie File\n# Paste your cookies here…",
            lines=4,
            max_lines=12,
        )
        with gr.Accordion("📖  How to get cookies", open=False):
            gr.HTML(
                '<div class="cookie-guide-step">'
                '<p><span class="cookie-step-num">1</span> '
                f'Install the <a href="{_EXT_URL}" target="_blank">'
                'Get cookies.txt locally</a> Chrome extension</p>'
                f'<img src="{_IMG_BASE}/p1.png" alt="Step 1: Install the Chrome extension" />'
                '</div>'
                '<div class="cookie-guide-step">'
                '<p><span class="cookie-step-num">2</span> '
                'Go to <a href="https://www.youtube.com" target="_blank">youtube.com</a>, '
                'make sure you are logged in, then click the extension icon</p>'
                f'<img src="{_IMG_BASE}/p2.png" alt="Step 2: Open extension on YouTube" />'
                '</div>'
                '<div class="cookie-guide-step">'
                '<p><span class="cookie-step-num">3</span> '
                'Click <strong>Copy</strong> to copy the cookies, '
                'then paste them in the text box above</p>'
                f'<img src="{_IMG_BASE}/p3.png" alt="Step 3: Copy cookies" />'
                '</div>'
            )

    # ── Voice & Language ──────────────────────────────────────────
    gr.Markdown("#### 🎤  VOICE & LANGUAGE", elem_classes="section-title")
    with gr.Group(elem_classes="card"):
        with gr.Row(equal_height=True):
            target_language = gr.Dropdown(
                choices=LANGUAGES,
                value="English",
                label="Output language",
                scale=1,
            )
            voice_type = gr.Radio(
                ["Voice Theme", "Preset Voice", "Custom Voice", "Auto-Clone"],
                value="Voice Theme",
                label="Voice source",
                scale=1,
            )

        # ── Voice Theme (default — easiest for non-technical users) ──
        with gr.Group(visible=True, elem_classes="voice-theme-group") as theme_group:
            gr.Markdown(
                "Pick a voice style — no files needed. "
                "A voice is generated automatically to match the theme.",
                elem_classes="openai-info",
            )
            with gr.Row(equal_height=True):
                # Build category buttons as a dropdown of grouped labels
                theme_category = gr.Radio(
                    choices=list(VOICE_THEMES.keys()),
                    value=list(VOICE_THEMES.keys())[0],
                    label="Category",
                    scale=1,
                )
                # Theme voices within the selected category
                _first_cat = list(VOICE_THEMES.keys())[0]
                _first_voices = list(VOICE_THEMES[_first_cat].keys())
                voice_theme = gr.Radio(
                    choices=_first_voices,
                    value=_first_voices[0],
                    label="Voice",
                    scale=1,
                )

            def _update_theme_voices(category):
                voices = list(VOICE_THEMES[category].keys())
                return gr.update(choices=voices, value=voices[0])

            theme_category.change(
                _update_theme_voices, theme_category, voice_theme,
            )

        # ── Preset Voice (HuggingFace profiles) ──
        with gr.Group(visible=False) as preset_group:
            voice_preset = gr.Dropdown(
                choices=VOICE_PRESETS,
                value=VOICE_PRESETS[0],
                allow_custom_value=True,
                label="Voice preset",
                info="Select a preset or type any profile name / local path",
            )

        # ── Custom Voice (upload your own) ──
        with gr.Group(visible=False) as custom_group:
            with gr.Row():
                voice_file = gr.Audio(
                    label="Reference audio (10-30 sec clip)",
                    type="filepath",
                    scale=1,
                )
                voice_script_text = gr.Textbox(
                    label="Transcript of the reference audio",
                    placeholder="Type the exact words spoken in your audio clip…",
                    lines=3,
                    scale=2,
                )

        # ── Auto-Clone (clone voice from source) ──
        with gr.Group(visible=False) as autoclone_group:
            gr.Markdown(
                "The speaker's voice is cloned directly from the source audio. "
                "No voice files or settings needed — the pipeline picks the "
                "best 20-60 s segment automatically.",
                elem_classes="openai-info",
            )

    def _toggle_voice(choice):
        return (
            gr.update(visible=(choice == "Voice Theme")),
            gr.update(visible=(choice == "Preset Voice")),
            gr.update(visible=(choice == "Custom Voice")),
            gr.update(visible=(choice == "Auto-Clone")),
        )
    voice_type.change(
        _toggle_voice, voice_type,
        [theme_group, preset_group, custom_group, autoclone_group],
    )

    # ── Advanced Settings ─────────────────────────────────────────
    with gr.Accordion("⚙️  Advanced Settings", open=True):
        with gr.Tabs():
            with gr.Tab("� Output"):
                output_type = gr.Radio(
                    ["Dubbed Audio", "Transcription Subtitles", "Translated Subtitles"],
                    value="Dubbed Audio",
                    label="What to produce",
                )
                force_reset = gr.Checkbox(
                    label="Force reset (discard cache, re-run all stages)",
                    value=False,
                )

            with gr.Tab("�🔌 API Override"):
                gr.Markdown(
                    "*Override the LLM provider settings above. "
                    "Leave empty to use defaults.*",
                    elem_classes="openai-info",
                )
                api_base_url = gr.Textbox(
                    label="API Base URL",
                    placeholder="Auto-set by LLM Provider selection",
                )
                llm_model = gr.Textbox(
                    label="LLM Model",
                    placeholder="Auto-set by LLM Provider selection",
                )

            with gr.Tab("📥 Download"):
                quality = gr.Dropdown(
                    ["Low (360p)", "Medium (720p)", "High (best)"],
                    value="Medium (720p)",
                    label="Video quality",
                )
                with gr.Row():
                    start_time = gr.Textbox(
                        label="Start time",
                        placeholder="00:01:30 or 90",
                    )
                    end_time = gr.Textbox(
                        label="End time",
                        placeholder="00:05:00 or 300",
                    )

            with gr.Tab("📝 Transcription"):
                transcribe_method = gr.Dropdown(
                    list(METHOD_MAP.keys()),
                    value="Faster Whisper (local GPU)",
                    label="Transcription method",
                    info="Cloud = needs OpenAI key  •  Local = faster, requires GPU",
                )
                whisper_model = gr.Textbox(
                    label="Model override",
                    placeholder="whisper-1 (cloud) / large-v3 (local)",
                )
                youtube_subs = gr.Checkbox(
                    label="Use YouTube subtitles",
                    value=False,
                    info="Download YouTube captions and compare with ASR to pick the best source",
                )

            with gr.Tab("🌐 Translation"):
                source_language = gr.Dropdown(
                    ["Auto-detect"] + LANGUAGES,
                    value="Auto-detect",
                    label="Source language",
                )
                with gr.Row():
                    words_per_second = gr.Slider(
                        0.0, 4.0, value=0.0, step=0.1,
                        label="Words per second",
                        info="0 = auto-estimate from source speech rate",
                    )
                    duration_budget = gr.Slider(
                        0.5, 1.0, value=0.85, step=0.05,
                        label="Duration budget",
                    )
                translate_technical = gr.Checkbox(
                    label="Translate technical terms",
                    value=False,
                )

            with gr.Tab("🗣️ TTS"):
                tts_engine = gr.Dropdown(
                    ["Qwen3-TTS"],
                    value="Qwen3-TTS",
                    label="TTS engine",
                )
                tts_dtype = gr.Dropdown(
                    ["bfloat16", "float16", "float32"],
                    value="bfloat16",
                    label="Model precision",
                    info="Weight dtype for Qwen3-TTS model",
                )

            with gr.Tab("🔊 Audio"):
                tempo_mode = gr.Dropdown(
                    ["Auto", "Off", "Dynamic", "Fixed"],
                    value="Auto",
                    label="Tempo mode",
                )
                max_tempo = gr.Slider(
                    1.0, 2.0, value=1.5, step=0.05,
                    label="Max tempo",
                )
                with gr.Row():
                    loudness_match = gr.Checkbox(
                        label="Match original loudness",
                        value=True,
                    )
                    mix_background = gr.Checkbox(
                        label="Mix background audio",
                        value=True,
                    )
                background_volume = gr.Slider(
                    0.0, 1.0, value=0.15, step=0.05,
                    label="Background volume",
                )

            with gr.Tab("📡 Streaming"):
                stream_llm = gr.Checkbox(
                    label="Stream LLM responses (live preview)",
                    value=False,
                    info="Show LLM output tokens in real-time in a separate log panel",
                )

    gr.HTML('<hr class="divider">')

    # ── Run Button ────────────────────────────────────────────────
    run_btn = gr.Button(
        "🎬  Start",
        variant="primary",
        size="lg",
        elem_classes="run-btn",
    )

    # ── Status & Logs ─────────────────────────────────────────────
    status = gr.Textbox(
        label="Status",
        interactive=False,
    )
    with gr.Accordion("📋 Pipeline Log", open=False):
        logs = gr.Textbox(
            label="Log output",
            lines=15,
            max_lines=40,
            interactive=False,
            autoscroll=True,
            elem_classes="log-box",
        )
    with gr.Accordion("📡 LLM Stream", open=False, visible=False) as llm_stream_section:
        llm_stream_box = gr.Textbox(
            label="LLM response (live)",
            lines=15,
            max_lines=60,
            interactive=False,
            autoscroll=True,
            elem_classes="log-box",
        )

    def _toggle_llm_stream_panel(enabled):
        return gr.update(visible=enabled, open=enabled)

    stream_llm.change(
        _toggle_llm_stream_panel, stream_llm, llm_stream_section,
    )

    # ── Results ───────────────────────────────────────────────────
    gr.Markdown("#### 📦  RESULTS", elem_classes="section-title")
    with gr.Group(elem_classes="results-card"):
        audio_output = gr.Audio(label="Dubbed Audio", type="filepath", visible=True)
        srt_output = gr.File(label="Subtitles (SRT)", visible=False)

    # ── Render Video ──────────────────────────────────────────────
    render_state = gr.State(value=None)

    with gr.Group(visible=False, elem_classes="render-card") as render_section:
        gr.Markdown(
            "#### 🎞️  RENDER VIDEO\n"
            "Combine your dubbed audio and subtitles into a downloadable video.",
            elem_classes="section-title",
        )

        with gr.Row(equal_height=True):
            render_dubbed = gr.Checkbox(label="Dubbed audio", value=True)
            render_orig_subs = gr.Checkbox(label="Original subtitles", value=False)
            render_trans_subs = gr.Checkbox(label="Translated subtitles", value=False)

        # Keep original / translated mutually exclusive
        def _exc_orig(val):
            return gr.update(value=False) if val else gr.update()
        def _exc_trans(val):
            return gr.update(value=False) if val else gr.update()
        render_orig_subs.change(_exc_orig, render_orig_subs, render_trans_subs)
        render_trans_subs.change(_exc_trans, render_trans_subs, render_orig_subs)

        with gr.Accordion("Subtitle style", open=False):
            with gr.Row(equal_height=True):
                sub_font_size = gr.Slider(
                    8, 32, value=14, step=1, label="Font size",
                )
                sub_position = gr.Dropdown(
                    ["Bottom", "Top", "Center"],
                    value="Bottom", label="Position",
                )
            with gr.Row(equal_height=True):
                sub_color = gr.Dropdown(
                    ["White", "Yellow", "Cyan"],
                    value="White", label="Font color",
                )
                sub_bg_alpha = gr.Slider(
                    0.0, 1.0, value=0.6, step=0.1, label="Background opacity",
                )

        render_btn = gr.Button(
            "🎬  Render Video",
            variant="primary",
            elem_classes="render-btn",
        )

        render_status = gr.Textbox(label="Render Status", interactive=False)
        with gr.Accordion("📋 Render Log", open=False):
            render_logs = gr.Textbox(
                label="Log output", lines=8, max_lines=20,
                interactive=False, autoscroll=True,
                elem_classes="log-box",
            )
        render_video_output = gr.Video(label="Rendered Video")

    def _show_render(paths):
        has_video = bool(paths and paths.get("video"))
        return gr.update(visible=has_video)

    render_btn.click(
        fn=render_video,
        inputs=[
            render_state,
            render_dubbed, render_orig_subs, render_trans_subs,
            sub_font_size, sub_position, sub_color, sub_bg_alpha,
        ],
        outputs=[render_status, render_logs, render_video_output],
    )

    # ── LLM provider toggle ───────────────────────────────────────
    # Auto-switches transcription method when LLM provider changes
    def _on_llm_provider_change(choice):
        is_ollama = (choice == "Ollama (Local — Free)")
        return (
            gr.update(visible=is_ollama),        # ollama_group
            gr.update(visible=not is_ollama),     # openai_group
            gr.update(value="Faster Whisper (local GPU)" if is_ollama
                      else "OpenAI Whisper (cloud)"),  # transcribe_method
        )
    llm_provider.change(
        _on_llm_provider_change, llm_provider,
        [ollama_group, openai_group, transcribe_method],
    )

    # ── Wire everything ───────────────────────────────────────────
    run_btn.click(
        fn=run_dubbing,
        inputs=[
            source_type, url_input, file_input,
            cookies_text,
            target_language, voice_type, voice_theme, voice_preset,
            voice_file, voice_script_text,
            llm_provider, ollama_model, openai_key,
            api_base_url, llm_model,
            quality, start_time, end_time,
            transcribe_method, whisper_model,
            source_language, words_per_second, duration_budget, translate_technical,
            tts_engine,
            tts_dtype,
            tempo_mode, max_tempo, loudness_match, mix_background, background_volume,
            output_type, force_reset,
            stream_llm,
            youtube_subs,
        ],
        outputs=[status, logs, llm_stream_box, audio_output, srt_output, render_state],
    ).then(
        fn=_show_render,
        inputs=[render_state],
        outputs=[render_section],
    )

    # Toggle result widgets based on output type selection
    def _on_output_type_change(choice):
        is_dub = (choice == "Dubbed Audio")
        return gr.update(visible=is_dub), gr.update(visible=not is_dub)
    output_type.change(
        _on_output_type_change, output_type,
        [audio_output, srt_output],
    )


# ═══════════════════════════════════════════════════════════════════════
#  Launch
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.launch(
        share=True,
        debug=True,
        show_error=True,
    )
