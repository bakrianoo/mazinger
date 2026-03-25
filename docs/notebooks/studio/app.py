"""Mazinger Studio — Gradio application entry point."""

import gradio as gr

from constants import LANGUAGES, VOICE_PRESETS, METHOD_MAP, OLLAMA_DEFAULT_MODEL
from theme import theme, CSS
from pipeline import run_dubbing


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
                ["Preset Voice", "Custom Voice"],
                value="Preset Voice",
                label="Voice source",
                scale=1,
            )

        with gr.Group(visible=True) as preset_group:
            voice_preset = gr.Dropdown(
                choices=VOICE_PRESETS,
                value=VOICE_PRESETS[0],
                allow_custom_value=True,
                label="Voice preset",
                info="Select a preset or type any profile name from HuggingFace",
            )

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

    def _toggle_voice(choice):
        return (
            gr.update(visible=(choice == "Preset Voice")),
            gr.update(visible=(choice == "Custom Voice")),
        )
    voice_type.change(_toggle_voice, voice_type, [preset_group, custom_group])

    # ── Advanced Settings ─────────────────────────────────────────
    with gr.Accordion("⚙️  Advanced Settings", open=False):
        with gr.Tabs():
            with gr.Tab("🔌 API Override"):
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

            with gr.Tab("🌐 Translation"):
                source_language = gr.Dropdown(
                    ["Auto-detect"] + LANGUAGES,
                    value="Auto-detect",
                    label="Source language",
                )
                with gr.Row():
                    words_per_second = gr.Slider(
                        1.0, 4.0, value=2.0, step=0.1,
                        label="Words per second",
                    )
                    duration_budget = gr.Slider(
                        0.5, 1.0, value=0.80, step=0.05,
                        label="Duration budget",
                    )
                translate_technical = gr.Checkbox(
                    label="Translate technical terms",
                    value=False,
                )

            with gr.Tab("🗣️ TTS"):
                tts_engine = gr.Dropdown(
                    ["Qwen", "Chatterbox"],
                    value="Qwen",
                    label="TTS engine",
                )
                tts_dtype = gr.Dropdown(
                    ["bfloat16", "float16", "float32"],
                    value="bfloat16",
                    label="Model precision",
                )
                with gr.Group(visible=False) as chatterbox_group:
                    with gr.Row():
                        chatterbox_exaggeration = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.05,
                            label="Exaggeration",
                        )
                        chatterbox_cfg = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.05,
                            label="CFG weight",
                        )

                def _toggle_cb(engine):
                    return gr.update(visible=(engine == "Chatterbox"))
                tts_engine.change(_toggle_cb, tts_engine, chatterbox_group)

            with gr.Tab("🔊 Audio"):
                tempo_mode = gr.Dropdown(
                    ["Auto", "Off", "Dynamic", "Fixed"],
                    value="Auto",
                    label="Tempo mode",
                )
                max_tempo = gr.Slider(
                    1.0, 2.0, value=1.3, step=0.05,
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

            with gr.Tab("📦 Output"):
                output_type = gr.Dropdown(
                    ["Audio", "Video"],
                    value="Audio",
                    label="Output type",
                    info="Audio = dubbed WAV  •  Video = mux into source",
                )
                force_reset = gr.Checkbox(
                    label="Force reset (discard cache, re-run all stages)",
                    value=False,
                )

    gr.HTML('<hr class="divider">')

    # ── Run Button ────────────────────────────────────────────────
    run_btn = gr.Button(
        "🎬  Start Dubbing",
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
            elem_classes="log-box",
        )

    # ── Results ───────────────────────────────────────────────────
    gr.Markdown("#### 📦  RESULTS", elem_classes="section-title")
    with gr.Group(elem_classes="results-card"):
        with gr.Row():
            audio_output = gr.Audio(label="Dubbed Audio", type="filepath")
            video_output = gr.Video(label="Dubbed Video")

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
            target_language, voice_type, voice_preset, voice_file, voice_script_text,
            llm_provider, ollama_model, openai_key,
            api_base_url, llm_model,
            quality, start_time, end_time,
            transcribe_method, whisper_model,
            source_language, words_per_second, duration_budget, translate_technical,
            tts_engine, tts_dtype, chatterbox_exaggeration, chatterbox_cfg,
            tempo_mode, max_tempo, loudness_match, mix_background, background_volume,
            output_type, force_reset,
        ],
        outputs=[status, logs, audio_output, video_output],
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
