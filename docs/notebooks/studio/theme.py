"""Gradio theme and CSS for Mazinger Studio."""

import gradio as gr

theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#eff6ff", c100="#dbeafe", c200="#bfdbfe", c300="#93c5fd",
        c400="#60a5fa", c500="#3b82f6", c600="#2563eb", c700="#1d4ed8",
        c800="#1e40af", c900="#1e3a8a", c950="#172554",
    ),
    secondary_hue=gr.themes.Color(
        c50="#f5f3ff", c100="#ede9fe", c200="#ddd6fe", c300="#c4b5fd",
        c400="#a78bfa", c500="#8b5cf6", c600="#7c3aed", c700="#6d28d9",
        c800="#5b21b6", c900="#4c1d95", c950="#2e1065",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f8fafc", c100="#f1f5f9", c200="#e2e8f0", c300="#cbd5e1",
        c400="#94a3b8", c500="#64748b", c600="#475569", c700="#334155",
        c800="#1e293b", c900="#0f172a", c950="#020617",
    ),
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    body_text_color_subdued="#94a3b8",
    body_text_color_subdued_dark="#94a3b8",

    block_background_fill="#1e293b",
    block_background_fill_dark="#1e293b",
    block_border_color="#334155",
    block_border_color_dark="#334155",
    block_border_width="1px",
    block_label_text_color="#94a3b8",
    block_label_text_color_dark="#94a3b8",
    block_label_background_fill="#1e293b",
    block_label_background_fill_dark="#1e293b",
    block_title_text_color="#e2e8f0",
    block_title_text_color_dark="#e2e8f0",
    block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.3)",
    block_shadow_dark="0 4px 6px -1px rgba(0, 0, 0, 0.3)",

    input_background_fill="#0f172a",
    input_background_fill_dark="#0f172a",
    input_border_color="#334155",
    input_border_color_dark="#334155",
    input_placeholder_color="#475569",
    input_placeholder_color_dark="#475569",

    button_primary_background_fill="linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)",
    button_primary_text_color="#ffffff",
    button_primary_border_color="transparent",
    button_primary_border_color_dark="transparent",

    button_secondary_background_fill="#334155",
    button_secondary_background_fill_dark="#334155",
    button_secondary_text_color="#e2e8f0",
    button_secondary_border_color="#475569",

    border_color_primary="#3b82f6",
    border_color_primary_dark="#3b82f6",

    shadow_spread="0px",

    checkbox_background_color="#0f172a",
    checkbox_background_color_dark="#0f172a",
    checkbox_background_color_selected="#3b82f6",
    checkbox_background_color_selected_dark="#3b82f6",
    checkbox_border_color="#475569",
    checkbox_border_color_dark="#475569",
    checkbox_label_text_color="#e2e8f0",

    slider_color="#3b82f6",
    slider_color_dark="#3b82f6",
)


CSS = """
/* ── Global layout ──────────────────────────────────────────────── */
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    background: #0f172a !important;
}
footer { display: none !important; }

/* ── Header ─────────────────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 1.5rem 1rem 0.5rem;
}
.app-header h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem !important;
}
.app-header p {
    color: #94a3b8 !important;
    font-size: 1rem;
    margin-top: 0 !important;
}

/* ── Section headings ───────────────────────────────────────────── */
.section-title {
    color: #e2e8f0 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin: 0.5rem 0 0.1rem !important;
    padding: 0 !important;
}

/* ── Cards ──────────────────────────────────────────────────────── */
.card {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
.card-highlight {
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(139,92,246,0.08)) !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* ── Inputs ─────────────────────────────────────────────────────── */
.gradio-container input[type="text"],
.gradio-container input[type="password"],
.gradio-container textarea,
.gradio-container select {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    transition: border-color 0.2s ease;
}
.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

/* ── Primary button ─────────────────────────────────────────────── */
.run-btn {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 0.85rem 2rem !important;
    border-radius: 12px !important;
    letter-spacing: 0.02em;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
}
.run-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
}

/* ── Log area ───────────────────────────────────────────────────── */
.log-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.5 !important;
    background: #020617 !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
}

/* ── Results area ───────────────────────────────────────────────── */
.results-card {
    background: linear-gradient(135deg, rgba(59,130,246,0.06), rgba(139,92,246,0.06)) !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* ── Accordion ──────────────────────────────────────────────────── */
.gradio-accordion {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}
.gradio-accordion > .label-wrap {
    background: #1e293b !important;
    color: #94a3b8 !important;
}

/* ── Tabs inside accordion ──────────────────────────────────────── */
.gradio-tab-nav button {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    border: none !important;
    background: transparent !important;
}
.gradio-tab-nav button.selected {
    color: #60a5fa !important;
    border-bottom: 2px solid #3b82f6 !important;
}

/* ── Divider ────────────────────────────────────────────────────── */
.divider {
    border: none !important;
    border-top: 1px solid #1e293b !important;
    margin: 0.5rem 0 !important;
}

/* ── File upload area ───────────────────────────────────────────── */
.gradio-file {
    border: 2px dashed #334155 !important;
    border-radius: 12px !important;
    background: rgba(15, 23, 42, 0.5) !important;
}

/* ── Dropdown list (options popup) ──────────────────────────────── */
.gradio-container ul[role="listbox"],
.gradio-container .options,
.gradio-container ul.options {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
.gradio-container ul[role="listbox"] li,
.gradio-container .options li,
.gradio-container ul.options li {
    color: #e2e8f0 !important;
    background: #1e293b !important;
}
.gradio-container ul[role="listbox"] li:hover,
.gradio-container .options li:hover,
.gradio-container ul.options li:hover,
.gradio-container ul[role="listbox"] li.selected,
.gradio-container .options li.selected {
    background: #334155 !important;
    color: #ffffff !important;
}

/* ── Ollama / OpenAI info text ──────────────────────────────────── */
.ollama-info p {
    color: #86efac !important;
    font-size: 0.85rem !important;
    margin: 0.3rem 0 0 !important;
}
.openai-info p {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    margin: 0.3rem 0 0 !important;
}

/* ── Cookie guide ───────────────────────────────────────────────── */
.cookie-guide-step {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 0.8rem !important;
    margin-bottom: 0.6rem !important;
}
.cookie-guide-step p {
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    margin: 0.4rem 0 0.5rem !important;
    line-height: 1.5 !important;
}
.cookie-guide-step img {
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
    max-width: 100% !important;
}
.cookie-guide-step a {
    color: #60a5fa !important;
    text-decoration: none !important;
}
.cookie-guide-step a:hover {
    text-decoration: underline !important;
}
.cookie-step-num {
    display: inline-block;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: #fff !important;
    font-weight: 700;
    width: 1.5rem;
    height: 1.5rem;
    line-height: 1.5rem;
    text-align: center;
    border-radius: 50%;
    font-size: 0.8rem;
    margin-right: 0.4rem;
}
"""
