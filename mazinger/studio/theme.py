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
    body_background_fill="#0a0f1e",
    body_background_fill_dark="#0a0f1e",
    body_text_color="#cbd5e1",
    body_text_color_dark="#cbd5e1",
    body_text_color_subdued="#8b98af",
    body_text_color_subdued_dark="#8b98af",

    block_background_fill="#162035",
    block_background_fill_dark="#162035",
    block_border_color="#243352",
    block_border_color_dark="#243352",
    block_border_width="1px",
    block_label_text_color="#64748b",
    block_label_text_color_dark="#64748b",
    block_label_background_fill="#162035",
    block_label_background_fill_dark="#162035",
    block_title_text_color="#94a3b8",
    block_title_text_color_dark="#94a3b8",
    block_shadow="0 2px 8px rgba(0, 0, 0, 0.4)",
    block_shadow_dark="0 2px 8px rgba(0, 0, 0, 0.4)",

    input_background_fill="#0a0f1e",
    input_background_fill_dark="#0a0f1e",
    input_border_color="#1e2d45",
    input_border_color_dark="#1e2d45",
    input_placeholder_color="#4a5a72",
    input_placeholder_color_dark="#4a5a72",

    button_primary_background_fill="linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%)",
    button_primary_text_color="#ffffff",
    button_primary_border_color="transparent",
    button_primary_border_color_dark="transparent",

    button_secondary_background_fill="#1e2d45",
    button_secondary_background_fill_dark="#1e2d45",
    button_secondary_text_color="#b0bdd0",
    button_secondary_border_color="#2d3f5c",

    border_color_primary="#2563eb",
    border_color_primary_dark="#2563eb",

    shadow_spread="0px",

    checkbox_background_color="#0a0f1e",
    checkbox_background_color_dark="#0a0f1e",
    checkbox_background_color_selected="#2563eb",
    checkbox_background_color_selected_dark="#2563eb",
    checkbox_border_color="#2d3f5c",
    checkbox_border_color_dark="#2d3f5c",
    checkbox_label_text_color="#b0bdd0",

    slider_color="#3b82f6",
    slider_color_dark="#3b82f6",
)


CSS = """
/* ── Global layout ──────────────────────────────────────────────── */
.gradio-container {
    max-width: 940px !important;
    margin: 0 auto !important;
    background: #0a0f1e !important;
}
footer { display: none !important; }

/* Text selection — readable on the dark theme */
.gradio-container ::selection {
    background: rgba(59, 130, 246, 0.35);
    color: #f1f5f9;
}

/* ── Header ─────────────────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 2rem 1rem 0.75rem;
}
.app-header h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem !important;
    letter-spacing: -0.02em !important;
}
.app-header p {
    color: #94a3b8 !important;
    font-size: 1rem;
    margin-top: 0 !important;
}

/* ── Section headings ───────────────────────────────────────────── */
.section-title {
    margin: 1.9rem 0 0.6rem !important;
    padding: 0 0 0 0.7rem !important;
    border-left: 3px solid #3b82f6 !important;
}
.section-title h4,
.section-title p {
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.45rem !important;
    color: #cbd5e1 !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    margin: 0 !important;
}

/* ── Cards ──────────────────────────────────────────────────────── */
/* Shared card base — every card variant inherits these */
.card,
.card-highlight,
.results-card,
.render-card {
    background: #162035 !important;
    border: 1px solid #263b58 !important;
    border-radius: 14px !important;
    padding: 1.1rem !important;
}
/* Accent variants only change the border / overlay */
.card-highlight,
.results-card {
    border-color: #2a4068 !important;
}
.card-highlight {
    position: relative !important;
}
.render-card {
    border-color: #3a2860 !important;
    margin-top: 0.75rem !important;
}
.card-highlight::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(59,130,246,0.06), rgba(139,92,246,0.04));
    pointer-events: none;
}

/* ── Inputs ─────────────────────────────────────────────────────── */
.gradio-container input[type="text"],
.gradio-container input[type="password"],
.gradio-container textarea,
.gradio-container select {
    background: #0a0f1e !important;
    border: 1px solid #263b58 !important;
    border-radius: 8px !important;
    color: #cbd5e1 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
    outline: none !important;
}
/* Kill the built-in Gradio info/description highlight on inputs */
.gradio-container .info,
.gradio-container .description,
span.info {
    background: transparent !important;
    color: #7c8aa3 !important;
    font-size: 0.79rem !important;
    line-height: 1.5 !important;
}

/* ── Text selection (subtle, not a glaring blue block) ──────────── */
.gradio-container ::selection,
.gradio-container *::selection {
    background: rgba(59, 130, 246, 0.30);
    color: #e2e8f0;
}

/* ── Radio buttons (Gradio-5 version-agnostic pills) ────────────── */
/* Gradio 5 renders each option as a <label> wrapping <input type=radio>.
   The selected <label> gets a default blue background — we style the
   label itself as the pill and neutralize that default fill. */

/* 1. Strip backgrounds from the radio container / row wrappers */
.gradio-container fieldset:has(input[type="radio"]),
.gradio-container .wrap:has(input[type="radio"]),
.gradio-container .wrap-inner:has(input[type="radio"]) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.gradio-container .wrap:has(input[type="radio"]),
.gradio-container .wrap-inner:has(input[type="radio"]) {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0.35rem !important;
}

/* 2. Each pill = the <label> directly wrapping a radio input */
.gradio-container label:has(> input[type="radio"]) {
    background: #1a2540 !important;
    border: 1px solid #2d3f5c !important;
    border-radius: 8px !important;
    padding: 0.34rem 0.85rem !important;
    margin: 0 !important;
    color: #94a3b8 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    box-shadow: none !important;
    white-space: nowrap !important;
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
}

/* 3. Hover */
.gradio-container label:has(> input[type="radio"]):hover {
    background: #1e2d45 !important;
    border-color: rgba(59, 130, 246, 0.5) !important;
    color: #cbd5e1 !important;
}

/* 4. Selected pill — match both Gradio's .selected class and :checked */
.gradio-container label.selected:has(> input[type="radio"]),
.gradio-container label:has(> input[type="radio"]:checked) {
    background: rgba(37, 99, 235, 0.20) !important;
    border-color: #3b82f6 !important;
    color: #93c5fd !important;
    font-weight: 600 !important;
}

/* 5. Pill text inherits colour; kill any inner span background */
.gradio-container label:has(> input[type="radio"]) span {
    background: transparent !important;
    color: inherit !important;
}

/* 6. Hide the native radio dot entirely */
.gradio-container input[type="radio"] {
    appearance: none !important;
    -webkit-appearance: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    position: absolute !important;
}
/* ── Group / form containers — prevent accent bleed ────────────── */
/* .styler is a Gradio 5 wrapper that inherits the primary color as bg */
.gradio-container .styler,
.gradio-container .gradio-group,
.gradio-container .form,
.gradio-container .gradio-group > .block,
.gradio-container fieldset {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.gradio-container .form,
.gradio-container fieldset {
    padding: 0.25rem 0 !important;
}

/* ── Primary button ─────────────────────────────────────────────── */
.run-btn {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 0.85rem 2rem !important;
    border-radius: 12px !important;
    letter-spacing: 0.02em !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.25) !important;
}
.run-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(59, 130, 246, 0.4) !important;
}

/* ── Secondary buttons ──────────────────────────────────────────── */
.gradio-container button.secondary,
.gradio-container button[variant="secondary"] {
    background: #1e2d45 !important;
    border: 1px solid #2d3f5c !important;
    color: #b0bdd0 !important;
    border-radius: 8px !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}
.gradio-container button.secondary:hover,
.gradio-container button[variant="secondary"]:hover {
    background: #263552 !important;
    color: #e2e8f0 !important;
}

/* ── Log area ───────────────────────────────────────────────────── */
.log-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.6 !important;
    background: #060a14 !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    max-height: 400px !important;
    overflow-y: auto !important;
    border-color: #263b58 !important;
}

/* ── Status box ─────────────────────────────────────────────────── */
.gradio-container .gradio-textbox:not(.log-box) textarea,
.gradio-container .gradio-textbox:not(.log-box) input {
    color: #cbd5e1 !important;
}

/* ── Results area ───────────────────────────────────────────────── */

.results-card .file-preview {
    background: #0a0f1e !important;
    border: 1px solid #263b58 !important;
    border-radius: 8px !important;
}
.results-card .file-preview .file-name,
.results-card a[download],
.results-card .upload-container .file-name,
.results-card .gradio-file a,
.results-card .gradio-file a:visited,
.results-card .gradio-file a:link,
.results-card .gradio-file span,
.results-card .gradio-file .name,
.results-card a,
.results-card a:visited,
.results-card a:link {
    color: #e2e8f0 !important;
}
.results-card .gradio-file .size,
.results-card .file-size {
    color: #94a3b8 !important;
}

/* ── Accordion ──────────────────────────────────────────────────── */
.gradio-accordion {
    background: #162035 !important;
    border: 1px solid #263b58 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}
.gradio-accordion > .label-wrap {
    background: #162035 !important;
    color: #8b98af !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.7rem 1rem !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}
.gradio-accordion > .label-wrap:hover {
    background: #1e2d45 !important;
    color: #cbd5e1 !important;
}

/* ── Tabs inside accordion ──────────────────────────────────────── */
.gradio-tab-nav {
    flex-wrap: wrap !important;
    overflow: visible !important;
    background: #0d1524 !important;
    border-bottom: 1px solid #1e2d45 !important;
    padding: 0 0.5rem !important;
    gap: 0.1rem !important;
}
/* hide the overflow "…" toggle and its dropdown */
.gradio-tab-nav .tab-nav-overflow,
.gradio-tab-nav .overflow-menu,
.tab-nav-overflow,
.overflow-menu {
    display: none !important;
}
/* Tab buttons */
.gradio-tab-nav button {
    color: #475569 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 0.75rem !important;
    border: none !important;
    background: transparent !important;
    white-space: nowrap !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
    margin-bottom: -1px !important;
}
.gradio-tab-nav button:hover {
    color: #94a3b8 !important;
}
.gradio-tab-nav button.selected {
    color: #60a5fa !important;
    border-bottom: 2px solid #3b82f6 !important;
    font-weight: 600 !important;
}
/* Overflow popup */
.gradio-tab-nav [role="menu"],
.gradio-tab-nav ul,
.gradio-tab-nav .tab-overflow-menu,
.tab-overflow-menu,
div[class*="overflow"] {
    background: #162035 !important;
    border: 1px solid #263b58 !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
}
.gradio-tab-nav [role="menu"] button,
.gradio-tab-nav ul button,
.tab-overflow-menu button {
    color: #94a3b8 !important;
    background: transparent !important;
}
.gradio-tab-nav [role="menu"] button:hover,
.gradio-tab-nav ul button:hover,
.tab-overflow-menu button:hover {
    background: #1e2d45 !important;
    color: #e2e8f0 !important;
}

/* ── Divider ────────────────────────────────────────────────────── */
.divider {
    border: none !important;
    border-top: 1px solid #1e2d45 !important;
    margin: 1.5rem 0 !important;
}

/* ── File upload area ───────────────────────────────────────────── */
.gradio-file {
    border: 2px dashed #1e2d45 !important;
    border-radius: 12px !important;
    background: rgba(10, 15, 30, 0.6) !important;
    transition: border-color 0.2s ease !important;
}
.gradio-file:hover {
    border-color: rgba(59,130,246,0.4) !important;
}
.gradio-file a,
.gradio-file a:visited,
.gradio-file a:link,
.gradio-file .wasm-file a,
.gradio-file td,
.gradio-file td a {
    color: #cbd5e1 !important;
}
.gradio-file .lite-file,
.gradio-file tr {
    background: #162035 !important;
    border-color: #263b58 !important;
}
.gradio-file .lite-file:hover,
.gradio-file tr:hover {
    background: #1e2d45 !important;
}

/* ── Dropdown list (options popup) ──────────────────────────────── */
.gradio-container ul[role="listbox"],
.gradio-container .options,
.gradio-container ul.options {
    background: #162035 !important;
    border: 1px solid #263b58 !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5) !important;
    padding: 0.3rem !important;
}
.gradio-container ul[role="listbox"] li,
.gradio-container .options li,
.gradio-container ul.options li {
    color: #94a3b8 !important;
    background: transparent !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.6rem !important;
    font-size: 0.875rem !important;
}
.gradio-container ul[role="listbox"] li:hover,
.gradio-container .options li:hover,
.gradio-container ul.options li:hover {
    background: #1e2d45 !important;
    color: #e2e8f0 !important;
}
.gradio-container ul[role="listbox"] li.selected,
.gradio-container .options li.selected {
    background: rgba(59,130,246,0.15) !important;
    color: #93c5fd !important;
}

/* ── Ollama / OpenAI info text ──────────────────────────────────── */
.ollama-info,
.ollama-info p {
    color: #4ade80 !important;
    font-size: 0.82rem !important;
    margin: 0.35rem 0 0 !important;
    line-height: 1.5 !important;
    background: transparent !important;
}
.openai-info,
.openai-info p {
    color: #7c8aa3 !important;
    font-size: 0.83rem !important;
    margin: 0.35rem 0 0 !important;
    line-height: 1.55 !important;
    background: transparent !important;
}
/* Prevent Gradio from adding a highlight/border to selected-state info text */
.gradio-container .wrap .info,
.gradio-container .wrap span.info {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Voice theme selector ───────────────────────────────────────── */
.voice-theme-group .gr-radio-group {
    gap: 0.4rem !important;
}
.voice-theme-group label span {
    font-size: 0.88rem !important;
}

/* ── Cookie guide ───────────────────────────────────────────────── */
.cookie-guide-step {
    background: #0a0f1e !important;
    border: 1px solid #263b58 !important;
    border-radius: 10px !important;
    padding: 0.8rem !important;
    margin-bottom: 0.6rem !important;
}
.cookie-guide-step p {
    color: #94a3b8 !important;
    font-size: 0.83rem !important;
    margin: 0.4rem 0 0.5rem !important;
    line-height: 1.5 !important;
}
.cookie-guide-step img {
    border-radius: 8px !important;
    border: 1px solid #263b58 !important;
    max-width: 100% !important;
}
.cookie-guide-step a {
    color: #60a5fa !important;
    text-decoration: none !important;
}
.cookie-guide-step a:hover {
    color: #93c5fd !important;
    text-decoration: underline !important;
}
.cookie-step-num {
    display: inline-block;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: #fff !important;
    font-weight: 700;
    width: 1.5rem;
    height: 1.5rem;
    line-height: 1.5rem;
    text-align: center;
    border-radius: 50%;
    font-size: 0.78rem;
    margin-right: 0.5rem;
    flex-shrink: 0;
}

/* ── Render video panel ─────────────────────────────────────────── */

.render-btn {
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.7rem 1.5rem !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 18px rgba(124, 58, 237, 0.25) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.render-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124, 58, 237, 0.4) !important;
}

/* ── Checkboxes ──────────────────────────────────────────────────── */
.gradio-container input[type="checkbox"] {
    accent-color: #3b82f6 !important;
    width: 15px !important;
    height: 15px !important;
}
.gradio-container .gradio-checkbox label {
    color: #94a3b8 !important;
    font-size: 0.875rem !important;
}

/* ── Slider ──────────────────────────────────────────────────────── */
.gradio-container input[type="range"] {
    accent-color: #3b82f6 !important;
}
.gradio-container .gradio-slider .head {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
}

/* ── Scrollbars ──────────────────────────────────────────────────── */
.gradio-container ::-webkit-scrollbar {
    width: 5px;
    height: 5px;
}
.gradio-container ::-webkit-scrollbar-track {
    background: #0a0f1e;
}
.gradio-container ::-webkit-scrollbar-thumb {
    background: #1e2d45;
    border-radius: 4px;
}
.gradio-container ::-webkit-scrollbar-thumb:hover {
    background: #2d3f5c;
}

/* ── Voice info box (autoclone description) ─────────────────────── */
.voice-info-box {
    background: #0d1524 !important;
    border: 1px solid #263b58 !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    margin-top: 0.25rem !important;
}
.voice-info-text p,
.voice-info-text strong {
    color: #94a3b8 !important;
    font-size: 0.84rem !important;
    line-height: 1.6 !important;
    background: transparent !important;
}
.voice-info-text strong {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}
"""
