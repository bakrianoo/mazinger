"""Gradio theme and CSS for Mazinger Studio.

Palette — "Charcoal Blue / Verdigris":

    #264653  Charcoal Blue   surfaces, borders, the whole dark base
    #2a9d8f  Verdigris       primary accent: selection, focus, progress
    #e9c46a  Tuscan Sun      highlight accent: links on hover, warnings
    #f4a261  Sandy Brown     secondary accent: links, render action
    #e76f51  Burnt Peach     destructive / error states

The studio is a dark-only UI, but Gradio's theme switcher can put the page in
either light or dark mode.  Two things keep both modes identical:

  * every colour is written through :func:`_mirror`, which sets both the light
    variable and its ``*_dark`` twin, and
  * the ``neutral_*`` ramp is **inverted** (``neutral_50`` is the darkest
    surface, ``neutral_950`` the lightest text).  Gradio's light-mode defaults
    read ``neutral_50`` for backgrounds and ``neutral_800`` for text, so any
    variable this file forgets still lands on a dark surface instead of white.
"""

import inspect

import gradio as gr

# ── Brand palette ───────────────────────────────────────────────────
CHARCOAL = "#264653"   # Charcoal Blue
VERDIGRIS = "#2a9d8f"  # Verdigris
TUSCAN = "#e9c46a"     # Tuscan Sun
SANDY = "#f4a261"      # Sandy Brown
PEACH = "#e76f51"      # Burnt Peach

# ── Surfaces, derived from Charcoal Blue ────────────────────────────
BG = "#0b171c"          # page background (deepest)
SUNKEN = "#101f26"      # inputs, log panes — recessed below a card
SURFACE = "#16303a"     # cards, blocks, popovers
SURFACE_ALT = "#1a323c" # hover / elevated surface
LINE = CHARCOAL         # visible borders
LINE_SOFT = "#1d3743"   # hairlines, dividers, group gaps
LINE_STRONG = "#31596a" # hovered borders

# ── Text ────────────────────────────────────────────────────────────
TEXT = "#dbe8ec"        # body copy
TEXT_DIM = "#93b0ba"    # labels, secondary copy
TEXT_MUTED = "#6b8f9c"  # hints, captions, placeholders
TEXT_FAINT = "#547482"  # placeholder text only

# ── Accent tints (used where a fill must stay readable) ─────────────
VERDIGRIS_SOFT = "rgba(42, 157, 143, 0.18)"
VERDIGRIS_FAINT = "rgba(42, 157, 143, 0.12)"
VERDIGRIS_LIGHT = "#7fd6c8"   # accent text on a dark fill
PEACH_SOFT = "rgba(231, 111, 81, 0.14)"
PEACH_LIGHT = "#f6b39f"       # error text on a dark fill


def _mirror(values: dict) -> dict:
    """Expand ``{name: value}`` into both light and ``name_dark`` variables.

    Names Gradio's ``Base.set()`` doesn't accept are dropped, so this file
    stays compatible across Gradio versions rather than raising TypeError on
    a variable that only exists in some of them.
    """
    accepted = set(inspect.signature(gr.themes.Base.set).parameters)
    out = {}
    for name, value in values.items():
        for key in (name, f"{name}_dark"):
            if key in accepted:
                out[key] = value
    return out


theme = gr.themes.Base(
    # Verdigris — the primary accent ramp (normal ordering: 50 lightest).
    primary_hue=gr.themes.Color(
        c50="#e8f7f4", c100="#c7ece6", c200="#9addd1", c300="#66c9ba",
        c400="#43b3a3", c500=VERDIGRIS, c600="#23847a", c700="#1d6b64",
        c800="#17544f", c900="#133f3d", c950="#0d2a29",
    ),
    # Sandy Brown — secondary accent, used for links.
    secondary_hue=gr.themes.Color(
        c50="#fef3e8", c100="#fde2c9", c200="#fbcda1", c300="#f8b77c",
        c400=SANDY, c500="#ef8b44", c600="#e2712c", c700="#bd5a23",
        c800="#954823", c900="#783c22", c950="#40200f",
    ),
    # Charcoal-teal neutrals, INVERTED: 50 is the darkest surface and 950 the
    # lightest text, so Gradio's light-mode fallbacks resolve to dark values.
    neutral_hue=gr.themes.Color(
        c50=BG, c100=SUNKEN, c200=SURFACE_ALT, c300=CHARCOAL,
        c400=TEXT_MUTED, c500=TEXT_DIM, c600="#a9c2ca", c700="#c3d7dd",
        c800=TEXT, c900="#e9f1f4", c950="#f4f9fa",
    ),
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    **_mirror({
        # ── Body ────────────────────────────────────────────────────
        "body_background_fill": BG,
        "body_text_color": TEXT,
        "body_text_color_subdued": TEXT_MUTED,

        # ── Core surfaces ───────────────────────────────────────────
        # background_fill_primary backs dropdown popups, tab overflow menus
        # and checkbox interiors; unset it defaults to plain white.
        "background_fill_primary": SURFACE,
        "background_fill_secondary": SURFACE_ALT,  # hover fills (tabs, items)
        "color_accent": VERDIGRIS,
        "color_accent_soft": VERDIGRIS_SOFT,

        # border_color_primary is also painted as the *background* of every
        # gr.Group (Gradio draws child separators by letting it show through
        # 1px gaps), so it must stay a quiet surface tone — an accent colour
        # here turns whole groups into solid blocks.
        "border_color_primary": LINE_SOFT,
        "border_color_accent": VERDIGRIS,
        "border_color_accent_subdued": "rgba(42, 157, 143, 0.45)",

        # ── Links (file sizes, download links, markdown) ────────────
        "link_text_color": SANDY,
        "link_text_color_hover": TUSCAN,
        "link_text_color_active": TUSCAN,
        "link_text_color_visited": SANDY,

        # ── Blocks ──────────────────────────────────────────────────
        "block_background_fill": SURFACE,
        "block_border_color": LINE_SOFT,
        "block_border_width": "1px",
        "block_radius": "12px",
        "block_shadow": "0 1px 3px rgba(0, 0, 0, 0.35)",
        "block_label_background_fill": SURFACE,
        "block_label_border_color": LINE_SOFT,
        "block_label_text_color": TEXT_MUTED,
        "block_title_text_color": TEXT_DIM,
        "block_info_text_color": TEXT_MUTED,
        "panel_background_fill": SURFACE_ALT,
        "panel_border_color": LINE_SOFT,
        "accordion_text_color": TEXT_DIM,
        "code_background_fill": SUNKEN,
        "container_radius": "10px",
        "shadow_spread": "0px",
        "shadow_drop": "0 1px 2px rgba(0, 0, 0, 0.35)",
        "shadow_drop_lg": "0 8px 24px rgba(0, 0, 0, 0.45)",

        # ── Inputs ──────────────────────────────────────────────────
        "input_background_fill": SUNKEN,
        "input_background_fill_hover": SUNKEN,
        "input_background_fill_focus": SUNKEN,
        "input_border_color": LINE,
        "input_border_color_hover": LINE_STRONG,
        "input_border_color_focus": VERDIGRIS,
        "input_border_width": "1px",
        "input_radius": "8px",
        "input_placeholder_color": TEXT_FAINT,
        "input_shadow": "none",
        "input_shadow_focus": "0 0 0 3px rgba(42, 157, 143, 0.15)",

        # ── Buttons ─────────────────────────────────────────────────
        "button_border_width": "1px",
        "button_large_radius": "10px",
        "button_medium_radius": "9px",
        "button_small_radius": "8px",
        "button_primary_background_fill":
            f"linear-gradient(135deg, {VERDIGRIS} 0%, #1f7f74 100%)",
        "button_primary_background_fill_hover":
            "linear-gradient(135deg, #34b3a3 0%, #26907f 100%)",
        "button_primary_border_color": "transparent",
        "button_primary_border_color_hover": "transparent",
        "button_primary_text_color": "#ffffff",
        "button_primary_text_color_hover": "#ffffff",
        "button_secondary_background_fill": SURFACE_ALT,
        "button_secondary_background_fill_hover": "#22404c",
        "button_secondary_border_color": LINE,
        "button_secondary_border_color_hover": LINE_STRONG,
        "button_secondary_text_color": TEXT_DIM,
        "button_secondary_text_color_hover": TEXT,
        "button_cancel_background_fill": PEACH_SOFT,
        "button_cancel_background_fill_hover": "rgba(231, 111, 81, 0.24)",
        "button_cancel_border_color": "rgba(231, 111, 81, 0.45)",
        "button_cancel_border_color_hover": PEACH,
        "button_cancel_text_color": PEACH_LIGHT,
        "button_cancel_text_color_hover": "#ffffff",

        # ── Radios & checkboxes (rendered as labelled pills) ────────
        "checkbox_background_color": SUNKEN,
        "checkbox_background_color_hover": SUNKEN,
        "checkbox_background_color_focus": SUNKEN,
        "checkbox_background_color_selected": VERDIGRIS,
        "checkbox_border_color": LINE,
        "checkbox_border_color_hover": LINE_STRONG,
        "checkbox_border_color_focus": VERDIGRIS,
        "checkbox_border_color_selected": VERDIGRIS,
        "checkbox_border_radius": "8px",
        "checkbox_border_width": "1px",
        "checkbox_shadow": "none",
        "checkbox_label_background_fill": SURFACE,
        "checkbox_label_background_fill_hover": SURFACE_ALT,
        "checkbox_label_background_fill_selected": VERDIGRIS_SOFT,
        "checkbox_label_border_color": LINE,
        "checkbox_label_border_color_hover": "rgba(42, 157, 143, 0.55)",
        "checkbox_label_border_color_selected": VERDIGRIS,
        "checkbox_label_border_width": "1px",
        "checkbox_label_gap": "6px",
        "checkbox_label_padding": "6px 14px",
        "checkbox_label_shadow": "none",
        "checkbox_label_text_color": TEXT_DIM,
        "checkbox_label_text_color_selected": VERDIGRIS_LIGHT,
        "checkbox_label_text_size": "0.875rem",

        # ── Sliders & progress ──────────────────────────────────────
        "slider_color": VERDIGRIS,
        "loader_color": VERDIGRIS,
        "stat_background_fill": VERDIGRIS,

        # ── Tables (this is what file rows are made of) ─────────────
        "table_even_background_fill": SURFACE,
        "table_odd_background_fill": SUNKEN,
        "table_border_color": LINE_SOFT,
        "table_row_focus": VERDIGRIS_FAINT,
        "table_text_color": TEXT,

        # ── Errors ──────────────────────────────────────────────────
        "error_background_fill": PEACH_SOFT,
        "error_border_color": PEACH,
        "error_border_width": "1px",
        "error_text_color": PEACH_LIGHT,
        "error_icon_color": PEACH,
    })
)


CSS = f"""
/* ── Palette tokens, for the hand-written rules below ────────────── */
:root {{
    --mz-charcoal: {CHARCOAL};
    --mz-verdigris: {VERDIGRIS};
    --mz-tuscan: {TUSCAN};
    --mz-sandy: {SANDY};
    --mz-peach: {PEACH};

    --mz-bg: {BG};
    --mz-sunken: {SUNKEN};
    --mz-surface: {SURFACE};
    --mz-surface-alt: {SURFACE_ALT};
    --mz-line: {LINE};
    --mz-line-soft: {LINE_SOFT};
    --mz-line-strong: {LINE_STRONG};

    --mz-text: {TEXT};
    --mz-text-dim: {TEXT_DIM};
    --mz-text-muted: {TEXT_MUTED};
}}

/* ── Global layout ──────────────────────────────────────────────── */
.gradio-container {{
    max-width: 940px !important;
    margin: 0 auto !important;
    background: var(--mz-bg) !important;
}}
footer {{ display: none !important; }}

/* Text selection — verdigris wash, never a glaring block */
.gradio-container ::selection,
.gradio-container *::selection {{
    background: rgba(42, 157, 143, 0.32);
    color: #eef7f5;
}}

/* ── Header ─────────────────────────────────────────────────────── */
.gradio-container .app-header {{
    text-align: center;
    padding: 2rem 1rem 0.75rem;
}}
/* Keep gradient values on a single line — Gradio's CSS rewriter drops
   declarations whose value is wrapped across lines. */
.gradio-container .app-header h1 {{
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    background-image: linear-gradient(120deg, var(--mz-verdigris) 0%, var(--mz-tuscan) 52%, var(--mz-sandy) 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0.4rem !important;
    letter-spacing: -0.02em !important;
}}
.gradio-container .app-header p {{
    color: var(--mz-text-dim) !important;
    font-size: 1rem;
    margin-top: 0 !important;
}}
/* Gradio stamps elem_classes on both the block wrapper and the inner .prose
   div, so decoration set on the outer must be neutralised on the inner. */
.gradio-container .prose.app-header {{
    padding: 0 !important;
}}

/* ── Section headings ───────────────────────────────────────────── */
.gradio-container .section-title {{
    margin: 1.9rem 0 0.6rem !important;
    padding: 0 0 0 0.7rem !important;
    border-left: 3px solid var(--mz-verdigris) !important;
    /* blocks are rounded by default, which would bend the accent bar */
    border-radius: 0 !important;
}}
.gradio-container .section-title h4,
.gradio-container .section-title p {{
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.45rem !important;
    color: var(--mz-text) !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    margin: 0 !important;
}}
/* …same duplicated-class caveat as .app-header: only the outer block keeps
   the accent bar, otherwise it draws twice. */
.gradio-container .prose.section-title {{
    border-left: none !important;
    margin: 0 !important;
    padding: 0 !important;
}}

/* ── Groups & cards ─────────────────────────────────────────────── */
/* Gradio paints every gr.Group with --border-color-primary so that 1px gaps
   between children read as separators.  Cards want a real surface instead. */
.gradio-container .gr-group {{
    background: var(--mz-surface) !important;
}}
/* A group renders its wrapper twice, both carrying the same elem_classes;
   without this the card's border and padding are drawn two deep. */
.gradio-container .gr-group > .gr-group {{
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}}
/* Gradio blanks the borders of a group's children with `border: none`, but
   those elements also carry an inline `border-style: solid` that outranks it —
   leaving a 3px currentColor frame, which is glaring against light text.
   Pin the width to zero instead; real nested groups keep their own border. */
.gradio-container .gr-group > :not(.absolute):not(.gr-group),
.gradio-container .gr-group .styler > :not(.absolute):not(.gr-group) {{
    border-width: 0 !important;
}}
.gradio-container .card,
.gradio-container .card-highlight,
.gradio-container .results-card,
.gradio-container .render-card {{
    background: var(--mz-surface) !important;
    border: 1px solid var(--mz-line) !important;
    border-radius: 14px !important;
    padding: 1.1rem !important;
}}
.gradio-container .card-highlight,
.gradio-container .results-card {{
    border-color: rgba(42, 157, 143, 0.35) !important;
}}
.gradio-container .card-highlight {{
    position: relative !important;
}}
.gradio-container .render-card {{
    border-color: rgba(244, 162, 97, 0.32) !important;
    margin-top: 0.75rem !important;
}}
.gradio-container .card-highlight::before {{
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 14px;
    background-image: linear-gradient(135deg, rgba(42, 157, 143, 0.07), rgba(244, 162, 97, 0.04));
    pointer-events: none;
}}
/* Nested groups (voice theme, provider panels) read as a sunken sub-panel */
.gradio-container .voice-theme-group,
.gradio-container .voice-info-box {{
    background: var(--mz-sunken) !important;
    border: 1px solid var(--mz-line-soft) !important;
    border-radius: 10px !important;
    padding: 0.85rem !important;
    margin-top: 0.25rem !important;
}}
/* Layout wrappers must never inherit a fill of their own */
.gradio-container .styler,
.gradio-container .form,
.gradio-container fieldset {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
/* A Radio renders as a <fieldset class="block padded">; overriding its padding
   here would pull it 6px up and 12px left of every sibling control, breaking
   label alignment across a Row. Only the inner .form wrapper is adjusted. */
.gradio-container .form {{
    padding: 0 !important;
}}

/* ── Row alignment ──────────────────────────────────────────────── */
/* A control with a label is taller than a bare button, so a Row of the two
   leaves the button floating at the top. Pin such rows to the baseline. */
.gradio-container .row-bottom {{
    align-items: flex-end !important;
}}
/* A bare button has no block padding, so flex-end lands it a block-padding
   below the neighbouring input's edge; give it back that offset. */
.gradio-container .row-bottom > button {{
    margin-bottom: var(--spacing-xl) !important;
}}

/* ── Inputs ─────────────────────────────────────────────────────── */
.gradio-container input[type="text"],
.gradio-container input[type="password"],
.gradio-container input[type="number"],
.gradio-container textarea,
.gradio-container select {{
    background: var(--mz-sunken) !important;
    border: 1px solid var(--mz-line) !important;
    border-radius: 8px !important;
    color: var(--mz-text) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}}
.gradio-container input[type="text"]:focus,
.gradio-container input[type="password"]:focus,
.gradio-container input[type="number"]:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {{
    border-color: var(--mz-verdigris) !important;
    box-shadow: 0 0 0 3px rgba(42, 157, 143, 0.15) !important;
    outline: none !important;
}}
/* Gradio's `info=` hint text — never let it pick up a highlight fill */
.gradio-container .info-text,
.gradio-container span.info {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--mz-text-muted) !important;
    font-size: 0.79rem !important;
    line-height: 1.5 !important;
}}

/* ── Radio pills ────────────────────────────────────────────────── */
/* Gradio styles the wrapping <label> from the checkbox_label_* theme
   variables; all that is left is hiding the native dot so it reads as a
   pill, and restoring a focus ring for keyboard users. */
.gradio-container label > input[type="radio"] {{
    appearance: none !important;
    -webkit-appearance: none !important;
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
}}
.gradio-container label:has(> input[type="radio"]) {{
    white-space: nowrap !important;
    cursor: pointer !important;
}}
.gradio-container label:has(> input[type="radio"]:focus-visible) {{
    outline: 2px solid var(--mz-verdigris) !important;
    outline-offset: 2px !important;
}}

/* ── Checkboxes ─────────────────────────────────────────────────── */
.gradio-container input[type="checkbox"] {{
    accent-color: var(--mz-verdigris) !important;
    width: 15px !important;
    height: 15px !important;
    /* checkbox_border_radius also rounds the radio pills, so square the box
       back off here — at 8px a 15px checkbox reads as a radio button. */
    border-radius: 4px !important;
}}

/* ── Sliders ────────────────────────────────────────────────────── */
/* The unfilled half of the track is Gradio's raw --neutral-200; pin it to a
   surface tone so it can't read as a bright bar on the dark background. */
.gradio-container input[type="range"] {{
    accent-color: var(--mz-verdigris) !important;
}}
.gradio-container input[type="range"]::-webkit-slider-runnable-track {{
    background-image: linear-gradient(to right, var(--mz-verdigris) var(--range_progress), var(--mz-surface-alt) var(--range_progress)) !important;
}}
.gradio-container input[type="range"]::-moz-range-track {{
    background: var(--mz-surface-alt) !important;
}}
.gradio-container input[type="range"]::-webkit-slider-thumb {{
    background-color: #eaf6f4 !important;
    box-shadow: 0 0 0 1px rgba(42, 157, 143, 0.6), 1px 1px 4px rgba(0,0,0,0.4) !important;
}}
.gradio-container input[type="range"]::-moz-range-thumb {{
    background-color: #eaf6f4 !important;
}}

/* ── Primary action button ──────────────────────────────────────── */
.gradio-container .run-btn {{
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 0.85rem 2rem !important;
    border-radius: 12px !important;
    letter-spacing: 0.02em !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 4px 20px rgba(42, 157, 143, 0.28) !important;
}}
.gradio-container .run-btn:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(42, 157, 143, 0.42) !important;
}}

/* ── Render button — the warm half of the palette ───────────────── */
.gradio-container .render-btn {{
    background: linear-gradient(135deg, var(--mz-sandy) 0%, var(--mz-peach) 100%) !important;
    color: #33170b !important;
    border-color: transparent !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.7rem 1.5rem !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 18px rgba(231, 111, 81, 0.25) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}}
.gradio-container .render-btn:hover {{
    background: linear-gradient(135deg, #f7b47e 0%, #ec8467 100%) !important;
    color: #33170b !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(231, 111, 81, 0.4) !important;
}}

/* ── Log panes ──────────────────────────────────────────────────── */
.gradio-container .log-box textarea {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.6 !important;
    background: #071216 !important;
    color: var(--mz-text-dim) !important;
    border: 1px solid var(--mz-line-soft) !important;
    border-radius: 8px !important;
    max-height: 400px !important;
    overflow-y: auto !important;
}}

/* ── Accordions ─────────────────────────────────────────────────── */
.gradio-container .gr-accordion {{
    background: var(--mz-surface) !important;
    border: 1px solid var(--mz-line-soft) !important;
    border-radius: 12px !important;
}}
.gradio-container .label-wrap {{
    color: var(--mz-text-dim) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: color 0.15s ease !important;
}}
.gradio-container .label-wrap:hover {{
    color: var(--mz-text) !important;
}}

/* ── Tabs ───────────────────────────────────────────────────────── */
/* Let every tab stay visible instead of collapsing into an overflow menu. */
.gradio-container .tab-wrapper {{
    height: auto !important;
}}
.gradio-container .tab-container {{
    height: auto !important;
    flex-wrap: wrap !important;
    overflow: visible !important;
}}
.gradio-container .overflow-menu {{ display: none !important; }}
.gradio-container .tab-container button {{
    color: var(--mz-text-muted) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    height: auto !important;
    padding: 0.5rem 0.8rem !important;
    border-radius: 8px 8px 0 0 !important;
}}
.gradio-container .tab-container button:hover:not(.selected) {{
    background-color: var(--mz-surface-alt) !important;
    color: var(--mz-text-dim) !important;
}}
.gradio-container .tab-container button.selected {{
    color: var(--mz-verdigris) !important;
    background-color: rgba(42, 157, 143, 0.10) !important;
    font-weight: 600 !important;
}}

/* ── Divider ────────────────────────────────────────────────────── */
.gradio-container .divider {{
    border: none !important;
    border-top: 1px solid var(--mz-line-soft) !important;
    margin: 1.5rem 0 !important;
}}

/* ── File upload / file lists ───────────────────────────────────── */
.gradio-container .file-preview {{
    color: var(--mz-text) !important;
}}
.gradio-container .file-preview .filename {{
    color: var(--mz-text) !important;
}}
.gradio-container .file-preview .download > a {{
    color: var(--mz-sandy) !important;
}}
.gradio-container .file-preview .download > a:hover {{
    color: var(--mz-tuscan) !important;
}}

/* ── Dropdown popup ─────────────────────────────────────────────── */
.gradio-container .options {{
    border: 1px solid var(--mz-line) !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
    padding: 0.3rem !important;
}}
.gradio-container .options .item {{
    color: var(--mz-text-dim) !important;
    border-radius: 6px !important;
    font-size: 0.875rem !important;
}}
.gradio-container .options .item:hover,
.gradio-container .options .item.active {{
    background: var(--mz-surface-alt) !important;
    color: var(--mz-text) !important;
}}
.gradio-container .options .item.selected {{
    background: rgba(42, 157, 143, 0.16) !important;
    color: {VERDIGRIS_LIGHT} !important;
}}

/* ── Provider notes ─────────────────────────────────────────────── */
.gradio-container .ollama-info,
.gradio-container .ollama-info p {{
    color: {VERDIGRIS_LIGHT} !important;
    font-size: 0.82rem !important;
    margin: 0.35rem 0 0 !important;
    line-height: 1.5 !important;
    background: transparent !important;
}}
.gradio-container .openai-info,
.gradio-container .openai-info p {{
    color: var(--mz-text-muted) !important;
    font-size: 0.83rem !important;
    margin: 0.35rem 0 0 !important;
    line-height: 1.55 !important;
    background: transparent !important;
}}

/* ── Voice theme selector ───────────────────────────────────────── */
.gradio-container .voice-theme-group label span {{
    font-size: 0.88rem !important;
}}
.gradio-container .voice-info-text p,
.gradio-container .voice-info-text strong {{
    color: var(--mz-text-dim) !important;
    font-size: 0.84rem !important;
    line-height: 1.6 !important;
    background: transparent !important;
}}
.gradio-container .voice-info-text strong {{
    color: var(--mz-text) !important;
    font-weight: 600 !important;
}}

/* ── Cookie guide ───────────────────────────────────────────────── */
.gradio-container .cookie-guide-step {{
    background: var(--mz-sunken) !important;
    border: 1px solid var(--mz-line-soft) !important;
    border-radius: 10px !important;
    padding: 0.8rem !important;
    margin-bottom: 0.6rem !important;
}}
.gradio-container .cookie-guide-step p {{
    color: var(--mz-text-dim) !important;
    font-size: 0.83rem !important;
    margin: 0.4rem 0 0.5rem !important;
    line-height: 1.5 !important;
}}
.gradio-container .cookie-guide-step img {{
    border-radius: 8px !important;
    border: 1px solid var(--mz-line-soft) !important;
    max-width: 100% !important;
}}
.gradio-container .cookie-guide-step a {{
    color: var(--mz-sandy) !important;
    text-decoration: none !important;
}}
.gradio-container .cookie-guide-step a:hover {{
    color: var(--mz-tuscan) !important;
    text-decoration: underline !important;
}}
.gradio-container .cookie-step-num {{
    display: inline-block;
    background: linear-gradient(135deg, var(--mz-verdigris), var(--mz-sandy));
    color: #0b171c !important;
    font-weight: 700;
    width: 1.5rem;
    height: 1.5rem;
    line-height: 1.5rem;
    text-align: center;
    border-radius: 50%;
    font-size: 0.78rem;
    margin-right: 0.5rem;
    flex-shrink: 0;
}}

/* ── Scrollbars ─────────────────────────────────────────────────── */
.gradio-container ::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
.gradio-container ::-webkit-scrollbar-track {{
    background: var(--mz-bg);
}}
.gradio-container ::-webkit-scrollbar-thumb {{
    background: var(--mz-charcoal);
    border-radius: 4px;
}}
.gradio-container ::-webkit-scrollbar-thumb:hover {{
    background: var(--mz-line-strong);
}}
"""
