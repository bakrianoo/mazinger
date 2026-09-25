"""The Editor tab relies on a handful of Gradio features.

Pinning them here means a Gradio upgrade that drops or renames one fails in
the test suite rather than when a user opens the Editor.  Each was checked to
exist from ``gradio==5.0`` — the floor declared in ``pyproject.toml``.
"""

import inspect

import pytest

gr = pytest.importorskip("gradio")
pd = pytest.importorskip("pandas")


def _params(fn):
    return inspect.signature(fn).parameters


def test_dataframe_row_selection_event():
    """Clicking a table row selects the chunk shown in the detail panel."""
    assert hasattr(gr.Dataframe, "select")
    assert hasattr(gr, "SelectData")


def test_dataframe_layout_options():
    """The chunk table is height-capped and wraps long text."""
    params = _params(gr.Dataframe.__init__)
    for name in ("max_height", "wrap", "column_widths", "datatype", "interactive"):
        assert name in params, name


def test_dataframe_accepts_a_pandas_styler():
    """Stale rows are highlighted through a Styler on the visible page."""
    df = pd.DataFrame({"text": ["a", "b"]})
    styler = df.style.apply(lambda row: ["background-color: #fee"] * len(row), axis=1)
    data = gr.Dataframe(value=styler).postprocess(styler)
    assert getattr(data, "metadata", None), "styling metadata was dropped"


def test_tabs_can_be_switched_programmatically():
    """"Open in Editor" switches from the Dub tab to the Editor tab."""
    assert "selected" in _params(gr.Tabs.__init__)
    assert "id" in _params(gr.Tab.__init__)


def test_launch_can_serve_project_files():
    """Audio players load segment WAVs straight from the project folder."""
    assert "allowed_paths" in _params(gr.Blocks.launch)
