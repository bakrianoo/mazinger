"""The Colab notebook is the on-ramp for users with no GPU, and it drifts easily.

Three things went wrong at once here and none of them are visible from reading
the notebook alone:

* it installed with plain ``pip``, which the README explicitly says cannot
  resolve ``mazinger[all]``;
* it pre-downloaded Faster Whisper while the Studio dropdown defaults to
  CohereX, so a new user waited on a ~1.5 GB model the default run never
  touches, then hit the gated-weights error anyway;
* Colab runs on a datacenter IP, which is exactly where YouTube withholds
  video formats.

Each assertion below pins one of those, plus the CLI flag that makes the
warm-up match the default.
"""

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "mazinger_colab.ipynb"


@pytest.fixture(scope="module")
def notebook():
    if not NOTEBOOK.exists():  # pragma: no cover — the file is checked in
        pytest.skip(f"notebook not found: {NOTEBOOK}")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code(notebook):
    """All code-cell source, concatenated."""
    return "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )


@pytest.fixture(scope="module")
def prose(notebook):
    """All markdown source, concatenated."""
    return "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )


class TestNotebookIsValid:
    def test_it_is_a_v4_notebook(self, notebook):
        assert notebook["nbformat"] == 4

    def test_colab_still_asks_for_the_t4(self, notebook):
        """Losing this metadata silently drops the user onto a CPU runtime."""
        meta = notebook["metadata"]
        assert meta.get("accelerator") == "GPU"
        assert meta.get("colab", {}).get("gpuType") == "T4"

    def test_every_cell_has_a_type_and_source(self, notebook):
        for i, cell in enumerate(notebook["cells"]):
            assert cell["cell_type"] in {"code", "markdown"}, i
            assert "".join(cell["source"]).strip(), f"cell {i} is empty"


class TestInstallCell:
    def test_mazinger_is_installed_with_uv(self, code):
        """mazinger[all] carries conflicting pins that only uv can resolve."""
        assert "uv pip install" in code

    def test_mazinger_is_not_installed_with_bare_pip(self, code):
        for line in code.splitlines():
            line = line.strip().lstrip("!").strip()
            if "mazinger[" in line and line.startswith("pip install"):
                pytest.fail(f"bare pip cannot resolve mazinger[all]: {line!r}")

    def test_uv_itself_is_bootstrapped(self, code):
        """Colab has no uv preinstalled, so `uv pip install` would be 'command not found'."""
        assert "pip install -q uv" in code or "pip install uv" in code

    def test_ffmpeg_is_installed(self, code):
        assert "ffmpeg" in code

    def test_a_po_token_provider_is_installed(self, code):
        """Colab is a datacenter IP; without this YouTube serves no formats."""
        assert "bgutil-ytdlp-pot-provider" in code


class TestLaunchCellMatchesTheStudioDefault:
    def test_it_warms_the_backend_the_studio_defaults_to(self, code):
        from mazinger.studio.constants import DEFAULT_TRANSCRIBE_LABEL, METHOD_MAP

        default = METHOD_MAP[DEFAULT_TRANSCRIBE_LABEL]
        assert default == "coherex", "default changed — update the notebook flag too"
        assert "--with-coherex" in code

    def test_it_does_not_warm_only_faster_whisper(self, code):
        """The old bug: downloading a model the default run never loads."""
        if "--with-faster-whisper" in code:
            assert "--with-coherex" in code

    def test_the_flag_it_passes_actually_exists(self, code):
        """A typo'd flag would fail at launch, after the long install cell."""
        from mazinger.cli import _build_parser

        parser = _build_parser()
        flags = [w for w in code.split() if w.startswith("--with-")]
        assert flags, "launch cell passes no --with-* flags"
        args = parser.parse_args(["web", *flags])
        assert args.with_coherex is True

    def test_ollama_is_still_set_up(self, code):
        assert "--with-ollama" in code


class TestNotebookExplainsTheGatedWeights:
    def test_it_mentions_the_hugging_face_sign_in(self, prose):
        """CohereX weights are gated — silence here is a dead end on first run."""
        assert "hugging face" in prose.lower()

    def test_it_names_the_no_sign_in_escape_hatch(self, prose):
        assert "Faster Whisper" in prose

    def test_it_links_the_gated_model_page(self, prose):
        from mazinger.transcribe import DEFAULT_COHEREX_MODEL

        assert DEFAULT_COHEREX_MODEL in prose
