"""CohereX is the Studio's default transcription backend.

The Gradio dropdown takes its default from the first entry of ``METHOD_MAP``,
so the ordering is load-bearing rather than cosmetic — reordering the dict
silently changes what every new user transcribes with.

CohereX also carries two prerequisites the Whisper backends do not (gated
weights, no language auto-detection), so the tests below also pin the guard
that turns a missing sign-in into an actionable message instead of a raw
HuggingFace 401 from inside the model loader.
"""

import pytest

from mazinger.studio import constants as C


class TestStudioDefault:
    def test_default_label_is_the_first_dropdown_entry(self):
        """The Gradio dropdown shows METHOD_MAP in order and defaults to [0]."""
        assert C.DEFAULT_TRANSCRIBE_LABEL == next(iter(C.METHOD_MAP))

    def test_default_resolves_to_coherex(self):
        assert C.METHOD_MAP[C.DEFAULT_TRANSCRIBE_LABEL] == "coherex"

    def test_every_label_maps_to_a_known_backend(self):
        from mazinger.transcribe import TranscribeMethod
        from typing import get_args

        valid = set(get_args(TranscribeMethod))
        assert set(C.METHOD_MAP.values()) <= valid

    def test_faster_whisper_is_still_offered(self):
        """The no-sign-in fallback must stay reachable from the UI."""
        assert "faster-whisper" in C.METHOD_MAP.values()

    def test_coherex_source_languages_are_all_selectable(self):
        """A CohereX language the dropdown cannot offer is unusable."""
        from mazinger.studio.constants import COHEREX_SOURCE_LANGUAGES

        assert COHEREX_SOURCE_LANGUAGES, "expected a non-empty list"

    def test_app_binds_the_dropdown_to_the_shared_constant(self):
        """A literal default in app.py would drift from METHOD_MAP unnoticed.

        Every other test here checks the constant; this is the one that checks
        the Gradio widget actually uses it.
        """
        import pathlib

        source = (pathlib.Path(__file__).resolve().parents[1]
                  / "mazinger" / "studio" / "app.py").read_text(encoding="utf-8")
        assert "value=DEFAULT_TRANSCRIBE_LABEL" in source


class TestOfferedLanguagesMatchTheBackend:
    """Cohere Transcribe never errors on a wrong language — it transcribes
    confidently in whichever one it is handed. Both directions matter: an
    unsupported option in the dropdown yields fluent nonsense, and a missing
    one makes a supported language unreachable."""

    @staticmethod
    def _offered():
        from mazinger.translate import lang_code_from_name

        return {lang_code_from_name(name) for name in C.COHEREX_SOURCE_LANGUAGES}

    def test_no_language_is_offered_that_coherex_cannot_transcribe(self):
        from mazinger.transcribe import COHEREX_LANGUAGES

        extra = sorted(self._offered() - set(COHEREX_LANGUAGES))
        assert not extra, f"Studio offers languages CohereX cannot transcribe: {extra}"

    def test_no_supported_language_is_missing_from_the_dropdown(self):
        from mazinger.transcribe import COHEREX_LANGUAGES

        missing = sorted(set(COHEREX_LANGUAGES) - self._offered())
        assert not missing, f"Studio omits languages CohereX supports: {missing}"


class TestAllExtraShipsCoherex:
    """The Studio default is only usable if ``mazinger[all]`` installs it."""

    @staticmethod
    def _extras():
        import pathlib
        import sys

        if sys.version_info >= (3, 11):
            import tomllib
        else:  # pragma: no cover - 3.10 has no tomllib
            tomllib = pytest.importorskip("tomli")

        path = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
        with path.open("rb") as fh:
            return tomllib.load(fh)["project"]["optional-dependencies"]

    def test_all_pulls_in_transcribe_coherex(self):
        assert "transcribe-coherex" in " ".join(self._extras()["all"])

    def test_all_coherex_alias_still_resolves(self):
        """Existing install commands and docs must keep working."""
        assert "all-coherex" in self._extras()

    def test_qwen_tts_is_not_installed_from_pypi(self):
        """It is vendored — the PyPI release pins an unusable transformers."""
        names = [d.split("[")[0].split(">")[0].split("=")[0].strip()
                 for d in self._extras()["tts"]]
        assert "qwen-tts" not in names


class TestColabCopyStaysInSync:
    """`docs/notebooks/studio` is fetched verbatim by the Colab notebook."""

    @staticmethod
    def _colab_constants():
        import pathlib

        path = pathlib.Path(__file__).resolve().parents[1] / "docs/notebooks/studio/constants.py"
        if not path.is_file():
            pytest.skip("Colab Studio copy not present")
        namespace: dict = {}
        source = path.read_text(encoding="utf-8")
        # Execute only up to the trailing package import the notebook copy makes.
        source = source.split("from mazinger.ollama_setup")[0]
        exec(compile(source, str(path), "exec"), namespace)  # noqa: S102
        return namespace

    def test_colab_copy_defaults_to_coherex_too(self):
        ns = self._colab_constants()
        assert ns["METHOD_MAP"][ns["DEFAULT_TRANSCRIBE_LABEL"]] == "coherex"


class TestGatedCredentialGuard:
    def test_explicit_token_is_accepted(self):
        from mazinger.transcribe import _require_hf_credentials

        _require_hf_credentials("hf_explicit")

    def test_a_stored_studio_login_is_accepted(self, monkeypatch):
        """Signing in via the Studio stores a token; HF_TOKEN must not be required."""
        import huggingface_hub

        from mazinger.transcribe import _require_hf_credentials

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_stored")
        _require_hf_credentials(None)

    def test_no_credentials_raises_something_actionable(self, monkeypatch):
        import huggingface_hub

        from mazinger.transcribe import _require_hf_credentials

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)

        with pytest.raises(RuntimeError) as exc:
            _require_hf_credentials(None)

        message = str(exc.value)
        assert "Sign in with Hugging Face" in message   # the Studio route
        assert "HF_TOKEN" in message                    # the CLI route
        assert "faster-whisper" in message              # the escape hatch


class TestWebWarmupMatchesTheDefault:
    """`mazinger web` should pre-download what the Studio actually uses.

    Warming Faster Whisper while the dropdown defaults to CohereX cost a new
    user a ~1.5 GB download of a model the default run never loads, and then
    stalled again on the CohereX weights at the first mission.
    """

    def test_there_is_a_flag_to_warm_the_default_backend(self):
        from mazinger.cli import _build_parser

        args = _build_parser().parse_args(["web", "--with-coherex"])
        assert args.with_coherex is True

    def test_the_help_hint_matches_the_real_default_model(self):
        """--help names the repo; a stale string here misleads about what downloads."""
        from mazinger.cli._web import _DEFAULT_COHEREX_HINT
        from mazinger.transcribe import DEFAULT_COHEREX_MODEL

        assert _DEFAULT_COHEREX_HINT == DEFAULT_COHEREX_MODEL

    def test_warmup_is_skipped_without_credentials(self, monkeypatch, caplog):
        """No sign-in must not abort the launch — the UI can still sign in."""
        import huggingface_hub

        from mazinger.cli import _web

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)

        def _boom(*a, **k):  # pragma: no cover — must never be reached
            raise AssertionError("attempted a gated download with no token")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", _boom)

        with caplog.at_level("WARNING"):
            _web._setup_coherex(None)   # must not raise

        assert "hugging face" in caplog.text.lower()

    def test_warmup_downloads_the_default_model_when_signed_in(self, monkeypatch):
        import huggingface_hub

        from mazinger.cli import _web
        from mazinger.transcribe import DEFAULT_COHEREX_MODEL

        seen = []
        monkeypatch.setenv("HF_TOKEN", "hf_dummy")
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_dummy")
        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", lambda repo, **k: seen.append(repo)
        )

        _web._setup_coherex(None)
        assert seen == [DEFAULT_COHEREX_MODEL]

    def test_a_failed_download_does_not_abort_the_launch(self, monkeypatch, caplog):
        """Gated terms not accepted yet is a warning, not a crash."""
        import huggingface_hub

        from mazinger.cli import _web

        monkeypatch.setenv("HF_TOKEN", "hf_dummy")
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_dummy")

        def _403(*a, **k):
            raise RuntimeError("403 Forbidden: gated repo")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", _403)

        with caplog.at_level("WARNING"):
            _web._setup_coherex(None)   # must not raise

        assert "403" in caplog.text
