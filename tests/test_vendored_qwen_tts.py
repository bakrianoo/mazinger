"""Guards for the vendored Qwen3-TTS fork (``mazinger/_vendor/qwen_tts``).

These cover the compatibility contract with transformers 5.x that upstream
qwen-tts does not satisfy.  They are deliberately CPU-only and download no
weights, because the failure they protect against is *silent*: when the RoPE
frequency buffers are not re-materialised the model still loads, still runs and
still emits audio — it just has no positional information, so generation never
terminates and the output is babble.  Nothing raises, so only an explicit test
catches a regression here.

See ``mazinger/_vendor/qwen_tts/NOTICE.md`` for the full rationale.
"""

import inspect
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mazinger._vendor.qwen_tts.core.models import modeling_qwen3_tts as M  # noqa: E402
from mazinger._vendor.qwen_tts.core.models.configuration_qwen3_tts import (  # noqa: E402
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from mazinger._vendor.qwen_tts.core.tokenizer_12hz import (  # noqa: E402
    modeling_qwen3_tts_tokenizer_v2 as T,
)
from mazinger._vendor.qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (  # noqa: E402
    Qwen3TTSTokenizerV2DecoderConfig,
)

#: Every rotary module in the vendored tree, with a config it can be built from.
ROTARY_CASES = [
    (M.Qwen3TTSTalkerRotaryEmbedding, Qwen3TTSTalkerConfig),
    (M.Qwen3TTSRotaryEmbedding, Qwen3TTSTalkerCodePredictorConfig),
    (T.Qwen3TTSTokenizerV2DecoderRotaryEmbedding, Qwen3TTSTokenizerV2DecoderConfig),
]
ROTARY_IDS = [cls.__name__ for cls, _ in ROTARY_CASES]


@pytest.mark.parametrize(("rotary_cls", "config_cls"), ROTARY_CASES, ids=ROTARY_IDS)
def test_rotary_class_name_matches_transformers_guard(rotary_cls, config_cls):
    """transformers keys buffer re-init off the literal class name.

    ``PreTrainedModel._init_weights`` restores ``inv_freq`` only for modules whose
    class name contains ``"RotaryEmbedding"``.  Upstream spelled one of these
    ``RotatoryEmbedding``, which silently excluded it.
    """
    assert "RotaryEmbedding" in rotary_cls.__name__


def test_misspelled_rotary_alias_still_resolves():
    """The upstream (misspelled) name is kept as an alias for compatibility."""
    assert (
        T.Qwen3TTSTokenizerV2DecoderRotatoryEmbedding
        is T.Qwen3TTSTokenizerV2DecoderRotaryEmbedding
    )


@pytest.mark.parametrize(("rotary_cls", "config_cls"), ROTARY_CASES, ids=ROTARY_IDS)
def test_rotary_module_exposes_reinit_hook(rotary_cls, config_cls):
    """The module must satisfy both halves of the transformers 5.x contract.

    ``original_inv_freq`` has to be a *registered buffer* (upstream assigned it as
    a plain attribute), and ``compute_default_rope_parameters`` has to exist and be
    callable as ``module.compute_default_rope_parameters(module.config)``.
    """
    module = rotary_cls(config_cls())

    assert "original_inv_freq" in module._buffers, "must be a registered buffer"
    assert callable(getattr(module, "compute_default_rope_parameters", None))

    inv_freq, _ = module.compute_default_rope_parameters(module.config)
    assert torch.isfinite(inv_freq).all()
    assert (inv_freq != 0).all(), "a zeroed table makes RoPE the identity"


@pytest.mark.parametrize(("rotary_cls", "config_cls"), ROTARY_CASES, ids=ROTARY_IDS)
def test_rotary_buffers_are_populated_on_construction(rotary_cls, config_cls):
    module = rotary_cls(config_cls())
    assert (module.inv_freq != 0).all()
    assert torch.equal(module.inv_freq, module.original_inv_freq)


def test_default_rope_matches_transformers_4_57_reference():
    """Golden values from transformers 4.57.3, the release upstream pins.

    Computed for the Qwen3-TTS talker geometry (``rope_theta=1e6``,
    ``head_dim=128``); these are the values the checkpoints were trained against,
    so drift here means every generation is subtly wrong.
    """
    config = Qwen3TTSTalkerConfig(rope_theta=1_000_000.0, head_dim=128)
    inv_freq, attention_factor = M._compute_default_rope_parameters(config)

    assert attention_factor == 1.0
    assert inv_freq.shape == (64,)
    torch.testing.assert_close(
        inv_freq[:3],
        torch.tensor([1.0, 0.8058422207832336, 0.6493816375732422]),
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "model_cls",
    [M.Qwen3TTSPreTrainedModel, M.Qwen3TTSTalkerTextPreTrainedModel],
    ids=["Qwen3TTSPreTrainedModel", "Qwen3TTSTalkerTextPreTrainedModel"],
)
def test_init_weights_override_restores_rotary_buffers(model_cls):
    """Both overrides must hand rotary modules back to the base implementation.

    transformers 5.x materialises models on the meta device and rebuilds
    non-persistent buffers through ``_init_weights``.  These two classes override
    it, so without an explicit delegation the base implementation — the only code
    that restores ``inv_freq`` — never runs.
    """
    rotary = M.Qwen3TTSTalkerRotaryEmbedding(Qwen3TTSTalkerConfig())
    expected = rotary.inv_freq.clone()

    # Simulate the post-meta-init state: buffer allocated but never filled.
    with torch.no_grad():
        rotary.inv_freq.zero_()
        rotary.original_inv_freq.zero_()
    assert not rotary.inv_freq.any()

    # `super()` needs a genuine instance, but not an initialised one.
    fake = object.__new__(model_cls)
    object.__setattr__(fake, "config", Qwen3TTSTalkerConfig())
    model_cls._init_weights(fake, rotary)

    assert rotary.inv_freq.any(), "rotary buffer was not re-materialised"
    torch.testing.assert_close(rotary.inv_freq, expected)


def test_init_weights_still_initialises_ordinary_modules():
    """Delegating the rotary case must not bypass the normal weight init."""
    linear = torch.nn.Linear(8, 8)
    with torch.no_grad():
        linear.weight.fill_(123.0)
        linear.bias.fill_(123.0)

    fake = object.__new__(M.Qwen3TTSTalkerTextPreTrainedModel)
    object.__setattr__(fake, "config", Qwen3TTSTalkerConfig())
    M.Qwen3TTSTalkerTextPreTrainedModel._init_weights(fake, linear)

    assert not torch.allclose(linear.weight, torch.full_like(linear.weight, 123.0))
    assert torch.equal(linear.bias, torch.zeros_like(linear.bias))


def test_mazinger_loads_the_vendored_engine_not_the_pypi_package():
    """`mazinger.tts` must not fall back to the conflicting PyPI distribution."""
    source = inspect.getsource(inspect.getmodule(_qwen_loader()))
    assert "from mazinger._vendor.qwen_tts import" in source
    assert "\n    from qwen_tts import" not in source


def _qwen_loader():
    from mazinger import tts

    return tts._load_qwen_model


def test_vendored_tree_carries_its_licence_and_notice():
    """Apache-2.0 §4 requires shipping the licence and stating modifications."""
    root = pathlib.Path(M.__file__).parents[2]   # .../mazinger/_vendor/qwen_tts
    assert (root / "LICENSE").is_file()
    notice = root / "NOTICE.md"
    assert notice.is_file()
    assert "Apache" in notice.read_text(encoding="utf-8")


def test_installed_transformers_is_within_the_supported_range():
    """The fork targets the `create_causal_mask` signature added in transformers 5.4.

    5.0–5.3 spell the parameter `input_embeds` and require `cache_position`, so the
    vendored call raises there.  This asserts the floor declared by the ``tts``
    extra actually matches what the code calls, and that the call still type-checks
    against whatever transformers is installed — a signature change is how this
    broke the last two times.
    """
    import inspect as _inspect

    from packaging.version import Version
    from transformers import __version__ as tf_version
    from transformers.masking_utils import create_causal_mask

    assert Version(tf_version) >= Version("5.4"), (
        f"transformers {tf_version} predates the create_causal_mask signature "
        "this fork targets; see mazinger/_vendor/qwen_tts/NOTICE.md"
    )

    params = _inspect.signature(create_causal_mask).parameters
    assert "inputs_embeds" in params, "renamed parameter — vendored call sites need updating"
    for name in ("config", "attention_mask", "past_key_values", "position_ids"):
        assert name in params, f"{name} missing from create_causal_mask signature"
    # We deliberately omit `cache_position`; it must not be mandatory.
    required = [
        n for n, p in params.items()
        if p.default is _inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert "cache_position" not in required


def test_importing_the_engine_does_not_require_sox_or_onnxruntime():
    """Those are 25 Hz-only dependencies; Mazinger loads 12 Hz models exclusively.

    ``sox`` also needs the SoX *binary* on PATH, so an eager import turns a
    missing system package into an import-time failure for every user.
    """
    import subprocess
    import sys

    program = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name.split('.')[0] in ('sox', 'onnxruntime') else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError(name + ' is blocked')\n"
        "sys.meta_path.insert(0, Block())\n"
        "from mazinger._vendor.qwen_tts import Qwen3TTSModel\n"
        "assert 'sox' not in sys.modules and 'onnxruntime' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(pathlib.Path(M.__file__).parents[5]),   # repo root, so `mazinger` imports
    )
    assert result.returncode == 0, result.stderr[-2000:]
