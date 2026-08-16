"""Runtime shims for third-party packages that lag behind their dependencies.

Mazinger overrides several upstream version pins (see ``[tool.uv]
override-dependencies`` in ``pyproject.toml``) so engines with mutually
exclusive requirements can share one environment. When a package then trips
over the newer dependency, the fix belongs here — applied lazily, right before
the offending import — rather than as a pin that would break another engine.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def patch_check_model_inputs() -> None:
    """Make ``@check_model_inputs()`` work on transformers 5.x.

    ``qwen-tts`` 0.1.1 targets transformers 4.57, where ``check_model_inputs``
    was a decorator *factory* invoked with parentheses. In transformers 5.x it
    is a plain decorator that takes the function directly, so the package's
    ``@check_model_inputs()`` raises ``TypeError: check_model_inputs() missing
    1 required positional argument: 'func'`` while its model module is being
    imported.

    Wrapping it to accept both call styles keeps transformers current (coherex
    requires >= 5.4) without patching site-packages. Idempotent.
    """
    from transformers.utils import generic

    original = generic.check_model_inputs
    if getattr(original, "_mazinger_dual_style", False):
        return

    def check_model_inputs(func=None, **kwargs):
        if callable(func):
            return original(func)  # bare: @check_model_inputs
        if kwargs:
            log.debug("Ignoring legacy check_model_inputs kwargs: %s", sorted(kwargs))
        return lambda f: original(f)  # called: @check_model_inputs(...)

    check_model_inputs._mazinger_dual_style = True
    generic.check_model_inputs = check_model_inputs
    log.debug("Patched transformers check_model_inputs for legacy call style")


#: Config classes qwen-tts reads ``pad_token_id`` from (as ``padding_idx``).
_QWEN_TTS_CONFIGS = ("Qwen3TTSTalkerConfig", "Qwen3TTSTalkerCodePredictorConfig")


def patch_qwen_tts_config_defaults() -> None:
    """Restore the ``pad_token_id`` config default qwen-tts still expects.

    transformers 4.x gave every ``PretrainedConfig`` a ``pad_token_id``
    defaulting to ``None``; 5.x moved the token-id attributes to
    ``GenerationConfig``, so reading one that the checkpoint does not define
    now raises ``AttributeError: 'Qwen3TTSTalkerConfig' object has no
    attribute 'pad_token_id'`` during ``from_pretrained``.

    The Qwen3-TTS checkpoints set ``pad_token_id`` to ``null`` (code predictor)
    or omit it (talker), so ``None`` is exactly what transformers 4.x resolved
    here — this restores the original ``padding_idx=None`` rather than guessing
    a value. Scoped to qwen-tts's own config classes so no other model's
    ``hasattr`` checks are affected. Call after importing ``qwen_tts``.
    """
    from qwen_tts.core.models import configuration_qwen3_tts as cfg

    for name in _QWEN_TTS_CONFIGS:
        klass = getattr(cfg, name, None)
        if klass is not None and "pad_token_id" not in vars(klass):
            klass.pad_token_id = None
            log.debug("Restored %s.pad_token_id default (None)", name)


def patch_default_rope() -> None:
    """Re-register the ``"default"`` RoPE initialiser transformers 5.x dropped.

    ``ROPE_INIT_FUNCTIONS`` lost its ``"default"`` entry in transformers 5.0
    (unscaled RoPE is now computed through ``config.rope_parameters``), but
    qwen-tts still looks it up whenever ``rope_scaling`` is unset — every
    Qwen3-TTS submodel — and dies with ``KeyError: 'default'``.

    The body below is a verbatim port of ``_compute_default_rope_parameters``
    from transformers 4.57.3, the version qwen-tts pins, so the frequencies
    match what the checkpoint was built against rather than a reimplementation.
    Only the missing key is added; transformers 5.x no longer reads it, so its
    own models are unaffected. Idempotent.
    """
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    log.debug("Registered legacy 'default' RoPE initialiser")


def patch_qwen_tts() -> None:
    """Apply every shim qwen-tts 0.1.1 needs to run on transformers 5.x."""
    patch_check_model_inputs()  # must precede the import (decorator runs at class body)
    import qwen_tts  # noqa: F401

    patch_qwen_tts_config_defaults()
    patch_default_rope()
