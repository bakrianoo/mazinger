# Qwen3-TTS — vendored fork

This directory contains a modified copy of **Qwen3-TTS 0.1.1**.

- Upstream: <https://github.com/Qwen/Qwen3-TTS>
- Copyright 2026 The Alibaba Qwen Team
- Licensed under the Apache License, Version 2.0 (see `LICENSE`)

Mazinger itself is MIT-licensed; this subtree remains Apache-2.0. Per Apache-2.0
§4(b), the modifications made to the original files are listed below.

## Why this is vendored rather than installed

The published `qwen-tts` 0.1.1 distribution hard-pins `transformers==4.57.3` and
`accelerate==1.12.0`. Mazinger runs it alongside `coherex` (needs
`transformers>=5.4`) and `omnivoice` (needs `>=5.3`), so those pins cannot be
satisfied in one environment, and a dependency pin cannot be relaxed from
outside the package.

Mazinger previously worked around this by overriding the pins in
`[tool.uv] override-dependencies` and monkey-patching the package at import
time from `mazinger/_compat.py`. That approach only ever covered the three
import-time failures below; it did not cover the mask-API changes or the
`cache_position` regression, which surface later and — in the last case —
silently. Vendoring turns those patches into ordinary, reviewable source edits.

## Modifications

Every change is marked inline with a `# [mazinger]` comment, so
`grep -rn '\[mazinger\]'` over this directory yields the complete list.

### Compatibility with transformers 5.x

| # | Change | Files |
|---|--------|-------|
| 1 | `@check_model_inputs()` → `@check_model_inputs`. In 4.57 it was a decorator *factory*; in 5.x it takes the function directly, so the parenthesised form raises `TypeError` while the module is imported. | `core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` |
| 2 | `ROPE_INIT_FUNCTIONS[self.rope_type]` → `.get(self.rope_type, _compute_default_rope_parameters)`. 5.x dropped the `"default"` entry (unscaled RoPE now goes through `config.rope_parameters`), but every Qwen3-TTS submodel looks it up when `rope_scaling` is unset. The fallback is a verbatim port of `_compute_default_rope_parameters` from transformers 4.57.3, so checkpoints keep the frequencies they were trained against. | `core/models/modeling_qwen3_tts.py`, `core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` |
| 3 | `config.pad_token_id` → `getattr(config, "pad_token_id", None)`. 5.x moved the token-id attributes to `GenerationConfig`; the Qwen3-TTS checkpoints set `pad_token_id` to `null` or omit it, so `None` is what 4.57 resolved here anyway. | `core/models/modeling_qwen3_tts.py` |
| 4 | `create_causal_mask(input_embeds=…)` → `inputs_embeds=…`, matching the renamed parameter. | `core/models/modeling_qwen3_tts.py`, `core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` |
| 5 | Dropped `cache_position=` from `create_causal_mask` / `create_sliding_window_causal_mask` calls — removed from the 5.x signature (the role is now filled by `position_ids`). | `core/models/modeling_qwen3_tts.py`, `core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` |
| 6 | `Qwen3TTSTalkerForConditionalGeneration.forward` now reconstructs `cache_position` from `past_key_values` when it arrives as `None`. See below. | `core/models/modeling_qwen3_tts.py` |
| 7 | Rotary modules register `original_inv_freq` as a real buffer and expose a `compute_default_rope_parameters` static method. | `core/models/modeling_qwen3_tts.py`, `core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` |
| 8 | The two `_init_weights` overrides delegate to `super()._init_weights(module)` for rotary modules. | `core/models/modeling_qwen3_tts.py` |
| 9 | `Qwen3TTSTokenizerV2DecoderRotatoryEmbedding` renamed to `…RotaryEmbedding` (upstream typo), with an alias under the old name. | `core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` |

#### On changes 7–9 — the silent one

These three are one bug with three causes, and it is the reason a "just relax the
pin" approach produces a model that loads, runs, raises nothing, and generates
garbage.

transformers 5.x builds models on the meta device and then re-materialises every
non-persistent buffer through `PreTrainedModel._init_weights`. `inv_freq` — the
RoPE frequency table — is such a buffer. Its re-initialisation is guarded by:

```python
elif "RotaryEmbedding" in module.__class__.__name__ and hasattr(module, "original_inv_freq"):
    rope_fn = (ROPE_INIT_FUNCTIONS[module.rope_type] if module.rope_type != "default"
               else module.compute_default_rope_parameters)
```

Qwen3-TTS failed that guard three separate ways:

1. Its rotary modules assign `self.original_inv_freq = self.inv_freq` as a plain
   attribute and never define `compute_default_rope_parameters` (change 7).
2. `Qwen3TTSPreTrainedModel` and `Qwen3TTSTalkerTextPreTrainedModel` both
   *override* `_init_weights` outright, so the base implementation — the only
   thing that restores the buffer — never runs (change 8).
3. The 12 Hz decoder's rotary class is spelled `RotatoryEmbedding`, which does
   not contain the substring `RotaryEmbedding`, so the name test fails (change 9).

The result: `inv_freq` stayed all zeros, so `cos ≡ 1` and `sin ≡ 0` and RoPE
became the identity — the model ran with **no positional information at all**.
Nothing raises. Weights load, shapes match, audio comes out. It is simply wrong:
generation never emits EOS and runs to `max_new_tokens` (~170 s of babble for a
one-sentence input).

#### On change 6

This is the only change that is not a mechanical rename, and the only one whose
absence corrupts output rather than raising at the call site.

`Qwen3TTSTalkerForConditionalGeneration.forward` reads `cache_position` to tell
prefill from decode:

```python
if cache_position is None or cache_position[0] == 0 or self.rope_deltas is None:
    position_ids, rope_deltas = self.get_rope_index(attention_mask)   # full length
else:
    delta = cache_position[0] + self.rope_deltas                      # incremental
```

transformers 4.57 always supplied `cache_position` from `generate()`. 5.x no
longer threads it into `forward`, so it arrives as `None` on *every* step and
the prefill branch is taken unconditionally. A single-token decode step then
gets `position_ids` for the whole sequence; RoPE broadcasts the 1-token query up
to the full length, and the corruption only surfaces further downstream as a
mask/key length mismatch inside attention (`Expected size … [16, 21] but got
[16, 11]`).

Since `cache_position[0]` is just the number of already-cached tokens, rebuilding
it from `past_key_values.get_seq_length()` restores the original semantics
exactly, and leaves the upstream branch untouched.

### Dependency trimming

| Change | Rationale |
|--------|-----------|
| `core/__init__.py` resolves the 25 Hz tokenizer lazily (PEP 562), and `inference/qwen3_tts_tokenizer.py` registers it through `_register_25hz()`, which degrades to a no-op when the import fails. | The 25 Hz tokenizer imports `sox` and `onnxruntime` at module scope, and `sox` additionally requires the SoX *binary* on `PATH`. Mazinger only loads 12 Hz models, which pulled that chain in transitively. Behaviour is unchanged when the packages are present. |
| Deleted `cli/demo.py` (and the `qwen-tts-demo` entry point). | Its only purpose was a standalone Gradio demo, and it was the sole reason `gradio` appeared in this package's dependencies. Mazinger has its own UI. |
| Deleted `__main__.py`. | Advertised the removed CLI entry points. |
| Added empty `__init__.py` to `inference/`, `core/tokenizer_12hz/`, `core/tokenizer_25hz/`, `core/tokenizer_25hz/vq/`, `core/tokenizer_25hz/vq/assets/`. | Upstream relied on PEP 420 namespace packages; explicit `__init__.py` files make the subpackages discoverable by `setuptools.find_packages` so they are included in the wheel. |

## Supported transformers range

The fork calls `create_causal_mask` with the signature introduced in
**transformers 5.4**. The API moved twice inside the 5.x line, so the floor is
not arbitrary:

| transformers | `create_causal_mask` signature | works? |
|---|---|---|
| 5.0 – 5.3 | `input_embeds` (singular), `cache_position` **required** | ✗ |
| 5.4 – 5.8 | `inputs_embeds`, `cache_position` optional and unused, `past_key_values`/`position_ids` keyword-only | ✓ |
| 5.9 – 5.15 | `cache_position` removed entirely | ✓ |

`ROPE_INIT_FUNCTIONS` lacks the `"default"` key and `_init_weights` gates rotary
re-initialisation on `original_inv_freq` across the *whole* 5.x line, so changes
2 and 7–9 are required for every supported version, not just recent ones.
`check_model_inputs` takes the function directly in every 5.x release, so the
bare decorator (change 1) is correct throughout.

Hence `transformers>=5.4` in the `tts` extra. Verified end-to-end on 5.13.1;
the table above was derived by inspecting each release's signature.

Two announced deprecations will eventually need attention: `layer_type_validation`
(imported by `configuration_qwen3_tts.py`) is slated for removal in transformers
5.20, and `check_model_inputs` is superseded by `merge_with_config_defaults`.
Both would fail loudly at import, not silently.

## Upgrading

When a new upstream release lands, prefer dropping the vendored tree and
returning to the PyPI package **if** the `transformers` pin has been relaxed.
Otherwise, re-vendor and re-apply the changes above — `grep -rn '\[mazinger\]'`
enumerates every site.
