"""Third-party source vendored into Mazinger.

Packages here are *not* Mazinger's own code.  They are upstream projects
copied in because their published distributions pin dependencies that
conflict with the rest of Mazinger's environment, and those pins cannot be
relaxed from the outside.

Nothing in this directory is public API.  Import the vendored modules through
the Mazinger wrapper that owns them (e.g. :mod:`mazinger.tts` for Qwen3-TTS)
rather than reaching in directly.

Contents
--------
``qwen_tts``
    Qwen3-TTS 0.1.1 (Apache-2.0), from https://github.com/Qwen/Qwen3-TTS.
    See ``qwen_tts/NOTICE.md`` for the full list of modifications.
"""
