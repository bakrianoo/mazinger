"""Tests for OmniVoice backend."""

import sys
import types
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from mazinger.tts import _OmniVoiceWrapper, load_model


def test_omnivoice_synthesize_kwargs():
    """Test that OmniVoice synthesize calls generate with correct kwargs."""
    model = MagicMock()
    # Mock return value of generate to be a list containing one np.ndarray
    model.generate.return_value = [np.array([0.1, 0.2, 0.3])]

    wrapper = _OmniVoiceWrapper(model, "ref.wav", "hello")
    audio, sr = wrapper.synthesize("test text", "English")

    assert sr == 24000
    np.testing.assert_array_equal(audio, np.array([0.1, 0.2, 0.3]))

    model.generate.assert_called_once_with(
        text="test text",
        ref_audio="ref.wav",
        ref_text="hello",
        language_id="en"
    )


def test_omnivoice_synthesize_without_ref_text():
    """Test that OmniVoice synthesize handles missing ref_text correctly."""
    model = MagicMock()
    model.generate.return_value = [np.array([0.4, 0.5])]

    wrapper = _OmniVoiceWrapper(model, "ref2.wav", None)
    audio, sr = wrapper.synthesize("test 2", "Arabic")

    model.generate.assert_called_once_with(
        text="test 2",
        ref_audio="ref2.wav",
        language_id="en"
    )
