from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestMLXTTSWrapper:
    def test_mlx_wrapper_exists(self):
        from mazinger.tts import _MLXTTSWrapper

        assert _MLXTTSWrapper is not None

    def test_mlx_wrapper_engine_attribute(self):
        from mazinger.tts import _MLXTTSWrapper

        assert _MLXTTSWrapper.engine == "mlx"

    def test_mlx_wrapper_init(self):
        from mazinger.tts import _MLXTTSWrapper

        model = MagicMock()
        wrapper = _MLXTTSWrapper(model, "ref.wav", "hello world")
        assert wrapper.model is model
        assert wrapper.ref_audio_path == "ref.wav"
        assert wrapper.ref_text == "hello world"

    def test_mlx_wrapper_init_no_ref_text(self):
        from mazinger.tts import _MLXTTSWrapper

        model = MagicMock()
        wrapper = _MLXTTSWrapper(model, "ref.wav")
        assert wrapper.ref_text is None

    def test_mlx_wrapper_synthesize_returns_numpy_tuple(self):
        from mazinger.tts import _MLXTTSWrapper

        model = MagicMock()
        result = MagicMock()
        result.audio = np.array([0.1, 0.2, 0.3])
        result.sample_rate = 24000
        model.generate.return_value = [result]
        wrapper = _MLXTTSWrapper(model, "ref.wav", "hello")
        audio, sr = wrapper.synthesize("test text", "English")
        assert isinstance(audio, np.ndarray)
        assert isinstance(sr, int)
        assert sr == 24000

    def test_mlx_wrapper_synthesize_calls_generate_with_correct_args(self):
        from mazinger.tts import _MLXTTSWrapper

        model = MagicMock()
        result = MagicMock()
        result.audio = np.array([0.1])
        result.sample_rate = 24000
        model.generate.return_value = [result]
        wrapper = _MLXTTSWrapper(model, "ref.wav", "hello")
        wrapper.synthesize("test text", "English")
        model.generate.assert_called_once_with(
            text="test text",
            ref_audio="ref.wav",
            ref_text="hello",
            lang_code="auto",
        )

    def test_mlx_wrapper_synthesize_default_sample_rate(self):
        from mazinger.tts import _MLXTTSWrapper

        model = MagicMock()
        result = MagicMock()
        result.audio = np.array([0.1])
        del result.sample_rate
        model.generate.return_value = [result]
        wrapper = _MLXTTSWrapper(model, "ref.wav", "hello")
        audio, sr = wrapper.synthesize("test text")
        assert sr == 24000

    def test_mlx_wrapper_unload_calls_gc_and_removes_from_cache(self):
        from mazinger.tts import _MLXTTSWrapper

        model = MagicMock()
        wrapper = _MLXTTSWrapper(model, "ref.wav", "hello")
        with (
            patch("mazinger.tts.gc.collect") as mock_gc,
            patch("mazinger.tts._remove_from_cache") as mock_remove,
        ):
            wrapper.unload()
            mock_gc.assert_called_once()
            mock_remove.assert_called_once_with(model)

    def test_load_model_mlx_qwen_branch(self):
        from mazinger.tts import load_model, _model_cache

        _model_cache.clear()
        with patch("mazinger.tts._load_mlx_model") as mock_load:
            mock_load.return_value = MagicMock()
            model = load_model(engine="mlx", mlx_model="mlx-test/Model")
            mock_load.assert_called_once_with("mlx-test/Model")
            assert model is mock_load.return_value

    def test_create_voice_prompt_mlx_qwen_returns_wrapper(self):
        from mazinger.tts import create_voice_prompt, _MLXTTSWrapper

        model = MagicMock()
        wrapper = create_voice_prompt(
            model,
            "ref.wav",
            "hello",
            engine="mlx",
        )
        assert isinstance(wrapper, _MLXTTSWrapper)
        assert wrapper.engine == "mlx"
        assert wrapper.ref_audio_path == "ref.wav"
        assert wrapper.ref_text == "hello"
