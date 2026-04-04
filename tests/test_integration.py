import argparse
import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestMLXCLIIntegration:
    def test_add_tts_engine_accepts_mlx_qwen(self):
        from mazinger.cli._groups import add_tts_engine

        p = argparse.ArgumentParser()
        add_tts_engine(p)
        args = p.parse_args(["--tts-engine", "mlx"])
        assert args.tts_engine == "mlx"

    def test_add_tts_engine_has_mlx_model_flag(self):
        from mazinger.cli._groups import add_tts_engine

        p = argparse.ArgumentParser()
        add_tts_engine(p)
        args = p.parse_args(["--mlx-model", "mlx-community/Test-Model"])
        assert args.mlx_model == "mlx-community/Test-Model"

    def test_add_tts_engine_mlx_model_default(self):
        from mazinger.cli._groups import add_tts_engine

        p = argparse.ArgumentParser()
        add_tts_engine(p)
        args = p.parse_args([])
        assert "mlx-community" in args.mlx_model
        assert "0.6B" in args.mlx_model

    def test_detect_device_returns_valid_string(self):
        from mazinger.cli._groups import detect_device

        result = detect_device()
        assert result in ("cuda", "mlx", "mps", "cpu")

    def test_resolve_device_auto_uses_detect_device(self):
        from mazinger.cli._groups import resolve_device, detect_device

        assert resolve_device("auto") == detect_device()

    def test_resolve_device_passes_through_explicit(self):
        from mazinger.cli._groups import resolve_device

        assert resolve_device("cuda") == "cuda"
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("mlx") == "mlx"


class TestMLXPipelineIntegration:
    def test_pipeline_dub_accepts_mlx_model_param(self):
        from mazinger.pipeline import MazingerDubber

        sig = inspect.signature(MazingerDubber.dub)
        assert "mlx_model" in sig.parameters

    def test_tts_load_model_accepts_mlx_model(self):
        from mazinger.tts import load_model

        sig = inspect.signature(load_model)
        assert "mlx_model" in sig.parameters

    def test_tts_create_voice_prompt_accepts_mlx_model(self):
        from mazinger.tts import create_voice_prompt

        sig = inspect.signature(create_voice_prompt)
        assert "mlx_model" in sig.parameters

    def test_full_tts_flow_mlx_qwen_with_mocks(self):
        from mazinger.tts import (
            load_model,
            create_voice_prompt,
            _MLXTTSWrapper,
            _model_cache,
        )

        _model_cache.clear()
        fake_model = MagicMock()
        fake_audio = np.array([0.1, 0.2])
        fake_result = MagicMock()
        fake_result.audio = fake_audio
        fake_result.sample_rate = 24000
        fake_model.generate.return_value = [fake_result]
        with patch("mazinger.tts._load_mlx_model", return_value=fake_model):
            model = load_model(engine="mlx", mlx_model="mlx-test/Model")
            wrapper = create_voice_prompt(
                model,
                "ref.wav",
                "hello",
                engine="mlx",
            )
            assert isinstance(wrapper, _MLXTTSWrapper)
            audio, sr = wrapper.synthesize("test text", "English")
            assert isinstance(audio, np.ndarray)
            assert isinstance(sr, int)
