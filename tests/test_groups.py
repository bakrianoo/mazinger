import sys
import types
import importlib
from unittest.mock import patch


def reload_groups():
    import mazinger.cli._groups as g

    importlib.reload(g)
    return g


def test_detect_device_cuda_available():
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        g = reload_groups()
        assert g.detect_device() == "cuda"


def test_detect_device_mlx_available_when_cuda_unavailable():
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_mlx_pkg = types.ModuleType("mlx")
    fake_mlx_core = types.ModuleType("mlx.core")
    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "mlx": fake_mlx_pkg,
            "mlx.core": fake_mlx_core,
        },
    ):
        g = reload_groups()
        assert g.detect_device() == "mlx"


def test_detect_device_mps_available_when_cuda_mlx_unavailable():
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: True)
    )
    with patch.dict(sys.modules, {"torch": fake_torch}):
        if "mlx.core" in sys.modules:
            del sys.modules["mlx.core"]
        g = reload_groups()
        assert g.detect_device() == "mps"


def test_detect_device_cpu_fallback_when_no_cuda_no_mlx_no_mps():
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    with patch.dict(sys.modules, {"torch": fake_torch}):
        if "mlx.core" in sys.modules:
            del sys.modules["mlx.core"]
        g = reload_groups()
        assert g.detect_device() == "cpu"


def test_no_torch_results_in_cpu_when_without_mlx():
    if "torch" in sys.modules:
        del sys.modules["torch"]
    if "mlx.core" in sys.modules:
        del sys.modules["mlx.core"]
    import builtins
    import types as _types

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return orig_import(name, *args, **kwargs)

    from unittest.mock import patch as _patch

    with _patch("builtins.__import__", side_effect=fake_import):
        g = reload_groups()
        assert g.detect_device() == "cpu"
