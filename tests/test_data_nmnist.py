import importlib
import sys
import types
import pytest


def _reload_module(modname: str):
    if modname in sys.modules:
        del sys.modules[modname]
    return importlib.import_module(modname)


def test_import_module_without_optional_deps(monkeypatch):
    # Simulate absence of tonic and numpy; import should still succeed
    monkeypatch.delitem(sys.modules, "tonic", raising=False)
    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    # Ensure a clean import for the target module
    monkeypatch.delitem(sys.modules, "dch_data.nmnist", raising=False)

    mod = _reload_module("dch_data.nmnist")
    assert hasattr(mod, "NmnistLoader")


def test_loader_raises_when_tonic_missing_on_use(monkeypatch, tmp_path):
    # Ensure fresh module
    monkeypatch.delitem(sys.modules, "dch_data.nmnist", raising=False)
    mod = _reload_module("dch_data.nmnist")

    loader = mod.NmnistLoader(root=str(tmp_path), split="train", download=False)

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "tonic":
            raise ModuleNotFoundError("No module named 'tonic'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as excinfo:
        _ = loader.load_one(0)

    msg = str(excinfo.value)
    assert "pip install tonic" in msg
    assert "conda install -c conda-forge tonic" in msg


def test_numpy_missing_on_use_path(monkeypatch, tmp_path):
    # Ensure fresh module
    monkeypatch.delitem(sys.modules, "dch_data.nmnist", raising=False)
    mod = _reload_module("dch_data.nmnist")

    loader = mod.NmnistLoader(root=str(tmp_path), split="train", download=False)

    # Build a minimal tonic stub so the code progresses past tonic import
    tonic_stub = types.ModuleType("tonic")

    class _NMNISTStub:
        def __init__(self, save_to, train, download):
            self._data = [("events", 0)]

        def __getitem__(self, idx):
            return self._data[idx]

        def __len__(self):
            return len(self._data)

    tonic_stub.datasets = types.SimpleNamespace(NMNIST=_NMNISTStub)

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "tonic":
            return tonic_stub
        if name == "numpy":
            raise ModuleNotFoundError("No module named 'numpy'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as excinfo:
        _ = loader.load_one(0)

    msg = str(excinfo.value)
    assert "pip install numpy" in msg
    assert "conda install -c conda-forge numpy" in msg