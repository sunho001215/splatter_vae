"""Fail-closed prebuilt loading must never enter gsplat's source-build fallback."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

import s4d.model.render as render


@pytest.fixture(autouse=True)
def uncached_renderer():
    render.require_prebuilt_renderer.cache_clear()
    yield
    render.require_prebuilt_renderer.cache_clear()


def _installed_package_without_initializer(monkeypatch, directory):
    attempted_imports = []
    spec_calls = []
    real_import = builtins.__import__
    real_find_spec = render.importlib.util.find_spec

    def guarded_import(name, *args, **kwargs):
        if name == "gsplat" or name.startswith("gsplat."):
            attempted_imports.append(name)
            raise AssertionError("gsplat initializer/JIT must not execute")
        return real_import(name, *args, **kwargs)

    def find_spec(name, *args, **kwargs):
        if name == "gsplat":
            spec_calls.append(name)
            return SimpleNamespace(submodule_search_locations=[str(directory)])
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(render.importlib.util, "find_spec", find_spec)
    monkeypatch.delitem(sys.modules, "gsplat.csrc", raising=False)
    return attempted_imports, spec_calls


@pytest.mark.parametrize("binaries", [(), ("csrc.so", "csrc_duplicate.so")])
def test_missing_or_ambiguous_binary_never_imports_gsplat_initializer(tmp_path, monkeypatch, binaries):
    for name in binaries:
        (tmp_path / name).touch()
    attempted_imports, _ = _installed_package_without_initializer(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="source/JIT builds are disabled"):
        render.require_prebuilt_renderer()
    assert attempted_imports == []
    assert "gsplat.csrc" not in sys.modules


def test_abi_error_raises_before_package_initializer_or_jit(tmp_path, monkeypatch):
    (tmp_path / "csrc.so").touch()
    attempted_imports, _ = _installed_package_without_initializer(monkeypatch, tmp_path)

    def incompatible_binary(spec):
        raise ImportError("synthetic missing ABI symbol")

    monkeypatch.setattr(render.importlib.util, "module_from_spec", incompatible_binary)
    with pytest.raises(RuntimeError, match="source/JIT builds are disabled") as error:
        render.require_prebuilt_renderer()
    assert isinstance(error.value.__cause__, ImportError)
    assert "synthetic missing ABI symbol" in str(error.value)
    assert attempted_imports == []
    assert "gsplat.csrc" not in sys.modules


def test_successful_prebuilt_validation_is_cached(tmp_path, monkeypatch):
    binary = tmp_path / "csrc.so"
    binary.touch()
    attempted_imports, spec_calls = _installed_package_without_initializer(monkeypatch, tmp_path)
    # This is a guard-only unit test, not evidence that a native CUDA binary works.
    compiled = SimpleNamespace(__file__=str(binary))
    monkeypatch.setitem(sys.modules, "gsplat.csrc", compiled)
    assert render.require_prebuilt_renderer() == str(binary)
    assert render.require_prebuilt_renderer() == str(binary)
    assert spec_calls == ["gsplat"]
    assert attempted_imports == []
    assert render.require_prebuilt_renderer.cache_info().hits == 1
