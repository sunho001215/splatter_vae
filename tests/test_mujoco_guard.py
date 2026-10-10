"""Synthetic rejection paths only; native EGL acceptance still requires actual renders."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from s4d.gpu_guard import ALLOWED_GPU_UUIDS, GPUIsolationError

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def helper():
    path = REPO / "mujoco_mig_setup.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    assert isinstance(tree.body[-1], ast.Expr)
    assert isinstance(tree.body[-1].value, ast.Call)
    assert tree.body[-1].value.func.id == "setup"
    tree.body.pop()
    module = ModuleType("egl_rejection_boundaries")
    module.__file__ = str(path)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("value", [None, "", "0", "MIG-forbidden", ",".join(ALLOWED_GPU_UUIDS[:2])])
def test_invalid_uuid_import_refuses_before_mujoco_or_native_libraries(value):
    code = f"""
import builtins, sys
sys.path.insert(0, {str(REPO)!r})
real_import = builtins.__import__
def block_native(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'gsplat', 'mujoco', 'metaworld'):
        raise AssertionError('native import before UUID rejection: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = block_native
try:
    import mujoco_mig_setup
except RuntimeError:
    assert not any(name in sys.modules for name in ('torch', 'gsplat', 'mujoco', 'metaworld'))
else:
    raise AssertionError('invalid CUDA environment accepted')
"""
    env = {**os.environ, "MUJOCO_EGL_DEVICE_ID": "0"}
    if value is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = value
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code], env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_proved_mapping_refuses_before_mujoco_import(helper, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ALLOWED_GPU_UUIDS[0])
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    monkeypatch.setattr(helper, "_find_egl_device_index_for_visible_cuda", lambda **kwargs: None)
    monkeypatch.setattr(helper, "_patch_mujoco_egl", lambda: pytest.fail("missing mapping must not import MuJoCo"))
    with pytest.raises(GPUIsolationError, match="No EGL device"):
        helper.setup()
    assert "MUJOCO_EGL_DEVICE_ID" not in os.environ
    assert helper._SELECTED_EGL_DEVICE_ID is None
    assert helper._SELECTED_GPU_UUID is None


def test_mapping_query_failure_never_falls_back(helper, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ALLOWED_GPU_UUIDS[0])
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")

    def failed_query(**kwargs):
        raise RuntimeError("synthetic EGL query failure")

    monkeypatch.setattr(helper, "_find_egl_device_index_for_visible_cuda", failed_query)
    monkeypatch.setattr(helper, "_patch_mujoco_egl", lambda: pytest.fail("failed query must not import MuJoCo"))
    with pytest.raises(RuntimeError, match="synthetic EGL query failure"):
        helper.setup()
    assert "MUJOCO_EGL_DEVICE_ID" not in os.environ


def test_ambiguous_cuda_mapping_refuses_egl_order(helper, monkeypatch):
    monkeypatch.setattr(helper, "_load_egl", lambda: object())
    monkeypatch.setattr(helper, "_query_egl_devices_ctypes", lambda: [object(), object()])

    def synthetic_query(device, attribute, result):
        result._obj.value = 0
        return 1

    monkeypatch.setattr(helper, "_get_egl_ext_function", lambda *args: synthetic_query)
    with pytest.raises(GPUIsolationError, match="Multiple EGL devices"):
        helper._find_egl_device_index_for_visible_cuda()


@pytest.mark.parametrize("selected", [None, "", "-1", "default", "0.0", "²"])
def test_display_requires_numeric_proved_selection_before_loading_egl(helper, monkeypatch, selected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ALLOWED_GPU_UUIDS[0])
    if selected is None:
        monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    else:
        monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", selected)
    monkeypatch.setattr(helper, "_load_egl", lambda: pytest.fail("invalid selection must not initialize EGL"))
    with pytest.raises(GPUIsolationError, match="proved numeric"):
        helper.create_initialized_egl_device_display_full()


def test_display_rejects_unproved_or_changed_selection_before_egl(helper, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ALLOWED_GPU_UUIDS[0])
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    monkeypatch.setattr(helper, "_load_egl", lambda: pytest.fail("unproved selection must not initialize EGL"))
    with pytest.raises(GPUIsolationError, match="changed after proved"):
        helper.create_initialized_egl_device_display_full()
    helper._SELECTED_EGL_DEVICE_ID = 1
    helper._SELECTED_GPU_UUID = ALLOWED_GPU_UUIDS[0]
    with pytest.raises(GPUIsolationError, match="changed after proved"):
        helper.create_initialized_egl_device_display_full()
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ALLOWED_GPU_UUIDS[1])
    with pytest.raises(GPUIsolationError, match="changed after proved"):
        helper.create_initialized_egl_device_display_full()


def test_display_rejects_out_of_range_selection_without_other_devices(helper, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ALLOWED_GPU_UUIDS[0])
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "1")
    helper._SELECTED_EGL_DEVICE_ID = 1
    helper._SELECTED_GPU_UUID = ALLOWED_GPU_UUIDS[0]
    monkeypatch.setattr(helper, "_load_egl", lambda: object())
    monkeypatch.setattr(helper, "_query_egl_devices_ctypes", lambda: [object()])
    monkeypatch.setattr(helper, "_get_egl_ext_function", lambda *args: pytest.fail("no other device may be tried"))
    with pytest.raises(GPUIsolationError, match="outside the 1 EGL devices"):
        helper.create_initialized_egl_device_display_full()
