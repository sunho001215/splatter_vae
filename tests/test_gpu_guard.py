"""Host-config and environment boundary checks, not native GPU acceptance evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from s4d import gpu_guard as guard

REPO = Path(__file__).resolve().parents[1]


def write_config(tmp_path, config):
    path = tmp_path / "host.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def config_for(name):
    return {
        "name": name,
        "gpu_uuids": list(guard.APPROVED_HOST_GPUS[name]),
        "runtime_root": str(guard.APPROVED_RUNTIME_ROOTS[name]) if name == "remote" else None,
    }


@pytest.mark.parametrize("name", ["local", "remote"])
def test_exact_host_config_selects_only_approved_uuids(tmp_path, name):
    config = config_for(name)
    config["gpu_uuids"].reverse()
    selected = guard.load_host_config(write_config(tmp_path, config))
    assert selected["name"] == name
    assert selected["gpu_uuids"] == guard.APPROVED_HOST_GPUS[name]
    assert selected["runtime_root"] == guard.APPROVED_RUNTIME_ROOTS[name]


def test_default_host_config_is_explicit_local_not_machine_detection():
    env = dict(os.environ)
    env.pop("S4D_HOST_CONFIG", None)
    code = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
from s4d import gpu_guard as guard
assert guard.HOST_NAME == 'local'
assert guard.HOST_CONFIG_PATH == guard.REPO / 'configs/hosts/local.yaml'
assert guard.ALLOWED_GPU_UUIDS == guard.APPROVED_HOST_GPUS['local']
assert not any(name in sys.modules for name in ('torch', 'gsplat', 'mujoco', 'metaworld'))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code], env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_host_allowlist_mapping_is_immutable():
    with pytest.raises(TypeError):
        guard.APPROVED_HOST_GPUS["local"] = guard.APPROVED_HOST_GPUS["remote"]
    with pytest.raises(TypeError):
        guard.HOST_CONFIG["name"] = "other"


@pytest.mark.parametrize(
    "change",
    [
        {"name": "autodetect"}, {"name": []}, {"gpu_uuids": "GPU-0"},
        {"gpu_uuids": ["GPU-00000000-0000-0000-0000-000000000000"]},
        {"gpu_uuids": list(guard.APPROVED_HOST_GPUS["remote"])},
        {"gpu_uuids": [guard.APPROVED_HOST_GPUS["local"][0]] * 2},
        {"gpu_uuids": [guard.APPROVED_HOST_GPUS["local"][0], []]},
        {"runtime_root": "/tmp"},
    ],
)
def test_invalid_host_config_fails_closed(tmp_path, change):
    config = {**config_for("local"), **change}
    with pytest.raises(guard.GPUIsolationError):
        guard.load_host_config(write_config(tmp_path, config))


@pytest.mark.parametrize("contents", ["", "[]", "name: [", "name: remote\ngpu_uuids: []\n"])
def test_missing_or_malformed_host_config_is_rejected(tmp_path, contents):
    path = tmp_path / "host.yaml"
    with pytest.raises(guard.GPUIsolationError):
        guard.load_host_config(path)
    path.write_text(contents)
    with pytest.raises(guard.GPUIsolationError):
        guard.load_host_config(path)
    with pytest.raises(guard.GPUIsolationError, match="absolute"):
        guard.load_host_config(Path("configs/hosts/local.yaml"))


@pytest.mark.parametrize("name", ["local", "remote"])
def test_import_selects_explicit_host_and_rejects_cross_host_before_torch(tmp_path, name):
    path = write_config(tmp_path, config_for(name))
    forbidden = guard.APPROVED_HOST_GPUS["remote" if name == "local" else "local"][0]
    code = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
from s4d import gpu_guard as guard
assert guard.HOST_NAME == {name!r}
assert guard.ALLOWED_GPU_UUIDS == guard.APPROVED_HOST_GPUS[{name!r}]
try:
    guard.enforce_allowed_gpus()
except guard.GPUIsolationError:
    assert not any(name in sys.modules for name in ('torch', 'gsplat', 'mujoco', 'metaworld'))
else:
    raise AssertionError('cross-host UUID accepted')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        env={**os.environ, "S4D_HOST_CONFIG": str(path), "CUDA_VISIBLE_DEVICES": forbidden},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[gpu_guard] FATAL:" in result.stderr


def test_invalid_selected_config_import_fails_before_gpu_aware_packages(tmp_path):
    path = write_config(tmp_path, {"name": "remote", "gpu_uuids": list(guard.APPROVED_HOST_GPUS["local"])})
    code = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
try:
    import s4d.gpu_guard
except RuntimeError:
    assert not any(name in sys.modules for name in ('torch', 'gsplat', 'mujoco', 'metaworld'))
else:
    raise AssertionError('invalid host config accepted')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code], env={**os.environ, "S4D_HOST_CONFIG": str(path)},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("raw", [None, "", " ", "0", "GPU-0", "MIG-0", ",", "GPU-0,"])
def test_invalid_cuda_environment_is_rejected(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", raw)
    with pytest.raises(guard.GPUIsolationError):
        guard.visible_device_uuids()


def test_cuda_environment_rejects_duplicates_and_accepts_selected_set(monkeypatch):
    uuid = guard.ALLOWED_GPU_UUIDS[0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{uuid},{uuid}")
    with pytest.raises(guard.GPUIsolationError, match="twice"):
        guard.visible_device_uuids()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(guard.ALLOWED_GPU_UUIDS))
    assert guard.visible_device_uuids() == list(guard.ALLOWED_GPU_UUIDS)


def test_repository_cache_and_evidence_paths_are_constrained(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "RUNTIME_ROOT", None)
    cache = tmp_path / ".cache/tmp"
    assert guard.validate_runtime_path(cache, repository=tmp_path, cache=True) == cache
    assert guard.validate_runtime_path(tmp_path / "docs", repository=tmp_path) == tmp_path / "docs"
    with pytest.raises(guard.GPUIsolationError, match=".cache"):
        guard.validate_runtime_path(tmp_path / "data", repository=tmp_path, cache=True)
    with pytest.raises(guard.GPUIsolationError, match="Runtime output"):
        guard.validate_runtime_path(tmp_path.parent, repository=tmp_path)
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(guard.GPUIsolationError, match="Runtime output"):
        guard.validate_runtime_path(tmp_path / "escape" / "other", repository=tmp_path)


def test_remote_runtime_root_allows_descendants_not_arbitrary_paths(tmp_path, monkeypatch):
    root = guard.APPROVED_RUNTIME_ROOTS["remote"]
    monkeypatch.setattr(guard, "RUNTIME_ROOT", root)
    assert guard.validate_runtime_path(root / "runtime/cache", repository=tmp_path) == root / "runtime/cache"
    for path in (root, root.parent / "outside", Path("/tmp")):
        with pytest.raises(guard.GPUIsolationError):
            guard.validate_runtime_path(path, repository=tmp_path)


def test_external_evidence_gate_remains_strict(tmp_path, monkeypatch):
    import _bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "REPO", tmp_path)
    monkeypatch.setattr(bootstrap, "source_fingerprints", lambda: {"helper.py": "sha256"})
    directory = tmp_path / "runtime/evidence"
    directory.mkdir(parents=True)
    monkeypatch.setenv("S4D_TEST_EVIDENCE_DIR", str(directory))
    report = {
        "full_suite_passed": True, "exit_code": 0,
        "counts": {"passed": 1, "error": 0, "failed": 0, "skipped": 0},
        "source_sha256": {"helper.py": "sha256"},
    }
    path = directory / "tests.json"
    path.write_text(json.dumps(report))
    bootstrap.require_passed_tests()
    for key in ("failed", "error", "skipped"):
        report["counts"][key] = 1
        path.write_text(json.dumps(report))
        with pytest.raises(RuntimeError, match="all tests"):
            bootstrap.require_passed_tests()
        report["counts"][key] = 0
    report["source_sha256"] = {"helper.py": "stale"}
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="stale"):
        bootstrap.require_passed_tests()


def test_root_egl_helper_is_fingerprinted():
    import _bootstrap as bootstrap

    assert "mujoco_mig_setup.py" in bootstrap.source_fingerprints()
