"""Isolation proof boundary checks; native acceptance still requires CUDA and EGL renders."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from s4d.gpu_guard import APPROVED_HOST_GPUS, HOST_NAME

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("isolation_checks", REPO / "scripts/check_gpu_isolation.py")
isolation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isolation)


def test_docker_pid_matching_requires_explicit_host_namespace():
    if Path("/.dockerenv").exists() or HOST_NAME == "remote":
        with pytest.raises(RuntimeError, match="--pid=host"):
            isolation.pid_namespace_evidence(False)
    else:
        assert isolation.pid_namespace_evidence(False)["nspid"]


@pytest.mark.parametrize("value", ["0", None])
def test_bad_gpu_environment_is_rejected_before_native_readiness(value):
    result = isolation.check_rejected(value)
    assert result["rejected"], result
    expected = "unset or empty" if value is None else "not an allowed GPU UUID"
    assert expected in result["child_log"]


def test_other_host_uuid_is_rejected_before_native_readiness():
    other = "remote" if HOST_NAME == "local" else "local"
    uuid = APPROVED_HOST_GPUS[other][0]
    result = isolation.check_rejected(uuid)
    assert result["rejected"], result
    assert "not an allowed GPU UUID" in result["child_log"]
