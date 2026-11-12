"""Remote MIG regime for the test session (remote MIG migration, 2026-11-11); imported by tests/conftest.py.

The conftest's GPU check uses the real allowlist (in the remote native suite of a MIG release: the container's own MIG
instance), and every module keeps seeing it, except the tests of the PHYSICAL launch-side regime listed below. Written
for the four physical remote GPUs, they run against that regime whether or not configs/hosts/remote-mig.yaml is present
in the checkout under test; the MIG regime itself is tested explicitly, with synthetic allowlists, in
tests/test_remote_mig.py. The swap is per test (monkeypatch, restored afterwards) and maps the k-th authorized instance
to the (k mod 4)-th physical GPU in the test module's own import-time constants and helper defaults; the CUDA/EGL mapping of this process's
own device (REMOTE_MIG_DEVICES) is never touched. Without a MIG allowlist nothing happens.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import MappingProxyType

import pytest

# a module file name (every test in it) or "<file>::<test function>"
PHYSICAL_REGIME_TESTS = {"test_remote_launch.py", "test_remote_jobs.py"}
NAMES = ("APPROVED_HOST_GPUS", "REMOTE_MIG_PARENTS", "REMOTE_DEVICES")


def physical_regime(monkeypatch, test_module) -> None:
    from ops.resources import gpu_guard as guard

    if not getattr(guard, "REMOTE_MIG_ACTIVE", False):
        return
    physical = tuple(guard.REMOTE_PHYSICAL_GPUS)
    mig = tuple(guard.APPROVED_HOST_GPUS["remote"])
    swaps = {id(guard.APPROVED_HOST_GPUS): MappingProxyType({**guard.APPROVED_HOST_GPUS, "remote": physical}),
             id(guard.REMOTE_MIG_PARENTS): (), id(guard.REMOTE_DEVICES): physical}
    modules = {id(m): m for m in list(sys.modules.values()) if m is not None}
    modules.update({id(v): v for v in vars(test_module).values() if isinstance(v, types.ModuleType)})
    for module in modules.values():
        for name in NAMES:
            value = getattr(module, name, None)
            if value is not None and id(value) in swaps:
                monkeypatch.setattr(module, name, swaps[id(value)])
        if getattr(module, "REMOTE_MIG_ACTIVE", None) is True:
            monkeypatch.setattr(module, "REMOTE_MIG_ACTIVE", False)
    from ops.containers import verification
    monkeypatch.setitem(verification.POLICIES, "native_suite", verification.PHYSICAL_NATIVE_POLICY)
    def mapped(value):
        return physical[mig.index(value) % len(physical)] if isinstance(value, str) and value in mig else value

    for name, value in list(vars(test_module).items()):
        if isinstance(value, str) and value in mig:
            monkeypatch.setattr(test_module, name, mapped(value))
        elif isinstance(value, types.FunctionType) and value.__module__ == test_module.__name__:
            # helpers whose default arguments were bound at import time (e.g. ``gpu=REMOTE``)
            if value.__defaults__ and any(isinstance(d, str) and d in mig for d in value.__defaults__):
                monkeypatch.setattr(value, "__defaults__", tuple(mapped(d) for d in value.__defaults__))
            if value.__kwdefaults__ and any(isinstance(d, str) and d in mig for d in value.__kwdefaults__.values()):
                monkeypatch.setattr(value, "__kwdefaults__", {k: mapped(d) for k, d in value.__kwdefaults__.items()})


@pytest.fixture(autouse=True)
def _s4d_remote_regime(request, monkeypatch):
    name = Path(str(request.node.fspath)).name
    if name in PHYSICAL_REGIME_TESTS or f"{name}::{request.node.originalname}" in PHYSICAL_REGIME_TESTS:
        physical_regime(monkeypatch, request.module)
    yield
