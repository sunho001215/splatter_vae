"""Operational code lives in ops/; the old s4d/ and scripts/ module paths stay importable as the same module objects."""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALIASES = {
    "s4d.remote_access": "ops.remote.access",
    "s4d.remote_artifact_check": "ops.remote.artifact_check",
    "s4d.remote_jobs": "ops.jobs.remote_backend",
    "s4d.lifecycle_cleanup": "ops.jobs.lifecycle",
    "s4d.replay_cleanup_host": "ops.jobs.replay_host",
    "s4d.run_identity": "ops.jobs.run_identity",
    "s4d.gpu_guard": "ops.resources.gpu_guard",
    "s4d.cross_host_equivalence": "ops.resources.host_equivalence",
    "s4d.remote_launch": "ops.containers.launch",
    "s4d.remote_verification": "ops.containers.verification",
    "scripts.jobs": "ops.jobs.scheduler",
    "scripts._nvidia_query": "ops.resources.nvidia_query",
}
# The only research-side imports of ops (lazy, inside functions): output-path authorization and the W&B host tag.
S4D_IMPORTS_OF_OPS = {
    ("s4d/data/__init__.py", "ops.resources.gpu_guard"),
    ("s4d/diag/wandb_log.py", "ops.resources.gpu_guard"),
}


@pytest.mark.parametrize("old,new", sorted(ALIASES.items()))
def test_old_module_path_is_the_same_module_object(old, new):
    module = importlib.import_module(new)
    assert importlib.import_module(old) is module and sys.modules[old] is module
    parent, leaf = old.rsplit(".", 1)
    assert getattr(importlib.import_module(parent), leaf) is module
    shim = REPO / (old.replace(".", "/") + ".py")
    assert len(shim.read_text().splitlines()) < 25  # an alias only: the implementation is not duplicated


def test_moved_modules_keep_their_repository_root():
    from ops.jobs import scheduler
    from ops.remote.access import RemoteClient
    from ops.resources import gpu_guard

    assert scheduler.REPO == gpu_guard.REPO == REPO
    assert scheduler.QUEUE == REPO / "experiments/queue.yaml" and scheduler.RUNS == REPO / "runs"
    assert RemoteClient.__init__.__code__.co_filename == str(REPO / "ops/remote/access.py")


def test_research_package_imports_ops_only_at_documented_points():
    found = set()
    for path in sorted((REPO / "s4d").rglob("*.py")):
        relative = str(path.relative_to(REPO))
        if relative.replace("/", ".")[:-3] in ALIASES:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            found.update((relative, name) for name in names if name == "ops" or name.startswith("ops."))
    assert found == S4D_IMPORTS_OF_OPS


@pytest.mark.parametrize("script", ["scripts/jobs.py", "s4d/remote_access.py", "ops/remote/access.py"])
def test_command_line_entry_points_run_isolated(script):
    result = subprocess.run([sys.executable, "-I", str(REPO / script), "--help"], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0 and result.stdout.startswith("usage: " + Path(script).name)
