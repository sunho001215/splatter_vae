"""Shared entry-point preamble: repo on sys.path, GPU guard before any CUDA/MuJoCo import."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for variable, suffix in (
    ("TORCH_HOME", "torch"),
    ("TORCH_EXTENSIONS_DIR", "extensions"),
    ("XDG_CACHE_HOME", "xdg"),
    ("TMPDIR", "tmp"),
    ("WANDB_CACHE_DIR", "wandb-cache"),
    ("WANDB_DATA_DIR", "wandb-data"),
):
    cache = REPO / ".cache" / suffix
    cache.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(cache)

from s4d.gpu_guard import enforce_allowed_gpus, enforce_mujoco_egl_device  # noqa: E402


def guard_gpus() -> list[dict]:
    """Validate CUDA_VISIBLE_DEVICES and torch's visible devices; returns the resolved mapping."""
    return enforce_allowed_gpus()


def guard_mujoco() -> int:
    """Import mujoco_mig_setup (selects the EGL device) and verify it. Call before importing mujoco/metaworld."""
    from s4d.gpu_guard import GPUIsolationError, visible_device_uuids

    if len(visible_device_uuids()) != 1:
        raise GPUIsolationError("MuJoCo rendering requires exactly one allowed GPU UUID.")
    import mujoco_mig_setup  # noqa: F401, PLC0415

    return enforce_mujoco_egl_device()


def source_fingerprints() -> dict[str, str]:
    """Fingerprint exactly the code, scripts, tests and configs exercised by the suite."""
    return {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("s4d", "scripts", "tests", "configs")
        for path in sorted((REPO / folder).rglob("*"))
        if path.is_file() and path.suffix in (".py", ".yaml", ".sh")
    }


def require_passed_tests() -> None:
    """Reject long work without a complete, passing suite for these exact sources."""
    path = REPO / "docs/tests.json"
    if not path.is_file():
        raise RuntimeError("all tests must pass before long work; run scripts/run_tests.py")
    evidence = json.loads(path.read_text())
    counts = evidence.get("counts", {})
    if (
        not evidence.get("full_suite_passed")
        or evidence.get("exit_code") != 0
        or any(counts.get(kind, 1) for kind in ("failed", "error", "skipped"))
        or counts.get("passed", 0) <= 0
    ):
        raise RuntimeError("all tests must pass before long work; native errors are not skipped")
    if evidence.get("source_sha256") != source_fingerprints():
        raise RuntimeError("test evidence is stale for these sources; rerun scripts/run_tests.py")
