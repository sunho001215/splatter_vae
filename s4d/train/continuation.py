"""Read-only source checkpoint staging and science checks, independent of GPU entry points."""

import hashlib
import os
import shutil
from pathlib import Path


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stage_checkpoint(source, destination, expected_sha):
    source, destination = Path(source).resolve(strict=True), Path(destination)
    if source == destination.resolve() or destination.is_symlink():
        raise ValueError("source checkpoint must remain read-only and separate")
    before = file_sha(source)
    if before != expected_sha:
        raise ValueError("source checkpoint sha256 differs from registration")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        tmp = destination.with_suffix(".copying")
        shutil.copyfile(source, tmp)
        if file_sha(tmp) != before:
            tmp.unlink()
            raise ValueError("checkpoint copy sha256 mismatch")
        os.replace(tmp, destination)
    if file_sha(destination) != before or file_sha(source) != before:
        raise ValueError("checkpoint source/copy sha256 mismatch after staging")
    return destination


def science_config(cfg):
    import copy

    cfg = copy.deepcopy(cfg)
    cfg.pop("run", None)
    for key in ("steps", "stop_step", "original_steps", "lr_schedule", "restarts"):
        cfg["train"].pop(key, None)

    def canonical(node):
        if isinstance(node, dict):
            return {k: canonical(v) for k, v in node.items()}
        if isinstance(node, list):
            return [canonical(v) for v in node]
        if isinstance(node, str) and "runs/realworld-inputs/" in node:
            return "/home/compu/kaist/sunho/realworld/" + node.split("runs/realworld-inputs/", 1)[1]
        return node

    return canonical(cfg)


def validate_resume_state(state, cfg, *, lower, upper, exact_step=None):
    """Reject a partial/reset optimizer or a continuation made with a different schedule."""
    step = state.get("step")
    if type(step) is not int or not lower <= step <= upper or (exact_step is not None and step != exact_step):
        raise ValueError("resume checkpoint step is outside this job")
    if science_config(state.get("config", {})) != science_config(cfg):
        raise ValueError("resume checkpoint science differs from baseline")
    if any(k not in state for k in ("optimizer", "scheduler", "rng")):
        raise ValueError("resume requires optimizer, scheduler and RNG state")
    moments = state["optimizer"].get("state", {})
    if (
        not moments
        or any(int(s.get("step", -1)) != step or "exp_avg" not in s or "exp_avg_sq" not in s for s in moments.values())
        or state["scheduler"].get("last_epoch") != step
    ):
        raise ValueError("resume optimizer moments/counters or scheduler are missing or inconsistent")
    original = cfg["train"]["original_steps"]
    old = state["config"]["train"]
    if step == original:
        if old["steps"] != original or old.get("lr_schedule", "cosine") != "cosine":
            raise ValueError("initial checkpoint must use the original cosine schedule")
    elif any(old.get(k) != cfg["train"].get(k) for k in ("steps", "original_steps", "lr_schedule", "restarts")):
        raise ValueError("resume checkpoint continuation schedule differs")
