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


def validate_adamw_state(saved, optimizer, step):
    """Validate every saved group/parameter against the actual optimizer before PyTorch can load it."""
    import torch

    groups, moments = saved.get("param_groups", []), saved.get("state", {})
    if not groups or len(groups) != len(optimizer.param_groups):
        raise ValueError("resume optimizer parameter groups are missing or inconsistent")
    ids = [pid for group in groups for pid in group.get("params", [])]
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(moments):
        raise ValueError("resume optimizer requires complete parameter-state coverage")
    for saved_group, live_group in zip(groups, optimizer.param_groups):
        if len(saved_group["params"]) != len(live_group["params"]):
            raise ValueError("resume optimizer parameter groups are missing or inconsistent")
        for pid, param in zip(saved_group["params"], live_group["params"]):
            state = moments[pid]
            counter = state.get("step")
            if torch.is_tensor(counter):
                counter = counter.item() if counter.numel() == 1 else None
            if counter != step:
                raise ValueError("resume optimizer counters are missing or inconsistent")
            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if not torch.is_tensor(value) or value.shape != param.shape:
                    raise ValueError("resume optimizer moment shapes are missing or inconsistent")


def validate_resume_state(state, cfg, *, optimizer, lower, upper, exact_step=None):
    """Reject a partial/reset optimizer or a continuation made with a different schedule."""
    step = state.get("step")
    if type(step) is not int or not lower <= step <= upper or (exact_step is not None and step != exact_step):
        raise ValueError("resume checkpoint step is outside this job")
    if science_config(state.get("config", {})) != science_config(cfg):
        raise ValueError("resume checkpoint science differs from baseline")
    if any(k not in state for k in ("optimizer", "scheduler", "rng")):
        raise ValueError("resume requires optimizer, scheduler and RNG state")
    validate_adamw_state(state["optimizer"], optimizer, step)
    if state["scheduler"].get("last_epoch") != step:
        raise ValueError("resume scheduler counter is missing or inconsistent")
    original = cfg["train"]["original_steps"]
    old = state["config"]["train"]
    if step == original:
        if old["steps"] != original or old.get("lr_schedule", "cosine") != "cosine":
            raise ValueError("initial checkpoint must use the original cosine schedule")
    elif any(old.get(k) != cfg["train"].get(k) for k in ("steps", "original_steps", "lr_schedule", "restarts")):
        raise ValueError("resume checkpoint continuation schedule differs")
