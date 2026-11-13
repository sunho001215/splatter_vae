"""Read-only source checkpoint staging and science checks, independent of GPU entry points."""

import hashlib
import math
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
    rngs = state.get("rng_by_rank")
    rngs = [state["rng"]] if rngs is None else rngs
    if not isinstance(rngs, list) or not rngs:
        raise ValueError("resume requires complete RNG state")
    import random

    import numpy as np
    import torch
    for rng in [state["rng"], *rngs]:
        try:
            torch.Generator().set_state(rng["torch"])
            random.Random().setstate(rng["python"])
            np.random.RandomState().set_state(rng["numpy"])
            if (not isinstance(rng["cuda"], list)
                    or any(not torch.is_tensor(v) or v.dtype != torch.uint8 or v.ndim != 1 or not v.numel()
                           for v in rng["cuda"])):
                raise ValueError("invalid CUDA RNG state")
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("resume requires complete RNG state") from exc
    validate_adamw_state(state["optimizer"], optimizer, step)
    schedule = state["scheduler"]
    if (not isinstance(schedule, dict) or type(schedule.get("last_epoch")) is not int
            or schedule["last_epoch"] != step or schedule.get("_step_count") != step + 1
            or any(not isinstance(schedule.get(key), list)
                   or len(schedule[key]) != len(optimizer.param_groups)
                   or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in schedule[key])
                   for key in ("base_lrs", "_last_lr"))):
        raise ValueError("resume scheduler state is missing or inconsistent")
    original = cfg["train"]["original_steps"]
    old = state["config"]["train"]
    if step == original:
        if old["steps"] != original or old.get("lr_schedule", "cosine") != "cosine":
            raise ValueError("initial checkpoint must use the original cosine schedule")
    elif any(old.get(k) != cfg["train"].get(k) for k in ("steps", "original_steps", "lr_schedule", "restarts")):
        raise ValueError("resume checkpoint continuation schedule differs")


def train(cfg, run_dir, train_dataset, ctx, logger, evaluate_fn, resume=None):
    """Continuation-only adapter around the unchanged training loop.

    Install the schedule for this invocation and validate the optimizer that the
    loop actually constructs before its unchanged checkpoint loader can run.
    The adapter is used only by train_ours_continuation.py, never legacy entries.
    """
    import json
    import random

    import numpy as np
    import torch

    from s4d.train import loop
    from s4d.train.schedule import build_optimizer, training_end

    if resume is None:
        raise ValueError("continuation requires a full-state resume checkpoint")
    end = training_end(cfg)
    state = torch.load(resume, map_location="cpu", weights_only=False)
    start = state["step"]
    legacy_build = loop.build_optimizer

    def scheduled_optimizer(model, config):
        nonlocal state
        optimizer, scheduler = build_optimizer(model, config, legacy_build=legacy_build)
        validate_resume_state(state, config, optimizer=optimizer,
                              lower=config["train"]["original_steps"], upper=end)
        state = None  # free the CPU checkpoint before the unchanged loader reads it
        return optimizer, scheduler

    def evaluate(model, step):
        # Probe windows depend on the evaluation step, independent of process restarts.
        for loader in getattr(evaluate_fn, "probe_loaders", {}).values():
            if loader.generator is not None:
                loader.generator.manual_seed(int(cfg.get("eval", {}).get("probe_seed", 0)) + step)
        if evaluate_fn is not None:
            evaluate_fn(model, step)

    # Checkpoints precede evaluation. An endpoint retry must replay its evaluation
    # before publishing success, even though the legacy loop has no remaining updates.
    def endpoint_optimizer(model, config):
        optimizer, scheduler = scheduled_optimizer(model, config)
        if start == end:
            from s4d.train.checkpoint import load_checkpoint

            load_checkpoint(resume, model.encoder, model.decoder, optimizer, scheduler, rank=ctx.rank)
            py_rng, np_rng = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[ctx.local_rank] if ctx.device.type == "cuda" else []):
                    loop.ddp.rank_zero_call(lambda: evaluate(model, end))
            finally:
                random.setstate(py_rng)
                np.random.set_state(np_rng)
        return optimizer, scheduler

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("completion.json", "failure.json", "continuation.json"):
        (run_dir / name).unlink(missing_ok=True)
    loop.build_optimizer = endpoint_optimizer
    try:
        loop.train(cfg, run_dir, train_dataset, ctx, logger, evaluate, resume=resume)
    finally:
        loop.build_optimizer = legacy_build
    if ctx.is_main:
        (run_dir / "continuation.json").write_text(json.dumps({
            "status": "completed", "start_step": start, "stop_step": end,
            "schedule_steps": cfg["train"]["steps"],
        }))
