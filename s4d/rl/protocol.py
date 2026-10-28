"""Resolution of one RL run's configuration: algorithm (DrM by default) + shared protocol + encoder + official per-task
overrides."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from s4d.config import apply_overrides, deep_update, load_config
from s4d.rl.env import check_pretraining_spacing

REPO = Path(__file__).resolve().parents[2]


def task_settings(task: str) -> dict:
    """{"difficulty", "drm_overrides"} of a Meta-World task (``configs/rl/tasks.yaml``)."""
    return yaml.safe_load((REPO / "configs/rl/tasks.yaml").read_text())[task]


def resolve_config(task: str, encoder: str, seed: int, overrides: list[str], algorithm: str = "drm") -> dict:
    """DrM: ``configs/rl/base.yaml`` + encoder file + overrides + the official DrM per-task overrides. Another
    downstream algorithm (user directive 2026-10-28; ``configs/rl/algorithms/<algorithm>.yaml``) keeps the shared
    protocol sections of base.yaml (env, eval, wandb) and the encoder file's ``vision``/``env`` entries, and REPLACES
    the ``agent`` and ``train`` sections (DrM values and DrM-specific encoder ``agent`` keys never leak into it); the
    DrM per-task overrides do not apply to it. Command-line overrides are applied last in both cases."""
    files = [REPO / "configs/rl/base.yaml", REPO / f"configs/rl/encoders/{encoder}.yaml"]
    settings = task_settings(task)
    if algorithm == "drm":
        cfg = load_config(files, overrides)
        cfg["agent"].update(settings["drm_overrides"])
    else:
        algo = load_config([REPO / f"configs/rl/algorithms/{algorithm}.yaml"])
        if algo.get("algorithm") != algorithm:
            raise ValueError(f"configs/rl/algorithms/{algorithm}.yaml declares algorithm {algo.get('algorithm')!r}")
        cfg = load_config(files)
        for section in ("agent", "train"):
            cfg[section] = algo.pop(section)
        cfg = apply_overrides(deep_update(cfg, algo), overrides)
    cfg.update(task=task, task_difficulty=settings["difficulty"], seed=seed, encoder_name=encoder)
    if cfg["vision"]["encoder_type"] == "splatter4d":
        export = torch.load(cfg["vision"]["export_path"], map_location="cpu", weights_only=True)
        check_pretraining_spacing(
            export["frame_strides"], int(export["encoder_config"]["num_frames"]), int(cfg["env"]["action_repeat"])
        )
        history = export.get("history_stride")  # future-objective exports: the encoder's one fixed history stride
        if history is not None and int(history) != int(cfg["env"]["action_repeat"]):
            raise ValueError(f"encoder history stride {history} differs from the RL frame spacing {cfg['env']['action_repeat']}")
    elif cfg["vision"]["encoder_type"] == "sincro":  # SinCro was pretrained on one fixed frame spacing
        export = torch.load(cfg["vision"]["export_path"], map_location="cpu", weights_only=True)
        check_pretraining_spacing(
            [export["frame_spacing"]], int(export["model_cfg"]["time_interval"]), int(cfg["env"]["action_repeat"])
        )
    return cfg
