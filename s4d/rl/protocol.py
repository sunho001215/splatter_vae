"""Resolution of one RL run's configuration: DrM + shared protocol + encoder + official per-task overrides."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from s4d.config import load_config
from s4d.rl.env import check_pretraining_spacing

REPO = Path(__file__).resolve().parents[2]


def task_settings(task: str) -> dict:
    """{"difficulty", "drm_overrides"} of a Meta-World task (``configs/rl/tasks.yaml``)."""
    return yaml.safe_load((REPO / "configs/rl/tasks.yaml").read_text())[task]


def resolve_config(task: str, encoder: str, seed: int, overrides: list[str]) -> dict:
    cfg = load_config([REPO / "configs/rl/base.yaml", REPO / f"configs/rl/encoders/{encoder}.yaml"], overrides)
    settings = task_settings(task)
    cfg["agent"].update(settings["drm_overrides"])
    cfg.update(task=task, task_difficulty=settings["difficulty"], seed=seed, encoder_name=encoder)
    if cfg["vision"]["encoder_type"] == "splatter4d":
        export = torch.load(cfg["vision"]["export_path"], map_location="cpu", weights_only=True)
        check_pretraining_spacing(
            export["frame_strides"], int(export["encoder_config"]["num_frames"]), int(cfg["env"]["action_repeat"])
        )
    return cfg
