"""Resolution of one RL run's configuration: reference protocol + encoder + per-task exploration schedule."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from s4d.config import load_config
from s4d.rl.env import check_pretraining_spacing

REPO = Path(__file__).resolve().parents[2]


def task_exploration(task: str) -> tuple[str, str]:
    """(MWM difficulty category, stddev schedule in agent steps) of a Meta-World task."""
    table = yaml.safe_load((REPO / "configs/rl/tasks.yaml").read_text())
    difficulty = table["tasks"][task]
    return difficulty, table["schedules"][difficulty]


def resolve_config(task: str, encoder: str, seed: int, overrides: list[str]) -> dict:
    cfg = load_config([REPO / "configs/rl/base.yaml", REPO / f"configs/rl/encoders/{encoder}.yaml"], overrides)
    difficulty, stddev_schedule = task_exploration(task)
    cfg["agent"]["stddev_schedule"] = stddev_schedule
    cfg.update(task=task, task_difficulty=difficulty, seed=seed, encoder_name=encoder)
    if cfg["vision"]["encoder_type"] == "splatter4d":
        export = torch.load(cfg["vision"]["export_path"], map_location="cpu", weights_only=True)
        check_pretraining_spacing(
            export["frame_strides"], int(export["encoder_config"]["num_frames"]), int(cfg["env"]["action_repeat"])
        )
    return cfg
