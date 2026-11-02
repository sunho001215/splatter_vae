"""Run-directory helpers shared by the RL trainers (scripts/train_rl.py, train_sac.py, train_ppo.py, train_dreamer.py,
train_flashsac.py): atomic checkpoint writes, the append-only metrics log and the process's resident memory."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def rss_gb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def atomic_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")
