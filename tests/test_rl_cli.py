"""End-to-end RL entry points in a one-UUID subprocess: DrM training, snapshot evaluation and crash resume."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TINY = [
    "train.num_seed_steps=20",
    "agent.num_expl_steps=20",
    "train.batch_size=16",
    "train.replay_size=500",
    "train.log_every_steps=10",
    "eval.every_steps=30",
    "train.checkpoint_every_steps=30",
    "eval.episodes_per_train_camera=1",
    "eval.episodes_per_heldout_camera=1",
    "eval.episodes_per_trajectory=1",
    "env.image_size=64",
    "wandb.enabled=false",
]


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0], "MUJOCO_GL": "egl"}
    result = subprocess.run(
        [sys.executable, "-I", str(REPO / script), *args], env=env, capture_output=True, text=True, timeout=900
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    return result


def test_drm_cnn_training_evaluation_and_resume(tmp_path):
    run_dir = tmp_path / "rl_cli"
    common = ["--task", "hammer", "--encoder", "cnn", "--seed", "3", "--run-dir", str(run_dir)]
    run("scripts/train_rl.py", *common, "--set", "train.num_train_steps=60", *TINY)
    assert json.loads((run_dir / "final.json").read_text())["steps"] == 60
    assert sorted(p.name for p in (run_dir / "snapshots").iterdir()) == ["step_0000030.pt", "step_0000060.pt"]
    records = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines()]
    assert records[-1]["step"] == 60 and "train/actor_dormant_ratio" in records[-1] and "train/stddev" in records[-1]
    assert (run_dir / "replay" / "atom.memmap").is_file(), "CNN replay is a disk memmap"

    run(
        "scripts/eval_rl.py",
        "--run-dir",
        str(run_dir),
        "--train-job",
        "none",
        "--workers",
        "2",
        "--envs-per-worker",
        "3",
        "--poll-seconds",
        "1",
    )
    evals = [json.loads(line) for line in (run_dir / "eval.jsonl").read_text().splitlines()]
    assert [e["step"] for e in evals] == [30, 60] and all(e["episodes"] == 12 for e in evals)
    assert all(0 <= e["heldout_cameras_success"] <= 1 and "traj_circular_success" in e for e in evals)

    (run_dir / "final.json").unlink()
    run("scripts/train_rl.py", *common, "--set", "train.num_train_steps=90", *TINY)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert {"event": "resumed", "step": 60} == {k: events[0][k] for k in ("event", "step")}
    assert json.loads((run_dir / "final.json").read_text())["steps"] == 90
