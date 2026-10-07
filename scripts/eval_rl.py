"""Evaluate every policy snapshot of one RL run with the full protocol (companion job of ``train_rl.py``).

Processes ``<run>/snapshots/step_*.pt`` in order, appending one record per snapshot to ``<run>/eval.jsonl``.
Exits when the training run has written ``final.json`` and every snapshot is evaluated, or when the
scheduler marks the training job failed. Launch with exactly one allowed GPU UUID.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus, guard_mujoco  # noqa: E402

GPU_MAPPING = guard_gpus()
EGL_DEVICE = guard_mujoco()

import torch  # noqa: E402
import yaml  # noqa: E402
from jobs import read_registry  # noqa: E402  (scripts/ is on sys.path)

from s4d.diag.wandb_log import init_wandb  # noqa: E402
from s4d.rl.agent import DrMAgent  # noqa: E402
from s4d.rl.env import env_kwargs  # noqa: E402
from s4d.rl.evaluate import evaluation_suite  # noqa: E402
from s4d.rl.vecenv import EnvPool  # noqa: E402

META_WORLD_ACTION_DIM = 4  # xyz end-effector delta + gripper


def evaluated_steps(path: Path) -> set[int]:
    return {json.loads(line)["step"] for line in path.read_text().splitlines()} if path.is_file() else set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--train-job", required=True, help="scheduler id of the training job")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--envs-per-worker", type=int, default=10)
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if REPO.resolve() not in run_dir.parents:
        raise ValueError("RL run directories must be inside the repository")
    while not (run_dir / "config.yaml").is_file():
        time.sleep(args.poll_seconds)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    device = torch.device("cuda")
    pool = EnvPool(cfg["task"], args.workers, args.envs_per_worker, env_kwargs(cfg))
    agent = DrMAgent(cfg, META_WORLD_ACTION_DIM, len(cfg["env"]["proprio_indices"]), device)
    agent.train(False)
    out = run_dir / "eval.jsonl"
    wandb_run = init_wandb(
        {**cfg, "eval_gpus": GPU_MAPPING},
        f"{run_dir.name}-eval",
        cfg["wandb"]["project"],
        bool(cfg["wandb"]["enabled"]),
        run_dir,
    )
    every = int(cfg["eval"]["every_steps"])
    try:
        while True:
            # Read the completion marker first: every snapshot is written before final.json.
            trained = (run_dir / "final.json").is_file()
            failed = read_registry().get(args.train_job, {}).get("status") == "failed"
            done_steps = evaluated_steps(out)
            pending = sorted(
                (p for p in (run_dir / "snapshots").glob("step_*.pt") if int(p.stem[5:]) not in done_steps),
                key=lambda p: int(p.stem[5:]),
            )
            if not pending and (trained or failed):
                break
            for path in pending:
                snapshot = torch.load(path, map_location=device, weights_only=True)
                agent.encoder.load_state_dict(snapshot["policy"]["encoder"])
                agent.actor.load_state_dict(snapshot["policy"]["actor"])
                step = int(snapshot["step"])
                t0 = time.time()
                metrics = evaluation_suite(pool, agent, step, step // every, int(cfg["seed"]), cfg["eval"])
                record = {"step": step, **metrics, "eval_seconds": time.time() - t0, "time": time.time()}
                with open(out, "a") as f:
                    f.write(json.dumps(record, allow_nan=False) + "\n")
                if wandb_run is not None:
                    wandb_run.log({f"eval/{k}": v for k, v in record.items() if k != "time"}, step=step)
            if not pending:
                time.sleep(args.poll_seconds)
    finally:
        pool.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
