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
from _bootstrap import REPO, guard_gpus, guard_mujoco, require_passed_tests  # noqa: E402

GPU_MAPPING = guard_gpus()
EGL_DEVICE = guard_mujoco()

import torch  # noqa: E402
import yaml  # noqa: E402
from jobs import read_registry  # noqa: E402  (scripts/ is on sys.path)

from s4d.diag.wandb_log import init_wandb  # noqa: E402
from s4d.gpu_guard import GPUIsolationError, validate_runtime_path  # noqa: E402
from s4d.rl.agent import DrMAgent  # noqa: E402
from s4d.rl.env import env_kwargs  # noqa: E402
from s4d.rl.evaluate import evaluation_suite  # noqa: E402
from s4d.rl.vecenv import EnvPool  # noqa: E402
from s4d.run_identity import record_run_identity  # noqa: E402

META_WORLD_ACTION_DIM = 4  # xyz end-effector delta + gripper


def evaluated_steps(path: Path) -> set[int]:
    return {json.loads(line)["step"] for line in path.read_text().splitlines()} if path.is_file() else set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--train-job", required=True, help="scheduler id of the training job")
    # Each worker process holds one MuJoCo renderer (~0.44 GB of GPU memory) shared by its environments; the pool
    # is opened only while snapshots are pending and closed when idle. 48 environments: 240 episodes in 5 waves.
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--envs-per-worker", type=int, default=8)
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    args = ap.parse_args()
    try:
        run_dir = validate_runtime_path(Path(args.run_dir), repository=REPO)
    except GPUIsolationError as exc:
        raise ValueError("RL run directories must be inside an authorized runtime root") from exc
    while not (run_dir / "config.yaml").is_file():
        time.sleep(args.poll_seconds)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    record_run_identity(
        cfg, REPO, run_dir / "eval_provenance",
        native_diagnostic_steps=int(cfg["train"]["num_train_steps"]),
    )
    if int(cfg["train"]["num_train_steps"]) > 1000:  # same exemption for short diagnostics as train_rl.py
        require_passed_tests()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # as in the official train_mw.py
    torch.set_num_threads(args.cpu_threads)
    pool = None
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
            if pending and pool is None:
                # Reference evaluation env seed: run seed + 1 (its 50 MT1 configurations differ from training's).
                pool = EnvPool(cfg["task"], int(cfg["seed"]) + 1, args.workers, args.envs_per_worker, env_kwargs(cfg))
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
                if pool is not None:  # release the renderers while waiting for the next snapshot
                    pool.close()
                    pool = None
                time.sleep(args.poll_seconds)
    finally:
        if pool is not None:
            pool.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
