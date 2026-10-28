"""Evaluate every policy snapshot of one RL run with the full protocol (companion job of ``train_rl.py`` and
``train_sac.py``; the run's ``config.yaml`` names its algorithm).

Processes ``<run>/snapshots/step_*.pt`` in order, appending one record per snapshot to ``<output>/eval.jsonl``
(``--output-dir``, default the run directory). Exits when the training run has written ``final.json`` and every
snapshot is evaluated, or when the scheduler marks the training job failed. Launch with exactly one allowed GPU UUID.

``--suites v2`` (camera suite v2, evaluation only, 2026-10-29; ``s4d.rl.camera_suites``) evaluates the
visibility-selected held-out cameras vis0..vis3 and the stronger moving paths, with matched references on the same
paired seeds (``--v2-reference``), and appends to a separate ``eval_cameras_v2.jsonl``: the original ``eval.jsonl``
history is never read or written. ``--snapshots final|last5`` re-evaluates the existing snapshots of a finished run
(the final one, or the last five evaluation points, e.g. 960k..1M) and exits; ``--train-job`` is then not needed.
"""

from __future__ import annotations

import argparse
import hashlib
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
from s4d.rl.camera_suites import (  # noqa: E402
    DEFINITION_FILE,
    SUITE_OUTPUTS,
    SUITE_PROVENANCE,
    SUITE_V2,
    SUITE_WANDB_PREFIX,
    SUITE_WANDB_SUFFIX,
    V2_REFERENCES,
    heldout_v2_cameras,
    reevaluation_steps,
)
from s4d.rl.env import env_kwargs  # noqa: E402
from s4d.rl.evaluate import evaluation_suite, evaluation_suite_v2  # noqa: E402
from s4d.rl.vecenv import EnvPool  # noqa: E402
from s4d.run_identity import record_run_identity  # noqa: E402

META_WORLD_ACTION_DIM = 4  # xyz end-effector delta + gripper


def evaluated_steps(path: Path) -> set[int]:
    return {json.loads(line)["step"] for line in path.read_text().splitlines()} if path.is_file() else set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--train-job", help="scheduler id of the training job (required unless --snapshots final|last5)")
    ap.add_argument("--suites", choices=tuple(SUITE_OUTPUTS), default="v1",
                    help="v1: the original protocol -> eval.jsonl; v2: camera suite v2 -> eval_cameras_v2.jsonl")
    ap.add_argument("--snapshots", choices=("all", "final", "last5"), default="all",
                    help="all: companion of a training job; final/last5: re-evaluate a finished run's snapshots")
    ap.add_argument("--v2-reference", choices=V2_REFERENCES, default="train+traj",
                    help="matched references re-run on the v2 seeds (training cameras, and the 1x paths)")
    # Each worker process holds one MuJoCo renderer (~0.44 GB of GPU memory) shared by its environments; the pool
    # is opened only while snapshots are pending and closed when idle. 48 environments: 240 episodes in 5 waves.
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--envs-per-worker", type=int, default=8)
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--output-dir", help="eval.jsonl, provenance and W&B files (a remote companion job's own run dir)")
    args = ap.parse_args()
    if args.snapshots == "all" and args.train_job is None:
        ap.error("--train-job is required unless --snapshots final|last5")
    try:
        run_dir = validate_runtime_path(Path(args.run_dir), repository=REPO)
        output_dir = run_dir if args.output_dir is None else validate_runtime_path(Path(args.output_dir), repository=REPO)
    except GPUIsolationError as exc:
        raise ValueError("RL run directories must be inside an authorized runtime root") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    while not (run_dir / "config.yaml").is_file():
        time.sleep(args.poll_seconds)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if args.snapshots != "all" and not (run_dir / "final.json").is_file():
        raise SystemExit("--snapshots final|last5 re-evaluates a finished run: final.json is missing")
    record_run_identity(
        cfg, REPO, output_dir / SUITE_PROVENANCE[args.suites],
        native_diagnostic_steps=int(cfg["train"]["num_train_steps"]),
    )
    if int(cfg["train"]["num_train_steps"]) > 1000:  # same exemption for short diagnostics as train_rl.py
        require_passed_tests()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # as in the official train_mw.py
    torch.set_num_threads(args.cpu_threads)
    pool = None
    drm = cfg.get("algorithm", "drm") == "drm"
    if drm:
        agent = DrMAgent(cfg, META_WORLD_ACTION_DIM, len(cfg["env"]["proprio_indices"]), device)
    else:  # downstream algorithms (user directive 2026-10-28): same evaluation, each algorithm's deterministic policy
        from s4d.rl.algorithms import evaluation_policy  # noqa: PLC0415

        agent = evaluation_policy(cfg, META_WORLD_ACTION_DIM, len(cfg["env"]["proprio_indices"]), device)
    agent.train(False)
    out = output_dir / SUITE_OUTPUTS[args.suites]
    wandb_run = init_wandb(
        {**cfg, "eval_gpus": GPU_MAPPING, **({"eval_suite": SUITE_V2} if args.suites == "v2" else {})},
        f"{run_dir.name}{SUITE_WANDB_SUFFIX[args.suites]}",
        cfg["wandb"]["project"],
        bool(cfg["wandb"]["enabled"]),
        output_dir,
    )
    every = int(cfg["eval"]["every_steps"])
    selected = None
    if args.snapshots != "all":
        available = [int(p.stem[5:]) for p in (run_dir / "snapshots").glob("step_*.pt")]
        selected = set(reevaluation_steps(available, int(cfg["train"]["num_train_steps"]), every, args.snapshots))
    definition = REPO / DEFINITION_FILE
    try:
        while True:
            # Read the completion marker first: every snapshot is written before final.json.
            trained = (run_dir / "final.json").is_file()
            failed = selected is None and read_registry().get(args.train_job, {}).get("status") == "failed"
            done_steps = evaluated_steps(out)
            pending = sorted(
                (p for p in (run_dir / "snapshots").glob("step_*.pt")
                 if int(p.stem[5:]) not in done_steps and (selected is None or int(p.stem[5:]) in selected)),
                key=lambda p: int(p.stem[5:]),
            )
            if not pending and (trained or failed or selected is not None):
                break
            if pending and pool is None:
                # Reference evaluation env seed: run seed + 1 (its 50 MT1 configurations differ from training's).
                pool = EnvPool(cfg["task"], int(cfg["seed"]) + 1, args.workers, args.envs_per_worker, env_kwargs(cfg))
            for path in pending:
                snapshot = torch.load(path, map_location=device, weights_only=True)
                if drm:
                    agent.encoder.load_state_dict(snapshot["policy"]["encoder"])
                    agent.actor.load_state_dict(snapshot["policy"]["actor"])
                else:
                    agent.load_policy(snapshot["policy"])
                step = int(snapshot["step"])
                t0 = time.time()
                views = int(env_kwargs(cfg).get("views", 1))
                if args.suites == "v1":
                    metrics = evaluation_suite(pool, agent, step, step // every, int(cfg["seed"]), cfg["eval"], views)
                    record = {"step": step, **metrics, "eval_seconds": time.time() - t0, "time": time.time()}
                else:
                    metrics = evaluation_suite_v2(pool, agent, step, step // every, int(cfg["seed"]), cfg["eval"],
                                                  views, args.v2_reference, cfg["task"])
                    record = {"step": step, "suite": SUITE_V2, "eval_index": step // every,
                              "v2_reference": args.v2_reference,
                              "heldout_v2_cameras": {k: list(v) for k, v in heldout_v2_cameras(cfg["task"]).items()},
                              "suite_definition_sha256": hashlib.sha256(definition.read_bytes()).hexdigest(),
                              **metrics, "eval_seconds": time.time() - t0, "time": time.time()}
                with open(out, "a") as f:
                    f.write(json.dumps(record, allow_nan=False) + "\n")
                if wandb_run is not None:
                    prefix = SUITE_WANDB_PREFIX[args.suites]
                    wandb_run.log({f"{prefix}/{k}": v for k, v in record.items() if isinstance(v, (int, float))
                                   and k != "time"}, step=step)
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
