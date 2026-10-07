"""DrM training on one Meta-World task with one encoder (official DrM loop on the shared reference protocol).

Launch through ``scripts/jobs.py`` with exactly one allowed GPU UUID. Evaluation runs in a companion job
(``scripts/eval_rl.py``) on the policy snapshots written here every ``eval.every_steps`` agent steps.
If ``<run>/checkpoints/latest.pt`` exists the run resumes from it, with the replay buffer returned to
its checkpointed contents.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus, guard_mujoco, require_passed_tests  # noqa: E402

GPU_MAPPING = guard_gpus()
EGL_DEVICE = guard_mujoco()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from s4d.config import dump_config  # noqa: E402
from s4d.diag.wandb_log import init_wandb  # noqa: E402
from s4d.rl.agent import DrMAgent  # noqa: E402
from s4d.rl.env import MetaWorldCameraEnv, TrainCameraSampler, env_kwargs  # noqa: E402
from s4d.rl.evaluate import policy_inputs  # noqa: E402
from s4d.rl.protocol import resolve_config  # noqa: E402
from s4d.rl.replay import Replay, replay_iterator  # noqa: E402

SHORT_RUN_STEPS = 1000


def git_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--encoder", required=True, help="configs/rl/encoders/<name>.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.path=value")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if REPO.resolve() not in run_dir.parents:
        raise ValueError("RL run directories must be inside the repository")
    cfg = resolve_config(args.task, args.encoder, args.seed, args.set)
    if int(cfg["train"]["num_train_steps"]) > SHORT_RUN_STEPS:  # short diagnostics (like 5-episode pilots) are exempt
        require_passed_tests()
    ckpt_dir, snap_dir = run_dir / "checkpoints", run_dir / "snapshots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    resume = latest.is_file()
    cfg["run"] = {"dir": str(run_dir), "git_commit": git_commit(), "gpus": GPU_MAPPING, "egl_device": EGL_DEVICE}
    dump_config(cfg, run_dir / "config.yaml")

    seed, tcfg = int(cfg["seed"]), cfg["train"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    env = MetaWorldCameraEnv(cfg["task"], seed, **env_kwargs(cfg))
    atexit.register(env.close)  # release EGL before interpreter teardown, also when the run crashes
    agent = DrMAgent(cfg, env.action_dim, env.proprio_dim, device)
    enc = agent.encoder
    disk = enc.backbone_trainable  # pixels on disk for end-to-end encoders, latents in RAM for frozen ones
    replay = Replay(
        run_dir / "replay" if disk else None,
        enc.replay_atom_shape,
        enc.replay_atom_dtype,
        env.proprio_dim,
        env.action_dim,
        tcfg["replay_size"],
        enc.replay_atom_frame_stack,
        tcfg["nstep"],
        mode="r+" if resume and disk else "w+",
        snapshot_dir=None if disk else run_dir / "replay",
    )
    sampler = TrainCameraSampler(seed)
    window = int(tcfg["rolling_window"])
    rolling_success, rolling_return = deque(maxlen=window), deque(maxlen=window)
    start_step, episode = 1, 0
    if resume:
        # Own checkpoint (RNG payload). Load on the CPU: RNG states must stay CPU byte tensors, and
        # load_state_dict moves weights and optimizer state to their parameters' device.
        state = torch.load(latest, map_location="cpu", weights_only=False)
        agent.load_state_dict(state["agent"])
        replay.restore(state["replay_meta"])
        start_step, episode = int(state["step"]) + 1, int(state["episode"])
        rolling_success.extend(state["rolling_success"])
        rolling_return.extend(state["rolling_return"])
        env.reset_count = int(state["train_reset_count"])
        sampler.rng.setstate(state["camera_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        append_jsonl(run_dir / "events.jsonl", {"event": "resumed", "step": start_step - 1, "time": time.time()})
    batches = None  # created at the first update, as in the reference loop

    wandb_run = init_wandb(cfg, run_dir.name, cfg["wandb"]["project"], bool(cfg["wandb"]["enabled"]), run_dir)

    def to_atom(policy_obs, latest_frame):
        if enc.replay_atom_is_stack_feature:
            return policy_obs[0].cpu().numpy()
        if enc.replay_atom_is_feature:
            return policy_obs[0, -1].cpu().numpy()
        return latest_frame

    def begin_episode():
        obs, proprio = env.reset(sampler())
        policy_obs = policy_inputs(agent, obs[None])
        replay.add_initial(to_atom(policy_obs, env.latest_frame()), proprio)
        return policy_obs, proprio

    policy_obs, proprio = begin_episode()
    episode_return = episode_success = 0.0
    metrics: dict = {}
    timers = {"act_env": 0.0, "update": 0.0, "checkpoint": 0.0}
    interval_start, interval_steps = time.time(), 0
    total = int(tcfg["num_train_steps"])
    for step in range(start_step, total + 1):
        t0 = time.time()
        # Uniform actions before num_expl_steps happen inside DrMAgent.act, as in the official agent.
        action = agent.act(policy_obs, proprio[None], step=step, eval_mode=False)[0].astype(np.float32)
        obs, proprio, reward, done, info = env.step(action)
        policy_obs = policy_inputs(agent, obs[None])
        continuation = float(tcfg["time_limit_continuation"]) if done else 1.0  # Meta-World only truncates
        replay.add(action, reward, continuation, to_atom(policy_obs, env.latest_frame()), proprio, done)
        episode_return += reward
        episode_success = max(episode_success, info["success"])
        if done:
            rolling_success.append(episode_success)
            rolling_return.append(episode_return)
            episode += 1
            if wandb_run is not None:
                wandb_run.log({"train/episode_success": episode_success, "train/episode_return": episode_return}, step=step)
            policy_obs, proprio = begin_episode()
            episode_return = episode_success = 0.0
        t1 = time.time()
        timers["act_env"] += t1 - t0
        if (
            step >= int(tcfg["num_seed_steps"])
            and len(replay) >= int(tcfg["batch_size"])
            and step % int(tcfg["update_every_steps"]) == 0
        ):
            if batches is None:
                batches = replay_iterator(
                    replay, tcfg["batch_size"], tcfg["discount"], tcfg["replay_num_workers"], seed + start_step
                )
            metrics = agent.update(batches, step)
            timers["update"] += time.time() - t1
            if "perturb_factor" in metrics:
                event = {"event": "perturbed", "step": step, "factor": metrics["perturb_factor"], "time": time.time()}
                append_jsonl(run_dir / "events.jsonl", event)
                if wandb_run is not None:
                    wandb_run.log({"train/perturb_event": 1.0, "train/perturb_factor": metrics["perturb_factor"]}, step=step)
        interval_steps += 1

        if step % int(tcfg["log_every_steps"]) == 0:
            record = {
                "step": step,
                "episode": episode,
                "fps": interval_steps / max(time.time() - interval_start, 1e-9),
                "rolling_success": float(np.mean(rolling_success)) if rolling_success else None,
                "rolling_return": float(np.mean(rolling_return)) if rolling_return else None,
                "gpu_max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                "rss_gb": rss_gb(),
                "replay_gb": replay.nbytes() / 2**30,
                "cpu_seconds": time.process_time(),
                **{f"time_{k}": v for k, v in timers.items()},
                **{f"train/{k}": v for k, v in metrics.items()},
                "time": time.time(),
            }
            append_jsonl(run_dir / "train.jsonl", record)
            if wandb_run is not None:
                wandb_run.log({k: v for k, v in record.items() if v is not None and k != "time"}, step=step)
            interval_start, interval_steps = time.time(), 0

        if step % int(cfg["eval"]["every_steps"]) == 0:
            policy = {"encoder": agent.encoder.state_dict(), "actor": agent.actor.state_dict()}
            atomic_save({"step": step, "policy": policy}, snap_dir / f"step_{step:07d}.pt")

        if step % int(tcfg["checkpoint_every_steps"]) == 0 or step == total:
            t2 = time.time()
            atomic_save(
                {
                    "step": step,
                    "episode": episode,
                    "agent": agent.state_dict(),
                    "replay_meta": replay.checkpoint(),
                    "rolling_success": list(rolling_success),
                    "rolling_return": list(rolling_return),
                    "train_reset_count": env.reset_count,
                    "camera_rng": sampler.rng.getstate(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                latest,
            )
            timers["checkpoint"] += time.time() - t2

    final = {"completed": True, "steps": total, "episodes": episode, "timers": timers, "rss_gb": rss_gb()}
    (run_dir / "final.json").write_text(json.dumps(final, indent=1))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
