"""DrQ-v2 on one Meta-World task with one encoder, following the reference training loop.

Launch through ``scripts/jobs.py`` with exactly one allowed GPU UUID. The run directory must be inside
the repository. If ``<run>/checkpoints/latest.pt`` exists the run resumes from it and reopens the replay
buffer; transitions written after that checkpoint stay in replay, so a resumed run is not bit-identical.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus, guard_mujoco  # noqa: E402

GPU_MAPPING = guard_gpus()
EGL_DEVICE = guard_mujoco()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from s4d.config import dump_config, load_config  # noqa: E402
from s4d.diag.wandb_log import init_wandb  # noqa: E402
from s4d.rl.agent import DrQv2Agent  # noqa: E402
from s4d.rl.env import MetaWorldCameraEnv, TrainCameraSampler, matching_frame_gap  # noqa: E402
from s4d.rl.evaluate import evaluation_suite, policy_inputs  # noqa: E402
from s4d.rl.replay import MemmapReplayBufferStorage, make_replay_loader  # noqa: E402


def encoder_frame_gap(export_path: str, action_repeat: int) -> int:
    payload = torch.load(export_path, map_location="cpu", weights_only=True)
    return matching_frame_gap(payload["frame_strides"], int(payload["encoder_config"]["num_frames"]), action_repeat)


def git_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def make_env(cfg: dict, seed: int) -> MetaWorldCameraEnv:
    e = cfg["env"]
    return MetaWorldCameraEnv(
        cfg["task"],
        seed,
        image_size=e["image_size"],
        frame_stack=e["frame_stack"],
        frame_gap=e["frame_gap"],
        action_repeat=e["action_repeat"],
        max_episode_steps=e["max_episode_steps"],
        proprio_indices=e["proprio_indices"],
    )


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
    cfg = load_config([REPO / "configs/rl/base.yaml", REPO / f"configs/rl/encoders/{args.encoder}.yaml"], args.set)
    cfg.update(task=args.task, seed=args.seed, encoder_name=args.encoder)
    if cfg["vision"]["encoder_type"] == "splatter4d":
        cfg["env"]["frame_gap"] = encoder_frame_gap(cfg["vision"]["export_path"], cfg["env"]["action_repeat"])
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    resume = latest.is_file()
    cfg["run"] = {"dir": str(run_dir), "git_commit": git_commit(), "gpus": GPU_MAPPING, "egl_device": EGL_DEVICE}
    dump_config(cfg, run_dir / "config.yaml")

    seed, tcfg, ecfg = int(cfg["seed"]), cfg["train"], cfg["eval"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    train_env = make_env(cfg, seed)
    eval_envs = [make_env(cfg, seed + 1 + 7919 * i) for i in range(int(ecfg["pool_size"]))]
    agent = DrQv2Agent(cfg, train_env.action_dim, train_env.proprio_dim, device)
    enc = agent.encoder
    storage = MemmapReplayBufferStorage(
        run_dir / "replay",
        enc.replay_atom_shape,
        enc.replay_atom_dtype,
        (train_env.proprio_dim,),
        (train_env.action_dim,),
        tcfg["replay_size"],
        enc.replay_atom_frame_stack,
        tcfg["nstep"],
        reset=not resume,
    )
    loader, _ = make_replay_loader(storage, tcfg["batch_size"], tcfg["replay_num_workers"], tcfg["discount"])
    sampler = TrainCameraSampler(seed)
    window = int(tcfg["rolling_window"])
    rolling_success, rolling_return = deque(maxlen=window), deque(maxlen=window)
    start_step, episode = 1, 0
    if resume:
        state = torch.load(latest, map_location=device, weights_only=False)  # own checkpoint (RNG payload)
        agent.load_state_dict(state["agent"])
        start_step, episode = int(state["step"]) + 1, int(state["episode"])
        rolling_success.extend(state["rolling_success"])
        rolling_return.extend(state["rolling_return"])
        train_env.reset_count = int(state["train_reset_count"])
        sampler.rng.setstate(state["camera_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        storage.resume()
        append_jsonl(run_dir / "events.jsonl", {"event": "resumed", "step": start_step - 1, "time": time.time()})

    wandb_run = init_wandb(cfg, run_dir.name, cfg["wandb"]["project"], bool(cfg["wandb"]["enabled"]), run_dir)

    def log(record: dict, step: int) -> None:
        if wandb_run is not None:
            wandb_run.log(record, step=step)

    def to_atom(policy_obs, latest_frame):
        if enc.replay_atom_is_stack_feature:
            return policy_obs[0].cpu().numpy()
        if enc.replay_atom_is_feature:
            return policy_obs[0, -1].cpu().numpy()
        return latest_frame

    def begin_episode():
        obs, proprio = train_env.reset(sampler())
        policy_obs = policy_inputs(agent, obs[None])
        storage.add_initial(to_atom(policy_obs, train_env.latest_frame()), proprio)
        return policy_obs, proprio

    policy_obs, proprio = begin_episode()
    episode_return = episode_success = 0.0
    replay_iter, metrics = None, {}
    timers = {"env": 0.0, "update": 0.0, "eval": 0.0}
    interval_start, interval_steps = time.time(), 0
    for step in range(start_step, int(tcfg["num_train_steps"]) + 1):
        t0 = time.time()
        if step <= int(tcfg["seed_steps"]):
            action = train_env.action_space.sample().astype(np.float32)
        else:
            action = agent.act(policy_obs, proprio[None], step=step, eval_mode=False)[0].astype(np.float32)
        obs, proprio, reward, done, info = train_env.step(action)
        policy_obs = policy_inputs(agent, obs[None])
        storage.add(action, reward, 0.0 if done else 1.0, to_atom(policy_obs, train_env.latest_frame()), proprio, done)
        episode_return += reward
        episode_success = max(episode_success, info["success"])
        t1 = time.time()
        timers["env"] += t1 - t0
        if (
            step > int(tcfg["seed_steps"])
            and len(storage) >= int(tcfg["batch_size"])
            and step % int(tcfg["update_every_steps"]) == 0
        ):
            replay_iter = replay_iter or iter(loader)
            metrics = agent.update(replay_iter, step)
            timers["update"] += time.time() - t1
        interval_steps += 1

        if done:
            rolling_success.append(episode_success)
            rolling_return.append(episode_return)
            episode += 1
            log({"train/episode_success": episode_success, "train/episode_return": episode_return}, step)
            policy_obs, proprio = begin_episode()
            episode_return = episode_success = 0.0

        if step % int(tcfg["log_every_steps"]) == 0:
            elapsed = time.time() - interval_start
            record = {
                "step": step,
                "episode": episode,
                "fps": interval_steps / max(elapsed, 1e-9),
                "rolling_success": float(np.mean(rolling_success)) if rolling_success else None,
                "rolling_return": float(np.mean(rolling_return)) if rolling_return else None,
                "gpu_max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                "cpu_seconds": time.process_time(),
                **{f"time_{k}": v for k, v in timers.items()},
                **{f"train/{k}": v for k, v in metrics.items()},
                "time": time.time(),
            }
            append_jsonl(run_dir / "train.jsonl", record)
            log({k: v for k, v in record.items() if v is not None and k != "time"}, step)
            interval_start, interval_steps = time.time(), 0

        if step % int(ecfg["every_steps"]) == 0:
            t2 = time.time()
            result = {"step": step, **evaluation_suite(eval_envs, agent, step, ecfg)}
            timers["eval"] += time.time() - t2
            append_jsonl(run_dir / "eval.jsonl", result)
            log(result, step)
            interval_start = time.time()

        if step % int(tcfg["checkpoint_every_steps"]) == 0 or step == int(tcfg["num_train_steps"]):
            atomic_save(
                {
                    "step": step,
                    "episode": episode,
                    "agent": agent.state_dict(),
                    "rolling_success": list(rolling_success),
                    "rolling_return": list(rolling_return),
                    "train_reset_count": train_env.reset_count,
                    "camera_rng": sampler.rng.getstate(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                latest,
            )

    final = {"completed": True, "steps": int(tcfg["num_train_steps"]), "episodes": episode, "timers": timers}
    (run_dir / "final.json").write_text(json.dumps(final, indent=1))
    if not tcfg["keep_replay"]:
        for path in (run_dir / "replay").glob("*.memmap"):
            path.unlink()
    train_env.close()
    for env in eval_envs:
        env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
