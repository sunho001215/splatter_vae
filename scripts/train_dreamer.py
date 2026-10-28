"""Dreamer v1 on one Meta-World task with one encoder (user directive 2026-10-28; ``s4d/rl/dreamer_v1.py``).

The dreamer-pytorch agent (upstream modules unchanged) trained with the loop of upstream ``main.py`` on the shared
reference protocol, launched and evaluated like ``scripts/train_sac.py`` (companion ``scripts/eval_rl.py`` on the policy
snapshots written every ``eval.every_steps`` agent steps; ``checkpoints/latest.pt`` resumes the run with its replay).

Loop correspondence with upstream ``main.py`` (agent steps; one episode = 125 agent steps):
- ``seed_episodes`` episodes of uniform random actions fill the replay first (main.py:170-179; here
  ``env.action_space.sample()``); they count toward the agent-step budget;
- then, per episode, ``collect_interval`` updates (world model, actor, value: main.py:340-509), followed by one
  collection episode with the exploration policy (main.py:530-560: posterior update, policy sample + N(0, 0.3) noise);
- every row stores (o_t, a_t, r_{t+1}, done_{t+1}) as main.py:175 / 553; ``done`` is the time limit (our only episode
  end), which Dreamer v1 bootstraps through (see ``s4d/rl/dreamer_v1.py``);
- evaluation by the companion job (deterministic ``get_action(det=True)`` as main.py:278 for ``--test``; our protocol's
  episodes, cameras and any-step success) instead of upstream's ``test_interval`` episodes; checkpoints every
  ``checkpoint_every_steps`` agent steps at an episode boundary instead of every 50 episodes (main.py:667-685), with the
  optimisers, RNG states and the replay (always, not only with ``--checkpoint-experience``).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
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
from s4d.gpu_guard import validate_runtime_path  # noqa: E402
from s4d.rl.dreamer_v1 import DreamerAgent  # noqa: E402
from s4d.rl.env import MetaWorldCameraEnv, TrainCameraSampler, env_kwargs, stack_observations  # noqa: E402
from s4d.rl.evaluate import policy_inputs  # noqa: E402
from s4d.rl.protocol import resolve_config  # noqa: E402
from s4d.run_identity import record_run_identity  # noqa: E402

SHORT_RUN_STEPS = 1000
ALGORITHM = "dreamer_v1"
LOSSES = ("observation_loss", "reward_loss", "kl_loss", "actor_loss", "value_loss", "imagined_return")


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

    run_dir = validate_runtime_path(Path(args.run_dir))
    cfg = resolve_config(args.task, args.encoder, args.seed, args.set, algorithm=ALGORITHM)
    tcfg = cfg["train"]
    total = int(tcfg["num_train_steps"])
    if total > SHORT_RUN_STEPS:  # short diagnostics are exempt, as in train_rl.py / train_sac.py
        require_passed_tests()
    horizon = int(cfg["env"]["max_episode_steps"]) // int(cfg["env"]["action_repeat"])
    if int(tcfg["checkpoint_every_steps"]) % horizon:
        raise ValueError(f"train.checkpoint_every_steps must be a multiple of the {horizon}-step episode")
    if int(tcfg["seed_episodes"]) * horizon <= int(tcfg["chunk_size"]):  # memory.py:42 draws below idx - chunk
        raise ValueError("the seed episodes must hold more rows than one chunk")
    ckpt_dir, snap_dir = run_dir / "checkpoints", run_dir / "snapshots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    resume = latest.is_file()
    cfg.setdefault("run", {}).update(dir=str(run_dir), gpus=GPU_MAPPING, egl_device=EGL_DEVICE)
    record_run_identity(cfg, REPO, run_dir, native_diagnostic_steps=total)
    dump_config(cfg, run_dir / "config.yaml")

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)  # upstream replay sampling draws from numpy's global RNG (memory.py:42)
    torch.manual_seed(seed)  # main.py:133-138
    device = torch.device("cuda")
    torch.set_num_threads(int(tcfg["cpu_threads"]))
    env = MetaWorldCameraEnv(cfg["task"], seed, **env_kwargs(cfg))
    atexit.register(env.close)
    agent = DreamerAgent(cfg, env.action_dim, env.proprio_dim, device)
    replay = agent.make_replay(run_dir / "replay", mode="r+" if resume else "w+")
    sampler = TrainCameraSampler(seed, int(env_kwargs(cfg).get("views", 1)))
    window = int(tcfg["rolling_window"])
    rolling_success, rolling_return = deque(maxlen=window), deque(maxlen=window)
    step, episode, updates = 0, 0, 0
    if resume:
        state = torch.load(latest, map_location="cpu", weights_only=False)
        agent.load_state_dict(state["agent"])
        replay.restore(state["replay_meta"])
        step, episode, updates = int(state["step"]), int(state["episode"]), int(state["updates"])
        rolling_success.extend(state["rolling_success"])
        rolling_return.extend(state["rolling_return"])
        env.reset_count = int(state["train_reset_count"])
        sampler.rng.setstate(state["camera_rng"])
        env.action_space.np_random.bit_generator.state = state["action_space_rng"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        append_jsonl(run_dir / "events.jsonl", {"event": "resumed", "step": step, "time": time.time()})

    wandb_run = init_wandb(cfg, run_dir.name, cfg["wandb"]["project"], bool(cfg["wandb"]["enabled"]), run_dir)

    def observe(obs):
        """Policy inputs of the observation and its replay row contents (64-px target frame, feature atom)."""
        inputs = policy_inputs(agent, stack_observations([obs]))
        frame = agent.target_frame(env.latest_frame())
        if agent.use_pixels:
            return inputs, frame, None
        return inputs, frame, (inputs[0] if agent.atom_stack == 1 else inputs[0, -1]).cpu().numpy()

    timers = {"act_env": 0.0, "update": 0.0, "checkpoint": 0.0}
    sums: dict[str, float] = {}
    interval_updates, interval_start, interval_steps = 0, time.time(), 0
    seed_episodes, collect_interval = int(tcfg["seed_episodes"]), int(tcfg["collect_interval"])
    every = int(cfg["eval"]["every_steps"])

    def log(step: int) -> None:
        nonlocal sums, interval_updates, interval_start, interval_steps
        record = {
            "step": step,
            "episode": episode,
            "updates": updates,
            "fps": interval_steps / max(time.time() - interval_start, 1e-9),
            "rolling_success": float(np.mean(rolling_success)) if rolling_success else None,
            "rolling_return": float(np.mean(rolling_return)) if rolling_return else None,
            "gpu_max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_max_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            "rss_gb": rss_gb(),
            "replay_gb": replay.nbytes() / 2**30,
            "cpu_seconds": time.process_time(),
            **{f"time_{k}": v for k, v in timers.items()},
            **{f"train/{k}": sums[k] / interval_updates for k in LOSSES if interval_updates},
            "time": time.time(),
        }
        append_jsonl(run_dir / "train.jsonl", record)
        if wandb_run is not None:
            wandb_run.log({k: v for k, v in record.items() if v is not None and k != "time"}, step=step)
        sums, interval_updates, interval_start, interval_steps = {}, 0, time.time(), 0

    while step < total:
        seeding = episode < seed_episodes
        if not seeding:  # main.py:340-509: collect_interval updates before each collection episode
            t1 = time.time()
            agent.train(True)
            for _ in range(collect_interval):
                metrics = agent.update(replay)
                for k in LOSSES:
                    sums[k] = sums.get(k, 0.0) + metrics[k]
                updates += 1
                interval_updates += 1
            timers["update"] += time.time() - t1
        t0 = time.time()
        agent.train(False)
        obs, proprio = env.reset(sampler())
        inputs, frame, atom = observe(obs)
        agent.begin_episodes(1)  # main.py:533-537: zero belief, state and previous action
        episode_return = episode_success = 0.0
        t, done = 0, False
        while not done and step < total:
            if seeding:
                action = env.action_space.sample().astype(np.float32)  # main.py:173
            else:
                action = agent.act(inputs, proprio[None], step=step, eval_mode=False)[0].astype(np.float32)
            obs, next_proprio, reward, done, info = env.step(action)
            replay.append(frame, atom, proprio, action, reward, done, t)  # main.py:175 / 553
            step, t = step + 1, t + 1
            interval_steps += 1
            episode_return += reward
            episode_success = max(episode_success, info["success"])
            if not done:
                inputs, frame, atom = observe(obs)
                proprio = next_proprio
            else:
                episode += 1
                rolling_success.append(episode_success)
                rolling_return.append(episode_return)
                if wandb_run is not None:
                    wandb_run.log({"train/episode_success": episode_success, "train/episode_return": episode_return},
                                  step=step)
            if step % int(tcfg["log_every_steps"]) == 0:
                timers["act_env"] += time.time() - t0
                t0 = time.time()
                log(step)
            if step % every == 0:
                atomic_save({"step": step, "algorithm": ALGORITHM, "policy": agent.policy_state()},
                            snap_dir / f"step_{step:07d}.pt")
        timers["act_env"] += time.time() - t0
        if (done and step % int(tcfg["checkpoint_every_steps"]) == 0) or step == total:
            t3 = time.time()
            atomic_save(
                {
                    "step": step,
                    "episode": episode,
                    "updates": updates,
                    "agent": agent.state_dict(),
                    "replay_meta": replay.checkpoint(),
                    "rolling_success": list(rolling_success),
                    "rolling_return": list(rolling_return),
                    "train_reset_count": env.reset_count,
                    "camera_rng": sampler.rng.getstate(),
                    "action_space_rng": env.action_space.np_random.bit_generator.state,
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                latest,
            )
            timers["checkpoint"] += time.time() - t3

    final = {"completed": True, "steps": total, "episodes": episode, "updates": updates, "timers": timers,
             "gpu_max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
             "gpu_max_reserved_gb": torch.cuda.max_memory_reserved() / 2**30, "rss_gb": rss_gb()}
    (run_dir / "final.json").write_text(json.dumps(final, indent=1))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
