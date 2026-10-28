"""SNeRL-code SAC on one Meta-World task with one encoder (user directive 2026-10-28; ``s4d/rl/snerl_sac.py``).

The official SNeRL agent (``CurlSacAgent``) trained with the loop of upstream ``snerl/train.py`` on the shared reference
protocol, launched and evaluated exactly like ``scripts/train_rl.py`` (companion ``scripts/eval_rl.py`` on the policy
snapshots written every ``eval.every_steps`` agent steps; ``checkpoints/latest.pt`` resumes the run with its replay).

Loop correspondence with upstream ``train.py`` (lines 270-334), agent step t = upstream step t - 1:
- uniform actions from the environment's action space while t <= ``init_steps`` (``env.action_space.sample()``),
  then a policy sample (``sample_action``: the squashed Gaussian sample);
- one ``agent.update`` per step once t > ``init_steps``, after choosing the action and before the environment step;
- ``done_bool`` 0 at the time limit (infinite bootstrap): our time-limit continuation 1.0;
- evaluation by the companion job on the deterministic mean (``select_action``), with our protocol's episodes, cameras
  and any-step success instead of upstream's 10 final-step-success episodes on fixed cameras.
Replay: our ``Replay`` with 1-step transitions (uniform over the last ``replay_size`` transitions, as upstream), frames
stored once per state (pixels / SNeRL views on disk; frozen-backbone features in RAM), stacks rebuilt by index with the
first frame repeated at episode start, as the environment's own history (upstream hard-codes 2 frames and builds the
first next-observation stack of an episode from the wrong frame; neither is reproduced).
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
from s4d.rl.env import MetaWorldCameraEnv, TrainCameraSampler, env_kwargs, stack_observations  # noqa: E402
from s4d.rl.evaluate import policy_inputs  # noqa: E402
from s4d.rl.protocol import resolve_config  # noqa: E402
from s4d.rl.replay import Replay, replay_iterator  # noqa: E402
from s4d.rl.snerl_sac import SNeRLSAC  # noqa: E402
from s4d.run_identity import record_run_identity  # noqa: E402

SHORT_RUN_STEPS = 1000
ALGORITHM = "snerl_sac"


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
    if int(cfg["train"]["num_train_steps"]) > SHORT_RUN_STEPS:  # short diagnostics are exempt, as in train_rl.py
        require_passed_tests()
    ckpt_dir, snap_dir = run_dir / "checkpoints", run_dir / "snapshots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    resume = latest.is_file()
    cfg.setdefault("run", {}).update(dir=str(run_dir), gpus=GPU_MAPPING, egl_device=EGL_DEVICE)
    record_run_identity(cfg, REPO, run_dir, native_diagnostic_steps=int(cfg["train"]["num_train_steps"]))
    dump_config(cfg, run_dir / "config.yaml")

    seed, tcfg = int(cfg["seed"]), cfg["train"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # upstream utils.set_seed_everywhere
    device = torch.device("cuda")
    torch.set_num_threads(int(tcfg["cpu_threads"]))
    env = MetaWorldCameraEnv(cfg["task"], seed, **env_kwargs(cfg))
    atexit.register(env.close)
    sac = SNeRLSAC(cfg, env.action_dim, env.proprio_dim, device)
    views = int(env_kwargs(cfg).get("views", 1))
    if sac.family == "frozen":
        atom_shape, atom_dtype, frame_stack = sac.features.atom_shape, sac.features.atom_dtype, sac.features.atom_frame_stack
        stored_proprio = env.proprio_dim
    else:  # pixels (CNN) or the SNeRL views; SNeRL also stores the views' normalised poses after the proprioception
        size = int(cfg["env"]["image_size"])
        atom_shape = (3, size, size) if sac.family == "pixel" else (views, 3, size, size)
        atom_dtype, frame_stack = np.uint8, int(cfg["env"]["frame_stack"])
        stored_proprio = env.proprio_dim + (views * 16 if sac.family == "nerf" else 0)
    disk = sac.family != "frozen"
    # Our replay counts states, the terminal state of every episode included; upstream counts transitions. Size the
    # state window so that replay_size transitions stay sampleable (125-step episodes: 126 states per 125 transitions).
    horizon = int(cfg["env"]["max_episode_steps"]) // int(cfg["env"]["action_repeat"])
    states = -(-int(tcfg["replay_size"]) * (horizon + 1) // horizon)
    replay = Replay(
        run_dir / "replay" if disk else None,
        atom_shape,
        atom_dtype,
        stored_proprio,
        env.action_dim,
        states,
        frame_stack,
        1,
        mode="r+" if resume and disk else "w+",
        snapshot_dir=None if disk else run_dir / "replay",
    )
    sampler = TrainCameraSampler(seed, views)
    window = int(tcfg["rolling_window"])
    rolling_success, rolling_return = deque(maxlen=window), deque(maxlen=window)
    start_step, episode = 1, 0
    if resume:
        state = torch.load(latest, map_location="cpu", weights_only=False)
        sac.load_state_dict(state["agent"])
        replay.restore(state["replay_meta"])
        start_step, episode = int(state["step"]) + 1, int(state["episode"])
        rolling_success.extend(state["rolling_success"])
        rolling_return.extend(state["rolling_return"])
        env.reset_count = int(state["train_reset_count"])
        sampler.rng.setstate(state["camera_rng"])
        env.action_space.np_random.bit_generator.state = state["action_space_rng"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        append_jsonl(run_dir / "events.jsonl", {"event": "resumed", "step": start_step - 1, "time": time.time()})
    batches = None

    wandb_run = init_wandb(cfg, run_dir.name, cfg["wandb"]["project"], bool(cfg["wandb"]["enabled"]), run_dir)

    def to_atom(inputs, obs):
        if sac.family == "pixel":
            return env.latest_frame()
        if sac.family == "nerf":
            return obs["rgb"][-1]
        return inputs[0].cpu().numpy() if frame_stack == 1 else inputs[0, -1].cpu().numpy()

    def stored(inputs, proprio):
        if sac.family != "nerf":
            return proprio
        return np.concatenate([proprio, inputs.poses[0, -1].reshape(-1).cpu().numpy()]).astype(np.float32)

    def observe(obs, proprio):
        inputs = policy_inputs(sac, stack_observations([obs]))
        return inputs, proprio, to_atom(inputs, obs), stored(inputs, proprio)

    def begin_episode():
        obs, proprio = env.reset(sampler())
        inputs, proprio, atom, extra = observe(obs, proprio)
        replay.add_initial(atom, extra)
        return inputs, proprio

    inputs, proprio = begin_episode()
    episode_return = episode_success = 0.0
    metrics: dict = {}
    timers = {"act_env": 0.0, "update": 0.0, "checkpoint": 0.0}
    interval_start, interval_steps = time.time(), 0
    total, init_steps = int(tcfg["num_train_steps"]), int(tcfg["init_steps"])
    for step in range(start_step, total + 1):
        t0, spent_updating = time.time(), 0.0
        if step <= init_steps:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = sac.act(inputs, proprio[None], step=step, eval_mode=False)[0].astype(np.float32)
        if step > init_steps and step % int(tcfg["update_every_steps"]) == 0:
            t1 = time.time()
            if batches is None:
                batches = replay_iterator(replay, tcfg["batch_size"], 1.0, seed + start_step, device)
            metrics = sac.update(batches, step - 1)  # upstream's 0-based step drives its update intervals
            spent_updating = time.time() - t1
            timers["update"] += spent_updating
        obs, proprio, reward, done, info = env.step(action)
        inputs, proprio, atom, extra = observe(obs, proprio)
        replay.add(action, reward, 1.0, atom, extra, done)  # continuation 1.0: upstream done_bool 0 at the time limit
        episode_return += reward
        episode_success = max(episode_success, info["success"])
        if done:
            rolling_success.append(episode_success)
            rolling_return.append(episode_return)
            episode += 1
            if wandb_run is not None:
                wandb_run.log({"train/episode_success": episode_success, "train/episode_return": episode_return}, step=step)
            inputs, proprio = begin_episode()
            episode_return = episode_success = 0.0
        timers["act_env"] += time.time() - t0 - spent_updating
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
                **{f"train/{k.replace('/', '_')}": v for k, v in metrics.items()},
                "time": time.time(),
            }
            append_jsonl(run_dir / "train.jsonl", record)
            if wandb_run is not None:
                wandb_run.log({k: v for k, v in record.items() if v is not None and k != "time"}, step=step)
            interval_start, interval_steps = time.time(), 0

        if step % int(cfg["eval"]["every_steps"]) == 0:
            atomic_save({"step": step, "algorithm": ALGORITHM, "policy": sac.policy_state()}, snap_dir / f"step_{step:07d}.pt")

        if step % int(tcfg["checkpoint_every_steps"]) == 0 or step == total:
            t3 = time.time()
            atomic_save(
                {
                    "step": step,
                    "episode": episode,
                    "agent": sac.state_dict(),
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

    final = {"completed": True, "steps": total, "episodes": episode, "timers": timers, "rss_gb": rss_gb()}
    (run_dir / "final.json").write_text(json.dumps(final, indent=1))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
