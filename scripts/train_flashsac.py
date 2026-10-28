"""FlashSAC on one Meta-World task with one encoder (user directive 2026-10-29; ``s4d/rl/flashsac.py``).

The official FlashSAC agent (third_party/flashsac @ 87edc90) with our visual adapter and no RL image augmentation,
trained with the loop of upstream ``train.py`` on the shared reference protocol, launched and evaluated exactly like
``scripts/train_sac.py`` (companion ``scripts/eval_rl.py`` on the policy snapshots written every ``eval.every_steps``
agent steps; ``checkpoints/latest.pt`` resumes the run with its replay and RNG states).

Loop correspondence with upstream ``train.py:113-192`` (one environment; agent step t = upstream interaction_step t):
- the action is a policy sample (``sample_actions(training=True)``: noise repetition) once the replay holds
  ``agent.buffer_min_length`` transitions, before that uniform from the environment's action space (train.py:124-127);
- the environment step; the transition is stored and the raw agent-step reward (sum over the 2 simulator steps) updates
  the reward normaliser, with the time limit as a truncation (``process_transition``, train.py:131-148);
- once the replay holds ``buffer_min_length`` transitions: ``update_counter += updates_per_interaction_step`` and one
  update per whole unit (train.py:152-159; UTD 0.5: one update every second agent step);
- evaluation by the companion job on the deterministic tanh(mean) with the protocol's episodes and cameras (upstream
  evaluates in-process on its own env every 10 % of training).
Replay: our ``Replay`` with 3-step transitions and ``end_windows`` (uniform over stored transitions; an n-step window
never crosses an episode boundary and, as upstream's n-step buffer, a window starting within the last 2 transitions of
a finished episode stops at its final state; the update bootstraps with gamma^n as upstream), frames stored once per
state (pixels on disk, frozen-backbone features in RAM), stacks rebuilt by index with the first frame repeated at an
episode start, as the environment's own history.
Resume: checkpoints fall on episode ends and are written before the next episode is prepared (its camera draw and
reset happen at the start of the next agent step), so a resumed run continues the camera, reset and RNG sequences.
Logged returns (train.jsonl, W&B) are raw environment returns; FlashSAC's reward normalisation is internal.
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
from s4d.rl.flashsac import FlashSAC  # noqa: E402
from s4d.rl.protocol import resolve_config  # noqa: E402
from s4d.rl.replay import Replay, replay_iterator  # noqa: E402
from s4d.run_identity import record_run_identity  # noqa: E402

SHORT_RUN_STEPS = 1000
ALGORITHM = "flashsac"


def rss_gb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def device_used_gb() -> float:
    """Memory in use on the (MIG) device: this process's CUDA context, allocator and MuJoCo EGL renderer, plus any
    other process on the same device."""
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**30


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

    seed, tcfg, acfg = int(cfg["seed"]), cfg["train"], cfg["agent"]
    random.seed(seed)  # upstream train.py:50-55
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    # train.py:57-61 also enables TF32 matmuls; not adopted, so that the frozen backbones compute the same features
    # here as in the shared evaluator (default precision, as for every other method).
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(int(tcfg["cpu_threads"]))
    env = MetaWorldCameraEnv(cfg["task"], seed, **env_kwargs(cfg))
    atexit.register(env.close)
    agent = FlashSAC(cfg, env.action_dim, env.proprio_dim, device)
    views = int(env_kwargs(cfg).get("views", 1))
    if agent.use_pixels:
        size = int(cfg["env"]["image_size"])
        atom_shape, atom_dtype, frame_stack = (3, size, size), np.uint8, int(cfg["env"]["frame_stack"])
    else:
        f = agent.features
        atom_shape, atom_dtype, frame_stack = f.atom_shape, f.atom_dtype, f.atom_frame_stack
    disk = agent.use_pixels
    # Our replay counts states, the terminal state of every episode included; upstream counts transitions. Size the
    # state window so that buffer_max_length transitions stay sampleable (126 states per 125-step episode).
    horizon = int(cfg["env"]["max_episode_steps"]) // int(cfg["env"]["action_repeat"])
    if int(tcfg["checkpoint_every_steps"]) % horizon or (
            int(tcfg["num_train_steps"]) > SHORT_RUN_STEPS and int(tcfg["num_train_steps"]) % horizon):
        # checkpoints then fall on episode ends, so a resume drops no stored transition (episodes are 125 steps)
        raise ValueError(f"checkpoint interval and budget must be multiples of the {horizon}-step episode")
    states = -(-int(acfg["buffer_max_length"]) * (horizon + 1) // horizon)
    replay = Replay(
        run_dir / "replay" if disk else None,
        atom_shape,
        atom_dtype,
        env.proprio_dim,
        env.action_dim,
        states,
        frame_stack,
        int(acfg["n_step"]),
        mode="r+" if resume and disk else "w+",
        snapshot_dir=None if disk else run_dir / "replay",
        end_windows=True,
    )
    sampler = TrainCameraSampler(seed, views)
    window = int(tcfg["rolling_window"])
    rolling_success, rolling_return = deque(maxlen=window), deque(maxlen=window)
    start_step, episode, update_counter, updates = 1, 0, 0.0, 0
    if resume:
        state = torch.load(latest, map_location="cpu", weights_only=False)
        agent.load_state_dict(state["agent"])
        replay.restore(state["replay_meta"])
        start_step, episode = int(state["step"]) + 1, int(state["episode"])
        update_counter, updates = float(state["update_counter"]), int(state["updates"])
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

    def observe(obs):
        """Policy input of an observation and its replay atom (the newest frame / frame feature, or the joint
        feature of the whole history)."""
        inputs = policy_inputs(agent, stack_observations([obs]))
        if agent.use_pixels:
            return inputs, env.latest_frame()
        return inputs, (inputs[0] if frame_stack == 1 else inputs[0, -1]).cpu().numpy()

    def begin_episode():
        obs, proprio = env.reset(sampler())
        inputs, atom = observe(obs)
        replay.add_initial(atom, proprio)
        return inputs, proprio

    # The next episode is prepared at the start of an agent step, so an episode-end checkpoint precedes its camera
    # draw and reset; a run (fresh or resumed at an episode end) starts by preparing one.
    inputs = proprio = None
    episode_return = episode_success = 0.0
    timers = {"act_env": 0.0, "update": 0.0, "checkpoint": 0.0}
    interval_start, interval_steps, interval_updates = time.time(), 0, 0
    total = int(tcfg["num_train_steps"])
    min_length, utd = int(acfg["buffer_min_length"]), float(tcfg["updates_per_interaction_step"])
    for step in range(start_step, total + 1):
        t0 = time.time()
        if inputs is None:
            inputs, proprio = begin_episode()
        if len(replay) >= min_length:  # train.py:124-127 (agent.can_start_training)
            action = agent.act(inputs, proprio[None], step=step, eval_mode=False)[0].astype(np.float32)
        else:
            action = env.action_space.sample().astype(np.float32)
        obs, proprio, reward, done, info = env.step(action)
        inputs, atom = observe(obs)
        replay.add(action, reward, 1.0, atom, proprio, done)  # continuation 1.0: the time limit is bootstrapped
        agent.observe_reward(reward, done)
        episode_return += reward
        episode_success = max(episode_success, info["success"])
        if done:
            rolling_success.append(episode_success)
            rolling_return.append(episode_return)
            episode += 1
            if wandb_run is not None:
                wandb_run.log({"train/episode_success": episode_success, "train/episode_return": episode_return}, step=step)
            inputs = proprio = None  # prepared at the start of the next agent step
            episode_return = episode_success = 0.0
        timers["act_env"] += time.time() - t0
        if len(replay) >= min_length:  # train.py:152-159
            t1 = time.time()
            if batches is None:
                batches = replay_iterator(replay, int(acfg["sample_batch_size"]), float(acfg["gamma"]),
                                          seed + start_step, device)
            update_counter += utd
            while update_counter >= 1:
                agent.update(batches)
                update_counter -= 1
                updates += 1
                interval_updates += 1
            timers["update"] += time.time() - t1
        interval_steps += 1

        if step % int(tcfg["log_every_steps"]) == 0:
            record = {
                "step": step,
                "episode": episode,
                "updates": updates,
                "fps": interval_steps / max(time.time() - interval_start, 1e-9),
                "interval_updates": interval_updates,
                "rolling_success": float(np.mean(rolling_success)) if rolling_success else None,
                "rolling_return": float(np.mean(rolling_return)) if rolling_return else None,  # raw env returns
                "reward_scale": agent.reward_scale(),
                "gpu_max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_max_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
                "gpu_device_used_gb": device_used_gb(),
                "rss_gb": rss_gb(),
                "replay_gb": replay.nbytes() / 2**30,
                "replay_transitions": len(replay),
                "cpu_seconds": time.process_time(),
                **{f"time_{k}": v for k, v in timers.items()},
                **{f"train/{k.replace('/', '_')}": v for k, v in agent.metrics().items()},
                "time": time.time(),
            }
            append_jsonl(run_dir / "train.jsonl", record)
            if wandb_run is not None:
                wandb_run.log({k: v for k, v in record.items() if v is not None and k != "time"}, step=step)
            interval_start, interval_steps, interval_updates = time.time(), 0, 0

        if step % int(cfg["eval"]["every_steps"]) == 0:
            atomic_save({"step": step, "algorithm": ALGORITHM, "policy": agent.policy_state()},
                        snap_dir / f"step_{step:07d}.pt")

        if step % int(tcfg["checkpoint_every_steps"]) == 0 or step == total:
            t3 = time.time()
            atomic_save(
                {
                    "step": step,
                    "episode": episode,
                    "updates": updates,
                    "update_counter": update_counter,
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
             "gpu_max_reserved_gb": torch.cuda.max_memory_reserved() / 2**30, "gpu_device_used_gb": device_used_gb(),
             "rss_gb": rss_gb()}
    (run_dir / "final.json").write_text(json.dumps(final, indent=1))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
