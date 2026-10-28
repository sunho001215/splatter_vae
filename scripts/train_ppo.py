"""PPO pilot on one Meta-World task with one encoder (user directive 2026-10-28; ``s4d/rl/sb3_ppo.py``).

Stable-Baselines3 v2.9.0 PPO (unchanged) on the shared reference protocol, launched and evaluated like
``scripts/train_rl.py`` / ``train_sac.py``: the companion ``scripts/eval_rl.py`` evaluates the policy snapshots written
every ``eval.every_steps`` agent steps (with their observation statistics); ``checkpoints/latest.pt`` (SB3 model zip +
VecNormalize statistics + counters) resumes the run. PPO is on-policy, so a resume loses at most the current rollout.
"""

from __future__ import annotations

import argparse
import atexit
import io
import json
import os
import pickle
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
from s4d.rl import sb3_ppo  # noqa: E402
from s4d.rl.env import env_kwargs  # noqa: E402
from s4d.rl.protocol import resolve_config  # noqa: E402
from s4d.run_identity import record_run_identity  # noqa: E402

SHORT_RUN_STEPS = 1000
ALGORITHM = "sb3_ppo"


def rss_gb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def atomic_save(payload, path: Path) -> None:
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
    if int(cfg["train"]["num_train_steps"]) > SHORT_RUN_STEPS:
        require_passed_tests()
    ckpt_dir, snap_dir = run_dir / "checkpoints", run_dir / "snapshots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    resume = latest.is_file()
    cfg.setdefault("run", {}).update(dir=str(run_dir), gpus=GPU_MAPPING, egl_device=EGL_DEVICE)
    record_run_identity(cfg, REPO, run_dir, native_diagnostic_steps=int(cfg["train"]["num_train_steps"]))
    dump_config(cfg, run_dir / "config.yaml")

    seed, tcfg, a = int(cfg["seed"]), cfg["train"], cfg["agent"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    torch.set_num_threads(int(tcfg["cpu_threads"]))
    sb3_ppo.load_sb3()
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    kwargs = env_kwargs(cfg)
    proprio_dim = len(cfg["env"]["proprio_indices"])
    n_envs = int(a["n_envs"])
    state = torch.load(latest, map_location="cpu", weights_only=False) if resume else None
    # a resume starts the workers' episode schedules afresh: offset their seeds by the resume count, so resumed training
    # never repeats the episodes (configurations, cameras) of earlier attempts (Codex review 2026-10-28)
    resumes = int(state["counters"].get("resumes", 0)) + 1 if resume else 0
    offset = resumes * sb3_ppo.SEED_STRIDE * n_envs
    fns = [sb3_ppo.make_env_fn(cfg["task"], seed + offset + sb3_ppo.SEED_STRIDE * i, kwargs, proprio_dim, cfg)
           for i in range(n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn")
    atexit.register(venv.close)
    cnn = sb3_ppo.is_cnn(cfg)
    if not cnn:
        venv = sb3_ppo.feature_vec_env(venv, sb3_ppo.FrozenFeatureExtractor(cfg, device), proprio_dim)
    norm_keys = ["proprio"] if cnn else None
    if resume:
        vecnorm = pickle.loads(state["vecnormalize"])  # VecNormalize.load without a file
        vecnorm.set_venv(venv)
        venv = vecnorm
        model = PPO.load(io.BytesIO(state["model_zip"]), env=venv, device=device)
        counters = {**state["counters"], "resumes": resumes}
        append_jsonl(run_dir / "events.jsonl", {"event": "resumed", "step": counters["num_timesteps"], "time": time.time()})
    else:
        venv = VecNormalize(venv, norm_obs=bool(a["normalize_observations"]), norm_reward=bool(a["normalize_rewards"]),
                            clip_obs=float(a["clip_obs"]), gamma=float(a["gamma"]), norm_obs_keys=norm_keys)
        model = PPO("MultiInputPolicy" if cnn else "MlpPolicy", venv, learning_rate=float(a["learning_rate"]),
                    n_steps=int(a["n_steps"]), batch_size=int(a["batch_size"]), n_epochs=int(a["n_epochs"]),
                    gamma=float(a["gamma"]), gae_lambda=float(a["gae_lambda"]), clip_range=float(a["clip_range"]),
                    ent_coef=float(a["ent_coef"]), vf_coef=float(a["vf_coef"]), max_grad_norm=float(a["max_grad_norm"]),
                    policy_kwargs=sb3_ppo.policy_kwargs(cfg), seed=seed, device=device, verbose=0)
        counters = {"num_timesteps": 0, "episodes": 0, "resumes": 0}
    total, every = int(tcfg["num_train_steps"]), int(cfg["eval"]["every_steps"])
    log_every, ckpt_every = int(tcfg["log_every_steps"]), int(tcfg["checkpoint_every_steps"])
    wandb_run = init_wandb(cfg, run_dir.name, cfg["wandb"]["project"], bool(cfg["wandb"]["enabled"]), run_dir)

    class Protocol(BaseCallback):
        """Snapshots on the evaluation grid, train.jsonl, checkpoints; episode return / success from the env infos."""

        def _on_training_start(self) -> None:
            n = self.training_env.num_envs
            self.ret, self.succ = np.zeros(n), np.zeros(n)
            window = int(tcfg["rolling_window"])
            self.roll_ret, self.roll_succ = deque(maxlen=window), deque(maxlen=window)
            done = self.num_timesteps
            self.next_snap = (done // every + 1) * every
            self.next_log = (done // log_every + 1) * log_every
            self.next_ckpt = (done // ckpt_every + 1) * ckpt_every
            self.started, self.interval_start, self.interval_steps = time.time(), time.time(), done

        def _on_step(self) -> bool:
            rewards = self.training_env.get_original_reward()  # environment return, not the normalised reward
            self.ret += rewards
            for i, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
                self.succ[i] = max(self.succ[i], info.get("success", 0.0))
                if done:
                    self.roll_ret.append(float(self.ret[i]))
                    self.roll_succ.append(float(self.succ[i]))
                    counters["episodes"] += 1
                    self.ret[i] = self.succ[i] = 0.0
            step = self.num_timesteps
            while step >= self.next_snap and self.next_snap <= total:
                policy = {"policy": self.model.policy.state_dict(),
                          "normalizer": sb3_ppo.normalizer_state(self.model.get_vec_normalize_env())}
                atomic_save({"step": self.next_snap, "algorithm": ALGORITHM, "agent_steps_taken": step,
                             "policy": policy}, snap_dir / f"step_{self.next_snap:07d}.pt")
                self.next_snap += every
            if step >= self.next_log:
                record = {"step": step, "episode": counters["episodes"],
                          "fps": (step - self.interval_steps) / max(time.time() - self.interval_start, 1e-9),
                          "rolling_success": float(np.mean(self.roll_succ)) if self.roll_succ else None,
                          "rolling_return": float(np.mean(self.roll_ret)) if self.roll_ret else None,
                          "gpu_max_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "rss_gb": rss_gb(),
                          "cpu_seconds": time.process_time(), "time": time.time(),
                          **{f"train/{k.split('/')[-1]}": float(v) for k, v in self.logger.name_to_value.items()
                             if isinstance(v, (int, float))}}
                append_jsonl(run_dir / "train.jsonl", record)
                if wandb_run is not None:
                    wandb_run.log({k: v for k, v in record.items() if v is not None and k != "time"}, step=step)
                self.next_log = (step // log_every + 1) * log_every
                self.interval_start, self.interval_steps = time.time(), step
            if step >= self.next_ckpt or step >= total:
                save_checkpoint(self.model, step)
                self.next_ckpt = (step // ckpt_every + 1) * ckpt_every
            return True

    def save_checkpoint(model, step):
        """One atomic file (Codex review 2026-10-28): the SB3 model zip (policy, optimiser, counters) and the
        VecNormalize statistics (VecNormalize.save's pickle) as bytes, with this run's counters."""
        buffer = io.BytesIO()
        model.save(buffer)
        atomic_save({"step": step, "counters": {**counters, "num_timesteps": step}, "model_zip": buffer.getvalue(),
                     "vecnormalize": pickle.dumps(model.get_vec_normalize_env())}, latest)

    model.learn(total_timesteps=total - model.num_timesteps, callback=Protocol(), reset_num_timesteps=not resume,
                progress_bar=False)
    save_checkpoint(model, model.num_timesteps)
    final = {"completed": True, "steps": total, "agent_steps_taken": int(model.num_timesteps),
             "episodes": counters["episodes"], "rss_gb": rss_gb()}
    (run_dir / "final.json").write_text(json.dumps(final, indent=1))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
