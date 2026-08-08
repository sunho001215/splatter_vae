from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import mujoco_mig_setup  # noqa: F401
except Exception:
    pass

import numpy as np
import torch
import yaml

from agents.drqv2.drqv2_metaworld import DrQv2MetaWorldAgent
from agents.drqv2.train_drqv2_metaworld import MetaWorldSingleCameraEnv, obs_to_policy_input, set_seed
from visualize.metaworld_camera_utils import camera_from_pose, load_yaml, lookat_up_from_drq, trajectory_poses, write_mp4


class MetaWorldCameraTrajectoryEnv(MetaWorldSingleCameraEnv):
    def __init__(self, cfg: Dict[str, Any], seed: int, poses, lookat: np.ndarray, loop: bool = True) -> None:
        self.trajectory_poses = list(poses)
        if not self.trajectory_poses:
            raise ValueError("At least one camera pose is required.")
        self.trajectory_lookat = np.asarray(lookat, dtype=np.float64)
        self.trajectory_loop = bool(loop)
        self.camera_sequence_step = 0
        super().__init__(cfg, seed=seed)

    def reset(self, camera_index=None):
        self.camera_sequence_step = 0
        return super().reset(camera_index=camera_index)

    def _trajectory_index(self) -> int:
        if self.trajectory_loop:
            return int(self.camera_sequence_step % len(self.trajectory_poses))
        return int(min(self.camera_sequence_step, len(self.trajectory_poses) - 1))

    def get_current_camera_name(self) -> str:
        return self.trajectory_poses[self._trajectory_index()].name

    def _render(self) -> np.ndarray:
        idx = self._trajectory_index()
        pose = self.trajectory_poses[idx]
        self.current_camera_index = idx
        self._renderer.update_scene(self._data, camera=camera_from_pose(pose, self.trajectory_lookat))
        img = np.asarray(self._renderer.render(), dtype=np.uint8)
        self.camera_sequence_step += 1
        frame_chw = np.transpose(img, (2, 0, 1)).copy()
        self._show_frame_cv2(frame_chw)
        return frame_chw


def load_policy_checkpoint(agent: DrQv2MetaWorldAgent, ckpt_path: str | Path, device: torch.device) -> int:
    payload = torch.load(str(ckpt_path), map_location=device)
    step = 0
    if isinstance(payload, dict) and "step" in payload:
        step = int(payload["step"])
    if isinstance(payload, dict) and "agent" in payload:
        payload = payload["agent"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported policy checkpoint payload in {ckpt_path}: {type(payload)!r}")
    required = ("encoder", "actor", "critic", "critic_target")
    if all(key in payload for key in required):
        agent.encoder.load_state_dict(payload["encoder"])
        agent.actor.load_state_dict(payload["actor"])
        agent.critic.load_state_dict(payload["critic"])
        agent.critic_target.load_state_dict(payload["critic_target"])
        for name, opt in (("actor_opt", agent.actor_opt), ("critic_opt", agent.critic_opt), ("encoder_opt", agent.encoder_opt)):
            if opt is None or name not in payload:
                continue
            try:
                opt.load_state_dict(payload[name])
            except ValueError as exc:
                print(f"[warn] Skipping incompatible optimizer state {name} from {ckpt_path}: {exc}")
    else:
        agent.load_state_dict(payload)
    agent.train(False)
    return step


def chw_to_hwc(frame: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(frame, dtype=np.uint8), (1, 2, 0)).copy()


def evaluate_policy_on_trajectory(
    *,
    name: str,
    cfg_path: str,
    ckpt_path: str,
    trajectory_cfg: Dict[str, Any],
    trajectory_name: str,
    poses,
    lookat: np.ndarray,
    args,
) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    cfg.setdefault("vision", {}).setdefault("img_height", int(cfg["env"]["image_height"]))
    cfg["vision"].setdefault("img_width", int(cfg["env"]["image_width"]))
    cfg["device"] = args.device
    cfg.setdefault("visualization", {})["enabled"] = False

    seed = int(args.seed)
    env = MetaWorldCameraTrajectoryEnv(cfg, seed=seed, poses=poses, lookat=lookat, loop=bool(args.loop_trajectory))
    device = torch.device(args.device)
    agent = DrQv2MetaWorldAgent(cfg, env.action_shape, env.proprio_shape, device)
    ckpt_step = load_policy_checkpoint(agent, ckpt_path, device)
    eval_step = int(args.policy_step) if args.policy_step is not None else ckpt_step

    returns: List[float] = []
    successes: List[float] = []
    video_paths: List[str] = []
    out_dir = Path(args.out_dir) / trajectory_name / name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for ep in range(int(args.num_episodes)):
            obs_pixels, proprio = env.reset()
            obs_for_policy = obs_to_policy_input(agent, obs_pixels)
            done = False
            ep_ret = 0.0
            ep_succ = 0.0
            frames = [chw_to_hwc(env.get_last_frame())] if ep < int(args.video_episodes) else []
            while not done:
                action = agent.act(obs_for_policy, proprio, step=eval_step, eval_mode=True)
                next_pixels, next_proprio, reward, done, info = env.step(action)
                obs_for_policy = obs_to_policy_input(agent, next_pixels)
                proprio = next_proprio
                ep_ret += float(reward)
                ep_succ = max(ep_succ, float(info.get("success", 0.0)))
                if ep < int(args.video_episodes):
                    frames.append(chw_to_hwc(env.get_last_frame()))
            returns.append(ep_ret)
            successes.append(ep_succ)
            if frames:
                video_path = out_dir / f"episode_{ep:03d}.mp4"
                write_mp4(video_path, frames, fps=float(args.fps))
                video_paths.append(str(video_path))
    finally:
        env.close()

    return {
        "policy": name,
        "config": cfg_path,
        "checkpoint": ckpt_path,
        "checkpoint_step": ckpt_step,
        "eval_step": eval_step,
        "trajectory": trajectory_name,
        "num_episodes": int(args.num_episodes),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns)) if returns else 0.0,
        "episode_successes": successes,
        "episode_returns": returns,
        "videos": video_paths,
    }


def write_summary(out_dir: Path, results: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps({"results": results}, indent=2))
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["trajectory", "policy", "success_rate", "return_mean", "return_std", "num_episodes", "checkpoint_step"])
        writer.writeheader()
        for item in results:
            writer.writerow({key: item[key] for key in writer.fieldnames})
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DrQ-v2 policies under disturbed lateral/circular camera trajectories.")
    parser.add_argument("--trajectory_config", required=True, help="DrQ-v2 config providing env.cameras/lookat/up for the disturbed camera path.")
    parser.add_argument("--policy", action="append", nargs=3, metavar=("NAME", "CONFIG", "CKPT"), required=True, help="Policy spec. Repeat as: --policy cnn config.yaml step.pt")
    parser.add_argument("--trajectory", choices=["lateral", "circular", "both"], default="both")
    parser.add_argument("--base_camera", default="cam1")
    parser.add_argument("--num_frames", type=int, default=72)
    parser.add_argument("--lateral_amplitude", type=float, default=0.12)
    parser.add_argument("--circular_azimuth_deg", type=float, default=10.0)
    parser.add_argument("--circular_elevation_deg", type=float, default=6.0)
    parser.add_argument("--loop_trajectory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--video_episodes", type=int, default=None, help="Number of episodes to save as videos. Defaults to --num_episodes.")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--out_dir", default="outputs/policy_camera_trajectory_eval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy_step", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.video_episodes is None:
        args.video_episodes = int(args.num_episodes)
    set_seed(int(args.seed))
    trajectory_cfg = load_yaml(args.trajectory_config)
    lookat, _up = lookat_up_from_drq(trajectory_cfg)
    trajectories = ["lateral", "circular"] if args.trajectory == "both" else [args.trajectory]

    results: List[Dict[str, Any]] = []
    for trajectory_name in trajectories:
        poses = trajectory_poses(
            trajectory_cfg,
            base_camera=args.base_camera,
            trajectory=trajectory_name,
            num_frames=int(args.num_frames),
            lateral_amplitude=float(args.lateral_amplitude),
            circular_azimuth_deg=float(args.circular_azimuth_deg),
            circular_elevation_deg=float(args.circular_elevation_deg),
        )
        for name, cfg_path, ckpt_path in args.policy:
            print(f"[eval] trajectory={trajectory_name} policy={name}")
            result = evaluate_policy_on_trajectory(
                name=name,
                cfg_path=cfg_path,
                ckpt_path=ckpt_path,
                trajectory_cfg=trajectory_cfg,
                trajectory_name=trajectory_name,
                poses=poses,
                lookat=lookat,
                args=args,
            )
            results.append(result)
            print(
                f"[done] {trajectory_name}/{name}: success={result['success_rate']:.3f}, "
                f"return={result['return_mean']:.2f}"
            )

    write_summary(Path(args.out_dir), results)


if __name__ == "__main__":
    main()
