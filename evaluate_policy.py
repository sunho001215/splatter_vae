#!/usr/bin/env python3
"""
evaluate_policy.py

Evaluate a trained LeRobot DiffusionPolicy in a robosuite environment.

Metrics (over N trials):
  1) Success rate: env._check_success()
  2) Average of (max reward per episode)
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2
import torch
import robosuite as suite

from dataset.robosuite_demo_collect.demo_collector.camera_config import (
    CameraDef,
    apply_custom_cameras_to_env,
    _look_at_quat_wxyz,
)
from dataset.robosuite_demo_collect.demo_collector.render import render_rgb
from policy_utils.pose10 import pose10_from_obs
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from utils.general_utils import (
    quat_xyzw_to_rotmat_np,
    mat_to_axisangle_np,
    rot6d_to_rotmat_np,
)

# ----------------------------- obs parsing -----------------------------

def pick_obs_keys(obs: Dict[str, Any]) -> Tuple[str, str, str]:
    """Best-effort selection of (eef_pos_key, eef_quat_key, gripper_key) from robosuite obs dict."""
    keys = list(obs.keys())

    pos_candidates = ["robot0_eef_pos", "eef_pos"]
    quat_candidates = ["robot0_eef_quat", "eef_quat"]
    grip_candidates = [
        "robot0_gripper_qpos",
        "robot0_gripper_pos",
        "robot0_gripper_state",
        "gripper_qpos",
        "gripper_pos",
        "gripper_state",
    ]

    def pick(cands, shape_pred) -> Optional[str]:
        for c in cands:
            if c in keys and shape_pred(np.asarray(obs[c]).shape):
                return c
        for k in keys:
            if any(tok in k for tok in cands) and shape_pred(np.asarray(obs[k]).shape):
                return k
        return None

    pos_k = pick(pos_candidates, lambda s: (len(s) >= 1 and s[-1] == 3))
    quat_k = pick(quat_candidates, lambda s: (len(s) >= 1 and s[-1] == 4))
    grip_k = pick(grip_candidates, lambda s: True)

    if pos_k is None or quat_k is None or grip_k is None:
        raise RuntimeError(
            "Could not locate required keys in robosuite obs dict.\n"
            f"Found keys: {keys}\n"
            f"Picked: pos={pos_k}, quat={quat_k}, grip={grip_k}"
        )
    return pos_k, quat_k, grip_k


def reduce_gripper(grip_raw: Any) -> float:
    """Convert robosuite gripper observation to a scalar (mean if vector)."""
    arr = np.asarray(grip_raw, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    return float(np.mean(arr.reshape(-1)))


# ----------------------------- camera setup -----------------------------

def make_single_spherical_camera(
    *,
    name: str,
    r: float,
    theta: float,
    phi: float,
    lookat: np.ndarray,
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    degrees: bool = True,
    fovy: float = 45.0,
) -> CameraDef:
    """
    Create exactly one MJCF camera around lookat, same convention as your collection code:
      - phi   = azimuth around +Z (0 along +X, +90 along +Y)
      - theta = elevation from XY plane (0 on plane, +90 straight up)
    """
    import math

    th = float(theta)
    ph = float(phi)
    if degrees:
        th = math.radians(th)
        ph = math.radians(ph)

    x = float(r) * math.cos(th) * math.cos(ph)
    y = float(r) * math.cos(th) * math.sin(ph)
    z = float(r) * math.sin(th)

    pos = np.asarray(lookat, dtype=np.float64).reshape(3) + np.array([x, y, z], dtype=np.float64)
    quat_wxyz = _look_at_quat_wxyz(pos, lookat, up=up)

    return CameraDef(
        name=str(name),
        pos=pos.tolist(),
        quat=np.asarray(quat_wxyz, dtype=np.float64).tolist(),  # MuJoCo expects wxyz
        fovy=float(fovy),
    )


# ----------------------------- policy helpers -----------------------------

def _get_cfg_value(policy: DiffusionPolicy, key: str, default: Any) -> Any:
    """DiffusionPolicy versions differ: try common config attribute names."""
    for attr in ("cfg", "config", "policy_config"):
        if hasattr(policy, attr):
            obj = getattr(policy, attr)
            if hasattr(obj, key):
                return getattr(obj, key)
    return default


def build_policy_batch(
    img_hist: deque[np.ndarray],
    state_hist: deque[np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Build the policy input from last n_obs_steps frames/states.
    Shapes we keep around:
      imgs: (n_obs, H, W, 3) uint8
      -> tensor (n_obs, 3, H, W) float in [0,1]
      states: (n_obs, 10) float
    """
    imgs = np.stack(list(img_hist), axis=0)  # (n_obs,H,W,3)
    img_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().float() / 255.0  # (n_obs,3,H,W)
    st = np.stack(list(state_hist), axis=0)  # (n_obs,10)
    st_t = torch.from_numpy(st).contiguous().float()  # (n_obs,10)

    # Put on device (no batch dim yet; we’ll try both layouts below)
    img_t = img_t.to(device)
    st_t = st_t.to(device)

    return {"_img_seq": img_t, "_state_seq": st_t}  # internal keys


@torch.no_grad()
def select_action10_step(policy: DiffusionPolicy, img: np.ndarray, state10: np.ndarray, device: torch.device) -> np.ndarray:
    """
    LeRobot 0.3.2: call select_action() with ONE observation.
    Policy internally:
      - queues the last n_obs_steps observations
      - samples an action chunk when needed
      - returns the next action from the cached chunk (replans every n_action_steps)
    """
    # img: (H,W,3) uint8
    img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0  # (3,H,W)
    img_t = img_t.unsqueeze(0).to(device)  # (1,3,H,W)

    st_t = torch.from_numpy(np.asarray(state10, dtype=np.float32)).unsqueeze(0).to(device)  # (1,10)

    batch = {
        "observation.image": img_t,
        "observation.state": st_t,
    }

    out = policy.select_action(batch)
    out = out.detach()

    return out[0].float().cpu().numpy()

def abs_pose10_to_env_action_delta(
    *,
    target10: np.ndarray,  # (10,)
    obs: Dict[str, Any],
    pos_k: str,
    quat_k: str,
    dpos_scale: float,
    drot_scale: float,
    action_dim: int,
    action_low: Optional[np.ndarray],
    action_high: Optional[np.ndarray],
) -> np.ndarray:
    """
    Convert absolute 10D pose target -> normalized delta command for env.step().

    Robosuite OSC-like action convention (common):
      [dpos(3), drot_axisangle(3), gripper(1)] in [-1,1] (approximately)
    """
    target10 = np.asarray(target10, dtype=np.float64).reshape(10)

    tgt_pos = target10[0:3]
    tgt_R = rot6d_to_rotmat_np(target10[3:9])
    tgt_grip = float(target10[9])

    cur_pos = np.asarray(obs[pos_k], dtype=np.float64).reshape(3)
    cur_quat = np.asarray(obs[quat_k], dtype=np.float64).reshape(4)  # xyzw
    cur_R = quat_xyzw_to_rotmat_np(cur_quat)

    dpos = tgt_pos - cur_pos
    dR = tgt_R @ cur_R.T
    drot = mat_to_axisangle_np(dR)  # axis*angle, (3,)

    dpos_cmd = np.clip(dpos / float(dpos_scale), -1.0, 1.0)
    drot_cmd = np.clip(drot / float(drot_scale), -1.0, 1.0)

    # Map predicted gripper scalar -> open/close command in {+1,-1}
    grip_cmd = float(np.clip(np.sign(tgt_grip), -1.0, 1.0))

    act7 = np.concatenate([dpos_cmd, drot_cmd, np.array([grip_cmd], dtype=np.float64)], axis=0).astype(np.float32)

    # Fit to env action dimension
    if act7.size < action_dim:
        act = np.pad(act7, (0, action_dim - act7.size), mode="constant")
    else:
        act = act7[:action_dim]

    # Clip to env bounds if available
    if action_low is not None and action_high is not None:
        act = np.clip(act, action_low, action_high)

    return act.astype(np.float32)


# ----------------------------- main eval loop -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    # Env / robot
    ap.add_argument("--env", type=str, required=True, choices=["Stack", "PickPlace", "NutAssembly", "Wipe"])
    ap.add_argument("--robot", type=str, default="Panda")

    # Policy checkpoint
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--dataset_stats", type=str, default="", help="default: <ckpt_dir>/dataset_stats.pt")
    ap.add_argument("--device", type=str, default="cuda:0")

    # Trials
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)

    # Rendering / camera input
    ap.add_argument("--img_h", type=int, default=224)
    ap.add_argument("--img_w", type=int, default=224)
    ap.add_argument("--render", action="store_true")

    # Single spherical camera parameters
    ap.add_argument("--cam_r", type=float, required=True)
    ap.add_argument("--cam_theta", type=float, required=True)
    ap.add_argument("--cam_phi", type=float, required=True)
    ap.add_argument("--cam_fovy", type=float, default=45.0)
    ap.add_argument("--cam_lookat", type=float, nargs=3, required=True)
    ap.add_argument("--cam_angles_in_rad", action="store_true")

    # Execution scaling (absolute pose -> normalized delta)
    ap.add_argument("--dpos_scale", type=float, default=0.1)
    ap.add_argument("--drot_scale", type=float, default=0.7)

    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if (torch.cuda.is_available() and "cpu" not in args.device) else "cpu")
    print(f"[info] device={device}")

    ckpt_dir = Path(args.ckpt_dir)
    stats_path = Path(args.dataset_stats) if args.dataset_stats else (ckpt_dir / "dataset_stats.pt")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing dataset_stats.pt at: {stats_path}")
    dataset_stats = torch.load(stats_path, map_location="cpu")

    # Load policy
    policy = DiffusionPolicy.from_pretrained(str(ckpt_dir), dataset_stats=dataset_stats).to(device)
    policy.eval()

    # Make evaluation deterministic if the encoder does random crop
    policy.config.crop_is_random = False

    # Read planning params from config
    n_obs_steps = int(_get_cfg_value(policy, "n_obs_steps", 2))
    horizon = int(_get_cfg_value(policy, "horizon", 16))
    n_action_steps = int(_get_cfg_value(policy, "n_action_steps", min(8, horizon)))
    print(f"[info] policy n_obs_steps={n_obs_steps} horizon={horizon} n_action_steps={n_action_steps}")

    # Create robosuite env
    env = suite.make(
        args.env,
        robots=[args.robot],
        has_renderer=bool(args.render),
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=10.0,
        horizon=max(1, int(args.max_steps)),
        hard_reset=False,
        renderer="mujoco",
    )

    # Reset once to access sim center
    obs = env.reset()

    # Inject ONE custom input camera
    cam_name = "eval_cam"
    cam_def = make_single_spherical_camera(
        name=cam_name,
        r=float(args.cam_r),
        theta=float(args.cam_theta),
        phi=float(args.cam_phi),
        lookat=np.array(args.cam_lookat, dtype=np.float64),
        up=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        degrees=(not args.cam_angles_in_rad),
        fovy=float(args.cam_fovy),
    )
    apply_custom_cameras_to_env(env, [cam_def])
    obs = env.reset()

    # Env action limits
    action_low, action_high = None, None
    action_dim = getattr(env, "action_dim", None)
    if action_dim is None and hasattr(env, "action_spec"):
        lo, hi = env.action_spec
        action_dim = int(np.asarray(lo).size)
        action_low = np.asarray(lo, dtype=np.float32).reshape(-1)
        action_high = np.asarray(hi, dtype=np.float32).reshape(-1)
    else:
        action_dim = int(action_dim) if action_dim is not None else 7
    print(f"[info] env action_dim={action_dim} (camera={cam_name})")

    # -------------------- evaluation --------------------
    successes = 0
    max_rewards: list[float] = []

    H, W = int(args.img_h), int(args.img_w)

    for trial in range(1, int(args.trials) + 1):
        obs = env.reset()
        policy.reset()

        pos_k, quat_k, grip_k = pick_obs_keys(obs)

        trial_max_reward = -float("inf")
        trial_success = False

        for step in range(int(args.max_steps)):
            # Current observation -> image + 10D state
            frame = render_rgb(env.sim, cam_name, H, W)
            grip = reduce_gripper(obs[grip_k])
            state10 = pose10_from_obs(np.asarray(obs[pos_k]), np.asarray(obs[quat_k]), grip)

            # Policy step (absolute 10D target)
            target10 = select_action10_step(policy, frame, state10, device)

            # Convert absolute pose target -> normalized delta action for robosuite
            act = abs_pose10_to_env_action_delta(
                target10=target10,
                obs=obs,
                pos_k=pos_k,
                quat_k=quat_k,
                dpos_scale=float(args.dpos_scale),
                drot_scale=float(args.drot_scale),
                action_dim=action_dim,
                action_low=action_low,
                action_high=action_high,
            )

            obs, reward, done, info = env.step(act)
            trial_max_reward = max(trial_max_reward, float(reward))

            # Visualize
            disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Robosuite Evaluation", disp)
            cv2.waitKey(1)

            if env._check_success():
                trial_success = True
                break
            if done:
                break

        successes += int(trial_success)
        max_rewards.append(trial_max_reward)

        print(
            f"[trial {trial:03d}/{args.trials}] "
            f"success={trial_success} max_reward={trial_max_reward:.4f}"
        )

    success_rate = float(successes) / float(args.trials)
    avg_max_reward = float(np.mean(np.asarray(max_rewards, dtype=np.float64))) if max_rewards else float("nan")

    print("\n==================== RESULTS ====================")
    print(f"env: {args.env}  robot: {args.robot}")
    print(f"trials: {args.trials}")
    print(f"camera: r={args.cam_r}, theta={args.cam_theta}, phi={args.cam_phi}, fovy={args.cam_fovy}")
    print(f"policy: n_obs_steps={n_obs_steps} horizon={horizon} n_action_steps={n_action_steps}")
    print(f"success_rate: {success_rate:.3f} ({successes}/{args.trials})")
    print(f"avg_max_reward: {avg_max_reward:.4f}")
    print("=================================================\n")

    env.close()


if __name__ == "__main__":
    main()
