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
from policies.utils.pose10 import pose10_from_obs
from policies.utils.dataset_hdf5 import pick_obs_keys, reduce_gripper
from policies.utils.config_loader import load_diffusion_config_json
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from utils.general_utils import (
    quat_xyzw_to_rotmat_np,
    mat_to_axisangle_np,
    rot6d_to_rotmat_np,
    read_json
)

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
    grip_cmd = 1.0 if tgt_grip >= 0.0 else -1.0

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

# ----------------------------- policy loading with custom encoder support -----------------------------

def _load_policy_with_optional_custom_encoder(
    *,
    ckpt_dir: Path,
    dataset_stats: Dict[str, Any],
    device: torch.device,
    vision_encoder_choice: str,   # "auto" | "standard" | "splatter_vae"
    config_json: str,             # "" -> default to <ckpt_dir>/config.json
) -> DiffusionPolicy:
    """
    Supports evaluating:
      - standard/original vision encoder (resnet...)
      - custom vision encoder (splatter_vae)

    vision_encoder_choice:
      - "auto": follow config.json's "vision_backbone"
      - "standard": force no custom backbone
      - "splatter_vae": force splatter custom backbone
    """
    ckpt_dir = Path(ckpt_dir)

    # Resolve config.json path (needed to build custom backbone)
    cfg_path = Path(config_json) if config_json else (ckpt_dir / "config.json")
    if not cfg_path.exists():
        if vision_encoder_choice in ("splatter_vae", "auto"):
            raise FileNotFoundError(
                f"Missing config.json needed for '{vision_encoder_choice}' at: {cfg_path}"
            )
        # standard-only can load without config.json
        return DiffusionPolicy.from_pretrained(str(ckpt_dir), dataset_stats=dataset_stats).to(device)

    # Load cfg + raw dict (same pattern as training)
    cfg, _ = load_diffusion_config_json(str(cfg_path))
    cfg_raw = read_json(cfg_path)

    # Decide which encoder to use
    vb = str(cfg_raw.get("vision_backbone", "")).lower()
    if vision_encoder_choice == "auto":
        vision_encoder_choice = "splatter_vae" if vb == "splatter_vae" else "standard"

    custom_backbone = None
    if vision_encoder_choice == "splatter_vae":
        from policies.encoders.splatter_encoder import make_splatter_custom_backbone

        custom_backbone = make_splatter_custom_backbone(
            dp_config=cfg,
            cfg_raw=cfg_raw,
            base_dir=cfg_path.parent,
            device=device,
        )
        print("[info] eval: using SplatterVAE custom backbone")
    else:
        print("[info] eval: using standard/original vision backbone")

    # Preferred: from_pretrained that accepts custom backbone (if supported by your LeRobot version)
    try:
        policy = DiffusionPolicy.from_pretrained(
            str(ckpt_dir),
            dataset_stats=dataset_stats,
            custom_vision_backbone=custom_backbone,
        ).to(device)
        return policy
    except TypeError:
        # Fallback: instantiate from cfg, then load weights manually
        pass

    policy = DiffusionPolicy(
        cfg,
        custom_vision_backbone=custom_backbone,
        dataset_stats=dataset_stats,
    ).to(device)

    # Manual weight load (common HF-style filenames)
    weight_candidates = [
        ckpt_dir / "model.safetensors",
        ckpt_dir / "pytorch_model.bin",
        ckpt_dir / "model.pt",
        ckpt_dir / "pytorch_model.pt",
    ]
    weight_path = next((p for p in weight_candidates if p.exists()), None)
    if weight_path is None:
        raise FileNotFoundError(
            f"Could not find model weights in {ckpt_dir}. "
            f"Tried: {[p.name for p in weight_candidates]}"
        )

    if weight_path.suffix == ".safetensors":
        from safetensors.torch import load_file as safetensors_load

        state_dict = safetensors_load(str(weight_path))
    else:
        state_dict = torch.load(weight_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys while loading (showing up to 20): {missing[:20]}")
    if unexpected:
        print(f"[warn] unexpected keys while loading (showing up to 20): {unexpected[:20]}")

    return policy


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
    ap.add_argument("--dpos_scale", type=float, default=0.024)
    ap.add_argument("--drot_scale", type=float, default=0.212)

    # add in main() argparse section (near policy checkpoint args)
    ap.add_argument(
        "--vision_encoder",
        type=str,
        default="auto",
        choices=["auto", "standard", "splatter_vae"],
        help="Choose vision encoder for eval. auto=follow config.json vision_backbone.",
    )
    ap.add_argument(
        "--policy_config_json",
        type=str,
        default="",
        help="Path to DiffusionPolicy config.json (default: <ckpt_dir>/config.json). "
            "Required for splatter_vae/auto if ckpt_dir lacks config.json.",
    )

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

    # Load policy (standard vs custom vision encoder)
    policy = _load_policy_with_optional_custom_encoder(
        ckpt_dir=ckpt_dir,
        dataset_stats=dataset_stats,
        device=device,
        vision_encoder_choice=args.vision_encoder,
        config_json=args.policy_config_json,
    )
    policy.eval()

    # Make evaluation deterministic if the encoder does random crop
    if hasattr(policy, "config") and hasattr(policy.config, "crop_is_random"):
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
