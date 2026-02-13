import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
import robosuite as suite

# Camera utils
from dataset.robosuite_demo_collect.demo_collector.camera_config import (
    CameraDef,
    apply_custom_cameras_to_env,
    _look_at_quat_wxyz,
)
from dataset.robosuite_demo_collect.demo_collector.render import render_rgb

# Policy 
from policies.utils.pose10 import pose10_from_obs
from policies.utils.dataloader import pick_obs_keys, reduce_gripper
from policies.models.vision import build_vision_encoder
from policies.models.policy import PolicyConfig, AlohaUnleashedFlowPolicy

# General utils
from utils.general_utils import (
    quat_xyzw_to_rotmat_np,
    mat_to_axisangle_np,
    rot6d_to_rotmat_np,
)

# Flow policy sampling
from policies.eval.chunked_policy_wrapper import ChunkExecCfg, ChunkedFlowPolicyWrapper
from policies.eval.flow_inference import FlowSampleCfg

# ----------------------------- yaml helpers -----------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _abs_path(maybe_rel: str, base_dir: Path) -> str:
    p = Path(maybe_rel)
    return str(p if p.is_absolute() else (base_dir / p).resolve())


def resolve_training_paths(train_cfg: Dict[str, Any], train_yaml_dir: Path) -> Dict[str, Any]:
    """
    Make common checkpoint paths absolute so eval doesn't depend on cwd.
    Especially important for splatter_vae checkpoints.
    """
    cfg = copy.deepcopy(train_cfg)

    # Example: vision.splatter_vae.checkpoint_path (adapt if your YAML uses different key)
    vis = cfg.get("vision", {})
    spl = vis.get("splatter_vae", {})
    if isinstance(spl, dict) and "checkpoint_path" in spl and spl["checkpoint_path"]:
        spl["checkpoint_path"] = _abs_path(str(spl["checkpoint_path"]), train_yaml_dir)
    vis["splatter_vae"] = spl
    cfg["vision"] = vis
    return cfg


# ----------------------------- camera setup (same as example) -----------------------------
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
    Same convention as your evaluation example. 
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
    Copy of the idea in your example: absolute target -> delta command.
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

    grip_cmd = 1.0 if tgt_grip >= 0.0 else -1.0
    act7 = np.concatenate([dpos_cmd, drot_cmd, np.array([grip_cmd], dtype=np.float64)], axis=0).astype(np.float32)

    if act7.size < action_dim:
        act = np.pad(act7, (0, action_dim - act7.size), mode="constant")
    else:
        act = act7[:action_dim]

    if action_low is not None and action_high is not None:
        act = np.clip(act, action_low, action_high)

    return act.astype(np.float32)


# ----------------------------- policy loading -----------------------------

def load_flow_policy_from_training_yaml(
    *,
    training_yaml_path: Path,
    eval_overrides: Dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    SCRIPT_DIR = Path(__file__).resolve().parent
    train_cfg = load_yaml(training_yaml_path)
    train_cfg = resolve_training_paths(train_cfg, SCRIPT_DIR)

    if eval_overrides:
        train_cfg = deep_update(train_cfg, eval_overrides)

    vision = build_vision_encoder(train_cfg)
    policy_cfg = PolicyConfig(**train_cfg["policy"])
    policy = AlohaUnleashedFlowPolicy(policy_cfg, vision).to(device)
    policy.eval()

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")

    # Your training checkpoint saves "policy_state_dict" + "config" + "step".
    state_dict = ckpt["policy_state_dict"] if isinstance(ckpt, dict) and "policy_state_dict" in ckpt else ckpt
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"[warning] loading checkpoint had missing keys: {missing}, unexpected keys: {unexpected}")
    return policy


# ----------------------------- main eval loop -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to eval YAML.")
    args = parser.parse_args()

    eval_yaml_path = Path(args.config).resolve()
    cfg = load_yaml(eval_yaml_path)

    # Resolve key paths relative to eval yaml
    SCRIPT_DIR = Path(__file__).resolve().parent
    training_yaml_path = Path(_abs_path(cfg["training"]["yaml_path"], SCRIPT_DIR))
    checkpoint_path = Path(_abs_path(cfg["checkpoint"]["path"], SCRIPT_DIR))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    # Load policy (vision encoder auto-selected from training yaml)
    policy = load_flow_policy_from_training_yaml(
        training_yaml_path=training_yaml_path,
        eval_overrides=cfg.get("model_overrides", {}),
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # Chunk wrapper (16 predicted, execute 8, replan)
    exec_cfg = ChunkExecCfg(
        pred_horizon=int(cfg["execution"].get("pred_horizon", 16)),
        exec_horizon=int(cfg["execution"].get("exec_horizon", 8)),
        action_dim=int(cfg["execution"].get("action_dim", 10)),
        proprio_dim=int(cfg["execution"].get("proprio_dim", 10)),
    )
    sample_cfg = FlowSampleCfg(
        ode_steps=int(cfg["inference"].get("ode_steps", 10)),
        method=str(cfg["inference"].get("method", "euler")),
        use_flow_matching_solver=bool(cfg["inference"].get("use_flow_matching_solver", True)),
        noise_std=float(cfg["inference"].get("noise_std", 1.0)),
    )
    wrapped = ChunkedFlowPolicyWrapper(policy, device=device, exec_cfg=exec_cfg, sample_cfg=sample_cfg)

    # Create robosuite env
    env_cfg = cfg["env"]
    env = suite.make(
        env_cfg["name"],
        robots=[env_cfg.get("robot", "Panda")],
        has_renderer=bool(env_cfg.get("render", False)),
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=float(env_cfg.get("control_freq", 10.0)),
        horizon=max(1, int(env_cfg.get("max_steps", 300))),
        hard_reset=False,
        renderer="mujoco",
    )

    # Reset once
    obs = env.reset()

    # Inject custom camera 
    cam_cfg = cfg["camera"]
    cam_name = str(cam_cfg.get("name", "eval_cam"))
    cam_def = make_single_spherical_camera(
        name=cam_name,
        r=float(cam_cfg["spherical"]["r"]),
        theta=float(cam_cfg["spherical"]["theta"]),
        phi=float(cam_cfg["spherical"]["phi"]),
        lookat=np.array(cam_cfg["spherical"]["lookat"], dtype=np.float64),
        up=np.array(cam_cfg["spherical"].get("up", [0.0, 0.0, 1.0]), dtype=np.float64),
        degrees=bool(cam_cfg["spherical"].get("degrees", True)),
        fovy=float(cam_cfg["spherical"].get("fovy", 45.0)),
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

    print(f"[info] device={device} camera={cam_name} env_action_dim={action_dim}")
    print(f"[info] pred_horizon={exec_cfg.pred_horizon} exec_horizon={exec_cfg.exec_horizon} ode_steps={sample_cfg.ode_steps}")

    H = int(cam_cfg.get("img_h", 224))
    W = int(cam_cfg.get("img_w", 224))
    dpos_scale = float(cfg["execution"].get("dpos_scale", 0.024))
    drot_scale = float(cfg["execution"].get("drot_scale", 0.212))

    trials = int(env_cfg.get("trials", 50))
    max_steps = int(env_cfg.get("max_steps", 300))
    show_window = bool(env_cfg.get("show_window", False))

    successes = 0
    max_rewards = []

    for trial in range(1, trials + 1):
        obs = env.reset()
        wrapped.reset()

        pos_k, quat_k, grip_k = pick_obs_keys(obs)
        trial_max_reward = -float("inf")
        trial_success = False

        for step in range(max_steps):
            frame = render_rgb(env.sim, cam_name, H, W)
            grip = reduce_gripper(obs[grip_k])
            state10 = pose10_from_obs(np.asarray(obs[pos_k]), np.asarray(obs[quat_k]), float(grip))

            # Build batch 
            img_t = torch.from_numpy(frame).permute(2, 0, 1).contiguous().float() / 255.0
            img_t = img_t.unsqueeze(0).to(device)
            st_t = torch.from_numpy(np.asarray(state10, dtype=np.float32)).unsqueeze(0).to(device)
            batch = {"observation.image": img_t, "observation.state": st_t}

            # Wrapper returns absolute 10D target for THIS step
            target10 = wrapped.select_action(batch)[0].detach().float().cpu().numpy()

            # Convert to env action delta (robosuite control)
            act = abs_pose10_to_env_action_delta(
                target10=target10,
                obs=obs,
                pos_k=pos_k,
                quat_k=quat_k,
                dpos_scale=dpos_scale,
                drot_scale=drot_scale,
                action_dim=action_dim,
                action_low=action_low,
                action_high=action_high,
            )

            obs, reward, done, info = env.step(act)

            trial_max_reward = max(trial_max_reward, float(reward))
            if hasattr(env, "_check_success") and env._check_success():
                trial_success = True

            if show_window:
                cv2.imshow("eval_cam", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if done:
                break

        successes += int(trial_success)
        max_rewards.append(trial_max_reward)
        print(f"[trial {trial:03d}/{trials}] success={trial_success} max_reward={trial_max_reward:.3f}")

    success_rate = float(successes) / float(max(1, trials))
    avg_max_reward = float(np.mean(max_rewards)) if max_rewards else float("nan")
    print("=======================================================")
    print(f"[result] success_rate={success_rate:.3f}  avg_max_reward={avg_max_reward:.3f}")
    print("=======================================================")


if __name__ == "__main__":
    main()
