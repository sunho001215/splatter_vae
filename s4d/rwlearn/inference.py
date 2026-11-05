"""Offline inference of a trained real-world FQL policy and conversion of its output to the robot's command convention.
NO robot I/O: nothing here opens a connection, publishes, or writes a command anywhere but the returned dict / files.

``OfflinePolicy`` = frozen baseline encoder (the export the dataset was built with, sha256-checked) + the trained FQL
agent (one-step policy). ``act`` takes an observation history exactly as the training data defines it (3 frames of
one camera - or 3 synchronised multi-view steps - in the training store's 128 px centre-crop format, plus the measured
flange pose and gripper register at the newest frame) and returns ``actions.to_robot_command``: the flange goal pose
in the base frame (the ``action/flange_pose`` convention), the gripper closure and Robotiq register.
``replay`` runs the policy over a recorded episode of a dataset file and compares with the recorded commands.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from s4d.rwlearn import actions as A
from s4d.rwlearn.fql import FQLAgent
from s4d.rwlearn.fql_data import CONDITIONS, FRAME_STACK, ObsEncoder, encoder_adapter, sha256_file


class OfflinePolicy:
    def __init__(self, checkpoint: str | Path, *, device="cpu", export_path: str | Path | None = None):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)  # own checkpoint (config + RNG payload)
        self.meta = ckpt["meta"]
        self.contract = A.ActionContract.from_json(self.meta["contract"])
        self.condition = self.meta["condition"]
        export_path = export_path or self.meta["export_path"]
        if sha256_file(export_path) != self.meta["export_sha256"]:
            raise ValueError("encoder export differs from the one the policy was trained on")
        self.device = torch.device(device)
        self.adapter = encoder_adapter(self.condition, export_path, int(self.meta["image_hw"][0])).to(self.device)
        self.adapter.backbone.eval()
        feat_dim, sequence = int(self.meta["feat_dim"]), bool(self.meta["sequence_feature"])
        state = ckpt["agent"]
        self.agent = FQLAgent.create(0, state["obs_dim"], state["action_dim"], state["config"], device=self.device,
                                     encoder=lambda: ObsEncoder(feat_dim, sequence, A.PROPRIO_DIM,
                                                                int(self.meta["proj_dim"])))
        self.agent.load_state_dict(state)
        self.agent.train_mode(False)
        self.sequence = sequence

    @torch.no_grad()
    def features(self, obs: dict) -> torch.Tensor:
        """obs: ``rgb`` (T=3, V, H, W, 3) uint8 oldest first [+ ``depth`` (T, V, H, W) m, ``K`` (V, 3, 3),
        ``c2w`` (V, 4, 4)] -> the frozen feature input of the policy (1, 3, D) or (1, D)."""
        rgb = torch.as_tensor(obs["rgb"], device=self.device)
        if rgb.shape[0] != FRAME_STACK:
            raise ValueError(f"expected a {FRAME_STACK}-frame history")
        x = rgb.float().div(255).permute(0, 1, 4, 2, 3)  # (T, V, 3, H, W)
        bb = self.adapter.backbone
        if self.sequence:
            return bb(x[:, 0].unsqueeze(0)).flatten(1)
        if self.adapter.observation is not None:
            o = {"rgb": x}
            if "depth" in obs:
                o["depth"] = torch.as_tensor(obs["depth"], device=self.device).float()
            if "K" in obs:
                o["K"] = torch.as_tensor(obs["K"], device=self.device).float().expand(FRAME_STACK, -1, -1, -1)
                o["c2w"] = torch.as_tensor(obs["c2w"], device=self.device).float().expand(FRAME_STACK, -1, -1, -1)
            return bb(o).flatten(1).unsqueeze(0)
        return bb(x[:, 0]).flatten(1).unsqueeze(0)

    @torch.no_grad()
    def act_from_features(self, feat: torch.Tensor, proprio: np.ndarray, measured_pose7: np.ndarray, *,
                          noise: torch.Tensor | None = None, binary_gripper: bool = False) -> dict:
        obs = {"feat": feat.to(self.device), "proprio": torch.as_tensor(proprio, device=self.device).float()[None]}
        a = self.agent.sample_actions(obs, noise).cpu().numpy()[0]
        out = A.to_robot_command(measured_pose7, a, self.contract, binary_gripper=binary_gripper)
        out["action_norm"] = a
        return out

    def act(self, obs: dict, measured_pose7: np.ndarray, gripper_register: float, **kw) -> dict:
        T = A.pose7_to_matrix(np.asarray(measured_pose7, np.float64))
        proprio = np.concatenate([T[:3, 3], A.rotation_6d(T[:3, :3]), [A.closure_from_register(gripper_register)]])
        return self.act_from_features(self.features(obs), proprio, measured_pose7, **kw)


def replay(policy: OfflinePolicy, dataset_path: str | Path, episode: str, timeline: str, *, seed: int = 0) -> dict:
    """Policy vs recorded commands over every transition of one recorded timeline, using the dataset's cached frozen
    features (identical to ``features`` on the store frames; tested). Errors in physical units."""
    gen = torch.Generator().manual_seed(seed)
    with h5py.File(dataset_path, "r") as f:
        g = f["episodes"][episode][timeline]
        feat, run, ok = g["features"][:], g["run"][:], g["transition_ok"][:].astype(bool)
        proprio, action_phys, pose = g["proprio"][:], g["action_phys"][:], g["measured_pose"][:]
    from s4d.rwlearn.transitions import history_indices

    pos_err, rot_err, grip_err = [], [], []
    for k in np.flatnonzero(ok):
        h = [k] if policy.sequence else history_indices(run, np.array([k]), FRAME_STACK)[0]
        x = torch.from_numpy(feat[h].astype(np.float32))
        x = x if policy.sequence else x[None]
        noise = torch.randn(1, A.ACTION_DIM, generator=gen)
        out = policy.act_from_features(x, proprio[k], A.matrix_to_pose7(pose[k]), noise=noise)
        pos_err.append(float(np.linalg.norm(out["delta_phys"][:3] - action_phys[k, :3])))
        Rp = A.apply_delta(np.eye(4), out["delta_phys"])[:3, :3]
        Rr = A.apply_delta(np.eye(4), action_phys[k, :6])[:3, :3]
        rot_err.append(float(np.degrees(np.arccos(np.clip((np.trace(Rp.T @ Rr) - 1) / 2, -1, 1)))))
        grip_err.append(abs(float(out["gripper_closure"]) - float(action_phys[k, 6])))
    return {"transitions": len(pos_err), "position_error_mm_median": 1e3 * float(np.median(pos_err)),
            "rotation_error_deg_median": float(np.median(rot_err)), "gripper_abs_error_mean": float(np.mean(grip_err)),
            "contract": json.dumps(policy.contract.to_json())}


def checkpoint_payload(agent: FQLAgent, data, condition: str, step: int, **extra) -> dict:
    """What ``OfflinePolicy`` loads: the agent state + the dataset's encoder / contract metadata."""
    meta = {"condition": condition, "export_path": data.attrs["export_path"],
            "export_sha256": data.attrs["export_sha256"], "contract": json.loads(data.attrs["contract"]),
            "image_hw": json.loads(data.attrs["image_hw"]), "feat_dim": data.feat_dim,
            "sequence_feature": data.sequence, "reward": data.attrs["reward"], "dataset": data.path,
            "proj_dim": int(agent.critic.encoder.out_dim - A.PROPRIO_DIM)}
    return {"agent": agent.state_dict(), "step": int(step), "meta": meta, **extra}


def conditions() -> list[str]:
    return sorted(CONDITIONS)
