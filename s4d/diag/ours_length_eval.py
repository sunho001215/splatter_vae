"""Fixed evaluation protocol for drawer-open length checkpoints; labels are grasp-site proxies, never object truth."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from s4d.diag.probes import fit_and_score, r2_score
from s4d.geometry import project


def grasp_anchor(robot):
    """Largest observed increase in recorded closure; midpoint of FK gripper surface at that frame.

    The compact-store probe column is closure (0 open), NOT opening width.
    Release is the first subsequent decrease past the midpoint of the closing edge.
    """
    state = np.asarray(robot["probe_state"])
    width = state[:, 6]  # closure, 0=open, 1=closed (legacy URDF map)
    valid = (
        np.isfinite(width)
        & np.isfinite(robot["gripper_points"]).all(axis=(1, 2))
        & np.asarray(robot.get("ok", np.ones(len(width))), bool)
    )
    edges = np.where(valid[:-1] & valid[1:], -np.diff(width), np.inf)
    grasp = int(np.argmin(edges)) + 1
    if not np.isfinite(edges[grasp - 1]) or edges[grasp - 1] >= -1e-5:
        raise ValueError("episode has no valid observed gripper closing edge; grasp annotation required")
    threshold = (width[grasp - 1] + width[grasp]) / 2
    later = np.flatnonzero(valid & (np.arange(len(width)) > grasp) & (width < threshold))
    release = int(later[0]) if len(later) else len(width)
    points = np.asarray(robot["gripper_points"][grasp])
    # The stored FK cloud includes the palm. Its most distal fifth approximates the finger contact region.
    distances = np.linalg.norm(points - state[grasp, :3], axis=-1)
    distal = points[distances >= np.quantile(distances, 0.8)]
    return dict(
        xyz=distal.mean(0).tolist(),
        grasp=grasp,
        release=release,
        frames=len(width),
        definition="largest closure increase / distal fifth of FK surface by flange distance; fixed initial grasp site",
    )


def handle_roi(batch, anchors, radius_m=0.035):
    """Project a fixed 35 mm radius around the demo's initial handle grasp site to each calibrated camera."""
    device = batch["K"].device
    xyz = torch.tensor([anchors[e]["xyz"] for e in batch["meta"]["episode"]], device=device, dtype=torch.float32)
    V = batch["K"].shape[1]
    uv, z = project(xyz[:, None, None].expand(-1, V, 1, -1), batch["K"], batch["w2c"])
    H, W = batch["images"].shape[-2:]
    y, x = torch.meshgrid(torch.arange(H, device=device) + 0.5, torch.arange(W, device=device) + 0.5, indexing="ij")
    rx = radius_m * batch["K"][..., 0, 0] / z[..., 0].clamp_min(1e-6)
    ry = radius_m * batch["K"][..., 1, 1] / z[..., 0].clamp_min(1e-6)
    mask = ((x - uv[..., 0, 0, None, None]) / rx[..., None, None]) ** 2 + (
        (y - uv[..., 0, 1, None, None]) / ry[..., None, None]
    ) ** 2 <= 1
    mask &= z[..., 0, None, None] > 0
    return mask[:, None, :, None].expand(-1, batch["images"].shape[1], -1, -1, -1, -1)


def probe_labels(batch, anchors):
    state = batch["probe_state"].cpu().float()
    center = torch.tensor([anchors[e]["xyz"] for e in batch["meta"]["episode"]])
    t = torch.tensor([row[1] for row in batch["meta"]["t_indices"]])
    phase = torch.tensor(
        [
            0 if ti < anchors[e]["grasp"] else 1 if ti < anchors[e]["release"] else 2
            for e, ti in zip(batch["meta"]["episode"], t)
        ]
    )
    return {
        "ee_initial_handle_relative_m": state[:, 1, :3] - center,
        "gripper_open_fraction": 1 - state[:, 1, 3:4],
        "phase_proxy": F.one_hot(phase, 3).float(),
        # Only the past and present are encoded for this probe; frame 2 is an unseen future target.
        "future_ee_displacement_m": state[:, 2, :3] - state[:, 1, :3],
    }


def track_errors(batch, out):
    """Sample rendered world flow on valid moving TrackEverything points and measure world and image flow EPE."""
    src, disp = batch["track_src"].float(), batch["track_disp"].float()
    B, P, V, N = src.shape[:4]
    K = batch["K"][:, None].expand(-1, P, -1, -1, -1)
    w2c = batch["w2c"][:, None].expand(-1, P, -1, -1, -1)
    uv, z = project(src, K, w2c)
    H, W = out["pred_disp"].shape[-2:]
    grid = (uv / uv.new_tensor([W, H]) * 2 - 1).reshape(B * P * V, N, 1, 2)
    pred = (
        F.grid_sample(out["pred_disp"].float().reshape(B * P * V, 3, H, W), grid, align_corners=False)
        .reshape(B, P, V, 3, N)
        .transpose(-1, -2)
    )
    valid = (
        (batch["track_weight"] > 0)
        & (disp.norm(dim=-1) > 0.005)
        & (z > 0)
        & (uv >= 0).all(-1)
        & (uv < uv.new_tensor([W, H])).all(-1)
    )
    pred_uv, _ = project(src + pred, K, w2c)
    target_uv, tz = project(src + disp, K, w2c)
    valid &= tz > 0
    return {
        "dynamic_track_epe_m": (pred - disp).norm(dim=-1)[valid],
        "dynamic_flow_epe_px": (pred_uv - target_uv).norm(dim=-1)[valid],
    }


class Means:
    def __init__(self):
        self.sums, self.counts = defaultdict(float), defaultdict(int)

    def add(self, key, values):
        values = torch.as_tensor(values).detach().double()
        values = values[torch.isfinite(values)]
        self.sums[key] += float(values.sum())
        self.counts[key] += values.numel()

    def result(self):
        return {k: {"value": self.sums[k] / n if n else None, "count": n} for k, n in self.counts.items()}


def score_probes(train, val, seed=0):
    """Train-only standardisation, fixed ridge and fixed 64-hidden-unit MLP; no validation tuning."""
    x, targets = train
    vx, vy = val
    result = fit_and_score(x, targets, {"validation": (vx, vy)})
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    x, vx = (x - mean) / std, (vx - mean) / std
    for name, y in targets.items():
        torch.manual_seed(seed)
        ym, ys = y.mean(0), y.std(0).clamp_min(1e-6)
        net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, y.shape[1]))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        for _ in range(200):
            opt.zero_grad()
            loss = F.mse_loss(net(x), (y - ym) / ys)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = net(vx) * ys + ym
            result[f"mlp_r2_{name}_validation"] = r2_score(pred, vy[name])
            result[f"mlp_rmse_{name}_validation"] = float((pred - vy[name]).square().mean().sqrt())
            if name == "phase_proxy":
                result["mlp_phase_accuracy_validation"] = float((pred.argmax(-1) == vy[name].argmax(-1)).float().mean())
    return result
