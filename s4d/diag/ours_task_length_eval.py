"""Length-checkpoint diagnostics; regions and grasp labels are measured proxies, never object truth."""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

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


def collect_grasp_anchors(loaders):
    """Use the same task/episode identity as window metadata, rather than a store locator.

    A missing closing edge invalidates only grasp-dependent labels, not the evaluation.
    """
    anchors = {}
    for loader in loaders:
        for ep in loader.dataset.episodes:
            ef = loader.dataset.store.episode(ep)
            key = f"{ef.task}/{ef.episode_id}"
            if key not in anchors:
                try:
                    anchors[key] = grasp_anchor(ef.robot(ef.reference_camera))
                except ValueError as exc:
                    anchors[key] = {"value": None, "reason": str(exc)}
    return anchors


def batch_anchors(batch, anchors):
    meta = batch["meta"]
    tasks = meta.get("task_name", meta.get("task", [None] * len(meta["episode"])))
    return [anchors.get(f"{task}/{ep}", anchors.get(ep)) for task, ep in zip(tasks, meta["episode"])]


def detail_region(cfg):
    """Retain the legacy drawer grasp ROI; otherwise use the task's measured dynamic workspace.

    SCENE_PARAMETERS contains rig look-at / global bounds, not small-detail annotations.
    Neither those nor a dynamic-to-scene fallback establishes a task-relevant region.
    The one-standard-deviation box is a dynamic-workspace proxy, not a semantic handle.
    """
    task = cfg.get("data", {}).get("task")
    if task == "drawer-open":
        return {"kind": "initial_grasp", "task": task, "radius_m": 0.035,
                "definition": "fixed initial grasp site; 35 mm projected radius"}
    path = cfg.get("model", {}).get("anchor_stats")
    reason = "no measured task-relevant dynamic workspace in model.anchor_stats"
    if path:
        stats = json.loads(Path(path).read_text())
        # A different task's statistics must never silently supply this task's region.
        source = str(stats.get("source", ""))
        if task and task not in Path(source).parts:
            reason = f"workspace statistics source does not identify task {task!r}"
        elif not stats.get("dynamic_fallback_to_scene", True):
            dynamic = stats.get("dynamic", {})
            mean, std = np.asarray(dynamic.get("mean", [])), np.asarray(dynamic.get("std", []))
            if (mean.shape == std.shape == (3,) and np.isfinite(mean).all() and np.isfinite(std).all()
                    and (std > 0).all() and dynamic.get("num_points", 0) > 0):
                return {"kind": "dynamic_workspace", "task": task, "mean": mean.tolist(), "std": std.tolist(),
                        "source": str(path), "definition": "projected dynamic mean +/- one std box; workspace proxy"}
    return {"kind": None, "task": task, "value": None, "reason": reason}


def detail_roi(batch, anchors, region):
    if region["kind"] == "initial_grasp":
        return handle_roi(batch, anchors, region["radius_m"])
    if region["kind"] is None:
        return None
    device = batch["K"].device
    mean = torch.tensor(region["mean"], device=device, dtype=torch.float32)
    std = torch.tensor(region["std"], device=device, dtype=torch.float32)
    signs = torch.tensor(list(itertools.product((-1, 1), repeat=3)), device=device)
    B, T, V, _, H, W = batch["images"].shape
    corners = (mean + std * signs).expand(B, V, -1, -1)
    uv, z = project(corners, batch["K"], batch["w2c"])
    lo, hi = uv.amin(-2), uv.amax(-2)
    y, x = torch.meshgrid(torch.arange(H, device=device) + 0.5, torch.arange(W, device=device) + 0.5, indexing="ij")
    mask = ((x >= lo[..., 0, None, None]) & (x <= hi[..., 0, None, None])
            & (y >= lo[..., 1, None, None]) & (y <= hi[..., 1, None, None])
            & (z > 0).all(-1)[..., None, None])
    return mask[:, None, :, None].expand(-1, T, -1, -1, -1, -1)


def handle_roi(batch, anchors, radius_m=0.035):
    """Project a fixed 35 mm radius around the demo's initial handle grasp site to each calibrated camera."""
    device = batch["K"].device
    sites = batch_anchors(batch, anchors)
    # NaN marks unavailable labels and projects to an empty mask, never an invented centre.
    xyz = torch.tensor([a["xyz"] if a and "xyz" in a else [float("nan")] * 3 for a in sites],
                       device=device, dtype=torch.float32)
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
    sites = batch_anchors(batch, anchors)
    center = torch.tensor([a["xyz"] if a and "xyz" in a else [float("nan")] * 3 for a in sites])
    phase = torch.full((len(sites), 3), float("nan"))
    for i, (a, row) in enumerate(zip(sites, batch["meta"]["t_indices"])):
        if a and "grasp" in a:
            label = 0 if row[1] < a["grasp"] else 1 if row[1] < a["release"] else 2
            phase[i] = F.one_hot(torch.tensor(label), 3).float()
    return {
        "ee_initial_handle_relative_m": state[:, 1, :3] - center,
        "gripper_open_fraction": 1 - state[:, 1, 3:4],
        "phase_proxy": phase,
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
    result = {}
    for name, all_y in targets.items():
        train_valid = torch.isfinite(all_y).all(-1) & torch.isfinite(x).all(-1)
        val_valid = torch.isfinite(vy[name]).all(-1) & torch.isfinite(vx).all(-1)
        tx, y, ex, ey = x[train_valid], all_y[train_valid], vx[val_valid], vy[name][val_valid]
        if len(tx) < 2 or not len(ex):
            reason = "not enough valid train/validation labels; initial grasp/phase may be unavailable"
            for key in (f"r2_{name}_validation", f"mlp_r2_{name}_validation", f"mlp_rmse_{name}_validation"):
                result[key] = None
                result.setdefault("not_applicable", {})[key] = reason
            if name == "phase_proxy":
                result["mlp_phase_accuracy_validation"] = None
                result["not_applicable"]["mlp_phase_accuracy_validation"] = reason
            continue
        result.update(fit_and_score(tx, {name: y}, {"validation": (ex, {name: ey})}))
        mean, std = tx.mean(0), tx.std(0).clamp_min(1e-6)
        tx, ex = (tx - mean) / std, (ex - mean) / std
        torch.manual_seed(seed)
        ym, ys = y.mean(0), y.std(0).clamp_min(1e-6)
        net = torch.nn.Sequential(torch.nn.Linear(tx.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, y.shape[1]))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        for _ in range(200):
            opt.zero_grad()
            loss = F.mse_loss(net(tx), (y - ym) / ys)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = net(ex) * ys + ym
            result[f"mlp_r2_{name}_validation"] = r2_score(pred, ey)
            result[f"mlp_rmse_{name}_validation"] = float((pred - ey).square().mean().sqrt())
            if name == "phase_proxy":
                result["mlp_phase_accuracy_validation"] = float((pred.argmax(-1) == ey.argmax(-1)).float().mean())
    return result
