"""Training-supported viewpoint domain of the self-render replacement variant (user amendment 2026-10-29, item 2).

The domain is defined ONLY from training-camera geometry (the six training cameras of s4d/data/metaworld/cameras.py:
orbit cameras around LOOKAT (0, 0.6, 0) at radius 1.0, fovy 45) and training observations (the six training cameras'
RGB, depth and body segmentation of training-split episodes, and training-episode body positions). It never reads the
evaluation cameras' images or depth (camera indices >= 6 of the data files), the held-out sets, policy results or any
other evaluation camera definition.

Candidates: orbit poses looking at LOOKAT at the rig radius (every training camera has distance 1.0), azimuth -180..175
and MuJoCo elevation -85..-5 in 5 degree steps (every training pose is a grid point). Per candidate and training frame,
the six training cameras' depth is back-projected and fused (as the reference's point-cloud synthetic views) and
z-buffer splatted into the candidate (s4d.diag.heldout.splat_batched semantics, 1 px per point):

  coverage        fraction of candidate pixels hit by an observed point;
  support2        among covered pixels, fraction whose front point is observed by >= 2 training cameras (a point is
                  observed by camera j when it projects inside j's image with |z_j - depth_j| <= 1 cm);
  ws_visible      workspace visibility: fraction of the robot/object surface points (bodies that move by > 1 cm within
                  some training episode, from xpos) visible from the candidate (in view, within 1 cm of its z-buffer);
  ws_support2     among candidate pixels showing a robot/object point, fraction observed by >= 2 training cameras.

Criteria (no threshold is tuned): for every analysis task, each of coverage, support2, ws_visible and ws_support2 >= its
minimum over the six training poses and the analysis tasks, i.e. a candidate must be at least as well supported as the
least supported training camera. (Draft rule, revised before freezing: coverage >= 0.9 = PATCH_COVERAGE. It excluded the
two 30-degree training poses themselves, whose own coverage is 0.83-0.87 because they see background beyond the far
plane, contradicting the definition "training poses and intermediate viewpoints"; coverage now uses the same
training-relative minimum as the other three measures.) The domain is
the set of supported cells in the 4-connected components (azimuth wraps) that contain a training pose; a cell covers its
grid point +-2.5 degrees. Representative renders of the decoded Gaussians of the csw4 checkpoints (training windows, t0)
at training poses, domain cells and rejected cells are written for inspection (not used as a criterion: high opacity is
not evidence of accurate geometry).

    CUDA_VISIBLE_DEVICES=<MIG uuid> python -E -s scripts/selfrender_domain.py --out runs/setup/selfrender-domain-v1 \
        [--write-domain configs/metaworld/selfrender_domain_v1.yaml]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401  (sys.path)
from _bootstrap import REPO, guard_gpus

guard_gpus()

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from s4d.data.metaworld.cameras import FOVY_DEG, LOOKAT, RADIUS, TRAIN_CAMERAS, orbit_rig  # noqa: E402
from s4d.geometry import lift_depth, project  # noqa: E402
from s4d.train.augment import PATCH_COVERAGE  # noqa: E402

DATA = Path("/home/ws/data/metaworld/splatter4d_v1")
TASKS = ("hammer", "pick-place", "bin-picking")
FRAMES_PER_TASK = 16
SEED = 20261029
H = W = 128
NEAR, FAR = 0.05, 3.0
OBS_TOL_M = 0.01  # observation / visibility tolerance (the Chamfer voxel size of chamfer_lean)
DYNAMIC_BODY_M = 0.01  # a body is robot/object when it moves by more than this within some training episode
AZIMUTHS = tuple(range(-180, 180, 5))
ELEVATIONS = tuple(range(-85, 0, 5))
CELL_DEG = 5.0
WINDOW_STRIDE = 2  # encoder windows for the representative renders: (t, t+2, t+4), the RL spacing
CHECKPOINTS = {t: Path(f"/home/ws/ws/splatter4d/runs/enc-csw4-seed0-{t}-200k/checkpoints/step_0200000.pt") for t in TASKS}


def train_poses() -> list[tuple[float, float]]:
    """(azimuth, elevation) of the six training cameras in the grid convention (azimuth in [-180, 180))."""
    return [(((phi + 180.0) % 360.0) - 180.0, -theta) for theta, phi in TRAIN_CAMERAS]


def dynamic_bodies(f: h5py.File, episodes: list[str]) -> np.ndarray:
    """Body ids that move by more than DYNAMIC_BODY_M within some training episode (robot links and task objects)."""
    moving = None
    for ep in episodes:
        xpos = f["episodes"][ep]["xpos"][:]
        span = np.linalg.norm(xpos.max(0) - xpos.min(0), axis=-1) > DYNAMIC_BODY_M
        moving = span if moving is None else moving | span
    return np.nonzero(moving)[0]


def analysis_frames(task: str) -> tuple[list[tuple[str, int]], list[str]]:
    split = json.loads((DATA / "splits" / f"{task}_seed0.json").read_text())
    rng = np.random.default_rng(SEED)
    with h5py.File(DATA / f"{task}.hdf5", "r") as f:
        eps = [split["train"][i] for i in sorted(rng.choice(len(split["train"]), FRAMES_PER_TASK, replace=False))]
        frames = [(ep, int(rng.integers(0, int(f["episodes"][ep].attrs["length"]) - 2 * WINDOW_STRIDE))) for ep in eps]
    return frames, split["train"]


def load_frame(f: h5py.File, ep: str, t: int, cams: dict, dyn: np.ndarray, device) -> dict:
    """Fused training-camera cloud of frame t: points (P,3), colours, valid, robot/object flag, observation count."""
    V = len(cams["train"])
    unit = float(f.attrs["depth_unit_m"])
    depth = torch.from_numpy(f["episodes"][ep]["depth"][t, :V].astype(np.float32) * unit).to(device)
    rgb = torch.from_numpy(f["episodes"][ep]["rgb"][t, :V]).to(device).float() / 255.0  # (V,H,W,3)
    body = torch.from_numpy(f["episodes"][ep]["body_id"][t].astype(np.int64)).to(device)  # (V,H,W)
    K, c2w, w2c = (cams[k].to(device) for k in ("K", "c2w", "w2c"))
    xyz = lift_depth(depth, K, c2w)  # (V,H,W,3)
    valid = (depth > 0) & (depth <= FAR)
    dyn_mask = torch.isin(body, torch.as_tensor(dyn, device=device))
    pts = xyz.reshape(-1, 3)
    # observation count: training cameras that see each point (inside, |z - depth| <= tol)
    uv, z = project(pts[None].expand(V, -1, 3), K, w2c)  # (V,P,2), (V,P)
    u, v = torch.floor(uv[..., 0]).long(), torch.floor(uv[..., 1]).long()
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > NEAR)
    d = depth[torch.arange(V, device=device)[:, None], v.clamp(0, H - 1), u.clamp(0, W - 1)]
    seen = inside & (d > 0) & ((z - d).abs() <= OBS_TOL_M)
    return {"points": pts, "colors": rgb.reshape(-1, 3), "valid": valid.reshape(-1), "dynamic": dyn_mask.reshape(-1),
            "n_obs": seen.sum(0), "depth": depth}


def splat_metrics(cloud: dict, w2c: torch.Tensor, K: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per candidate camera (C,4,4): coverage, support2, ws_visible, ws_support2, ws_pixels, and the splat RGB."""
    C = len(w2c)
    keep_pt = cloud["valid"]
    pts, n_obs, dyn = cloud["points"][keep_pt], cloud["n_obs"][keep_pt], cloud["dynamic"][keep_pt]
    P = len(pts)
    uv, z = project(pts[None].expand(C, P, 3), K[None].expand(C, 3, 3), w2c)
    u, v = torch.floor(uv[..., 0]).long(), torch.floor(uv[..., 1]).long()
    keep = (z > NEAR) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    cam = torch.arange(C, device=pts.device)[:, None].expand(C, P)
    index = (cam * H + v.clamp(0, H - 1)) * W + u.clamp(0, W - 1)
    zbuf = torch.full((C * H * W,), float("inf"), device=pts.device)
    zbuf.scatter_reduce_(0, index[keep], z[keep], reduce="amin")
    front = keep & (z <= zbuf[index])
    covered = torch.isfinite(zbuf).view(C, H * W)
    sup = torch.zeros(C * H * W, device=pts.device)
    sup.scatter_reduce_(0, index[front], (n_obs[None].expand(C, P)[front] >= 2).float(), reduce="amax")
    dpix = torch.zeros(C * H * W, device=pts.device)
    dpix.scatter_reduce_(0, index[front], dyn[None].expand(C, P)[front].float(), reduce="amax")
    sup, dpix = sup.view(C, H * W), dpix.view(C, H * W)
    visible = keep & (z <= zbuf[index] + OBS_TOL_M)
    n_dyn = dyn.sum().clamp_min(1)
    ws_pix = dpix.sum(1)
    rgb = torch.zeros(C * H * W, 3, device=pts.device)
    rgb[index[front]] = cloud["colors"][keep_pt][None].expand(C, P, 3)[front]
    return {
        "coverage": covered.float().mean(1),
        "support2": (sup * covered).sum(1) / covered.sum(1).clamp_min(1),
        "ws_visible": (visible & dyn[None]).sum(1) / n_dyn,
        "ws_support2": torch.where(ws_pix > 0, (sup * dpix).sum(1) / ws_pix.clamp_min(1), torch.zeros_like(ws_pix)),
        "ws_pixels": ws_pix / (H * W),
        "rgb": rgb.view(C, H, W, 3),
        "covered": covered.view(C, H, W),
    }


METRICS = ("coverage", "support2", "ws_visible", "ws_support2", "ws_pixels")


def support_maps(device) -> dict:
    grid = [(a, e) for e in ELEVATIONS for a in AZIMUTHS]
    rig = orbit_rig([(a, e, RADIUS) for a, e in grid], H, W)
    w2c_all = torch.as_tensor(rig["w2c"], device=device)
    K = torch.as_tensor(rig["K"][0], device=device)
    out = {"grid": grid, "tasks": {}}
    for task in TASKS:
        frames, train_eps = analysis_frames(task)
        with h5py.File(DATA / f"{task}.hdf5", "r") as f:
            names = [n.decode() for n in f["cameras/names"][:]]
            train = np.nonzero(f["cameras/is_train"][:])[0]
            assert list(train) == list(range(6)) and names[:6] == [f"train{i}" for i in range(6)]
            cams = {"train": train,
                    **{k: torch.from_numpy(f["cameras"][k][:6].astype(np.float32)) for k in ("K", "c2w", "w2c")}}
            # the rig in the data equals cameras.py: the training poses' orbit_rig matrices reproduce the stored ones
            ref = orbit_rig([(a, e, RADIUS) for a, e in train_poses()], H, W)
            assert np.allclose(ref["c2w"], cams["c2w"].numpy(), atol=1e-4), "data rig differs from cameras.py"
            assert np.allclose(ref["K"], cams["K"].numpy(), atol=1e-3)
            dyn = dynamic_bodies(f, train_eps)
            body_names = json.loads(f.attrs["body_names"])
            sums = {m: torch.zeros(len(grid), device=device) for m in METRICS}
            for ep, t in frames:
                cloud = load_frame(f, ep, t, cams, dyn, device)
                for s in range(0, len(grid), 72):
                    res = splat_metrics(cloud, w2c_all[s : s + 72], K)
                    for m in METRICS:
                        sums[m][s : s + 72] += res[m]
        out["tasks"][task] = {
            "frames": frames,
            "dynamic_bodies": [body_names[i] for i in dyn],
            **{m: (sums[m] / len(frames)).cpu().numpy() for m in METRICS},
        }
        print(task, "done", flush=True)
    return out


def components(supported: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    """4-connected components (azimuth wraps) of ``supported`` (E,A) containing the seed cells."""
    E, A = supported.shape
    keep = np.zeros_like(supported)
    stack = [s for s in seeds if supported[s]]
    while stack:
        e, a = stack.pop()
        if keep[e, a]:
            continue
        keep[e, a] = True
        for ne, na in ((e + 1, a), (e - 1, a), (e, (a + 1) % A), (e, (a - 1) % A)):
            if 0 <= ne < E and supported[ne, na] and not keep[ne, na]:
                stack.append((ne, na))
    return keep


def decide(maps: dict) -> dict:
    E, A = len(ELEVATIONS), len(AZIMUTHS)
    idx = {g: i for i, g in enumerate(maps["grid"])}
    tp = [idx[(int(a), int(e))] for a, e in train_poses()]
    per_task = maps["tasks"]
    thresholds = {}
    for m in ("coverage", "support2", "ws_visible", "ws_support2"):
        thresholds[m] = float(min(per_task[t][m][tp].min() for t in TASKS))
    ok = np.ones(len(maps["grid"]), dtype=bool)
    for t in TASKS:
        for m, thr in thresholds.items():
            ok &= per_task[t][m] >= thr - 1e-9
    supported = ok.reshape(E, A)
    seeds = [(ELEVATIONS.index(int(e)), AZIMUTHS.index(int(a))) for a, e in train_poses()]
    domain = components(supported, seeds)
    worst = {m: np.min([per_task[t][m] for t in TASKS], axis=0) for m in METRICS}
    return {"thresholds": thresholds, "supported": supported, "domain": domain, "worst": worst, "train_index": tp}


def solid_angle(el: float) -> float:
    lo, hi = math.radians(el - CELL_DEG / 2), math.radians(el + CELL_DEG / 2)
    return math.radians(CELL_DEG) * (math.sin(hi) - math.sin(lo))


@torch.no_grad()
def representative_renders(maps: dict, decision: dict, out: Path, device) -> dict:
    """Decoded Gaussians of the csw4 checkpoints on training windows, rendered at t0 into training poses, domain cells
    and rejected cells; agreement with the fused training cloud splat on its covered pixels (PSNR) and alpha."""
    from PIL import Image

    from s4d.model.render import render_rgbd
    from s4d.train.loop import build_model

    rng = np.random.default_rng(SEED + 1)
    grid = maps["grid"]
    dom = np.flatnonzero(decision["domain"].reshape(-1))
    rej = np.flatnonzero(~decision["domain"].reshape(-1))
    picks = {"train": np.asarray(decision["train_index"]), "domain": rng.choice(dom, 12, replace=False),
             "rejected": rng.choice(rej, 12, replace=False)}
    order = [(k, i) for k in ("train", "domain", "rejected") for i in picks[k]]
    rig = orbit_rig([(grid[i][0], grid[i][1], RADIUS) for _, i in order], H, W)
    w2c = torch.as_tensor(rig["w2c"], device=device)
    K = torch.as_tensor(rig["K"], device=device)
    stats = {}
    for task in TASKS:
        ckpt = CHECKPOINTS[task]
        if not ckpt.is_file():
            stats[task] = {"missing_checkpoint": str(ckpt)}
            continue
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = build_model(state["config"]).to(device).eval()
        model.encoder.load_state_dict(state["encoder"])
        model.decoder.load_state_dict(state["decoder"])
        frames = maps["tasks"][task]["frames"][:4]
        rows, acc = [], {k: {"psnr_vs_cloud": [], "alpha_mean": [], "alpha_token_eligible": []} for k in picks}
        with h5py.File(DATA / f"{task}.hdf5", "r") as f:
            dyn = dynamic_bodies(f, json.loads((DATA / "splits" / f"{task}_seed0.json").read_text())["train"])
            cams = {"train": np.arange(6), **{k: torch.from_numpy(f["cameras"][k][:6].astype(np.float32))
                                             for k in ("K", "c2w", "w2c")}}
            for ep, t in frames:
                rgb = f["episodes"][ep]["rgb"][[t, t + WINDOW_STRIDE, t + 2 * WINDOW_STRIDE], :6]  # (T,6,H,W,3)
                x = torch.from_numpy(rgb).to(device).permute(1, 0, 4, 2, 3).float() / 255.0  # (6,T,3,H,W)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    slots = model.encoder(x, None, mask_ratio=0.0)["slots"]
                    gs = model.decoder(slots[:1])  # source: training camera 0
                xyz0 = gs.xyz.float()[:, None]
                gsf = type(gs)(gs.xyz.float(), gs.scales.float(), gs.quats.float(), gs.opacity.float(), gs.rgb.float(),
                               gs.delta01.float(), gs.delta12.float(), gs.group)
                r = render_rgbd(gsf, xyz0, w2c[None], K[None], H, W, NEAR, FAR)
                g_rgb, g_alpha = r["rgb"][0, 0].clamp(0, 1), r["alpha"][0, 0]  # (C,3,H,W), (C,1,H,W)
                cloud = load_frame(f, ep, t, cams, dyn, device)
                sp = splat_metrics(cloud, w2c, K[0])
                cov = sp["covered"]
                se = ((g_rgb.permute(0, 2, 3, 1) - sp["rgb"]) ** 2).mean(-1)
                mse = (se * cov).sum((1, 2)) / cov.sum((1, 2)).clamp_min(1)
                psnr = -10 * torch.log10(mse.clamp_min(1e-10))
                frac = torch.nn.functional.avg_pool2d(g_alpha, 16).flatten(1)
                for j, (kind, _) in enumerate(order):
                    acc[kind]["psnr_vs_cloud"].append(float(psnr[j]))
                    acc[kind]["alpha_mean"].append(float(g_alpha[j].mean()))
                    acc[kind]["alpha_token_eligible"].append(float((frac[j] >= PATCH_COVERAGE).float().mean()))
                gs_row = (g_rgb.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                pc_row = (sp["rgb"].cpu().numpy() * 255).astype(np.uint8)
                if len(rows) < 4:  # the image keeps the first 2 windows; the statistics use all 4
                    rows.append(np.concatenate(list(gs_row), 1))
                    rows.append(np.concatenate(list(pc_row), 1))
        Image.fromarray(np.concatenate(rows, 0)).save(out / f"renders_{task}.jpg", quality=85)
        stats[task] = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in acc.items()}
    stats["columns"] = [{"kind": k, "azimuth": grid[i][0], "elevation": grid[i][1]} for k, i in order]
    stats["layout"] = ("per task JPEG: for each of the first 2 of the 4 training windows, a row of decoded-Gaussian "
                       "renders (t0) then a row of "
                       "the fused training-cloud splat, columns as listed (6 training poses, 12 domain cells, 12 rejected)")
    return stats


def support_png(decision: dict, out: Path) -> None:
    from PIL import Image

    img = np.zeros((len(ELEVATIONS), len(AZIMUTHS), 3), dtype=np.uint8)
    img[decision["supported"]] = (90, 90, 160)
    img[decision["domain"]] = (60, 170, 90)
    for a, e in train_poses():
        img[ELEVATIONS.index(int(e)), AZIMUTHS.index(int(a))] = (240, 200, 40)
    Image.fromarray(np.kron(img, np.ones((12, 12, 1), dtype=np.uint8))).save(out / "domain_map.png")


def relate_eval(domain_path: Path, grid_json: Path, suite: str = "v1") -> dict:
    """AFTER freezing (never an input of the domain): where the existing evaluation cameras lie relative to the frozen
    domain -- the fixed held-out cameras eval0..3 (s4d/rl/env.py HELDOUT_PATHS) and the lateral / circular trajectories
    around train1 (trajectory_path). A pose is inside when its (azimuth, elevation) falls in a domain cell (+-2.5 deg);
    its distance is reported separately (the domain is at the rig radius 1.0). For each pose: the angle to the nearest
    domain cell centre and the training-data support values of the nearest grid point."""
    from s4d.rl.env import HELDOUT_PATHS, trajectory_path
    from s4d.train.augment import load_viewpoint_domain

    domain = load_viewpoint_domain(domain_path)
    full = json.loads(grid_json.read_text())
    grid = np.asarray(full["grid"], dtype=np.float64)
    cells = domain["cells"]

    def unit(az, el):
        az, el = np.radians(az), np.radians(el)
        return np.stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), -1)

    def describe(cam) -> dict:
        az = ((cam.azimuth + 180.0) % 360.0) - 180.0
        d_az = (az - cells[:, 0] + 180.0) % 360.0 - 180.0
        tol = domain["half"] + 1e-9
        inside = bool(((np.abs(d_az) <= tol) & (np.abs(cam.elevation - cells[:, 1]) <= tol)).any())
        ang = np.degrees(np.arccos(np.clip(unit(cells[:, 0], cells[:, 1]) @ unit(az, cam.elevation), -1, 1)))
        g = int(np.argmin(np.degrees(np.arccos(np.clip(unit(grid[:, 0], grid[:, 1]) @ unit(az, cam.elevation), -1, 1)))))
        support = {t: {m: full["tasks"][t][m][g] for m in ("coverage", "support2", "ws_visible", "ws_support2")}
                   for t in full["tasks"]}
        return {"azimuth": round(az, 3), "elevation": round(cam.elevation, 3), "distance": round(cam.distance, 4),
                "inside": inside, "nearest_domain_cell_deg": round(float(ang.min()), 2),
                "nearest_grid_point": [int(grid[g, 0]), int(grid[g, 1])], "grid_point_in_domain": bool(full["domain"][g]),
                "support_at_grid_point": support}

    def azimuth_range(poses) -> list[float]:
        # Wrap-aware: offsets from the path's first pose, so a path around +-180 deg is not reported as -180..180.
        a0 = poses[0]["azimuth"]
        offsets = [((p["azimuth"] - a0 + 180.0) % 360.0) - 180.0 for p in poses]
        return [round(a0 + min(offsets), 3), round(a0 + max(offsets), 3)]

    out = {"domain": str(domain_path), "fixed": {name: describe(path[0]) for name, path in HELDOUT_PATHS.items()}}
    kinds = ("lateral", "circular")
    if suite == "v2":
        # Camera suite v2 (s4d.rl.camera_suites; frozen before any policy evaluation): the common held-out rig
        # vis0..3, every task-specific exception camera, and the stronger moving levels.
        from s4d.rl.camera_suites import HELDOUT_V2_TASK_EXCEPTIONS, STRONGER_TRAJECTORIES
        from s4d.rl.env import HELDOUT_V2_PATHS, orbit_camera

        out = {"domain": str(domain_path),
               "fixed": {name: describe(path[0]) for name, path in HELDOUT_V2_PATHS.items()},
               "task_exceptions": {f"{task}:{name}": describe(orbit_camera(*cam))
                                   for task, cams in HELDOUT_V2_TASK_EXCEPTIONS.items() for name, cam in cams.items()}}
        kinds = tuple(STRONGER_TRAJECTORIES)
    for kind in kinds:
        poses = [describe(c) for c in trajectory_path(kind)]
        out[f"traj_{kind}"] = {
            "poses": len(poses), "inside_fraction": float(np.mean([p["inside"] for p in poses])),
            "azimuth_range": azimuth_range(poses),
            "elevation_range": [min(p["elevation"] for p in poses), max(p["elevation"] for p in poses)],
            "distance_range": [min(p["distance"] for p in poses), max(p["distance"] for p in poses)],
            "max_nearest_domain_cell_deg": max(p["nearest_domain_cell_deg"] for p in poses),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-domain", default=None, help="freeze: write the domain YAML here")
    ap.add_argument("--no-renders", action="store_true")
    ap.add_argument("--relate-eval", default=None, help="after freezing: relate the evaluation cameras to this domain YAML")
    ap.add_argument("--suite", choices=("v1", "v2"), default="v1",
                    help="--relate-eval: v1 = eval0..3 and the 1x paths; v2 = camera suite v2 (own output file)")
    args = ap.parse_args()
    out = Path(args.out)
    out = out if out.is_absolute() else REPO / out
    out.mkdir(parents=True, exist_ok=True)
    if args.relate_eval:
        result = relate_eval(Path(args.relate_eval), out / "support_grid.json", args.suite)
        name = "eval_relationship.json" if args.suite == "v1" else "eval_relationship_cameras_v2.json"
        (out / name).write_text(json.dumps(result, indent=1))
        print(json.dumps({k: v if k.startswith("traj") else {n: {f: d[f] for f in ("azimuth", "elevation", "inside",
                          "nearest_domain_cell_deg", "grid_point_in_domain")} for n, d in v.items()}
                          for k, v in result.items() if k != "domain"}, indent=1))
        return
    device = torch.device("cuda")
    maps = support_maps(device)
    decision = decide(maps)
    support_png(decision, out)
    grid = maps["grid"]
    full = {
        "grid": [list(g) for g in grid],
        "thresholds": decision["thresholds"],
        "tasks": {t: {"frames": d["frames"], "dynamic_bodies": d["dynamic_bodies"],
                      **{m: [round(float(x), 5) for x in d[m]] for m in METRICS}} for t, d in maps["tasks"].items()},
        "supported": decision["supported"].reshape(-1).tolist(),
        "domain": decision["domain"].reshape(-1).tolist(),
        "train_poses": train_poses(),
        "train_pose_values": {t: {m: [round(float(maps["tasks"][t][m][i]), 5) for i in decision["train_index"]]
                                  for m in METRICS} for t in TASKS},
    }
    if not args.no_renders:
        full["renders"] = representative_renders(maps, decision, out, device)
    (out / "support_grid.json").write_text(json.dumps(full, indent=1))
    cells = [i for i, d in enumerate(decision["domain"].reshape(-1)) if d]
    area = sum(solid_angle(grid[i][1]) for i in cells)
    print(json.dumps({"thresholds": decision["thresholds"], "supported_cells": int(decision["supported"].sum()),
                      "domain_cells": len(cells), "domain_solid_angle_sr": area,
                      "renders": full.get("renders")}, indent=1))
    if args.write_domain:
        doc = {
            "version": "selfrender_domain_v1",
            "frozen": "2026-10-29",
            "definition": "scripts/selfrender_domain.py (module docstring); full grid: "
                          + str(out.relative_to(REPO) / "support_grid.json"),
            "rig": {"lookat": list(LOOKAT), "distance": RADIUS, "fovy_deg": FOVY_DEG, "image": [H, W],
                    "train_poses_az_el": [list(p) for p in train_poses()]},
            "analysis": {"tasks": list(TASKS), "frames_per_task": FRAMES_PER_TASK, "seed": SEED,
                         "split": "training episodes of splits/<task>_seed0.json", "cameras": "training cameras 0-5 only",
                         "observation_tolerance_m": OBS_TOL_M, "dynamic_body_motion_m": DYNAMIC_BODY_M,
                         "dynamic_bodies": {t: maps["tasks"][t]["dynamic_bodies"] for t in TASKS}},
            "criteria": {"coverage_min": round(decision["thresholds"]["coverage"], 5),
                         "support2_min": round(decision["thresholds"]["support2"], 5),
                         "ws_visible_min": round(decision["thresholds"]["ws_visible"], 5),
                         "ws_support2_min": round(decision["thresholds"]["ws_support2"], 5),
                         "rule": "every task's mean >= each minimum; 4-connected components containing a training pose"},
            "cell_deg": CELL_DEG,
            "sampling": "cell with probability proportional to its solid angle, then azimuth uniform and sin(elevation) "
                        "uniform within the cell (uniform on the sphere over the domain); distance = rig radius; "
                        "look-at = LOOKAT; one camera per rendered view, fixed over the three reconstruction times",
            "summary": {"domain_cells": len(cells), "supported_cells_all_components": int(decision["supported"].sum()),
                        "domain_solid_angle_sr": round(area, 5),
                        "elevation_range": [min(grid[i][1] for i in cells), max(grid[i][1] for i in cells)]},
            "cells": [[int(grid[i][0]), int(grid[i][1]),
                       *[round(float(decision["worst"][m][i]), 4)
                         for m in ("coverage", "support2", "ws_visible", "ws_support2")]]
                      for i in cells],
            "cells_columns": ["azimuth_deg", "elevation_deg", "coverage", "support2", "ws_visible", "ws_support2"],
        }
        path = Path(args.write_domain)
        path = path if path.is_absolute() else REPO / path
        path.write_text("# Frozen training-supported viewpoint domain of variant selfrender_replace_v2 (generated by "
                        "scripts/selfrender_domain.py; do not edit by hand).\n"
                        "# Per-cell values: the minimum over the analysis tasks of the per-task means.\n"
                        + yaml.safe_dump(doc, sort_keys=False, width=120, default_flow_style=None))
        print("wrote", path)


if __name__ == "__main__":
    main()
