"""Display data for the interactive episode viewer (``scripts/realworld_viewer.py``): measured/preprocessed outputs
only, nothing is predicted and the processed data are never modified.

* timeline: the synchronized sets of all views (``depth_stages.frame_sets``), set times from the reference camera,
  recording pauses (gaps longer than ``gap_factor`` frame periods) reported explicitly;
* point clouds: one view's depth (``foundation_stereo_depth`` = IR stereo, or ``sensor_depth`` = raw RealSense)
  unprojected with the stored K and moved to the robot base with the accepted calibration, coloured by the same frame's
  RGB. Display-only downsampling (pixel stride, optional finer stride inside a region of interest) and an optional
  base-frame crop;
* robot: link poses from the composed URDF at the recorded joints and gripper state (forward kinematics), with the
  visual meshes of the arm/gripper/coupling descriptions;
* trails: tracks classified ``moving`` by the ``track_motion`` stage, from ONE query window per camera and frame (the
  window in whose second half the frame lies, so every frame has up to half a window of history). As in PointWorld's
  flow timelines, a line joins two ADJACENT frames only when both samples are valid and in the same motion segment:
  lines are display interpolation between measured samples (returned separately as points) and break at every invalid
  sample, segment break and recording pause; tracks of different cameras or windows are never joined.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from s4d.realworld import io as rio


@dataclass
class ViewerEpisode:
    root: Path
    episode_id: str
    meta: object
    streams: list
    sets: dict
    T_cam_base: np.ndarray  # (V,4,4)
    times_s: np.ndarray  # (S,) reference-camera set times from the first set
    period_s: float
    gaps: list = field(default_factory=list)  # [(set index k, seconds between k-1 and k)]

    @property
    def n_sets(self) -> int:
        return len(self.times_s)


def load_episode(root: Path, episode_id: str, *, gap_factor: float = 1.5) -> ViewerEpisode:
    from s4d.realworld.depth_stages import configured_sets_policy, frame_sets
    from s4d.realworld.pipeline import accepted_calibration, camera_views, load_meta

    root = Path(root)
    meta = load_meta(root, episode_id)
    streams = [v.stream_id for v in camera_views(meta)]
    policy = configured_sets_policy(root, episode_id)
    sets = frame_sets(meta, streams, **policy)
    if not len(sets["index"]):
        raise ValueError(f"{episode_id}: no synchronized camera sets under policy {policy}")
    T = np.stack([accepted_calibration(root, episode_id, s) for s in streams])
    stamps = sets["stamps_ns"][:, 0].astype(np.int64)
    times = (stamps - stamps[0]) * 1e-9
    period = float(np.median(np.diff(times)))
    gaps = [(int(k), float(times[k] - times[k - 1])) for k in range(1, len(times))
            if times[k] - times[k - 1] > gap_factor * period]
    return ViewerEpisode(root, episode_id, meta, streams, sets, T, times, period, gaps)


def segment_of_set(ve: ViewerEpisode) -> np.ndarray:
    """Recording segment (0, 1, ...) of every set: increments at each pause."""
    seg = np.zeros(ve.n_sets, np.int32)
    for k, _ in ve.gaps:
        seg[k:] += 1
    return seg


# ------------------------------------------------------------------------------------------------ point clouds


def view_rgbd(ve: ViewerEpisode, v: int, frame_index: int, source: str, root: Path | None = None):
    """RGB of the view and the depth of stage ``source`` read from ``root`` (default: the episode root; a depth variant
    root links the same normalized frames)."""
    from s4d.realworld.depth_stages import load_view_depth
    from s4d.realworld.pipeline import read_frames

    stream = ve.streams[v]
    rgb = read_frames(ve.root, ve.episode_id, stream, [frame_index])[0]
    depth, valid, K = load_view_depth(root or ve.root, ve.episode_id, stream, [frame_index], source)
    if ve.meta.stream(stream).channel_order == "BGR":
        rgb = rgb[..., ::-1]
    return rgb, depth[0], valid[0], np.asarray(K, float)


def cloud(ve: ViewerEpisode, v: int, frame_index: int, source: str, *, stride: int = 4, roi_stride: int | None = None,
          roi_box: np.ndarray | None = None, crop_box: np.ndarray | None = None, root: Path | None = None):
    """(points_base (N,3) float32, rgb (N,3) uint8) of one view; ``*_box`` = [[xmin,ymin,zmin],[xmax,ymax,zmax]] in
    the base frame. ``roi_stride`` samples pixels inside ``roi_box`` more densely (display only)."""
    from s4d.realworld.geometry import invert_T, pixel_centers, transform_points, unproject

    if roi_stride is not None and stride % roi_stride:
        raise ValueError(f"stride {stride} must be a multiple of roi_stride {roi_stride}")
    rgb, depth, valid, K = view_rgbd(ve, v, frame_index, source, root)
    H, W = depth.shape
    fine = roi_stride or stride
    ii, jj = np.meshgrid(np.arange(0, H, fine), np.arange(0, W, fine), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    ok = valid[ii, jj] & np.isfinite(depth[ii, jj]) & (depth[ii, jj] > 0)
    ii, jj = ii[ok], jj[ok]
    uv = pixel_centers(H, W)[ii, jj]
    pts = transform_points(invert_T(ve.T_cam_base[v]), unproject(uv, depth[ii, jj], K))
    keep = np.ones(len(pts), bool)
    if roi_stride is not None and roi_stride != stride:  # coarse outside the region of interest
        coarse = (ii % stride == 0) & (jj % stride == 0)
        keep = coarse | inside(pts, roi_box)
    if crop_box is not None:
        keep &= inside(pts, crop_box)
    return pts[keep].astype(np.float32), rgb[ii[keep], jj[keep]].astype(np.uint8)


def inside(points: np.ndarray, box) -> np.ndarray:
    if box is None:
        return np.ones(len(points), bool)
    box = np.asarray(box, float)
    return ((points >= box[0]) & (points <= box[1])).all(1)


# ------------------------------------------------------------------------------------------------ robot


def robot_state_at_sets(ve: ViewerEpisode):
    """Joint vector, gripper state and validity at every set (reference-camera time)."""
    from s4d.realworld.pipeline import robot_state_at

    q, g, ok, _ = robot_state_at(ve.meta, ve.sets["stamps_ns"][:, 0].astype(np.float64))
    return q, g, ok


def link_poses(geom, q7, grip) -> dict:
    """{link: T_base_link (4,4)} by forward kinematics of the composed URDF (base = the URDF root)."""
    return geom.model.fk(geom.q(q7, grip))


def urdf_visuals(urdf_path: Path, packages: dict) -> dict:
    """{link: [(mesh_path, T_link_mesh (4,4), scale (3,))]} of a description URDF's VISUAL meshes (package:// resolved)."""
    from s4d.realworld.robot import origin_T

    out = {}
    root = ET.fromstring(Path(urdf_path).read_text())
    for link in root.findall("link"):
        items = []
        for vis in link.findall("visual"):
            g = vis.find("geometry")
            m = g.find("mesh") if g is not None else None
            if m is None:
                continue
            name = m.get("filename")
            for pkg, directory in packages.items():
                name = name.replace(f"package://{pkg}/", f"{Path(directory)}/")
            name = re.sub(r"^file://", "", name)
            scale = np.array([float(x) for x in (m.get("scale") or "1 1 1").split()], float)
            items.append((name, origin_T(vis.find("origin")), scale))
        if items:
            out[link.get("name")] = items
    return out


# ------------------------------------------------------------------------------------------------ trails


@dataclass
class CameraTracks:
    """Moving tracks of one camera: per lift3d chunk the frames, base-frame positions, validity and segments."""

    chunks: list  # [{"query", "frames" (T,), "xyz" (T,M,3), "seg" (T,M)}] moving tracks only
    window_of_frame: dict  # frame index -> chunk position (second-half rule)


def load_camera_tracks(root: Path, episode_id: str, stream: str) -> CameraTracks:
    """Moving tracks (``track_motion`` class 2) of every query window of one camera, from lift3d positions."""
    from s4d.realworld.track_motion import MOVING

    lift = rio.read_record(rio.stage_dir(root, episode_id, "lift3d"))
    chunks = []
    for name in lift["summary"][stream]["chunks"]:
        with np.load(rio.stage_dir(root, episode_id, "lift3d") / name) as z:
            frames, xyz = z["frame_indices"], z["xyz_base"]
        with np.load(rio.stage_dir(root, episode_id, "track_motion") / name) as z:
            keep = np.flatnonzero(z["class"] == MOVING)
            seg = z["segment"][:, keep]
        chunks.append({"name": name, "query": int(frames[0]), "frames": frames.astype(np.int64),
                       "xyz": xyz[:, keep].astype(np.float32), "seg": seg.astype(np.int16)})
    chunks.sort(key=lambda c: c["query"])
    window = {}
    for i, c in enumerate(chunks):  # later chunks overwrite: a frame maps to the newest window with >= half history
        half = len(c["frames"]) // 2
        for f in c["frames"][half:] if i else c["frames"]:
            window[int(f)] = i
    for i, c in enumerate(chunks):  # frames only covered by a first half (episode start/end) fall back to that window
        for f in c["frames"]:
            window.setdefault(int(f), i)
    return CameraTracks(chunks, window)


def trail_segments(ct: CameraTracks, frame_index: int, *, length: int = 8, max_tracks: int | None = None,
                   seed: int = 0, break_frames: set | None = None):
    """Display trails at ``frame_index``: (segments (M,2,3), age in frames of each segment's end (M,), measured
    samples (P,3), sample age (P,), track ids (M,)). Segments join ADJACENT valid samples of one motion segment within
    the last ``length`` frames; a frame in ``break_frames`` (first frame after a recording pause) starts over."""
    i = ct.window_of_frame.get(int(frame_index))
    empty = (np.zeros((0, 2, 3), np.float32), np.zeros(0), np.zeros((0, 3), np.float32), np.zeros(0),
             np.zeros(0, np.int64))
    if i is None:
        return empty
    c = ct.chunks[i]
    frames = list(c["frames"])
    if frame_index not in frames or c["xyz"].shape[1] == 0:
        return empty
    t_now = frames.index(frame_index)
    t0 = max(0, t_now - length)
    if break_frames:
        for t in range(t0 + 1, t_now + 1):
            if int(frames[t]) in break_frames:
                t0 = t
    cols = np.arange(c["xyz"].shape[1])
    if max_tracks is not None and len(cols) > max_tracks:
        cols = np.sort(np.random.default_rng(seed + c["query"]).choice(cols, max_tracks, replace=False))
    xyz, seg = c["xyz"][t0:t_now + 1, cols], c["seg"][t0:t_now + 1, cols]
    valid = (seg >= 0) & np.isfinite(xyz).all(-1)
    segs, ages, ids = [], [], []
    for a in range(len(xyz) - 1):  # adjacent frames only: no line across an invalid sample
        b = a + 1
        m = valid[a] & valid[b] & (seg[a] == seg[b])
        if m.any():
            segs.append(np.stack([xyz[a, m], xyz[b, m]], 1))
            ages.append(np.full(int(m.sum()), len(xyz) - 1 - b))
            ids.append(cols[m])
    pts_age = np.repeat(np.arange(len(xyz) - 1, -1, -1)[:, None], len(cols), 1)
    if not segs:
        return (np.zeros((0, 2, 3), np.float32), np.zeros(0), xyz[valid].astype(np.float32), pts_age[valid],
                np.zeros(0, np.int64))
    return (np.concatenate(segs).astype(np.float32), np.concatenate(ages), xyz[valid].astype(np.float32),
            pts_age[valid], np.concatenate(ids))


def turbo(x: np.ndarray) -> np.ndarray:
    """Google's polynomial approximation of the turbo colormap (PointWorld colours flows with turbo over time)."""
    x = np.clip(np.asarray(x, np.float64), 0, 1)[..., None]
    c = np.array([[0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
                  [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
                  [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973]])
    return np.clip(np.concatenate([x ** k for k in range(6)], -1) @ c.T, 0, 1)


def gripper_flow(geom, q: np.ndarray, g: np.ndarray, ok: np.ndarray, k: int, *, length: int, points: int = 48,
                 seed: int = 0, break_sets: set | None = None):
    """Robot (gripper) flow at set ``k``: trajectories of ``points`` fixed gripper-surface samples from forward
    kinematics of the RECORDED joints/gripper state over the last ``length`` sets (segments (M,2,3), age (M,)).
    Breaks where the robot state is unavailable and after a recording pause, like the tracks."""
    links = geom.gripper_links()
    pool = np.concatenate([geom.surface.link_points[n] for n in links])
    owner = np.concatenate([[n] * len(geom.surface.link_points[n]) for n in links])
    pick = np.random.default_rng(seed).choice(len(pool), min(points, len(pool)), replace=False)
    k0 = max(0, k - length)
    if break_sets:
        for kk in range(k0 + 1, k + 1):
            if kk in break_sets:
                k0 = kk
    P = []
    for kk in range(k0, k + 1):
        if not ok[kk]:
            P.append(np.full((len(pick), 3), np.nan))
            continue
        fk = geom.model.fk(geom.q(q[kk], g[kk]))
        P.append(np.stack([fk[owner[i]][:3, :3] @ pool[i] + fk[owner[i]][:3, 3] for i in pick]))
    P = np.asarray(P)
    segs, ages = [], []
    for a in range(len(P) - 1):
        m = np.isfinite(P[a]).all(-1) & np.isfinite(P[a + 1]).all(-1)
        if m.any():
            segs.append(np.stack([P[a, m], P[a + 1, m]], 1))
            ages.append(np.full(int(m.sum()), len(P) - 2 - a))
    if not segs:
        return np.zeros((0, 2, 3), np.float32), np.zeros(0)
    return np.concatenate(segs).astype(np.float32), np.concatenate(ages)


def describe(ve: ViewerEpisode) -> dict:
    return {"episode": ve.episode_id, "sets": ve.n_sets, "duration_s": float(ve.times_s[-1]),
            "period_s": ve.period_s, "pauses": [{"before_set": k, "gap_s": g} for k, g in ve.gaps],
            "cameras": ve.streams, "max_sync_ns": ve.sets["max_sync_ns"],
            "unmatched_frames": ve.sets.get("unmatched", {s: [] for s in ve.streams})}


def write_json(path: Path, doc: dict) -> None:
    Path(path).write_text(json.dumps(doc, indent=1, default=str))

