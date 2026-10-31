"""AllTracker 2D tracks -> per-camera 3D tracks in the robot base frame.

AllTracker (aharley/alltracker, pinned in configs/realworld/upstream_pins.yaml) returns, per query frame, dense
flow to every frame of a window plus two sigmoid channels: visibility (channel 0) and confidence (channel 1,
trained as P(error <= 12 px)). Its trajectories are ``flow + grid`` with ``grid = linspace(0, W-1)``, i.e. integer
pixel centres; ``alltracker_to_tracks2d`` converts to the repository's continuous convention (+0.5).

Lifting samples the *same frame's* metric depth (reference = this camera's rectified left image, from the stereo
stage) at the tracked pixel with nearest-pixel lookup (never interpolating across depth edges), unprojects with
the matching intrinsics and maps to the base frame with the episode's accepted calibration. Every (time, point)
gets an explicit status; invalid entries are NaN. Point ids are camera-local: no identity is ever assumed across
cameras (``global_id`` namespaces ids by camera and query frame).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from s4d.realworld.geometry import invert_T, transform_points, unproject

STATUS = {
    0: "valid",
    1: "occluded",  # tracker visibility below threshold
    2: "low_confidence",  # tracker confidence below threshold
    3: "out_of_image",
    4: "no_depth",  # stereo depth invalid at the pixel
    5: "depth_edge",  # depth discontinuity in the 3x3 neighbourhood
    6: "not_tracked",  # frame outside the tracker window
}
VALID, OCCLUDED, LOW_CONF, OUT, NO_DEPTH, EDGE, NOT_TRACKED = range(7)


@dataclass
class Tracks2D:
    camera_id: str
    query_frame: int  # stream frame index of the query
    frame_indices: np.ndarray  # (T,) stream frame indices covered
    timestamps_ns: np.ndarray  # (T,)
    point_ids: np.ndarray  # (N,) camera-local ids
    uv: np.ndarray  # (T,N,2) continuous pixel coords (NaN where untracked)
    visibility: np.ndarray  # (T,N) in [0,1]
    confidence: np.ndarray  # (T,N) in [0,1]
    image_hw: tuple
    source: dict = field(default_factory=dict)  # model, commit, checkpoint, is_mock, args

    def global_ids(self) -> np.ndarray:
        return np.array([f"{self.camera_id}/q{self.query_frame}/{int(i)}" for i in self.point_ids])


@dataclass
class Tracks3D:
    camera_id: str
    query_frame: int
    frame_indices: np.ndarray
    timestamps_ns: np.ndarray
    point_ids: np.ndarray
    xyz_base: np.ndarray  # (T,N,3) float32, NaN where status != valid
    xyz_cam: np.ndarray  # (T,N,3) float32, NaN where status != valid
    status: np.ndarray  # (T,N) uint8, see STATUS
    visibility: np.ndarray
    confidence: np.ndarray
    source: dict = field(default_factory=dict)

    @property
    def valid(self) -> np.ndarray:
        return self.status == VALID


def query_grid(hw: tuple, stride: int, margin: int = 0) -> np.ndarray:
    """(N,2) integer (row, col) query pixels on a regular grid."""
    H, W = hw
    ii, jj = np.meshgrid(np.arange(margin, H - margin, stride), np.arange(margin, W - margin, stride), indexing="ij")
    return np.stack([ii.ravel(), jj.ravel()], 1)


def alltracker_to_tracks2d(flows: np.ndarray, visconf: np.ndarray, query_pixels: np.ndarray, *, camera_id: str,
                           query_frame: int, frame_indices, timestamps_ns, source: dict) -> Tracks2D:
    """Sample AllTracker's dense output at query pixels.

    flows (T,2,H,W) in pixels relative to the query grid, visconf (T,2,H,W) sigmoids (vis, conf);
    query_pixels (N,2) integer (row, col). Output uv are continuous (+0.5 from AllTracker's integer centres).
    """
    T, _, H, W = flows.shape
    r, c = query_pixels[:, 0], query_pixels[:, 1]
    uv = np.stack([flows[:, 0, r, c] + c, flows[:, 1, r, c] + r], -1) + 0.5
    return Tracks2D(camera_id, int(query_frame), np.asarray(frame_indices), np.asarray(timestamps_ns, np.int64),
                    np.arange(len(r), dtype=np.int64), uv.astype(np.float32), visconf[:, 0, r, c].astype(np.float32),
                    visconf[:, 1, r, c].astype(np.float32), (H, W), dict(source))


def lift_tracks(tracks: Tracks2D, depths: np.ndarray, K: np.ndarray, T_cam_base: np.ndarray, *,
                vis_threshold: float = 0.5, conf_threshold: float = 0.5, edge_rel: float = 0.05) -> Tracks3D:
    """depths (T,H,W) metric camera-z on the tracked image grid (0 invalid), K continuous-convention."""
    T, N = tracks.uv.shape[:2]
    H, W = depths.shape[1:]
    if (H, W) != tuple(tracks.image_hw):
        raise ValueError("depth and tracked image grids differ; resize with tracked intrinsics first")
    status = np.full((T, N), VALID, np.uint8)
    uv = tracks.uv
    finite = np.isfinite(uv).all(-1)
    status[~finite] = NOT_TRACKED
    inside = finite & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
    status[finite & ~inside] = OUT
    status[(status == VALID) & (tracks.confidence < conf_threshold)] = LOW_CONF
    status[(status == VALID) & (tracks.visibility < vis_threshold)] = OCCLUDED
    j = np.clip(np.floor(np.nan_to_num(uv[..., 0], nan=-1)).astype(int), 0, W - 1)
    i = np.clip(np.floor(np.nan_to_num(uv[..., 1], nan=-1)).astype(int), 0, H - 1)
    t_idx = np.repeat(np.arange(T)[:, None], N, 1)
    z = depths[t_idx, i, j]
    status[(status == VALID) & ~(np.isfinite(z) & (z > 0))] = NO_DEPTH
    # depth edge test on the 3x3 neighbourhood (all neighbours valid and within edge_rel of the centre)
    padded = np.pad(depths, ((0, 0), (1, 1), (1, 1)))
    neigh = np.stack([padded[t_idx, i + di, j + dj] for di in range(3) for dj in range(3)], -1)
    edge = (neigh <= 0).any(-1) | (np.abs(neigh - z[..., None]).max(-1) > edge_rel * np.maximum(z, 1e-6))
    status[(status == VALID) & edge] = EDGE
    ok = status == VALID
    cam = np.full((T, N, 3), np.nan, np.float32)
    cam[ok] = unproject(uv[ok], z[ok], K)
    base = np.full((T, N, 3), np.nan, np.float32)
    base[ok] = transform_points(invert_T(T_cam_base), cam[ok])
    src = dict(tracks.source)
    src.update({"vis_threshold": vis_threshold, "conf_threshold": conf_threshold, "edge_rel": edge_rel,
                "depth_lookup": "nearest pixel on the tracked (rectified left) grid"})
    return Tracks3D(tracks.camera_id, tracks.query_frame, tracks.frame_indices, tracks.timestamps_ns, tracks.point_ids,
                    base, cam, status, tracks.visibility, tracks.confidence, src)


def status_counts(tracks: Tracks3D) -> dict:
    return {STATUS[k]: int((tracks.status == k).sum()) for k in STATUS}
