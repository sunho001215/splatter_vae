"""Track chunk contract of the ``track2d`` / ``lift3d`` stages (written by ``trackeverything_stage``): per-sample
status codes, the 2D/3D track records, the query grid and the birth-depth support test.

Pixel coordinates are continuous (pixel centres at +0.5). 3D samples are the same-frame metric points in the robot base
frame (via the episode's accepted calibration). Every (time, point) has an explicit status; invalid entries are NaN.
Point ids are camera-local: no identity is ever assumed across cameras (``global_id`` namespaces ids by camera and
query frame).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STATUS = {
    0: "valid",
    1: "occluded",  # tracker visibility below threshold
    2: "low_confidence",  # tracker confidence below threshold
    3: "out_of_image",
    4: "no_depth",  # depth invalid at the pixel
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


def depth_support_status(depths: np.ndarray, uv: np.ndarray, edge_rel: float = 0.05) -> np.ndarray:
    """Depth support at continuous pixels: floor lookup and a 3x3 relative-spread (depth edge) test.

    Zero padding treats image boundaries as edges. A missing centre is NO_DEPTH; a missing/nonfinite neighbour or
    one differing by more than 5% of the centre is EDGE. Callers supply finite, in-bounds query pixels.
    """
    j, i = np.floor(uv[..., 0]).astype(int), np.floor(uv[..., 1]).astype(int)
    t_idx = np.arange(len(depths))[:, None]
    z = depths[t_idx, i, j]
    padded = np.pad(depths, ((0, 0), (1, 1), (1, 1)))
    neigh = np.stack([padded[t_idx, i + di, j + dj] for di in range(3) for dj in range(3)], -1)
    edge = (~np.isfinite(neigh) | (neigh <= 0)).any(-1)
    edge |= np.abs(neigh - z[..., None]).max(-1) > edge_rel * np.maximum(z, 1e-6)
    return np.where(~np.isfinite(z) | (z <= 0), NO_DEPTH, np.where(edge, EDGE, VALID)).astype(np.uint8)
