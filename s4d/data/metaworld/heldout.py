"""Reader for the near-view held-out camera sets (review item 1a); no MuJoCo import, safe in loader workers.

The files are written by ``s4d.data.metaworld.heldout_render`` (``scripts/render_heldout_sets.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

DATA_ROOT = Path("/home/ws/data/metaworld/splatter4d_v1")


def heldout_path(task: str, root: str | Path = DATA_ROOT) -> Path:
    return Path(root) / "heldout_sets" / f"{task}.hdf5"


class HeldoutSets:
    """Reader for one task's held-out-set file; ``window(ep, t_idx)`` returns per-set tensors for a sample."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._file: h5py.File | None = None
        with h5py.File(self.path, "r") as f:
            if not f.attrs["complete"]:
                raise ValueError(f"{self.path} is incomplete")
            self.depth_unit = float(f.attrs["depth_unit_m"])
            self.episodes = set(f["episodes"].keys())
            cams = f["cameras"]
            sets = [s.decode() if isinstance(s, bytes) else str(s) for s in cams["set"][:]]
            self.columns = {name: np.asarray([i for i, s in enumerate(sets) if s == name]) for name in json.loads(f.attrs["sets"])}
            self.K, self.c2w, self.w2c = (np.asarray(cams[k][:], dtype=np.float32) for k in ("K", "c2w", "w2c"))

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def __contains__(self, episode: str) -> bool:
        return episode in self.episodes

    def window(self, episode: str, t_idx: list[int]) -> dict[str, np.ndarray]:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        g = self._file["episodes"][episode]
        rgb = np.stack([g["rgb"][t] for t in t_idx])  # (T,V,H,W,3)
        depth = np.stack([g["depth"][t] for t in t_idx]).astype(np.float32) * self.depth_unit  # (T,V,H,W)
        out = {}
        for name, cols in self.columns.items():
            out[f"{name}_images"] = np.ascontiguousarray(rgb[:, cols].transpose(0, 1, 4, 2, 3))
            out[f"{name}_depth"] = np.ascontiguousarray(depth[:, cols][:, :, None])
            out[f"{name}_K"], out[f"{name}_w2c"], out[f"{name}_c2w"] = self.K[cols], self.w2c[cols], self.c2w[cols]
        return out
