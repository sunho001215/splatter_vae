"""Meta-World window dataset producing the batch contract, with ground-truth 3D motion.

Motion is derived from simulator state only: per-pixel body id + body poses give
exact rigid displacements. Segmentation is never used as a loss mask.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from s4d.data import METAWORLD_ROOT, writable_path
from s4d.data.contract import MOTION_SCORE_SCALE_M, PAIRS, T_WINDOW
from s4d.geometry import lift_depth, rigid_body_displacement

# Frame strides in simulator steps. RL observations are 2 simulator steps apart (action repeat 2),
# which lies inside the pretraining distribution; validation reports the RL spacing and the largest stride.
TRAIN_STRIDES = (2, 4, 6)
VAL_STRIDES = (2, 6)
PROBE_DIM = 11  # hand xyz, gripper opening, object-1 xyz, object-1 quaternion (wxyz)


def list_episodes(path: str | Path) -> list[str]:
    with h5py.File(path, "r") as f:
        return sorted(f["episodes"].keys())


def split_episodes(
    path: str | Path, train_ratio: float = 0.96, seed: int = 0, manifest_dir: str | Path | None = None
) -> tuple[list[str], list[str]]:
    """Deterministic episode split; the manifest is written next to the data file."""
    path = Path(path)
    manifest_dir = writable_path(manifest_dir or path.parent / "splits", METAWORLD_ROOT)
    episodes = list_episodes(path)
    rng = random.Random(seed)
    shuffled = episodes[:]
    rng.shuffle(shuffled)
    n_train = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_ratio))))
    train, val = sorted(shuffled[:n_train]), sorted(shuffled[n_train:])
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"{path.stem}_seed{seed}.json"
    manifest.write_text(
        json.dumps(
            {"file": path.name, "seed": seed, "train_ratio": train_ratio, "train": train, "validation": val}, indent=1
        )
    )
    return train, val


class MetaworldWindowDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        episodes: list[str],
        strides=TRAIN_STRIDES,
        with_eval: bool = False,
        seed: int = 0,
    ):
        """One sample per (episode, start frame, stride). Start frames are kept only where *every* stride fits,
        so each sample's stride is exactly uniform over ``strides``."""
        self.path = str(path)
        self.episodes = list(episodes)
        self.strides = tuple(int(s) for s in strides)
        self.with_eval = with_eval
        self._file: h5py.File | None = None
        with h5py.File(self.path, "r") as f:
            self.task = str(f.attrs["task"])
            self.depth_unit = float(f.attrs["depth_unit_m"])
            self.dt_seconds = float(f.attrs["dt_seconds"])
            self.background_body = int(f.attrs["background_body"])
            self.body_names = json.loads(f.attrs["body_names"])
            cams = f["cameras"]
            self.K = torch.from_numpy(cams["K"][:]).float()
            self.c2w = torch.from_numpy(cams["c2w"][:]).float()
            self.w2c = torch.from_numpy(cams["w2c"][:]).float()
            is_train = cams["is_train"][:].astype(bool)
            self.train_cams = np.where(is_train)[0]
            self.eval_cams = np.where(~is_train)[0]
            self.samples: list[tuple[str, int, tuple[int, ...]]] = []
            for ep in self.episodes:
                length = int(f["episodes"][ep].attrs["length"])
                for t0 in range(length - (T_WINDOW - 1) * max(self.strides)):
                    self.samples.extend((ep, t0, (s,)) for s in self.strides)
        if not self.samples:
            raise ValueError(f"no temporal windows in {self.path} for episodes {self.episodes[:3]}...")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def _h5(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        ep_key, t0, valid = self.samples[index]
        stride = valid[0]
        t_idx = [t0 + k * stride for k in range(T_WINDOW)]
        ep = self._h5()["episodes"][ep_key]
        n_cams = len(self.K) if self.with_eval else len(self.train_cams)
        if not np.array_equal(self.train_cams, np.arange(len(self.train_cams))):
            raise ValueError("training cameras must be stored first in the camera table")
        rgb = torch.from_numpy(np.stack([ep["rgb"][t, :n_cams] for t in t_idx]))
        depth = torch.from_numpy(np.stack([ep["depth"][t, :n_cams] for t in t_idx]).astype(np.float32)) * self.depth_unit
        body = torch.from_numpy(ep["body_id"][t_idx].astype(np.int64))  # (T,6,H,W)
        xpos = torch.from_numpy(ep["xpos"][t_idx]).float()  # (T,nb,3)
        xquat = torch.from_numpy(ep["xquat"][t_idx]).float()
        obs = torch.from_numpy(ep["obs"][t_idx]).float()
        tr = torch.as_tensor(self.train_cams)
        V = len(tr)
        images = rgb[:, tr].permute(0, 1, 4, 2, 3).contiguous()  # (T,V,3,H,W)
        depth_tr = depth[:, tr]  # (T,V,H,W)
        H, W = depth_tr.shape[-2:]
        K_tr, c2w_tr, w2c_tr = self.K[tr], self.c2w[tr], self.w2c[tr]

        xyz = lift_depth(depth_tr, K_tr[None].expand(T_WINDOW, -1, -1, -1), c2w_tr[None].expand(T_WINDOW, -1, -1, -1))
        valid_depth = depth_tr > 0
        bg = body == self.background_body
        safe_body = torch.where(bg, torch.zeros_like(body), body)
        nb = xpos.shape[1]

        def disp(a: int, b: int) -> torch.Tensor:
            pts = xyz[a].reshape(V, H * W, 3)
            d = rigid_body_displacement(
                pts,
                safe_body[a].reshape(V, H * W),
                xpos[a][None].expand(V, nb, 3),
                xquat[a][None].expand(V, nb, 4),
                xpos[b][None].expand(V, nb, 3),
                xquat[b][None].expand(V, nb, 4),
            ).reshape(V, H, W, 3)
            keep = (valid_depth[a] & ~bg[a])[..., None]
            return torch.where(keep, d, torch.zeros_like(d))

        all_disp = {(a, b): disp(a, b) for a in range(T_WINDOW) for b in range(T_WINDOW) if a != b}
        motion3d = torch.stack([all_disp[(a, b)].permute(0, 3, 1, 2) for a, b, _ in PAIRS])  # (P,V,3,H,W)
        motion_weight = torch.stack([valid_depth[a].float()[:, None] for a, _, _ in PAIRS])
        score = []
        for t in range(T_WINDOW):
            mags = torch.stack([all_disp[(t, b)].norm(dim=-1) for b in range(T_WINDOW) if b != t]).amax(0)
            score.append((mags / MOTION_SCORE_SCALE_M).clamp(0.0, 1.0)[:, None])
        sample = {
            "images": images,
            "K": K_tr.clone(),
            "w2c": w2c_tr.clone(),
            "c2w": c2w_tr.clone(),
            "depth": depth_tr[:, :, None].contiguous(),
            "motion3d": motion3d.contiguous(),
            "motion_weight": motion_weight.contiguous(),
            "motion_score": torch.stack(score).contiguous(),
            "probe_state": obs[:, :PROBE_DIM].contiguous(),
            "meta": {
                "task": self.task,
                "episode": ep_key,
                "t_indices": t_idx,
                "stride": int(stride),
                "dt_seconds": self.dt_seconds,
            },
        }
        if self.with_eval:
            ev = torch.as_tensor(self.eval_cams)
            sample["eval_images"] = rgb[:, ev].permute(0, 1, 4, 2, 3).contiguous()
            sample["eval_K"] = self.K[ev].clone()
            sample["eval_w2c"] = self.w2c[ev].clone()
            sample["eval_depth"] = depth[:, ev][:, :, None].contiguous()
        return sample
