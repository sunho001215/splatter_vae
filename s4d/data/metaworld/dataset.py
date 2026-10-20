"""Meta-World window dataset producing the batch contract, with ground-truth 3D motion.

Motion is derived from simulator state only: per-pixel body id + body poses give
exact rigid displacements. Segmentation is never used as a loss mask.

``objective="future"`` (variant action_future): a sample is a current time t and a stride k. The batch-contract fields
(images, depth, motion3d, motion_weight, motion_score) describe the TARGET window (t, t+k, t+2k) and serve only as
supervision; the encoder's inputs are separate: ``history_images``/``history_depth`` and ``history_motion_score``
(computed from the history frames alone) of the window (t-2k, t-k, t); ``actions`` = the recorded actions[t:t+2k]
(action[i] takes observation i to observation i+1, s4d/data/metaworld/collect.py), zero-padded to MAX_ACTIONS, with
``action_count`` = 2k. Anchors are kept only where every stride's history, targets and actions lie inside the episode.
``probe_state`` describes the history window (the frames the encoder sees).
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
from s4d.data.metaworld.heldout import HeldoutSets
from s4d.geometry import lift_depth, rigid_body_displacement

# Frame strides in simulator steps. RL observations are 2 simulator steps apart (action repeat 2),
# which lies inside the pretraining distribution; validation reports the RL spacing and the largest stride.
TRAIN_STRIDES = (2, 4, 6)
VAL_STRIDES = (2, 6)
PROBE_DIM = 11  # hand xyz, gripper opening, object-1 xyz, object-1 quaternion (wxyz)
OBJECTIVES = ("reconstruct", "future")
MAX_ACTIONS = 12  # 2k for the largest training stride k = 6


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
    manifest = manifest_dir / f"{path.stem}_seed{seed}.json"
    expected = {"file": path.name, "seed": seed, "train_ratio": train_ratio, "train": train, "validation": val}
    if manifest.is_file():
        if json.loads(manifest.read_text()) != expected:
            raise ValueError(f"split manifest differs from the deterministic split: {manifest}")
    else:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(expected, indent=1))
    return train, val


class MetaworldWindowDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        episodes: list[str],
        strides=TRAIN_STRIDES,
        with_eval: bool = False,
        seed: int = 0,
        heldout: str | Path | None = None,
        objective: str = "reconstruct",
    ):
        """One sample per (episode, start frame, stride). Start frames are kept only where *every* stride fits,
        so each sample's stride is exactly uniform over ``strides``.

        ``heldout``: near/trajectory held-out-set file (validation episodes only); with ``with_eval`` its views are
        added as ``<set>_images/_depth/_K/_w2c/_c2w`` for episodes it contains."""
        self.path = str(path)
        self.episodes = list(episodes)
        self.strides = tuple(int(s) for s in strides)
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        self.objective = objective
        if objective == "future" and 2 * max(self.strides) > MAX_ACTIONS:
            raise ValueError(f"future objective supports strides up to {MAX_ACTIONS // 2}")
        self.with_eval = with_eval
        self._file: h5py.File | None = None
        self.heldout = HeldoutSets(heldout) if (with_eval and heldout is not None) else None
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
            # only the future objective reads actions (fixtures of the original method may have none)
            self.action_dim = int(f["episodes"][self.episodes[0]]["action"].shape[1]) if objective == "future" else 0
            span = (T_WINDOW - 1) * max(self.strides)
            for ep in self.episodes:
                length = int(f["episodes"][ep].attrs["length"])
                if objective == "future":  # sample index = current time t: history t-2k >= 0, target t+2k <= length-1
                    for t in range(span, length - span):
                        self.samples.extend((ep, t, (s,)) for s in self.strides)
                else:
                    for t0 in range(length - span):
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
        ep_key, t_anchor, valid = self.samples[index]
        stride = valid[0]
        if self.objective == "reconstruct":
            return self._window(ep_key, [t_anchor + k * stride for k in range(T_WINDOW)], stride)
        # future: targets (t, t+k, t+2k) carry every supervision field; the history (t-2k, t-k, t) feeds the encoder
        target_idx = [t_anchor + k * stride for k in range(T_WINDOW)]
        history_idx = [t_anchor - (T_WINDOW - 1 - k) * stride for k in range(T_WINDOW)]
        sample = self._window(ep_key, target_idx, stride)
        history = self._window(ep_key, history_idx, stride)
        sample["history_images"] = history["images"]
        sample["history_depth"] = history["depth"]
        sample["history_motion_score"] = history["motion_score"]
        sample["probe_state"] = history["probe_state"]
        ep = self._h5()["episodes"][ep_key]
        count = 2 * stride
        seq = torch.from_numpy(ep["action"][t_anchor : t_anchor + count]).float()  # actions[t:t+2k]
        if seq.shape[0] != count:
            raise ValueError(f"{ep_key}: actions[{t_anchor}:{t_anchor + count}] incomplete")
        actions = torch.zeros(MAX_ACTIONS, seq.shape[1])
        actions[:count] = seq
        sample["actions"] = actions
        sample["action_count"] = torch.tensor(count, dtype=torch.int64)
        sample["meta"]["history_t_indices"] = history_idx
        if self.with_eval:
            for key in [k for k in history if k.endswith("_images") and k != "images"]:
                sample[f"history_{key}"] = history[key]
        return sample

    def _window(self, ep_key: str, t_idx: list[int], stride: int) -> dict:
        """Every contract field of the frames ``t_idx`` (the original method's sample)."""
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
            if self.heldout is not None and ep_key in self.heldout:
                sample.update({k: torch.from_numpy(v) for k, v in self.heldout.window(ep_key, t_idx).items()})
        return sample
