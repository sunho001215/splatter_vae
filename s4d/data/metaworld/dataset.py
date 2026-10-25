"""Meta-World window dataset producing the batch contract, with ground-truth 3D motion.

Motion is derived from simulator state only: per-pixel body id + body poses give
exact rigid displacements. Segmentation is never used as a loss mask.

``objective="future"`` (variant future_k2): a sample is a current time t and a FUTURE stride k2 drawn uniformly from
``future_strides`` (1..6); the encoder's HISTORY stride is fixed (``history_stride`` = 2 stored environment steps, the
registered RL observation spacing). The batch-contract fields (images, depth, motion3d, motion_weight, motion_score)
describe the TARGET window (t, t+k2, t+2*k2) and serve only as supervision; the encoder's inputs are separate:
``history_images``/``history_depth`` and ``history_motion_score`` (from the history frames alone) of the window
(t-4, t-2, t), identical for every k2; ``actions`` = the recorded actions[t:t+2*k2] (action[i] takes observation i to
observation i+1, s4d/data/metaworld/collect.py), zero-padded to MAX_ACTIONS, with ``future_stride`` = k2. Anchors are
kept only where the history and the longest future (and its actions) lie inside the episode, and each anchor yields one
sample per k2, so k2 is exactly uniform without episode-boundary bias. ``probe_state`` and ``meta["stride"]`` (the
spacing every encoder-side metric uses) describe the history; ``meta["future_stride"]`` the targets.
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
HISTORY_STRIDE = 2  # future objective: encoder frames (t-4, t-2, t), the RL observation spacing (action repeat 2)
FUTURE_STRIDES = (1, 2, 3, 4, 5, 6)
MAX_ACTIONS = 12  # 2 * the largest future stride


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
        history_stride: int = HISTORY_STRIDE,
        future_strides=FUTURE_STRIDES,
        eligibility_future: int | None = None,
        hardneg: dict | None = None,
    ):
        """One sample per (episode, start frame, stride). Start frames are kept only where *every* stride fits,
        so each sample's stride is exactly uniform over ``strides``. Under the future objective ``strides`` is unused:
        the encoder stride is ``history_stride`` (fixed at 2) and ``future_strides`` are the sampled target strides;
        anchors need room for ``eligibility_future`` (default: the largest of ``future_strides``), so loaders that
        evaluate one k2 each share the identical anchors (and encoder windows) when given the training maximum.

        ``heldout``: near/trajectory held-out-set file (validation episodes only); with ``with_eval`` its views are
        added as ``<set>_images/_depth/_K/_w2c/_c2w`` for episodes it contains.

        ``hardneg`` (encoder pilot hn; reconstruction objective only): {max_gap, min_hand_m, min_gripper} of
        s4d/data/hardneg.py; every sample then also carries one frame-disjoint window of its episode and stride that
        passed the robot-state rule (``hardneg_*`` contract fields; training cameras, images only). None (default):
        the sample is exactly the window itself and no random number is drawn."""
        self.path = str(path)
        self.episodes = list(episodes)
        self.strides = tuple(int(s) for s in strides)
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        self.objective = objective
        self.history_stride = int(history_stride)
        self.future_strides = tuple(int(k) for k in future_strides)
        if objective == "future":
            if self.history_stride != HISTORY_STRIDE:
                raise ValueError(f"the future objective's history stride is fixed at {HISTORY_STRIDE} (RL spacing)")
            if not self.future_strides or min(self.future_strides) < 1 or 2 * max(self.future_strides) > MAX_ACTIONS:
                raise ValueError(f"future strides must lie in 1..{MAX_ACTIONS // 2}, got {self.future_strides}")
            self.eligibility_future = int(eligibility_future or max(self.future_strides))
            if self.eligibility_future < max(self.future_strides) or 2 * self.eligibility_future > MAX_ACTIONS:
                raise ValueError("eligibility_future must cover every future stride and stay within MAX_ACTIONS")
            self.strides = ()  # never used under this objective: the history stride is fixed
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
            for ep in self.episodes:
                length = int(f["episodes"][ep].attrs["length"])
                if objective == "future":  # anchor t: history t-4 >= 0; longest target t+2*max(k2) <= length-1
                    first, end = (T_WINDOW - 1) * self.history_stride, length - (T_WINDOW - 1) * self.eligibility_future
                    for t in range(first, end):
                        self.samples.extend((ep, t, (k2,)) for k2 in self.future_strides)
                else:
                    span = (T_WINDOW - 1) * max(self.strides)
                    for t0 in range(length - span):
                        self.samples.extend((ep, t0, (s,)) for s in self.strides)
        if not self.samples:
            raise ValueError(f"no temporal windows in {self.path} for episodes {self.episodes[:3]}...")
        self.hardneg = None
        if hardneg is not None:
            if objective != "reconstruct":
                raise ValueError("hard negatives are defined for the reconstruction objective only")
            from s4d.data.hardneg import HardNegativeSampler, robot_state  # noqa: PLC0415

            with h5py.File(self.path, "r") as f:  # robot state only (proprio indices 0-3); no object state is read
                robot = {ep: robot_state(f["episodes"][ep]["obs"][:, :4], "metaworld") for ep in self.episodes}
            self.hardneg = HardNegativeSampler(robot, **hardneg)

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
        if self.objective == "reconstruct":
            stride = valid[0]
            sample = self._window(ep_key, [t_anchor + k * stride for k in range(T_WINDOW)], stride)
            if self.hardneg is not None:
                sample.update(self._hardneg(ep_key, t_anchor, stride, sample["images"]))
            return sample
        # future: targets (t, t+k2, t+2*k2) carry every supervision field; the fixed history (t-4, t-2, t) feeds the
        # encoder. Encoder-side metadata (meta["stride"], probe_state) is the history's; the future stride is separate.
        future_stride, h = valid[0], self.history_stride
        target_idx = [t_anchor + k * future_stride for k in range(T_WINDOW)]
        history_idx = [t_anchor - (T_WINDOW - 1 - k) * h for k in range(T_WINDOW)]
        sample = self._window(ep_key, target_idx, h)
        history = self._window(ep_key, history_idx, h)
        sample["history_images"] = history["images"]
        sample["history_depth"] = history["depth"]
        sample["history_motion_score"] = history["motion_score"]
        sample["probe_state"] = history["probe_state"]
        ep = self._h5()["episodes"][ep_key]
        count = 2 * future_stride
        seq = torch.from_numpy(ep["action"][t_anchor : t_anchor + count]).float()  # actions[t:t+2*k2]
        if seq.shape[0] != count:
            raise ValueError(f"{ep_key}: actions[{t_anchor}:{t_anchor + count}] incomplete")
        actions = torch.zeros(MAX_ACTIONS, seq.shape[1])
        actions[:count] = seq
        sample["actions"] = actions
        sample["future_stride"] = torch.tensor(future_stride, dtype=torch.int64)
        sample["meta"].update(history_t_indices=history_idx, history_stride=h, future_stride=future_stride)
        if self.with_eval:
            for key in [k for k in history if k.endswith("_images") and k != "images"]:
                sample[f"history_{key}"] = history[key]
        return sample

    def _hardneg(self, ep_key: str, t0: int, stride: int, images: torch.Tensor) -> dict:
        """The hard-negative window of sample (ep_key, t0, stride): images of the training cameras only."""
        t_neg = self.hardneg.sample(ep_key, t0, stride)
        if t_neg is None:
            return {"hardneg_images": torch.zeros_like(images), "hardneg_valid": torch.tensor(False),
                    "hardneg_t0": torch.tensor(-1, dtype=torch.int64),
                    "hardneg_hand_disp": torch.tensor(-1.0, dtype=torch.float32)}
        ep = self._h5()["episodes"][ep_key]
        V = len(self.train_cams)
        rgb = torch.from_numpy(np.stack([ep["rgb"][t_neg + k * stride, :V] for k in range(T_WINDOW)]))
        return {"hardneg_images": rgb.permute(0, 1, 4, 2, 3).contiguous(), "hardneg_valid": torch.tensor(True),
                "hardneg_t0": torch.tensor(t_neg, dtype=torch.int64),
                "hardneg_hand_disp": torch.tensor(self.hardneg.hand_displacement(ep_key, t0, t_neg, stride),
                                                  dtype=torch.float32)}

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
