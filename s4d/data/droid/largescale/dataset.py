"""Map-style, lazily-read DROID x PointWorld x DROID-3D window dataset implementing the shared batch contract.

* Windows ``(i, i+s, i+2s)`` stay inside one PointWorld clip (track identities are per clip) and use only canonical
  frames whose external-camera depth frame was validated as the same raw step as the RLDS RGB (``frames.valid``).
* Nothing is decoded or materialized up front: ``__getitem__`` preads three JPEGs per camera from the released RLDS
  shards, three depth frames per camera and three rows of each camera's tracks from the PointWorld HDF5 files.
  Open files are bounded per worker (``HandlePool``), HDF5 chunk caches are small, and pickling drops all handles,
  so DataLoader workers and DDP ranks each open their own.
* Sharding/resume come from the trainer's ``ResumableDistributedSampler`` over this deterministic window list
  (sorted by episode, clip, start); per-item randomness is a pure function of ``(seed, index)``.
* Encoder inputs are only ``images`` (and the contract's motion score). Robot state, actions-free joint readings and
  all privileged geometry (tracks, gripper points, wrist pose/depth) are supervision fields; nothing here alters the
  encoder input contract. Wrist camera data are emitted as separate ``wrist_*`` fields because the wrist pose varies
  per frame while the contract's training views have one pose per window.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from s4d.data.contract import PAIRS, validate_batch
from s4d.data.droid.gripper import gripper_points
from s4d.data.droid.largescale import calibration as C
from s4d.data.droid.largescale.geometry import (
    invert,
    pose6_to_matrix,
    rebase_static_camera_points,
    scale_intrinsics,
    transform_points,
)
from s4d.data.droid.largescale.join import load_manifest, wrist_pose_w2c
from s4d.data.droid.largescale.pointworld_source import DEFAULT_LAYOUT, PointWorldReader, clip_bounds
from s4d.data.droid.largescale.rlds_index import RldsStepReader
from s4d.data.droid.pointworld import depth_test, splat_score, track_targets

NATIVE_WH = (320, 180)
EXT_ROLES = ("ext1", "ext2")


def stable_hash(*parts) -> int:
    return int.from_bytes(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest(), "little")


def episode_split(episode: str, validation_percent: float) -> str:
    """Episode-disjoint, order-independent split."""
    return "validation" if stable_hash("split", episode) % 10000 < validation_percent * 100 else "train"


def resize(array: np.ndarray, height: int, width: int, *, nearest: bool) -> np.ndarray:
    from PIL import Image

    if array.shape[:2] == (height, width):
        return array.copy()
    image = Image.fromarray(array)
    return np.asarray(
        image.resize((width, height), Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR)
    ).copy()


def corrected_ee_pose(cartesian6, q, dq) -> np.ndarray:
    """EE pose under DROID-3D joint offsets: ``FK(q+dq) @ inv(FK(q)) @ T_base_ee(cartesian)`` (keeps the controller's
    flange->tool offset; exact if ``cartesian_position`` is the controller's FK of ``q``)."""
    return C.franka_fk(np.asarray(q) + np.asarray(dq)) @ invert(C.franka_fk(q)) @ pose6_to_matrix(cartesian6)


class DroidLargeScaleDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        *,
        pointworld_root,
        rlds_index_dir,
        rlds_data_root,
        split: str = "train",
        validation_percent: float = 2.0,
        stride: int = 3,
        image_height: int = 144,
        image_width: int = 256,
        max_tracks: int = 4096,
        with_wrist: bool = True,
        gripper_offset_m: float = -0.019,
        seed: int = 0,
        max_open_files: int = 8,
        layout: dict = DEFAULT_LAYOUT,
        validate: bool = True,
        with_eval: bool = False,
    ):
        self.rows = sorted(load_manifest(manifest_path), key=lambda r: r["episode"])
        if split not in ("train", "validation", "all"):
            raise ValueError("split must be train, validation or all")
        self.stride, self.H, self.W = int(stride), int(image_height), int(image_width)
        self.max_tracks, self.with_wrist, self.seed = int(max_tracks), bool(with_wrist), int(seed)
        self.gripper_offset_m, self.validate = float(gripper_offset_m), bool(validate)
        self.pw = PointWorldReader(pointworld_root, layout, max_open=max_open_files)
        self.rlds = RldsStepReader(rlds_index_dir, rlds_data_root, max_open_files=max_open_files)
        self.windows: list[tuple[int, str, int]] = []
        for e, row in enumerate(self.rows):
            if split != "all" and episode_split(row["episode"], validation_percent) != split:
                continue
            valid = np.ones(len(row["timeline"]["canonical_timestamps"]), bool)
            for role in EXT_ROLES:
                valid &= np.asarray(row["frames"]["valid"][role], bool)
            for clip in sorted(row["clips"], key=clip_bounds):
                start, end = clip_bounds(clip)
                for i in range(start, end - 2 * self.stride):
                    if valid[[i, i + self.stride, i + 2 * self.stride]].all():
                        self.windows.append((e, clip, i))
        if not self.windows:
            raise ValueError(f"no valid windows in split {split!r}")

    def __len__(self) -> int:
        return len(self.windows)

    def close(self) -> None:
        self.pw.close()
        self.rlds.close()

    # ------------------------------------------------------------------------------------------------ helpers
    def _subsample(self, n: int, *key) -> np.ndarray:
        if n <= self.max_tracks:
            return np.arange(n)
        rng = np.random.default_rng(stable_hash(self.seed, *key))
        return np.sort(rng.choice(n, self.max_tracks, replace=False))

    def _ext_geometry(self, row: dict, role: str):
        cam = row["cameras"][role]
        K = np.asarray(cam["K"], np.float64)
        return K, np.asarray(cam["w2c"], np.float64), np.asarray(cam["w2c_pointworld"], np.float64)

    def _wrist_w2c(self, row: dict, state: dict) -> np.ndarray | None:
        wrist = row["cameras"].get("wrist")
        return None if wrist is None else wrist_pose_w2c(wrist, state)

    # ------------------------------------------------------------------------------------------------ item
    def __getitem__(self, index: int) -> dict:
        e, clip, i0 = self.windows[index]
        row = self.rows[e]
        episode = row["episode"]
        start, _ = clip_bounds(clip)
        canon = np.array([i0, i0 + self.stride, i0 + 2 * self.stride])
        rel = canon - start
        raw = row["timeline"]["stride"] * canon
        wrist_cal = row["cameras"].get("wrist") if self.with_wrist else None
        cams = (*EXT_ROLES, "wrist") if self.with_wrist else EXT_ROLES
        rl = row["rlds"]
        state = self.rlds.read(
            rl["shard"], rl["row"], raw, cameras=cams, numeric=("cartesian_position", "gripper_position", "joint_position")
        )
        flow = self.pw.flow(episode)[clip]
        # frame mapping re-check on the three frames (validated for the whole episode when the manifest was built)
        pw_pose = flow["gripper_pose"][rel.min() : rel.max() + 1][rel - rel.min()]
        if np.abs(pw_pose[:, :3] - state["cartesian_position"][:, :3]).max() > 2e-3:
            raise RuntimeError(f"{episode} {clip} {canon.tolist()}: RLDS/PointWorld robot state mismatch")

        if row["calibration"] == "droid3d":
            ee = corrected_ee_pose(state["cartesian_position"], state["joint_position"], row["droid3d"]["dq"])
        else:
            ee = pose6_to_matrix(state["cartesian_position"])
        local = gripper_points(np.zeros((3, 6)), state["gripper_position"].reshape(-1), self.gripper_offset_m)
        grip = transform_points(ee, local.astype(np.float64)).astype(np.float32)  # (T,G,3)

        images, depths_r, Ks, w2cs, scores, srcs, disps, weights, counts, gweights = ([] for _ in range(10))
        ext_points, ext_vis = [], []
        sample_ids = {}
        for role in EXT_ROLES:
            K_nat, w2c, w2c_pw = self._ext_geometry(row, role)
            serial = row["cameras"][role]["serial"]
            group = row["depth_groups"][role]
            depth_nat, _ = self.pw.depth_frames(episode, group, np.asarray(row["frames"]["depth_index"][role])[canon])
            cam = flow[f"camera_{serial}_ext"]
            lo, hi = int(rel.min()), int(rel.max()) + 1
            n_all = cam["scene_flows"].shape[1]
            keep = self._subsample(n_all, episode, clip, serial)
            sample_ids[role] = keep
            sel = rel - lo
            points = cam["scene_flows"][lo:hi][sel][:, keep].astype(np.float32)  # world positions, not displacements
            vis = (cam["scene_visibility"][lo:hi][sel] & cam["scene_depth_valid_mask"][lo:hi][sel])[:, keep]
            if row["calibration"] == "droid3d":
                points = rebase_static_camera_points(points, w2c_pw, w2c).astype(np.float32)
            gvis = np.ones((3, grip.shape[1]), bool)
            for t in range(3):
                vis[t] &= depth_test(points[t], K_nat, w2c, depth_nat[t], continuous=True)[0]
                gvis[t] &= depth_test(grip[t], K_nat, w2c, depth_nat[t], continuous=True)[0]
            K = scale_intrinsics(K_nat, NATIVE_WH, (self.W, self.H))
            d_r = np.stack([resize(d, self.H, self.W, nearest=True) for d in depth_nat])
            scores.append(splat_score(np.concatenate([points, grip], 1), np.concatenate([vis, gvis], 1), K, w2c, d_r))
            src, disp, weight = track_targets(points, vis)
            pad = self.max_tracks - len(keep)
            srcs.append(np.pad(src, ((0, 0), (0, pad), (0, 0))))
            disps.append(np.pad(disp, ((0, 0), (0, pad), (0, 0))))
            weights.append(np.pad(weight, ((0, 0), (0, pad))))
            counts.append(len(keep))
            gweights.append(gvis[[a for a, _, _ in PAIRS]].astype(np.float32))
            images.append(np.stack([resize(im, self.H, self.W, nearest=False) for im in state[role]]))
            depths_r.append(d_r)
            Ks.append(K)
            w2cs.append(w2c)
            ext_points.append(points)
            ext_vis.append(vis)

        w2c_arr = np.stack(w2cs)
        sample = {
            "images": torch.from_numpy(np.stack(images, 1).transpose(0, 1, 4, 2, 3).copy()),
            "K": torch.from_numpy(np.stack(Ks).astype(np.float32)),
            "w2c": torch.from_numpy(w2c_arr.astype(np.float32)),
            "c2w": torch.from_numpy(invert(w2c_arr).astype(np.float32)),
            "depth": torch.from_numpy(np.stack(depths_r, 1)[:, :, None].astype(np.float32)),
            "motion_score": torch.from_numpy(np.stack(scores, 1)),
            "track_src": torch.from_numpy(np.stack(srcs, 1)),
            "track_disp": torch.from_numpy(np.stack(disps, 1)),
            "track_weight": torch.from_numpy(np.stack(weights, 1)),
            "track_count": torch.tensor(counts, dtype=torch.int64),
            "gripper_points": torch.from_numpy(grip),
            "gripper_weight": torch.from_numpy(np.stack(gweights, 1)),
            "probe_state": torch.from_numpy(
                np.concatenate([state["cartesian_position"], state["gripper_position"]], -1).astype(np.float32)
            ),
            "robot_joint_position": torch.from_numpy(state["joint_position"].astype(np.float32)),
        }
        if self.with_wrist:
            sample.update(self._wrist_fields(row, episode, canon, state, wrist_cal, ext_points, ext_vis))
        sample["meta"] = {
            "task": "droid_largescale",
            "episode": episode,
            "clip": clip,
            "t_indices": tuple(int(x) for x in canon),
            "raw_indices": tuple(int(x) for x in raw),
            "stride": self.stride,
            "camera_serials": tuple(row["cameras"][r]["serial"] for r in EXT_ROLES),
            "calibration": row["calibration"],
        }
        if self.validate:
            validate_batch({k: v if k == "meta" else v.unsqueeze(0) for k, v in sample.items()})
        return sample

    def _wrist_fields(self, row, episode, canon, state, wrist_cal, ext_points, ext_vis) -> dict:
        H, W, P, V = self.H, self.W, len(PAIRS), len(EXT_ROLES)
        valid = np.zeros(3, bool)
        K_r = np.eye(3)
        w2c = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
        depth_r = np.zeros((3, H, W), np.float32)
        track_w = np.zeros((P, V, self.max_tracks), np.float32)
        group = row["depth_groups"].get("wrist")
        if wrist_cal is not None and group is not None:
            valid = np.asarray(row["frames"]["valid"]["wrist"], bool)[canon]
            K_nat = np.asarray(wrist_cal["K"], np.float64)
            K_r = scale_intrinsics(K_nat, NATIVE_WH, (W, H))
            w2c = self._wrist_w2c(row, state)
            depth_nat, _ = self.pw.depth_frames(episode, group, np.asarray(row["frames"]["depth_index"]["wrist"])[canon])
            if row.get("wrist_qa", {}).get("depth_rotation_deg") == 180:  # released wrist depth stored rotated (decided from data)
                depth_nat = np.ascontiguousarray(depth_nat[:, ::-1, ::-1])
            depth_nat[~valid] = 0
            depth_r = np.stack([resize(d, H, W, nearest=True) for d in depth_nat])
            for p, (a, _, _) in enumerate(PAIRS):
                if not valid[a]:
                    continue
                for v in range(V):
                    seen = depth_test(ext_points[v][a], K_nat, w2c[a], depth_nat[a], continuous=True)[0] & ext_vis[v][a]
                    track_w[p, v, : len(seen)] = seen
        else:
            K_r = scale_intrinsics(np.array([[100.0, 0, 160], [0, 100.0, 90], [0, 0, 1]]), NATIVE_WH, (W, H))
        images = np.stack([resize(im, H, W, nearest=False) for im in state["wrist"]]).transpose(0, 3, 1, 2)
        return {
            "wrist_images": torch.from_numpy(images.copy()),
            "wrist_depth": torch.from_numpy(depth_r[:, None].astype(np.float32)),
            "wrist_K": torch.from_numpy(K_r.astype(np.float32)),
            "wrist_w2c": torch.from_numpy(w2c.astype(np.float32)),
            "wrist_c2w": torch.from_numpy(invert(w2c).astype(np.float32)),
            "wrist_valid": torch.from_numpy(valid.copy()),
            "wrist_track_weight": torch.from_numpy(track_w),
        }


def worker_open_files() -> int:
    """Open file descriptors of this process (Linux), for bounded-handle tests and diagnostics."""
    return len(os.listdir("/proc/self/fd"))
