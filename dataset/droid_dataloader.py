"""
DROID Dataset Dataloader for SplatterVAE Training
===================================================

Produces the same batch format as RoboSuiteMultiViewTemporalHDF5Dataset:
    image_i_t, image_j_t, image_i_t1, image_j_t1, T_ij, K_i, K_j

Two-step pipeline:
    1. Preprocess: RLDS episodes -> per-episode frame directories (one-time)
    2. Train: PyTorch Dataset reads frame directories (fast random access)

Data sources:
    - DROID RLDS v1.0.1:  gs://gresearch/robotics/droid/1.0.1
    - cam2cam_extrinsics.json from https://huggingface.co/KarlP/droid
    - intrinsics.json from https://huggingface.co/KarlP/droid
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info


# ============================================================================
# Utility functions
# ============================================================================

def image_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    """uint8 RGB [0,255] -> float32 [-1,1], shape (3,H,W)."""
    img = img_rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(img).permute(2, 0, 1)


def invert_4x4(m: np.ndarray) -> np.ndarray:
    """Invert a rigid-body 4x4 transform (bottom row = [0,0,0,1])."""
    R, t = m[:3, :3], m[:3, 3:4]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = R.T
    out[:3, 3] = (-R.T @ t)[:, 0]
    return out


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize (H,W,3) image to (size,size,3)."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def scale_intrinsics(K: np.ndarray, oh: int, ow: int, nh: int, nw: int) -> np.ndarray:
    """Scale 3x3 intrinsics matrix for a resized image."""
    K = K.copy()
    K[0, 0] *= nw / ow;  K[0, 2] *= nw / ow   # fx, cx
    K[1, 1] *= nh / oh;  K[1, 2] *= nh / oh   # fy, cy
    return K


# ============================================================================
# Parsing DROID calibration JSONs
# ============================================================================

def parse_cam2cam_entry(entry: dict) -> Optional[dict]:
    """
    Parse one episode entry from cam2cam_extrinsics.json.

    Actual format (from inspecting the file):
        {
            "relative_path": "Lab/status/date/timestamp",
            "left_cam": {
                "pose": [[4 floats], [4 floats], [4 floats], [4 floats]],  # 4x4 c2w matrix
                "focal": float,
                "principal_point": [cx, cy]
            },
            "right_cam": { ... same structure ... },
            "metric_type": "num_matches",
            "quality_metric": int
        }

    Returns dict with K_left, K_right, T_left_to_right, relative_path,
    or None if parsing fails.
    """
    if "left_cam" not in entry or "right_cam" not in entry:
        return None

    left, right = entry["left_cam"], entry["right_cam"]

    # Parse 4x4 pose matrices (camera-to-world)
    try:
        pose_left = np.array(left["pose"], dtype=np.float32)    # (4,4)
        pose_right = np.array(right["pose"], dtype=np.float32)  # (4,4)
    except (ValueError, KeyError):
        return None
    if pose_left.shape != (4, 4) or pose_right.shape != (4, 4):
        return None

    # Build 3x3 intrinsics from focal + principal_point
    # Note: focal is a single scalar (fx == fy for ZED cameras)
    # These are at SVO native resolution (1280x720)
    fl, pl = float(left["focal"]), left["principal_point"]
    fr, pr = float(right["focal"]), right["principal_point"]
    K_left  = np.array([[fl, 0, pl[0]], [0, fl, pl[1]], [0, 0, 1]], dtype=np.float32)
    K_right = np.array([[fr, 0, pr[0]], [0, fr, pr[1]], [0, 0, 1]], dtype=np.float32)

    # Relative transform: cam_left -> cam_right
    T_left_to_right = invert_4x4(pose_right) @ pose_left

    return {
        "K_left": K_left,  "K_right": K_right,
        "pose_left": pose_left,  "pose_right": pose_right,
        "T_left_to_right": T_left_to_right,
        "relative_path": entry.get("relative_path", ""),
        "quality_metric": entry.get("quality_metric", 0),
    }


# ============================================================================
# Preprocessing: RLDS -> frame directories (run once)
# ============================================================================

class DROIDPreprocessor:
    """
    Convert DROID RLDS episodes into per-episode frame directories
    for fast random-access during training.

    Output layout:
        <output_dir>/<episode_id>/
            metadata.json
            cam_left/00000.png, 00001.png, ...
            cam_right/00000.png, 00001.png, ...
    """

    def __init__(self, rlds_dir: str, output_dir: str,
                 cam2cam_json: str, intrinsics_json: str,
                 img_size: int = 128, min_quality: int = 0):
        self.rlds_dir = rlds_dir
        self.output_dir = output_dir
        self.img_size = img_size
        self.min_quality = min_quality

        print("[DROIDPreprocessor] Loading calibration JSONs...")
        with open(cam2cam_json) as f:
            cam2cam_raw = json.load(f)
        # intrinsics.json is no longer needed — cam2cam has focal + principal_point

        # Parse cam2cam and build relative_path -> (episode_id, calibration) lookup
        self.path_to_calib: Dict[str, Tuple[str, dict]] = {}
        parsed_ok = 0
        for ep_id, entry in cam2cam_raw.items():
            parsed = parse_cam2cam_entry(entry)
            if parsed is None or parsed["quality_metric"] < self.min_quality:
                continue
            rel_path = parsed["relative_path"]
            if rel_path:
                self.path_to_calib[rel_path] = (ep_id, parsed)
                parsed_ok += 1

        print(f"  cam2cam: {len(cam2cam_raw)} raw -> {parsed_ok} usable entries")

    def _match_rlds_path(self, file_path: str) -> Optional[Tuple[str, dict]]:
        """
        Match an RLDS file_path to a cam2cam calibration entry.

        RLDS path example:
            gs://xembodiment_data/r2d2/r2d2-data-full/AUTOLab/success/.../trajectory.h5
        relative_path in cam2cam example:
            AUTOLab/success/2023-11-11/Sat_Nov_11_16:50:39_2023

        Strategy: extract last 4 path components (Lab/status/date/timestamp).
        """
        dir_path = file_path.rsplit("/", 1)[0] if "/" in file_path else file_path
        parts = dir_path.rstrip("/").split("/")

        # Try last-4 components: Lab/status/date/timestamp
        if len(parts) >= 4:
            key = "/".join(parts[-4:])
            if key in self.path_to_calib:
                return self.path_to_calib[key]
        return None

    def run(self, max_episodes: Optional[int] = None):
        """Iterate RLDS episodes, decode frames, save to disk."""
        import tensorflow_datasets as tfds

        os.makedirs(self.output_dir, exist_ok=True)
        builder = tfds.builder_from_directory(self.rlds_dir)
        ds = builder.as_dataset(split="train")

        processed, skipped = 0, 0
        skip_reasons = {"no_path_match": 0, "too_short": 0}

        # SVO native resolution -> RLDS resolution -> target resolution
        svo_h, svo_w = 720, 1280
        rlds_h, rlds_w = 180, 320

        for episode in ds:
            file_path = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")

            # Match to calibration
            match = self._match_rlds_path(file_path)
            if match is None:
                skip_reasons["no_path_match"] += 1
                skipped += 1
                continue
            ep_id, calib = match

            # Skip if already processed (allows resuming)
            ep_dir = os.path.join(self.output_dir, ep_id)
            meta_path = os.path.join(ep_dir, "metadata.json")
            if os.path.isfile(meta_path):
                processed += 1
                if max_episodes and processed >= max_episodes:
                    break
                continue

            # Scale intrinsics: SVO(1280x720) -> RLDS(320x180) -> target(img_size x img_size)
            K_left  = scale_intrinsics(calib["K_left"],  svo_h, svo_w, rlds_h, rlds_w)
            K_right = scale_intrinsics(calib["K_right"], svo_h, svo_w, rlds_h, rlds_w)
            K_left  = scale_intrinsics(K_left,  rlds_h, rlds_w, self.img_size, self.img_size)
            K_right = scale_intrinsics(K_right, rlds_h, rlds_w, self.img_size, self.img_size)

            # Decode and save frames
            os.makedirs(os.path.join(ep_dir, "cam_left"), exist_ok=True)
            os.makedirs(os.path.join(ep_dir, "cam_right"), exist_ok=True)

            frame_idx = 0
            for step in episode["steps"]:
                obs = step["observation"]
                img_l = resize_image(obs["exterior_image_1_left"].numpy(), self.img_size)
                img_r = resize_image(obs["exterior_image_2_left"].numpy(), self.img_size)
                cv2.imwrite(os.path.join(ep_dir, "cam_left",  f"{frame_idx:05d}.png"),
                            cv2.cvtColor(img_l, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(ep_dir, "cam_right", f"{frame_idx:05d}.png"),
                            cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))
                frame_idx += 1

            if frame_idx < 2:
                skip_reasons["too_short"] += 1
                skipped += 1
                continue

            # Camera poses and relative transform
            c2w_left, c2w_right = calib["pose_left"], calib["pose_right"]
            w2c_left, w2c_right = invert_4x4(c2w_left), invert_4x4(c2w_right)

            metadata = {
                "episode_id": ep_id,
                "num_frames": frame_idx,
                "cameras": {
                    "cam_left":  {"K": K_left.tolist(),  "w2c": w2c_left.tolist(),  "c2w": c2w_left.tolist()},
                    "cam_right": {"K": K_right.tolist(), "w2c": w2c_right.tolist(), "c2w": c2w_right.tolist()},
                },
                "T_left_to_right": calib["T_left_to_right"].tolist(),
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            processed += 1
            if processed % 500 == 0:
                print(f"  Processed {processed} (skipped {skipped})")
            if max_episodes and processed >= max_episodes:
                break

        print(f"[DROIDPreprocessor] Done. Processed={processed}, Skipped={skipped}")
        print(f"  Skip reasons: {skip_reasons}")


# ============================================================================
# PyTorch Dataset
# ============================================================================

class DROIDMultiViewTemporalDataset(Dataset):
    """
    DROID dataset for SplatterVAE training.
    Reads pre-processed frame directories and produces batches matching:
        image_i_t, image_j_t, image_i_t1, image_j_t1, T_ij, K_i, K_j
    """

    def __init__(self, dataset_dir: str, episode_ids: List[str],
                 img_size: int = 128, min_time_gap: int = 10,
                 max_frames_per_episode: Optional[int] = None, seed: int = 0):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.min_time_gap = min_time_gap
        self.rng = random.Random(seed)

        self.episode_meta: Dict[str, Dict[str, Any]] = {}
        self.episode_lengths: Dict[str, int] = {}
        self.samples: List[Tuple[str, int]] = []

        for ep_id in episode_ids:
            meta_path = os.path.join(dataset_dir, ep_id, "metadata.json")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)

            nf = min(int(meta["num_frames"]),
                     max_frames_per_episode or float("inf"))
            if nf < 2:
                continue

            cl, cr = meta["cameras"]["cam_left"], meta["cameras"]["cam_right"]
            self.episode_meta[ep_id] = {
                "K_left":    np.array(cl["K"],   dtype=np.float32),
                "K_right":   np.array(cr["K"],   dtype=np.float32),
                "w2c_left":  np.array(cl["w2c"], dtype=np.float32),
                "c2w_left":  np.array(cl["c2w"], dtype=np.float32),
                "w2c_right": np.array(cr["w2c"], dtype=np.float32),
                "c2w_right": np.array(cr["c2w"], dtype=np.float32),
            }
            self.episode_lengths[ep_id] = nf
            self.samples.extend((ep_id, t) for t in range(nf))

        print(f"[DROIDDataset] {len(self.episode_meta)} episodes, {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_t1(self, T: int, t: int) -> int:
        """Sample t1 with |t1-t| >= min_time_gap when possible."""
        far = [i for i in range(T) if abs(i - t) >= self.min_time_gap]
        if far:
            return self.rng.choice(far)
        others = [i for i in range(T) if i != t]
        return self.rng.choice(others) if others else t

    def _load_image(self, ep_id: str, cam: str, idx: int) -> np.ndarray:
        path = os.path.join(self.dataset_dir, ep_id, cam, f"{idx:05d}.png")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_id, t = self.samples[idx]
        T = self.episode_lengths[ep_id]
        meta = self.episode_meta[ep_id]
        t1 = self._sample_t1(T, t)

        # Randomly swap which camera is view_i vs view_j
        if self.rng.random() < 0.5:
            ci, cj = "cam_left", "cam_right"
            K_i, K_j = meta["K_left"], meta["K_right"]
            w2c_j, c2w_i = meta["w2c_right"], meta["c2w_left"]
        else:
            ci, cj = "cam_right", "cam_left"
            K_i, K_j = meta["K_right"], meta["K_left"]
            w2c_j, c2w_i = meta["w2c_left"], meta["c2w_right"]

        # T_ij: cam_i -> cam_j (same as robosuite dataloader)
        T_ij = (w2c_j @ c2w_i).astype(np.float32)

        return {
            "image_i_t":  image_to_tensor(self._load_image(ep_id, ci, t)),
            "image_j_t":  image_to_tensor(self._load_image(ep_id, cj, t)),
            "image_i_t1": image_to_tensor(self._load_image(ep_id, ci, t1)),
            "image_j_t1": image_to_tensor(self._load_image(ep_id, cj, t1)),
            "T_ij": torch.from_numpy(T_ij),
            "K_i":  torch.from_numpy(K_i.copy()),
            "K_j":  torch.from_numpy(K_j.copy()),
            "episode_id": ep_id, "t": t, "t1": t1,
            "view_i": ci, "view_j": cj,
        }


# ============================================================================
# DataLoader builder (drop-in for build_train_valid_loaders_robosuite)
# ============================================================================

def _worker_init_fn(worker_id: int):
    info = get_worker_info()
    if info is not None:
        info.dataset.rng = random.Random(info.seed)


def build_train_valid_loaders_droid(
    dataset_dir: str, batch_size: int = 64, num_workers: int = 8,
    pin_memory: bool = True, train_ratio: float = 0.96, seed: int = 42,
    img_size: int = 128, min_time_gap: int = 25,
    max_frames_per_episode: Optional[int] = None,
    max_episodes: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/valid loaders with same batch format as robosuite version."""
    all_ids = sorted(d for d in os.listdir(dataset_dir)
                     if os.path.isfile(os.path.join(dataset_dir, d, "metadata.json")))
    if max_episodes:
        all_ids = all_ids[:max_episodes]

    rng = random.Random(seed)
    rng.shuffle(all_ids)
    n = len(all_ids)
    n_train = max(1, min(int(n * train_ratio), n - 1)) if n > 1 else n
    train_ids, valid_ids = all_ids[:n_train], all_ids[n_train:]
    print(f"[DROID] Total={n}, Train={len(train_ids)}, Valid={len(valid_ids)}")

    mk = lambda ids, s: DROIDMultiViewTemporalDataset(
        dataset_dir, ids, img_size, min_time_gap, max_frames_per_episode, s)
        
    train_loader = DataLoader(mk(train_ids, seed), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=True, worker_init_fn=_worker_init_fn)
    valid_loader = DataLoader(mk(valid_ids, seed + 999), batch_size=batch_size, shuffle=True,
                              num_workers=max(1, num_workers // 2), pin_memory=pin_memory,
                              drop_last=True, worker_init_fn=_worker_init_fn)
    return train_loader, valid_loader


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DROID Preprocessing for SplatterVAE")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("preprocess")
    p.add_argument("--rlds_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cam2cam_json", required=True, help="cam2cam_extrinsics.json")
    p.add_argument("--intrinsics_json", default=None, help="(unused, kept for compat)")
    p.add_argument("--id2path_json", default=None, help="(unused, kept for compat)")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--min_quality", type=int, default=0)

    args = parser.parse_args()
    if args.command == "preprocess":
        DROIDPreprocessor(
            rlds_dir=args.rlds_dir, output_dir=args.output_dir,
            cam2cam_json=args.cam2cam_json,
            intrinsics_json=args.intrinsics_json or "",
            img_size=args.img_size, min_quality=args.min_quality,
        ).run(max_episodes=args.max_episodes)
    else:
        parser.print_help()