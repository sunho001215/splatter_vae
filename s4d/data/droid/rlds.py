"""CPU-only, deterministic RLDS access and independent episode/state verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

IMAGE_KEYS = ("exterior_image_1_left", "exterior_image_2_left")


def scene_suffix(value: str | bytes) -> str:
    """Discard storage bucket prefixes, retaining lab/success/date/recording identity."""
    text = value.decode() if isinstance(value, bytes) else str(value)
    text = text.replace("\\", "/").rstrip("/")
    for suffix in ("/trajectory.h5", "/recordings/MP4", "/recordings"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    parts = text.split("/")
    for marker in ("success", "failure"):
        if marker in parts:
            i = parts.index(marker)
            return "/".join(parts[max(0, i - 1) :])
    return text


def load_episode(root: str | Path, ordinal: int, split: str = "train") -> tuple[dict, dict]:
    """Absolute TFDS slices follow the canonical sequential-shard ordering.

    TensorFlow receives no visible GPU before any dataset operation. Entry points
    must have already enforced the shared allowed-GPU environment guard.
    """
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds

    root = Path(root).resolve(strict=True)
    builder_dir = root if (root / "dataset_info.json").is_file() else root / "1.0.1"
    builder = tfds.builder_from_directory(str(builder_dir))
    options = tf.data.Options()
    options.deterministic = True
    read_config = tfds.ReadConfig(
        options=options,
        try_autocache=False,
        interleave_cycle_length=1,
        interleave_block_length=1,
        num_parallel_calls_for_interleave_files=1,
        num_parallel_calls_for_decode=1,
        skip_prefetch=True,
    )
    dataset = builder.as_dataset(
        split=f"{split}[{ordinal}:{ordinal + 1}]",
        shuffle_files=False,
        read_config=read_config,
    )
    raw = next(iter(tfds.as_numpy(dataset)))
    steps = list(raw["steps"])
    if not steps:
        raise ValueError("matched RLDS episode has no steps")
    observation = steps[0]["observation"]
    keys = (*IMAGE_KEYS, "cartesian_position", "gripper_position", "joint_position")
    missing = set(keys) - set(observation)
    if missing:
        raise ValueError(f"matched episode misses fields {sorted(missing)}")
    arrays = {k: np.stack([s["observation"][k] for s in steps]) for k in keys}
    metadata = {k: v.decode() if isinstance(v, bytes) else str(v) for k, v in raw["episode_metadata"].items()}
    metadata.update(split=split, ordinal=int(ordinal), steps=len(steps))
    return arrays, metadata


def verify_path(metadata: dict, scene_path: str) -> None:
    expected = scene_suffix(scene_path)
    matches = [scene_suffix(metadata[k]) == expected for k in ("file_path", "recording_folderpath")]
    if not all(matches):
        raise ValueError(f"RLDS metadata mismatch: expected {expected!r}, got {metadata}")


def verify_states(raw: dict, flow_file) -> dict:
    """Verify each canonical camera timestep against the raw even RLDS step.

    Pose agreement proves the timestamp mapping independently of index ratio.
    PW gripper_positions is a joint angle; its conversion factor is .725.
    """
    from scipy.spatial.transform import Rotation

    xyz_errors, rotation_errors, gripper_errors, reverse_errors = [], [], [], []
    for key in flow_file:
        if ":" not in key:
            continue
        start, end = map(int, key.split(":"))
        indices = 2 * np.arange(start, end)
        if indices[-1] >= len(raw["cartesian_position"]):
            raise ValueError("canonical timeline exceeds matched RLDS episode")
        group = flow_file[key]
        expected = group["gripper_pose"][:]
        pose = raw["cartesian_position"][indices].astype(np.float64)
        xyz_errors.extend(np.linalg.norm(pose[:, :3] - expected[:, :3], axis=-1))
        r1 = Rotation.from_euler("xyz", pose[:, 3:])
        r2 = Rotation.from_quat(expected[:, 3:])
        rotation_errors.extend((r1.inv() * r2).magnitude())
        grip = raw["gripper_position"][indices].reshape(-1)
        stored = group["gripper_positions"][:] / 0.725
        gripper_errors.extend(np.abs(grip - stored))
        reverse_errors.extend(np.abs((1 - grip) - stored))
    if not xyz_errors:
        raise ValueError("no real clip states available for alignment")
    result = {
        "state_observations": len(xyz_errors),
        "xyz_median_m": float(np.median(xyz_errors)),
        "xyz_max_m": float(np.max(xyz_errors)),
        "euler_xyz_rotation_median_rad": float(np.median(rotation_errors)),
        "euler_xyz_rotation_max_rad": float(np.max(rotation_errors)),
        "gripper_direct_median": float(np.median(gripper_errors)),
        "gripper_direct_max": float(np.max(gripper_errors)),
        "gripper_reversed_median": float(np.median(reverse_errors)),
        "canonical_to_raw_stride": 2,
        "raw_gripper_sign": "0=open,1=closed",
        "pose_units": "meters,radians",
        "pw_gripper_joint_factor": 0.725,
    }
    if result["xyz_max_m"] > 0.005 or result["euler_xyz_rotation_max_rad"] > 0.02:
        raise ValueError(f"raw pose timeline/convention check failed: {result}")
    if result["gripper_direct_max"] > 0.02:
        raise ValueError(f"raw gripper sign/scaling check failed: {result}")
    return result


def save_matched_episode(path: Path, raw: dict, metadata: dict) -> None:
    """Save only newly read raw observations, never old derived teacher arrays."""
    np.savez_compressed(
        path, **{k: raw[k] for k in (*IMAGE_KEYS, "cartesian_position", "gripper_position", "joint_position")}
    )
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
