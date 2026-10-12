"""Convert the inspected real episode, without importing upstream teacher code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from s4d.data import DROID_CACHE_ROOT, writable_path
from s4d.data.droid.gripper import calibrate_offset, gripper_depth_residuals, gripper_points
from s4d.data.droid.pointworld import (
    PAIR_TIMES,
    clip_windows,
    depth_test,
    nearest_timestamps,
    resize_intrinsics,
    splat_score,
    track_targets,
)
from s4d.data.droid.rlds import IMAGE_KEYS, load_episode, load_saved_episode, verify_path, verify_states

RESOLUTIONS = {"scratch": (144, 256), "dinov2": (140, 252)}
EPISODE = "AUTOLab+0d4edc83+2023-10-21-19h-07m-04s"
REVISION = "dd9aaeec94bb14e27ab6b16b6e4aa0dbcf3ef56f"
FORMAT = "splatter4d_pointworld_sample_v2_tracks"


def sample_paths(root: Path) -> dict:
    return {
        "flow": root / "flow_episode" / f"{EPISODE}_flows.h5",
        "depth": root / "depth_episode" / f"{EPISODE}_depth.h5",
        "cameras": root / "camera_episode" / f"{EPISODE}_cameras.json",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_geometry(flow, depth, cameras: dict) -> dict:
    clips = sorted((key for key in flow if ":" in key), key=lambda key: tuple(map(int, key.split(":"))))
    serials = [str(flow.attrs[f"ext{i}_cam_serial"]) for i in (1, 2)]
    K, w2c, depths, indices, errors = [], [], [], [], []
    timestamps = np.array(json.loads(flow.attrs["canonical_timestamps"]), dtype=np.int64)
    for serial in serials:
        group = flow[clips[0]][f"camera_{serial}_ext"]
        intrinsic, extrinsic = group["intrinsic"][:], group["extrinsic"][:]
        if not np.allclose(extrinsic, cameras[serial]["optimized_extrinsics"], atol=1e-6):
            raise ValueError("camera JSON and flow extrinsics disagree")
        for key in clips:
            clip_camera = flow[key][f"camera_{serial}_ext"]
            if not np.allclose(clip_camera["intrinsic"][:], intrinsic, atol=1e-6):
                raise ValueError("sample intrinsics unexpectedly change between clips")
            if not np.allclose(clip_camera["extrinsic"][:], extrinsic, atol=1e-6):
                raise ValueError("sample static extrinsics unexpectedly change between clips")
        stamps = depth[f"{serial}+ext"]["timestamps"][:]
        idx = nearest_timestamps(stamps, timestamps)
        indices.append(idx)
        errors.append(np.abs(stamps[idx] - timestamps))
        # Wrist is intentionally never loaded.
        all_depth = depth[f"{serial}+ext"]["depth"][:].astype(np.float32) / 1000
        depths.append(all_depth[idx])
        K.append(intrinsic)
        w2c.append(extrinsic)
    return {
        "clips": clips,
        "serials": serials,
        "K": np.array(K, np.float32),
        "w2c": np.array(w2c, np.float32),
        "depths": np.stack(depths, axis=1),
        "timestamps": timestamps,
        "depth_indices": np.stack(indices, axis=1),
        "timestamp_errors_ms": np.stack(errors, axis=1),
    }


def resize_image(array: np.ndarray, height: int, width: int, *, depth: bool = False) -> np.ndarray:
    image = Image.fromarray(array)
    method = Image.Resampling.NEAREST if depth else Image.Resampling.BILINEAR
    return np.asarray(image.resize((width, height), method)).copy()


def convert_sample(
    sample_root: Path, output: Path, *, rlds_root: Path | None = None, raw_episode: tuple[Path, Path] | None = None
) -> dict:
    """Only real matched RLDS observations are admitted; never PW pose substitution.

    The episode is reread from ``rlds_root`` or taken from a saved ``raw_episode`` (npz, metadata json) that an
    earlier reread wrote; either way it must match the clip's scene path and per-step raw states.
    """
    if (rlds_root is None) == (raw_episode is None):
        raise ValueError("give exactly one of rlds_root or raw_episode")
    output = writable_path(output, DROID_CACHE_ROOT).resolve(strict=True)
    sample_root = sample_root.resolve(strict=True)
    paths = sample_paths(sample_root)
    if any(output.iterdir()):
        raise FileExistsError(f"conversion output must be empty: {output}")
    license_path = sample_root / "metadata_download/LICENSE.pdf"
    (output / "LICENSE.pdf").write_bytes(license_path.read_bytes())
    with h5py.File(paths["flow"], "r") as flow, h5py.File(paths["depth"], "r") as depth:
        cameras = json.loads(paths["cameras"].read_text())
        hint = json.loads((sample_root / "rlds_match_hint.json").read_text())
        if rlds_root is not None:
            raw, metadata = load_episode(rlds_root, hint["rlds_ordinal"], hint["rlds_split"])
        else:
            raw, metadata = load_saved_episode(*raw_episode)
            if (int(metadata["ordinal"]), metadata["split"]) != (int(hint["rlds_ordinal"]), hint["rlds_split"]):
                raise ValueError("saved matched episode is not the hinted RLDS episode")
        verify_path(metadata, str(flow.attrs["scene_path"]))
        alignment = verify_states(raw, flow)
        np.savez_compressed(output / "matched_raw_external.npz", **raw)
        (output / "matched_raw_metadata.json").write_text(json.dumps(metadata, indent=2))
        geom = episode_geometry(flow, depth, cameras)
        # Keep the latest clip temporally disjoint. There is only one episode,
        # so this is an intra-episode sample split, never a corpus evaluation.
        validation_clip = geom["clips"][-1]
        validation_start = int(validation_clip.split(":")[0])
        train_clips = [key for key in geom["clips"][:-1] if int(key.split(":")[1]) <= validation_start]
        if len(train_clips) != len(geom["clips"]) - 1:
            raise ValueError("last-clip validation overlaps a training clip")
        fit_indices = np.unique(np.concatenate([np.arange(*map(int, key.split(":"))) for key in train_clips]))
        fit_indices = fit_indices[::2]
        raw_indices = 2 * fit_indices
        calibration = calibrate_offset(
            raw["cartesian_position"][raw_indices],
            raw["gripper_position"][raw_indices],
            geom["K"],
            geom["w2c"],
            geom["depths"][fit_indices],
        )
        calibration["fit_canonical_indices"] = fit_indices.tolist()
        all_gripper = gripper_points(raw["cartesian_position"][::2], raw["gripper_position"][::2], calibration["offset_m"])
        heldout = np.arange(*map(int, validation_clip.split(":")))
        heldout_errors = gripper_depth_residuals(all_gripper[heldout], geom["K"], geom["w2c"], geom["depths"][heldout])
        calibration["heldout_median_residual_m"] = float(np.median(heldout_errors)) if len(heldout_errors) else None
        calibration["heldout_point_observations"] = len(heldout_errors)
        calibration["heldout_canonical_indices"] = heldout.tolist()
        # Every window pads each camera's tracks to the largest per-camera track count of the episode.
        capacity = max(
            flow[key][f"camera_{serial}_ext"]["scene_flows"].shape[1] for key in geom["clips"] for serial in geom["serials"]
        )
        windows = []
        for key in geom["clips"]:
            for indices in clip_windows(key):
                windows.append(
                    {
                        "clip": key,
                        "canonical_indices": list(indices),
                        "raw_indices": [2 * i for i in indices],
                        "split": "validation" if key == validation_clip else "train",
                    }
                )
        manifest = {
            "format": FORMAT,
            "episode": EPISODE,
            "dataset": "nvidia/PointWorld-DROID",
            "revision": REVISION,
            "license": "NVIDIA License; non-commercial use except NVIDIA and affiliates",
            "license_source": str(license_path),
            "license_sha256": sha256(license_path),
            "license_cache_copy": "LICENSE.pdf",
            "source_sha256": {key: sha256(path) for key, path in paths.items()},
            "matched_rlds": metadata,
            "matching": {"attempted": 1, "matched": 1, "rate": 1.0, "scope": "selected real sample only"},
            "raw_state_alignment": alignment,
            "gripper_calibration": calibration,
            "camera_serials": geom["serials"],
            "windows": windows,
            "resolutions": {key: list(value) for key, value in RESOLUTIONS.items()},
            "split_policy": "last clip validation, temporally disjoint, same episode; not generalization evidence",
            "intrinsics_convention": (
                "add .5 to native integer principal point then scale rows independently; continuous centers"
            ),
            "depth_resize": "nearest neighbor, zero invalid, metric camera-z",
            "motion_target": (
                "per-camera 3D track arrays, never splatted: source position, displacement and weight per pair; "
                "weight 1 where PointWorld visibility, depth-valid mask and the native 2 cm surface test pass at the "
                "source time and both positions are finite; zero padding after track_count"
            ),
            "gripper_target": (
                "gripper FK points per time; per-camera weight 1 where the native 2 cm surface test passes at the "
                "source time"
            ),
            "track_capacity": capacity,
            "motion_pairs": [[0, 1, 0], [1, 2, 1], [0, 2, 0]],
            "motion_score": (
                "max norm over all three point displacement pairs / .03, clipped, splatted with nearest-z on each "
                "time (scene and gripper points)"
            ),
            "depth_timestamp_indices": geom["depth_indices"].tolist(),
            "depth_timestamp_median_error_ms": np.median(geom["timestamp_errors_ms"], axis=0).tolist(),
            "depth_timestamp_max_error_ms": geom["timestamp_errors_ms"].max(axis=0).tolist(),
        }
        # Reuse each clip's trajectories across both resized variants. Random
        # access does not retain open source downloads in training workers.
        outputs = {}
        for backbone, (height, width) in RESOLUTIONS.items():
            path = output / f"{backbone}.h5"
            outputs[backbone] = h5py.File(path, "x")
            outputs[backbone].attrs["height"] = height
            outputs[backbone].attrs["width"] = width
        try:
            for index, window in enumerate(windows):
                key = window["clip"]
                start = int(key.split(":")[0])
                canon = np.array(window["canonical_indices"])
                relative = canon - start
                raw_idx = canon * 2
                native_depth = geom["depths"][canon]
                for backbone, (height, width) in RESOLUTIONS.items():
                    K = np.stack([resize_intrinsics(k, width, height) for k in geom["K"]])
                    images = np.stack(
                        [np.stack([resize_image(raw[k][t], height, width) for k in IMAGE_KEYS]) for t in raw_idx]
                    ).transpose(0, 1, 4, 2, 3)
                    resized_depth = np.stack(
                        [np.stack([resize_image(d, height, width, depth=True) for d in frame]) for frame in native_depth]
                    )
                    gripper = all_gripper[canon]
                    srcs, disps, weights, counts, grip_weights, scores = [], [], [], [], [], []
                    for view, serial in enumerate(geom["serials"]):
                        camera = flow[key][f"camera_{serial}_ext"]
                        points = camera["scene_flows"][relative].astype(np.float32)
                        visible = camera["scene_visibility"][relative] & camera["scene_depth_valid_mask"][relative]
                        grip_visible = np.ones((3, len(gripper[0])), bool)
                        for t in range(3):
                            visible[t] &= depth_test(points[t], geom["K"][view], geom["w2c"][view], native_depth[t, view])[0]
                            grip_visible[t] &= depth_test(
                                gripper[t], geom["K"][view], geom["w2c"][view], native_depth[t, view]
                            )[0]
                        scores.append(
                            splat_score(
                                np.concatenate([points, gripper], axis=1),
                                np.concatenate([visible, grip_visible], axis=1),
                                K[view],
                                geom["w2c"][view],
                                resized_depth[:, view],
                            )
                        )
                        src, disp, weight = track_targets(points, visible)
                        pad = ((0, 0), (0, capacity - points.shape[1]))
                        srcs.append(np.pad(src, (*pad, (0, 0))))
                        disps.append(np.pad(disp, (*pad, (0, 0))))
                        weights.append(np.pad(weight, pad))
                        counts.append(points.shape[1])
                        grip_weights.append(grip_visible[[a for a, _ in PAIR_TIMES]].astype(np.float32))
                    group = outputs[backbone].create_group(f"windows/{index:05d}")
                    tensors = {
                        "images": images,
                        "depth": resized_depth[:, :, None],
                        "K": K,
                        "w2c": geom["w2c"],
                        "c2w": np.linalg.inv(geom["w2c"]).astype(np.float32),
                        "motion_score": np.stack(scores, axis=1),
                        "track_src": np.stack(srcs, axis=1),
                        "track_disp": np.stack(disps, axis=1),
                        "track_weight": np.stack(weights, axis=1),
                        "track_count": np.array(counts, np.int64),
                        "gripper_points": gripper.astype(np.float32),
                        "gripper_weight": np.stack(grip_weights, axis=1),
                        "probe_state": np.concatenate(
                            [raw["cartesian_position"][raw_idx], raw["gripper_position"][raw_idx]], axis=-1
                        ).astype(np.float32),
                    }
                    for name, array in tensors.items():
                        group.create_dataset(name, data=array, compression="lzf" if array.ndim >= 3 else None)
                print(f"converted {index + 1}/{len(windows)} {key} {canon.tolist()}", flush=True)
        finally:
            for handle in outputs.values():
                handle.close()
        manifest["cache_sha256"] = {key: sha256(output / f"{key}.h5") for key in outputs}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
        return manifest
