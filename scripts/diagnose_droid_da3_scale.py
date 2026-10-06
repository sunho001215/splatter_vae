"""Read-only cached-pair diagnostic; never repairs or approves production depth.

Use the isolated DA3 environment and one authorized CUDA UUID. Cached JPEGs
are used, so the re-inference is not a bit-exact replay of raw-RLDS inference.
Sparse stereo evidence is conditional on the supplied calibration being correct.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from dataset.droid.codecs import decode_jpeg, decode_numeric_array
from dataset.droid.preprocessed_manifest import load_stage0_manifest, shard_path
from dataset.droid.records import unpack_real_rgb_record
from dataset.droid.shards import IndexedTarReader, write_json_atomic


def triangulate_matches(points, K, w2c):
    """Positive-depth, >=1 degree parallax, <=1px reprojection stereo points."""
    projection = K @ w2c[:, :3]
    homogeneous = cv2.triangulatePoints(
        projection[0], projection[1], points[0].T, points[1].T
    ).T
    with np.errstate(divide="ignore", invalid="ignore"):
        world = homogeneous[:, :3] / homogeneous[:, 3:]
        camera = np.einsum("vij,nj->vni", w2c[:, :3, :3], world)
        camera += w2c[:, None, :3, 3]
        projected = np.einsum("vij,vnj->vni", K, camera)
        xy = projected[..., :2] / projected[..., 2:]
        error = np.linalg.norm(xy - points, axis=-1).max(axis=0)
        centers = np.linalg.inv(w2c)[:, :3, 3]
        rays = world[None] - centers[:, None]
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        angle = np.degrees(np.arccos(np.clip((rays[0] * rays[1]).sum(-1), -1, 1)))
    depth = camera[..., 2]
    valid = (
        np.isfinite(world).all(-1)
        & np.isfinite(error)
        & (error <= 1.0)
        & (angle >= 1.0)
        & (depth > 0.01).all(0)
        & (depth < 20.0).all(0)
    )
    return depth[:, valid], points[:, valid], error[valid]


def sparse_stereo(rgb, K, w2c):
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    features = [
        sift.detectAndCompute(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), None) for x in rgb
    ]
    if any(descriptor is None or len(descriptor) < 2 for _, descriptor in features):
        return np.empty((2, 0)), np.empty((2, 0, 2)), np.empty(0), 0
    matcher = cv2.BFMatcher()
    forward = matcher.knnMatch(features[0][1], features[1][1], k=2)
    reverse = matcher.knnMatch(features[1][1], features[0][1], k=2)
    reverse_good = {
        (a.trainIdx, a.queryIdx) for a, b in reverse if a.distance < 0.7 * b.distance
    }
    matches = [
        a
        for a, b in forward
        if a.distance < 0.7 * b.distance and (a.queryIdx, a.trainIdx) in reverse_good
    ]
    if not matches:
        return np.empty((2, 0)), np.empty((2, 0, 2)), np.empty(0), 0
    points = np.array(
        [
            [features[0][0][m.queryIdx].pt for m in matches],
            [features[1][0][m.trainIdx].pt for m in matches],
        ],
        dtype=np.float64,
    )
    depth, points, error = triangulate_matches(points, K, w2c)
    return depth, points, error, len(matches)


def percentiles(value):
    value = np.asarray(value)
    value = value[np.isfinite(value)]
    if not value.size:
        return None
    return dict(
        zip(
            ("p01", "p05", "p50", "p95", "p99"),
            np.percentile(value, (1, 5, 50, 95, 99)).tolist(),
            strict=True,
        )
    )


def depth_stats(depth, stereo_depth, points):
    result = {"depth_m": [percentiles(x) for x in depth]}
    if points.shape[1]:
        values = np.stack(
            [
                cv2.remap(
                    depth[v],
                    points[v, :, 0].astype(np.float32)[None],
                    points[v, :, 1].astype(np.float32)[None],
                    cv2.INTER_LINEAR,
                )[0]
                for v in range(2)
            ]
        )
        result["depth_over_triangulated_depth"] = [
            percentiles(x) for x in values / stereo_depth
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/ws/data/droid_stage0_preprocessed")
    parser.add_argument(
        "--sample", action="append", required=True, help="shard_id:global_key"
    )
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    allowed = {
        "GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce",
        "GPU-d09f0338-71b9-d915-3c7f-e99754a3b639",
        "GPU-6b33eef2-80c5-547e-6a61-7ab81c14d63b",
    }
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in allowed:
        raise ValueError("Select exactly one authorized CUDA UUID.")
    from preprocessing.da3.official import DA3DROIDTeacher, _resize_valid_depth

    manifest = load_stage0_manifest(args.root)
    teacher = DA3DROIDTeacher(
        Path(__file__).resolve().parents[1] / "third_party/Depth-Anything-3",
        cache_dir=Path(args.root) / "metadata/model_cache",
        device="cuda:0",
    )
    report = {
        "diagnostic_only": True,
        "input": "cached Q95 JPEG, not raw RLDS replay",
        "manifest_signature": manifest["schema_signature"],
        "teacher": teacher.metadata(),
        "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
        "samples": [],
    }
    for specification in args.sample:
        shard, key = specification.split(":")
        shard = int(shard)
        key = f"{int(key):010d}"
        entry = next(
            x
            for x in manifest["episodes"]
            if x["shard_id"] == shard
            and x["global_retained_start"]
            <= int(key)
            < x["global_retained_start"] + x["retained_count"]
        )
        rgb_reader = IndexedTarReader(shard_path(args.root, "rgb", shard))
        depth_reader = IndexedTarReader(shard_path(args.root, "da3", shard))
        try:
            payload = rgb_reader.read(key, "rgb", verify=True)
            rgb = np.stack([decode_jpeg(x) for x in unpack_real_rgb_record(payload)])
            cached = (
                decode_numeric_array(
                    depth_reader.read(key, "depth", verify=True)
                ).astype(np.float32)
                * 0.001
            )
        finally:
            rgb_reader.close()
            depth_reader.close()
        K = np.array([x["intrinsics_rlds"] for x in entry["exterior_cameras"]])
        w2c = np.array([x["w2c"] for x in entry["exterior_cameras"]])
        stereo_depth, points, reprojection_error, mutual_count = sparse_stereo(
            rgb, K, w2c
        )
        captured = {}
        original = teacher.model._align_to_input_extrinsics_intrinsics

        def capture(
            extrinsics,
            intrinsics,
            prediction,
            *rest,
            captured=captured,
            original=original,
            **kwargs,
        ):
            captured["depth"] = _resize_valid_depth(prediction.depth.copy(), 180, 320)
            captured["extrinsics"] = prediction.extrinsics.copy()
            captured["intrinsics"] = prediction.intrinsics.copy()
            return original(extrinsics, intrinsics, prediction, *rest, **kwargs)

        with patch.object(
            teacher.model, "_align_to_input_extrinsics_intrinsics", side_effect=capture
        ):
            output = teacher(rgb, K, w2c)
        sample = {
            "shard_id": shard,
            "key": key,
            "episode_id": entry["episode_id"],
            "raw_timestep": (int(key) - entry["global_retained_start"]) * 3,
            "input_rgb_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "calibrated_baseline_m": output["input_baseline_m"],
            "pose_alignment_scale": output["pose_alignment_scale"],
            "nested_metric_scale_factor": output["scale_factor"],
            "predicted_extrinsics_before_alignment": captured["extrinsics"].tolist(),
            "predicted_intrinsics_before_alignment": captured["intrinsics"].tolist(),
            "sift_mutual_ratio_matches": mutual_count,
            "calibrated_triangulation_matches": points.shape[1],
            "triangulated_depth_m": [percentiles(x) for x in stereo_depth],
            "triangulation_reprojection_error_px": percentiles(reprojection_error),
            "cached": depth_stats(cached, stereo_depth, points),
            "replay_after_alignment": depth_stats(
                output["metric_depth"], stereo_depth, points
            ),
            "replay_before_alignment": depth_stats(
                captured["depth"], stereo_depth, points
            ),
        }
        report["samples"].append(sample)
        write_json_atomic(args.report, report)
        print(
            f"{key}: stereo={points.shape[1]}/{mutual_count}, scale={output['pose_alignment_scale']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
