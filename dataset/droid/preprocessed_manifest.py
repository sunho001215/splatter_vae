from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .calibration import load_calibration_manifest
from .safety import DEFAULT_DROID_ROOT, validate_derived_root
from .shards import write_json_atomic

STAGE0_DATASET_SCHEMA_VERSION = 2
RETAINED_RAW_STRIDE = 3
TEMPORAL_GAP_RAW = 6
TEMPORAL_WINDOW = 3
RETAINED_WINDOW_OFFSETS = (0, 2, 4)
SOURCE_HEIGHT = 180
SOURCE_WIDTH = 320
LAGER_SIZE = 256

FOUNDATION_MODEL_PROVENANCE = {
    "da3": {
        "repository": "ByteDance-Seed/Depth-Anything-3",
        "repository_revision": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
        "model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        "checkpoint_revision": "b2359bdf726fb44ef62acca04d629dcf158053e7",
    },
    "megaflow": {
        "repository": "cvg/megaflow",
        "repository_revision": "ee5b61813db0a76ac0db9034899aade72a0d230c",
        "model_id": "megaflow-flow",
        "checkpoint_repository": "Kristen-Z/MegaFlow",
        "checkpoint_revision": "b4c5c33800b8fa88e047d2eb70ae74b0feca606d",
    },
    "lagernvs": {
        "repository": "facebookresearch/lagernvs",
        "repository_revision": "665f727aba8298a04ff4c040fd6279a32ef23017",
        "model_id": "facebook/lagernvs_dl3dv_2-6_v_256",
        "checkpoint_revision": "4026552953a72c5fb037501564dc673dd73c574e",
    },
}

DEFAULT_LAGER_SCENE_CENTER = (0.6268912475, 0.0371947839, 0.1571336175)
OFFLINE_TEACHER_PROCESSING = {
    "da3": {
        "inference_group": "synchronized_cam_a_cam_b_only",
        "camera_extrinsics": "opencv_colmap_w2c",
        "process_resolution": 504,
        "process_resolution_method": "upper_bound_resize",
        "align_to_input_extrinsic_scale": True,
        "two_view_pose_scale_alignment": "predicted_baseline_over_input_baseline",
        "use_ray_pose": False,
        "reference_view_strategy": "first",
        "output_resolution": [180, 320],
    },
    "megaflow": {
        "flow_direction": "forward_t_to_tplus6",
        "native_resolution": [180, 320],
        "refinement_iterations": 8,
        "maximum_sequence_frames": 12,
        "amp_dtype": "bf16",
        "attention_backend": "pytorch_sdpa",
    },
    "lagernvs": {
        "conditioning_views": 2,
        "targets_per_timestep": 4,
        "canonical_resolution": [256, 256],
        "canonical_focal_px": 186.5,
        "alpha_bands_cam_a": [[0.15, 0.25], [0.25, 0.35]],
        "symmetric_alpha_resampling": True,
        "random_alpha_resample_attempts": 4,
        "alpha_safety_grid_candidates": 17,
        "excluded_center_distance": 0.15,
        "configured_scene_center": list(DEFAULT_LAGER_SCENE_CENTER),
        "geometry_centered_arc_fallback": True,
        "translation_max_baseline_fraction": 0.03,
        "rotation_max_degrees": 3.0,
        "minimum_source_coverage": 0.60,
        "minimum_geometry_distance_m": 0.08,
        "geometry_clearance_statistic": "point_distance_quantile",
        "geometry_clearance_quantile": 0.01,
        "minimum_optical_axis_cosine": 0.35,
        "coverage_sample_stride": 2,
        "coverage_dilation_kernel": 7,
        "source_reconstructor_calls_per_timestep": 1,
        "target_renderer_calls_per_timestep": 4,
        "attention_backend": "pytorch_sdpa",
    },
}


def sample_key(global_retained_index: int) -> str:
    value = int(global_retained_index)
    if value < 0:
        raise ValueError("Global retained indices must be nonnegative.")
    return f"{value:010d}"


def retained_count(num_steps: int) -> int:
    return max(0, (int(num_steps) + RETAINED_RAW_STRIDE - 1) // RETAINED_RAW_STRIDE)


def window_count(num_steps: int) -> int:
    return max(0, retained_count(num_steps) - RETAINED_WINDOW_OFFSETS[-1])


def flow_timestamp_count(num_steps: int) -> int:
    return max(0, retained_count(num_steps) - TEMPORAL_GAP_RAW // RETAINED_RAW_STRIDE)


def retained_raw_indices(num_steps: int) -> tuple[int, ...]:
    return tuple(range(0, int(num_steps), RETAINED_RAW_STRIDE))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_signature(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _pipeline_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable payload contract shared by pilot and full manifests.

    Dataset selection and shard assignment intentionally do not participate, so a
    pilot can authorize a full manifest only when every codec, geometry, temporal,
    and teacher-provenance decision is identical.
    """

    return {
        name: value[name]
        for name in (
            "schema_version",
            "source_root",
            "source_calibration_sha256",
            "eligibility",
            "retained_raw_stride",
            "temporal_gap_raw",
            "temporal_window",
            "retained_window_offsets",
            "source_resolution",
            "lager_resolution",
            "encoding",
            "foundation_models",
            "teacher_processing",
        )
    }


def _write_jsonl_gzip_atomic(path: Path, entries: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as raw:
        with (
            gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed,
            io.TextIOWrapper(compressed, encoding="utf-8") as stream,
        ):
            for entry in entries:
                stream.write(json.dumps(entry, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def _theoretical_uncompressed_bytes(counts: Mapping[str, int]) -> dict[str, int]:
    retained = int(counts["retained_timesteps"])
    flow_fields = int(counts["flow_fields"])
    episodes = int(counts["eligible_episodes"])
    values = {
        "real_rgb": retained * 2 * SOURCE_HEIGHT * SOURCE_WIDTH * 3,
        "depth_uint16": retained * 2 * SOURCE_HEIGHT * SOURCE_WIDTH * 2,
        "flow_int16": flow_fields * 2 * SOURCE_HEIGHT * SOURCE_WIDTH * 2,
        "lager_rgb": retained * 4 * LAGER_SIZE * LAGER_SIZE * 3,
        "lager_support_bitpacked": retained * 4 * LAGER_SIZE * LAGER_SIZE // 8,
        # K plus c2w/w2c for two cameras, stored once per episode.
        "real_camera_geometry_float32": episodes * 2 * (9 + 16 + 16) * 4,
        # Three 4x4 poses per target plus 64 scalar/vector diagnostics.
        "lager_pose_metadata_float32": retained * (4 * 3 * 16 + 64) * 4,
        "lager_pose_metadata_int16": retained * 8 * 2,
        "lager_pose_metadata_flags": retained * 12,
    }
    values["total"] = sum(values.values())
    return values


def _eligible_entries(
    entries: Sequence[Mapping[str, Any]],
    episode_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for raw in entries:
        if not bool(raw.get("valid")):
            continue
        episode_id = str(raw.get("episode_id") or "")
        if not episode_id:
            raise ValueError("A Stage-0-valid manifest entry has no episode_id.")
        if raw.get("dataset_split") not in {"train", "validation"}:
            raise ValueError(
                f"Stage-0-valid episode {episode_id} has no canonical split."
            )
        if episode_ids is not None and episode_id not in episode_ids:
            continue
        cameras = raw.get("exterior_cameras")
        if not isinstance(cameras, list) or len(cameras) != 2:
            raise ValueError(f"Stage-0-valid episode {episode_id} lacks two cameras.")
        selected.append(dict(raw))
    selected.sort(key=lambda item: (str(item["rlds_split"]), int(item["rlds_ordinal"])))
    if episode_ids is not None:
        missing = episode_ids - {str(entry["episode_id"]) for entry in selected}
        if missing:
            raise ValueError(
                f"Requested pilot episodes are not Stage-0 valid: {sorted(missing)}"
            )
    if not selected:
        raise ValueError("No Stage-0-valid episodes were selected.")
    return selected


def _assign_shards(
    entries: Sequence[Mapping[str, Any]], target_retained_per_shard: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = int(target_retained_per_shard)
    if target <= 0:
        raise ValueError("target_retained_per_shard must be positive.")
    output: list[dict[str, Any]] = []
    shard_entries: list[list[int]] = []
    global_retained = 0
    global_window = 0
    current: list[int] = []
    current_count = 0
    for entry_index, raw in enumerate(entries):
        count = retained_count(int(raw["num_steps"]))
        if current and current_count + count > target:
            shard_entries.append(current)
            current = []
            current_count = 0
        shard_id = len(shard_entries)
        enriched = dict(raw)
        enriched.update(
            {
                "manifest_episode_index": entry_index,
                "retained_raw_stride": RETAINED_RAW_STRIDE,
                "retained_count": count,
                "global_retained_start": global_retained,
                "training_window_count": window_count(int(raw["num_steps"])),
                "global_window_start": global_window,
                "shard_id": shard_id,
            }
        )
        output.append(enriched)
        current.append(entry_index)
        current_count += count
        global_retained += count
        global_window += enriched["training_window_count"]
    if current:
        shard_entries.append(current)

    shards = []
    for shard_id, episode_indices in enumerate(shard_entries):
        first = output[episode_indices[0]]
        last = output[episode_indices[-1]]
        count = sum(int(output[index]["retained_count"]) for index in episode_indices)
        shards.append(
            {
                "shard_id": shard_id,
                "episode_start": episode_indices[0],
                "episode_stop": episode_indices[-1] + 1,
                "episode_count": len(episode_indices),
                "retained_count": count,
                "global_retained_start": int(first["global_retained_start"]),
                "global_retained_stop": int(last["global_retained_start"])
                + int(last["retained_count"]),
            }
        )
    return output, shards


def build_stage0_manifest(
    calibration_manifest: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    droid_root: str | os.PathLike[str] = DEFAULT_DROID_ROOT,
    target_retained_per_shard: int = 4096,
    episode_ids: Sequence[str] | None = None,
    mode: str = "full",
    jpeg_quality: int = 95,
    jpeg_subsampling: str = "4:4:4",
    numeric_compression_level: int = 3,
    write: bool = True,
) -> dict[str, Any]:
    if mode not in {"full", "pilot"}:
        raise ValueError("Manifest mode must be 'full' or 'pilot'.")
    if not 1 <= int(jpeg_quality) <= 100:
        raise ValueError("JPEG quality must lie in [1,100].")
    if jpeg_subsampling not in {"4:4:4", "4:2:2", "4:2:0"}:
        raise ValueError("Unsupported JPEG chroma subsampling.")
    if not 0 <= int(numeric_compression_level) <= 9:
        raise ValueError("Numeric Zstd compression level must lie in [0,9].")
    calibration_path = Path(calibration_manifest).expanduser().resolve()
    root = validate_derived_root(output_root, droid_root)
    selected_ids = (
        None if episode_ids is None else {str(value) for value in episode_ids}
    )
    entries = _eligible_entries(
        load_calibration_manifest(calibration_path), selected_ids
    )
    episodes, shards = _assign_shards(entries, target_retained_per_shard)
    counts = {
        "eligible_episodes": len(episodes),
        "raw_timesteps": sum(int(entry["num_steps"]) for entry in episodes),
        "raw_exterior_frames": sum(2 * int(entry["num_steps"]) for entry in episodes),
        "retained_timesteps": sum(int(entry["retained_count"]) for entry in episodes),
        "training_windows": sum(
            int(entry["training_window_count"]) for entry in episodes
        ),
        "flow_timestamps": sum(
            flow_timestamp_count(int(entry["num_steps"])) for entry in episodes
        ),
    }
    counts.update(
        {
            "flow_fields": counts["flow_timestamps"] * 2,
            "real_rgb_images": counts["retained_timesteps"] * 2,
            "depth_maps": counts["retained_timesteps"] * 2,
            "lager_jpeg_images": counts["retained_timesteps"] * 4,
        }
    )
    core = {
        "schema_version": STAGE0_DATASET_SCHEMA_VERSION,
        "mode": mode,
        "source_root": str(Path(droid_root).expanduser().resolve()),
        "source_calibration_manifest": str(calibration_path),
        "source_calibration_sha256": _sha256_file(calibration_path),
        "eligibility": "canonical_stage0_calibration_valid_only",
        "retained_raw_stride": RETAINED_RAW_STRIDE,
        "temporal_gap_raw": TEMPORAL_GAP_RAW,
        "temporal_window": TEMPORAL_WINDOW,
        "retained_window_offsets": list(RETAINED_WINDOW_OFFSETS),
        "source_resolution": [SOURCE_HEIGHT, SOURCE_WIDTH],
        "lager_resolution": [LAGER_SIZE, LAGER_SIZE],
        "encoding": {
            "real_rgb": {
                "format": "jpeg",
                "quality": int(jpeg_quality),
                "chroma_subsampling": str(jpeg_subsampling),
                "optimize": False,
                "progressive": False,
            },
            "lagernvs_rgb": {
                "format": "jpeg",
                "quality": int(jpeg_quality),
                "chroma_subsampling": str(jpeg_subsampling),
                "optimize": False,
                "progressive": False,
            },
            "depth": {
                "dtype": "uint16",
                "units": "millimeters",
                "invalid": 0,
                "resolution": [SOURCE_HEIGHT, SOURCE_WIDTH],
            },
            "flow": {
                "dtype": "int16",
                "fixed_point_scale": 64,
                "invalid_sentinel": -32768,
                "resolution": [SOURCE_HEIGHT, SOURCE_WIDTH],
                "direction": "forward_t_to_tplus6",
            },
            "numeric_compression": {
                "container": "blosc",
                "codec": "zstd",
                "compression_level": int(numeric_compression_level),
                "shuffle": "bitshuffle",
            },
            "tar_stream_compression": None,
        },
        "foundation_models": FOUNDATION_MODEL_PROVENANCE,
        "teacher_processing": OFFLINE_TEACHER_PROCESSING,
        "target_retained_per_shard": int(target_retained_per_shard),
        "counts": counts,
        "theoretical_uncompressed_bytes": _theoretical_uncompressed_bytes(counts),
        "split_episode_counts": dict(
            Counter(str(entry["dataset_split"]) for entry in episodes)
        ),
        "shards": shards,
    }
    core["pipeline_signature"] = _canonical_signature(_pipeline_contract(core))
    core["schema_signature"] = _canonical_signature(core)
    if write:
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("schema_signature") != core["schema_signature"]:
                raise FileExistsError(
                    f"Refusing to replace an incompatible dataset manifest at {manifest_path}."
                )
        _write_jsonl_gzip_atomic(root / "manifests" / "episodes.jsonl.gz", episodes)
        write_json_atomic(manifest_path, core)
    return {**core, "episodes": episodes}


def load_stage0_manifest(root: str | os.PathLike[str]) -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != STAGE0_DATASET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Stage-0 dataset schema at {dataset_root}.")
    signed = dict(manifest)
    recorded_schema_signature = signed.pop("schema_signature", None)
    if recorded_schema_signature != _canonical_signature(signed):
        raise ValueError(f"Stage-0 manifest signature is invalid at {dataset_root}.")
    if manifest.get("pipeline_signature") != _canonical_signature(
        _pipeline_contract(manifest)
    ):
        raise ValueError(f"Stage-0 pipeline signature is invalid at {dataset_root}.")
    with gzip.open(dataset_root / "manifests" / "episodes.jsonl.gz", "rt") as stream:
        episodes = [json.loads(line) for line in stream if line.strip()]
    if len(episodes) != int(manifest["counts"]["eligible_episodes"]):
        raise ValueError("Stage-0 episode manifest count is inconsistent.")
    return {**manifest, "episodes": episodes}


def shard_path(root: str | os.PathLike[str], stage: str, shard_id: int) -> Path:
    dataset_root = Path(root).expanduser().resolve()
    if stage == "final":
        directory = dataset_root / "shards"
        prefix = "stage0"
    elif stage in {"rgb", "da3", "megaflow", "lagernvs"}:
        directory = dataset_root / "staging" / stage / "shards"
        prefix = stage
    else:
        raise ValueError(f"Unknown preprocessing stage {stage!r}.")
    return directory / f"{prefix}-{int(shard_id):05d}.tar"


@dataclass(frozen=True)
class WindowReference:
    episode_index: int
    retained_start: int
    retained_indices: tuple[int, int, int]
    raw_timesteps: tuple[int, int, int]


def resolve_window(episode: Mapping[str, Any], retained_start: int) -> WindowReference:
    start = int(retained_start)
    count = int(episode["retained_count"])
    indices = tuple(start + offset for offset in RETAINED_WINDOW_OFFSETS)
    if start < 0 or indices[-1] >= count:
        raise IndexError(
            f"Retained window {indices} is outside episode {episode['episode_id']} ({count})."
        )
    raw = tuple(index * RETAINED_RAW_STRIDE for index in indices)
    if raw[1] - raw[0] != TEMPORAL_GAP_RAW or raw[2] - raw[1] != TEMPORAL_GAP_RAW:
        raise RuntimeError(
            "Stage-0 temporal window contract is internally inconsistent."
        )
    return WindowReference(
        episode_index=int(episode["manifest_episode_index"]),
        retained_start=start,
        retained_indices=indices,
        raw_timesteps=raw,
    )
