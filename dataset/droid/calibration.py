from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from .safety import DEFAULT_DROID_ROOT, validate_derived_root

OFFICIAL_CALIBRATION_REPOSITORY = "KarlP/droid"
OFFICIAL_CALIBRATION_REVISION = "main"
OFFICIAL_CALIBRATION_BASE_URL = "https://huggingface.co/KarlP/droid/resolve/main"
OFFICIAL_CALIBRATION_FILES = (
    "episode_id_to_path.json",
    "camera_serials.json",
    "intrinsics.json",
    "cam2base_extrinsics.json",
    "cam2cam_extrinsics.json",
    "cam2base_extrinsic_superset.json",
)
RLDS_IMAGE_WIDTH = 320
RLDS_IMAGE_HEIGHT = 180
OFFICIAL_INTRINSIC_WIDTH = 1280
OFFICIAL_INTRINSIC_HEIGHT = 720


@dataclass(frozen=True)
class CalibrationThresholds:
    rotation_orthonormal_atol: float = 0.02
    rotation_determinant_atol: float = 0.02
    maximum_translation_m: float = 5.0
    minimum_exterior_baseline_m: float = 0.05
    maximum_exterior_baseline_m: float = 3.0
    principal_point_margin_fraction: float = 0.25
    direct_relative_max_rotation_deg: float = 20.0
    direct_relative_max_translation_m: float = 0.35
    direction_minimum_samples: int = 100
    direction_maximum_samples: int = 25_000


@dataclass(frozen=True)
class RLDSEpisodeMetadata:
    rlds_split: str
    rlds_ordinal: int
    file_path: str
    recording_folderpath: str
    num_steps: int


def _json_open(path: Path, mode: str):
    if ".gz" in path.suffixes:
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def download_official_calibration(
    output_dir: str | os.PathLike[str],
    *,
    droid_root: str | os.PathLike[str] = DEFAULT_DROID_ROOT,
    overwrite: bool = False,
) -> dict[str, dict[str, object]]:
    """Download the official post-hoc files outside the read-only data root."""
    root = validate_derived_root(output_dir, droid_root)
    root.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, dict[str, object]] = {}
    for filename in OFFICIAL_CALIBRATION_FILES:
        destination = root / filename
        if overwrite or not destination.is_file():
            request = urllib.request.Request(
                f"{OFFICIAL_CALIBRATION_BASE_URL}/{filename}",
                headers={"User-Agent": "splatter-vae-droid-calibration"},
            )
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
            os.replace(temporary, destination)
        digest = hashlib.sha256()
        with destination.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        provenance[filename] = {
            "bytes": destination.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    metadata = {
        "repository": OFFICIAL_CALIBRATION_REPOSITORY,
        "revision": OFFICIAL_CALIBRATION_REVISION,
        "files": provenance,
    }
    (root / "provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def load_official_calibration(
    calibration_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    root = Path(calibration_dir).expanduser().resolve()
    missing = [
        filename
        for filename in OFFICIAL_CALIBRATION_FILES
        if not (root / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Official calibration directory {root} is missing: {missing}."
        )
    output: dict[str, Any] = {}
    for filename in OFFICIAL_CALIBRATION_FILES:
        with (root / filename).open("r", encoding="utf-8") as stream:
            output[filename.removesuffix(".json")] = json.load(stream)
    return output


def normalize_episode_path(value: str | bytes | os.PathLike[str]) -> str:
    """Normalize raw, gs://, local, and RLDS trajectory paths to one episode path."""
    if isinstance(value, bytes):
        text = value.decode("utf-8")
    else:
        text = os.fspath(value)
    text = unquote(text.strip()).replace("\\", "/")
    parsed = urlparse(text)
    if parsed.scheme:
        text = parsed.path
    text = "/" + text.lstrip("/")
    for marker in ("/r2d2-data-full/", "/droid_raw/1.0.1/", "/droid_raw/1.0.0/"):
        if marker in text:
            text = "/" + text.split(marker, 1)[1]
            break
    for marker in ("/trajectory", "/recordings", "/metadata_"):
        if marker in text:
            text = text.split(marker, 1)[0]
    parts = [part for part in PurePosixPath(text).parts if part not in ("/", "", ".")]
    return "/".join(parts)


class EpisodePathMatcher:
    def __init__(self, episode_id_to_path: Mapping[str, str]):
        exact_candidates: dict[str, list[str]] = defaultdict(list)
        suffixes: dict[str, list[str]] = defaultdict(list)
        for episode_id, path in episode_id_to_path.items():
            normalized = normalize_episode_path(path)
            exact_candidates[normalized].append(str(episode_id))
            parts = normalized.split("/")
            for count in (4, 5, 6):
                if len(parts) >= count:
                    suffixes["/".join(parts[-count:])].append(str(episode_id))
        self._exact = {
            key: values[0]
            for key, values in exact_candidates.items()
            if len(set(values)) == 1
        }
        self._ambiguous_exact = {
            key for key, values in exact_candidates.items() if len(set(values)) > 1
        }
        self._suffix = {
            key: values[0] for key, values in suffixes.items() if len(set(values)) == 1
        }

    def is_ambiguous(self, normalized_path: str) -> bool:
        return normalized_path in self._ambiguous_exact

    def match(
        self, file_path: str, recording_folderpath: str = ""
    ) -> tuple[str | None, str]:
        candidates = []
        for raw in (file_path, recording_folderpath):
            if raw:
                normalized = normalize_episode_path(raw)
                if normalized and normalized not in candidates:
                    candidates.append(normalized)
        for candidate in candidates:
            if candidate in self._exact:
                return self._exact[candidate], candidate
            parts = candidate.split("/")
            for count in (6, 5, 4):
                if len(parts) >= count:
                    suffix = "/".join(parts[-count:])
                    if suffix in self._suffix:
                        return self._suffix[suffix], candidate
        return None, candidates[0] if candidates else ""


def euler_xyz_to_matrix(angles: Sequence[float]) -> np.ndarray:
    values = np.asarray(angles, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("XYZ Euler angles must contain three finite radians.")
    x, y, z = values
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rotation_x = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    rotation_y = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    rotation_z = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    # scipy Rotation.from_euler("xyz") uses this extrinsic composition.
    return rotation_z @ rotation_y @ rotation_x


def cam2base_pose_matrix(extrinsics: Sequence[float]) -> np.ndarray:
    values = np.asarray(extrinsics, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("cam2base must be a finite [tx,ty,tz,rx,ry,rz] vector.")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = euler_xyz_to_matrix(values[3:])
    transform[:3, 3] = values[:3]
    return transform


def parse_intrinsics(
    record: Mapping[str, Any] | Sequence[float],
) -> tuple[np.ndarray, int, int]:
    if isinstance(record, Mapping):
        values = record.get("cameraMatrix")
        width = int(record.get("width", 0))
        height = int(record.get("height", 0))
    else:
        values = record
        width, height = 0, 0
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape == (4,):
        fx, cx, fy, cy = vector
        K = np.array(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)))
    elif vector.shape == (3, 3):
        K = vector.copy()
    else:
        raise ValueError(f"Unsupported intrinsic shape: {vector.shape}.")
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Intrinsics must be finite with positive focal lengths.")
    return K.astype(np.float64), width, height


def resize_intrinsics(
    K: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    source_width, source_height = map(int, source_size)
    target_width, target_height = map(int, target_size)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Intrinsic source and target dimensions must be positive.")
    output = np.asarray(K, dtype=np.float64).copy()
    sx = target_width / source_width
    sy = target_height / source_height
    output[0, 0] *= sx
    output[0, 2] *= sx
    output[1, 1] *= sy
    output[1, 2] *= sy
    return output


def transform_error(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[float, float]:
    delta = np.linalg.inv(reference) @ candidate
    cosine = np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    rotation_deg = math.degrees(math.acos(float(cosine)))
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    return rotation_deg, translation_m


def validate_transform(
    transform: np.ndarray,
    thresholds: CalibrationThresholds,
) -> str | None:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        return "invalid_extrinsic_shape"
    if not np.isfinite(matrix).all():
        return "nonfinite_extrinsic"
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-6):
        return "invalid_homogeneous_row"
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=float(thresholds.rotation_orthonormal_atol),
    ):
        return "nonorthonormal_rotation"
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > float(thresholds.rotation_determinant_atol):
        return "invalid_rotation_determinant"
    if float(np.linalg.norm(matrix[:3, 3])) > float(thresholds.maximum_translation_m):
        return "implausible_translation"
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return "singular_extrinsic"
    if not np.isfinite(inverse).all():
        return "nonfinite_inverse"
    return None


def validate_intrinsics(
    K: np.ndarray,
    width: int,
    height: int,
    thresholds: CalibrationThresholds,
) -> str | None:
    matrix = np.asarray(K, dtype=np.float64)
    if matrix.shape != (3, 3):
        return "invalid_intrinsic_shape"
    if not np.isfinite(matrix).all():
        return "nonfinite_intrinsics"
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        return "nonpositive_focal_length"
    margin_x = float(width) * thresholds.principal_point_margin_fraction
    margin_y = float(height) * thresholds.principal_point_margin_fraction
    if not (-margin_x <= matrix[0, 2] <= width + margin_x):
        return "implausible_principal_point"
    if not (-margin_y <= matrix[1, 2] <= height + margin_y):
        return "implausible_principal_point"
    return None


def released_ext2_from_ext1(relative_entry: Mapping[str, Any]) -> np.ndarray:
    """Return T_ext2<-ext1 from the official cam2cam left/right pose fields."""
    left = np.asarray(relative_entry["left_cam"]["pose"], dtype=np.float64)
    right = np.asarray(relative_entry["right_cam"]["pose"], dtype=np.float64)
    if left.shape != (4, 4) or right.shape != (4, 4):
        raise ValueError("cam2cam left/right poses must both be 4x4 matrices.")
    return np.linalg.inv(right) @ left


def _numeric_pose_keys(entry: Mapping[str, Any]) -> list[str]:
    return [
        str(key)
        for key, value in entry.items()
        if str(key).isdigit() and isinstance(value, list)
    ]


def infer_cam2cam_direction(
    direct_entries: Mapping[str, Any],
    relative_entries: Mapping[str, Any],
    camera_serials: Mapping[str, Any],
    thresholds: CalibrationThresholds,
) -> dict[str, Any]:
    """Numerically verify the relative direction using dual-direct episodes."""
    forward_errors: list[tuple[float, float]] = []
    inverse_errors: list[tuple[float, float]] = []
    episode_ids = sorted(
        set(direct_entries) & set(relative_entries) & set(camera_serials)
    )
    if len(episode_ids) > thresholds.direction_maximum_samples:
        step = len(episode_ids) / thresholds.direction_maximum_samples
        episode_ids = [
            episode_ids[int(index * step)]
            for index in range(thresholds.direction_maximum_samples)
        ]
    for episode_id in episode_ids:
        serial_map = camera_serials[episode_id]
        serial_a = str(serial_map.get("ext1_cam_serial", ""))
        serial_b = str(serial_map.get("ext2_cam_serial", ""))
        direct = direct_entries[episode_id]
        if serial_a not in direct or serial_b not in direct:
            continue
        try:
            base_from_a = cam2base_pose_matrix(direct[serial_a])
            base_from_b = cam2base_pose_matrix(direct[serial_b])
            direct_b_from_a = np.linalg.inv(base_from_b) @ base_from_a
            released = released_ext2_from_ext1(relative_entries[episode_id])
            if not np.isfinite(released).all():
                continue
            forward_errors.append(transform_error(direct_b_from_a, released))
            inverse_errors.append(
                transform_error(direct_b_from_a, np.linalg.inv(released))
            )
        except (KeyError, ValueError, np.linalg.LinAlgError):
            continue
    if len(forward_errors) < int(thresholds.direction_minimum_samples):
        raise ValueError(
            f"Only {len(forward_errors)} dual-direct episodes were usable; cannot verify cam2cam direction."
        )
    forward = np.asarray(forward_errors)
    inverse = np.asarray(inverse_errors)
    forward_score = float(np.median(forward[:, 0]) + 20.0 * np.median(forward[:, 1]))
    inverse_score = float(np.median(inverse[:, 0]) + 20.0 * np.median(inverse[:, 1]))
    if not forward_score < inverse_score:
        raise ValueError(
            "Official cam2cam direction verification unexpectedly selected the inverse."
        )
    return {
        "direction": "T_ext2<-ext1 = inv(right_cam.pose) @ left_cam.pose",
        "num_comparisons": len(forward_errors),
        "forward_median_rotation_deg": float(np.median(forward[:, 0])),
        "forward_median_translation_m": float(np.median(forward[:, 1])),
        "inverse_median_rotation_deg": float(np.median(inverse[:, 0])),
        "inverse_median_translation_m": float(np.median(inverse[:, 1])),
    }


def _invalid_entry(
    metadata: RLDSEpisodeMetadata,
    episode_id: str | None,
    normalized_path: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "rlds_path": normalized_path,
        "rlds_file_path": metadata.file_path,
        "rlds_recording_folderpath": metadata.recording_folderpath,
        "rlds_split": metadata.rlds_split,
        "rlds_ordinal": int(metadata.rlds_ordinal),
        "num_steps": int(metadata.num_steps),
        "valid": False,
        "failure_reason": reason,
    }


def build_calibration_entry(
    metadata: RLDSEpisodeMetadata,
    official: Mapping[str, Any],
    matcher: EpisodePathMatcher,
    thresholds: CalibrationThresholds,
) -> dict[str, Any]:
    episode_id, normalized_path = matcher.match(
        metadata.file_path, metadata.recording_folderpath
    )
    if episode_id is None:
        reason = (
            "ambiguous_path_match"
            if matcher.is_ambiguous(normalized_path)
            else "path_match_failed"
        )
        return _invalid_entry(metadata, None, normalized_path, reason)
    serials_all = official["camera_serials"]
    intrinsics_all = official["intrinsics"]
    if episode_id not in serials_all:
        return _invalid_entry(
            metadata, episode_id, normalized_path, "missing_camera_serials"
        )
    if episode_id not in intrinsics_all:
        return _invalid_entry(
            metadata, episode_id, normalized_path, "missing_intrinsics"
        )
    serial_map = serials_all[episode_id]
    serial_a = str(serial_map.get("ext1_cam_serial", ""))
    serial_b = str(serial_map.get("ext2_cam_serial", ""))
    if not serial_a or not serial_b or serial_a == serial_b:
        return _invalid_entry(
            metadata, episode_id, normalized_path, "invalid_exterior_serial_mapping"
        )
    intrinsics_entry = intrinsics_all[episode_id]
    if serial_a not in intrinsics_entry or serial_b not in intrinsics_entry:
        return _invalid_entry(
            metadata, episode_id, normalized_path, "missing_exterior_intrinsics"
        )
    cameras: dict[str, dict[str, Any]] = {}
    for logical_id, serial in (("exterior_1", serial_a), ("exterior_2", serial_b)):
        try:
            K_original, source_width, source_height = parse_intrinsics(
                intrinsics_entry[serial]
            )
        except (TypeError, ValueError):
            return _invalid_entry(
                metadata, episode_id, normalized_path, "invalid_intrinsics"
            )
        if source_width <= 0 or source_height <= 0:
            return _invalid_entry(
                metadata, episode_id, normalized_path, "missing_intrinsic_resolution"
            )
        if (source_width, source_height) != (
            OFFICIAL_INTRINSIC_WIDTH,
            OFFICIAL_INTRINSIC_HEIGHT,
        ):
            return _invalid_entry(
                metadata,
                episode_id,
                normalized_path,
                "unexpected_official_intrinsic_resolution",
            )
        intrinsic_reason = validate_intrinsics(
            K_original, source_width, source_height, thresholds
        )
        if intrinsic_reason:
            return _invalid_entry(
                metadata, episode_id, normalized_path, intrinsic_reason
            )
        K_rlds = resize_intrinsics(
            K_original,
            (source_width, source_height),
            (RLDS_IMAGE_WIDTH, RLDS_IMAGE_HEIGHT),
        )
        rlds_reason = validate_intrinsics(
            K_rlds, RLDS_IMAGE_WIDTH, RLDS_IMAGE_HEIGHT, thresholds
        )
        if rlds_reason:
            return _invalid_entry(metadata, episode_id, normalized_path, rlds_reason)
        cameras[logical_id] = {
            "logical_id": logical_id,
            "rlds_image_key": (
                "exterior_image_1_left"
                if logical_id == "exterior_1"
                else "exterior_image_2_left"
            ),
            "serial": serial,
            "intrinsics_original": K_original.tolist(),
            "intrinsics_original_resolution": [source_width, source_height],
            "intrinsics_rlds": K_rlds.tolist(),
            "intrinsics_rlds_resolution": [RLDS_IMAGE_WIDTH, RLDS_IMAGE_HEIGHT],
        }

    direct_sources = (
        ("cam2base_extrinsic_superset", official["cam2base_extrinsic_superset"]),
        ("cam2base_extrinsics", official["cam2base_extrinsics"]),
    )
    direct_pose: dict[str, tuple[np.ndarray, str, Mapping[str, Any]]] = {}
    for source_name, source_entries in direct_sources:
        entry = source_entries.get(episode_id, {})
        for logical_id, serial in (("exterior_1", serial_a), ("exterior_2", serial_b)):
            if logical_id not in direct_pose and serial in entry:
                try:
                    direct_pose[logical_id] = (
                        cam2base_pose_matrix(entry[serial]),
                        source_name,
                        entry,
                    )
                except ValueError:
                    return _invalid_entry(
                        metadata, episode_id, normalized_path, "invalid_cam2base"
                    )

    relative_entry = official["cam2cam_extrinsics"].get(episode_id)
    relative_b_from_a: np.ndarray | None = None
    if relative_entry is not None:
        try:
            relative_b_from_a = released_ext2_from_ext1(relative_entry)
            reason = validate_transform(relative_b_from_a, thresholds)
            if reason:
                relative_b_from_a = None
        except (KeyError, ValueError, np.linalg.LinAlgError):
            relative_b_from_a = None

    poses: dict[str, tuple[np.ndarray, str, str]] = {}
    for logical_id, (pose, source_name, source_entry) in direct_pose.items():
        serial = cameras[logical_id]["serial"]
        release_source = str(
            source_entry.get(f"{serial}_source", source_entry.get("source", "unknown"))
        )
        poses[logical_id] = (pose, "direct", f"{source_name}:{release_source}")

    if len(poses) == 1 and relative_b_from_a is not None:
        if "exterior_1" in poses:
            base_from_a = poses["exterior_1"][0]
            base_from_b = base_from_a @ np.linalg.inv(relative_b_from_a)
            poses["exterior_2"] = (base_from_b, "derived", "cam2cam_extrinsics")
        else:
            base_from_b = poses["exterior_2"][0]
            base_from_a = base_from_b @ relative_b_from_a
            poses["exterior_1"] = (base_from_a, "derived", "cam2cam_extrinsics")
    if len(poses) != 2:
        return _invalid_entry(
            metadata, episode_id, normalized_path, "insufficient_exterior_geometry"
        )

    if len(direct_pose) == 2 and relative_b_from_a is not None:
        direct_relative = (
            np.linalg.inv(direct_pose["exterior_2"][0]) @ direct_pose["exterior_1"][0]
        )
        rotation_error, translation_error = transform_error(
            direct_relative, relative_b_from_a
        )
        if (
            rotation_error > thresholds.direct_relative_max_rotation_deg
            or translation_error > thresholds.direct_relative_max_translation_m
        ):
            return _invalid_entry(
                metadata, episode_id, normalized_path, "direct_relative_disagreement"
            )
    else:
        rotation_error = None
        translation_error = None

    for logical_id, (c2w, pose_flag, source_name) in poses.items():
        reason = validate_transform(c2w, thresholds)
        if reason:
            return _invalid_entry(metadata, episode_id, normalized_path, reason)
        cameras[logical_id].update(
            {
                "c2w": c2w.tolist(),
                "w2c": np.linalg.inv(c2w).tolist(),
                "pose_flag": pose_flag,
                "calibration_source": source_name,
            }
        )
    baseline = float(
        np.linalg.norm(poses["exterior_1"][0][:3, 3] - poses["exterior_2"][0][:3, 3])
    )
    if (
        not thresholds.minimum_exterior_baseline_m
        <= baseline
        <= thresholds.maximum_exterior_baseline_m
    ):
        return _invalid_entry(
            metadata, episode_id, normalized_path, "implausible_exterior_baseline"
        )
    return {
        "episode_id": episode_id,
        "rlds_path": normalized_path,
        "rlds_file_path": metadata.file_path,
        "rlds_recording_folderpath": metadata.recording_folderpath,
        "rlds_split": metadata.rlds_split,
        "rlds_ordinal": int(metadata.rlds_ordinal),
        "num_steps": int(metadata.num_steps),
        "exterior_cameras": [cameras["exterior_1"], cameras["exterior_2"]],
        "exterior_baseline_m": baseline,
        "direct_relative_rotation_error_deg": rotation_error,
        "direct_relative_translation_error_m": translation_error,
        "valid": True,
        "failure_reason": None,
    }


def _split_group(entry: Mapping[str, Any]) -> str:
    parts = str(entry.get("rlds_path", "")).split("/")
    lab = parts[0] if parts else "unknown"
    date = parts[2] if len(parts) > 2 else "unknown"
    return f"{lab}/{date}"


def assign_episode_splits(
    entries: Sequence[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one.")
    valid_entries = [entry for entry in entries if entry.get("valid")]
    groups = sorted({_split_group(entry) for entry in valid_entries})
    validation_groups = {
        group
        for group in groups
        if int.from_bytes(
            hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big"
        )
        / float(2**64)
        < validation_fraction
    }
    if groups and not validation_groups:
        validation_groups.add(groups[-1])
    if len(validation_groups) == len(groups) and len(groups) > 1:
        validation_groups.remove(groups[0])
    split_ids: dict[str, list[str]] = {"train": [], "validation": []}
    for entry in valid_entries:
        split = "validation" if _split_group(entry) in validation_groups else "train"
        entry["dataset_split"] = split
        split_ids[split].append(str(entry["episode_id"]))
    return split_ids


def calibration_statistics(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reasons = Counter(
        str(entry.get("failure_reason"))
        for entry in entries
        if not bool(entry.get("valid"))
    )
    valid = [entry for entry in entries if entry.get("valid")]
    pose_flags = Counter(
        camera["pose_flag"]
        for entry in valid
        for camera in entry.get("exterior_cameras", [])
    )
    return {
        "total_episodes_inspected": len(entries),
        "successful_path_matches": sum(
            entry.get("episode_id") is not None for entry in entries
        ),
        "calibration_valid_episodes": len(valid),
        "rejected_episodes": len(entries) - len(valid),
        "rejection_reason_counts": dict(sorted(reasons.items())),
        "direct_pose_count": int(pose_flags["direct"]),
        "derived_pose_count": int(pose_flags["derived"]),
        "train_episode_count": sum(
            entry.get("dataset_split") == "train" for entry in valid
        ),
        "validation_episode_count": sum(
            entry.get("dataset_split") == "validation" for entry in valid
        ),
    }


def write_calibration_manifest(
    entries: Sequence[Mapping[str, Any]],
    output_path: str | os.PathLike[str],
    *,
    droid_root: str | os.PathLike[str] = DEFAULT_DROID_ROOT,
) -> None:
    destination = Path(output_path).expanduser().resolve(strict=False)
    validate_derived_root(destination.parent, droid_root).mkdir(
        parents=True, exist_ok=True
    )
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with _json_open(temporary, "wt") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
    os.replace(temporary, destination)


def load_calibration_manifest(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    manifest = Path(path).expanduser().resolve()
    with _json_open(manifest, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare_calibration_manifest(
    episode_metadata: Iterable[RLDSEpisodeMetadata],
    calibration_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    thresholds: CalibrationThresholds | None = None,
    validation_fraction: float = 0.02,
    split_seed: int = 42,
    split_output_path: str | os.PathLike[str] | None = None,
    droid_root: str | os.PathLike[str] = DEFAULT_DROID_ROOT,
) -> dict[str, Any]:
    thresholds = CalibrationThresholds() if thresholds is None else thresholds
    official = load_official_calibration(calibration_dir)
    direction = infer_cam2cam_direction(
        official["cam2base_extrinsic_superset"],
        official["cam2cam_extrinsics"],
        official["camera_serials"],
        thresholds,
    )
    matcher = EpisodePathMatcher(official["episode_id_to_path"])
    entries = [
        build_calibration_entry(metadata, official, matcher, thresholds)
        for metadata in episode_metadata
    ]
    splits = assign_episode_splits(
        entries, validation_fraction=validation_fraction, seed=split_seed
    )
    write_calibration_manifest(entries, output_path, droid_root=droid_root)
    output = Path(output_path).expanduser().resolve()
    stats = calibration_statistics(entries)
    payload = {
        "schema_version": 1,
        "official_repository": OFFICIAL_CALIBRATION_REPOSITORY,
        "official_revision": OFFICIAL_CALIBRATION_REVISION,
        "intrinsics_source_resolution_verified_from_release_metadata": True,
        "official_intrinsics_resolution": [
            OFFICIAL_INTRINSIC_WIDTH,
            OFFICIAL_INTRINSIC_HEIGHT,
        ],
        "rlds_resolution": [RLDS_IMAGE_WIDTH, RLDS_IMAGE_HEIGHT],
        "cam2cam_direction_validation": direction,
        "thresholds": asdict(thresholds),
        "statistics": stats,
    }
    (output.parent / "calibration_statistics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    split_output = (
        output.parent / "episode_splits.json"
        if split_output_path is None
        else Path(split_output_path).expanduser().resolve(strict=False)
    )
    split_output = (
        validate_derived_root(split_output.parent, droid_root) / split_output.name
    )
    split_output.parent.mkdir(parents=True, exist_ok=True)
    split_temporary = split_output.with_suffix(split_output.suffix + ".partial")
    split_temporary.write_text(
        json.dumps({"seed": split_seed, **splits}, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(split_temporary, split_output)
    return payload
