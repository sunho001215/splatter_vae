from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .codecs import (
    pack_byte_strings,
    pack_support_masks,
    unpack_byte_strings,
    unpack_support_masks,
)

RGB_RECORD_MAGIC = b"ST0RGB01"
LAGER_RECORD_MAGIC = b"ST0LGR04"
LAGER_METADATA_MAGIC = b"ST0LGM04"
LAGER_V3_RECORD_MAGIC = b"ST0LGR03"
LAGER_V3_METADATA_MAGIC = b"ST0LGM03"
LEGACY_LAGER_RECORD_MAGIC = b"ST0LGR02"
LEGACY_LAGER_METADATA_MAGIC = b"ST0LGM02"
TIMESTEP_METADATA_MAGIC = b"ST0MET01"
_LAGER_HEADER = struct.Struct("<8sHH")
_TIMESTEP_HEADER = struct.Struct("<8sQIIIH")

_LEGACY_FLOAT_FIELDS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("target_c2w", (4, 4, 4)),
    ("target_w2c", (4, 4, 4)),
    ("base_c2w", (4, 4, 4)),
    ("alpha", (4,)),
    ("baseline", (4,)),
    ("translation_perturbation", (4, 3)),
    ("translation_perturbation_magnitude", (4,)),
    ("rotation_perturbation_degrees", (4,)),
    ("source_coverage", (4,)),
    ("minimum_geometry_distance", (4,)),
    ("geometry_clearance_distance", (4,)),
    ("target_scene_center", (4, 3)),
    ("distance_from_camera_a", (4,)),
    ("distance_from_camera_b", (4,)),
)
_V3_FLOAT_FIELDS: tuple[tuple[str, tuple[int, ...]], ...] = (
    *_LEGACY_FLOAT_FIELDS,
    ("minimum_source_coverage_threshold", (4,)),
    ("minimum_geometry_distance_threshold", (4,)),
)
_FLOAT_FIELDS: tuple[tuple[str, tuple[int, ...]], ...] = (
    *_V3_FLOAT_FIELDS,
    ("source_clearance_reference", (4,)),
)
_INT16_FIELDS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("rejected_candidates", (4,)),
    ("alpha_resample_count", (4,)),
)
_LEGACY_UINT8_FIELDS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("fallback_used", (4,)),
    ("scene_centered_arc_used", (4,)),
    ("geometry_scene_center_used", (4,)),
)
_UINT8_FIELDS: tuple[tuple[str, tuple[int, ...]], ...] = (
    *_LEGACY_UINT8_FIELDS,
    ("safety_translation_limit_escalated", (4,)),
    ("safety_thresholds_relaxed", (4,)),
)


def lager_record_contract() -> dict[str, Any]:
    return {
        "schema_version": 4,
        "record_magic": LAGER_RECORD_MAGIC.decode("ascii"),
        "metadata_magic": LAGER_METADATA_MAGIC.decode("ascii"),
        "float_fields": [name for name, _shape in _FLOAT_FIELDS],
        "int16_fields": [name for name, _shape in _INT16_FIELDS],
        "boolean_fields": [name for name, _shape in _UINT8_FIELDS],
        "previous_schema_version": 3,
        "previous_record_magic": LAGER_V3_RECORD_MAGIC.decode("ascii"),
        "previous_metadata_magic": LAGER_V3_METADATA_MAGIC.decode("ascii"),
        "legacy_schema_version": 2,
        "legacy_record_magic": LEGACY_LAGER_RECORD_MAGIC.decode("ascii"),
        "legacy_metadata_magic": LEGACY_LAGER_METADATA_MAGIC.decode("ascii"),
        "legacy_missing_safety_tier_policy": "ordinary_strict_defaults",
        "additional_uncompressed_bytes_per_timestamp_vs_v2": 56,
    }


def pack_real_rgb_record(jpegs: Sequence[bytes]) -> bytes:
    if len(jpegs) != 2:
        raise ValueError("Each retained timestamp requires two real RGB JPEGs.")
    return pack_byte_strings(tuple(jpegs), magic=RGB_RECORD_MAGIC)


def unpack_real_rgb_record(payload: bytes) -> tuple[bytes, bytes]:
    values = unpack_byte_strings(payload, magic=RGB_RECORD_MAGIC, expected_count=2)
    return values[0], values[1]


def pack_timestep_metadata(
    *,
    global_retained_index: int,
    episode_index: int,
    episode_id: str,
    retained_index: int,
    raw_timestep: int,
) -> bytes:
    identifier = str(episode_id).encode("utf-8")
    if not identifier or len(identifier) > 65535:
        raise ValueError("Episode identifiers must contain 1..65535 UTF-8 bytes.")
    values = (
        int(global_retained_index),
        int(episode_index),
        int(retained_index),
        int(raw_timestep),
    )
    if min(values) < 0 or max(values[1:]) > 0xFFFFFFFF:
        raise ValueError("Timestep metadata indices are outside their encoding range.")
    return (
        _TIMESTEP_HEADER.pack(TIMESTEP_METADATA_MAGIC, *values, len(identifier))
        + identifier
    )


def unpack_timestep_metadata(payload: bytes) -> dict[str, int | str]:
    if len(payload) < _TIMESTEP_HEADER.size:
        raise ValueError("Timestep metadata is truncated.")
    magic, global_index, episode_index, retained_index, raw_timestep, length = (
        _TIMESTEP_HEADER.unpack_from(payload)
    )
    if magic != TIMESTEP_METADATA_MAGIC:
        raise ValueError("Timestep metadata magic is invalid.")
    if len(payload) != _TIMESTEP_HEADER.size + int(length):
        raise ValueError("Timestep metadata length is invalid.")
    try:
        episode_id = payload[_TIMESTEP_HEADER.size :].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Timestep episode identifier is not UTF-8.") from exc
    return {
        "global_retained_index": int(global_index),
        "episode_index": int(episode_index),
        "episode_id": episode_id,
        "retained_index": int(retained_index),
        "raw_timestep": int(raw_timestep),
    }


def _pack_lager_metadata(metadata: Mapping[str, Any]) -> bytes:
    float_values = []
    for name, shape in _FLOAT_FIELDS:
        value = np.asarray(metadata[name], dtype="<f4")
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(
                f"Lager metadata {name} must be finite with shape {shape}."
            )
        float_values.append(value.reshape(-1))
    int16_values = []
    for name, shape in _INT16_FIELDS:
        value = np.asarray(metadata[name], dtype="<i2")
        if value.shape != shape:
            raise ValueError(f"Lager metadata {name} must have shape {shape}.")
        int16_values.append(value.reshape(-1))
    uint8_values = []
    for name, shape in _UINT8_FIELDS:
        value = np.asarray(metadata[name], dtype=np.uint8)
        if value.shape != shape:
            raise ValueError(f"Lager metadata {name} must have shape {shape}.")
        uint8_values.append(value.reshape(-1))
    floats = np.concatenate(float_values).astype("<f4", copy=False)
    ints = np.concatenate(int16_values).astype("<i2", copy=False)
    flags = np.concatenate(uint8_values).astype(np.uint8, copy=False)
    return (
        _LAGER_HEADER.pack(LAGER_METADATA_MAGIC, floats.size, ints.size)
        + floats.tobytes()
        + ints.tobytes()
        + flags.tobytes()
    )


def _unpack_lager_metadata(payload: bytes) -> dict[str, np.ndarray]:
    view = memoryview(payload)
    if len(view) < _LAGER_HEADER.size:
        raise ValueError("Lager metadata payload is truncated.")
    magic, float_count, int_count = _LAGER_HEADER.unpack_from(view)
    if magic == LAGER_METADATA_MAGIC:
        float_fields = _FLOAT_FIELDS
        uint8_fields = _UINT8_FIELDS
        schema_version = 4
    elif magic == LAGER_V3_METADATA_MAGIC:
        float_fields = _V3_FLOAT_FIELDS
        uint8_fields = _UINT8_FIELDS
        schema_version = 3
    elif magic == LEGACY_LAGER_METADATA_MAGIC:
        float_fields = _LEGACY_FLOAT_FIELDS
        uint8_fields = _LEGACY_UINT8_FIELDS
        schema_version = 2
    else:
        raise ValueError("Lager metadata header is invalid.")
    expected_floats = sum(np.prod(shape, dtype=int) for _, shape in float_fields)
    expected_ints = sum(np.prod(shape, dtype=int) for _, shape in _INT16_FIELDS)
    expected_flags = sum(np.prod(shape, dtype=int) for _, shape in uint8_fields)
    if float_count != expected_floats or int_count != expected_ints:
        raise ValueError("Lager metadata header is invalid.")
    float_start = _LAGER_HEADER.size
    int_start = float_start + int(float_count) * 4
    flag_start = int_start + int(int_count) * 2
    if len(view) != flag_start + int(expected_flags):
        raise ValueError("Lager metadata byte length is invalid.")
    floats = np.frombuffer(view[float_start:int_start], dtype="<f4")
    ints = np.frombuffer(view[int_start:flag_start], dtype="<i2")
    flags = np.frombuffer(view[flag_start:], dtype=np.uint8)
    output: dict[str, np.ndarray] = {}
    offset = 0
    for name, shape in float_fields:
        count = int(np.prod(shape))
        output[name] = floats[offset : offset + count].reshape(shape).copy()
        offset += count
    offset = 0
    for name, shape in _INT16_FIELDS:
        count = int(np.prod(shape))
        output[name] = ints[offset : offset + count].reshape(shape).copy()
        offset += count
    offset = 0
    for name, shape in uint8_fields:
        count = int(np.prod(shape))
        output[name] = (
            flags[offset : offset + count].reshape(shape).astype(np.bool_, copy=True)
        )
        offset += count
    if schema_version == 2:
        # Schema-2 records were produced before exceptional recovery tiers
        # existed, so their absence unambiguously means ordinary strict gates.
        output["minimum_source_coverage_threshold"] = np.full(
            (4,), 0.60, dtype=np.float32
        )
        output["minimum_geometry_distance_threshold"] = np.full(
            (4,), 0.08, dtype=np.float32
        )
        output["safety_translation_limit_escalated"] = np.zeros((4,), dtype=np.bool_)
        output["safety_thresholds_relaxed"] = np.zeros((4,), dtype=np.bool_)
    return output


def pack_lager_record(
    jpegs: Sequence[bytes],
    metadata: Mapping[str, Any],
    support_masks: np.ndarray,
) -> bytes:
    if len(jpegs) != 4:
        raise ValueError("Every retained timestamp requires four LagerNVS JPEGs.")
    mask_payload = pack_support_masks(np.asarray(support_masks, dtype=np.bool_))
    return pack_byte_strings(
        (*tuple(jpegs), _pack_lager_metadata(metadata), mask_payload),
        magic=LAGER_RECORD_MAGIC,
    )


def unpack_lager_record(
    payload: bytes,
) -> tuple[tuple[bytes, bytes, bytes, bytes], dict[str, np.ndarray], np.ndarray]:
    stored_magic = bytes(memoryview(payload)[:8])
    if stored_magic not in {
        LAGER_RECORD_MAGIC,
        LAGER_V3_RECORD_MAGIC,
        LEGACY_LAGER_RECORD_MAGIC,
    }:
        raise ValueError(f"Unexpected Lager record magic {stored_magic!r}.")
    values = unpack_byte_strings(payload, magic=stored_magic, expected_count=6)
    expected_metadata_magic = {
        LAGER_RECORD_MAGIC: LAGER_METADATA_MAGIC,
        LAGER_V3_RECORD_MAGIC: LAGER_V3_METADATA_MAGIC,
        LEGACY_LAGER_RECORD_MAGIC: LEGACY_LAGER_METADATA_MAGIC,
    }[stored_magic]
    if bytes(memoryview(values[4])[:8]) != expected_metadata_magic:
        raise ValueError("Lager record and metadata schema versions do not match.")
    metadata = _unpack_lager_metadata(values[4])
    support = unpack_support_masks(values[5])
    if support.shape != (4, 1, 256, 256):
        raise ValueError("Lager record contains an invalid support-mask shape.")
    return (values[0], values[1], values[2], values[3]), metadata, support
