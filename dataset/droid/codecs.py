from __future__ import annotations

import io
import json
import math
import struct
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numcodecs import Blosc
from PIL import Image
from PIL import __version__ as pillow_version

DEPTH_INVALID_U16 = np.uint16(0)
DEPTH_METERS_PER_UNIT = 0.001
FLOW_INVALID_I16 = np.int16(-32768)
FLOW_FIXED_POINT_SCALE = 64.0
FLOW_MIN_PIXELS = -32767.0 / FLOW_FIXED_POINT_SCALE
FLOW_MAX_PIXELS = 32767.0 / FLOW_FIXED_POINT_SCALE

_ARRAY_MAGIC = b"ST0ARR01"
_ARRAY_HEADER = struct.Struct("<8sBBBB4I")
_BYTE_HEADER = struct.Struct("<8sH")
_LENGTH = struct.Struct("<I")
_DTYPE_TO_CODE = {np.dtype("<u2"): 1, np.dtype("<i2"): 2}
_CODE_TO_DTYPE = {value: key for key, value in _DTYPE_TO_CODE.items()}


@dataclass(frozen=True)
class JPEGCodecConfig:
    quality: int = 95
    subsampling: str = "4:4:4"
    optimize: bool = False
    progressive: bool = False

    def __post_init__(self) -> None:
        if not 1 <= int(self.quality) <= 100:
            raise ValueError("JPEG quality must lie in [1,100].")
        if self.subsampling not in {"4:4:4", "4:2:2", "4:2:0"}:
            raise ValueError("JPEG subsampling must be 4:4:4, 4:2:2, or 4:2:0.")

    @property
    def pillow_subsampling(self) -> int:
        return {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}[self.subsampling]

    def metadata(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "format": "JPEG",
            "encoder": "Pillow",
            "pillow_version": pillow_version,
        }


@dataclass(frozen=True)
class NumericCodecConfig:
    cname: str = "zstd"
    compression_level: int = 3
    shuffle: str = "bitshuffle"

    def __post_init__(self) -> None:
        if self.cname != "zstd":
            raise ValueError("Stage-0 numeric payloads require Zstd.")
        if self.shuffle != "bitshuffle":
            raise ValueError("Stage-0 numeric payloads require bitshuffle.")
        if not 0 <= int(self.compression_level) <= 9:
            raise ValueError("Blosc compression level must lie in [0,9].")

    def build(self) -> Blosc:
        return Blosc(
            cname=self.cname,
            clevel=int(self.compression_level),
            shuffle=Blosc.BITSHUFFLE,
        )

    def metadata(self) -> dict[str, Any]:
        import numcodecs

        return {
            **asdict(self),
            "container": "blosc",
            "numcodecs_version": numcodecs.__version__,
        }


DEFAULT_NUMERIC_CODEC = NumericCodecConfig()


def _rgb_hwc_u8(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3:
        raise ValueError(f"JPEG input must be three-dimensional, got {value.shape}.")
    if value.shape[-1] == 3:
        output = value
    elif value.shape[0] == 3:
        output = np.moveaxis(value, 0, -1)
    else:
        raise ValueError(f"JPEG input must have three RGB channels, got {value.shape}.")
    if output.dtype != np.uint8:
        if not np.issubdtype(output.dtype, np.floating):
            raise TypeError(f"Unsupported JPEG input dtype {output.dtype}.")
        scale = 255.0 if float(np.nanmax(output, initial=0.0)) <= 1.0 + 1.0e-6 else 1.0
        output = np.rint(np.clip(output * scale, 0.0, 255.0)).astype(np.uint8)
    return np.ascontiguousarray(output)


def encode_jpeg(image: np.ndarray, config: JPEGCodecConfig) -> bytes:
    output = io.BytesIO()
    Image.fromarray(_rgb_hwc_u8(image), mode="RGB").save(
        output,
        format="JPEG",
        quality=int(config.quality),
        subsampling=config.pillow_subsampling,
        optimize=bool(config.optimize),
        progressive=bool(config.progressive),
    )
    return output.getvalue()


def decode_jpeg(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        output = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(output)


def pack_byte_strings(items: Sequence[bytes], *, magic: bytes) -> bytes:
    if len(magic) != 8:
        raise ValueError("Framed-byte magic must contain exactly eight bytes.")
    if len(items) > 65535:
        raise ValueError("A framed payload may contain at most 65535 items.")
    lengths = b"".join(_LENGTH.pack(len(item)) for item in items)
    return _BYTE_HEADER.pack(magic, len(items)) + lengths + b"".join(items)


def unpack_byte_strings(
    payload: bytes | bytearray | memoryview,
    *,
    magic: bytes,
    expected_count: int | None = None,
) -> tuple[bytes, ...]:
    view = memoryview(payload)
    if len(view) < _BYTE_HEADER.size:
        raise ValueError("Framed payload is truncated.")
    stored_magic, count = _BYTE_HEADER.unpack_from(view)
    if stored_magic != magic:
        raise ValueError(f"Unexpected framed payload magic {stored_magic!r}.")
    if expected_count is not None and count != int(expected_count):
        raise ValueError(f"Expected {expected_count} framed items, found {count}.")
    table_stop = _BYTE_HEADER.size + count * _LENGTH.size
    if len(view) < table_stop:
        raise ValueError("Framed payload length table is truncated.")
    lengths = [
        _LENGTH.unpack_from(view, _BYTE_HEADER.size + index * _LENGTH.size)[0]
        for index in range(count)
    ]
    offset = table_stop
    output = []
    for length in lengths:
        stop = offset + int(length)
        if stop > len(view):
            raise ValueError("Framed item extends beyond its payload.")
        output.append(bytes(view[offset:stop]))
        offset = stop
    if offset != len(view):
        raise ValueError("Framed payload contains trailing bytes.")
    return tuple(output)


def encode_numeric_array(
    array: np.ndarray, config: NumericCodecConfig = DEFAULT_NUMERIC_CODEC
) -> bytes:
    value = np.asarray(array)
    dtype = value.dtype.newbyteorder("<")
    if dtype not in _DTYPE_TO_CODE:
        raise TypeError(f"Unsupported numeric payload dtype {value.dtype}.")
    value = np.ascontiguousarray(value, dtype=dtype)
    if not 1 <= value.ndim <= 4:
        raise ValueError("Numeric payloads support one to four dimensions.")
    shape = tuple(int(item) for item in value.shape) + (0,) * (4 - value.ndim)
    header = _ARRAY_HEADER.pack(
        _ARRAY_MAGIC,
        1,
        _DTYPE_TO_CODE[dtype],
        value.ndim,
        0,
        *shape,
    )
    return header + bytes(config.build().encode(value))


def decode_numeric_array(
    payload: bytes | bytearray | memoryview,
    config: NumericCodecConfig = DEFAULT_NUMERIC_CODEC,
) -> np.ndarray:
    view = memoryview(payload)
    if len(view) <= _ARRAY_HEADER.size:
        raise ValueError("Numeric payload is truncated.")
    magic, version, dtype_code, ndim, reserved, *shape = _ARRAY_HEADER.unpack_from(view)
    if magic != _ARRAY_MAGIC or version != 1 or reserved != 0:
        raise ValueError("Numeric payload header is invalid or unsupported.")
    if dtype_code not in _CODE_TO_DTYPE or not 1 <= ndim <= 4:
        raise ValueError("Numeric payload dtype or rank is unsupported.")
    resolved_shape = tuple(int(value) for value in shape[:ndim])
    if any(value <= 0 for value in resolved_shape) or any(shape[ndim:]):
        raise ValueError("Numeric payload shape is invalid.")
    decoded = config.build().decode(view[_ARRAY_HEADER.size :])
    output = np.frombuffer(decoded, dtype=_CODE_TO_DTYPE[dtype_code])
    expected = math.prod(resolved_shape)
    if output.size != expected:
        raise ValueError(
            f"Numeric payload has {output.size} values, expected {expected}."
        )
    return output.reshape(resolved_shape).copy()


def depth_meters_to_u16(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    millimeters = np.rint(depth * 1000.0)
    millimeters = np.clip(millimeters, 1.0, 65535.0)
    return np.where(valid, millimeters, 0.0).astype("<u2")


def depth_u16_to_meters(depth_u16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.asarray(depth_u16, dtype="<u2")
    valid = encoded > DEPTH_INVALID_U16
    return encoded.astype(np.float32) * DEPTH_METERS_PER_UNIT, valid


def flow_pixels_to_i16(flow_pixels: np.ndarray) -> np.ndarray:
    flow = np.asarray(flow_pixels, dtype=np.float32)
    if flow.shape[-3] != 2:
        raise ValueError("Flow arrays must have a two-channel component axis.")
    finite = np.isfinite(flow).all(axis=-3, keepdims=True)
    fixed = np.rint(np.where(finite, flow, 0.0) * FLOW_FIXED_POINT_SCALE)
    fixed = np.clip(fixed, -32767.0, 32767.0).astype("<i2")
    return np.where(finite, fixed, FLOW_INVALID_I16).astype("<i2", copy=False)


def flow_i16_to_pixels(flow_i16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.asarray(flow_i16, dtype="<i2")
    if encoded.shape[-3] != 2:
        raise ValueError("Encoded flow arrays must have a two-channel component axis.")
    valid = (encoded != FLOW_INVALID_I16).all(axis=-3, keepdims=True)
    flow = encoded.astype(np.float32) / FLOW_FIXED_POINT_SCALE
    flow = np.where(valid, flow, 0.0).astype(np.float32, copy=False)
    return flow, valid


def pack_support_masks(masks: np.ndarray) -> bytes:
    value = np.asarray(masks, dtype=np.bool_)
    if value.shape[-3:] != (1, 256, 256):
        raise ValueError("Lager support masks must end in (1,256,256).")
    metadata = json.dumps(
        {"shape": list(value.shape), "bitorder": "little"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    packed = np.packbits(value.reshape(-1), bitorder="little").tobytes()
    return pack_byte_strings((metadata, packed), magic=b"ST0MASK1")


def unpack_support_masks(payload: bytes) -> np.ndarray:
    metadata_bytes, packed = unpack_byte_strings(
        payload, magic=b"ST0MASK1", expected_count=2
    )
    metadata = json.loads(metadata_bytes)
    shape = tuple(int(value) for value in metadata["shape"])
    if shape[-3:] != (1, 256, 256) or metadata.get("bitorder") != "little":
        raise ValueError("Lager support-mask metadata is invalid.")
    count = math.prod(shape)
    unpacked = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8), count=count, bitorder="little"
    )
    return unpacked.reshape(shape).astype(np.bool_, copy=False)
