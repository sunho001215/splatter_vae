"""Bounded binary frames and in-memory numpy archives; no repository imports."""

from __future__ import annotations

import hashlib
import io
import json
import struct
from typing import BinaryIO

import numpy as np

MAX_PAYLOAD = 2 << 30


class FrameError(RuntimeError):
    """Invalid, truncated, oversized, or corrupted frame."""


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = io.BytesIO()
    remaining = size
    while remaining:
        block = stream.read(min(remaining, 1 << 20))
        if not block:
            raise FrameError("truncated frame")
        data.write(block)
        remaining -= len(block)
    return data.getvalue()


def _write_all(stream: BinaryIO, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if written is None or written <= 0:
            raise OSError("frame write made no progress")
        view = view[written:]


def write_frame(stream: BinaryIO, header: dict, payload: bytes = b"") -> None:
    """Write one frame, including its payload checksum, and flush it."""
    header = dict(header)
    if payload:
        header["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    else:
        header.pop("payload_sha256", None)
    try:
        encoded = json.dumps(header, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FrameError(f"invalid JSON header: {exc}") from exc
    if len(encoded) > 0xffffffff:
        raise FrameError("header exceeds wire length limit")
    _write_all(stream, b"S4DF\x01" + struct.pack(">I", len(encoded)))
    _write_all(stream, encoded)
    _write_all(stream, struct.pack(">Q", len(payload)))
    _write_all(stream, payload)
    stream.flush()


def read_frame(stream: BinaryIO, max_header: int = 1 << 20,
               max_payload: int = MAX_PAYLOAD) -> tuple[dict, bytes]:
    """Read exactly one frame. Only EOF before its first byte is a clean EOF."""
    first = stream.read(1)
    if not first:
        raise EOFError("end of frame stream")
    if first + _read_exact(stream, 3) != b"S4DF":
        raise FrameError("bad frame magic")
    if _read_exact(stream, 1) != b"\x01":
        raise FrameError("unsupported frame version")
    header_len = struct.unpack(">I", _read_exact(stream, 4))[0]
    if header_len > max_header:
        raise FrameError(f"header length {header_len} exceeds limit {max_header}")
    try:
        header = json.loads(_read_exact(stream, header_len).decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise FrameError(f"invalid JSON header: {exc}") from exc
    if not isinstance(header, dict):
        raise FrameError("frame header must be an object")
    payload_len = struct.unpack(">Q", _read_exact(stream, 8))[0]
    if payload_len > max_payload:
        raise FrameError(f"payload length {payload_len} exceeds limit {max_payload}")
    payload = _read_exact(stream, payload_len)
    if payload_len and header.get("payload_sha256") != hashlib.sha256(payload).hexdigest():
        raise FrameError("payload sha256 mismatch")
    return header, payload


def pack_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize numeric arrays without temporary files or object pickles."""
    for name, array in arrays.items():
        if not isinstance(name, str) or not isinstance(array, np.ndarray):
            raise ValueError("arrays must map string names to numpy arrays")
        if array.dtype.hasobject:
            raise ValueError("object arrays require pickle and are refused")
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def unpack_arrays(payload: bytes) -> dict[str, np.ndarray]:
    """Read an in-memory npz archive with pickle disabled."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}
