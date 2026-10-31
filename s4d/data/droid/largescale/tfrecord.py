"""TensorFlow-free TFRecord framing and ``tf.train.Example`` wire-format walking for DROID RLDS 1.0.1.

The released DROID RLDS (``gs://gresearch/robotics/droid/1.0.1``, ``features.json``/``dataset_info.json``) is
``fileFormat: tfrecord`` with 2048 train shards named ``droid_101-train.tfrecord-XXXXX-of-02048``; every record is one
whole episode serialized by TFDS as a flat ``tf.train.Example`` whose keys are ``episode_metadata/<name>`` and
``steps/<path>`` (sequence features flattened over the steps). Images are ``jpeg`` (180x320x3), numeric tensors use
``encoding: none`` (plain packed ``float_list``/``int64_list``).

Reading a whole episode record (tens of MB) for a three-frame window wastes I/O, so the indexer records the byte
span of every per-step JPEG and of each packed numeric payload *inside* the record. A window then preads only the
frames it needs. Nothing here imports TensorFlow; the wire format is walked directly.

TFRecord framing: ``uint64 length | uint32 masked_crc32c(length) | data | uint32 masked_crc32c(data)``.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import numpy as np

_CRC_TABLE: list[int] | None = None


def _table() -> list[int]:
    global _CRC_TABLE
    if _CRC_TABLE is None:
        table = []
        for i in range(256):
            crc = i
            for _ in range(8):
                crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
            table.append(crc)
        _CRC_TABLE = table
    return _CRC_TABLE


def crc32c(data: bytes) -> int:
    """Castagnoli CRC (pure Python; used for 8-byte headers always and for payloads only on request)."""
    table, crc = _table(), 0xFFFFFFFF
    for byte in data:
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc(data: bytes) -> int:
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


class TFRecordError(ValueError):
    pass


def iter_record_spans(fd: int, *, size: int | None = None, verify_payload: bool = False):
    """Yield ``(data_offset, data_length)`` of every record by reading 12-byte headers and seeking past payloads.

    Memory is O(1); payloads are only read when ``verify_payload`` is set.
    """
    size = os.fstat(fd).st_size if size is None else size
    pos = 0
    while pos < size:
        header = os.pread(fd, 12, pos)
        if len(header) != 12:
            raise TFRecordError(f"truncated TFRecord header at byte {pos}")
        length, length_crc = struct.unpack("<QI", header)
        if masked_crc(header[:8]) != length_crc:
            raise TFRecordError(f"corrupt TFRecord length CRC at byte {pos}")
        start = pos + 12
        if start + length + 4 > size:
            raise TFRecordError(f"truncated TFRecord payload at byte {pos}")
        if verify_payload:
            data = os.pread(fd, length, start)
            (data_crc,) = struct.unpack("<I", os.pread(fd, 4, start + length))
            if masked_crc(data) != data_crc:
                raise TFRecordError(f"corrupt TFRecord payload CRC at byte {pos}")
        yield start, length
        pos = start + length + 4


def write_records(path, payloads) -> list[tuple[int, int]]:
    """Write a TFRecord file (fixtures and tools). Returns the ``(offset, length)`` of each payload."""
    spans = []
    with open(path, "wb") as stream:
        for data in payloads:
            header = struct.pack("<Q", len(data))
            stream.write(header + struct.pack("<I", masked_crc(header)))
            spans.append((stream.tell(), len(data)))
            stream.write(data + struct.pack("<I", masked_crc(data)))
    return spans


# ---------------------------------------------------------------------------------------------- protobuf wire format
def _varint(buf, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise TFRecordError("varint too long")


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            return bytes(out)


def _fields(buf, start: int, end: int):
    """Yield ``(field_number, wire_type, value_start, value_end)``; length-delimited values only carry their span."""
    pos = start
    while pos < end:
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 2:
            length, pos = _varint(buf, pos)
            yield number, wire, pos, pos + length
            pos += length
        elif wire == 0:
            value_start = pos
            _, pos = _varint(buf, pos)
            yield number, wire, value_start, pos
        elif wire == 5:
            yield number, wire, pos, pos + 4
            pos += 4
        elif wire == 1:
            yield number, wire, pos, pos + 8
            pos += 8
        else:
            raise TFRecordError(f"unsupported wire type {wire}")
    if pos != end:
        raise TFRecordError("protobuf field overran its message")


@dataclass
class FeatureSpan:
    """Where one Example feature's values live inside the record payload (offsets relative to the payload)."""

    kind: str  # "bytes" | "float" | "int64"
    elements: list[tuple[int, int]] = field(default_factory=list)  # bytes: one (offset, length) per element
    packed: tuple[int, int] | None = None  # float/int64: one packed payload span
    unpacked: list[tuple[int, int]] = field(default_factory=list)  # float/int64 written without packing


_KINDS = {1: "bytes", 2: "float", 3: "int64"}


def walk_example(buf, keys: set[str] | None = None) -> dict[str, FeatureSpan]:
    """Locate the value spans of ``tf.train.Example`` features without copying any payload.

    Example{1: Features{1: map<string, Feature>}}; Feature oneof {1: BytesList, 2: FloatList, 3: Int64List}; each
    list has ``repeated value = 1`` (packed for numbers in proto3).
    """
    view = memoryview(buf)
    out: dict[str, FeatureSpan] = {}
    for number, wire, s, e in _fields(view, 0, len(view)):
        if number != 1 or wire != 2:
            continue
        for fnum, fwire, ms, me in _fields(view, s, e):
            if fnum != 1 or fwire != 2:
                continue
            name, feature = None, None
            for enum, ewire, vs, ve in _fields(view, ms, me):
                if enum == 1 and ewire == 2:
                    name = bytes(view[vs:ve]).decode()
                elif enum == 2 and ewire == 2:
                    feature = (vs, ve)
            if name is None or feature is None or (keys is not None and name not in keys):
                continue
            span = None
            for knum, kwire, ls, le in _fields(view, *feature):
                if kwire != 2 or knum not in _KINDS:
                    continue
                span = FeatureSpan(_KINDS[knum])
                for vnum, vwire, xs, xe in _fields(view, ls, le):
                    if vnum != 1:
                        continue
                    if span.kind == "bytes":
                        span.elements.append((xs, xe - xs))
                    elif vwire == 2:
                        if span.packed is not None:
                            raise TFRecordError(f"feature {name!r} has two packed payloads")
                        span.packed = (xs, xe - xs)
                    else:
                        span.unpacked.append((xs, xe - xs))
            if span is None:  # an empty Feature: no kind set
                span = FeatureSpan("bytes")
            out[name] = span
    return out


def decode_numeric(raw: bytes, kind: str) -> np.ndarray:
    """Decode a packed float32 / int64 payload."""
    if kind == "float":
        if len(raw) % 4:
            raise TFRecordError("packed float payload is not a multiple of 4 bytes")
        return np.frombuffer(raw, dtype="<f4").copy()
    if kind == "int64":
        values, pos = [], 0
        while pos < len(raw):
            value, pos = _varint(raw, pos)
            values.append(value - (1 << 64) if value >= 1 << 63 else value)
        return np.asarray(values, dtype=np.int64)
    raise TFRecordError(f"not a numeric feature kind: {kind}")


def feature_values(buf, span: FeatureSpan) -> np.ndarray | list[bytes]:
    """Materialize one feature's values from a payload that is already in memory."""
    if span.kind == "bytes":
        return [bytes(buf[o : o + n]) for o, n in span.elements]
    if span.packed is not None:
        return decode_numeric(bytes(buf[span.packed[0] : span.packed[0] + span.packed[1]]), span.kind)
    if span.kind == "float":
        return np.array([struct.unpack("<f", bytes(buf[o : o + 4]))[0] for o, _ in span.unpacked], np.float32)
    return np.array([_varint(buf, o)[0] for o, _ in span.unpacked], np.int64)


# ---------------------------------------------------------------------------------------------- Example encoding
def _ld(number: int, payload: bytes) -> bytes:
    return _encode_varint((number << 3) | 2) + _encode_varint(len(payload)) + payload


def encode_example(features: dict[str, tuple[str, object]]) -> bytes:
    """Serialize a ``tf.train.Example`` (fixtures). ``features[name] = (kind, values)`` with kind bytes/float/int64.

    Byte-compatible with TensorFlow's serializer for the subset used here (packed numbers, proto3).
    """
    entries = b""
    for name in sorted(features):
        kind, values = features[name]
        if kind == "bytes":
            body = b"".join(_ld(1, bytes(v)) for v in values)
            feature = _ld(1, body)
        elif kind == "float":
            packed = np.asarray(values, dtype="<f4").tobytes()
            feature = _ld(2, _ld(1, packed) if len(packed) else b"")
        elif kind == "int64":
            packed = b"".join(_encode_varint(int(v) & ((1 << 64) - 1)) for v in np.asarray(values).reshape(-1))
            feature = _ld(3, _ld(1, packed) if packed else b"")
        else:
            raise ValueError(kind)
        entries += _ld(1, _ld(1, name.encode()) + _ld(2, feature))
    return _ld(1, entries)
