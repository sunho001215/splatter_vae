"""Wire protocol of the real-world policy server. numpy + stdlib only (the robot-side client never imports torch).

One message = ``u32`` little-endian header length | UTF-8 JSON header | raw array bytes. The header carries ``type``,
``v`` (protocol version), the message fields and an array table ``arrays`` = [{name, dtype, shape, offset, nbytes}]
describing the payload that follows: C-order, little-endian, concatenated in table order without gaps
(``payload_nbytes`` = sum of ``nbytes``). Arrays are sent at their exact dtype (uint8 RGB, float32 metres, bool
validity, float32 normalized actions, float64 poses), never re-encoded or compressed.

Session flow over ONE persistent TCP connection (TCP_NODELAY):

  server -> WELCOME {server, nonce, auth_required, policies: {id: unbound InputSpec JSON}}
  client -> HELLO   {policy_id, choices, resume: {session_id, last_ack} | null, auth, takeover}
  server -> SPEC    {session_id, resumed, input_spec (bound), limits, last_seq_received, redeliver}
  client -> STEP    {session_id, seq, frames: [only NEW frames], sets: [new multi-view sets], request}
  server -> ACTION  {session_id, seq, ack, status "ok", reference_id, outputs + output_arrays, server_ms}
          | ERROR   {session_id, seq, ack, code, detail, history_reset, fatal}
  client -> RESET   {session_id, seq}          server -> RESET_OK {session_id, seq, ack}
  client -> PING    {t_client_ns}              server -> PONG {t_client_ns, t_server_ns}
  client -> BYE     {session_id}               (server ends the session and closes the connection)

Every request seq is per session, strictly increasing; a repeated seq returns the cached response (no re-inference).
"""

from __future__ import annotations

import json
import math
import select
import socket
import struct
import time
from dataclasses import dataclass, field

import numpy as np

PROTOCOL = "s4d-policy-serving"
VERSION = 1
PREFIX = struct.Struct("<I")
ACTION_DIM = 7  # FQL action dimension (dx dy dz rx ry rz gripper); DeployPolicy validates explicit noise as (1, 7)

WELCOME, HELLO, SPEC, STEP, ACTION, ERROR = "WELCOME", "HELLO", "SPEC", "STEP", "ACTION", "ERROR"
RESET, RESET_OK, PING, PONG, BYE = "RESET", "RESET_OK", "PING", "PONG", "BYE"
TYPES = frozenset({WELCOME, HELLO, SPEC, STEP, ACTION, ERROR, RESET, RESET_OK, PING, PONG, BYE})

# Error codes (ERROR.code). ``fatal`` errors are followed by the server closing the connection.
ERROR_CODES = {
    "version": "protocol version mismatch (fatal)",
    "auth": "shared-token handshake failed (fatal)",
    "bad_request": "malformed message, payload sizes, unknown fields or inputs the deploy contract refuses",
    "session_unknown": "no such (live) session: expired, ended, evicted or the server restarted; HELLO again",
    "session_busy": "the policy already serves another live session (HELLO takeover=true evicts it explicitly)",
    "session_taken_over": "this session was evicted by another client's takeover: the robot loop must stop",
    "incomplete_history": "a frame (or multi-view set) required at the trained history spacing is missing",
    "irregular_spacing": "frame ids or capture times do not follow the trained history spacing",
    "unsynchronized": "multi-view run-start padding not step-aligned across cameras (cross-camera offsets and M5 "
                      "motion are logged warnings since 2026-11-12, never this refusal)",
    "binding_refused": "measured pose / gripper sample not bound to the observation as in training (sample age, ids)",
    "duplicate": "repeated frame id, or a repeated request seq whose cached response was evicted",
    "stale": "superseded by a newer request of the session, or its ttl expired before execution",
    "overloaded": "the bounded inference queue is full",
    "shutting_down": "the server is stopping",
    "internal": "unexpected server failure (logged on the server)",
}

# Wire dtypes (numpy ``dtype.str``): exactly the arrays the deploy contract uses.
DTYPES = frozenset({"|u1", "|b1", "<u2", "<i4", "<i8", "<f4", "<f8"})
MAX_NDIM = 6


class ProtocolError(Exception):
    """A message violates the protocol. ``code`` is an ERROR code; ``fatal`` closes the connection."""

    def __init__(self, code: str, detail: str, *, fatal: bool = False):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail, self.fatal = code, detail, fatal


class Stopped(Exception):
    """The reader was asked to stop while waiting for a message."""


@dataclass
class Message:
    header: dict
    arrays: dict = field(default_factory=dict)
    nbytes: int = 0  # bytes on the wire (prefix + header + payload)
    t_recv_ns: int = 0  # time.monotonic_ns() when the last byte was read (receiver's clock)

    @property
    def type(self) -> str:
        return self.header["type"]

    def get(self, key, default=None):
        return self.header.get(key, default)


def _wire_array(a) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    if not a.flags.c_contiguous:
        a = a.copy(order="C")  # (np.ascontiguousarray would turn a 0-d scalar into shape (1,))
    if a.dtype.str not in DTYPES:
        raise ProtocolError("bad_request", f"unsupported wire dtype {a.dtype.str}")
    if a.ndim > MAX_NDIM:
        raise ProtocolError("bad_request", f"array rank {a.ndim} > {MAX_NDIM}")
    return a


def encode(header: dict, arrays: dict | None = None) -> bytes:
    """Header dict (``type`` required) + named arrays -> one wire message. ``v``, ``arrays``, ``payload_nbytes`` are set
    here. Non-finite floats are refused in the JSON header (``allow_nan=False``); arrays carry them unchanged."""
    if header.get("type") not in TYPES:
        raise ProtocolError("bad_request", f"unknown message type {header.get('type')!r}")
    table, blobs, offset = [], [], 0
    for name, value in (arrays or {}).items():
        a = _wire_array(value)
        blob = a.reshape(-1).view(np.uint8)
        table.append({"name": str(name), "dtype": a.dtype.str, "shape": list(a.shape), "offset": offset,
                      "nbytes": int(blob.nbytes)})
        blobs.append(blob)
        offset += int(blob.nbytes)
    head = {**header, "v": header.get("v", VERSION), "arrays": table, "payload_nbytes": offset}
    raw = json.dumps(head, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return b"".join([PREFIX.pack(len(raw)), raw, *blobs])


def _no_constant(name: str):
    raise ValueError(f"non-finite JSON number {name} refused")


def loads_strict(text: str):
    """JSON without NaN / Infinity (Python's json accepts them by default; the wire never carries them)."""
    return json.loads(text, parse_constant=_no_constant)


def parse_header(raw: bytes | bytearray | memoryview, *, max_payload_bytes: int) -> dict:
    try:
        header = loads_strict(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:  # JSONDecodeError is a ValueError
        raise ProtocolError("bad_request", f"header is not strict UTF-8 JSON: {exc}", fatal=True) from exc
    if not isinstance(header, dict) or header.get("type") not in TYPES:
        raise ProtocolError("bad_request", "header must be an object with a known type", fatal=True)
    if type(header.get("v")) is not int:
        raise ProtocolError("version", "header has no integer protocol version", fatal=True)
    n = header.get("payload_nbytes")
    if type(n) is not int or n < 0 or n > max_payload_bytes:
        raise ProtocolError("bad_request", f"payload_nbytes {n!r} outside [0, {max_payload_bytes}]", fatal=True)
    _check_table(header.get("arrays"), n)
    return header


def _check_table(table, payload_nbytes: int) -> None:
    if not isinstance(table, list):
        raise ProtocolError("bad_request", "array table must be a list", fatal=True)
    names, offset = set(), 0
    for e in table:
        if not isinstance(e, dict) or set(e) != {"name", "dtype", "shape", "offset", "nbytes"}:
            raise ProtocolError("bad_request", f"array table entry {e!r} malformed", fatal=True)
        name, dtype, shape = e["name"], e["dtype"], e["shape"]
        if not isinstance(name, str) or not name or name in names:
            raise ProtocolError("bad_request", f"array name {name!r} empty or repeated", fatal=True)
        if dtype not in DTYPES:
            raise ProtocolError("bad_request", f"{name}: dtype {dtype!r} not allowed", fatal=True)
        if not isinstance(shape, list) or len(shape) > MAX_NDIM or any(type(s) is not int or s < 0 for s in shape):
            raise ProtocolError("bad_request", f"{name}: invalid shape {shape!r}", fatal=True)
        if e["offset"] != offset:
            raise ProtocolError("bad_request", f"{name}: arrays must be contiguous in table order", fatal=True)
        expected = math.prod(shape) * np.dtype(dtype).itemsize
        if e["nbytes"] != expected:
            raise ProtocolError("bad_request", f"{name}: nbytes {e['nbytes']} != {expected} for {dtype}{shape}",
                                fatal=True)
        names.add(name)
        offset += expected
    if offset != payload_nbytes:
        raise ProtocolError("bad_request", f"array table covers {offset} bytes, payload has {payload_nbytes}",
                            fatal=True)


def parse_arrays(header: dict, payload: bytearray) -> dict:
    """Zero-copy views into the received payload (writable: the payload is a fresh bytearray)."""
    out = {}
    for e in header["arrays"]:
        count = math.prod(e["shape"])
        if e["dtype"] == "|b1" and count and np.frombuffer(payload, np.uint8, count, e["offset"]).max() > 1:
            raise ProtocolError("bad_request", f"{e['name']}: bool bytes must be 0 or 1", fatal=True)
        out[e["name"]] = np.frombuffer(payload, dtype=np.dtype(e["dtype"]), count=count,
                                       offset=e["offset"]).reshape(tuple(e["shape"]))
    return out


def decode(data: bytes, *, max_header_bytes: int = 1 << 20, max_payload_bytes: int = 1 << 26) -> Message:
    """One complete message held in memory (tests, offline checks)."""
    if len(data) < PREFIX.size:
        raise ProtocolError("bad_request", "truncated message prefix", fatal=True)
    (n,) = PREFIX.unpack_from(data)
    if n == 0 or n > max_header_bytes:
        raise ProtocolError("bad_request", f"header length {n} outside [1, {max_header_bytes}]", fatal=True)
    header = parse_header(data[PREFIX.size:PREFIX.size + n], max_payload_bytes=max_payload_bytes)
    payload = bytearray(data[PREFIX.size + n:])
    if len(payload) != header["payload_nbytes"]:
        raise ProtocolError("bad_request", "payload length differs from payload_nbytes", fatal=True)
    return Message(header, parse_arrays(header, payload), len(data))


class Wire:
    """Blocking message I/O over one connected TCP socket.

    The socket keeps ONE fixed timeout (``io_timeout_s``) for every recv/send call, so concurrent senders and the reader
    never change each other's timeout. Waiting for the NEXT message uses ``select`` polls (stop checks, deadlines);
    once its first byte arrived a message must complete within ``io_timeout_s``.
    """

    def __init__(self, sock: socket.socket, *, max_header_bytes: int = 1 << 20, max_payload_bytes: int = 1 << 26,
                 io_timeout_s: float = 5.0):
        self.sock = sock
        self.max_header_bytes, self.max_payload_bytes = int(max_header_bytes), int(max_payload_bytes)
        self.io_timeout_s = float(io_timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.io_timeout_s)
        self.closed = False
        self.bytes_sent = 0
        self.bytes_received = 0

    def send(self, data: bytes) -> None:
        self.sock.sendall(data)
        self.bytes_sent += len(data)

    def _read_exact(self, n: int, deadline: float) -> bytearray:
        """``n`` bytes of a message already started. Not complete by ``deadline`` -> ConnectionError: the stream position
        is lost, so the connection must be dropped (never a TimeoutError that a caller might retry on this stream)."""
        buf = bytearray(n)
        view, got = memoryview(buf), 0
        while got < n:
            left = deadline - time.monotonic()
            try:
                ready = select.select([self.sock], [], [], max(0.0, left))[0]  # bytes already here are read
                k = self.sock.recv_into(view[got:], n - got) if ready else None
            except (ValueError, OSError) as exc:
                raise ConnectionError(f"connection failed mid-message ({got}/{n} bytes): {exc}") from exc
            if k is None:
                raise ConnectionError(f"message incomplete at the deadline ({got}/{n} bytes); stream position lost")
            if k == 0:
                raise ConnectionError(f"peer closed mid-message ({got}/{n} bytes)")
            got += k
        return buf

    def wait_readable(self, timeout_s: float | None, should_stop=None, poll_s: float = 0.2) -> bool:
        """True when bytes (or EOF) are readable; False on timeout. ``should_stop()`` -> raise Stopped."""
        end = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            if should_stop is not None and should_stop():
                raise Stopped()
            left = poll_s if end is None else min(poll_s, end - time.monotonic())
            if left <= 0:
                return False
            if self.closed:
                raise ConnectionError("connection closed")
            try:
                readable, _, _ = select.select([self.sock], [], [], left)
            except (ValueError, OSError) as exc:  # closed by another thread meanwhile
                raise ConnectionError(f"connection closed ({exc})") from exc
            if readable:
                return True

    def recv(self, timeout_s: float | None = None, should_stop=None) -> Message | None:
        """Next message; None on a clean EOF between messages; TimeoutError when none STARTED within ``timeout_s``.
        A started message must complete within ``timeout_s`` (from the call) and ``io_timeout_s``, else
        ConnectionError (the connection is unusable)."""
        start = time.monotonic()
        if not self.wait_readable(timeout_s, should_stop):
            raise TimeoutError("no message within the wait time")
        try:
            first = self.sock.recv(PREFIX.size)
        except OSError as exc:
            raise ConnectionError(f"connection failed: {exc}") from exc
        if not first:
            return None
        deadline = time.monotonic() + self.io_timeout_s
        if timeout_s is not None:
            deadline = min(deadline, start + timeout_s)
        prefix = bytes(first) + bytes(self._read_exact(PREFIX.size - len(first), deadline))
        (n,) = PREFIX.unpack(prefix)
        if n == 0 or n > self.max_header_bytes:
            raise ProtocolError("bad_request", f"header length {n} outside [1, {self.max_header_bytes}]", fatal=True)
        header = parse_header(self._read_exact(n, deadline), max_payload_bytes=self.max_payload_bytes)
        payload = self._read_exact(header["payload_nbytes"], deadline)
        t_recv = time.monotonic_ns()
        size = PREFIX.size + n + len(payload)
        self.bytes_received += size
        return Message(header, parse_arrays(header, payload), size, t_recv)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


def error_message(code: str, detail: str, *, session_id: str | None = None, seq: int | None = None,
                  history_reset: bool = False, fatal: bool = False) -> bytes:
    if code not in ERROR_CODES:
        raise ValueError(f"unknown error code {code!r}")
    return encode({"type": ERROR, "code": code, "detail": str(detail), "session_id": session_id, "seq": seq,
                   "ack": seq, "history_reset": bool(history_reset), "fatal": bool(fatal)})


# ------------------------------------------------------------------------------------------------ JSON / outputs
def jsonable(value):
    """Plain JSON value of numpy scalars/arrays inside dicts/lists (floats keep their exact repr round trip)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"not JSON-serializable: {type(value).__name__}")


def pack_outputs(outputs: dict, prefix: str = "out.") -> tuple[dict, dict, dict]:
    """DeployPolicy.infer outputs -> (JSON fields, array markers, arrays). numpy arrays and numpy scalars travel as
    arrays at their exact dtype (scalars as shape ()), everything else as JSON."""
    plain, markers, arrays = {}, {}, {}
    for key, value in outputs.items():
        if isinstance(value, (np.ndarray, np.generic)):
            name = prefix + key
            arrays[name] = np.asarray(value)
            markers[key] = {"array": name, "scalar": isinstance(value, np.generic)}
        else:
            plain[key] = jsonable(value)
    return plain, markers, arrays


def unpack_outputs(plain: dict, markers: dict, arrays: dict) -> dict:
    out = dict(plain)
    for key, m in markers.items():
        a = arrays[m["array"]]
        out[key] = a[()] if m["scalar"] else a
    return out


def validation_noise(seed: int) -> np.ndarray:
    """Explicit FQL noise of validation mode from an integer seed: identical on client, server and direct runs."""
    if type(seed) is not int or seed < 0:
        raise ValueError("noise seed must be a nonnegative integer")
    return np.random.default_rng(seed).standard_normal((1, ACTION_DIM), dtype=np.float32)
