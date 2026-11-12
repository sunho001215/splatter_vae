"""Robot-side CPU client of the policy server. numpy + stdlib only: importing this module never imports torch.

    client = PolicyClient(host, port, policy_id="dynamo-drawer-open", choices={...}, token=...)
    client.connect()                                   # WELCOME -> HELLO -> SPEC (bound InputSpec JSON)
    client.add_frame_native(cam, frame_id, capture_ns, rgb, pose7=..., gripper_register=...)   # crop/resize on CPU
    result = client.step({cam: frame_id}, reference_id, pose7, gripper_register, pose_time_ns=..., gripper_time_ns=...)
    if result.terminal: stop the robot loop (another client took the policy over)
    ok, why = check_action(result, reference_id=..., now_ns=time.monotonic_ns(), max_age_ms=...)

The client buffers the last frames of each camera and sends only frames the server does not hold yet; the server
assembles the trained history. Every timing is on THIS host's ``time.monotonic_ns`` clock; live capture times come from
``DeviceClock`` (the recorder's mapping of the camera's global-time stamp, so the frame times follow the training
convention). Disconnects are retried with backoff and the session is resumed (session id + last ack); unacknowledged
requests are re-sent with their ORIGINAL seq (the server answers a repeated seq from its cache, never re-infers). A lost
session (server restart / expiry) opens a new one: in-flight requests then return ``session_unknown`` and the buffered
frames are re-sent with the next step. A session taken over by another client is TERMINAL: every step returns
``code == "session_taken_over"`` (``StepResult.terminal``) until ``connect()`` is called again explicitly.

An action is bound to the measured pose of its observation (``reference_id``): execute its absolute ``flange_pose``
goal (through the robot-side safety layer) or nothing - never re-apply its delta to a newer pose. This module never
talks to a robot.
"""

from __future__ import annotations

import collections
import hashlib
import hmac
import socket
import time
from dataclasses import dataclass, field

import numpy as np

from ops.policy_serving import protocol as P
from ops.policy_serving.config import RESTRICTED_EXECUTION_CAPABILITY

# What this client implements (declared at HELLO only when the offered entry requires it, so a HELLO to a server
# without requirements is unchanged): a restricted entry executable by user decision is shown with its failed
# criterion and decision before START (scripts/robot_pc/infer.py); dry run / replay never execute.
CAPABILITIES = frozenset({RESTRICTED_EXECUTION_CAPABILITY})

TERMINAL_CODES = frozenset({"session_taken_over"})


class ServingError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


class ConnectionLost(ServingError):
    pass


class SessionTakenOver(ServingError):
    """The robot loop's STOP signal: another client holds this policy (takeover, or the policy is busy when this client
    tried to renew its lost session). ``submit`` raises it; ``step`` returns it as a terminal ``StepResult``."""

    def __init__(self, detail: str):
        super().__init__("session_taken_over", detail)


@dataclass
class PendingFrame:
    camera_id: str
    frame_id: int
    capture_ns: int
    rgb: np.ndarray
    pose7: list
    gripper_register: float
    depth: np.ndarray | None = None
    valid: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    preprocess_ns: tuple[int, int] | None = None  # (start, end) on the client monotonic clock
    sent: bool = False  # sent at least once: a re-send (renewal / history reset) is not new work for e2e timing


@dataclass
class StepResult:
    status: str  # "ok" | "error" | "timeout"
    seq: int
    session_id: str | None = None
    code: str | None = None
    detail: str | None = None
    reference_id: object = None
    outputs: dict = field(default_factory=dict)
    server_ms: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    request_bytes: int = 0
    response_bytes: int = 0
    frames_sent: int = 0
    history_reset: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def terminal(self) -> bool:
        """Stop the robot loop: this client no longer holds the policy (``connect()`` again only deliberately)."""
        return self.code in TERMINAL_CODES


@dataclass
class _Inflight:
    seq: int
    data: bytes
    reference_id: object
    end_ids: dict
    frames_sent: int
    preprocess_ms: float
    t_origin_ns: int
    t_serialize_ns: int
    t_serialized_ns: int
    capture_ns: int
    resent_frames: int = 0
    resent_wait_ms: float = 0.0
    t_sent_ns: int = 0


def check_action(result: StepResult, *, reference_id, now_ns: int, max_age_ms: float,
                 session_id: str | None = None) -> tuple[bool, str]:
    """Whether an action may be executed NOW. It is usable only for the measured pose it is bound to (same
    ``reference_id``; the robot executes its absolute ``flange_pose``, never its delta re-applied to a newer pose) and
    only while its observation is at most ``max_age_ms`` old. The age is measured from the observation's CAPTURE time
    (``DeviceClock``: the camera's global-time stamp on this host's monotonic clock, as the recorder stamps training
    frames), not from frame arrival, so it includes the camera-to-host latency."""
    if not result.ok:
        return False, f"no action ({result.status}: {result.code} {result.detail})"
    if result.reference_id != reference_id:
        return False, f"bound to measured pose {result.reference_id!r}, not {reference_id!r}"
    if session_id is not None and result.session_id != session_id:
        return False, "action from another session"
    age_ms = (int(now_ns) - int(result.outputs["obs_ids"]["observation_time_ns"])) / 1e6
    if age_ms < 0:
        return False, f"observation {-age_ms:.1f} ms in the future: frames not stamped on this clock"
    if age_ms > max_age_ms:
        return False, f"expired: observation {age_ms:.1f} ms old > {max_age_ms} ms"
    return True, f"usable: observation {age_ms:.1f} ms old"


def bound_command(result: StepResult) -> dict:
    """The executable part of an action: absolute goal pose + gripper, with the measured pose it was computed from."""
    o = result.outputs
    out = {k: o[k] for k in ("flange_pose", "gripper_closure", "gripper_register", "measured_pose7") if k in o}
    if "execution_flange_pose" in o:
        out["execution_flange_pose"] = o["execution_flange_pose"]
    out["reference_id"] = result.reference_id
    return out


class PolicyClient:
    def __init__(self, host: str, port: int, *, policy_id: str, choices: dict, token: bytes | None = None,
                 connect_timeout_s: float = 3.0, io_timeout_s: float = 5.0, step_timeout_s: float = 0.5,
                 reconnect_backoff_s=(0.05, 0.1, 0.25, 0.5, 1.0), ttl_ms: float = 100.0, takeover: bool = False,
                 frame_buffer: int | None = None, max_inflight: int = 4, max_payload_bytes: int = 32 << 20,
                 name: str = "policy-client", hello_timeout_s: float = 20.0):
        self.host, self.port, self.policy_id = host, int(port), policy_id
        self.choices = choices if callable(choices) else P.jsonable(choices)  # callable(unbound spec) -> dict
        self.token, self.takeover, self.name = token, bool(takeover), name
        self.connect_timeout_s, self.io_timeout_s = float(connect_timeout_s), float(io_timeout_s)
        # HELLO -> SPEC wait: the server loads a policy lazily on its first selection (load + warm-up, seconds under
        # GPU contention), so it is longer than the TCP connect / WELCOME wait
        self.hello_timeout_s = max(float(hello_timeout_s), self.connect_timeout_s)
        self.step_timeout_s, self.ttl_ms = float(step_timeout_s), float(ttl_ms)
        self.backoff = tuple(float(x) for x in reconnect_backoff_s) or (0.1,)
        self.frame_buffer, self.max_inflight, self.max_payload_bytes = frame_buffer, int(max_inflight), max_payload_bytes
        self.wire: P.Wire | None = None
        self.welcome: dict | None = None
        self.session_id: str | None = None
        self.spec: dict | None = None
        self.spec_info: dict | None = None  # the bound policy's CURRENT catalog labels (SPEC ``info``), not WELCOME's
        self.limits: dict | None = None
        self.taken_over: str | None = None  # terminal: why this client no longer holds the policy
        self.last_ack = 0  # newest seq answered in this session (resume)
        self.last_action_seq = 0  # newest seq answered with an ACTION: an older ACTION arriving later is refused
        self._seq = 0
        self._renewing = False
        self._buffer: dict[str, collections.OrderedDict] = {}
        self._added: dict[str, int] = {}
        self._tip: dict[str, int] = {}
        self._sets: collections.OrderedDict = collections.OrderedDict()  # key -> [surface_disp_m, sent]
        self._inflight: collections.OrderedDict[int, _Inflight] = collections.OrderedDict()
        self._results: dict[int, StepResult] = {}
        self._pongs: collections.deque = collections.deque(maxlen=16)
        self._capacity = 1
        self._deadline: float | None = None  # absolute deadline of the step being waited for (monotonic s)
        self.stats = collections.Counter()

    # ------------------------------------------------------------------------------------------------ connection
    @property
    def cameras(self) -> list:
        return list(self.spec["camera_ids"])

    def connect(self) -> dict:
        """Open the connection and a NEW session (also after a takeover); returns the bound InputSpec JSON."""
        self.taken_over = None
        self._open()
        self._hello(None)
        return self.spec

    def _budget(self, limit: float) -> float:
        """``limit`` bounded by the deadline of the step being waited for (reconnects/renewals never outlast it)."""
        if self._deadline is None:
            return limit
        return max(0.05, min(limit, self._deadline - time.monotonic()))

    def _open(self) -> None:
        if self.wire is not None:
            self.wire.close()
        sock = socket.create_connection((self.host, self.port), timeout=self._budget(self.connect_timeout_s))
        self.wire = P.Wire(sock, io_timeout_s=self.io_timeout_s, max_payload_bytes=self.max_payload_bytes)
        msg = self.wire.recv(self._budget(self.connect_timeout_s))
        if msg is not None and msg.type == P.ERROR:
            raise ServingError(msg.header["code"], msg.header["detail"])
        if msg is None or msg.type != P.WELCOME:
            raise ServingError("bad_request", f"expected WELCOME, got {None if msg is None else msg.type}")
        if msg.header["v"] != P.VERSION:
            raise ServingError("version", f"server protocol {msg.header['v']} != client {P.VERSION}")
        self.welcome = msg.header

    def _hello(self, resume: dict | None, timeout_s: float | None = None) -> dict:
        offered = self.welcome.get("policies", {})
        if self.policy_id not in offered:
            raise ServingError("bad_request", f"server does not serve {self.policy_id!r}; serves {sorted(offered)}")
        required = (offered[self.policy_id].get("info") or {}).get("requires_client") or []
        lacking = sorted(set(required) - CAPABILITIES)
        if lacking:
            raise ServingError("bad_request", f"{self.policy_id} requires client capabilities {lacking} that this "
                                              "client does not implement")
        if callable(self.choices):
            self.choices = P.jsonable(self.choices(offered[self.policy_id]["input_spec"]))
        header = {"type": P.HELLO, "policy_id": self.policy_id, "choices": self.choices, "client": self.name,
                  "takeover": self.takeover, "resume": resume}
        if required:
            header["capabilities"] = sorted(set(required))
        if self.welcome["auth_required"]:
            if self.token is None:
                raise ServingError("auth", "server requires the shared token")
            header["auth"] = hmac.new(self.token, self.welcome["nonce"].encode(), hashlib.sha256).hexdigest()
        self.wire.send(P.encode(header))
        deadline = time.monotonic() + (self.hello_timeout_s if timeout_s is None else timeout_s)
        while True:
            msg = self.wire.recv(max(0.0, deadline - time.monotonic()))
            if msg is None:
                raise ConnectionLost("disconnected", "server closed during HELLO")
            if msg.type == P.SPEC:
                break
            if msg.type == P.ERROR and msg.header.get("seq") is None:  # the HELLO's own answer
                raise ServingError(msg.header["code"], msg.header["detail"])
            self._handle(msg)  # late answers of the previous session's requests (never a nested HELLO: _renewing)
        h = msg.header
        bound_requires = sorted(set((h.get("info") or {}).get("requires_client") or []) - CAPABILITIES)
        if bound_requires:  # the bound policy's CURRENT labels require what this client does not implement
            raise ServingError("bad_request", f"{self.policy_id} (bound) requires client capabilities "
                                              f"{bound_requires} that this client does not implement")
        if resume is None or h["session_id"] != self.session_id:
            self._new_session_state(h["session_id"])
        self.spec, self.limits = h["input_spec"], h["limits"]
        self.spec_info = dict(h.get("info") or {})
        span = (self.spec["history_length"] - 1) * self.spec["frame_spacing"] + 1
        cap = max(span, int(self.frame_buffer or span))
        for cam in self.spec["camera_ids"]:
            old = self._buffer.get(cam)
            self._buffer[cam] = collections.OrderedDict() if old is None else old
            while len(self._buffer[cam]) > cap:
                self._buffer[cam].popitem(last=False)
        self._capacity = cap
        return h

    def _new_session_state(self, session_id: str) -> None:
        lost = list(self._inflight)
        for seq in lost:
            self._results[seq] = StepResult("error", seq, self.session_id, "session_unknown",
                                            f"server lost session {self.session_id}; new session {session_id}")
        self._inflight.clear()
        self.session_id = session_id
        self.last_ack = self.last_action_seq = 0
        self._tip.clear()
        for v in self._sets.values():
            v[1] = False
        if lost:
            self.stats["sessions_renewed"] += 1

    def _taken(self, detail: str) -> None:
        """Terminal: answer every in-flight request with ``session_taken_over`` and refuse further steps."""
        self.taken_over = detail
        for seq in list(self._inflight):
            self._results[seq] = StepResult("error", seq, self.session_id, "session_taken_over", detail)
        self._inflight.clear()
        self.stats["taken_over"] += 1

    def _renew(self) -> None:
        """The server no longer knows this session (expired / restarted): open a new one on this connection, once. If
        another client now holds the policy, this client is taken over (terminal)."""
        if self._renewing:
            self.stats["renew_nested_ignored"] += 1
            return
        self._renewing = True
        try:
            self._hello(None, timeout_s=self._budget(self.hello_timeout_s))
        except ServingError as exc:
            if exc.code == "session_busy":
                self._taken(f"policy held by another client after this session was lost: {exc.detail}")
            else:  # disconnect, refusal (e.g. policy withdrawn), auth: an error result, never an exception
                self._renew_failed(exc, "session_unknown" if isinstance(exc, ConnectionLost) else exc.code)
        except (TimeoutError, OSError, ConnectionError, P.ProtocolError) as exc:
            self._renew_failed(exc)
        finally:
            self._renewing = False

    def _renew_failed(self, exc, code: str = "session_unknown") -> None:
        """Renewal not done (deadline, disconnect): drop the connection (a late SPEC must never be mistaken for an
        answer) and answer the in-flight requests; the next step reconnects (a still-held policy -> takeover)."""
        self.stats["renew_failed"] += 1
        if self.wire is not None:
            self.wire.close()
            self.wire = None
        for seq in list(self._inflight):
            self._results[seq] = StepResult("error", seq, self.session_id, code,
                                            f"session renewal did not complete ({exc})")
        self._inflight.clear()

    def _recover(self, deadline: float) -> None:
        """Reconnect with backoff and resume (or renew) the session; re-send unacknowledged requests."""
        self.stats["reconnects"] += 1
        attempt = 0
        while True:
            if self.wire is not None:
                self.wire.close()
                self.wire = None
            delay = self.backoff[min(attempt, len(self.backoff) - 1)]
            attempt += 1
            if time.monotonic() + delay > deadline:
                raise ConnectionLost("disconnected", f"could not reconnect to {self.host}:{self.port} in time")
            time.sleep(delay)
            try:
                self._open()
                try:
                    h = self._hello({"session_id": self.session_id, "last_ack": self.last_ack},  # bounded by the
                                    timeout_s=max(0.05, deadline - time.monotonic()))  # caller's step deadline
                except ServingError as exc:
                    if exc.code != "session_unknown":
                        raise
                    self._renew()
                    return
                for inf in list(self._inflight.values()):
                    if inf.seq > h["last_seq_received"]:
                        self.wire.send(inf.data)
                        inf.t_sent_ns = time.monotonic_ns()
                self.stats["resumes"] += 1
                return
            except ConnectionLost:
                continue
            except (OSError, TimeoutError, ConnectionError, P.ProtocolError):
                continue

    def close(self) -> None:
        if self.wire is not None:
            try:
                if self.session_id is not None and self.taken_over is None:
                    self.wire.send(P.encode({"type": P.BYE, "session_id": self.session_id}))
            except OSError:
                pass
            self.wire.close()
            self.wire = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------------------------------------ frames
    def _check_geometry(self, camera_id: str, native_hw, K) -> tuple[list, list | None]:
        """The frame's native colour grid (and, for calibrated sessions, its intrinsics at the training grid) must be the
        ones bound at HELLO: the server crops K from the session's native grid, so a different stream would silently
        get wrong intrinsics."""
        if camera_id not in self._buffer:
            raise ValueError(f"camera {camera_id!r} not in this session {self.cameras}")
        want = [int(x) for x in self.spec["native_hw"][camera_id]]
        got = [int(x) for x in native_hw]
        if got != want:
            raise ValueError(f"{camera_id}: native grid {got} differs from the session's {want}")
        if not self.spec["needs_calibration"]:
            return got, None
        if K is None:
            raise ValueError(f"{camera_id}: this condition needs the frame intrinsics (K_native) to verify them")
        K = np.asarray(K, np.float64)
        bound = np.asarray(self.spec["calibration"][camera_id]["K"], np.float64)
        if K.shape != (3, 3) or not np.allclose(K, bound, rtol=0, atol=1e-9):
            raise ValueError(f"{camera_id}: frame intrinsics {K.tolist()} differ from the session's {bound.tolist()}")
        return got, K.tolist()

    def add_frame(self, camera_id: str, frame_id: int, capture_ns: int, rgb: np.ndarray, *, pose7, gripper_register,
                  native_hw, K=None, depth: np.ndarray | None = None, valid: np.ndarray | None = None,
                  metadata: dict | None = None, preprocess_ns: tuple[int, int] | None = None) -> None:
        """An already preprocessed frame (training grid) of a native colour frame of ``native_hw`` (with ``K``, its
        intrinsics at the training grid, for calibrated conditions). ``frame_id`` is the camera's acquisition counter
        (gaps stay gaps), ``capture_ns`` its capture time on the common client monotonic clock; pose/gripper measured
        at it."""
        hw, K = self._check_geometry(camera_id, native_hw, K)
        if type(frame_id) is not int or type(capture_ns) is not int:
            raise ValueError("frame_id and capture_ns must be integers")
        if frame_id <= self._added.get(camera_id, -1):
            raise ValueError(f"{camera_id}: frame {frame_id} is not newer than {self._added[camera_id]}")
        pose = [float(x) for x in np.asarray(pose7, np.float64).reshape(7)]
        meta = {**dict(metadata or {}), "native_hw": hw}
        if K is not None:
            meta["K"] = K
        buf = self._buffer[camera_id]
        buf[frame_id] = PendingFrame(camera_id, frame_id, capture_ns, np.ascontiguousarray(rgb), pose,
                                     float(gripper_register), None if depth is None else np.ascontiguousarray(depth),
                                     None if valid is None else np.ascontiguousarray(valid), meta, preprocess_ns)
        self._added[camera_id] = frame_id
        while len(buf) > self._capacity:
            buf.popitem(last=False)

    def add_frame_native(self, camera_id: str, frame_id: int, capture_ns: int, rgb: np.ndarray, *, pose7,
                         gripper_register, depth: np.ndarray | None = None, valid: np.ndarray | None = None,
                         K_native: np.ndarray | None = None, native_correction: dict | None = None,
                         metadata: dict | None = None) -> dict:
        """Native colour frame (+ native depth ALIGNED to colour, metres, with validity) -> the exact training crop +
        resize of ``s4d.rwlearn.deploy.preprocess_frame`` on this CPU, timed, then ``add_frame``. Refused unless the
        frame's grid (and intrinsics, for calibrated conditions) are the session's. Depth is labelled native RealSense
        with its session native correction (the deploy contract checks both)."""
        from s4d.rwlearn.deploy import preprocess_frame

        self._check_geometry(camera_id, np.shape(rgb)[:2], None if not self.spec["needs_calibration"] else
                             self.spec["calibration"][camera_id]["K"])  # grid first: before any work
        if self.spec["needs_calibration"] and K_native is None:
            raise ValueError(f"{camera_id}: this condition needs K_native of the frame to verify its intrinsics")
        t0 = time.monotonic_ns()
        out = preprocess_frame(rgb, depth=depth, valid=valid, K_native=K_native,
                               out_size=int(self.spec["image_hw"][0]), native_correction=native_correction)
        t1 = time.monotonic_ns()
        meta = dict(metadata or {})
        if depth is not None:
            meta.setdefault("depth_source", "native_realsense")
            meta.setdefault("native_correction", native_correction)
        self.add_frame(camera_id, frame_id, capture_ns, out["rgb"], pose7=pose7, gripper_register=gripper_register,
                       native_hw=np.shape(rgb)[:2], K=out.get("K"), depth=out.get("depth"), valid=out.get("valid"),
                       metadata=meta, preprocess_ns=(t0, t1))
        return out

    def add_capture(self, camera_id: str, capture: dict, *, pose7, gripper_register, metadata: dict | None = None):
        """A ``RealSenseCapture.read()`` result (already preprocessed on this CPU)."""
        meta = {**capture.get("metadata", {}), **(metadata or {})}
        self.add_frame(camera_id, capture["frame_id"], capture["capture_ns"], capture["rgb"], pose7=pose7,
                       gripper_register=gripper_register, native_hw=capture["native_hw"], K=capture.get("K"),
                       depth=capture.get("depth"), valid=capture.get("valid"), metadata=meta,
                       preprocess_ns=capture.get("preprocess_ns"))

    def add_sync_set(self, frame_ids: dict, surface_disp_m: float | None) -> None:
        """Multi-view set evidence (mutual-nearest frames + robot-surface motion, criterion M5) for one step;
        ``None`` = no valid robot state (refused by the server, never accepted). Recorded replay takes the value from
        the training store (``s4d.rwlearn.deploy.replay.m5_from_store``); LIVE multi-view must not run until the
        science side provides a validated M5 helper (API.md R3)."""
        key = tuple(int(frame_ids[c]) for c in self.cameras)
        if set(frame_ids) != set(self.cameras):
            raise ValueError(f"set must name exactly {self.cameras}")
        self._sets[key] = [None if surface_disp_m is None else float(surface_disp_m), False]
        while len(self._sets) > self._capacity:
            self._sets.popitem(last=False)

    def reset(self, timeout_s: float | None = None) -> None:
        """Explicit history reset (e.g. a new episode): server history, local frame buffers and in-flight requests."""
        if self.taken_over is not None:
            raise SessionTakenOver(self.taken_over)
        self._seq += 1
        seq = self._seq
        data = P.encode({"type": P.RESET, "session_id": self.session_id, "seq": seq})
        self._inflight.clear()
        self._results.clear()
        self._tip.clear()
        self._added.clear()
        self._sets.clear()
        for buf in self._buffer.values():
            buf.clear()
        self._inflight[seq] = _Inflight(seq, data, None, {}, 0, 0.0, 0, 0, 0, 0)
        self._send(data, time.monotonic() + (timeout_s or self.step_timeout_s))
        result = self.wait(seq, timeout_s)
        if result.status != "ok":
            raise ServingError(result.code or result.status, f"RESET failed: {result.detail}")

    # ------------------------------------------------------------------------------------------------ steps
    def _check_binding(self, request: dict) -> None:
        """Training binding rule before sending: the measured pose / gripper are the latest samples at or before the
        observation time (newest end frame), at most ``pose_max_age_ns`` / ``gripper_max_age_ns`` old."""
        t = [pf.capture_ns for c in self.cameras for f, pf in self._buffer[c].items() if f == request["end_ids"][c]]
        if len(t) != len(self.cameras):
            return  # end frame no longer buffered: the server's deploy contract still checks the binding
        obs = max(t)
        for key, bound in (("pose_time_ns", self.spec.get("pose_max_age_ns")),
                           ("gripper_time_ns", self.spec.get("gripper_max_age_ns"))):
            if bound is not None and not 0 <= obs - request[key] <= int(bound):
                raise ValueError(f"{key} {request[key]} is not the sample of the observation at {obs} "
                                 f"(training: 0 <= age <= {int(bound) / 1e6:.0f} ms)")

    def submit(self, end_ids: dict, reference_id, pose7, gripper_register, *, pose_time_ns: int, gripper_time_ns: int,
               ttl_ms: float | None = None, pad_run_start: bool = False, noise: np.ndarray | None = None,
               noise_seed: int | None = None) -> int:
        """Send one inference request (only frames the server does not hold yet); returns its seq.

        ``pose7`` / ``gripper_register``: the measured robot / gripper samples bound to the observation (training rule:
        the latest sample at or before the observation time, <= 15 / 100 ms old) and their sample times on the SAME
        client monotonic clock as the frames (checked here). ``pad_run_start``: the training run-start padding (first
        frames of a run); otherwise an incomplete history is refused. Raises ``SessionTakenOver`` once taken over."""
        if self.taken_over is not None:
            raise SessionTakenOver(self.taken_over)
        t_serialize = time.monotonic_ns()
        if len(self._inflight) >= self.max_inflight:
            raise ServingError("overloaded", f"{len(self._inflight)} requests already in flight")
        request = {"end_ids": {c: int(end_ids[c]) for c in self.cameras}, "reference_id": reference_id,
                   "pose7": [float(x) for x in np.asarray(pose7, np.float64).reshape(7)],
                   "pose_time_ns": int(pose_time_ns), "gripper_register": float(gripper_register),
                   "gripper_time_ns": int(gripper_time_ns), "ttl_ms": float(ttl_ms or self.ttl_ms),
                   "pad_run_start": bool(pad_run_start)}
        self._check_binding(request)
        frames, arrays, origin, pre_ms, sent, sending = [], {}, None, 0.0, {}, []
        resent, resent_origin = 0, None
        for cam in self.cameras:
            for fid, pf in self._buffer[cam].items():
                if fid <= self._tip.get(cam, -1):
                    continue
                i = len(frames)
                names = {"rgb": f"f{i}.rgb"}
                arrays[names["rgb"]] = pf.rgb
                if pf.depth is not None:
                    names["depth"], names["valid"] = f"f{i}.depth", f"f{i}.valid"
                    arrays[names["depth"]], arrays[names["valid"]] = pf.depth, pf.valid
                frames.append({"camera_id": cam, "frame_id": fid, "capture_ns": pf.capture_ns, "pose7": pf.pose7,
                               "gripper_register": pf.gripper_register, "arrays": names, "metadata": pf.metadata})
                if pf.sent:  # re-send after a renewal / history reset: its preprocessing is not this step's work
                    resent += 1
                    if pf.preprocess_ns is not None:
                        resent_origin = pf.preprocess_ns[0] if resent_origin is None else \
                            min(resent_origin, pf.preprocess_ns[0])
                elif pf.preprocess_ns is not None:
                    pre_ms += (pf.preprocess_ns[1] - pf.preprocess_ns[0]) / 1e6
                    origin = pf.preprocess_ns[0] if origin is None else min(origin, pf.preprocess_ns[0])
                sent[cam] = fid
                sending.append(pf)
        sets = [{"frame_ids": dict(zip(self.cameras, key)), "surface_disp_m": v[0]}
                for key, v in self._sets.items() if not v[1]]
        if noise is not None and noise_seed is not None:
            raise ValueError("give noise OR noise_seed")
        if noise is not None:
            noise = np.asarray(noise)
            if noise.dtype != np.float32 or noise.shape != (1, P.ACTION_DIM):
                raise ValueError(f"validation noise must be float32 (1, {P.ACTION_DIM})")
            arrays["noise"], request["noise"] = noise, "noise"
        if noise_seed is not None:
            request["noise_seed"] = int(noise_seed)
        self._seq += 1
        seq = self._seq
        data = P.encode({"type": P.STEP, "session_id": self.session_id, "seq": seq, "frames": frames, "sets": sets,
                         "request": request, "t_client_ns": t_serialize}, arrays)
        t_serialized = time.monotonic_ns()
        capture = max((pf.capture_ns for c in self.cameras for f, pf in self._buffer[c].items()
                       if f == request["end_ids"][c]), default=0)
        inf = _Inflight(seq, data, reference_id, request["end_ids"], len(frames), pre_ms,
                        t_serialize if origin is None else origin, t_serialize, t_serialized, capture, resent,
                        0.0 if resent_origin is None else (t_serialize - resent_origin) / 1e6)
        self._inflight[seq] = inf
        self._tip.update(sent)  # the bytes are kept until acknowledged: a resume re-sends them with this seq
        for pf in sending:
            pf.sent = True
        for v in self._sets.values():
            v[1] = True
        self._send(data, time.monotonic() + self.step_timeout_s)
        inf.t_sent_ns = time.monotonic_ns()
        return seq

    def _send(self, data: bytes, deadline: float) -> None:
        self._deadline = deadline
        try:
            if self.wire is None:
                raise ConnectionError("not connected")
            self.wire.send(data)
        except (OSError, ConnectionError):
            self._recover(deadline)  # re-sends every unacknowledged request, including this one
        finally:
            self._deadline = None

    def step(self, end_ids: dict, reference_id, pose7, gripper_register, *, pose_time_ns: int, gripper_time_ns: int,
             ttl_ms: float | None = None, pad_run_start: bool = False, noise: np.ndarray | None = None,
             noise_seed: int | None = None, timeout_s: float | None = None) -> StepResult:
        """One request + its response. Never raises for a lost connection or a taken-over session: those return an
        error ``StepResult`` (``terminal`` for a takeover - the robot loop's stop signal)."""
        try:
            seq = self.submit(end_ids, reference_id, pose7, gripper_register, pose_time_ns=pose_time_ns,
                              gripper_time_ns=gripper_time_ns, ttl_ms=ttl_ms, pad_run_start=pad_run_start,
                              noise=noise, noise_seed=noise_seed)
        except SessionTakenOver as exc:
            return StepResult("error", self._seq, self.session_id, exc.code, exc.detail)
        except ConnectionLost as exc:
            self._inflight.pop(self._seq, None)  # never re-send a request the caller has given up on
            self.stats["disconnected"] += 1
            return StepResult("error", self._seq, self.session_id, exc.code, exc.detail)
        return self.wait(seq, timeout_s)

    def wait(self, seq: int, timeout_s: float | None = None) -> StepResult:
        """The response of ``seq``. Older unclaimed results are dropped (superseded); later responses for requests
        no longer awaited are dropped as stale. Every read is bounded by this call's deadline; a message cut by the
        deadline or the network makes the connection unusable: it is dropped and resumed."""
        deadline = time.monotonic() + (self.step_timeout_s if timeout_s is None else timeout_s)
        self._deadline = deadline
        try:
            return self._wait(seq, deadline)
        finally:
            self._deadline = None

    def _wait(self, seq: int, deadline: float) -> StepResult:
        while seq not in self._results:
            if seq not in self._inflight:
                return StepResult("error", seq, self.session_id, "unknown_seq", "not in flight")
            left = deadline - time.monotonic()
            if left <= 0:
                self._inflight.pop(seq, None)
                self.stats["timeouts"] += 1
                return StepResult("timeout", seq, self.session_id, None, "no response within the step timeout")
            try:
                if self.wire is None:
                    raise ConnectionError("not connected")
                msg = self.wire.recv(left)
            except TimeoutError:  # no message STARTED in time (a started one raises ConnectionError)
                continue
            except P.ProtocolError as exc:
                if exc.fatal:
                    raise ServingError(exc.code, exc.detail) from exc
                continue
            except (OSError, ConnectionError):
                msg = None
                self.stats["connection_errors"] += 1
            if msg is None:
                try:
                    self._recover(deadline)
                except ConnectionLost as exc:
                    self._inflight.pop(seq, None)
                    self.stats["disconnected"] += 1
                    return StepResult("error", seq, self.session_id, exc.code, exc.detail)
                continue
            self._handle(msg)
        result = self._results.pop(seq)
        for old in [s for s in self._results if s < seq]:
            self._results.pop(old)
            self.stats["superseded_results"] += 1
        return result

    def ping(self, timeout_s: float = 1.0) -> float:
        if self.wire is None:
            raise ConnectionLost("disconnected", "not connected")
        t0 = time.monotonic_ns()
        self.wire.send(P.encode({"type": P.PING, "t_client_ns": t0}))
        deadline = time.monotonic() + timeout_s
        while True:
            while self._pongs:
                pong = self._pongs.popleft()
                if pong["t_client_ns"] == t0:
                    return (time.monotonic_ns() - t0) / 1e6
            msg = self.wire.recv(max(0.0, deadline - time.monotonic()))
            if msg is None:
                raise ConnectionLost("disconnected", "server closed")
            self._handle(msg)

    # ------------------------------------------------------------------------------------------------ responses
    def _handle(self, msg: P.Message) -> None:
        h, kind = msg.header, msg.type
        if kind == P.PONG:
            self._pongs.append(h)
            return
        if kind not in (P.ACTION, P.ERROR, P.RESET_OK):
            self.stats["unexpected_messages"] += 1
            return
        if kind == P.ERROR:
            if h.get("fatal"):
                raise ServingError(h["code"], h["detail"])
            if h["code"] == "session_taken_over":
                if h.get("session_id") == self.session_id:
                    self._taken(h["detail"])
                return
            if h["code"] == "session_unknown":  # expired / evicted / server restarted: renew on this connection
                self.stats["session_unknown"] += 1
                if h.get("session_id") == self.session_id and self.taken_over is None:
                    self._renew()  # once per lost session (pipelined requests repeat it; nested calls ignored)
                return
            if h.get("seq") is None:
                raise ServingError(h["code"], h["detail"])
        seq = h.get("seq")
        if h.get("session_id") != self.session_id:
            self.stats["foreign_responses"] += 1
            return
        inf = self._inflight.pop(seq, None)
        if inf is None:  # duplicate, superseded locally (timeout) or from before a reset
            self.stats["dropped_responses"] += 1
            return
        self.last_ack = max(self.last_ack, seq)
        if kind == P.ACTION and seq < self.last_action_seq:
            self._results[seq] = StepResult("error", seq, self.session_id, "stale",
                                            f"out-of-order action (seq {self.last_action_seq} already acted)")
            self.stats["out_of_order"] += 1
            return
        if kind == P.RESET_OK:
            self._results[seq] = StepResult("ok", seq, self.session_id)
            return
        if kind == P.ERROR:
            if h.get("history_reset"):
                self._tip.clear()
                for v in self._sets.values():
                    v[1] = False
            self._results[seq] = StepResult("error", seq, self.session_id, h["code"], h["detail"],
                                            response_bytes=msg.nbytes, request_bytes=len(inf.data),
                                            frames_sent=inf.frames_sent, history_reset=bool(h.get("history_reset")))
            return
        self.last_action_seq = seq
        outputs = P.unpack_outputs(h["outputs"], h["output_arrays"], msg.arrays)
        t_decoded = time.monotonic_ns()
        ids = outputs["obs_ids"]
        if h["reference_id"] != inf.reference_id or ids["reference_id"] != inf.reference_id \
                or any(ids["frame_ids"][c][-1] != inf.end_ids[c] for c in inf.end_ids):
            raise ServingError("bad_response", f"response seq {seq} is bound to another observation")
        server_ms = h["server_ms"]
        rtt = (msg.t_recv_ns - inf.t_serialized_ns) / 1e6
        timings = {"preprocess_ms": inf.preprocess_ms, "serialize_ms": (inf.t_serialized_ns - inf.t_serialize_ns) / 1e6,
                   "send_ms": (inf.t_sent_ns - inf.t_serialized_ns) / 1e6, "rtt_ms": rtt,
                   "network_ms": rtt - server_ms["total_ms"], "decode_ms": (t_decoded - msg.t_recv_ns) / 1e6,
                   "e2e_ms": (t_decoded - inf.t_origin_ns) / 1e6,  # first NEW frame's preprocessing -> decoded action
                   "capture_to_action_ms": (t_decoded - inf.capture_ns) / 1e6,
                   "resent_frames": inf.resent_frames, "resent_wait_ms": inf.resent_wait_ms}
        self._results[seq] = StepResult("ok", seq, self.session_id, None, None, inf.reference_id, outputs, server_ms,
                                        timings, len(inf.data), msg.nbytes, inf.frames_sent)


# ---------------------------------------------------------------------------------------------------- RealSense
GLOBAL_TIME = "global_time"
_DOMAINS = {0: "hardware_clock", 1: "system_time", 2: "global_time"}  # rs2_timestamp_domain


def timestamp_domain_name(domain) -> str:
    """librealsense timestamp domain (enum, its str, or the recorded integer) -> its name."""
    if isinstance(domain, (int, np.integer)):
        return _DOMAINS.get(int(domain), f"unknown({int(domain)})")
    name = getattr(domain, "name", None) or str(domain)
    return str(name).strip().lower().replace(" ", "_").split(".")[-1]


class DeviceClock:
    """Capture times on this host's CLOCK_MONOTONIC exactly as the panda_teleop_episode v2 recorder stamps the frames
    the policies were trained on (``cameras/cam_<serial>/t`` and ``stereo/t``):

        t = hw_timestamp_ms / 1e3 - (t_recv_unix - t_recv)

    i.e. the librealsense GLOBAL_TIME stamp (device clock steered to the host CLOCK_REALTIME, ms) minus the
    realtime-to-monotonic offset sampled together when the frame is received, refreshed for every frame. No pipeline
    latency is added: the capture time is the device stamp, not the arrival. Other timestamp domains are refused.
    The recorder's source is not in this repository; the rule is read from its recordings: it holds to <= 0.36 us on
    every colour and stereo frame of /home/ws/data/real_world/{drawer-open,mug-hanging,covering,pick-and-place,test}/
    episode_00000[01].h5 (5964 frames, timestamp_domain 2 = global_time), see the panda_teleop adapter docstring."""

    def __init__(self, realtime_ns=time.time_ns, monotonic_ns=time.monotonic_ns):
        self.realtime_ns, self.monotonic_ns = realtime_ns, monotonic_ns

    def offset_ns(self) -> int:
        """CLOCK_REALTIME - CLOCK_MONOTONIC now (both read together; monotonic bracketed around the realtime read)."""
        m0 = self.monotonic_ns()
        r = self.realtime_ns()
        m1 = self.monotonic_ns()
        return int(r) - (int(m0) + int(m1)) // 2

    def map(self, device_ms: float, domain, *, offset_ns: int | None = None) -> int:
        name = timestamp_domain_name(domain)
        if name != GLOBAL_TIME:
            raise ValueError(f"timestamp domain {name!r}: only global_time maps onto the training frame times")
        offset = self.offset_ns() if offset_ns is None else int(offset_ns)
        return int(round(float(device_ms) * 1e6)) - offset


class RealSenseCapture:
    """Optional live capture of ONE camera through ``s4d.rwlearn.deploy.realsense`` (pyrealsense2 imported here only).

    ``depth=True``: colour + native z16 depth, ``rs.align`` to colour and the shared crop/resize (``RealSenseRGBD``);
    ``depth=False``: colour only + ``preprocess_frame``. The stream must have the session's native grid (and colour
    intrinsics ``K_native`` when given: required for calibrated conditions); checked at start and on every frame.
    Frames carry the colour frame number (acquisition counter) and the recorder's capture time (``DeviceClock``);
    pass a result to ``PolicyClient.add_capture``."""

    def __init__(self, serial: str, *, native_hw, fps: int = 15, depth: bool = False, correction: dict | None = None,
                 K_native=None, out_size: int = 128, rs_module=None, pipeline=None, clock: DeviceClock | None = None,
                 keep_native: bool = False):
        if rs_module is None:
            import pyrealsense2 as rs_module
        from s4d.rwlearn.deploy.realsense import RealSenseRGBD

        self.rs, self.serial, self.depth, self.out_size = rs_module, str(serial), bool(depth), int(out_size)
        self.keep_native = bool(keep_native)  # also return the native RGB frame (session recording)
        self.native_hw = tuple(int(x) for x in native_hw)
        self.K_native = None if K_native is None else np.asarray(K_native, np.float64)
        h, w = self.native_hw
        self.pipeline = pipeline or rs_module.pipeline()
        cfg = rs_module.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs_module.stream.color, w, h, rs_module.format.rgb8, fps)
        if self.depth:
            cfg.enable_stream(rs_module.stream.depth, w, h, rs_module.format.z16, fps)
        self.profile = self.pipeline.start(cfg)
        try:
            self._check_stream(self.profile.get_stream(rs_module.stream.color).as_video_stream_profile())
        except ValueError:
            self.pipeline.stop()
            raise
        self.rgbd = RealSenseRGBD(self.profile, correction=correction, out_size=out_size,
                                  rs_module=rs_module) if self.depth else None
        self.clock = clock or DeviceClock()

    def _check_stream(self, video_profile) -> np.ndarray:
        from s4d.rwlearn.deploy.realsense import color_intrinsics

        intr = video_profile.get_intrinsics()
        if (int(intr.height), int(intr.width)) != self.native_hw:
            raise ValueError(f"{self.serial}: colour stream {intr.height}x{intr.width} is not the session's "
                             f"{self.native_hw[0]}x{self.native_hw[1]}")
        K = color_intrinsics(intr)
        if self.K_native is not None and not np.allclose(K, self.K_native, rtol=0, atol=1e-9):
            raise ValueError(f"{self.serial}: colour intrinsics {K.tolist()} differ from the session's "
                             f"{self.K_native.tolist()}")
        return K

    def read(self, timeout_ms: int = 1000) -> dict:
        from s4d.rwlearn.deploy import preprocess_frame

        frames = self.pipeline.wait_for_frames(timeout_ms)
        offset = self.clock.offset_ns()  # sampled at receipt, as the recorder does (t_recv_unix - t_recv)
        color = frames.get_color_frame()
        K = self._check_stream(color.profile.as_video_stream_profile())
        domain = color.get_frame_timestamp_domain()
        color_ms, number = float(color.get_timestamp()), int(color.get_frame_number())
        capture_ns = self.clock.map(color_ms, domain, offset_ns=offset)
        t0 = time.monotonic_ns()
        native_depth = depth_scale = None
        if self.keep_native and self.rgbd is not None:  # the camera's own z16 frame (not aligned), before processing
            d = frames.get_depth_frame()
            if d:
                native_depth = np.array(np.asanyarray(d.get_data()), np.uint16, copy=True)
                depth_scale = float(d.get_units())
        if self.rgbd is not None:
            out = self.rgbd.process(frames)
        else:
            rgb = np.asanyarray(color.get_data())
            if rgb.shape[:2] != self.native_hw:
                raise ValueError(f"{self.serial}: colour frame {rgb.shape[:2]} is not the session's grid")
            out = preprocess_frame(rgb, K_native=K, out_size=self.out_size)
            out["metadata"] = {"color_timestamp_ms": color_ms, "color_frame_number": number,
                               "color_timestamp_domain": timestamp_domain_name(domain)}
        t1 = time.monotonic_ns()
        if self.keep_native:
            out["native_rgb"] = np.array(np.asanyarray(color.get_data()), copy=True)
            if native_depth is not None:  # recorded only for depth policies (the rgbd path); never IR
                out["native_depth"], out["depth_scale_m"] = native_depth, depth_scale
        out.update(frame_id=number, capture_ns=capture_ns, preprocess_ns=(t0, t1), native_hw=list(self.native_hw))
        out["K"] = np.asarray(out["K"]).tolist()
        return out

    def stop(self) -> None:
        self.pipeline.stop()
