"""Persistent real-world policy server: ONE process, policies loaded once, one GPU worker, per-session history.

    .venv/bin/python -I ops/policy_serving/server.py serve --config configs/realworld/policy_serving.yaml --policies ID,...

Science stays in ``s4d.rwlearn.deploy``: ``DeployPolicy.reset_session`` binds a session (cameras, native grids,
static calibration, native depth correction), ``HistoryWindow`` holds the frames and assembles the trained history,
``DeployPolicy.infer`` validates and runs encoder + FQL + action conversion. This module only moves bytes, keeps
sessions/sequence numbers/queues and maps the deploy contract's refusals to explicit error codes:

- clients send only NEW frames; the history is assembled server-side at the trained spacing. A missing frame (or
  multi-view set) is an explicit ``incomplete_history`` error, never a silent re-spacing;
- one worker thread runs inference (one GPU); the queue is bounded; a newer request of a session supersedes its older
  queued one (``stale``); a request whose ``ttl_ms`` (counted from its arrival) expired before execution is ``stale``;
- a repeated seq returns the cached response without re-inference; reconnect with resume (session id + last ack)
  re-delivers responses completed meanwhile; sessions expire after ``session_idle_timeout_s`` without traffic;
- sessions of one policy share the loaded model (``reset_session(concurrent=True)`` / ``end_session``); at most
  ``max_sessions_per_policy`` live sessions per policy (default 1: one controller); another HELLO gets
  ``session_busy`` unless it asks ``takeover`` (evicts the least recently active session);
- optional shared-token handshake: HMAC-SHA256(token, per-connection nonce); the token file path is given at launch
  and the token is never sent, stored in source or logged.
Nothing here talks to a robot.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import queue
import secrets
import select
import signal
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # ``python ops/policy_serving/server.py``: make the repository importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from ops.policy_serving import protocol as P  # noqa: E402

LOG = logging.getLogger("ops.policy_serving.server")
HELLO_CHOICES = frozenset({"camera_id", "native_hw", "calibration", "native_depth_correction"})
FRAME_KEYS = frozenset({"camera_id", "frame_id", "capture_ns", "pose7", "gripper_register", "arrays", "metadata"})
REQUEST_KEYS = frozenset({"end_ids", "reference_id", "pose7", "pose_time_ns", "gripper_register", "gripper_time_ns",
                          "ttl_ms", "pad_run_start", "noise", "noise_seed"})


class Reject(Exception):
    """A request refused with an explicit ERROR code (never a silent fallback)."""

    def __init__(self, code: str, detail: str, *, history_reset: bool = False):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail, self.history_reset = code, detail, history_reset


def classify(exc: Exception) -> str:
    """Wire error code of a deploy refusal: by its CLASS (``s4d.rwlearn.deploy.errors``), never its message text."""
    from s4d.rwlearn.deploy import errors as E

    for cls, code in ((E.IncompleteHistory, "incomplete_history"), (E.IrregularHistory, "irregular_spacing"),
                      (E.Unsynchronized, "unsynchronized"), (E.BindingRefused, "binding_refused"),
                      (E.SessionMismatch, "session_unknown")):
        if isinstance(exc, cls):
            return code
    return "bad_request"


def is_loopback(host: str) -> bool:
    if host in ("localhost", "ip6-localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _numbers(values, n: int) -> bool:
    return isinstance(values, list) and len(values) == n and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in values)


def warmup_choices(spec, native_hw=(480, 640)) -> dict:
    """Synthetic session for start-up warm-up only (labelled; never used for a client's actions)."""
    cams = tuple(spec.trained_camera_orders[0]) if spec.views > 1 else (spec.trained_camera_orders[0][0],)
    h, w = (int(x) for x in native_hw)
    out = {"native_hw": {c: (h, w) for c in cams}}
    if spec.views == 1:
        out["camera_id"] = cams[0]
    if spec.needs_calibration:
        K = [[float(max(h, w)), 0.0, w / 2], [0.0, float(max(h, w)), h / 2], [0.0, 0.0, 1.0]]
        out["calibration"] = {c: {"K_native": K, "c2w": np.eye(4).tolist()} for c in cams}
    if "depth" in spec.modalities:
        out["native_depth_correction"] = {c: {"stage": "sensor_depth", "factor": 1.0, "validated": False,
                                              "evidence": "server warm-up: synthetic input, no correction"}
                                          for c in cams}
    return out


@dataclass(eq=False)
class ServedPolicy:
    policy_id: str
    policy: object  # s4d.rwlearn.deploy.DeployPolicy
    info: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)  # held during infer and reset_session
    sessions: dict = field(default_factory=dict)  # live sessions of this policy (id -> Session)
    warmup_ms: dict | None = None


@dataclass(eq=False)
class Job:
    session: Session
    seq: int
    window: object
    pose7: np.ndarray
    gripper: float
    obs_ids: dict
    noise: np.ndarray | None
    ttl_ms: float
    deadline_ns: int
    t_recv_ns: int
    t_enqueued_ns: int
    prep_ms: float


class Session:
    def __init__(self, sid: str, served: ServedPolicy, spec, history, capacity: int):
        self.id, self.served, self.spec, self.history = sid, served, spec, history
        self.capacity = capacity
        # mirror of the history: camera -> {frame id: capture_ns}, oldest first, same capacity as HistoryWindow
        self.held = {c: collections.OrderedDict() for c in spec.camera_ids}
        self.sets: collections.OrderedDict = collections.OrderedDict()  # frame-id tuple -> surface_disp_m
        self.last_seq = 0
        self.responses: collections.OrderedDict = collections.OrderedDict()  # seq -> encoded response
        self.pending: Job | None = None
        self.running: Job | None = None
        self.conn: Connection | None = None
        self.closed = False
        self.last_activity = time.monotonic()
        self.hist_lock = threading.Lock()
        self.counts = collections.Counter()

    def clear_history(self) -> None:
        self.history.reset()
        self.sets.clear()
        for d in self.held.values():
            d.clear()

    def hold(self, cam: str, frame) -> None:
        held = self.held[cam]
        held[frame.seq] = frame.timestamp_ns
        while len(held) > self.capacity:
            held.popitem(last=False)


class Connection:
    """One client connection: a reader thread (dispatch) and a sender thread draining a bounded outbox, so the
    inference worker never blocks on a slow client (a client that stops reading is disconnected when its outbox is
    full). All outbound messages of a connection go through the outbox in order."""

    OUTBOX = 256
    PRE_SESSION_BYTES = 64 << 10  # header and payload cap until HELLO bound a session (authenticated if a token is set)

    def __init__(self, server: PolicyServer, sock: socket.socket, addr):
        self.server, self.addr = server, addr
        self.wire = P.Wire(sock, max_header_bytes=min(self.PRE_SESSION_BYTES, server.max_header_bytes),
                           max_payload_bytes=min(self.PRE_SESSION_BYTES, server.max_payload_bytes),
                           io_timeout_s=server.io_timeout_s)
        self.session: Session | None = None
        self.nonce = secrets.token_hex(16)
        self.authenticated = server.token is None
        self.outbox: queue.Queue = queue.Queue(self.OUTBOX)
        self.closing = False
        self.thread = threading.Thread(target=self.run, name=f"policy-conn-{addr}", daemon=True)
        self.sender = threading.Thread(target=self._send_loop, name=f"policy-send-{addr}", daemon=True)

    def start(self) -> None:
        self.sender.start()
        self.thread.start()

    def bound(self) -> None:
        """A session is bound on this (authenticated) connection: the configured message limits apply from now on."""
        self.wire.max_header_bytes = self.server.max_header_bytes
        self.wire.max_payload_bytes = self.server.max_payload_bytes

    def send(self, data: bytes) -> bool:
        if self.closing or self.wire.closed:
            return False
        try:
            self.outbox.put_nowait(data)
            return True
        except queue.Full:
            LOG.warning("client %s stopped reading (%d messages pending); closing the connection", self.addr,
                        self.OUTBOX)
            self.close()
            return False

    def _send_loop(self) -> None:
        while True:
            data = self.outbox.get()
            if data is None:
                break
            try:
                self.wire.send(data)
            except OSError as exc:
                if not self.wire.closed:
                    LOG.info("send to %s failed (%s); closing the connection", self.addr, exc)
                break
        self.wire.close()

    def close(self, *, flush: bool = False) -> None:
        """``flush``: deliver what is queued (e.g. a fatal ERROR), then close; else close now."""
        self.closing = True
        if not flush:
            self.wire.close()
        try:
            self.outbox.put_nowait(None)
        except queue.Full:
            self.wire.close()

    def run(self) -> None:
        srv = self.server
        try:
            self.send(P.encode({"type": P.WELCOME, "protocol": P.PROTOCOL, "server": srv.name, "nonce": self.nonce,
                                "auth_required": srv.token is not None, "policies": srv.policy_specs(),
                                "t_server_ns": time.monotonic_ns()}))
            while not srv.stop_event.is_set():
                try:
                    msg = self.wire.recv(None, srv.stop_event.is_set)
                except P.Stopped:
                    break
                if msg is None:
                    break
                try:
                    if not srv.dispatch(self, msg):
                        break
                except P.ProtocolError as exc:
                    self.send(P.error_message(exc.code, exc.detail, fatal=exc.fatal))
                    if exc.fatal:
                        break
        except P.ProtocolError as exc:
            self.send(P.error_message(exc.code, exc.detail, fatal=True))
        except (OSError, TimeoutError, ConnectionError) as exc:
            if not srv.stop_event.is_set():
                LOG.info("connection %s ended: %s", self.addr, exc)
        except Exception:  # noqa: BLE001 - one broken connection must never take the server down
            LOG.exception("connection %s failed", self.addr)
        finally:
            srv.drop_connection(self)


class PolicyServer:
    """Serve ``{policy_id: DeployPolicy}`` (loaded once by the caller) on one TCP port."""

    def __init__(self, policies: dict, *, host: str = "127.0.0.1", port: int = 0, token: bytes | None = None,
                 max_queue: int = 4, session_idle_timeout_s: float = 30.0, response_cache: int = 32,
                 io_timeout_s: float = 5.0, max_header_bytes: int = 1 << 20, max_payload_bytes: int = 32 << 20,
                 max_frames_per_step: int = 64, history_capacity: int | None = None, warmup_iterations: int = 0,
                 max_sessions_per_policy: int = 1, max_connections: int = 16,
                 warmup_native_hw=(480, 640), info: dict | None = None, name: str = "s4d-policy-server"):
        if not policies:
            raise ValueError("no policies to serve")
        if token is not None and not token:
            raise ValueError("empty shared token")
        if token is None and not is_loopback(host):
            raise ValueError(f"binding {host!r} beyond loopback requires the shared token (--token-file)")
        self.max_connections = int(max_connections)
        if self.max_connections < 1:
            raise ValueError("max_connections must be positive")
        self.policies = {pid: ServedPolicy(pid, p, dict((info or {}).get(pid, {}))) for pid, p in policies.items()}
        self.host, self.port, self.token, self.name = host, int(port), token, name
        self.max_queue, self.idle_timeout_s = int(max_queue), float(session_idle_timeout_s)
        self.response_cache, self.io_timeout_s = int(response_cache), float(io_timeout_s)
        self.max_header_bytes, self.max_payload_bytes = int(max_header_bytes), int(max_payload_bytes)
        self.max_frames_per_step, self.history_capacity = int(max_frames_per_step), history_capacity
        self.warmup_iterations, self.warmup_native_hw = int(warmup_iterations), tuple(warmup_native_hw)
        self.max_sessions_per_policy = int(max_sessions_per_policy)
        if self.max_queue < 1 or self.response_cache < 1 or self.max_sessions_per_policy < 1:
            raise ValueError("max_queue, response_cache and max_sessions_per_policy must be positive")
        self.stop_event = threading.Event()
        self.stopping = False
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.queue: collections.deque = collections.deque()
        self.sessions: dict[str, Session] = {}
        self.connections: set[Connection] = set()
        self.counts = collections.Counter()
        self.listener: socket.socket | None = None
        self.threads: list[threading.Thread] = []
        self._worker_exit = False

    # ------------------------------------------------------------------------------------------------ lifecycle
    def warm_up(self) -> None:
        for sp in self.policies.values():
            if self.warmup_iterations <= 0:
                continue
            spec = sp.policy.input_spec
            wid = f"warmup-{uuid.uuid4().hex}"
            sp.policy.reset_session(session_id=wid, concurrent=True, **warmup_choices(spec, self.warmup_native_hw))
            try:
                sp.warmup_ms = sp.policy.warm_up(iterations=self.warmup_iterations)
            finally:
                sp.policy.end_session(wid)
            LOG.info("warm-up %s (%s): %s", sp.policy_id, spec.condition, sp.warmup_ms)

    def start(self) -> tuple[str, int]:
        self.warm_up()
        self.listener = socket.create_server((self.host, self.port), reuse_port=False)
        self.listener.settimeout(0.5)
        self.host, self.port = self.listener.getsockname()[:2]
        for target, name in ((self._accept_loop, "policy-accept"), (self._worker, "policy-worker"),
                             (self._reaper, "policy-reaper")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self.threads.append(t)
        LOG.info("serving %s on %s:%d", sorted(self.policies), self.host, self.port)
        return self.host, self.port

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.shutdown()

    def shutdown(self, timeout_s: float = 5.0) -> None:
        """Stop accepting, answer queued requests with ``shutting_down``, finish the running one, close connections."""
        with self.lock:
            if self.stopping:
                return
            self.stopping = True
            drained = list(self.queue)
            self.queue.clear()
            for job in drained:
                job.session.pending = None
        for job in drained:
            self._respond(job.session, job.seq, self._error(job.session, job.seq, "shutting_down", "server stopping"))
        with self.lock:
            for s in list(self.sessions.values()):  # releases the deploy API's live sessions too
                if s.running is None:
                    self._end_session(s, "server shutdown")
        self.stop_event.set()
        with self.cond:
            self._worker_exit = True
            self.cond.notify_all()
        for t in self.threads:
            t.join(timeout_s)
        if self.listener is not None:
            self.listener.close()
        with self.lock:
            conns = list(self.connections)
        for c in conns:
            c.close(flush=True)  # deliver what is queued (shutting_down answers, the last action), then close
        for c in conns:
            c.sender.join(timeout_s)
            c.wire.close()
            c.thread.join(timeout_s)
        LOG.info("stopped; counts %s", dict(self.counts))

    def policy_specs(self) -> dict:
        return {pid: {"input_spec": sp.policy.input_spec.to_json(), "info": P.jsonable(sp.info)}
                for pid, sp in self.policies.items()}

    def stats(self) -> dict:
        with self.lock:
            return {"counts": dict(self.counts), "sessions": len(self.sessions), "queue": len(self.queue),
                    "connections": len(self.connections)}

    # ------------------------------------------------------------------------------------------------ threads
    def _accept_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                readable, _, _ = select.select([self.listener], [], [], 0.2)
                if not readable:
                    continue
                sock, addr = self.listener.accept()
            except OSError:
                if self.stop_event.is_set():
                    return
                continue
            with self.lock:
                full = len(self.connections) >= self.max_connections
                self.counts["connections_refused" if full else "connections"] += 1
            if full:  # answered and closed at once: no threads for connections over the cap
                LOG.warning("connection from %s refused: %d connections open", addr, self.max_connections)
                try:
                    sock.settimeout(1.0)
                    sock.sendall(P.error_message("overloaded", f"{self.max_connections} connections open",
                                                 fatal=True))
                    sock.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                sock.close()
                continue
            conn = Connection(self, sock, addr)
            with self.lock:
                self.connections.add(conn)
            conn.start()

    def drop_connection(self, conn: Connection) -> None:
        conn.close(flush=True)
        with self.lock:
            self.connections.discard(conn)
            s = conn.session
            if s is not None and s.conn is conn:
                s.conn = None  # the session survives for a resume until it idles out
                s.last_activity = time.monotonic()

    def _reaper(self) -> None:
        while not self.stop_event.wait(0.25):
            now = time.monotonic()
            with self.lock:
                for s in list(self.sessions.values()):
                    if s.pending is None and s.running is None and now - s.last_activity > self.idle_timeout_s:
                        self._end_session(s, f"idle > {self.idle_timeout_s} s")

    def _end_session(self, s: Session, reason: str, *, taken_over: bool = False) -> None:
        """Caller holds ``self.lock``. ``taken_over``: tell the evicted client at once (its stop signal)."""
        if s.closed:
            return
        s.closed = True
        self.sessions.pop(s.id, None)
        s.served.sessions.pop(s.id, None)
        s.served.policy.end_session(s.id)  # its windows are refused from now on (the deploy API's live set)
        if s.pending is not None:
            try:
                self.queue.remove(s.pending)
            except ValueError:
                pass
            s.pending = None
        if s.conn is not None and s.conn.session is s:
            s.conn.session = None
            if taken_over:  # queued on its outbox (non-blocking); a disconnected client learns it at renewal
                s.conn.send(P.error_message("session_taken_over", reason, session_id=s.id))
        s.conn = None
        self.counts["sessions_ended"] += 1
        LOG.info("session %s (%s) ended: %s; counts %s", s.id, s.served.policy_id, reason, dict(s.counts))

    def _worker(self) -> None:
        while True:
            with self.cond:
                while not self.queue and not self._worker_exit:
                    self.cond.wait(0.5)
                if not self.queue:
                    return
                job = self.queue.popleft()
                s = job.session
                if s.pending is job:
                    s.pending = None
                if s.closed:
                    continue
                s.running = job
            try:
                data = self._execute(job)
            except Exception as exc:  # noqa: BLE001 - the only inference thread must survive and answer
                LOG.exception("request failed (session %s seq %d)", s.id, job.seq)
                data = self._error(s, job.seq, "internal", f"{type(exc).__name__}: {exc}")
            finally:
                with self.lock:
                    if s.running is job:
                        s.running = None
                    s.last_activity = time.monotonic()
            self._respond(s, job.seq, data)

    def _execute(self, job: Job) -> bytes:
        s = job.session
        t_start = time.monotonic_ns()
        if self.stopping:
            return self._error(s, job.seq, "shutting_down", "server stopping")
        if t_start > job.deadline_ns:
            self.counts["stale_ttl"] += 1
            return self._error(s, job.seq, "stale", f"ttl {job.ttl_ms} ms expired before execution "
                                                    f"(queued {(t_start - job.t_enqueued_ns) / 1e6:.2f} ms)")
        sp = s.served
        with sp.lock:
            if s.closed or s.id not in sp.policy.live_sessions:
                return self._error(s, job.seq, "session_unknown", "session ended before execution")
            try:
                out = sp.policy.infer(job.window, job.pose7, job.gripper, job.obs_ids, noise=job.noise)
            except ValueError as exc:
                return self._error(s, job.seq, classify(exc), f"deploy contract refused the request: {exc}")
            except Exception as exc:  # noqa: BLE001
                LOG.exception("inference failed (session %s seq %d)", s.id, job.seq)
                return self._error(s, job.seq, "internal", f"{type(exc).__name__}: {exc}")
        t_done = time.monotonic_ns()
        plain, markers, arrays = P.pack_outputs(out)
        t = out.get("timings", {})
        server_ms = {"prep_ms": job.prep_ms, "queue_ms": (t_start - job.t_enqueued_ns) / 1e6,
                     "validation_ms": t.get("validation_ms"), "encoder_ms": t.get("encoder_ms"),
                     "fql_ms": t.get("fql_ms"), "policy_ms": t.get("policy_ms"),
                     "infer_ms": (t_done - t_start) / 1e6}
        server_ms["total_ms"] = (time.monotonic_ns() - job.t_recv_ns) / 1e6
        s.counts["actions"] += 1
        self.counts["actions"] += 1
        return P.encode({"type": P.ACTION, "session_id": s.id, "seq": job.seq, "ack": job.seq, "status": "ok",
                         "reference_id": job.obs_ids["reference_id"], "outputs": plain, "output_arrays": markers,
                         "server_ms": P.jsonable(server_ms)}, arrays)

    # ------------------------------------------------------------------------------------------------ responses
    def _error(self, s: Session | None, seq, code: str, detail: str, history_reset: bool = False,
               session_id: str | None = None) -> bytes:
        with self.lock:
            self.counts[f"error_{code}"] += 1
            if s is not None:
                s.counts[f"error_{code}"] += 1
        return P.error_message(code, detail, session_id=session_id if s is None else s.id, seq=seq,
                               history_reset=history_reset)

    def _respond(self, s: Session, seq: int, data: bytes) -> None:
        """Cache (for duplicates and resume) THEN deliver to the session's current connection, if any."""
        with self.lock:
            s.responses[seq] = data
            while len(s.responses) > self.response_cache:
                s.responses.popitem(last=False)
            conn = s.conn
        if conn is not None:
            conn.send(data)

    # ------------------------------------------------------------------------------------------------ dispatch
    def dispatch(self, conn: Connection, msg: P.Message) -> bool:
        """Handle one message; False closes the connection."""
        if msg.header["v"] != P.VERSION:
            conn.send(P.error_message("version", f"server speaks protocol {P.VERSION}, got {msg.header['v']}",
                                      fatal=True))
            return False
        kind = msg.type
        if kind == P.HELLO:
            return self._hello(conn, msg)
        if not conn.authenticated:
            conn.send(P.error_message("auth", "HELLO with the shared-token handshake first", fatal=True))
            return False
        s = conn.session
        if s is not None:
            with self.lock:
                s.last_activity = time.monotonic()
        if kind == P.PING:
            conn.send(P.encode({"type": P.PONG, "t_client_ns": msg.get("t_client_ns"),
                                "t_server_ns": time.monotonic_ns(), "session_id": None if s is None else s.id}))
            return True
        if kind == P.BYE:
            with self.lock:
                if s is not None:
                    self._end_session(s, "client BYE")
            return False
        if kind not in (P.STEP, P.RESET):
            raise P.ProtocolError("bad_request", f"unexpected {kind} from a client")
        seq = msg.get("seq")
        if s is None or msg.get("session_id") != s.id:
            claimed = msg.get("session_id") if isinstance(msg.get("session_id"), str) else None
            conn.send(self._error(None, seq, "session_unknown", "no live session on this connection; HELLO first",
                                  session_id=claimed))
            return True
        if type(seq) is not int or seq <= 0:
            conn.send(self._error(s, seq, "bad_request", "seq must be a positive integer"))
            return True
        with self.lock:
            if seq <= s.last_seq:
                cached = s.responses.get(seq)
                inflight = any(j is not None and j.seq == seq for j in (s.pending, s.running))
                s.counts["duplicate_seq"] += 1
            else:
                s.last_seq, cached, inflight = seq, None, None
        if inflight is not None:  # a repeated seq: cached response, in-flight (delivered once ready) or evicted
            if cached is not None:
                conn.send(cached)
            elif not inflight:
                conn.send(self._error(s, seq, "duplicate", f"seq {seq} already processed; response evicted"))
            return True
        if kind == P.RESET:
            self._reset(s, seq)
            return True
        try:
            job = self._prepare(s, seq, msg)
        except Reject as r:
            self._respond(s, seq, self._error(s, seq, r.code, r.detail, r.history_reset))
            return True
        self._submit(job, conn)
        return True

    def _hello(self, conn: Connection, msg: P.Message) -> bool:
        h = msg.header
        if conn.session is not None:
            raise P.ProtocolError("bad_request", "this connection already carries a session")
        if self.token is not None:
            expected = hmac.new(self.token, conn.nonce.encode(), hashlib.sha256).hexdigest()
            given = h.get("auth")
            if not isinstance(given, str) or not hmac.compare_digest(given, expected):
                self.counts["auth_failures"] += 1
                LOG.warning("authentication failed from %s", conn.addr)
                conn.send(P.error_message("auth", "shared-token handshake failed", fatal=True))
                return False
        conn.authenticated = True
        sp = self.policies.get(h.get("policy_id"))
        if sp is None:
            raise P.ProtocolError("bad_request", f"unknown policy_id {h.get('policy_id')!r}; "
                                                 f"served: {sorted(self.policies)}")
        resume = h.get("resume")
        if resume is not None:
            return self._resume(conn, sp, resume)
        choices = h.get("choices")
        if not isinstance(choices, dict) or not set(choices) <= HELLO_CHOICES:
            raise P.ProtocolError("bad_request", f"choices must be an object with keys from {sorted(HELLO_CHOICES)}")
        choices = dict(choices)
        if isinstance(choices.get("native_hw"), dict):
            choices["native_hw"] = {c: tuple(v) if isinstance(v, list) else v for c, v in choices["native_hw"].items()}
        from s4d.rwlearn.deploy import HistoryWindow

        refusal = None
        with sp.lock:  # serialises session binding with this policy's inference
            with self.lock:
                if len(sp.sessions) >= self.max_sessions_per_policy:
                    oldest = min(sp.sessions.values(), key=lambda x: x.last_activity)
                    if h.get("takeover") is not True:
                        idle = time.monotonic() - oldest.last_activity
                        refusal = ("session_busy", f"policy {sp.policy_id} serves {len(sp.sessions)} session(s) "
                                                   f"(max {self.max_sessions_per_policy}; least active {oldest.id} "
                                                   f"idle {idle:.1f} s, expires after {self.idle_timeout_s} s idle); "
                                                   "HELLO takeover=true evicts it")
                    else:
                        self._end_session(oldest, f"evicted by takeover from {conn.addr}", taken_over=True)
            if refusal is None:
                sid = uuid.uuid4().hex
                try:
                    spec = sp.policy.reset_session(session_id=sid, concurrent=True, **choices)
                    history = HistoryWindow(spec, capacity=self.history_capacity)
                except (ValueError, TypeError, KeyError) as exc:
                    refusal = ("bad_request", f"session choices refused: {exc}")
            if refusal is None:
                s = Session(sid, sp, spec, history, history.capacity)
                try:  # the SPEC is built BEFORE the session is registered: a failure leaves nothing live behind
                    spec_msg = self._spec_message(s, resumed=False, redeliver=[])
                except (ValueError, TypeError) as exc:
                    sp.policy.end_session(sid)
                    refusal = ("bad_request", f"session spec not serialisable: {exc}")
            if refusal is None:
                with self.lock:
                    self.sessions[sid] = s
                    sp.sessions[sid] = s
                    s.conn, conn.session = conn, s
                    self.counts["sessions_opened"] += 1
        if refusal is not None:
            conn.send(self._error(None, None, *refusal))
            return True
        LOG.info("session %s opened for %s from %s (cameras %s)", sid, sp.policy_id, conn.addr, spec.camera_ids)
        conn.bound()
        conn.send(spec_msg)
        return True

    def _resume(self, conn: Connection, sp: ServedPolicy, resume) -> bool:
        if not isinstance(resume, dict) or not isinstance(resume.get("session_id"), str) \
                or type(resume.get("last_ack", 0)) is not int:
            raise P.ProtocolError("bad_request", "resume must be {session_id: str, last_ack: int}")
        with self.lock:
            s = self.sessions.get(resume["session_id"])
            if s is None or s.closed or s.served is not sp:
                conn.send(self._error(None, None, "session_unknown",
                                      f"cannot resume {resume['session_id']}: expired, ended or unknown"))
                return True
            old, s.conn, conn.session = s.conn, conn, s
            s.last_activity = time.monotonic()
            redeliver = sorted((q, d) for q, d in s.responses.items() if q > resume.get("last_ack", 0))
            spec_msg = self._spec_message(s, resumed=True, redeliver=[q for q, _ in redeliver])
            self.counts["resumes"] += 1
        if old is not None and old is not conn:
            old.session = None
            old.close()
        LOG.info("session %s resumed from %s (redeliver %s)", s.id, conn.addr, [q for q, _ in redeliver])
        conn.bound()
        conn.send(spec_msg)
        for _, data in redeliver:
            conn.send(data)
        return True

    def _spec_message(self, s: Session, *, resumed: bool, redeliver: list) -> bytes:
        return P.encode({"type": P.SPEC, "session_id": s.id, "policy_id": s.served.policy_id, "resumed": resumed,
                         "input_spec": s.spec.to_json(), "info": P.jsonable(s.served.info),
                         "last_seq_received": s.last_seq, "redeliver": redeliver,
                         "limits": {"max_queue": self.max_queue, "max_payload_bytes": self.max_payload_bytes,
                                    "max_header_bytes": self.max_header_bytes,
                                    "max_frames_per_step": self.max_frames_per_step,
                                    "session_idle_timeout_s": self.idle_timeout_s,
                                    "response_cache": self.response_cache, "history_capacity": s.capacity,
                                    "max_sessions_per_policy": self.max_sessions_per_policy},
                         "t_server_ns": time.monotonic_ns()})

    def _reset(self, s: Session, seq: int) -> None:
        with s.hist_lock:
            s.clear_history()
        with self.lock:
            old = s.pending
            if old is not None:
                self.queue.remove(old)
                s.pending = None
            s.counts["resets"] += 1
        if old is not None:
            self._respond(s, old.seq, self._error(s, old.seq, "stale", f"superseded by RESET seq {seq}"))
        self._respond(s, seq, P.encode({"type": P.RESET_OK, "session_id": s.id, "seq": seq, "ack": seq}))

    # ------------------------------------------------------------------------------------------------ STEP
    def _prepare(self, s: Session, seq: int, msg: P.Message) -> Job:
        """Validate the message, push the NEW frames, assemble the trained history window (connection thread).

        A message refused BEFORE its frames were pushed also clears the session history (``history_reset``) when it
        carried frames or sets: the client has counted them as delivered, so it must re-send its buffered history."""
        try:
            parsed, new_sets, request = self._validate_step(s, msg)
        except Reject as r:
            if msg.header.get("frames") or msg.header.get("sets"):
                with s.hist_lock:
                    s.clear_history()
                r.history_reset = True
                r.detail += "; session history reset - resend the history"
            raise
        return self._assemble(s, seq, msg, parsed, new_sets, *request)

    def _validate_step(self, s: Session, msg: P.Message):
        from s4d.rwlearn.deploy import Frame

        h, arrays, spec = msg.header, msg.arrays, s.spec
        frames, sets, req = h.get("frames", []), h.get("sets", []), h.get("request")
        if not isinstance(frames, list) or not isinstance(sets, list) or not isinstance(req, dict):
            raise Reject("bad_request", "STEP needs frames (list), sets (list) and request (object)")
        if len(frames) > self.max_frames_per_step:
            raise Reject("bad_request", f"{len(frames)} frames > max_frames_per_step {self.max_frames_per_step}")
        if not set(req) <= REQUEST_KEYS:
            raise Reject("bad_request", f"unknown request fields {sorted(set(req) - REQUEST_KEYS)}")
        referenced, parsed = set(), []
        for f in frames:
            if not isinstance(f, dict) or not set(f) <= FRAME_KEYS or not FRAME_KEYS - {"metadata"} <= set(f):
                raise Reject("bad_request", f"frame entry must have keys {sorted(FRAME_KEYS)}")
            cam, fid, t_ns = f["camera_id"], f["frame_id"], f["capture_ns"]
            if cam not in spec.camera_ids:
                raise Reject("bad_request", f"camera {cam!r} is not in this session {list(spec.camera_ids)}")
            if type(fid) is not int or fid < 0 or type(t_ns) is not int or t_ns < 0:
                raise Reject("bad_request", "frame_id and capture_ns must be nonnegative integers")
            names = f["arrays"]
            if not isinstance(names, dict) or "rgb" not in names or not set(names) <= {"rgb", "depth", "valid"}:
                raise Reject("bad_request", "frame arrays must name rgb [+ depth + valid]")
            if any(n not in arrays or n in referenced for n in names.values()):
                raise Reject("bad_request", "frame array missing from the payload or referenced twice")
            referenced.update(names.values())
            pose = f["pose7"]
            if not _numbers(pose, 7) or not _numbers([f["gripper_register"]], 1):
                raise Reject("bad_request", "frame pose7 (7 numbers) and gripper_register (number) required")
            meta = dict(f.get("metadata") or {})
            if not isinstance(f.get("metadata", {}), dict):
                raise Reject("bad_request", "frame metadata must be an object")
            if meta.get("native_hw") != [int(x) for x in spec.native_hw[cam]]:
                raise Reject("bad_request", f"{cam}: frame native grid {meta.get('native_hw')} differs from the "
                                            f"session's {list(spec.native_hw[cam])} (crop and intrinsics would be wrong)")
            if spec.needs_calibration:
                K = meta.get("K")
                want = np.asarray(spec.calibration[cam]["K"], np.float64)
                if not (isinstance(K, list) and np.shape(K) == (3, 3) and all(_numbers(r, 3) for r in K)
                        and np.allclose(np.asarray(K, np.float64), want, rtol=0, atol=1e-9)):
                    raise Reject("bad_request", f"{cam}: frame intrinsics {K} differ from the session calibration "
                                                f"{want.tolist()} at the training grid")
            meta.update(measured_pose7=pose, gripper_register=f["gripper_register"])
            parsed.append((cam, Frame(fid, t_ns, arrays[names["rgb"]], arrays.get(names.get("depth")),
                                      arrays.get(names.get("valid")), meta)))
        last = {}
        for cam, fr in parsed:
            if fr.seq <= last.get(cam, -1):
                raise Reject("bad_request", f"{cam}: frame ids must increase within a message")
            last[cam] = fr.seq
        noise = None
        if "noise" in req and "noise_seed" in req:
            raise Reject("bad_request", "give noise OR noise_seed")
        if "noise" in req:
            name = req["noise"]
            if name not in arrays or name in referenced:
                raise Reject("bad_request", "noise array missing from the payload")
            referenced.add(name)
            noise = arrays[name]
            if noise.dtype != np.float32 or noise.shape != (1, P.ACTION_DIM):
                raise Reject("bad_request", f"noise must be float32 (1, {P.ACTION_DIM})")
        elif "noise_seed" in req:
            try:
                noise = P.validation_noise(req["noise_seed"])
            except ValueError as exc:
                raise Reject("bad_request", str(exc)) from exc
        if set(arrays) != referenced:
            raise Reject("bad_request", f"unreferenced payload arrays {sorted(set(arrays) - referenced)}")
        end_ids, ref = req.get("end_ids"), req.get("reference_id")
        if not isinstance(end_ids, dict) or set(end_ids) != set(spec.camera_ids) \
                or any(type(v) is not int for v in end_ids.values()):
            raise Reject("bad_request", f"end_ids must give an integer frame id for each of {list(spec.camera_ids)}")
        if type(ref) not in (str, int):
            raise Reject("bad_request", "reference_id (str or int) of the bound measured pose required")
        ttl = req.get("ttl_ms")
        if not isinstance(ttl, (int, float)) or isinstance(ttl, bool) or not 0 < ttl <= 60_000:
            raise Reject("bad_request", "ttl_ms must be a number in (0, 60000]")
        pose7, grip = req.get("pose7"), req.get("gripper_register")
        if not _numbers(pose7, 7) or not _numbers([grip], 1):
            raise Reject("bad_request", "request pose7 (7 numbers) and gripper_register (number) required")
        pose_t, grip_t, pad = req.get("pose_time_ns"), req.get("gripper_time_ns"), req.get("pad_run_start", False)
        if type(pose_t) is not int or type(grip_t) is not int:
            raise Reject("bad_request", "pose_time_ns and gripper_time_ns (int, client monotonic ns of the measured "
                                        "robot / gripper samples) required")
        if type(pad) is not bool:
            raise Reject("bad_request", "pad_run_start must be a boolean")
        new_sets = []
        for st in sets:
            if not isinstance(st, dict) or set(st) != {"frame_ids", "surface_disp_m"} \
                    or not isinstance(st["frame_ids"], dict) or set(st["frame_ids"]) != set(spec.camera_ids) \
                    or any(type(v) is not int for v in st["frame_ids"].values()):
                raise Reject("bad_request", "set entries are {frame_ids: {camera: int}, surface_disp_m: number|null}")
            disp = st["surface_disp_m"]
            if disp is not None and (not isinstance(disp, (int, float)) or isinstance(disp, bool)):
                raise Reject("bad_request", "surface_disp_m must be a number or null (no valid robot state)")
            new_sets.append((tuple(st["frame_ids"][c] for c in spec.camera_ids), np.inf if disp is None else
                             float(disp)))
        if new_sets and spec.views == 1:
            raise Reject("bad_request", "multi-view sets sent to a single-view session")
        return parsed, new_sets, (end_ids, ref, ttl, pose7, grip, pose_t, grip_t, pad, noise)

    def _assemble(self, s: Session, seq: int, msg: P.Message, parsed: list, new_sets: list, end_ids, ref, ttl, pose7,
                  grip, pose_t, grip_t, pad, noise) -> Job:
        spec = s.spec
        with s.hist_lock:
            # Frames at or before a camera's history tip: an identical re-send of held frames is skipped (idempotent);
            # anything else (the client re-sent its buffer after a history reset, a restarted camera counter) rebuilds
            # the session history from this message - reported as history_rebuilt, never silently merged.
            rebuild = False
            for cam, fr in parsed:
                held = s.held[cam]
                tip = next(reversed(held)) if held else -1
                if fr.seq <= tip and held.get(fr.seq) != fr.timestamp_ns:
                    rebuild = True
            if rebuild:
                s.clear_history()
                s.counts["history_rebuilds"] += 1
            try:
                for cam, fr in parsed:
                    if fr.seq in s.held[cam]:
                        continue  # identical re-send of a held frame
                    s.history.push(cam, fr)
                    s.hold(cam, fr)
            except ValueError as exc:
                s.clear_history()
                raise Reject("bad_request", f"frame refused by the deploy contract ({exc}); session history reset "
                                            "- resend the history", history_reset=True) from exc
            for key, disp in new_sets:
                s.sets[key] = disp
                s.sets.move_to_end(key)
            while len(s.sets) > s.capacity:
                s.sets.popitem(last=False)
            L, k, cams = spec.history_length, spec.frame_spacing, spec.camera_ids
            if not pad:  # detailed refusal; with pad_run_start the deploy history decides (padding at a run start only)
                wanted = {c: [end_ids[c] - j * k for j in range(L - 1, -1, -1)] for c in cams}
                missing = {c: [i for i in ids if i not in s.held[c]] for c, ids in wanted.items()}
                missing = {c: ids for c, ids in missing.items() if ids}
                if missing:
                    raise Reject("incomplete_history", f"missing frames {missing} for the trained history "
                                                       f"(length {L}, spacing {k}); never re-spaced")
                if spec.views > 1:
                    absent = [key for key in (tuple(wanted[c][j] for c in cams) for j in range(L))
                              if key not in s.sets]
                    if absent:
                        raise Reject("incomplete_history", f"missing multi-view set evidence (M5) for sets {absent}")
            disp = {key[0]: d for key, d in s.sets.items()} if spec.views > 1 else None  # reference seq -> M5
            try:
                window = s.history.assemble(dict(end_ids), surface_disp_m=disp, pad_run_start=pad)
            except ValueError as exc:
                raise Reject(classify(exc), str(exc)) from exc
            if spec.views > 1:  # every assembled step must be a set the client declared, not just its reference frame
                steps = [tuple(window.frames[c][j].seq for c in cams) for j in range(L)]
                foreign = [key for key in steps if key not in s.sets]
                if foreign:
                    raise Reject("unsynchronized", f"assembled steps {foreign} are not declared multi-view sets")
        obs_ids = window.binding(ref, pose_time_ns=pose_t, gripper_time_ns=grip_t)
        now = time.monotonic_ns()
        return Job(s, seq, window, np.asarray(pose7, np.float64), float(grip), obs_ids, noise, float(ttl),
                   msg.t_recv_ns + int(ttl * 1e6), msg.t_recv_ns, now, (now - msg.t_recv_ns) / 1e6)

    def _submit(self, job: Job, conn: Connection) -> None:
        s, superseded, refusal, closed = job.session, None, None, False
        with self.lock:
            if self.stopping:
                refusal = ("shutting_down", "server stopping")
            elif s.closed:
                closed = True
                self.counts["dropped_closed_session"] += 1
            else:
                if s.pending is not None:
                    superseded = s.pending
                    self.queue.remove(superseded)
                    s.pending = None
                if len(self.queue) >= self.max_queue:
                    refusal = ("overloaded", f"inference queue full ({self.max_queue})")
                else:
                    job.t_enqueued_ns = time.monotonic_ns()
                    self.queue.append(job)
                    s.pending = job
                    self.cond.notify()
        if superseded is not None:
            self._respond(s, superseded.seq, self._error(s, superseded.seq, "stale",
                                                         f"superseded by seq {job.seq} before execution"))
        if refusal is not None:
            self._respond(s, job.seq, self._error(s, job.seq, *refusal))
        if closed:  # ended (expiry / eviction / BYE) while the request was prepared: answered on its connection
            LOG.info("seq %d of session %s dropped: session ended before queuing", job.seq, s.id)
            conn.send(self._error(None, job.seq, "session_unknown", "session ended before the request was queued",
                                  session_id=s.id))


# ---------------------------------------------------------------------------------------------------- loading
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def enforce_gpu(device: str, gpu_uuid: str | None):
    """CUDA serving only on the ONE configured GPU, checked by the repository's GPU guard (``ops.resources.gpu_guard``:
    host config from ``S4D_HOST_CONFIG``, ``CUDA_VISIBLE_DEVICES`` by UUID, torch's device UUID) instead of trusting the
    container to expose a single device. Must run before torch is imported. CPU serving: no GPU, nothing to check."""
    if not str(device).startswith("cuda"):
        return None
    if device not in ("cuda", "cuda:0"):
        raise ValueError(f"device {device!r}: serve on cuda / cuda:0 = the configured GPU (CUDA_VISIBLE_DEVICES)")
    if not gpu_uuid:
        raise ValueError("CUDA serving needs server.gpu_uuid (--gpu-uuid): the one GPU this server may use")
    from ops.resources import gpu_guard

    entries = gpu_guard.visible_device_uuids()  # fails closed: unset / not a UUID / not allowed on this host config
    if entries != [gpu_uuid]:
        raise gpu_guard.GPUIsolationError(f"CUDA_VISIBLE_DEVICES={entries} is not exactly the configured {gpu_uuid}")
    return gpu_guard.enforce_allowed_gpus(verbose=False)  # torch sees exactly that device (UUID checked)


def load_policy(entry: dict, *, root, device: str, allow_fixture: bool = False):
    """One configured policy through the public deploy API (sha256 check; the training contract comes from the
    deploy-contract sidecar JSON or the FQL dataset, both checked against the checkpoint by the deploy API). FIXTURE
    checkpoints (synthetic test weights) are refused unless ``allow_fixture``."""
    from ops.policy_serving.config import resolve
    from s4d.rwlearn.deploy import DeployPolicy

    ckpt = resolve(root, entry["checkpoint"])
    digest = sha256_file(ckpt)
    if entry.get("sha256") and entry["sha256"] != digest:
        raise ValueError(f"{ckpt}: sha256 {digest} != configured {entry['sha256']}")
    export = resolve(root, entry.get("export"))
    contract = entry.get("training_contract") or entry.get("dataset")
    policy = DeployPolicy(ckpt, device=device, export_path=export,
                          training=None if contract is None else str(resolve(root, contract)))
    if policy.input_spec.fixture and not allow_fixture:
        raise ValueError(f"{ckpt}: a FIXTURE checkpoint (synthetic weights) is never served without --allow-fixture")
    info = {"checkpoint": str(ckpt), "checkpoint_sha256": digest, "condition": policy.input_spec.condition,
            "fixture": policy.input_spec.fixture, "task": entry.get("task"), "device": device}
    return policy, info


def read_token(path) -> bytes:
    p = Path(path).expanduser()
    if p.stat().st_mode & 0o077:
        LOG.warning("token file %s is readable by group/others", p)
    token = p.read_bytes().strip()
    if not token:
        raise ValueError(f"token file {p} is empty")
    return token


def main(argv=None) -> int:
    from ops.policy_serving.config import load_config

    ap = argparse.ArgumentParser(prog="ops/policy_serving/server.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="load policies once, warm up, serve until SIGINT/SIGTERM")
    s.add_argument("--config", type=Path, default=None, help="default configs/realworld/policy_serving.yaml")
    s.add_argument("--policies", default=None, help="comma-separated policy ids (default server.default_policies)")
    s.add_argument("--bind", default=None)
    s.add_argument("--port", type=int, default=None, help="0 = ephemeral (reported in the READY line)")
    s.add_argument("--device", default=None)
    s.add_argument("--checkpoint-root", default=None)
    s.add_argument("--token-file", default=None, help="shared-token file (never logged)")
    s.add_argument("--warmup-iterations", type=int, default=None)
    s.add_argument("--threads", type=int, default=None, help="CPU threads (OMP/MKL) for CPU inference")
    s.add_argument("--gpu-uuid", default=None, help="the one GPU (UUID) for --device cuda (default server.gpu_uuid)")
    s.add_argument("--allow-fixture", action="store_true", help="serve FIXTURE checkpoints (tests only)")
    s.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(args, "log_level", "INFO"), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if cfg["protocol_version"] not in (None, P.VERSION):
        raise SystemExit(f"config protocol_version {cfg['protocol_version']} != {P.VERSION}")
    sc = cfg["server"]
    threads = args.threads or sc["torch_threads"]
    if threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = str(int(threads))
    ids = args.policies.split(",") if args.policies else list(sc["default_policies"])
    if not ids:
        raise SystemExit("no policies selected (--policies or server.default_policies)")
    unknown = [i for i in ids if i not in cfg["policies"]]
    if unknown:
        raise SystemExit(f"unknown policy ids {unknown}; configured: {sorted(cfg['policies'])}")
    device = args.device or sc["device"]
    bind = args.bind or sc["bind_host"]
    if not args.token_file and not is_loopback(bind):  # refused before anything is loaded
        raise SystemExit(f"binding {bind!r} beyond loopback requires --token-file")
    gpus = enforce_gpu(device, args.gpu_uuid or sc["gpu_uuid"])  # before any torch import
    root = args.checkpoint_root or sc["checkpoint_root"]
    loaded, info = {}, {}
    for pid in ids:
        t0 = time.monotonic()
        loaded[pid], info[pid] = load_policy(cfg["policies"][pid], root=root, device=device,
                                             allow_fixture=args.allow_fixture)
        info[pid]["load_s"] = round(time.monotonic() - t0, 3)
    server = PolicyServer(loaded, host=bind, port=sc["port"] if args.port is None else args.port,
                          token=read_token(args.token_file) if args.token_file else None,
                          max_queue=sc["max_queue"], session_idle_timeout_s=sc["session_idle_timeout_s"],
                          response_cache=sc["response_cache"], io_timeout_s=sc["io_timeout_s"],
                          max_header_bytes=sc["max_header_bytes"], max_payload_bytes=sc["max_payload_bytes"],
                          max_frames_per_step=sc["max_frames_per_step"], history_capacity=sc["history_capacity"],
                          max_sessions_per_policy=sc["max_sessions_per_policy"],
                          max_connections=sc["max_connections"],
                          warmup_iterations=sc["warmup_iterations"] if args.warmup_iterations is None
                          else args.warmup_iterations, warmup_native_hw=sc["warmup_native_hw"], info=info)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    host, port = server.start()
    print(json.dumps({"event": "ready", "host": host, "port": port, "protocol": P.VERSION, "pid": os.getpid(),
                      "auth": server.token is not None, "gpus": gpus,
                      "policies": {pid: {**P.jsonable(info[pid]), "warmup_ms": P.jsonable(sp.warmup_ms)}
                                   for pid, sp in server.policies.items()}}), flush=True)
    while not stop.wait(1.0):
        pass
    server.shutdown()
    print(json.dumps({"event": "stopped", **server.stats()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
