"""Persistent real-world policy server: ONE process, one GPU worker, per-session history, selectable policies.

    .venv/bin/python -I ops/policy_serving/server.py serve --config configs/realworld/policy_serving.yaml \
        [--catalog catalog.json] [--preload ID,...] [--policies ID,...]

Policy registry: the config's static ``policies`` + the staged CATALOG (``config.load_catalog``: ids, aliases, status
deployable / quarantined, advertised unbound input spec). Every offered id is advertised in WELCOME without loading;
a policy is loaded on the first HELLO that selects it (sha256 + training-contract checks, warm-up, advertised spec ==
loaded spec), stays resident afterwards and is evicted least-recently-used only under the GPU memory budget
(``gpu_memory_budget_mb`` / ``max_resident_policies``) and never while it has a live session. The catalog is re-read
on SIGHUP and when a HELLO names an unknown id (file changed); live sessions keep the object they were bound to.
Quarantined entries are refused unless ``--allow-quarantined`` (then labelled in every info).

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
import gc
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
from ops.policy_serving.config import RESTRICTED_EXECUTION_CAPABILITY, SERVABLE_STATUSES, valid_sha256  # noqa: E402

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


IDENTITY_KEYS = ("checkpoint", "sha256", "export", "training_contract", "dataset", "input_spec", "actor")


class BudgetExceeded(Exception):
    """No room for one more resident policy (GPU memory budget / resident cap; in-use policies are never evicted)."""


@dataclass(eq=False)
class CatalogEntry:
    """One selectable policy of the registry (config ``policies`` / catalog entry / a pre-loaded constructor policy)."""
    policy_id: str
    entry: dict  # checkpoint / export / training_contract | dataset / sha256 (+ catalog labels)
    status: str = "deployable"  # deployable | quarantined | error (could not be described)
    reasons: tuple = ()
    spec: dict | None = None  # advertised unbound InputSpec JSON (normalised)
    source: str = "config"  # config | catalog | preloaded
    spec_checked: bool = False  # spec == the one described from the checkpoint files (not only catalog-supplied)

    def identity(self) -> str:
        """What the loaded object depends on: the staged artefacts and whether the entry may be served at all. Labels
        (deployable <-> restricted, reasons, criteria, usage) refresh without reloading the model."""
        servable = self.status in SERVABLE_STATUSES
        return json.dumps([servable if self.status != "error" else "error",
                           {k: self.entry.get(k) for k in IDENTITY_KEYS}], sort_keys=True, default=str)

    def info(self) -> dict:
        """Published labels. FAIL CLOSED: ``execution_eligible`` is true only if the entry says exactly true;
        ``usage`` defaults to dry_run_replay_only."""
        e = self.entry
        out = {"condition": e.get("condition") or (self.spec or {}).get("condition"), "task": e.get("task"),
               "seed": e.get("seed"), "run": e.get("run"), "status": self.status, "flags": list(e.get("flags", [])),
               "checkpoint_sha256": e.get("sha256"), "usage": e.get("usage", "dry_run_replay_only"),
               "execution_eligible": e.get("execution_eligible") is True, "actor": e.get("actor", "onestep"),
               "execution_decision": e.get("execution_decision") if e.get("execution_eligible") is True else None,
               "usage_reasons": list(e.get("usage_reasons", [])), "criteria": e.get("criteria", {}),
               "reasons": list(self.reasons), "requires_client": self.requires_client()}
        return out

    def requires_client(self) -> list:
        """Client capabilities a HELLO must declare to bind this entry: a restricted entry with an execution grant
        (a criterion FAILED, executable by user decision) needs a client that shows the failure and the decision
        before START (``RESTRICTED_EXECUTION_CAPABILITY``); an older client is refused at HELLO instead of executing it
        without the notice."""
        e = self.entry
        return [RESTRICTED_EXECUTION_CAPABILITY] if self.status == "restricted" and \
            e.get("execution_eligible") is True else []


@dataclass(eq=False)
class ServedPolicy:
    policy_id: str
    policy: object  # s4d.rwlearn.deploy.DeployPolicy
    info: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)  # held during infer and reset_session
    sessions: dict = field(default_factory=dict)  # live sessions of this policy (id -> Session)
    warmup_ms: dict | None = None
    entry: CatalogEntry | None = None  # the registry entry this object was loaded from
    pinned: bool = False  # given pre-loaded to the constructor: never evicted (cannot be re-loaded)
    footprint: int = 0  # bytes of parameters, buffers and optimizer state on the device
    last_used: float = 0.0  # time.monotonic() of the last HELLO / inference
    claims: int = 0  # HELLOs between acquiring and binding (never evicted meanwhile)
    retired: bool = False  # its catalog entry was removed / changed: unloaded once its last session ends
    withdrawn: dict | None = None  # labels published for a retired object's remaining sessions (never eligible)


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
            srv.reload_catalog(only_if_changed=True)  # a newly staged policy is offered without a restart
            self.send(P.encode({"type": P.WELCOME, "protocol": P.PROTOCOL, "server": srv.name, "nonce": self.nonce,
                                "auth_required": srv.token is not None, "policies": srv.policy_specs(),
                                "unavailable": srv.unavailable(), "t_server_ns": time.monotonic_ns()}))
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
    """Serve a registry of policies on one TCP port: ``policies`` = ``{policy_id: DeployPolicy}`` loaded by the caller
    (pinned: resident for the server's lifetime) and/or ``entries`` (config ``policies`` entries) + ``catalog`` (path
    of a staged catalog JSON), loaded lazily on the first HELLO that selects them."""

    def __init__(self, policies: dict | None = None, *, entries: dict | None = None, catalog=None,
                 checkpoint_root=None, device: str = "cpu", allow_fixture: bool = False,
                 allow_quarantined: bool = False, offered=None, preload=(), gpu_memory_budget_bytes: int | None = None,
                 max_resident_policies: int | None = None, host: str = "127.0.0.1", port: int = 0,
                 token: bytes | None = None, max_queue: int = 4, session_idle_timeout_s: float = 30.0,
                 response_cache: int = 32, io_timeout_s: float = 5.0, max_header_bytes: int = 1 << 20,
                 max_payload_bytes: int = 32 << 20, max_frames_per_step: int = 64, history_capacity: int | None = None,
                 warmup_iterations: int = 0, max_sessions_per_policy: int = 1, max_connections: int = 16,
                 warmup_native_hw=(480, 640), info: dict | None = None, name: str = "s4d-policy-server"):
        if not policies and not entries and catalog is None:
            raise ValueError("no policies to serve")
        if token is not None and not token:
            raise ValueError("empty shared token")
        if token is None and not is_loopback(host):
            raise ValueError(f"binding {host!r} beyond loopback requires the shared token (--token-file)")
        self.max_connections = int(max_connections)
        if self.max_connections < 1:
            raise ValueError("max_connections must be positive")
        self.host, self.port, self.token, self.name = host, int(port), token, name
        self.max_queue, self.idle_timeout_s = int(max_queue), float(session_idle_timeout_s)
        self.response_cache, self.io_timeout_s = int(response_cache), float(io_timeout_s)
        self.max_header_bytes, self.max_payload_bytes = int(max_header_bytes), int(max_payload_bytes)
        self.max_frames_per_step, self.history_capacity = int(max_frames_per_step), history_capacity
        self.warmup_iterations, self.warmup_native_hw = int(warmup_iterations), tuple(warmup_native_hw)
        self.max_sessions_per_policy = int(max_sessions_per_policy)
        if self.max_queue < 1 or self.response_cache < 1 or self.max_sessions_per_policy < 1:
            raise ValueError("max_queue, response_cache and max_sessions_per_policy must be positive")
        if max_resident_policies is not None and int(max_resident_policies) < 1:
            raise ValueError("max_resident_policies must be positive (or None)")
        self.checkpoint_root, self.device = checkpoint_root, str(device)
        self.allow_fixture, self.allow_quarantined = bool(allow_fixture), bool(allow_quarantined)
        self.budget_bytes = None if gpu_memory_budget_bytes is None else int(gpu_memory_budget_bytes)
        self.max_resident = None if max_resident_policies is None else int(max_resident_policies)
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
        # registry: entries (selectable ids), aliases, resident served objects, in-flight load reservations
        self.static_entries = {str(k): dict(v) for k, v in (entries or {}).items()}
        self.catalog_path = None if catalog is None else Path(catalog)
        self.catalog_stamp, self.catalog_sha256 = None, None
        self.policies: dict[str, ServedPolicy] = {}
        self.reserved: dict[str, int] = {}
        self.known_footprint: dict[str, int] = {}  # entry identity -> measured footprint (budget estimate)
        self._load_locks: dict[str, threading.Lock] = collections.defaultdict(threading.Lock)
        self._reload_lock = threading.Lock()
        self.pinned_entries: dict[str, CatalogEntry] = {}
        now = time.monotonic()
        for pid, p in (policies or {}).items():
            actor = getattr(p, "actor", "onestep")
            if actor != "onestep":  # experimental actors only through validated catalog entries
                raise ValueError(f"pre-loaded policy {pid}: actor {actor!r} must be served from a catalog entry")
            e = CatalogEntry(pid, {}, spec=spec_json(p.input_spec), source="preloaded")
            self.pinned_entries[pid] = e
            self.policies[pid] = ServedPolicy(pid, p, dict((info or {}).get(pid, {})), entry=e, pinned=True,
                                              footprint=policy_footprint_bytes(p), last_used=now)
        self.entries, self.aliases = self._build_table({})
        if self.catalog_path is not None:
            self.catalog_stamp, self.catalog_sha256 = _stamp(self.catalog_path), sha256_file(self.catalog_path)
        self.offered = None
        if offered:
            unknown = [x for x in offered if self._resolve(x) is None]
            if unknown:
                raise ValueError(f"unknown policy ids {unknown}; registry: {sorted(self.entries)}")
            self.offered = {self._resolve(x) for x in offered}
        self.preload = [self._resolve(x) or x for x in preload]
        unknown = [x for x in self.preload if x not in self.entries]
        if unknown:
            raise ValueError(f"unknown preload ids {unknown}")

    # ------------------------------------------------------------------------------------------------ registry
    def _build_table(self, previous: dict) -> tuple[dict, dict]:
        """Entries + aliases from the pinned policies, the static config entries and the catalog file (raises on an
        invalid catalog or an id collision). Unchanged entries keep their previous object (and described spec)."""
        from ops.policy_serving.config import load_catalog

        table: dict[str, CatalogEntry] = dict(self.pinned_entries)
        raw = [(pid, e, "config") for pid, e in self.static_entries.items()]
        aliases = {}
        if self.catalog_path is not None:
            cat = load_catalog(self.catalog_path)
            raw += [(pid, e, "catalog") for pid, e in cat["policies"].items()]
            aliases = dict(cat["aliases"])
        for pid, e, source in raw:
            if pid in table:
                raise ValueError(f"policy id {pid!r} defined twice (pre-loaded / config / catalog)")
            spec = e.get("input_spec")
            entry = CatalogEntry(pid, dict(e), e.get("status", "deployable"), tuple(e.get("reasons", ())),
                                 None if spec is None else spec_json(spec), source)
            old = previous.get(pid)
            if old is not None and old.source == source and old.identity() == entry.identity():
                entry.spec = old.spec  # same artefacts: keep the described spec; labels come from the new entry
                entry.spec_checked = old.spec_checked
                if old.status == "error":
                    entry.status, entry.reasons = old.status, old.reasons
            elif entry.spec is None:
                try:
                    entry.spec = describe_entry(entry.entry, self.checkpoint_root)
                    entry.spec_checked = True
                except Exception as exc:  # noqa: BLE001 - one broken entry must not take the registry down
                    LOG.error("policy %s cannot be described (%s: %s); not offered", pid, type(exc).__name__, exc)
                    entry.status, entry.reasons = "error", (f"cannot be described: {type(exc).__name__}: {exc}",)
            if entry.status != "error" and entry.entry.get("execution_eligible") is True and not entry.spec_checked:
                # an execution-eligible entry is advertised only with the spec its checkpoint describes (meta read,
                # no model build), so its published condition / contract are never just catalog claims (Codex
                # review of f69a9d2); _load compares the loaded policy with it again
                try:
                    described = describe_entry(entry.entry, self.checkpoint_root)
                except Exception as exc:  # noqa: BLE001
                    described, why = None, f"cannot be described: {type(exc).__name__}: {exc}"
                else:
                    why = None if described == entry.spec else "catalog input_spec differs from the checkpoint's"
                if why is None:
                    entry.spec_checked = True
                else:
                    LOG.error("policy %s: %s; not offered", pid, why)
                    entry.status, entry.reasons = "error", (why,)
            label, described = entry.entry.get("condition"), (entry.spec or {}).get("condition")
            if entry.status != "error" and label is not None and label != described:
                # the advertised spec is checked against the loaded checkpoint (_load), so this binds the published
                # condition (and any condition-scoped execution grant) to the weights (Codex review of 0f61212, P0)
                LOG.error("policy %s: catalog condition %r != checkpoint condition %r; not offered", pid, label,
                          described)
                entry.status, entry.reasons = "error", (f"catalog condition {label!r} differs from the checkpoint's "
                                                        f"condition {described!r}",)
            table[pid] = entry
        for alias in aliases:
            if alias in table:
                raise ValueError(f"alias {alias!r} shadows a policy id")
        return table, aliases

    def _resolve(self, requested) -> str | None:
        """Policy id of a requested id or alias (None: unknown)."""
        if not isinstance(requested, str):
            return None
        with self.lock:
            pid = self.aliases.get(requested, requested)
            return pid if pid in self.entries else None

    def _is_offered(self, pid: str) -> bool:
        return self.offered is None or pid in self.offered

    def _refusal(self, entry: CatalogEntry) -> str | None:
        """Why an offered entry cannot be served (None: servable)."""
        if entry.status == "error":
            return f"policy {entry.policy_id} is unavailable: {'; '.join(entry.reasons)}"
        if entry.status == "quarantined" and not self.allow_quarantined:
            return f"policy {entry.policy_id} is quarantined: {'; '.join(entry.reasons)}"
        return None

    def policy_specs(self) -> dict:
        """WELCOME ``policies``: every offered servable id and alias with its advertised (unbound) input spec."""
        with self.lock:
            out = {}
            for pid, e in self.entries.items():
                if self._is_offered(pid) and self._refusal(e) is None:
                    sp = self.policies.get(pid)
                    resident = sp is not None and sp.entry is not None and sp.entry.identity() == e.identity()
                    out[pid] = {"input_spec": e.spec, "info": {**P.jsonable(sp.info if resident else {}),
                                                               **e.info(), "resident": resident}}
            for alias, pid in self.aliases.items():
                if pid in out:
                    out[alias] = {"input_spec": out[pid]["input_spec"], "info": {**out[pid]["info"], "alias_of": pid}}
            return out

    def unavailable(self) -> dict:
        """WELCOME ``unavailable``: offered ids that are refused (quarantined without --allow-quarantined, errors)."""
        with self.lock:
            return {pid: {"status": e.status, "reasons": list(e.reasons)} for pid, e in self.entries.items()
                    if self._is_offered(pid) and self._refusal(e) is not None}

    def reload_catalog(self, *, only_if_changed: bool = False) -> dict | None:
        """Re-read the catalog (SIGHUP / unknown-id HELLO). Invalid catalog: the old table stays (logged). Removed or
        changed entries are no longer offered in their old form; their resident objects are unloaded once they have
        no live session. Returns the change summary (None: no catalog, unchanged file or invalid)."""
        if self.catalog_path is None:
            return None
        with self._reload_lock:
            try:
                stamp = _stamp(self.catalog_path)
            except OSError as exc:
                LOG.error("catalog %s unreadable (%s); keeping the current registry", self.catalog_path, exc)
                return None
            if only_if_changed and stamp == self.catalog_stamp:
                return None
            with self.lock:
                previous = dict(self.entries)
            try:
                digest = sha256_file(self.catalog_path)
                entries, aliases = self._build_table(previous)
            except Exception as exc:  # noqa: BLE001
                LOG.error("catalog %s invalid (%s: %s); keeping the current registry", self.catalog_path,
                          type(exc).__name__, exc)
                return None
            with self.lock:
                added = sorted(set(entries) - set(previous))
                removed = sorted(set(previous) - set(entries))
                changed = sorted(p for p in set(entries) & set(previous)
                                 if entries[p].identity() != previous[p].identity())
                self.entries, self.aliases = entries, aliases
                self.catalog_stamp, self.catalog_sha256 = stamp, digest
                for pid, sp in list(self.policies.items()):
                    e = entries.get(pid)
                    if sp.pinned:
                        continue
                    if e is not None and e.identity() == sp.entry.identity():
                        sp.entry = e  # same artefacts: refreshed labels (status, criteria, usage) are published
                        sp.retired, sp.withdrawn = False, None  # servable again (e.g. quarantine lifted)
                        continue
                    sp.retired = True
                    sp.withdrawn = {"status": "withdrawn", "usage": "none", "execution_eligible": False,
                                    "execution_decision": None,
                                    "usage_reasons": ["withdrawn from the catalog: finish or end this session"],
                                    "reasons": ["removed from the catalog"] if e is None else
                                    [f"catalog entry changed ({e.status})", *e.reasons]}
                    if not sp.sessions and not sp.claims:
                        self._unload(sp, "catalog entry removed" if e is None else "catalog entry changed")
                self.counts["catalog_reloads"] += 1
            summary = {"added": added, "removed": removed, "changed": changed, "catalog_sha256": digest}
            log_event("catalog_reloaded", **summary)
            return summary

    # ------------------------------------------------------------------------------------------------ residency
    def _acquire(self, pid: str, *, takeover: bool) -> tuple[ServedPolicy | None, tuple | None]:
        """The resident object of ``pid`` (loaded now if needed) with one claim held, or ``(None, (code, detail))``.
        Called WITHOUT the server lock; one load per id at a time."""
        with self._load_locks[pid]:
            with self.lock:
                entry = self.entries.get(pid)
                if entry is None:
                    return None, ("bad_request", f"policy {pid} left the catalog")
                refusal = self._refusal(entry)  # re-checked here: a reload may have quarantined it since HELLO
                if refusal is not None:
                    return None, ("bad_request", refusal)
                sp = self.policies.get(pid)
                if sp is not None and not sp.pinned and sp.entry.identity() != entry.identity():
                    if sp.claims:  # another HELLO is binding to the previous version right now
                        return None, ("session_busy", f"policy {pid} changed in the catalog while another HELLO "
                                                      "binds its previous version; retry")
                    if sp.sessions and not takeover:
                        return None, ("session_busy", f"policy {pid} changed in the catalog; its previous version "
                                                      f"still serves {len(sp.sessions)} session(s); HELLO "
                                                      "takeover=true evicts them")
                    for s in list(sp.sessions.values()):
                        self._end_session(s, "evicted by takeover: catalog entry changed", taken_over=True)
                    self._unload(sp, "catalog entry changed")
                    sp = None
                if sp is not None:
                    sp.claims += 1
                    sp.last_used = time.monotonic()
                    return sp, None
            try:
                return self._load(pid, entry), None
            except BudgetExceeded as exc:
                with self.lock:
                    self.counts["budget_refusals"] += 1
                return None, ("overloaded", str(exc))
            except Exception as exc:  # noqa: BLE001 - a broken checkpoint is refused, never served
                LOG.exception("policy %s could not be loaded", pid)
                return None, ("bad_request", f"policy {pid} could not be loaded: {type(exc).__name__}: {exc}")

    @staticmethod
    def _current_info(sp: ServedPolicy) -> dict:
        """Load-time info + the CURRENT catalog labels of the entry it was loaded from (refreshed on reload)."""
        return {**sp.info, **(sp.entry.info() if sp.entry is not None and sp.entry.source != "preloaded" else {}),
                **(sp.withdrawn or {})}

    def _release(self, sp: ServedPolicy) -> None:
        with self.lock:
            sp.claims -= 1
            if sp.retired and not sp.sessions and not sp.claims:
                self._unload(sp, "retired catalog entry, last claim released")

    def _estimate(self, entry: CatalogEntry) -> int:
        known = self.known_footprint.get(entry.identity())
        if known is not None:
            return known
        from ops.policy_serving.config import resolve

        total = 0
        for key in ("checkpoint", "export"):
            if entry.entry.get(key):
                try:
                    total += resolve(self.checkpoint_root, entry.entry[key]).stat().st_size
                except OSError:
                    pass
        return total

    def _make_room(self, pid: str, estimate: int) -> None:
        """Caller holds ``self.lock``. Evict least-recently-used idle policies until ``estimate`` more bytes and one
        more resident policy fit; raise ``BudgetExceeded`` otherwise (policies in use are never evicted)."""
        def fits() -> bool:
            used = sum(sp.footprint for sp in self.policies.values()) + sum(self.reserved.values())
            count = len(self.policies) + len(self.reserved)
            return (self.budget_bytes is None or used + estimate <= self.budget_bytes) and \
                (self.max_resident is None or count + 1 <= self.max_resident)

        victims = sorted((sp for sp in self.policies.values()
                          if not sp.pinned and not sp.sessions and not sp.claims and sp.policy_id not in self.reserved),
                         key=lambda sp: sp.last_used)
        while not fits() and victims:
            self._unload(victims.pop(0), f"least recently used; room for {pid}")
        if not fits():
            busy = sorted(p for p, sp in self.policies.items() if sp.sessions or sp.claims or sp.pinned)
            used = sum(sp.footprint for sp in self.policies.values()) + sum(self.reserved.values())
            raise BudgetExceeded(
                f"cannot load {pid}: GPU memory budget {_mb(self.budget_bytes)} MB, resident {_mb(used)} MB + "
                f"{_mb(estimate)} MB needed, max_resident_policies {self.max_resident}; resident policies in use "
                f"(live sessions / pinned): {busy}")

    def _unload(self, sp: ServedPolicy, reason: str) -> None:
        """Caller holds ``self.lock``. Drop the resident object (memory returns once nothing references it)."""
        if self.policies.get(sp.policy_id) is not sp:
            return
        del self.policies[sp.policy_id]
        sp.retired = True
        self.counts["policies_evicted"] += 1
        if self.device.startswith("cuda"):
            gc.collect()
            import torch

            torch.cuda.empty_cache()
        log_event("policy_evicted", policy_id=sp.policy_id, reason=reason, footprint_bytes=sp.footprint)

    def _load(self, pid: str, entry: CatalogEntry) -> ServedPolicy:
        """Load + check + warm one entry (outside the server lock; the caller holds its load lock) and make it
        resident with one claim held."""
        estimate = self._estimate(entry)
        with self.lock:
            self._make_room(pid, estimate)
            self.reserved[pid] = estimate
        try:
            cuda = self.device.startswith("cuda")
            if cuda:
                import torch

                before = torch.cuda.memory_allocated()
            t0 = time.monotonic()
            policy, info = load_policy(entry.entry, root=self.checkpoint_root, device=self.device,
                                       allow_fixture=self.allow_fixture)
            load_s = time.monotonic() - t0
            if spec_json(policy.input_spec) != entry.spec:
                LOG.error("policy %s: catalog input_spec differs from the checkpoint's; refused", pid)
                raise ValueError("catalog input_spec differs from the checkpoint")
            warmup_ms = self._warm(pid, policy)
            footprint = policy_footprint_bytes(policy)
            delta = (torch.cuda.memory_allocated() - before) if cuda else None
        except BaseException:
            with self.lock:
                self.reserved.pop(pid, None)
            raise
        info.update(entry.info(), load_s=round(load_s, 3))
        sp = ServedPolicy(pid, policy, info, warmup_ms=warmup_ms, entry=entry, footprint=footprint,
                          last_used=time.monotonic(), claims=1)
        with self.lock:  # reservation -> resident in ONE step (no window for another load to take the slot)
            self.reserved.pop(pid, None)
            self.known_footprint[entry.identity()] = footprint
            current = self.entries.get(pid)
            stale = current is None or current.identity() != entry.identity() or self._refusal(current) is not None
            try:
                if not stale:
                    self._make_room(pid, footprint)  # the MEASURED footprint must fit, not only the estimate
            except BudgetExceeded:
                self.counts["budget_refusals_measured"] += 1
                raise
            if stale:
                raise ValueError(f"catalog entry of {pid} changed or became unservable during the load; retry")
            sp.entry = current  # same artefacts; labels refreshed during the load are the ones published
            self.policies[pid] = sp
            self.counts["policies_loaded"] += 1
            resident = sorted(self.policies)
        log_event("policy_loaded", policy_id=pid, load_s=round(load_s, 3), warmup_ms=warmup_ms,
                  footprint_bytes=footprint, cuda_allocated_delta_bytes=delta, resident=resident)
        return sp

    def _warm(self, pid: str, policy) -> dict | None:
        if self.warmup_iterations <= 0:
            return None
        spec = policy.input_spec
        wid = f"warmup-{uuid.uuid4().hex}"
        policy.reset_session(session_id=wid, concurrent=True, **warmup_choices(spec, self.warmup_native_hw))
        try:
            timings = policy.warm_up(iterations=self.warmup_iterations)
        finally:
            policy.end_session(wid)
        LOG.info("warm-up %s (%s): %s", pid, spec.condition, timings)
        return timings

    # ------------------------------------------------------------------------------------------------ lifecycle
    def warm_up(self) -> None:
        """Warm the pinned (constructor-given) policies, then load + warm the ``preload`` ids."""
        for sp in list(self.policies.values()):
            if sp.pinned and sp.warmup_ms is None:
                sp.warmup_ms = self._warm(sp.policy_id, sp.policy)
        for pid in self.preload:
            sp, refusal = self._acquire(pid, takeover=False) if self._refusal(self.entries[pid]) is None \
                else (None, ("bad_request", self._refusal(self.entries[pid])))
            if refusal is not None:
                raise ValueError(f"preload {pid}: {refusal[0]}: {refusal[1]}")
            self._release(sp)

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
        LOG.info("serving %s on %s:%d", sorted(self.policy_specs()), self.host, self.port)
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

    def ready_info(self) -> dict:
        """The registry part of the READY line: offered ids, aliases, resident policies, unavailable ids, catalog."""
        offered = self.policy_specs()
        with self.lock:
            resident = {pid: {**P.jsonable(self._current_info(sp)), "warmup_ms": P.jsonable(sp.warmup_ms),
                              "footprint_bytes": sp.footprint} for pid, sp in self.policies.items()}
            aliases = {a: p for a, p in self.aliases.items() if a in offered}
        return {"offered": sorted(p for p in offered if p not in aliases), "aliases": aliases, "resident": resident,
                "unavailable": self.unavailable(), "catalog": None if self.catalog_path is None else
                str(self.catalog_path), "catalog_sha256": self.catalog_sha256,
                "gpu_memory_budget_mb": _mb(self.budget_bytes), "max_resident_policies": self.max_resident}

    def stats(self) -> dict:
        with self.lock:
            return {"counts": dict(self.counts), "sessions": len(self.sessions), "queue": len(self.queue),
                    "connections": len(self.connections), "resident": sorted(self.policies)}

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
            conn.session = None  # a finished connection must not keep its session (and the model) reachable
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
                s = None  # noqa: F841 - no session (and its model) kept alive between passes

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
        if s.served.retired and not s.served.sessions and not s.served.claims:
            self._unload(s.served, "retired catalog entry, last session ended")

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
                    job = s = None
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
            # drop the finished request before waiting: the job / session / window reference the policy, which must
            # be freeable once evicted (otherwise its tensors stay allocated while the next policy loads)
            job = s = data = None  # noqa: F841

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
            sp.last_used = time.monotonic()
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
        requested = h.get("policy_id")
        resume = h.get("resume")
        declared = h.get("capabilities")
        declared = frozenset(declared) if isinstance(declared, list) and all(isinstance(x, str) for x in declared) \
            else frozenset()
        if resume is not None:  # a live session may run on a resident object whose entry left the catalog meanwhile
            with self.lock:
                pid = self._resolve(requested) or (requested if isinstance(requested, str) and
                                                   requested in self.policies else None)
                sp = None if pid is None else self.policies.get(pid)
            if pid is None:
                raise P.ProtocolError("bad_request", f"unknown policy_id {requested!r}")
            if sp is None:
                conn.send(self._error(None, None, "session_unknown", f"cannot resume: policy {pid} is not resident "
                                                                     "(evicted, retired or the server restarted)"))
                return True
            return self._resume(conn, sp, resume, capabilities=declared)
        pid = self._resolve(requested)
        if pid is None and self.reload_catalog(only_if_changed=True) is not None:
            pid = self._resolve(requested)  # the staged catalog gained the id since the last read
        if pid is None or not self._is_offered(pid):
            raise P.ProtocolError("bad_request", f"unknown policy_id {requested!r}; served: "
                                                 f"{sorted(self.policy_specs())}")
        choices = h.get("choices")
        if not isinstance(choices, dict) or not set(choices) <= HELLO_CHOICES:
            raise P.ProtocolError("bad_request", f"choices must be an object with keys from {sorted(HELLO_CHOICES)}")
        choices = dict(choices)
        if isinstance(choices.get("native_hw"), dict):
            choices["native_hw"] = {c: tuple(v) if isinstance(v, list) else v for c, v in choices["native_hw"].items()}
        with self.lock:
            entry = self.entries.get(pid)
            unavailable = f"policy {pid} left the catalog" if entry is None else self._refusal(entry)
        if unavailable is None:
            unavailable = self._capability_refusal(entry, declared)
        if unavailable is not None:
            conn.send(self._error(None, None, "bad_request", unavailable))
            return True
        sp, refusal = self._acquire(pid, takeover=h.get("takeover") is True)  # lazy load (never under self.lock)
        if refusal is not None:
            conn.send(self._error(None, None, *refusal))
            return True
        try:
            return self._bind(conn, sp, choices, takeover=h.get("takeover") is True, capabilities=declared)
        finally:
            self._release(sp)

    @staticmethod
    def _capability_refusal(entry: CatalogEntry | None, declared) -> str | None:
        """Why a HELLO (new session or resume) lacks a client capability the CURRENT entry requires (None: fine)."""
        lacking = [] if entry is None else [c for c in entry.requires_client() if c not in declared]
        if not lacking:
            return None
        return (f"policy {entry.policy_id} requires client capabilities {lacking} (a restricted policy executable by "
                "user decision: update the robot-PC client, which shows its failed criterion and the decision before "
                "START)")

    def _bind(self, conn: Connection, sp: ServedPolicy, choices: dict, *, takeover: bool,
              capabilities=frozenset()) -> bool:
        from s4d.rwlearn.deploy import HistoryWindow

        refusal = None
        with sp.lock:  # serialises session binding with this policy's inference
            with self.lock:
                current = self.entries.get(sp.policy_id)
                if self.policies.get(sp.policy_id) is not sp or sp.retired:
                    refusal = ("session_busy", f"policy {sp.policy_id} was replaced or unloaded during this HELLO; "
                                               "retry")
                elif not sp.pinned and (current is None or self._refusal(current) is not None
                                        or current.identity() != sp.entry.identity()):
                    refusal = ("bad_request", f"policy {sp.policy_id} is no longer servable in this form: "
                                              f"{'left the catalog' if current is None else self._refusal(current)}")
                elif (lack := self._capability_refusal(current, capabilities)) is not None:
                    refusal = ("bad_request", lack)  # a grant published since the HELLO's first check
                elif len(sp.sessions) >= self.max_sessions_per_policy:
                    oldest = min(sp.sessions.values(), key=lambda x: x.last_activity)
                    if not takeover:
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
                with self.lock:  # final check: a reload may have withdrawn the policy while the session was bound
                    current = self.entries.get(sp.policy_id)
                    if self.policies.get(sp.policy_id) is not sp or sp.retired or (
                            not sp.pinned and (current is None or self._refusal(current) is not None
                                               or current.identity() != sp.entry.identity())):
                        refusal = ("bad_request", f"policy {sp.policy_id} was withdrawn while this session was being "
                                                  "bound")
                    elif (lack := self._capability_refusal(current, capabilities)) is not None:
                        refusal = ("bad_request", lack)  # a grant published while this session was being bound
                    else:
                        try:  # SPEC encoded under the lock with the CURRENT labels, before registration
                            spec_msg = self._spec_message(s, resumed=False, redeliver=[])
                        except (ValueError, TypeError) as exc:
                            refusal = ("bad_request", f"session spec not serialisable: {exc}")
                    if refusal is None:
                        self.sessions[sid] = s
                        sp.sessions[sid] = s
                        s.conn, conn.session = conn, s
                        sp.last_used = time.monotonic()
                        self.counts["sessions_opened"] += 1
                if refusal is not None:
                    sp.policy.end_session(sid)
        if refusal is not None:
            conn.send(self._error(None, None, *refusal))
            return True
        LOG.info("session %s opened for %s from %s (cameras %s)", sid, sp.policy_id, conn.addr, spec.camera_ids)
        conn.bound()
        conn.send(spec_msg)
        return True

    def _resume(self, conn: Connection, sp: ServedPolicy, resume, *, capabilities=frozenset()) -> bool:
        if not isinstance(resume, dict) or not isinstance(resume.get("session_id"), str) \
                or type(resume.get("last_ack", 0)) is not int:
            raise P.ProtocolError("bad_request", "resume must be {session_id: str, last_ack: int}")
        with self.lock:
            s = self.sessions.get(resume["session_id"])
            if s is None or s.closed or s.served is not sp:
                conn.send(self._error(None, None, "session_unknown",
                                      f"cannot resume {resume['session_id']}: expired, ended or unknown"))
                return True
            lack = None if sp.pinned else self._capability_refusal(self.entries.get(sp.policy_id), capabilities)
            if lack is not None:  # a grant published since this session was bound by a client without the notice
                conn.send(self._error(None, None, "bad_request", lack))
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
                         "input_spec": s.spec.to_json(), "info": P.jsonable(self._current_info(s.served)),
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


def _stamp(path) -> tuple[int, int]:
    st = Path(path).stat()
    return st.st_mtime_ns, st.st_size


def _mb(n: int | None) -> float | None:
    return None if n is None else round(n / (1 << 20), 1)


def log_event(event: str, **fields) -> None:
    """One JSON line per registry event (load, eviction, catalog reload) in the server log."""
    LOG.info("%s", json.dumps({"event": event, **P.jsonable(fields)}))


def spec_json(spec) -> dict:
    """Normalised JSON of an (unbound) InputSpec or its ``to_json()`` dict, comparable across a JSON round trip."""
    return json.loads(json.dumps(P.jsonable(spec.to_json() if hasattr(spec, "to_json") else spec), sort_keys=True))


def describe_entry(entry: dict, root=None) -> dict:
    """Advertised unbound input spec of a policy entry WITHOUT building the model: the checkpoint's meta + its
    training contract (deploy-contract sidecar or FQL dataset; the deploy API's own reader and checks) ->
    ``spec_json(InputSpec.from_meta(...))``, identical to the loaded ``DeployPolicy.input_spec`` (checked at load)."""
    import torch

    from ops.policy_serving.config import resolve
    from s4d.rwlearn.deploy.contracts import InputSpec, read_training_contract

    meta = torch.load(resolve(root, entry["checkpoint"]), map_location="cpu", weights_only=False, mmap=True)["meta"]
    contract = entry.get("training_contract") or entry.get("dataset")
    training = read_training_contract(meta, None if contract is None else str(resolve(root, contract)))
    return spec_json(InputSpec.from_meta(meta, training))


def policy_footprint_bytes(policy) -> int:
    """Bytes of every parameter, buffer and optimizer-state tensor of a loaded DeployPolicy (agent networks + frozen
    encoder adapter): the device memory the resident policy holds (deterministic on CPU and CUDA)."""
    import torch

    off = policy.offline
    modules = list(off.agent.modules().values())
    adapter = getattr(off, "adapter", None)
    for m in (adapter, getattr(adapter, "backbone", None)):
        if isinstance(m, torch.nn.Module):
            modules.append(m)
    tensors = [t for m in modules for t in (*m.parameters(), *m.buffers())]
    optimizer = getattr(off.agent, "optimizer", None)
    if optimizer is not None:
        tensors += [v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)]
    seen, total = set(), 0
    for t in tensors:
        if id(t) not in seen:
            seen.add(id(t))
            total += t.numel() * t.element_size()
    return total


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
    if entry.get("execution_eligible") is True and not valid_sha256(entry.get("sha256")):
        raise ValueError(f"{ckpt}: an execution-eligible entry is served only against its checkpoint sha256")
    digest = sha256_file(ckpt)
    if entry.get("sha256") and entry["sha256"] != digest:
        raise ValueError(f"{ckpt}: sha256 {digest} != configured {entry['sha256']}")
    export = resolve(root, entry.get("export"))
    contract = entry.get("training_contract") or entry.get("dataset")
    policy = DeployPolicy(ckpt, device=device, export_path=export,
                          training=None if contract is None else str(resolve(root, contract)),
                          actor=entry.get("actor", "onestep"))
    if policy.input_spec.fixture and not allow_fixture:
        raise ValueError(f"{ckpt}: a FIXTURE checkpoint (synthetic weights) is never served without --allow-fixture")
    info = {"checkpoint": str(ckpt), "checkpoint_sha256": digest, "condition": policy.input_spec.condition,
            "fixture": policy.input_spec.fixture, "task": entry.get("task"), "device": device, "actor": policy.actor}
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
    from ops.policy_serving.config import load_config, resolve

    ap = argparse.ArgumentParser(prog="ops/policy_serving/server.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="advertise the registry, load selected policies lazily, serve until SIGINT/SIGTERM "
                                     "(SIGHUP re-reads the catalog)")
    s.add_argument("--config", type=Path, default=None, help="default configs/realworld/policy_serving.yaml")
    s.add_argument("--catalog", default=None, help="staged catalog JSON (relative: under the checkpoint root; "
                                                   "default server.catalog; 'none' = config policies only)")
    s.add_argument("--policies", default=None, help="comma-separated ids/aliases to OFFER (default "
                                                    "server.default_policies, else every registry entry)")
    s.add_argument("--preload", default=None, help="comma-separated ids loaded + warmed before READY "
                                                   "(default server.preload)")
    s.add_argument("--bind", default=None)
    s.add_argument("--port", type=int, default=None, help="0 = ephemeral (reported in the READY line)")
    s.add_argument("--device", default=None)
    s.add_argument("--checkpoint-root", default=None)
    s.add_argument("--token-file", default=None, help="shared-token file (never logged)")
    s.add_argument("--warmup-iterations", type=int, default=None)
    s.add_argument("--threads", type=int, default=None, help="CPU threads (OMP/MKL) for CPU inference")
    s.add_argument("--gpu-uuid", default=None, help="the one GPU (UUID) for --device cuda (default server.gpu_uuid)")
    s.add_argument("--allow-fixture", action="store_true", help="serve FIXTURE checkpoints (tests only)")
    s.add_argument("--allow-quarantined", action="store_true",
                   help="serve catalog entries marked quarantined (pipeline checks only; labelled in every info)")
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
    device = args.device or sc["device"]
    bind = args.bind or sc["bind_host"]
    if not args.token_file and not is_loopback(bind):  # refused before anything is loaded
        raise SystemExit(f"binding {bind!r} beyond loopback requires --token-file")
    root = args.checkpoint_root or sc["checkpoint_root"]
    catalog = args.catalog if args.catalog is not None else sc["catalog"]
    catalog = None if catalog in (None, "", "none") else resolve(root, catalog)
    if catalog is not None and not catalog.is_file():
        raise SystemExit(f"catalog {catalog} not found (stage it, or --catalog none for config policies only)")
    if not cfg["policies"] and catalog is None:
        raise SystemExit("no policies: neither config policies nor a catalog")
    offered = args.policies.split(",") if args.policies else list(sc["default_policies"]) or None
    preload = args.preload.split(",") if args.preload else list(sc["preload"])
    gpus = enforce_gpu(device, args.gpu_uuid or sc["gpu_uuid"])  # before any torch import
    budget = sc["gpu_memory_budget_mb"]
    try:
        server = PolicyServer(entries=cfg["policies"], catalog=catalog, checkpoint_root=root, device=device,
                              allow_fixture=args.allow_fixture, allow_quarantined=args.allow_quarantined,
                              offered=offered, preload=preload,
                              gpu_memory_budget_bytes=None if budget is None else int(budget * (1 << 20)),
                              max_resident_policies=sc["max_resident_policies"], host=bind,
                              port=sc["port"] if args.port is None else args.port,
                              token=read_token(args.token_file) if args.token_file else None,
                              max_queue=sc["max_queue"], session_idle_timeout_s=sc["session_idle_timeout_s"],
                              response_cache=sc["response_cache"], io_timeout_s=sc["io_timeout_s"],
                              max_header_bytes=sc["max_header_bytes"], max_payload_bytes=sc["max_payload_bytes"],
                              max_frames_per_step=sc["max_frames_per_step"], history_capacity=sc["history_capacity"],
                              max_sessions_per_policy=sc["max_sessions_per_policy"],
                              max_connections=sc["max_connections"],
                              warmup_iterations=sc["warmup_iterations"] if args.warmup_iterations is None
                              else args.warmup_iterations, warmup_native_hw=sc["warmup_native_hw"])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    stop, reload = threading.Event(), threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    signal.signal(signal.SIGHUP, lambda *_: reload.set())
    host, port = server.start()
    print(json.dumps({"event": "ready", "host": host, "port": port, "protocol": P.VERSION, "pid": os.getpid(),
                      "auth": server.token is not None, "gpus": gpus, "device": device, **server.ready_info()}),
          flush=True)
    while not stop.wait(0.5):
        if reload.is_set():
            reload.clear()
            summary = server.reload_catalog()
            print(json.dumps({"event": "catalog_reloaded", "ok": summary is not None, **(summary or {})}),
                  flush=True)
    server.shutdown()
    print(json.dumps({"event": "stopped", **server.stats()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
