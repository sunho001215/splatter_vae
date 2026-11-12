"""Network policy serving (ops/policy_serving) on CPU over loopback with tiny labelled FIXTURE policies
(``s4d.rwlearn.deploy.fixtures``; never trained evidence). No skips: every condition builds synthetic weights.

Direct ``DeployPolicy.infer`` == network result (identical observations + explicit FQL noise) for every condition
(single-view RGB, multi-view RGB + cameras, single / multi-view RGB-D, e2e CNN pixel path, ours fixture), plus the
transport contract: only-new-frames history, reconnect + resume, session renewal, incomplete history / run-start
padding, irregular spacing, explicit reset, duplicate seq / frame, ttl + superseded staleness, bounded-queue overload,
version / auth / payload refusals, session ownership, graceful shutdown, client action-expiry checks and the bench.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ops.policy_serving import bench
from ops.policy_serving import protocol as P
from ops.policy_serving.client import (
    DeviceClock,
    PolicyClient,
    RealSenseCapture,
    ServingError,
    SessionTakenOver,
    StepResult,
    bound_command,
    check_action,
)
from ops.policy_serving.config import CATALOG_SCHEMA, DEFAULT_PATH, load_catalog, load_config, validate_catalog
from ops.policy_serving.server import (
    PolicyServer,
    classify,
    describe_entry,
    enforce_gpu,
    load_policy,
    main,
    policy_footprint_bytes,
    sha256_file,
    spec_json,
)
from s4d.rwlearn.deploy import Frame, HistoryWindow, preprocess_frame
from s4d.rwlearn.deploy.fixtures import FIXTURE_CONDITIONS, make_fixture_policy, session_choices

NATIVE_HW = (128, 160)
T0 = 5_000_000_000
POSE = [0.31, -0.05, 0.42, 0.0, 0.0, 0.0, 1.0]
GRIP = 117.0


@pytest.fixture(scope="module")
def policies(tmp_path_factory):
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    root = tmp_path_factory.mktemp("serving-fixtures")
    out = {c: make_fixture_policy(root / c, c) for c in FIXTURE_CONDITIONS}
    out["reviwo_b"] = make_fixture_policy(root / "reviwo_b", "reviwo")  # a second, independently loaded policy
    yield out
    torch.set_num_threads(old)


def choices_of(policy) -> dict:
    out = session_choices(policy, "unused")
    out.pop("session_id")
    return out


@contextlib.contextmanager
def serve(pols: dict, **kw):
    kw.setdefault("warmup_iterations", 0)
    srv = PolicyServer(pols, **kw)
    srv.start()
    try:
        yield srv
    finally:
        srv.shutdown()


class Feed:
    """Synthetic native frames -> the client (CPU crop/resize); keeps the client's preprocessed frames so the direct
    path pushes exactly the same Frame contents."""

    def __init__(self, client: PolicyClient, choices: dict, seed: int = 3):
        self.client, self.choices, self.rng = client, choices, np.random.default_rng(seed)
        self.spec = client.spec
        self.frames = {c: [] for c in self.spec["camera_ids"]}
        self.depth = "depth" in self.spec["modalities"]

    def t(self, k: int, v: int) -> int:
        return T0 + int(round(k * self.spec["frame_period_s"] * 1e9)) + v * 1_000_000

    def add(self, k: int, *, jitter_ns: int = 0, disp: float = 0.001) -> None:
        for v, cam in enumerate(self.spec["camera_ids"]):
            rgb = self.rng.integers(0, 256, (*NATIVE_HW, 3), dtype=np.uint8)
            d = valid = corr = None
            if self.depth:
                d = self.rng.uniform(0.3, 1.2, NATIVE_HW).astype(np.float32)
                d[::5, ::7] = 0
                valid, corr = d > 0, self.choices["native_depth_correction"][cam]
            out = self.client.add_frame_native(cam, k, self.t(k, v) + jitter_ns, rgb, pose7=POSE, gripper_register=GRIP,
                                               depth=d, valid=valid, native_correction=corr,
                                               K_native=self.K_native(cam))
            meta = {"depth_source": "native_realsense", "native_correction": corr} if self.depth else {}
            self.frames[cam].append(Frame(k, self.t(k, v) + jitter_ns, out["rgb"], out.get("depth"), out.get("valid"),
                                          meta))
        if self.spec["views"] > 1:
            self.client.add_sync_set({c: k for c in self.spec["camera_ids"]}, disp)

    def K_native(self, cam):
        return (self.choices.get("calibration") or {}).get(cam, {}).get("K_native")

    def geometry(self, cam) -> dict:
        """native_hw (+ the session K at the training grid) for frames added with ``add_frame``."""
        out = {"native_hw": NATIVE_HW}
        if self.spec["needs_calibration"]:
            out["K"] = self.spec["calibration"][cam]["K"]
        return out

    def obs_time(self, k: int) -> int:
        return max(self.t(k, v) for v in range(len(self.spec["camera_ids"])))

    def request(self, k: int, ref=None) -> dict:
        t = self.obs_time(k)
        return {"end_ids": {c: k for c in self.spec["camera_ids"]}, "reference_id": ref or f"pose-{k}",
                "pose7": POSE, "gripper_register": GRIP, "pose_time_ns": t - 3_000_000,
                "gripper_time_ns": t - 40_000_000}

    def step(self, k: int, ref=None, **kw) -> StepResult:
        r = self.request(k, ref)
        return self.client.step(r.pop("end_ids"), r.pop("reference_id"), r.pop("pose7"), r.pop("gripper_register"),
                                **r, **kw)

    def submit(self, k: int, ref=None, **kw) -> int:
        r = self.request(k, ref)
        return self.client.submit(r.pop("end_ids"), r.pop("reference_id"), r.pop("pose7"), r.pop("gripper_register"),
                                  **r, **kw)


def client_for(srv, pid, policy, **kw) -> tuple[PolicyClient, Feed]:
    choices = choices_of(policy)
    kw.setdefault("step_timeout_s", 60.0)
    c = PolicyClient("127.0.0.1", srv.port, policy_id=pid, choices=choices, **kw)
    c.connect()
    return c, Feed(c, choices)


def direct(policy, feed: Feed, k: int, *, noise, pad: bool = False, ref=None) -> dict:
    """The same history + request through DeployPolicy directly (after the server released the policy)."""
    spec = policy.reset_session(session_id=f"direct-{time.monotonic_ns()}", **feed.choices)
    h = HistoryWindow(spec)
    for cam, frames in feed.frames.items():
        for f in frames:
            if f.seq <= k:
                h.push(cam, f)
    ref_frames = feed.frames[spec.camera_ids[0]]
    disp = {f.seq: 0.001 for f in ref_frames} if spec.views > 1 else None
    window = h.assemble({c: k for c in spec.camera_ids}, surface_disp_m=disp, pad_run_start=pad)
    r = feed.request(k, ref)
    ids = window.binding(r["reference_id"], pose_time_ns=r["pose_time_ns"], gripper_time_ns=r["gripper_time_ns"])
    return policy.infer(window, np.asarray(POSE, np.float64), GRIP, ids, noise=noise)


def wait_until(predicate, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


def running(srv) -> bool:
    with srv.lock:
        return any(s.running is not None for s in srv.sessions.values())


def assert_same(net: dict, ref: dict) -> None:
    """Every output bit-identical (dtype included) except timings and the (necessarily different) session ids."""
    assert set(net) == set(ref)
    for key, want in ref.items():
        got = net[key]
        if key == "timings":
            continue
        if key in ("obs_ids", "metadata"):
            assert {k: v for k, v in got.items() if k != "session_id"} == \
                   {k: v for k, v in want.items() if k != "session_id"}, key
        elif isinstance(want, (np.ndarray, np.generic)):
            assert np.asarray(got).dtype == np.asarray(want).dtype, key
            np.testing.assert_array_equal(got, want, err_msg=key)
        else:
            assert got == want, key


# ------------------------------------------------------------------------------------------------ protocol
def test_protocol_roundtrip_exact_dtypes_and_refusals():
    arrays = {"u8": np.arange(6, dtype=np.uint8).reshape(2, 3), "b": np.array([True, False]),
              "f4": np.float32([1.5, np.nan]), "f8": np.float64(0.1), "i8": np.int64([-3]),
              "u2": np.uint16([65535]), "be": np.arange(3, dtype=">f4")}
    msg = P.decode(P.encode({"type": P.PING, "x": 1}, arrays))
    assert msg.type == P.PING and msg.header["v"] == P.VERSION and msg.header["x"] == 1
    for k, a in arrays.items():
        assert msg.arrays[k].dtype == np.asarray(a).dtype.newbyteorder("<") and msg.arrays[k].shape == np.shape(a)
        np.testing.assert_array_equal(msg.arrays[k], a)
    plain, markers, packed = P.pack_outputs({"a": np.float32([1, 2]), "s": np.float64(2.5), "j": {"x": np.int64(3)}})
    out = P.unpack_outputs(plain, markers, P.decode(P.encode({"type": P.PING}, packed)).arrays)
    assert type(out["s"]) is np.float64 and out["j"] == {"x": 3} and out["a"].dtype == np.float32
    with pytest.raises(P.ProtocolError, match="dtype"):
        P.encode({"type": P.PING}, {"c": np.complex64([1j])})
    good = P.encode({"type": P.PING}, {"x": np.zeros(4, np.uint8)})
    (n,) = P.PREFIX.unpack_from(good)
    header = json.loads(good[4:4 + n])
    for mutate, match in ((lambda h: h["arrays"][0].update(nbytes=5), "nbytes"),
                          (lambda h: h["arrays"][0].update(offset=1), "contiguous"),
                          (lambda h: h["arrays"][0].update(dtype="<c8"), "dtype"),
                          (lambda h: h.update(payload_nbytes=1 << 40), "payload_nbytes"),
                          (lambda h: h.pop("v"), "version")):
        bad = json.loads(json.dumps(header))
        mutate(bad)
        raw = json.dumps(bad).encode()
        with pytest.raises(P.ProtocolError, match=match) as e:
            P.decode(P.PREFIX.pack(len(raw)) + raw + good[4 + n:])
        assert e.value.fatal
    with pytest.raises(P.ProtocolError, match="header length"):
        P.decode(P.PREFIX.pack(1 << 30) + b"{}", max_header_bytes=1024)
    assert np.array_equal(P.validation_noise(7), P.validation_noise(7)) and P.validation_noise(7).dtype == np.float32


def test_client_side_modules_import_without_torch():
    code = ("import sys; import ops.policy_serving.client, ops.policy_serving.protocol, ops.policy_serving.bench, "
            "ops.policy_serving.config; from s4d.rwlearn.deploy import preprocess_frame; "
            "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if m.startswith('torch'))")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-c", code], check=True, cwd=repo,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": repo})


def test_error_codes_follow_the_deploy_refusal_classes():
    from s4d.rwlearn.deploy import errors as E

    cases = {E.IncompleteHistory: "incomplete_history", E.IrregularHistory: "irregular_spacing",
             E.Unsynchronized: "unsynchronized", E.BindingRefused: "binding_refused",
             E.SessionMismatch: "session_unknown", E.ContractViolation: "bad_request", ValueError: "bad_request"}
    for cls, code in cases.items():
        assert classify(cls("any wording")) == code and code in P.ERROR_CODES


def test_config_file_loads_and_refuses_unknown_keys(tmp_path):
    cfg = load_config(DEFAULT_PATH)
    assert cfg["protocol_version"] == P.VERSION and cfg["policies"] == {}  # the staged catalog is the single source
    assert cfg["server"]["catalog"] == "catalog.json" and cfg["server"]["gpu_memory_budget_mb"] > 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"server": {"prot": 1}}))
    with pytest.raises(ValueError, match="unknown server keys"):
        load_config(bad)


def test_catalog_validation_refuses_malformed_catalogs(tmp_path):
    good = {"checkpoint": "a/policy.pt", "sha256": "0" * 64, "condition": "reviwo", "task": "t", "status": "deployable"}
    q = {**good, "status": "quarantined", "reasons": ["vision-blind"]}
    ok = validate_catalog({"schema": CATALOG_SCHEMA, "policies": {"a-s0": good, "q-s0": q}, "aliases": {"a": "a-s0"}})
    assert set(ok["policies"]) == {"a-s0", "q-s0"} and ok["aliases"] == {"a": "a-s0"}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"schema": CATALOG_SCHEMA, "policies": {"a-s0": good}}))
    assert load_catalog(path)["aliases"] == {}
    cases = (({"schema": "other/1"}, "schema"), ({"schema": CATALOG_SCHEMA, "extra": 1}, "unknown catalog keys"),
             ({"schema": CATALOG_SCHEMA, "policies": {"a": {**good, "typo": 1}}}, "unknown keys"),
             ({"schema": CATALOG_SCHEMA, "policies": {"a": {**good, "status": "maybe"}}}, "status"),
             ({"schema": CATALOG_SCHEMA, "policies": {"a": {**good, "status": "quarantined"}}}, "reasons"),
             ({"schema": CATALOG_SCHEMA, "policies": {"a": {k: v for k, v in good.items() if k != "sha256"}}},
              "missing keys"),
             ({"schema": CATALOG_SCHEMA, "policies": {"a": {**good, "dataset": "d.h5", "training_contract": "c"}}},
              "OR"),
             ({"schema": CATALOG_SCHEMA, "policies": {"a": good}, "aliases": {"b": "nope"}}, "unknown policy"),
             ({"schema": CATALOG_SCHEMA, "policies": {"a": good, "b": good}, "aliases": {"b": "a"}}, "shadows"))
    for raw, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_catalog(raw)


# ------------------------------------------------------------------------------------------------ equality
@pytest.mark.parametrize("condition", FIXTURE_CONDITIONS)
def test_direct_equals_network_with_only_new_frames(policies, condition):
    policy = policies[condition]
    with serve({condition: policy}) as srv:
        c, feed = client_for(srv, condition, policy)
        V = len(feed.spec["camera_ids"])
        for k in range(3):
            feed.add(k)
        r2 = feed.step(2, noise_seed=11)
        feed.add(3)
        explicit = P.validation_noise(99)
        r3 = feed.step(3, noise=explicit)
        c.close()
        assert srv.counts["actions"] == 2
    assert r2.ok and r3.ok, (r2.code, r2.detail, r3.code, r3.detail)
    assert (r2.frames_sent, r3.frames_sent) == (3 * V, V)
    assert r3.request_bytes < r2.request_bytes
    assert_same(r2.outputs, direct(policy, feed, 2, noise=P.validation_noise(11)))
    assert_same(r3.outputs, direct(policy, feed, 3, noise=explicit))
    assert r3.outputs["obs_ids"]["frame_ids"] == {cam: [1, 2, 3] for cam in feed.spec["camera_ids"]}
    assert r3.outputs["metadata"]["fixture"] is True
    for key in ("prep_ms", "queue_ms", "encoder_ms", "fql_ms", "infer_ms", "total_ms"):
        assert r3.server_ms[key] >= 0
    assert r3.timings_ms["e2e_ms"] >= r3.timings_ms["rtt_ms"] > 0


def test_run_start_padding_and_incomplete_history(policies):
    for pid in ("reviwo", "mvp3d_multiview"):
        policy = policies[pid]
        with serve({pid: policy}) as srv:
            c, feed = client_for(srv, pid, policy)
            feed.add(0)
            r = feed.step(0, noise_seed=1)
            assert (r.status, r.code) == ("error", "incomplete_history") and "never re-spaced" in r.detail
            r = feed.step(0, noise_seed=1, pad_run_start=True)
            assert r.ok and r.outputs["metadata"]["padded"] is True
            assert r.outputs["obs_ids"]["frame_ids"][feed.spec["camera_ids"][0]] == [0, 0, 0]
            feed.add(1)
            feed.add(2)
            feed.add(4)  # frame 3 was dropped: a gap ends the run, it is never re-spaced
            r = feed.step(4, noise_seed=2)
            assert r.code == "incomplete_history" and "3" in r.detail
            r = feed.step(4, noise_seed=2, pad_run_start=True)  # the training rule: a run start after the gap
            assert r.ok and r.outputs["obs_ids"]["frame_ids"][feed.spec["camera_ids"][0]] == [4, 4, 4]
            c.close()
            assert srv.counts["actions"] == 2
        assert_same(r.outputs, direct(policy, feed, 4, noise=P.validation_noise(2), pad=True))


def test_irregular_spacing_and_missing_multiview_evidence(policies):
    with serve({"reviwo": policies["reviwo"]}) as srv:
        c, feed = client_for(srv, "reviwo", policies["reviwo"])
        feed.add(0)
        feed.add(1, jitter_ns=5_000_000)  # 5 ms off the trained 66.7 ms spacing (tolerance 1 %)
        feed.add(2)
        r = feed.step(2)
        assert r.code == "irregular_spacing"
        c.close()
    with serve({"mvp3d_multiview": policies["mvp3d_multiview"]}) as srv:
        c, feed = client_for(srv, "mvp3d_multiview", policies["mvp3d_multiview"])
        for k in range(3):
            feed.add(k, disp=0.02 if k == 1 else 0.001)  # set 1 moved 2 cm: fails M5
        assert feed.step(2).code == "unsynchronized"
        feed.add(3)
        c._sets.pop((4, 4, 4), None)
        for cam in feed.spec["camera_ids"]:
            c.add_frame(cam, 4, feed.t(4, 0), feed.frames[cam][-1].rgb, pose7=POSE, gripper_register=GRIP,
                        depth=feed.frames[cam][-1].depth, valid=feed.frames[cam][-1].valid,
                        metadata=feed.frames[cam][-1].metadata, **feed.geometry(cam))
        r = feed.step(4)  # frame 4 of every camera without its declared set
        assert r.code in ("incomplete_history", "irregular_spacing")
        c.close()
        assert srv.counts["actions"] == 0


def test_explicit_reset_clears_the_history(policies):
    with serve({"reviwo": policies["reviwo"]}) as srv:
        c, feed = client_for(srv, "reviwo", policies["reviwo"])
        for k in range(3):
            feed.add(k)
        assert feed.step(2).ok
        c.reset()
        assert c.stats["reconnects"] == 0
        r = feed.step(2)
        assert r.code == "incomplete_history" and r.frames_sent == 0
        for k in (10, 11, 12):
            feed.add(k)
        r = feed.step(12)
        assert r.ok and r.frames_sent == 3
        c.close()


# ------------------------------------------------------------------------------------------------ sequencing
def test_duplicate_seq_returns_cached_response_and_frame_resends_are_idempotent(policies):
    with serve({"reviwo": policies["reviwo"]}) as srv:
        c, feed = client_for(srv, "reviwo", policies["reviwo"])
        for k in range(3):
            feed.add(k)
        seq = feed.submit(2, noise_seed=4)
        data = c._inflight[seq].data
        first = c.wait(seq)
        assert first.ok and srv.counts["actions"] == 1
        c.wire.send(data)  # the same seq again (e.g. a retransmission)
        again = c.wire.recv(10.0)
        assert again.type == P.ACTION and again.header["seq"] == seq and again.nbytes == first.response_bytes
        np.testing.assert_array_equal(again.arrays["out.action_norm"], first.outputs["action_norm"])
        assert srv.counts["actions"] == 1  # no re-inference
        c._tip.clear()  # re-send frames the server already holds, under a NEW seq: identical frames are skipped
        r = feed.step(2, noise_seed=4)
        assert r.ok and r.frames_sent == 3 and srv.counts["actions"] == 2
        np.testing.assert_array_equal(r.outputs["action_norm"], first.outputs["action_norm"])
        assert not next(iter(srv.sessions.values())).counts["history_rebuilds"]
        c.close()


def test_stale_ttl_and_superseded_requests(policies):
    with serve({"reviwo": policies["reviwo"]}) as srv:
        c, feed = client_for(srv, "reviwo", policies["reviwo"])
        for k in range(3):
            feed.add(k)
        r = feed.step(2, ttl_ms=0.001)
        assert r.code == "stale" and "ttl" in r.detail
        lock = srv.policies["reviwo"].lock
        lock.acquire()  # keep the single worker busy with the first request
        try:
            s1 = feed.submit(2, "a")
            wait_until(lambda: running(srv))
            s2 = feed.submit(2, "b")
            wait_until(lambda: len(srv.queue) == 1)
            s3 = feed.submit(2, "c")
            wait_until(lambda: srv.counts["error_stale"] == 2)
        finally:
            lock.release()
        r1, r2, r3 = c.wait(s1), c.wait(s2), c.wait(s3)
        assert r1.ok and r3.ok and r3.reference_id == "c"
        assert r2.code == "stale" and "superseded" in r2.detail
        assert srv.counts["actions"] == 2
        c.close()


def test_bounded_queue_overload(policies):
    pols = {"a": policies["reviwo"], "b": policies["reviwo_b"]}
    with serve(pols, max_queue=1) as srv:
        ca, fa = client_for(srv, "a", pols["a"])
        cb, fb = client_for(srv, "b", pols["b"])
        for k in range(3):
            fa.add(k)
            fb.add(k)
        lock = srv.policies["a"].lock
        lock.acquire()
        try:
            s1 = fa.submit(2, "a1")  # running (blocked on the policy)
            wait_until(lambda: running(srv))
            sb = fb.submit(2, "b1")  # queued: the queue is now full
            wait_until(lambda: len(srv.queue) == 1)
            s2 = fa.submit(2, "a2")
            r2 = ca.wait(s2)
        finally:
            lock.release()
        assert r2.code == "overloaded"
        assert ca.wait(s1).ok and cb.wait(sb).ok
        ca.close()
        cb.close()


# ------------------------------------------------------------------------------------------------ connections
def test_reconnect_resumes_session_without_reinference(policies):
    with serve({"reviwo": policies["reviwo"]}) as srv:
        c, feed = client_for(srv, "reviwo", policies["reviwo"])
        sid = c.session_id
        for k in range(3):
            feed.add(k)
        assert feed.step(2).ok
        feed.add(3)
        seq = feed.submit(3)  # the request reaches the server, then the connection breaks
        c.wire.sock.shutdown(socket.SHUT_RDWR)
        r = c.wait(seq)
        assert r.ok and r.frames_sent == 1 and c.session_id == sid and c.stats["resumes"] == 1
        c.wire.close()  # broken BEFORE the next request: it is re-sent with its original seq after the resume
        feed.add(4)
        r = feed.step(4)
        assert r.ok and r.frames_sent == 1 and c.session_id == sid and c.stats["resumes"] == 2
        assert srv.counts["actions"] == 3 and srv.counts["resumes"] == 2
        c.close()


def test_expired_session_is_renewed_and_history_resent(policies):
    with serve({"reviwo": policies["reviwo"]}, session_idle_timeout_s=0.3) as srv:
        c, feed = client_for(srv, "reviwo", policies["reviwo"])
        sid = c.session_id
        for k in range(3):
            feed.add(k)
        assert feed.step(2).ok
        deadline = time.monotonic() + 10
        while srv.sessions and time.monotonic() < deadline:
            time.sleep(0.05)
        idle_start = time.monotonic()
        while srv.sessions:
            assert time.monotonic() - idle_start < 10
            time.sleep(0.05)
        feed.add(3)
        r = feed.step(3)
        assert r.code == "session_unknown" and c.session_id != sid
        feed.add(4)
        r = feed.step(4)  # the buffered history (frames 2, 3 re-sent + new 4) goes to the new session
        assert r.ok and r.frames_sent == 3
        # (C) re-sent frames are not this step's work: e2e starts at the NEW frame, their wait is reported apart
        assert r.timings_ms["resent_frames"] == 2 and r.timings_ms["resent_wait_ms"] >= 300
        assert r.timings_ms["e2e_ms"] < r.timings_ms["resent_wait_ms"]
        c.close()


def test_concurrent_sessions_share_one_loaded_policy(policies):
    with serve({"reviwo": policies["reviwo"]}, max_sessions_per_policy=2) as srv:
        (a, fa), (b, fb) = (client_for(srv, "reviwo", policies["reviwo"]) for _ in range(2))
        assert a.session_id != b.session_id
        for k in range(3):
            fa.add(k)
            fb.add(k)  # same seed: identical frames in both sessions
        sa, sb = fa.submit(2, noise_seed=5), fb.submit(2, noise_seed=5)
        ra, rb = a.wait(sa), b.wait(sb)
        assert ra.ok and rb.ok
        np.testing.assert_array_equal(ra.outputs["action_norm"], rb.outputs["action_norm"])
        assert ra.outputs["metadata"]["session_id"] == a.session_id != rb.outputs["metadata"]["session_id"]
        with pytest.raises(ServingError, match="session_busy"):
            client_for(srv, "reviwo", policies["reviwo"])
        a.close()
        b.close()


def test_session_ownership_takeover_and_bye(policies):
    with serve({"reviwo": policies["reviwo"]}) as srv:
        a, fa = client_for(srv, "reviwo", policies["reviwo"])
        with pytest.raises(ServingError, match="session_busy"):
            client_for(srv, "reviwo", policies["reviwo"])
        b, fb = client_for(srv, "reviwo", policies["reviwo"], takeover=True)
        for k in range(3):
            fb.add(k)
        assert fb.step(2).ok
        b.close()  # BYE ends the session at once
        deadline = time.monotonic() + 5
        while srv.sessions and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not srv.sessions
        for k in range(3):
            fa.add(k)
        r = fa.step(2)  # a was evicted by b's takeover: a TERMINAL result, never an exception (robot loop stop signal)
        assert r.code == "session_taken_over" and r.terminal and not r.ok
        assert fa.step(2).terminal and srv.counts["sessions_opened"] == 2  # nothing sent, no automatic re-HELLO
        a.connect()  # only a deliberate reconnect opens a new session
        assert fa.step(2).ok
        a.close()


def test_version_auth_and_payload_refusals(policies, caplog):
    token = b"fixture-shared-token-0123"
    caplog.set_level(logging.DEBUG)
    with serve({"reviwo": policies["reviwo"]}, token=token, max_payload_bytes=1 << 20) as srv:
        def raw():
            w = P.Wire(socket.create_connection(("127.0.0.1", srv.port), timeout=5))
            welcome = w.recv(5)
            assert welcome.type == P.WELCOME and welcome.header["auth_required"]
            return w, welcome.header

        w, _ = raw()
        w.send(P.encode({"type": P.HELLO, "v": 99, "policy_id": "reviwo", "choices": {}}))
        err = w.recv(5)
        assert err.header["code"] == "version" and err.header["fatal"] and w.recv(5) is None
        w.close()
        for tok in (None, b"wrong"):
            with pytest.raises(ServingError, match="auth"):
                client_for(srv, "reviwo", policies["reviwo"], token=tok)
        c, feed = client_for(srv, "reviwo", policies["reviwo"], token=token)
        for k in range(3):
            feed.add(k)
        assert feed.step(2).ok
        cam = feed.spec["camera_ids"][0]
        c.add_frame(cam, 3, feed.t(3, 0), np.zeros((16, 16, 3), np.uint8), pose7=POSE, gripper_register=GRIP,
                    **feed.geometry(cam))
        r = feed.step(3)
        assert r.code == "bad_request" and r.history_reset and "deploy contract" in r.detail
        c.close()
        w, welcome = raw()  # a malformed array table is fatal: the stream cannot be re-synchronised
        good = P.encode({"type": P.PING}, {"x": np.zeros(4, np.uint8)})
        (n,) = P.PREFIX.unpack_from(good)
        header = json.loads(good[4:4 + n])
        header["arrays"][0]["nbytes"] = 3
        body = json.dumps(header).encode()
        w.send(P.PREFIX.pack(len(body)) + body)  # refused from the header alone (no unread payload -> clean close)
        err = w.recv(5)
        assert err.header["code"] == "bad_request" and err.header["fatal"] and w.recv(5) is None
        w.close()
        w, welcome = raw()
        n = 2 << 20  # declared payload above max_payload_bytes: refused from the header alone, nothing is read
        body = json.dumps({"type": P.PING, "v": P.VERSION, "payload_nbytes": n, "arrays": [
            {"name": "x", "dtype": "|u1", "shape": [n], "offset": 0, "nbytes": n}]}).encode()
        w.send(P.PREFIX.pack(len(body)) + body)
        err = w.recv(5)
        assert err.header["code"] == "bad_request" and "payload_nbytes" in err.header["detail"]
        w.close()
    assert token.decode() not in caplog.text


def test_unreferenced_arrays_and_wrong_depth_dtype_are_refused(policies):
    pid = "mvp3d_single"
    with serve({pid: policies[pid]}) as srv:
        c, feed = client_for(srv, pid, policies[pid])
        feed.add(0)
        cam = feed.spec["camera_ids"][0]
        f = feed.frames[cam][0]
        c.add_frame(cam, 1, feed.t(1, 0), f.rgb, pose7=POSE, gripper_register=GRIP, depth=f.depth.astype(np.float64),
                    valid=f.valid, metadata=f.metadata, **feed.geometry(cam))
        r = feed.step(1, pad_run_start=True)
        assert r.code == "bad_request" and r.history_reset
        seq_data = P.encode({"type": P.STEP, "session_id": c.session_id, "seq": 99, "frames": [], "sets": [],
                             "request": feed.request(0)}, {"stray": np.zeros(3, np.uint8)})
        c.wire.send(seq_data)
        msg = c.wire.recv(10)
        assert msg.header["code"] == "bad_request" and "unreferenced" in msg.header["detail"]
        c.close()
        assert srv.counts["actions"] == 0


def test_graceful_shutdown_answers_queued_requests(policies):
    pols = {"a": policies["reviwo"], "b": policies["reviwo_b"]}
    srv = PolicyServer(pols, warmup_iterations=1)
    srv.start()
    assert "encoder_ms" in srv.policies["a"].warmup_ms
    ca, fa = client_for(srv, "a", pols["a"])
    cb, fb = client_for(srv, "b", pols["b"])
    for k in range(3):
        fa.add(k)
        fb.add(k)
    lock = srv.policies["a"].lock
    lock.acquire()
    stopper = None
    try:
        sa = fa.submit(2)
        wait_until(lambda: running(srv))
        sb = fb.submit(2)
        wait_until(lambda: len(srv.queue) == 1)
        stopper = threading.Thread(target=srv.shutdown)
        stopper.start()
        rb = cb.wait(sb, 10)
    finally:
        lock.release()
        if stopper is not None:
            stopper.join(30)
    assert rb.code == "shutting_down"
    ra = ca.wait(sa, 2)
    assert ra.status in ("ok", "error")
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", srv.port), timeout=2)


# ------------------------------------------------------------------------------------------------ client helpers
def test_action_expiry_and_binding_helpers():
    outputs = {"obs_ids": {"observation_time_ns": 1_000_000_000}, "flange_pose": np.zeros(7),
               "gripper_closure": np.float64(0.2), "gripper_register": 48, "measured_pose7": np.ones(7)}
    r = StepResult("ok", 3, "s", reference_id="pose-3", outputs=outputs)
    assert check_action(r, reference_id="pose-3", now_ns=1_050_000_000, max_age_ms=100, session_id="s")[0]
    assert not check_action(r, reference_id="pose-4", now_ns=1_050_000_000, max_age_ms=100)[0]
    ok, why = check_action(r, reference_id="pose-3", now_ns=1_200_000_000, max_age_ms=100)
    assert not ok and "expired" in why
    assert not check_action(r, reference_id="pose-3", now_ns=900_000_000, max_age_ms=100)[0]
    assert not check_action(StepResult("timeout", 4), reference_id="pose-3", now_ns=0, max_age_ms=100)[0]
    cmd = bound_command(r)
    assert cmd["reference_id"] == "pose-3" and set(cmd) >= {"flange_pose", "measured_pose7"}


# Recorded (hw_timestamp_ms, t_recv, t_recv_unix, t) of /home/ws/data/real_world/drawer-open/episode_000000.h5,
# cameras/cam_135122070988 rows 0, 1, 2, 100 (timestamp_domain 2 = global_time): the recorder's own frame times.
RECORDED_COLOUR_STAMPS = [
    (1793879729587.492, 187335.617039116, 1793879729.6524234, 187335.55210772334),
    (1793879729654.2065, 187335.684022002, 1793879729.7194061, 187335.61882238844),
    (1793879729720.9211, 187335.750481397, 1793879729.7858653, 187335.68553736815),
    (1793879736258.9705, 187342.288514136, 1793879736.3238978, 187342.22358679643),
]


def test_device_clock_is_the_recorders_global_time_mapping():
    """#2: capture time = global-time stamp - (realtime - monotonic at receipt), refreshed per frame, no latency added."""
    clock = DeviceClock()
    for hw_ms, t_recv, t_recv_unix, t in RECORDED_COLOUR_STAMPS:
        got = clock.map(hw_ms, 2, offset_ns=int(round((t_recv_unix - t_recv) * 1e9)))
        assert abs(got - t * 1e9) < 1_000  # the training frame time to < 1 us
        assert t_recv * 1e9 - got > 60e6  # arrival is ~65 ms later and never enters the capture time
    assert clock.map(1.0, "timestamp_domain.global_time", offset_ns=0) == 1_000_000
    for domain in (0, 1, "hardware_clock", "timestamp_domain.system_time"):
        with pytest.raises(ValueError, match="global_time"):
            clock.map(1.0, domain, offset_ns=0)
    ticks = iter([100, 102, 200, 204])
    fake = DeviceClock(realtime_ns=lambda: 5_000_000_000_000, monotonic_ns=lambda: next(ticks))
    assert (fake.offset_ns(), fake.offset_ns()) == (5_000_000_000_000 - 101, 5_000_000_000_000 - 202)  # refreshed


def fake_realsense(rgb, intr, *, domain="timestamp_domain.global_time"):
    video = SimpleNamespace(get_intrinsics=lambda: intr)
    video.as_video_stream_profile = lambda: video
    color = SimpleNamespace(profile=video, get_data=lambda: rgb, get_timestamp=lambda: 1234.5,
                            get_frame_number=lambda: 77, get_frame_timestamp_domain=lambda: domain)
    calls = []

    class Pipe:
        def start(self, cfg):
            calls.append(cfg.streams)
            return SimpleNamespace(get_stream=lambda kind: video)

        def wait_for_frames(self, timeout_ms):
            return SimpleNamespace(get_color_frame=lambda: color)

        def stop(self):
            calls.append("stop")

    class Config:
        def __init__(self):
            self.streams = []

        def enable_device(self, serial):
            self.streams.append(("device", serial))

        def enable_stream(self, *a):
            self.streams.append(a)

    rs = SimpleNamespace(pipeline=Pipe, config=Config, stream=SimpleNamespace(color="color", depth="depth"),
                         format=SimpleNamespace(rgb8="rgb8", z16="z16"))
    return rs, calls


def test_realsense_capture_checks_session_geometry_and_stamps_capture_time():
    """#2 + #4 with a fake SDK: recorder capture time; the stream must be the session's grid and intrinsics."""
    from s4d.rwlearn.deploy.realsense import color_intrinsics

    intr = SimpleNamespace(width=160, height=128, fx=100.0, fy=100.0, ppx=79.5, ppy=63.5, coeffs=[0] * 5)
    rgb = np.random.default_rng(0).integers(0, 256, (128, 160, 3), dtype=np.uint8)
    K = color_intrinsics(intr)
    clock = DeviceClock(realtime_ns=lambda: 1_000_000_000_000, monotonic_ns=lambda: 400_000)
    rs, calls = fake_realsense(rgb, intr)
    cap = RealSenseCapture("123", native_hw=(128, 160), rs_module=rs, out_size=32, K_native=K, clock=clock)
    out = cap.read()
    cap.stop()
    np.testing.assert_array_equal(out["rgb"], preprocess_frame(rgb, out_size=32)["rgb"])
    assert out["frame_id"] == 77 and out["native_hw"] == [128, 160] and calls[-1] == "stop"
    assert out["capture_ns"] == 1_234_500_000 - (1_000_000_000_000 - 400_000)  # global-time stamp, no latency
    assert calls[0][0] == ("device", "123")
    for kwargs, match in (({"native_hw": (480, 640)}, "not the session"),
                          ({"native_hw": (128, 160), "K_native": K + np.diag([0, 0, 0]) + [[0, 0, 28], [0, 0, 0],
                                                                                          [0, 0, 0]]},
                           "intrinsics")):
        rs, calls = fake_realsense(rgb, intr)
        with pytest.raises(ValueError, match=match):
            RealSenseCapture("123", rs_module=rs, out_size=32, clock=clock, **kwargs)
        assert calls[-1] == "stop"
    rs, _ = fake_realsense(rgb, intr, domain="timestamp_domain.hardware_clock")
    with pytest.raises(ValueError, match="global_time"):
        RealSenseCapture("123", native_hw=(128, 160), rs_module=rs, out_size=32, clock=clock).read()


def test_bench_reports_payload_and_latency_breakdown(policies):
    pid = "mvp3d_multiview"
    with serve({pid: policies[pid]}) as srv:
        c = PolicyClient("127.0.0.1", srv.port, policy_id=pid, choices=bench.synthetic_choices((128, 160)),
                         step_timeout_s=60)
        c.connect()
        res = bench.run(c, mode="max", steps=4, warmup=1, native_hw=(128, 160), timeout_s=60, noise_seed=0)
        c.reset()
        paced = bench.run(c, mode="paced", steps=3, warmup=0, hz=50, native_hw=(128, 160), timeout_s=60)
        c.close()
    assert res["ok"] == 4 and not res["errors"] and paced["ok"] == 3
    assert res["payload_bytes"]["first_frames"] == 9 and res["payload_bytes"]["steady_frames"] == 3
    assert res["payload_bytes"]["first_request"] > 2.5 * res["payload_bytes"]["steady_request_median"]
    for key in ("e2e_ms", "preprocess_ms", "serialize_ms", "network_ms"):
        assert res["client_ms"][key]["n"] == 4
    assert res["server_ms"]["encoder_ms"]["p50"] > 0 and res["achieved_hz"] > 0
    json.dumps(res, allow_nan=False)


# ------------------------------------------------------------------------------------------------ review 7f7c15a
def test_contract_workers_config_list_includes_policy_serving():
    """#1: the release suite's config inventory lists configs/realworld/policy_serving.yaml (and it parses)."""
    import yaml

    from ops.policy_serving.config import REPO

    configs = sorted(p.stem for p in (REPO / "configs/realworld").glob("*.yaml"))
    assert "policy_serving" in configs
    assert isinstance(yaml.safe_load((REPO / "configs/realworld/policy_serving.yaml").read_text()), dict)
    text = (REPO / "tests/test_realworld_contract_workers.py").read_text()
    assert '"inference_server", "mug_hanging", "policy_serving", "upstream_pins"' in text


def test_client_refuses_state_samples_outside_the_training_binding(policies):
    """#2: pose / gripper must be the samples at or before the observation, <= 15 / 100 ms old (checked before
    sending); check_action ages the action from the observation's capture time."""
    with serve({"reviwo": policies["reviwo"]}) as srv:
        c, feed = client_for(srv, "reviwo", policies["reviwo"])
        for k in range(3):
            feed.add(k)
        t, end = feed.obs_time(2), {cam: 2 for cam in feed.spec["camera_ids"]}
        for pose_t, grip_t, match in ((t - 60_000_000, t, "pose_time_ns"), (t + 1_000_000, t, "pose_time_ns"),
                                      (t, t - 150_000_000, "gripper_time_ns")):
            with pytest.raises(ValueError, match=match):
                c.submit(end, "x", POSE, GRIP, pose_time_ns=pose_t, gripper_time_ns=grip_t)
        assert all(s.last_seq == 0 for s in srv.sessions.values())  # nothing was sent
        r = feed.step(2)
        assert r.ok
        assert check_action(r, reference_id="pose-2", now_ns=t + 50_000_000, max_age_ms=60)[0]
        assert not check_action(r, reference_id="pose-2", now_ns=t + 70_000_000, max_age_ms=60)[0]
        c.close()


def test_wire_mid_message_timeout_is_a_lost_connection_bounded_by_the_deadline():
    """#3: a message cut mid-way raises ConnectionError at the CALLER's deadline (not TimeoutError after io_timeout)."""
    listener = socket.create_server(("127.0.0.1", 0))
    b = socket.create_connection(listener.getsockname()[:2])
    a, _ = listener.accept()
    listener.close()
    w = P.Wire(a, io_timeout_s=5.0)
    data = P.encode({"type": P.PING}, {"x": np.zeros(1000, np.uint8)})
    b.sendall(data[:30])
    t0 = time.monotonic()
    with pytest.raises(ConnectionError, match="stream position lost"):
        w.recv(0.3)
    assert time.monotonic() - t0 < 1.5
    with pytest.raises(TimeoutError):  # nothing started: a plain timeout, the stream is intact
        P.Wire(b, io_timeout_s=5.0).recv(0.1)
    a.close()
    b.close()


def test_client_reconnects_after_a_response_cut_mid_message(policies):
    """#3 end to end: a stalled half response -> the client drops the connection and resumes; no ServingError."""
    spec = policies["reviwo"].input_spec.bind(session_id="fake", **choices_of(policies["reviwo"])).to_json()
    listener = socket.create_server(("127.0.0.1", 0))
    port, hellos = listener.getsockname()[1], []

    def fake_server():
        conns = []
        try:
            for i in range(2):
                sock, _ = listener.accept()
                conns.append(sock)
                w = P.Wire(sock)
                w.send(P.encode({"type": P.WELCOME, "nonce": "n", "auth_required": False,
                                 "policies": {"reviwo": {"input_spec": spec}}}))
                hello = w.recv(5)
                hellos.append(hello.header.get("resume"))
                w.send(P.encode({"type": P.SPEC, "session_id": "s1", "resumed": i == 1, "input_spec": spec,
                                 "limits": {}, "last_seq_received": i, "redeliver": []}))
                w.recv(5)  # a STEP (the first, then the second one re-sent after the resume)
                if i == 0:  # half a response, then silence
                    w.send(P.encode({"type": P.PING}, {"x": np.zeros(4000, np.uint8)})[:40])
            time.sleep(3)
        except OSError:
            pass
        for sock in conns:
            sock.close()

    thread = threading.Thread(target=fake_server, daemon=True)
    thread.start()
    c = PolicyClient("127.0.0.1", port, policy_id="reviwo", choices=choices_of(policies["reviwo"]), step_timeout_s=1.0)
    c.connect()
    feed = Feed(c, choices_of(policies["reviwo"]))
    for k in range(3):
        feed.add(k)
    t0 = time.monotonic()
    r = feed.step(2)  # cut at the step deadline: an error result, never ServingError("header length 0")
    assert r.status == "error" and r.code == "disconnected" and time.monotonic() - t0 < 2.0  # not io_timeout (5 s)
    assert c.stats["connection_errors"] == 1 and c.wire is None
    r = feed.step(2)  # the next step reconnects and resumes the same session, re-sending its request
    assert r.status == "timeout" and c.stats["resumes"] == 1 and hellos[1] == {"session_id": "s1", "last_ack": 0}
    c.wire.close()
    listener.close()
    thread.join(5)


def test_frames_of_another_native_geometry_are_refused(policies):
    """#4: client side (grid, intrinsics) and server side (declared native grid / training-grid K per frame)."""
    pid = "mvp3d_single"
    with serve({pid: policies[pid]}) as srv:
        c, feed = client_for(srv, pid, policies[pid])
        cam = feed.spec["camera_ids"][0]
        rgb = np.zeros((*NATIVE_HW, 3), np.uint8)
        d = np.ones(NATIVE_HW, np.float32)
        corr = feed.choices["native_depth_correction"][cam]
        K = np.asarray(feed.K_native(cam), np.float64)
        with pytest.raises(ValueError, match="native grid"):
            c.add_frame_native(cam, 0, feed.t(0, 0), np.zeros((120, 160, 3), np.uint8), pose7=POSE,
                               gripper_register=GRIP, depth=np.ones((120, 160), np.float32),
                               valid=np.ones((120, 160), bool), native_correction=corr, K_native=K)
        with pytest.raises(ValueError, match="K_native"):
            c.add_frame_native(cam, 0, feed.t(0, 0), rgb, pose7=POSE, gripper_register=GRIP, depth=d, valid=d > 0,
                               native_correction=corr)
        shifted = K.copy()
        shifted[0, 2] += 28
        with pytest.raises(ValueError, match="intrinsics"):
            c.add_frame_native(cam, 0, feed.t(0, 0), rgb, pose7=POSE, gripper_register=GRIP, depth=d, valid=d > 0,
                               native_correction=corr, K_native=shifted)
        for key, value in (("native_hw", [480, 640]), ("K", (np.asarray(feed.spec["calibration"][cam]["K"]) +
                                                             [[0, 0, 1], [0, 0, 0], [0, 0, 0]]).tolist())):
            c.reset()
            feed.add(0)
            c._buffer[cam][0].metadata[key] = value  # a client that skipped its own check
            r = feed.step(0, pad_run_start=True)
            assert r.code == "bad_request" and r.history_reset and ("native grid" in r.detail or
                                                                    "intrinsics" in r.detail)
        assert srv.counts["actions"] == 0
        c.close()


def test_worker_survives_a_failure_after_inference(policies, monkeypatch):
    """#5: an exception after infer() (packing/encoding) answers 'internal'; the worker stays alive."""
    real, calls = P.pack_outputs, []

    def flaky(outputs, *a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("injected encoding failure")
        return real(outputs, *a, **k)

    monkeypatch.setattr(P, "pack_outputs", flaky)
    with serve({"reviwo": policies["reviwo"]}) as srv:
        c, feed = client_for(srv, "reviwo", policies["reviwo"])
        for k in range(3):
            feed.add(k)
        r = feed.step(2)
        assert r.code == "internal" and "injected" in r.detail
        assert feed.step(2).ok and all(t.is_alive() for t in srv.threads)
        c.close()


def test_non_finite_json_and_unserialisable_spec_leave_no_session(policies, monkeypatch):
    """#6: NaN / Infinity are refused at parse time; a SPEC that cannot be built leaves no live session behind."""
    with serve({"reviwo": policies["reviwo"]}) as srv:
        w = P.Wire(socket.create_connection(("127.0.0.1", srv.port), timeout=5))
        assert w.recv(5).type == P.WELCOME
        body = b'{"type": "HELLO", "v": 1, "policy_id": "reviwo", "choices": {"native_hw": {"cam_a": [NaN, 160]}}, ' \
               b'"arrays": [], "payload_nbytes": 0}'
        w.send(P.PREFIX.pack(len(body)) + body)
        err = w.recv(5)
        assert err.header["code"] == "bad_request" and "non-finite" in err.header["detail"] and err.header["fatal"]
        w.close()
        with pytest.raises(ValueError, match="non-finite"):
            P.loads_strict('{"x": Infinity}')
        live_before = set(policies["reviwo"].live_sessions)
        real = PolicyServer._spec_message

        def broken(self, s, **kw):
            raise ValueError("Out of range float values are not JSON compliant")

        monkeypatch.setattr(PolicyServer, "_spec_message", broken)
        with pytest.raises(ServingError, match="bad_request"):
            client_for(srv, "reviwo", policies["reviwo"])
        assert not srv.sessions and not srv.policies["reviwo"].sessions
        assert not set(policies["reviwo"].live_sessions) - live_before
        monkeypatch.setattr(PolicyServer, "_spec_message", real)
        c, _ = client_for(srv, "reviwo", policies["reviwo"])  # not session_busy: nothing leaked
        c.close()


def test_resent_history_after_a_pipelined_reset_rebuilds_instead_of_refusing(policies):
    """#7: frames refused before the push are re-sent (history_reset); a re-sent buffer older than frames a pipelined
    request already pushed rebuilds the history (no 'duplicate' dead end)."""
    with serve({"reviwo": policies["reviwo"]}) as srv:
        c, feed = client_for(srv, "reviwo", policies["reviwo"])
        for k in range(3):
            feed.add(k)
        assert feed.step(2).ok
        feed.add(3)
        sa = feed.submit(3, ttl_ms=10 ** 6)  # refused before its frame was pushed -> history reset
        feed.add(4)
        sb = feed.submit(4)  # pipelined before the client saw the reset: the server holds only frame 4
        ra, rb = c.wait(sa), c.wait(sb)
        assert ra.code == "bad_request" and ra.history_reset and rb.code == "incomplete_history"
        feed.add(5)
        r = feed.step(5)  # re-sends 3, 4, 5 -> rebuilt
        assert r.ok and r.outputs["obs_ids"]["frame_ids"][feed.spec["camera_ids"][0]] == [3, 4, 5]
        assert next(iter(srv.sessions.values())).counts["history_rebuilds"] == 1
        c.close()


def test_request_of_a_session_closed_meanwhile_is_answered(policies, monkeypatch):
    """#7: _submit for a session ended while its request was prepared answers session_unknown (recorded)."""
    real = PolicyServer._prepare

    def prepare_then_end(self, s, seq, msg):
        job = real(self, s, seq, msg)
        with self.lock:
            self._end_session(s, "test: ended during preparation")
        return job

    with serve({"reviwo": policies["reviwo"]}) as srv:
        c, feed = client_for(srv, "reviwo", policies["reviwo"])
        for k in range(3):
            feed.add(k)
        monkeypatch.setattr(PolicyServer, "_prepare", prepare_then_end)
        r = feed.step(2)
        assert r.code == "session_unknown" and srv.counts["dropped_closed_session"] == 1
        monkeypatch.setattr(PolicyServer, "_prepare", real)
        assert feed.step(2).ok  # renewed once, buffered history re-sent
        c.close()


def test_pipelined_session_unknown_renews_once_without_nested_hello(policies):
    """#8: two in-flight requests of an expired session -> exactly one renewal, no ServingError."""
    with serve({"reviwo": policies["reviwo"]}, session_idle_timeout_s=0.3) as srv:
        c, feed = client_for(srv, "reviwo", policies["reviwo"])
        for k in range(3):
            feed.add(k)
        assert feed.step(2).ok
        wait_until(lambda: not srv.sessions)
        s1, s2 = feed.submit(2, "a"), feed.submit(2, "b")
        r1, r2 = c.wait(s1), c.wait(s2)
        assert r1.code == r2.code == "session_unknown"
        assert srv.counts["sessions_opened"] == 2 and c.stats["renew_nested_ignored"] >= 1
        assert feed.step(2).ok
        c.close()


def test_auth_required_beyond_loopback_pre_session_cap_and_connection_cap(policies, tmp_path):
    """#9: no token -> no non-loopback bind (constructor and CLI, before loading); pre-session messages are capped;
    connections are capped."""
    with pytest.raises(ValueError, match="token"):
        PolicyServer({"reviwo": policies["reviwo"]}, host="0.0.0.0")
    PolicyServer({"reviwo": policies["reviwo"]}, host="0.0.0.0", token=b"t")  # constructed, never started
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"policies": {"x": {"checkpoint": "/nonexistent/policy.pt"}}}))
    with pytest.raises(SystemExit, match="token"):
        main(["serve", "--config", str(cfg), "--policies", "x", "--bind", "0.0.0.0", "--device", "cpu"])
    with serve({"reviwo": policies["reviwo"]}, max_connections=2) as srv:
        w1 = P.Wire(socket.create_connection(("127.0.0.1", srv.port), timeout=5))
        assert w1.recv(5).type == P.WELCOME
        n = 100 << 10  # a 100 KiB message before any HELLO: refused from its header (no unread payload)
        body = json.dumps({"type": P.PING, "v": P.VERSION, "payload_nbytes": n, "arrays": [
            {"name": "x", "dtype": "|u1", "shape": [n], "offset": 0, "nbytes": n}]}).encode()
        w1.send(P.PREFIX.pack(len(body)) + body)
        err = w1.recv(5)
        assert err.header["code"] == "bad_request" and "payload_nbytes" in err.header["detail"] and err.header["fatal"]
        w1.close()
        wait_until(lambda: not srv.connections)
        keep = [P.Wire(socket.create_connection(("127.0.0.1", srv.port), timeout=5)) for _ in range(2)]
        assert all(w.recv(5).type == P.WELCOME for w in keep)
        w3 = P.Wire(socket.create_connection(("127.0.0.1", srv.port), timeout=5))
        err = w3.recv(5)
        assert err.header["code"] == "overloaded" and err.header["fatal"] and w3.recv(5) is None
        assert srv.counts["connections_refused"] == 1
        for w in (*keep, w3):
            w.close()


def test_replay_demo_labels_and_first_request():
    """#10 + (D): depth provenance is labelled from the alignment actually used; the first request is the first one
    sent (rows skipped before a request carry no bytes)."""
    import importlib.util

    from ops.policy_serving.config import REPO

    spec = importlib.util.spec_from_file_location("policy_replay_demo", REPO / "scripts/policy_replay_demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    for alignment in ("rs_align", "sensor_depth"):
        meta = demo.depth_metadata({"depth": 1, "alignment": alignment, "depth_timestamp_ns": 5})
        assert meta == {"alignment": f"{alignment} (recorded replay)", "depth_timestamp_ns": 5}
    assert demo.depth_metadata({"rgb": 1}) == {}
    rows = [{"status": "skipped", "request_bytes": 0}, {"status": "ok", "request_bytes": 50056},
            {"status": "ok", "request_bytes": 50055}]
    assert demo.first_request(rows) == 50056 and demo.first_request(rows[:1]) is None


def test_fixture_checkpoints_are_not_served_by_default(tmp_path):
    """#11: the CLI loader refuses FIXTURE weights unless explicitly allowed."""
    make_fixture_policy(tmp_path / "fx", "reviwo")
    entry = {"checkpoint": str(tmp_path / "fx/policy.pt"), "export": str(tmp_path / "fx/encoder.pt")}
    with pytest.raises(ValueError, match="FIXTURE"):
        load_policy(entry, root=None, device="cpu")
    policy, info = load_policy(entry, root=None, device="cpu", allow_fixture=True)
    assert info["fixture"] is True and policy.input_spec.fixture


def test_evicted_client_gets_a_terminal_result_never_an_exception(policies):
    """(A): a client evicted while disconnected renews into a busy policy -> terminal 'session_taken_over'."""
    with serve({"reviwo": policies["reviwo"]}) as srv:
        a, fa = client_for(srv, "reviwo", policies["reviwo"], step_timeout_s=10)
        for k in range(3):
            fa.add(k)
        assert fa.step(2).ok
        a.wire.close()  # a drops off the network; b takes the policy over meanwhile
        b, _ = client_for(srv, "reviwo", policies["reviwo"], takeover=True)
        r = fa.step(2)  # reconnect -> resume refused -> renewal finds the policy busy
        assert r.terminal and r.code == "session_taken_over" and "another client" in r.detail
        with pytest.raises(SessionTakenOver):
            fa.submit(2, "again")
        b.close()
        a.close()


def test_gpu_is_enforced_by_the_guard_not_the_container(monkeypatch):
    """(B): CUDA serving only on the configured UUID, checked by ops.resources.gpu_guard."""
    from ops.resources.gpu_guard import GPUIsolationError

    assert enforce_gpu("cpu", None) is None
    with pytest.raises(ValueError, match="gpu_uuid"):
        enforce_gpu("cuda", None)
    with pytest.raises(ValueError, match="cuda:0"):
        enforce_gpu("cuda:1", "GPU-x")
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    other = next(u for u in ("GPU-9bcf38b6-8240-e910-6548-ca715643dc11", "GPU-00000000-0000-0000-0000-000000000000")
                 if u != visible)  # host-independent: never this process's device (also inside shkim_rw)
    with pytest.raises(GPUIsolationError):
        enforce_gpu("cuda", other)
    mapping = enforce_gpu("cuda", visible)
    assert len(mapping) == 1 and mapping[0]["uuid"] == visible
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(GPUIsolationError):
        enforce_gpu("cuda", visible)  # bare indices are refused by the guard


# ------------------------------------------------------------------------------------------------ catalog registry
def stage_fixture(root: Path, name: str, condition: str = "reviwo", normalize_q_loss: bool = False, **extra):
    """A FIXTURE checkpoint (+ export) written under ``root/name`` and its catalog entry (paths relative to root);
    ``normalize_q_loss``: an a10nq-trained checkpoint (agent.config)."""
    policy = make_fixture_policy(root / name, condition, normalize_q_loss=normalize_q_loss)
    entry = {"checkpoint": f"{name}/policy.pt", "sha256": sha256_file(root / name / "policy.pt"),
             "condition": condition, "task": "fixture-task", "seed": 0, "run": f"fixture-{name}",
             "status": "deployable", "flags": ["fixture"], "reasons": [], **extra}
    if (root / name / "encoder.pt").exists():
        entry["export"] = f"{name}/encoder.pt"
    return policy, entry


def write_catalog(path: Path, policies: dict, aliases: dict | None = None) -> Path:
    path.write_text(json.dumps({"schema": CATALOG_SCHEMA, "generated_utc": "fixture", "criteria_sha256": None,
                                "policies": policies, "aliases": aliases or {}}, indent=1))
    return path


@contextlib.contextmanager
def serve_catalog(root: Path, catalog: Path, **kw):
    kw.setdefault("warmup_iterations", 0)
    srv = PolicyServer(catalog=catalog, checkpoint_root=root, device="cpu", allow_fixture=True, **kw)
    srv.start()
    try:
        yield srv
    finally:
        srv.shutdown()


def welcome(srv) -> dict:
    w = P.Wire(socket.create_connection(("127.0.0.1", srv.port), timeout=5))
    try:
        msg = w.recv(5)
        assert msg.type == P.WELCOME
        return msg.header
    finally:
        w.close()


def raw_hello(srv, pid: str, choices: dict) -> P.Message:
    """HELLO without the client's own offered-id check (the server's refusal is what is tested)."""
    w = P.Wire(socket.create_connection(("127.0.0.1", srv.port), timeout=5))
    try:
        assert w.recv(5).type == P.WELCOME
        w.send(P.encode({"type": P.HELLO, "policy_id": pid, "choices": P.jsonable(choices)}))
        return w.recv(30)
    finally:
        w.close()


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    root = tmp_path_factory.mktemp("serving-catalog")
    out = {name: stage_fixture(root, name, cond) for name, cond in
           (("r1", "reviwo"), ("r2", "reviwo"), ("r3", "reviwo"), ("r1b", "reviwo"), ("s1", "sincro"),
            ("e1", "e2e_cnn"))}
    out["d1"] = stage_fixture(root, "d1", "dynamo", normalize_q_loss=True)  # an a10nq-trained DynaMo checkpoint
    out["d0"] = stage_fixture(root, "d0", "dynamo")  # the same architecture trained in the legacy setting
    yield root, out
    torch.set_num_threads(old)


def test_lazy_load_on_first_hello_alias_and_advertised_spec(staged, tmp_path):
    root, fx = staged
    (pol, e), (spol, se) = fx["r1"], fx["s1"]
    cat = write_catalog(tmp_path / "catalog.json", {"reviwo-x-s0": e, "sincro-x-s0": se},
                        {"reviwo-x": "reviwo-x-s0"})
    with serve_catalog(root, cat, warmup_iterations=1) as srv:
        assert srv.policies == {} and srv.counts["policies_loaded"] == 0  # nothing loaded before a selection
        hello = welcome(srv)
        offered = hello["policies"]
        assert set(offered) == {"reviwo-x-s0", "sincro-x-s0", "reviwo-x"} and hello["unavailable"] == {}
        assert offered["reviwo-x"]["info"]["alias_of"] == "reviwo-x-s0" and not offered["reviwo-x-s0"]["info"]["resident"]
        assert offered["reviwo-x-s0"]["input_spec"] == spec_json(pol.input_spec) == describe_entry(e, root)
        assert offered["sincro-x-s0"]["input_spec"] == spec_json(spol.input_spec)
        assert srv.policies == {}  # advertising never loads
        c, feed = client_for(srv, "reviwo-x", pol)  # the alias selects (and loads) the canonical policy
        assert srv.counts["policies_loaded"] == 1 and set(srv.policies) == {"reviwo-x-s0"}
        sp = srv.policies["reviwo-x-s0"]
        assert sp.warmup_ms is not None and sp.footprint == policy_footprint_bytes(pol) > 0
        for k in range(3):
            feed.add(k)
        r = feed.step(2, noise_seed=11)
        c.close()
        assert r.ok, (r.code, r.detail)
        c2, feed2 = client_for(srv, "reviwo-x-s0", pol)  # resident: reused, not reloaded
        assert srv.counts["policies_loaded"] == 1 and srv.policies["reviwo-x-s0"] is sp
        assert welcome(srv)["policies"]["reviwo-x-s0"]["info"]["resident"] is True
        c2.close()
    assert_same(r.outputs, direct(pol, feed, 2, noise=P.validation_noise(11)))  # lazily loaded == direct, bitwise


def test_spec_mismatch_and_quarantine_are_refused(staged, tmp_path):
    root, fx = staged
    (pol, e), spol = fx["r1"], fx["s1"][0]
    # advertises sincro's contract (label consistent with it: the validator refuses a label/spec mismatch) for a reviwo
    # checkpoint: only the load-time comparison with the checkpoint can catch it
    lying = {**e, "condition": "sincro", "input_spec": spec_json(spol.input_spec)}
    q = {**fx["r2"][1], "status": "quarantined", "reasons": ["vision-blind adapter input (fixture label)"]}
    cat = write_catalog(tmp_path / "catalog.json", {"lying-s0": lying, "q-s0": q})
    with serve_catalog(root, cat) as srv:
        hello = welcome(srv)
        assert set(hello["policies"]) == {"lying-s0"}
        assert hello["unavailable"] == {"q-s0": {"status": "quarantined", "reasons": q["reasons"]}}
        err = raw_hello(srv, "lying-s0", choices_of(spol))
        assert err.header["code"] == "bad_request" and "input_spec differs" in err.header["detail"]
        assert srv.policies == {}  # refused, never served
        err = raw_hello(srv, "q-s0", choices_of(pol))
        assert err.header["code"] == "bad_request" and "quarantined" in err.header["detail"]
        assert srv.policies == {}
    with serve_catalog(root, cat, allow_quarantined=True) as srv:
        info = welcome(srv)["policies"]["q-s0"]["info"]
        assert info["status"] == "quarantined" and info["reasons"] == q["reasons"]
        c, feed = client_for(srv, "q-s0", fx["r2"][0])
        for k in range(3):
            feed.add(k)
        assert feed.step(2).ok and srv.policies["q-s0"].info["status"] == "quarantined"
        c.close()


def test_budget_lru_eviction_never_evicts_a_live_session(staged, tmp_path):
    root, fx = staged
    pol = fx["r1"][0]
    cat = write_catalog(tmp_path / "catalog.json", {n: fx[n][1] for n in ("r1", "r2", "r3")})
    fp = policy_footprint_bytes(pol)
    probe = PolicyServer(catalog=cat, checkpoint_root=root, device="cpu", allow_fixture=True)
    est = probe._estimate(probe.entries["r2"])
    budget = 2 * max(fp, est) + 1
    assert fp + est <= budget < 2 * fp + est  # two fit, a third forces an eviction
    with serve_catalog(root, cat, gpu_memory_budget_bytes=budget) as srv:
        a, _ = client_for(srv, "r1", pol)
        a.close()  # BYE: r1 stays resident, idle
        wait_until(lambda: not srv.policies["r1"].sessions)
        b, fb = client_for(srv, "r2", pol)
        assert set(srv.policies) == {"r1", "r2"}
        c, fc = client_for(srv, "r3", pol)  # r1 is the least recently used idle policy -> evicted
        assert set(srv.policies) == {"r2", "r3"} and srv.counts["policies_evicted"] == 1
        with pytest.raises(ServingError, match="overloaded") as exc:  # r2 and r3 have live sessions
            client_for(srv, "r1", pol)
        assert "live sessions" in str(exc.value) and set(srv.policies) == {"r2", "r3"}
        for k in range(3):
            fb.add(k)
        assert fb.step(2).ok  # the live session was never disturbed
        b.close()
        wait_until(lambda: not srv.policies["r2"].sessions)
        a2, _ = client_for(srv, "r1", pol)  # r2 idle now -> evicted for r1
        assert set(srv.policies) == {"r1", "r3"} and srv.counts["policies_loaded"] == 4
        for cl in (a2, c):
            cl.close()


def test_max_resident_policies_cap(staged, tmp_path):
    root, fx = staged
    pol = fx["r1"][0]
    cat = write_catalog(tmp_path / "catalog.json", {n: fx[n][1] for n in ("r1", "r2")})
    with serve_catalog(root, cat, max_resident_policies=1) as srv:
        a, _ = client_for(srv, "r1", pol)
        with pytest.raises(ServingError, match="overloaded"):
            client_for(srv, "r2", pol)
        a.close()
        wait_until(lambda: not srv.policies["r1"].sessions)
        b, _ = client_for(srv, "r2", pol)
        assert set(srv.policies) == {"r2"}
        b.close()


def test_catalog_reload_offers_new_entries_and_swaps_changed_ones_after_their_session(staged, tmp_path):
    root, fx = staged
    pol = fx["r1"][0]
    path = tmp_path / "catalog.json"
    write_catalog(path, {"r1": fx["r1"][1]})
    with serve_catalog(root, path) as srv:
        a, fa = client_for(srv, "r1", pol)
        old = srv.policies["r1"]
        write_catalog(path, {"r1": fx["r1"][1], "r2": fx["r2"][1]})
        b, _ = client_for(srv, "r2", pol)  # the WELCOME of a new connection re-reads the changed catalog
        assert srv.counts["catalog_reloads"] == 1
        b.close()
        w = P.Wire(socket.create_connection(("127.0.0.1", srv.port), timeout=5))
        assert "r3" not in w.recv(5).header["policies"]
        write_catalog(path, {"r1": fx["r1"][1], "r2": fx["r2"][1], "r3": fx["r3"][1]})
        w.send(P.encode({"type": P.HELLO, "policy_id": "r3", "choices": P.jsonable(choices_of(pol))}))
        assert w.recv(30).type == P.SPEC  # an unknown id in a HELLO re-reads the catalog
        w.close()
        assert srv.counts["catalog_reloads"] == 2
        write_catalog(path, {"r1": fx["r1b"][1], "r3": fx["r3"][1]})  # r1 changed (new files), r2 removed
        summary = srv.reload_catalog()
        assert summary["changed"] == ["r1"] and summary["removed"] == ["r2"]
        assert "r2" not in srv.policies  # idle: unloaded at once
        assert srv.policies["r1"] is old and old.retired  # still serves its live session
        for k in range(3):
            fa.add(k)
        assert fa.step(2).ok
        a.wire.close()  # a dropped connection resumes onto the retired object
        assert fa.step(2).ok and srv.counts["resumes"] == 1 and srv.policies["r1"] is old
        with pytest.raises(ServingError, match="session_busy"):
            client_for(srv, "r1", pol)
        a.close()
        wait_until(lambda: "r1" not in srv.policies)  # unloaded when its last session ended
        c, fc = client_for(srv, "r1", pol)
        assert srv.policies["r1"] is not old and srv.policies["r1"].info["checkpoint"].endswith("r1b/policy.pt")
        for k in range(3):
            fc.add(k)
        assert fc.step(2).ok
        c.close()
        path.write_text("{not json")
        assert srv.reload_catalog() is None and set(srv.entries) == {"r1", "r3"}  # invalid: old table kept


def test_preload_and_ready_info(staged, tmp_path):
    root, fx = staged
    cat = write_catalog(tmp_path / "catalog.json", {"e1": fx["e1"][1], "r1": fx["r1"][1]}, {"legacy": "r1"})
    with pytest.raises(ValueError, match="unknown preload"):
        PolicyServer(catalog=cat, checkpoint_root=root, device="cpu", allow_fixture=True, preload=["nope"])
    with pytest.raises(ValueError, match="unknown policy ids"):
        PolicyServer(catalog=cat, checkpoint_root=root, device="cpu", allow_fixture=True, offered=["nope"])
    with serve_catalog(root, cat, preload=["legacy"], warmup_iterations=1, offered=["r1"]) as srv:
        assert set(srv.policies) == {"r1"} and srv.policies["r1"].warmup_ms is not None
        info = srv.ready_info()
        assert info["offered"] == ["r1"] and info["aliases"] == {"legacy": "r1"}
        assert set(info["resident"]) == {"r1"} and info["catalog_sha256"] == sha256_file(cat)
        assert set(welcome(srv)["policies"]) == {"r1", "legacy"}  # e1 is not offered (--policies restriction)
        c, feed = client_for(srv, "r1", fx["r1"][0])
        assert srv.counts["policies_loaded"] == 1  # preloaded, not loaded again
        c.close()
    with pytest.raises(SystemExit, match="catalog"):
        main(["serve", "--config", str(DEFAULT_PATH), "--catalog", str(tmp_path / "missing.json"), "--device", "cpu"])


def test_cli_serves_the_catalog_lazily_and_reloads_on_sighup(staged, tmp_path):
    """The entry point: READY lists offered / aliases / preloaded ids, SIGHUP re-reads the catalog, SIGTERM stops."""
    import signal

    from ops.policy_serving.config import REPO

    root, fx = staged
    cat = write_catalog(tmp_path / "catalog.json", {"r1": fx["r1"][1]}, {"legacy": "r1"})
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"server": {"warmup_iterations": 1, "gpu_memory_budget_mb": 64}}))
    proc = subprocess.Popen([sys.executable, "-I", str(REPO / "ops/policy_serving/server.py"), "serve", "--config",
                             str(cfg), "--catalog", str(cat), "--checkpoint-root", str(root), "--device", "cpu",
                             "--port", "0", "--allow-fixture", "--preload", "legacy", "--threads", "2"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=REPO)
    try:
        ready = json.loads(proc.stdout.readline())
        assert ready["event"] == "ready" and ready["offered"] == ["r1"] and ready["aliases"] == {"legacy": "r1"}
        assert set(ready["resident"]) == {"r1"} and ready["resident"]["r1"]["warmup_ms"] is not None
        assert ready["gpu_memory_budget_mb"] == 64 and ready["catalog_sha256"] == sha256_file(cat)
        write_catalog(cat, {"r1": fx["r1"][1], "r2": fx["r2"][1]}, {"legacy": "r1"})
        proc.send_signal(signal.SIGHUP)
        reloaded = json.loads(proc.stdout.readline())
        assert reloaded["event"] == "catalog_reloaded" and reloaded["ok"] and reloaded["added"] == ["r2"]
        proc.send_signal(signal.SIGTERM)
        assert json.loads(proc.stdout.readline())["event"] == "stopped" and proc.wait(30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(10)


# ------------------------------------------------------------------------------------------------ labels, races (POL)
def test_label_refresh_keeps_the_resident_model_and_execution_eligibility_fails_closed(staged, tmp_path):
    root, fx = staged
    pol, e = fx["r1"]
    cat = write_catalog(tmp_path / "catalog.json", {"r1": e})
    with serve_catalog(root, cat) as srv:
        info = welcome(srv)["policies"]["r1"]["info"]
        assert info["execution_eligible"] is False and info["usage"] == "dry_run_replay_only"  # field absent: closed
        c, _ = client_for(srv, "r1", pol)
        c.close()
        sp = srv.policies["r1"]
        restricted = {**e, "status": "restricted", "reasons": ["e1 LayerNorm-spread FAIL (fixture label)"],
                      "usage": "dry_run_replay_only", "execution_eligible": False, "usage_reasons": ["decision"],
                      "criteria": {"e1_layernorm_spread": {"pass": False, "value": 0.35, "threshold": ">= 0.5"}}}
        write_catalog(cat, {"r1": restricted})
        summary = srv.reload_catalog()
        assert summary["changed"] == [] and srv.policies["r1"] is sp and srv.counts["policies_loaded"] == 1
        info = welcome(srv)["policies"]["r1"]["info"]
        assert info["status"] == "restricted" and info["criteria"]["e1_layernorm_spread"]["pass"] is False
        assert info["resident"] is True and info["execution_eligible"] is False
        spec_msg = raw_hello(srv, "r1", choices_of(pol))
        assert spec_msg.type == P.SPEC and spec_msg.header["info"]["status"] == "restricted"
        assert spec_msg.header["info"]["usage_reasons"] == ["decision"]
        decision = {"decided_by": "user", "at": "2026-11-10T21:10+09:00", "text": "fixture decision"}
        # user decision 2026-11-12 ~16:00: a legacy frozen-encoder entry (r1: reviwo, no FQL-setting prefix) is never
        # execution-eligible: the validator refuses the catalog and the current (restricted) table stays
        write_catalog(cat, {"r1": {**e, "usage": "execution", "execution_eligible": True,
                                   "execution_decision": decision}})
        assert srv.reload_catalog() is None
        info = welcome(srv)["policies"]["r1"]["info"]
        assert info["execution_eligible"] is False and info["status"] == "restricted"
        assert srv._current_info(srv.policies["r1"])["execution_eligible"] is False  # what SPEC publishes


def _dynamo_grant_entries(root, fx):
    """A REAL DynaMo fixture checkpoint staged as a restricted entry (+ its granted form), with its described spec."""
    pol, e = fx["d1"]
    spec = describe_entry(e, root)
    assert spec["condition"] == "dynamo"
    sidecar = root / "d1" / "deploy_contract.json"  # fixtures carry their contract in the checkpoint meta; the file
    sidecar.write_text('{"fixture": true}\n')     # binds the deployment validation to the served contract
    restricted = {**e, "condition": "dynamo", "run": "rwnq-20261112-fql-dynamo-fixture-task-s0", "seed": 0,
                  "training_contract": "d1/deploy_contract.json",
                  "flags": ["fixture", "fql_setting_a10nq"], "input_spec": spec, "status": "restricted",
                  "reasons": ["e1 LayerNorm-spread FAIL (fixture label)"],
                  "usage": "dry_run_replay_only", "execution_eligible": False, "usage_reasons": ["decision"],
                  "criteria": {"e1_layernorm_spread": {"pass": False, "value": 0.35, "threshold": ">= 0.5"}}}
    grant = {"decided_by": "user", "at": "2026-11-12", "text": "fixture DynaMo decision",
             "scope": {"condition": "dynamo", "fql_setting": "a10nq"},
             "acknowledged_failures": ["e1_layernorm_spread"], "checkpoint_sha256": e["sha256"]}
    from ops.policy_serving.config import input_spec_digest

    validation = {"run": restricted["run"], "checkpoint_sha256": e["sha256"], "pass": True, "at": "fixture",
                  "record": "fixture.json", "training_contract_sha256": sha256_file(sidecar),
                  "input_spec_sha256": input_spec_digest(spec),
                  "checks": {k: {"pass": True} for k in ("l2_offline", "server_network", "robotpc_client")}}
    granted = {**restricted, "usage": "execution", "execution_eligible": True, "execution_decision": grant,
               "deploy_validation": validation}
    return pol, e, restricted, grant, granted


def hello_with(srv, pid: str, choices: dict, *, capabilities=None, resume=None) -> P.Message:
    w = P.Wire(socket.create_connection(("127.0.0.1", srv.port), timeout=5))
    try:
        assert w.recv(5).type == P.WELCOME
        h = {"type": P.HELLO, "policy_id": pid, "choices": P.jsonable(choices), "resume": resume}
        if capabilities is not None:
            h["capabilities"] = capabilities
        w.send(P.encode(h))
        return w.recv(30)
    finally:
        w.close()


def test_restricted_grant_is_scoped_bound_published_and_needs_the_client_capability(staged, tmp_path):
    """User decision 2026-11-12 (DynaMo): a restricted entry executes only under a scoped grant that acknowledges its
    FAILED criterion and is bound to its checkpoint; the failure stays published, the model stays resident across the
    label change, and the server binds the granted entry (new session or resume) only for a client declaring the notice
    capability (an older client is refused; Codex reviews of 1847fdd / 0f61212)."""
    from ops.policy_serving.config import RESTRICTED_EXECUTION_CAPABILITY

    root, fx = staged
    pol, e, restricted, grant, granted = _dynamo_grant_entries(root, fx)
    pid = "dynamo-fixture-task-s0-a10nq"
    cat = write_catalog(tmp_path / "catalog.json", {pid: restricted})
    with serve_catalog(root, cat) as srv:
        assert welcome(srv)["policies"][pid]["info"]["requires_client"] == []
        c0, _ = client_for(srv, pid, pol)  # not granted: any client binds (dry run / replay)
        assert c0.spec_info["execution_eligible"] is False and c0.spec_info["status"] == "restricted"
        old_session = c0.session_id
        sp = srv.policies[pid]
        for bad in ({**granted, "execution_decision": {**grant, "checkpoint_sha256": "0" * 64}},  # other weights
                    {**granted, "sha256": None, "execution_decision": {k: v for k, v in grant.items()
                                                                       if k != "checkpoint_sha256"}},
                    {**granted, "execution_decision": {k: v for k, v in grant.items() if k != "scope"}},
                    {**granted, "condition": "reviwo", "run": "rwnq-20261112-fql-reviwo-fixture-task-s0"},
                    {**granted, "run": "r-fql-dynamo-fixture-task-s0", "flags": ["fixture"]},  # legacy DynaMo
                    {**granted, "deploy_validation": None},  # not deployment-validated
                    {k: v for k, v in granted.items() if k != "input_spec"},  # no checkpoint-bound spec
                    {**granted, "status": "quarantined"}):
            write_catalog(cat, {pid: bad})
            assert srv.reload_catalog() is None  # refused by the validator: the current table stays
            assert welcome(srv)["policies"][pid]["info"]["execution_eligible"] is False
        write_catalog(cat, {pid: granted})
        summary = srv.reload_catalog()
        assert summary["changed"] == [] and srv.policies[pid] is sp  # label refresh, model stays resident
        info = welcome(srv)["policies"][pid]["info"]
        assert info["status"] == "restricted" and info["criteria"]["e1_layernorm_spread"]["pass"] is False
        assert info["execution_eligible"] is True and info["execution_decision"] == grant
        assert info["requires_client"] == [RESTRICTED_EXECUTION_CAPABILITY]
        # the session bound BEFORE the grant cannot resume without the capability (it would receive an eligible SPEC)
        refused = hello_with(srv, pid, choices_of(pol), resume={"session_id": old_session, "last_ack": 0})
        assert refused.type == P.ERROR and "requires client capabilities" in refused.header["detail"]
        c0.close()
        refused = hello_with(srv, pid, choices_of(pol))  # an older client's new HELLO: no capabilities declared
        assert refused.type == P.ERROR and "requires client capabilities" in refused.header["detail"]
        c, _ = client_for(srv, pid, pol)  # this client declares the capability it implements
        published = c.spec_info  # what SPEC publishes: the CURRENT labels of the bound policy
        assert published["status"] == "restricted" and published["execution_eligible"] is True
        assert published["execution_decision"]["acknowledged_failures"] == ["e1_layernorm_spread"]
        assert published["checkpoint_sha256"] == e["sha256"]
        c.close()


def test_a_grant_published_while_a_session_is_being_bound_is_enforced_at_the_final_check(staged, tmp_path):
    root, fx = staged
    pol, e, restricted, grant, granted = _dynamo_grant_entries(root, fx)
    pid = "dynamo-fixture-task-s0-a10nq"
    cat = write_catalog(tmp_path / "catalog.json", {pid: restricted})
    with serve_catalog(root, cat) as srv:
        c, _ = client_for(srv, pid, pol)
        c.close()
        sp = srv.policies[pid]
        real = sp.policy.reset_session

        def publish_then_bind(**kw):  # the reload lands between the HELLO's checks and the session registration
            write_catalog(cat, {pid: granted})
            assert srv.reload_catalog() is not None
            return real(**kw)

        sp.policy.reset_session = publish_then_bind
        try:
            refused = hello_with(srv, pid, choices_of(pol))  # no capabilities; the grant was not there at HELLO
        finally:
            sp.policy.reset_session = real
        assert refused.type == P.ERROR and "requires client capabilities" in refused.header["detail"]
        assert not sp.sessions


def test_a_legacy_trained_checkpoint_under_a_new_setting_identity_is_never_advertised_eligible(staged, tmp_path):
    """Codex review of 7518c4f: eligibility follows the setting the WEIGHTS trained with (agent.config)."""
    root, fx = staged
    _, _, _, grant, granted = _dynamo_grant_entries(root, fx)
    lpol, le = fx["d0"]  # same architecture, legacy setting (normalize_q_loss false)
    pid = "dynamo-fixture-task-s0-a10nq"
    forged = {**granted, "checkpoint": le["checkpoint"], "sha256": le["sha256"], "export": le.get("export"),
              "input_spec": describe_entry(le, root),
              "execution_decision": {**grant, "checkpoint_sha256": le["sha256"]},
              "deploy_validation": {**granted["deploy_validation"], "checkpoint_sha256": le["sha256"]}}
    from ops.policy_serving.config import input_spec_digest

    forged["deploy_validation"]["input_spec_sha256"] = input_spec_digest(forged["input_spec"])
    validate_catalog({"schema": CATALOG_SCHEMA, "policies": {pid: forged}, "aliases": {}})  # labels are consistent
    cat = write_catalog(tmp_path / "catalog.json", {pid: forged})
    with serve_catalog(root, cat) as srv:
        w = welcome(srv)
        assert pid not in w["policies"] and "is not the a10nq FQL setting" in str(w["unavailable"][pid]["reasons"])
    cat = write_catalog(tmp_path / "catalog2.json", {pid: granted})  # the a10nq-trained checkpoint: offered
    with serve_catalog(root, cat) as srv:
        assert welcome(srv)["policies"][pid]["info"]["execution_eligible"] is True
    stale = {**granted, "deploy_validation": {**granted["deploy_validation"], "training_contract_sha256": "b" * 64}}
    cat = write_catalog(tmp_path / "catalog3.json", {pid: stale})  # validated against another training contract
    with serve_catalog(root, cat) as srv:
        w = welcome(srv)
        assert pid not in w["policies"] and "another training contract" in str(w["unavailable"][pid]["reasons"])


def test_a_condition_label_that_differs_from_the_checkpoint_is_never_offered_or_granted(staged, tmp_path):
    """Codex review of 0f61212 (P0): a DynaMo grant on a ReViWo checkpoint must not publish condition=dynamo."""
    root, fx = staged
    rpol, re_ = fx["r1"]
    _, _, restricted, grant, granted = _dynamo_grant_entries(root, fx)
    pid = "dynamo-fixture-task-s0-a10nq"
    reviwo_spec = describe_entry(re_, root)
    relabelled = {**granted, "checkpoint": re_["checkpoint"], "sha256": re_["sha256"], "export": re_.get("export"),
                  "execution_decision": {**grant, "checkpoint_sha256": re_["sha256"]},
                  "deploy_validation": {**granted["deploy_validation"], "checkpoint_sha256": re_["sha256"]}}
    from ops.policy_serving.config import input_spec_digest

    with pytest.raises(ValueError, match="differs from its input_spec"):  # label vs the ReViWo checkpoint's spec
        validate_catalog({"schema": CATALOG_SCHEMA, "policies": {pid: {
            **relabelled, "input_spec": reviwo_spec,
            "deploy_validation": {**relabelled["deploy_validation"], "input_spec_sha256": input_spec_digest(reviwo_spec)}}},
            "aliases": {}})
    # a deployable entry labelled dynamo without input_spec: described from the ReViWo checkpoint -> not offered
    plain = {k: v for k, v in re_.items()}
    cat = write_catalog(tmp_path / "catalog.json", {"dynamo-fixture-task-s0-a10nq": {**plain, "condition": "dynamo"}})
    with serve_catalog(root, cat) as srv:
        w = welcome(srv)
        assert pid not in w["policies"] and "differs from the checkpoint" in str(w["unavailable"][pid]["reasons"])
    # labels AND input_spec edited to dynamo on ReViWo weights: the eligible entry's spec is checked against the
    # checkpoint before it is advertised (never published as an eligible DynaMo), and the load would refuse it too
    forged = {**relabelled, "input_spec": {**reviwo_spec, "condition": "dynamo"}}
    forged["deploy_validation"] = {**relabelled["deploy_validation"],
                                   "input_spec_sha256": input_spec_digest(forged["input_spec"])}
    validate_catalog({"schema": CATALOG_SCHEMA, "policies": {pid: forged}, "aliases": {}})
    cat = write_catalog(tmp_path / "catalog2.json", {pid: forged})
    with serve_catalog(root, cat) as srv:
        from ops.policy_serving.config import RESTRICTED_EXECUTION_CAPABILITY

        w = welcome(srv)
        assert pid not in w["policies"] and "differs from the checkpoint" in str(w["unavailable"][pid]["reasons"])
        refused = hello_with(srv, pid, choices_of(rpol), capabilities=[RESTRICTED_EXECUTION_CAPABILITY])
        assert refused.type == P.ERROR and pid not in srv.policies  # never loaded, never bound
    # a label-only refresh (same artefacts: the retained-spec fast path) that changes the condition is refused too
    cat = write_catalog(tmp_path / "catalog3.json", {"reviwo-fixture-task-s0": plain})
    with serve_catalog(root, cat) as srv:
        assert "reviwo-fixture-task-s0" in welcome(srv)["policies"]
        write_catalog(cat, {"reviwo-fixture-task-s0": {**plain, "condition": "dynamo"}})
        assert srv.reload_catalog() is not None
        w = welcome(srv)
        assert "reviwo-fixture-task-s0" not in w["policies"]
        assert "differs from the checkpoint" in str(w["unavailable"]["reviwo-fixture-task-s0"]["reasons"])


def test_eligible_entries_are_advertised_only_with_their_checkpoint_spec_and_unchanged_labels_pass_both_stages(
        staged, tmp_path):
    """Positive control: a granted DynaMo entry and a deployable eligible entry with checkpoint-consistent specs are
    offered; WELCOME -> SPEC -> fresh WELCOME labels are equal, so the launcher's binding / START checks pass."""
    import importlib.util

    from ops.policy_serving.config import REPO

    root, fx = staged
    pol, e, restricted, grant, granted = _dynamo_grant_entries(root, fx)
    rpol, re_ = fx["e1"]  # an end-to-end CNN entry: the class that stays eligible (user decision 2026-11-12 ~16:00)
    decision = {"decided_by": "user", "at": "2026-11-10T21:10+09:00", "text": "fixture decision"}
    dep = {**re_, "input_spec": describe_entry(re_, root), "usage": "execution", "execution_eligible": True,
           "execution_decision": decision}
    spec = importlib.util.spec_from_file_location("infer_for_test", REPO / "scripts/robot_pc/infer.py")
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)
    cat = write_catalog(tmp_path / "catalog.json", {"dynamo-fixture-task-s0-a10nq": granted,
                                                    "e2e_cnn-fixture-task-s0": dep})
    with serve_catalog(root, cat) as srv:
        for pid, p in (("dynamo-fixture-task-s0-a10nq", pol), ("e2e_cnn-fixture-task-s0", rpol)):
            selected = welcome(srv)["policies"][pid]["info"]
            assert selected["execution_eligible"] is True
            c, _ = client_for(srv, pid, p)
            bound = infer.confirm_policy(selected, c.spec_info, execute=True, stage="binding")
            infer.confirm_policy(bound, welcome(srv)["policies"][pid]["info"], execute=True, stage="START re-check")
            c.close()


def _blocking_loads(monkeypatch, gate: threading.Event, started: threading.Event):
    import ops.policy_serving.server as server_mod

    real = server_mod.load_policy

    def slow(*a, **k):
        started.set()
        assert gate.wait(30)
        return real(*a, **k)

    monkeypatch.setattr(server_mod, "load_policy", slow)


def _hello_in_thread(srv, pid, pol, out: dict):
    def run():
        try:
            out["client"] = client_for(srv, pid, pol)[0]
        except ServingError as exc:
            out["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_entry_quarantined_while_loading_is_never_published_or_bound(staged, tmp_path, monkeypatch):
    root, fx = staged
    pol, e = fx["r1"]
    cat = write_catalog(tmp_path / "catalog.json", {"r1": e})
    gate, started, out = threading.Event(), threading.Event(), {}
    with serve_catalog(root, cat) as srv:
        _blocking_loads(monkeypatch, gate, started)
        t = _hello_in_thread(srv, "r1", pol, out)
        assert started.wait(20)
        write_catalog(cat, {"r1": {**e, "status": "quarantined", "reasons": ["decided while loading"]}})
        assert srv.reload_catalog() is not None
        gate.set()
        t.join(30)
        assert "client" not in out and "unservable" in str(out["error"]) and srv.policies == {}


def test_concurrent_loads_respect_the_resident_cap(staged, tmp_path, monkeypatch):
    """Blocks load A AFTER the model is built and warmed, at the reservation -> resident boundary (the former gap)."""
    import ops.policy_serving.server as server_mod

    root, fx = staged
    pol = fx["r1"][0]
    cat = write_catalog(tmp_path / "catalog.json", {n: fx[n][1] for n in ("r1", "r2")})
    gate, started, a, b = threading.Event(), threading.Event(), {}, {}
    real = server_mod.ServedPolicy

    def at_boundary(pid, *args, **kw):
        if pid == "r1":
            started.set()
            assert gate.wait(30)
        return real(pid, *args, **kw)

    with serve_catalog(root, cat, max_resident_policies=1) as srv:
        monkeypatch.setattr(server_mod, "ServedPolicy", at_boundary)
        ta = _hello_in_thread(srv, "r1", pol, a)
        assert started.wait(20)
        tb = _hello_in_thread(srv, "r2", pol, b)
        tb.join(30)  # the reservation of the running load counts: refused at once
        assert "overloaded" in str(b.get("error"))
        gate.set()
        ta.join(30)
        assert "client" in a and set(srv.policies) == {"r1"}
        a["client"].close()


def test_measured_footprint_is_enforced_against_the_budget(staged, tmp_path, monkeypatch):
    root, fx = staged
    pol, e = fx["r1"]
    cat = write_catalog(tmp_path / "catalog.json", {"r1": e})
    fp = policy_footprint_bytes(pol)
    with serve_catalog(root, cat, gpu_memory_budget_bytes=fp - 1) as srv:
        monkeypatch.setattr(srv, "_estimate", lambda entry: 1)  # an estimate that fits, a model that does not
        with pytest.raises(ServingError, match="overloaded"):
            client_for(srv, "r1", pol)
        assert srv.policies == {} and srv.counts["budget_refusals_measured"] == 1


def test_an_evicted_policy_is_not_kept_alive_by_the_worker(staged, tmp_path):
    import gc
    import weakref

    root, fx = staged
    pol, e = fx["r3"]
    cat = write_catalog(tmp_path / "catalog.json", {"r3": e})
    with serve_catalog(root, cat) as srv:
        c, feed = client_for(srv, "r3", pol)
        for k in range(3):
            feed.add(k)
        assert feed.step(2).ok
        ref = weakref.ref(srv.policies["r3"].policy)
        c.close()
        wait_until(lambda: not srv.policies["r3"].sessions)
        write_catalog(cat, {})
        srv.reload_catalog()
        assert "r3" not in srv.policies

        def freed():
            gc.collect()
            return ref() is None

        wait_until(freed, 10)


def test_reaper_and_withdrawal_release_models_and_labels(staged, tmp_path):
    """A session ended by the idle reaper does not keep its model alive; a withdrawn policy's live session sees
    withdrawn, never-eligible labels; restoring the entry re-enables it; cleared reasons are published as []."""
    import gc
    import weakref

    root, fx = staged
    pol, e = fx["r2"]
    restricted = {**e, "status": "restricted", "reasons": ["LN FAIL (fixture)"], "usage": "dry_run_replay_only",
                  "execution_eligible": False}
    cat = write_catalog(tmp_path / "catalog.json", {"r2": restricted})
    with serve_catalog(root, cat, session_idle_timeout_s=0.5) as srv:
        c, _ = client_for(srv, "r2", pol)
        sp = srv.policies["r2"]
        write_catalog(cat, {"r2": {**e, "status": "quarantined", "reasons": ["withdrawn (fixture)"]}})
        srv.reload_catalog()
        assert sp.retired and sp.sessions  # live session kept on the retired object
        info = srv._current_info(sp)
        assert info["status"] == "withdrawn" and info["execution_eligible"] is False and info["usage"] == "none"
        assert info["execution_decision"] is None
        write_catalog(cat, {"r2": e})  # entry restored (deployable, no reasons): re-enabled, labels refreshed
        srv.reload_catalog()
        assert not sp.retired and srv._current_info(sp)["reasons"] == [] and srv._current_info(sp)["status"] == \
            "deployable"
        c.wire.close()  # disconnect without BYE: the idle reaper ends the session
        wait_until(lambda: not sp.sessions)
        ref = weakref.ref(sp.policy)
        sp = None  # the test must not keep the model alive
        write_catalog(cat, {})
        srv.reload_catalog()

        def freed():
            gc.collect()
            return ref() is None

        wait_until(freed, 10)


def test_withdrawal_during_session_binding_is_refused(staged, tmp_path, monkeypatch):
    root, fx = staged
    pol, e = fx["r1b"]
    cat = write_catalog(tmp_path / "catalog.json", {"r1b": e})
    with serve_catalog(root, cat, preload=["r1b"]) as srv:
        sp = srv.policies["r1b"]
        real = sp.policy.reset_session

        def reload_meanwhile(**kw):
            out = real(**kw)
            write_catalog(cat, {"r1b": {**e, "status": "quarantined", "reasons": ["during bind (fixture)"]}})
            srv.reload_catalog()
            return out

        monkeypatch.setattr(sp.policy, "reset_session", reload_meanwhile)
        with pytest.raises(ServingError, match="withdrawn while this session was being bound"):
            client_for(srv, "r1b", pol)
        assert not sp.sessions and srv.sessions == {} and sp.policy.live_sessions == ()


def test_client_declares_capabilities_only_when_required_and_refuses_unknown_requirements():
    """Mixed versions (Codex review of 1847fdd, P1): a HELLO to a server without requirements (e.g. f19de4d) is
    unchanged; a requirement this client does not implement is refused before anything is sent."""
    from ops.policy_serving.client import CAPABILITIES, ServingError
    from ops.policy_serving.config import RESTRICTED_EXECUTION_CAPABILITY

    assert RESTRICTED_EXECUTION_CAPABILITY in CAPABILITIES
    sent = []

    class Wire:
        def send(self, data):
            sent.append(data)
            raise ConnectionError("stop after HELLO")  # only the HELLO header matters here

    def hello_header(requires):
        c = PolicyClient("127.0.0.1", 1, policy_id="p", choices={})
        c.welcome = {"policies": {"p": {"input_spec": {}, "info": {"requires_client": requires}}},
                     "auth_required": False}
        c.wire = Wire()
        sent.clear()
        with pytest.raises(ConnectionError):
            c._hello(None)
        return P.decode(sent[0]).header

    assert "capabilities" not in hello_header([])
    assert hello_header([RESTRICTED_EXECUTION_CAPABILITY])["capabilities"] == [RESTRICTED_EXECUTION_CAPABILITY]
    c = PolicyClient("127.0.0.1", 1, policy_id="p", choices={})
    c.welcome = {"policies": {"p": {"input_spec": {}, "info": {"requires_client": ["unknown/9"]}}},
                 "auth_required": False}
    c.wire = Wire()
    sent.clear()
    with pytest.raises(ServingError, match="does not implement"):
        c._hello(None)
    assert sent == []

    class SpecWire:  # the WELCOME offered no requirement, the bound SPEC does
        def send(self, data):
            sent.append(data)

        def recv(self, timeout):
            return P.decode(P.encode({"type": P.SPEC, "session_id": "s", "input_spec": {}, "limits": {},
                                      "info": {"requires_client": ["unknown/9"]}}))

    c = PolicyClient("127.0.0.1", 1, policy_id="p", choices={})
    c.welcome = {"policies": {"p": {"input_spec": {}, "info": {}}}, "auth_required": False}
    c.wire = SpecWire()
    with pytest.raises(ServingError, match=r"\(bound\) requires client capabilities \['unknown/9'\]"):
        c._hello(None)
