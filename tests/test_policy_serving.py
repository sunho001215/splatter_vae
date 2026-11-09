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
    StepResult,
    bound_command,
    check_action,
)
from ops.policy_serving.config import DEFAULT_PATH, load_config
from ops.policy_serving.server import PolicyServer, classify
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
                                               depth=d, valid=valid, native_correction=corr)
            meta = {"depth_source": "native_realsense", "native_correction": corr} if self.depth else {}
            self.frames[cam].append(Frame(k, self.t(k, v) + jitter_ns, out["rgb"], out.get("depth"), out.get("valid"),
                                          meta))
        if self.spec["views"] > 1:
            self.client.add_sync_set({c: k for c in self.spec["camera_ids"]}, disp)

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
    assert cfg["protocol_version"] == P.VERSION and cfg["policies"]
    assert all("checkpoint" in e and "sha256" in e for e in cfg["policies"].values())
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"server": {"prot": 1}}))
    with pytest.raises(ValueError, match="unknown server keys"):
        load_config(bad)


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
                        metadata=feed.frames[cam][-1].metadata)
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
def test_duplicate_seq_returns_cached_response_and_duplicate_frames_are_refused(policies):
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
        c._tip.clear()  # force the client to re-send frames the server already holds, under a NEW seq
        r = feed.step(2)
        assert r.code == "duplicate" and "already held" in r.detail
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
        feed.add(3)
        r = feed.step(3)
        assert r.code == "session_unknown" and c.session_id != sid
        r = feed.step(3)  # the buffered history (frames 1..3) goes to the new session
        assert r.ok and r.frames_sent == 3
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
        r = fa.step(2)  # a's session was evicted: renewed on its connection; the step is reported lost
        assert r.code == "session_unknown"
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
        c.add_frame(cam, 3, feed.t(3, 0), np.zeros((16, 16, 3), np.uint8), pose7=POSE, gripper_register=GRIP)
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
                    valid=f.valid, metadata=f.metadata)
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


def test_device_clock_and_realsense_colour_capture_with_fake_sdk():
    clock = DeviceClock(calibration_frames=3)
    stamps, ready = [], []
    for i, jitter in enumerate((5_000_000, 1_000_000, 9_000_000, 30_000_000, 0)):
        stamps.append(clock.map(100.0 + 66.7 * i, 10_000_000_000 + int(66.7e6 * i) + jitter))
        ready.append(clock.calibrated)
    assert ready == [False, False, True, True, True]  # frames before the frozen offset are flagged unusable
    assert np.allclose(np.diff(stamps[2:]), 66.7e6, atol=1)  # arrival jitter never enters the capture times
    assert abs(stamps[2] - (10_000_000_000 + int(66.7e6 * 2) + 1_000_000)) <= 1  # minimum-latency offset (1 ms)

    intr = SimpleNamespace(width=160, height=128, fx=100.0, fy=100.0, ppx=79.5, ppy=63.5, coeffs=[0] * 5)
    profile = SimpleNamespace(get_intrinsics=lambda: intr)
    profile.as_video_stream_profile = lambda: profile
    rgb = np.random.default_rng(0).integers(0, 256, (128, 160, 3), dtype=np.uint8)
    color = SimpleNamespace(profile=profile, get_data=lambda: rgb, get_timestamp=lambda: 1234.5,
                            get_frame_number=lambda: 77, get_frame_timestamp_domain=lambda: "global_time")
    calls = []

    class Pipe:
        def start(self, cfg):
            calls.append(cfg.streams)
            return None

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
    cap = RealSenseCapture("123", width=160, height=128, fps=15, rs_module=rs, out_size=32)
    out = cap.read()
    cap.stop()
    np.testing.assert_array_equal(out["rgb"], preprocess_frame(rgb, out_size=32)["rgb"])
    assert out["frame_id"] == 77 and out["capture_ns"] > 0 and calls[-1] == "stop"
    assert calls[0][0] == ("device", "123")


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
