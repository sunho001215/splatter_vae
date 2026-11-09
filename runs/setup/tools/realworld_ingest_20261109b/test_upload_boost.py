"""Tests for the temporary dataset-upload boost: the worker's fail-safe lease reading (ingest_worker.boost_level /
Worker.upload_budget), reductions vs increases during a transfer, and the controller's decisions (upload_boost.py).
Everything runs in scratch directories under /home/ws/ws/realworld-ops/ingest/scratch/tests/ (removed afterwards).

  TMPDIR=/home/ws/ws/realworld-ops/ingest/scratch/tests /home/ws/ws/splatter4d/.venv/bin/python -I -B -m pytest -q \
      -p no:cacheprovider test_upload_boost.py test_ingest.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import signal
import types
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ingest_worker as W  # noqa: E402
import upload_boost as U  # noqa: E402
from test_ingest import LocalShellRemote  # noqa: E402

SCRATCH = Path("/home/ws/ws/realworld-ops/ingest/scratch/tests")
BASELINE = {"echo_ms": {"median": 50.0, "p90": 65.0, "n": 30}, "ctrl_srtt_ms": {"median": 17.0, "p90": 20.0},
            "retrans_frac": 0.001, "kib_cap": 600}


@pytest.fixture()
def state(request, monkeypatch) -> Path:
    d = SCRATCH / ("boost-" + re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)[:70] + f"-{os.getpid()}")
    shutil.rmtree(d, ignore_errors=True)
    (d / W.BOOST_STATE).mkdir(parents=True)
    (d / "access").mkdir()
    (d / "access/download-budget.json").write_bytes(b'{"kib_per_s": 200}\n')
    monkeypatch.setattr(U, "ACCESS_DIR", d / "access")
    Clock.current = None
    real_boottime = W.boottime
    monkeypatch.setattr(W, "boottime", lambda: Clock.current.t if Clock.current else real_boottime())
    (d / "upload-budget.json").write_text(json.dumps({"kib_per_s": 600}))
    (d / "upload-boost/baseline.json").write_text(json.dumps(BASELINE))
    (d / "upload-boost/original-policy.json").write_text(json.dumps(
        {"original": {"dataset_upload_kib_per_s": 600},
         "files": {str(d / "upload-budget.json"): {"sha256": U.budget_sha(d)}}}))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def policy(d: Path, **kw) -> dict:
    now = kw.pop("now", time.time())
    doc = {"schema": W.BOOST_SCHEMA, "state": "active", "aggregate_kib_per_s": 8192, "original_kib_per_s": 600,
           "lease": {**W.controller_identity(), "expires_at_unix": now + U.LEASE_S,
                     "expires_at_boottime": W.boottime() + U.LEASE_S}}
    doc.update(kw)
    (d / W.BOOST_POLICY).write_text(json.dumps(doc))
    return doc


class SlowLocalRemote(LocalShellRemote):
    """Simulate a slower source/disk for finite-budget integration tests."""
    def rsync_command(self, local, remote, kib, append=True):
        cmd, env = super().rsync_command(local, remote, kib, append)
        return ["--bwlimit=2048" if a.startswith("--bwlimit=") and int(a.split("=")[1]) > 2048 else a
                for a in cmd], env


# --------------------------------------------------------------------------------------------- worker: lease reading


def test_boost_level_is_fail_safe(state):
    now = time.time()
    assert W.boost_level(state, 600, now) == (None, "no boost policy")
    policy(state, now=now)
    assert W.boost_level(state, 600, now) == (8192, "boost lease valid")
    bad = [dict(schema="rw-upload-boost/0"), dict(state="standby"), dict(state="restored"),
           dict(aggregate_kib_per_s=65536), dict(aggregate_kib_per_s=1000), dict(aggregate_kib_per_s=True),
           dict(aggregate_kib_per_s="8192"), dict(original_kib_per_s=300), dict(lease=None),
           dict(lease={"expires_at_unix": now - 1}), dict(lease={"expires_at_unix": now + 3600}),
           dict(lease={"expires_at_unix": True}), dict(lease={"expires_at_unix": "soon"}),
           dict(lease={"expires_at_unix": float("nan")}), dict(lease={"expires_at_unix": float("inf")}),
           dict(lease=[1]), dict(lease="x"), dict(aggregate_kib_per_s=[8192])]
    for kw in bad:
        policy(state, now=now, **kw)
        level, why = W.boost_level(state, 600, now)
        assert level is None, (kw, why)
    policy(state, now=now)
    assert W.boost_level(state, 0, now)[0] is None        # a coordinator pause always wins
    assert W.boost_level(state, 300, now)[0] is None      # base changed by the coordinator: boost void
    (state / W.BOOST_POLICY).write_text("{not json")
    assert W.boost_level(state, 600, now)[0] is None
    assert W.boost_level(state, 600, now + 151)[0] is None  # lapsed lease = original


def test_worker_budget_uses_lease_and_logs_changes(state):
    w = W.Worker(state, raw_root=state / "raw", tasks={"drawer-open": {"drawer-opening"}}, remote=None)
    assert w.upload_budget() == (600, "base")
    policy(state, aggregate_kib_per_s=16384)
    assert w.upload_budget() == (16384, "boost")
    assert w.upload_budget() == (16384, "boost")
    (state / "upload-budget.json").write_text(json.dumps({"kib_per_s": 0}))
    assert w.upload_budget() == (0, "base")
    (state / "upload-budget.json").write_text(json.dumps({"kib_per_s": 600}))
    policy(state, lease={"expires_at_unix": time.time() - 1})
    assert w.upload_budget() == (600, "base")
    ev = [json.loads(x) for x in (state / "events.jsonl").read_text().splitlines()]
    boost = [(e["applied"], e["kib_per_s"], e["reason"]) for e in ev if e["event"] == "upload_boost"]
    assert boost == [(False, 600, "no boost policy"), (True, 16384, "boost lease valid"),
                     (False, 0, "boost granted for base 600, base is 0"), (False, 600, "boost lease expired")]


def test_reduction_is_enforced_during_a_long_scan_and_mirror_is_never_boosted(state, monkeypatch):
    src = state / "episode_000001.h5"
    src.write_bytes(os.urandom(12 * 1024 * 1024))
    remote = SlowLocalRemote(state / "remote")
    part = state / "remote/part/episode_000001.h5.part"
    part.parent.mkdir(parents=True)
    w = W.Worker(state, raw_root=state / "raw", tasks={"drawer-open": {"drawer-opening"}}, remote=remote)
    w.state.doc["counts"] = {}
    policy(state, aggregate_kib_per_s=8192)
    entry = {"key": "drawer-open/episode_000001.h5", "size": src.stat().st_size, "transfer": {}}
    seen = {}

    def long_scan(skip_key=None):  # 8 s of "validations"; the lease lapses after 1 s
        t0 = time.time()
        while time.time() - t0 < 8:
            if time.time() - t0 > 1 and "expired" not in seen:
                policy(state, aggregate_kib_per_s=8192, lease={"expires_at_unix": time.time() - 1})
                seen["expired"] = time.time()
            time.sleep(0.5)
            if w._inflight and w._inflight["reduced_to"] is not None and "reduced" not in seen:
                seen["reduced"] = time.time()
        w.last_scan = time.time()
    monkeypatch.setattr(W, "SCAN_DURING_TRANSFER_S", 0.0)
    monkeypatch.setattr(w, "scan", long_scan)
    monkeypatch.setattr(w, "write_status", lambda *a, **k: None)
    rc, err = w._run_rsync(src, str(part), 8192, entry)
    assert rc == 0, err and part.read_bytes() == src.read_bytes()
    assert seen["reduced"] - seen["expired"] < 3.0          # watchdog: enforced during the scan, not after it
    assert [next(a for a in c if a.startswith("--bwlimit")) for c in remote.rsync_calls] == \
        ["--bwlimit=8192", "--bwlimit=600"]
    w._mirror_dirty = True
    policy(state, aggregate_kib_per_s=8192)
    w.mirror({})
    assert "--bwlimit=600" in remote.rsync_calls[-1]


def test_increase_waits_for_next_transfer_reduction_restarts_at_once(state):
    """Real rsync (local): 12 MiB at 2 MiB/s. A raised lease does not interrupt it; an expired lease restarts it at the
    original 600 KiB/s from the partial bytes, and the copy completes intact."""
    src = state / "episode_000001.h5"
    src.write_bytes(os.urandom(12 * 1024 * 1024))
    remote = SlowLocalRemote(state / "remote")
    part = state / "remote/part/episode_000001.h5.part"
    part.parent.mkdir(parents=True)
    w = W.Worker(state, raw_root=state / "raw", tasks={"drawer-open": {"drawer-opening"}}, remote=remote)
    w.state.doc["counts"] = {}
    policy(state, aggregate_kib_per_s=8192)
    entry = {"key": "drawer-open/episode_000001.h5", "size": src.stat().st_size, "transfer": {}}

    def later():
        time.sleep(1.0)
        policy(state, aggregate_kib_per_s=16384)                      # increase: must not restart rsync
        time.sleep(2.5)
        policy(state, aggregate_kib_per_s=16384, lease={"expires_at_unix": time.time() - 1})  # lapsed: reduce now
    updater = threading.Thread(target=later)
    updater.start()
    rc, err = w._run_rsync(src, str(part), 8192, entry)
    updater.join(timeout=10)
    assert not updater.is_alive()
    assert rc == 0, err
    assert part.read_bytes() == src.read_bytes()
    ev = [json.loads(x) for x in (state / "events.jsonl").read_text().splitlines()]
    changes = [(e["old"], e["new"]) for e in ev if e["event"] == "budget_changed_mid_transfer"]
    assert changes == [(8192, 600)]
    assert [next(a for a in c if a.startswith("--bwlimit")) for c in remote.rsync_calls] == \
        ["--bwlimit=8192", "--bwlimit=600"]
    assert entry["transfer"]["kib_per_s"] == 600


# ---------------------------------------------------------------------------------------------------- controller


class FakeSampler:
    def __init__(self):
        self.next = {}
        self.events = []

    def sample(self, measure=True):
        return dict(self.next)

    def new_events(self):
        ev, self.events = self.events, []
        return ev


class Clock:
    current = None

    def __init__(self, t=1_800_000_000.0):
        self.t = t
        Clock.current = self

    def __call__(self):
        return self.t


def sample(clock, cap=600, rate=None, echo=50.0, srtt=17.0, **kw):
    s = {"t": clock.t, "alive": True, "version": W.WORKER_VERSION, "base_kib": 600, "remaining": 50,
         "all_done": False, "rsync_pid": 4242 if cap else None, "kib_cap": cap, "key": "drawer-open/e.h5",
         "echo_ms": echo, "ctrl_srtt_ms": srtt, "rate_kib": rate if rate is not None else
         (50000.0 if cap == W.UNCAPPED else 0.97 * cap if cap else None), "rate_cap": cap, "retrans": 1, "segs": 1000}
    if echo is not None:  # the real sampler sets "echo" only when the ControlMaster exists
        s["echo"] = "ok"
    s.update(kw)
    return s


def iso(t):
    return W.now_iso(t)


def verified(clock, cap, n=1, size=100 * 2 ** 20):
    out = []
    for i in range(n):
        key = f"drawer-open/episode_{int(clock.t) % 10 ** 6:06d}{i}.h5"
        out += [{"t": iso(clock.t - 60), "event": "transfer_start", "key": key, "kib_per_s": cap},
                {"t": iso(clock.t - 1), "event": "transfer_verified", "key": key, "kib_per_s": cap,
                 "sent_bytes": size, "rsync_s": 50.0, "duration_s": 53.0}]
    return out


def run(ctl, clock, seconds, events_every=None, **kw):
    """Advance the controller by `seconds` in SAMPLE_S ticks with samples built from kw (cap follows the level)."""
    end = clock.t + seconds
    while clock.t < end:
        clock.t += U.SAMPLE_S
        cap = kw.get("cap", ctl.level or 600)
        ctl.sampler.next = sample(clock, **{**kw, "cap": cap})
        if events_every and int(clock.t) % events_every < U.SAMPLE_S:
            ctl.sampler.events += verified(clock, cap)
        ctl.tick()


def run_until_change(ctl, clock, max_ticks=10, **kw):
    start = ctl.level
    for _ in range(max_ticks):
        run(ctl, clock, U.SAMPLE_S, **kw)
        if ctl.level != start:
            return


def enter(ctl, clock, **kw):
    """Tick until the controller boosts (entry needs ENTRY_QUIET_S without hard failure and a measured healthy window),
    then one more tick (as the earlier 'first tick enters' tests assumed)."""
    for _ in range(int(U.ENTRY_QUIET_S / U.SAMPLE_S) + U.ENTRY_SAMPLES + 4):
        run(ctl, clock, U.SAMPLE_S, **kw)
        if ctl.level:
            run(ctl, clock, U.SAMPLE_S, **kw)
            return
    raise AssertionError(f"no entry: {ctl.reason}")


def lease_level(state, clock):
    return W.boost_level(state, 600, clock.t)[0]


def test_controller_steps_up_while_healthy_and_holds_at_ceiling(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    assert ctl.level == 8192 and lease_level(state, clock) == 8192
    pol = json.loads((state / W.BOOST_POLICY).read_text())
    assert pol["state"] == "active" and 0 < pol["lease"]["expires_at_unix"] - clock.t <= W.BOOST_MAX_LEASE_S
    run(ctl, clock, U.DWELL_S - 60, events_every=180)
    assert ctl.level == 8192
    run(ctl, clock, 120, events_every=180)
    assert ctl.level == 16384
    run(ctl, clock, U.DWELL_S + 60, events_every=180)
    assert ctl.level == 32768
    run(ctl, clock, U.DWELL_S + 60, events_every=180)
    assert ctl.level == W.UNCAPPED and lease_level(state, clock) == W.UNCAPPED
    run(ctl, clock, 3 * U.DWELL_S, events_every=180)
    assert ctl.level == W.UNCAPPED
    assert set(ctl.healthy) == {8192, 16384, 32768}
    ctl.write_levels({"t": clock.t})  # mixed finite/UNCAPPED reporting is serializable
    assert lease_level(state, Clock(clock.t + U.LEASE_S + 1)) is None   # controller gone -> lease lapses


def test_not_cap_bound_holds_and_reports_bottleneck(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    run(ctl, clock, 2 * U.DWELL_S, events_every=180, rate=1200.0)   # 2 MiB/s cap, only 1.2 MiB/s delivered
    assert ctl.level == 8192
    notes = [json.loads(x) for x in (state / W.BOOST_STATE / "history.jsonl").read_text().splitlines()]
    assert any(n["event"] == "note" and "not cap-bound" in n["detail"] for n in notes)


def test_soft_breach_drops_to_last_healthy_level(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    run(ctl, clock, U.DWELL_S + 60, events_every=180)
    assert ctl.level == 16384
    run_until_change(ctl, clock, echo=400.0)             # control latency 8x baseline at the 4 MiB/s cap
    assert ctl.level == 8192 and lease_level(state, clock) == 8192
    assert ctl.blocked(16384, clock.t) and not ctl.blocked(8192, clock.t)
    run(ctl, clock, 2 * U.DWELL_S, events_every=180)      # 16384 in cooldown: hold at 8192
    assert ctl.level == 8192
    run_until_change(ctl, clock, srtt=160.0)              # path RTT breach at 8192 -> original
    assert ctl.level == 0 and lease_level(state, clock) is None


def test_hard_failure_restores_original_with_cooldown(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    assert ctl.level == 8192
    ctl.sampler.events.append({"t": iso(clock.t), "event": "transfer_error", "key": "k",
                               "error": "remote part sha256 aaa != local bbb (part removed, full retransfer)"})
    run(ctl, clock, U.SAMPLE_S)
    assert ctl.level == 0 and lease_level(state, clock) is None
    assert "sha256 mismatch" in json.loads((state / W.BOOST_POLICY).read_text())["reason"]
    run(ctl, clock, U.COOLDOWN_S - 120)
    assert ctl.level == 0                                   # cooldown of the failed level
    run(ctl, clock, 240)
    assert ctl.level == 8192


def test_stall_and_echo_failures_are_hard(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    run(ctl, clock, (U.STALL_SAMPLES - 1) * U.SAMPLE_S, rate=0.0)
    assert ctl.level == 8192                                # a resume's prefix verification is not a stall
    run(ctl, clock, U.SAMPLE_S, rate=0.0)
    assert ctl.level == 0 and "stalled" in ctl.reason
    ctl.failures.clear()
    ctl.last_hard_at = 0
    enter(ctl, clock)
    assert ctl.level == 8192
    run(ctl, clock, 2 * U.SAMPLE_S, echo=None)
    assert ctl.level == 8192                                # no master: no sample, not a failure
    for _ in range(1):
        clock.t += U.SAMPLE_S
        ctl.sampler.next = sample(clock, cap=8192, echo=None)
        ctl.sampler.next["echo"] = "exit 255"
        ctl.tick()
    assert ctl.level == 0 and "echo failed" in ctl.reason


def test_isolated_latency_spikes_do_not_trigger(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    for i in range(40):                                     # the baseline's pattern: 1 in 4 samples at 130-250 ms
        run(ctl, clock, U.SAMPLE_S, echo=250.0 if i % 4 == 0 else 50.0, srtt=52.0 if i % 3 == 0 else 20.0)
    assert ctl.level >= 8192 and not ctl.failures
    run_until_change(ctl, clock, echo=251.0)                # sustained: most samples high -> soft
    assert ctl.failures and "control echoes" in ctl.reason


def test_missing_measurements_restore_and_unhealthy_entry_is_refused(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    run(ctl, clock, U.ENTRY_QUIET_S + 120, echo=400.0)      # unhealthy at the original budget: never enter
    assert ctl.level == 0 and not ctl.failures
    enter(ctl, clock)
    run_until_change(ctl, clock, max_ticks=int(U.MISSING_S / U.SAMPLE_S) + 4, echo=None)            # boosted but no control-channel measurement
    assert ctl.level == 0 and "missing evidence" in ctl.reason
    enter(ctl, clock)
    run_until_change(ctl, clock, max_ticks=int(U.MISSING_S / U.SAMPLE_S) + 4, rate=None, **{"rate_kib": None})  # uploading but no data-flow measurement
    assert ctl.level == 0 and "data-flow" in ctl.reason


def test_remote_conflict_is_hard(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    ctl.sampler.events.append({"t": iso(clock.t), "event": "remote_conflict", "key": "k", "outcome": "PUBLISHED",
                               "remote_sha": "aa"})
    run(ctl, clock, U.SAMPLE_S)
    assert ctl.level == 0 and "remote_conflict" in ctl.reason


def test_two_failures_cap_the_boost_below(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    ctl.fail(32768, clock.t - 60)
    ctl.fail(32768, clock.t - 30)
    assert ctl.ceiling(clock.t) == 16384 and ctl.blocked(32768, clock.t) and not ctl.blocked(16384, clock.t)


def test_standby_when_idle_worker_gone_or_base_changed_then_reentry(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    run(ctl, clock, U.DWELL_S + 60, events_every=180)
    assert ctl.level == 16384
    run(ctl, clock, U.IDLE_GRACE_S - 60, cap=None, remaining=7)
    assert ctl.level == 16384                                 # gaps between transfers keep the level
    run(ctl, clock, 120, cap=None, remaining=7)
    assert ctl.level == 0 and "no upload for" in ctl.reason  # pending but nothing uploads (e.g. back-off)
    enter(ctl, clock, remaining=7)
    assert ctl.level > 0                                     # the worker uploads again
    run(ctl, clock, U.SAMPLE_S, cap=None, remaining=0)
    assert ctl.level > 0  # a batch ends after the idle grace
    run(ctl, clock, U.IDLE_GRACE_S + 60, cap=None, remaining=0)
    assert ctl.level == 0 and json.loads((state / W.BOOST_POLICY).read_text())["state"] == "idle"
    enter(ctl, clock, remaining=3)                            # new arrivals: re-enter at the recently healthy level
    assert ctl.level == 8192 or ctl.level == 16384
    for bad, text in ((dict(alive=False), "not running"), (dict(version="rw-ingest-worker/2"), "cannot read"),
                      (dict(base_kib=300), "coordinator")):
        run(ctl, clock, 60, **bad)
        assert ctl.level == 0 and text in ctl.reason, (bad, ctl.reason)
        enter(ctl, clock)
        assert ctl.level > 0


def test_force_restore_and_stop_win_over_a_running_controller(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    assert lease_level(state, clock) == 8192
    U.cmd_restore(state, "test")
    assert lease_level(state, clock) is None
    clock.t += U.SAMPLE_S
    ctl.sampler.next = sample(clock, cap=8192)
    assert ctl.tick() == "stop"                               # the loop may not re-activate after a restore
    assert json.loads((state / W.BOOST_POLICY).read_text())["state"] == "restored"


def test_retire_when_all_tasks_complete(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    clock.t += U.SAMPLE_S
    ctl.sampler.next = sample(clock, all_done=True, remaining=0, cap=None,
                              counts={t: {"transferred": W.TARGET_PER_TASK} for t in W.TASKS})
    assert ctl.tick() == "retire"
    U.retire(state, ctl)
    pol = json.loads((state / W.BOOST_POLICY).read_text())
    ret = json.loads((state / W.BOOST_STATE / "retired.json").read_text())
    assert pol["state"] == "retired" and lease_level(state, clock) is None and ret["original_restored"] is True
    assert U.main(["run", "--state-dir", str(state)]) == 0    # a retired boost never starts again


def test_run_span_aggregate():
    xs = [{"cap": 8192, "final_cap": 8192, "t_start": 0, "t_end": 50, "bytes": 1},
          {"cap": 8192, "final_cap": 8192, "t_start": 52, "t_end": 100, "bytes": 1},
          {"cap": 16384, "final_cap": 16384, "t_start": 101, "t_end": 120, "bytes": 1},
          {"cap": 8192, "final_cap": 8192, "t_start": 500, "t_end": 550, "bytes": 1}]
    assert U.run_span(xs, 8192) == (150.0, [2.0])
    assert U.run_span(xs, 16384) == (19.0, [])


def test_quiet_period_is_taken_from_the_record(state):
    now = time.time()
    with open(state / "events.jsonl", "w") as f:
        f.write(json.dumps({"t": iso(now - 1200), "event": "transfer_error", "key": "old"}) + "\n")
    assert U.last_hard_failure(state, now) == 0.0                       # older than ENTRY_QUIET_S: quiet shown
    with open(state / "events.jsonl", "a") as f:
        f.write(json.dumps({"t": iso(now - 120), "event": "remote_unreachable"}) + "\n")
    assert abs(U.last_hard_failure(state, now) - (now - 120)) < 2          # recent: wait out the quiet period
    with open(state / W.BOOST_STATE / "history.jsonl", "w") as f:
        f.write(json.dumps({"t": iso(now - 30), "event": "level", "reason": "soft: x"}) + "\n")
    assert abs(U.last_hard_failure(state, now) - (now - 30)) < 2


def test_hard_failure_wins_over_missing_evidence_and_evidence_survives_step_up(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    run(ctl, clock, U.MISSING_S - 60, echo=None)
    ctl.sampler.events.append({"t": iso(clock.t), "event": "transfer_error", "key": "k", "error": "rsync exit 12"})
    run(ctl, clock, 120, echo=None)
    assert ctl.level == 0 and ctl.reason.startswith("hard:") and ctl.blocked(8192, clock.t)
    ctl2 = U.Controller(state, sampler=FakeSampler(), clock=clock)
    ctl2.failures.clear()
    ctl2.last_hard_at = 0
    enter(ctl2, clock)
    run(ctl2, clock, U.DWELL_S + 60, events_every=180)
    assert ctl2.level == 16384
    ctl2.last_echo_at = clock.t - U.MISSING_S - 1                      # stale evidence is not reset by a step up
    run(ctl2, clock, U.SAMPLE_S, echo=None)
    assert ctl2.level == 0 and "missing evidence" in ctl2.reason


def test_sampler_keeps_completed_short_uploads_per_cap(state, monkeypatch):
    """Transfers shorter than a sample (8 MiB/s: ~32 s) are measured by the 5-s polls; the marker disappearing after
    an upload must not erase what was accumulated, and a new cap is accounted separately."""
    sp = U.Sampler.__new__(U.Sampler)
    sp.dir, sp.flow_prev, sp.flow_last, sp.acc = state, None, None, {}
    clock = Clock(1000.0)
    flow = {"inode": 1, "bytes": 0}
    monkeypatch.setattr(U.time, "time", clock)
    monkeypatch.setattr(U, "children", lambda pid: [pid + 1] if pid in (10, 20) else [])
    monkeypatch.setattr(U, "proc_comm", lambda pid: "ssh")
    monkeypatch.setattr(U, "socket_inodes", lambda pid: {flow["inode"]})
    monkeypatch.setattr(U, "cpu_ticks", lambda pid: 0)
    info = lambda: {flow["inode"]: {"bytes_acked": flow["bytes"], "total_retrans": 0, "data_segs_out": flow["bytes"] // 1448,  # noqa: E731
                                    "srtt_ms": 20.0, "cwnd": 100, "rwnd_limited_us": 0, "sndbuf_limited_us": 0}}
    monkeypatch.setattr(U, "tcp_flows", info)
    marker = state / "rsync-current.json"
    marker.write_text(json.dumps({"pid": 10, "kib_per_s": 8192}))
    for _ in range(5):                      # 25 s at 8 MiB/s
        sp.poll_flows()
        clock.t += 5
        flow["bytes"] += 8192 * 1024 * 5
    sp.poll_flows()
    marker.unlink()                         # upload done: verification/publish, no marker
    sp.poll_flows()
    s = sp.consume_rates({})
    assert s["rate_cap"] == 8192 and abs(s["rate_kib"] - 8192) < 1 and s["rate_secs"] == 25.0
    marker.write_text(json.dumps({"pid": 20, "kib_per_s": 16384}))
    flow["inode"] = 2
    for _ in range(3):
        sp.poll_flows()
        clock.t += 5
        flow["bytes"] += 16384 * 1024 * 5
    marker.write_text(json.dumps({"pid": 10, "kib_per_s": 8192}))
    flow["inode"] = 3
    sp.poll_flows()
    s = sp.consume_rates({})
    assert s["rate_cap"] == 16384 and abs(s["rate_kib"] - 16384) < 1 and sp.acc == {}


def test_late_watchdog_latch_restarts_instead_of_failing(state, monkeypatch):
    src = state / "episode_000001.h5"
    src.write_bytes(os.urandom(2 * 1024 * 1024))
    remote = SlowLocalRemote(state / "remote")
    part = state / "remote/part/episode_000001.h5.part"
    part.parent.mkdir(parents=True)
    w = W.Worker(state, raw_root=state / "raw", tasks={"drawer-open": {"drawer-opening"}}, remote=remote)
    w.state.doc["counts"] = {}
    policy(state, aggregate_kib_per_s=8192)
    entry = {"key": "drawer-open/episode_000001.h5", "size": src.stat().st_size, "transfer": {}}
    calls = []

    def racing_watchdog(cur, stop):  # latches only after rsync ended by itself (the race window)
        calls.append(cur["kib"])
        if len(calls) == 1:
            cur["proc"].wait()
            cur["reduced_to"] = 600
    monkeypatch.setattr(w, "_budget_watchdog", racing_watchdog)
    rc, err = w._run_rsync(src, str(part), 8192, entry)
    assert rc == 0, err
    assert [next(a for a in c if a.startswith("--bwlimit")) for c in remote.rsync_calls] == \
        ["--bwlimit=8192", "--bwlimit=600"]
    assert part.read_bytes() == src.read_bytes() and entry["transfer"]["kib_per_s"] == 600


@pytest.mark.parametrize("cap", [8192, 16384, 32768, W.UNCAPPED])
def test_ladder_rsync_argv(state, cap):
    remote = LocalShellRemote(state / "remote")
    cmd, _ = remote.rsync_command(state / "episode_000001.h5", str(state / "part"), cap)
    assert "--partial" in cmd and "--append-verify" in cmd
    caps = [arg for arg in cmd if arg.startswith("--bwlimit")]
    assert caps == ([] if cap == W.UNCAPPED else [f"--bwlimit={cap}"])
    policy(state, aggregate_kib_per_s=cap)
    assert W.boost_level(state, 600)[0] == cap


@pytest.mark.parametrize("change,breach", [
    ({"echo_ms": 250.0, "ctrl_srtt_ms": 150.0, "retrans": 20}, False),
    ({"echo_ms": 250.1}, True),
    ({"ctrl_srtt_ms": 150.1}, True),
    ({"retrans": 21}, True),
    ({"echo_ms": 1000.1}, True),
    ({"echo": "exit 255", "echo_ms": None}, True),
])
def test_absolute_limits(state, change, breach):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    ctl.at_level = [{**sample(clock), **change} for _ in range(U.WINDOW)]
    hard, soft = ctl.assess(ctl.at_level[-1], [])
    assert bool(hard or soft) is breach
    assert U.limits(BASELINE) == U.limits({}) == {
        "echo_ms": 250.0, "ctrl_srtt_ms": 150.0, "retrans_frac": 0.02, "slow_echo_ms": 1000.0}


def test_any_slow_echo_even_when_p90_is_low(state, monkeypatch):
    clock = Clock()
    ctl = U.Controller(state, clock=clock)
    monkeypatch.setattr(U, "WINDOW", 20)
    ctl.at_level = [sample(clock) for _ in range(19)] + [sample(clock, echo=1001)]
    assert U.pct([s["echo_ms"] for s in ctl.at_level], .9) == 50
    assert any("> 1000" in why for why in ctl.assess(ctl.at_level[-1], [])[1])


def test_failure_memory_expires_including_after_restart(state):
    clock = Clock()
    ctl = U.Controller(state, clock=clock)
    for cap in (8192, W.UNCAPPED):
        ctl.fail(cap, clock.t - 60)
        ctl.fail(cap, clock.t - 30)
    assert ctl.ceiling(clock.t) == 0
    ctl.write_policy("idle")  # serialize mixed numeric and string memory keys
    clock.t += U.FAILURE_MEMORY_S + 1
    ctl2 = U.Controller(state, clock=clock)
    assert not ctl2.blocked(8192, clock.t) and not ctl2.blocked(W.UNCAPPED, clock.t)
    assert ctl2.ceiling(clock.t) == W.UNCAPPED and not ctl2.failures


def test_activation_exact_settings_idle_restore_and_later_batch(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    before = U.previous_settings(state)
    enter(ctl, clock)
    first = ctl.activation["activation_id"]
    settings = ctl.activation["previous_settings"]
    assert settings == before
    for rec in settings.values():
        data = base64.b64decode(rec["bytes_base64"])
        assert hashlib.sha256(data).hexdigest() == rec["sha256"]
    run(ctl, clock, U.IDLE_GRACE_S + U.SAMPLE_S, cap=None, remaining=0)
    pol = json.loads((state / W.BOOST_POLICY).read_text())
    assert pol["state"] == "idle" and pol["lease"] is None and ctl.activation is None
    ended = U.activation_records(ctl.bdir)[first]
    assert ended["start_unix"] < ended["end_unix"] and ended["end_reason"].startswith("no eligible")
    assert ended["levels_reached"] == [0, 8192]
    assert U.previous_settings(state) == before
    enter(ctl, clock)
    assert ctl.activation["activation_id"] != first


def test_six_hour_expiry_rest_and_reactivation_survive_restart(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    first = ctl.activation["activation_id"]
    clock.t = ctl.activation["start_unix"] + U.MAX_ACTIVATION_S
    ctl.sampler.next = sample(clock, cap=8192)
    assert ctl.tick() == "expired"
    assert ctl.level == 0 and W.boost_level(state, 600, clock.t)[0] is None
    assert U.activation_records(ctl.bdir)[first]["end_reason"] == "activation expired after 6 h"
    ctl2 = U.Controller(state, sampler=FakeSampler(), clock=clock)
    run(ctl2, clock, U.EXPIRY_REST_S - U.SAMPLE_S)
    assert ctl2.state == "expired" and ctl2.activation is None
    run(ctl2, clock, U.SAMPLE_S)
    assert ctl2.activation and ctl2.activation["activation_id"] != first
    enter(ctl2, clock)
    assert ctl2.level > 0


def test_stale_active_policy_restored_before_measurement(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    activation = ctl.activation["activation_id"]
    # An unclean exit leaves its active policy + open journal, with no end record.
    ctl2 = U.Controller(state, sampler=FakeSampler(), clock=clock)
    pol = json.loads((state / W.BOOST_POLICY).read_text())
    assert pol["state"] == "restored" and pol["lease"] is None
    assert "stale active" in pol["reason"]
    assert W.boost_level(state, 600, clock.t)[0] is None
    assert U.activation_records(ctl2.bdir)[activation]["end_reason"].startswith("startup recovery")


@pytest.mark.parametrize("field,value", [
    ("boot_id", "another-boot"), ("host", "another-host"),
    ("pid", 2 ** 30), ("start_time_ticks", -1),
    ("expires_at_boottime", -1),
])
def test_lease_identity_and_boot_mismatch_fall_back(state, field, value):
    doc = policy(state, aggregate_kib_per_s=W.UNCAPPED)
    doc["lease"][field] = value
    (state / W.BOOST_POLICY).write_text(json.dumps(doc))
    assert W.boost_level(state, 600)[0] is None


def test_wall_clock_rollback_does_not_extend_lease(state):
    clock = Clock(time.time())
    policy(state, now=clock.t, aggregate_kib_per_s=W.UNCAPPED)
    wall_now = clock.t
    clock.t += U.LEASE_S + 1  # boot time advances while wall time moves backwards
    assert W.boost_level(state, 600, wall_now - 100)[0] is None


def test_uncapped_watchdog_lease_lapse_restarts_partial_and_sets_pdeathsig(state, monkeypatch):
    clock = Clock(time.time())
    src = state / "episode_000001.h5"
    src.write_bytes(b"test")
    part = state / "remote/part"
    remote = LocalShellRemote(state / "remote")
    w = W.Worker(state, raw_root=state / "raw", remote=remote)
    w.last_scan = clock.t
    lease_start = clock.t
    policy(state, now=clock.t, aggregate_kib_per_s=W.UNCAPPED)
    spawned, cuts = [], []

    class Proc:
        pid = os.getpid()
        def __init__(self, cmd, **kw):
            self.returncode = None if not spawned else 0
            self.cmd, self.kw = cmd, kw
            spawned.append(self)
        def poll(self):
            return self.returncode
        def terminate(self):
            if self.returncode is None:
                cuts.append(clock.t)
                self.returncode = -signal.SIGTERM
        def wait(self, timeout=None):
            return self.returncode

    class Event:
        stopped = False
        def wait(self, seconds):
            clock.t += seconds
            return self.stopped
        def set(self):
            self.stopped = True

    class Thread:
        def __init__(self, target, args, **kw):
            self.target, self.args = target, args
        def start(self):
            self.target(*self.args)
        def is_alive(self):
            return False

    monkeypatch.setattr(W.subprocess, "Popen", Proc)
    monkeypatch.setattr(W.threading, "Event", Event)
    monkeypatch.setattr(W.threading, "Thread", Thread)
    monkeypatch.setattr(W.time, "time", clock)
    monkeypatch.setattr(W, "_kill_stray_uploads", lambda path: None)
    entry = {"key": "covering/episode_000001.h5", "size": 4, "transfer": {}}
    rc, _ = w._run_rsync(src, str(part), W.UNCAPPED, entry)
    assert rc == 0 and len(spawned) == 2
    assert not any(a.startswith("--bwlimit") for a in spawned[0].cmd)
    assert "--bwlimit=600" in spawned[1].cmd
    assert all("--partial" in p.cmd and "--append-verify" in p.cmd for p in spawned)
    assert 0 <= cuts[0] - (lease_start + U.LEASE_S) <= 2.0 < U.LEASE_S
    assert entry["transfer"]["kib_per_s"] == 600
    # Inspect the exact preexec_fn passed to every rsync Popen without forking or changing this process's signals.
    calls = []
    monkeypatch.setattr(W, "_LIBC", types.SimpleNamespace(prctl=lambda *args: calls.append(args) or 0))
    monkeypatch.setattr(W.os, "getppid", os.getpid)
    monkeypatch.setattr(W.os, "setpriority", lambda *args: None)
    monkeypatch.setattr(W.signal, "signal", lambda *args: None)
    for proc in spawned:
        proc.kw["preexec_fn"]()
    assert calls == [(1, signal.SIGKILL), (1, signal.SIGKILL)]


def test_retirement_requires_every_named_task_and_ignores_v1_retirement(state, monkeypatch):
    clock = Clock()
    v1 = state / "upload-boost/retired.json"
    v1.write_text('{"v1": "retired"}\n')
    old_bytes = v1.read_bytes()
    sp = U.Sampler.__new__(U.Sampler)
    sp.dir = state
    for counts, expected in [
        ({t: {"transferred": 100, "target": 100} for t in ("drawer-open", "mug-hanging")}, False),
        ({t: {"transferred": 100, "target": 1} for t in W.TASKS}, True),
        ({t: {"transferred": 99 if t == "covering" else 100, "target": 1} for t in W.TASKS}, False),
        ({t: {"transferred": 100} for t in (*list(W.TASKS)[:3], "unrelated")}, False),
    ]:
        (state / "status.json").write_text(json.dumps({"counts": counts}))
        assert sp.worker()["all_done"] is expected
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    assert ctl.level == 8192
    # CLI run must also pass the v2 retirement gate without any real sampler/network/process start.
    monkeypatch.setattr(U.os, "setpriority", lambda *args: None)
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.setattr(U, "Sampler", lambda state_dir: FakeSampler())
    monkeypatch.setattr(U.Controller, "run", lambda self: 17)
    assert U.main(["run", "--state-dir", str(state)]) == 17
    assert v1.read_bytes() == old_bytes


def test_restore_closes_activation_once_and_stop_cannot_reopen_it(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    activation = ctl.activation["activation_id"]
    U.cmd_restore(state, "test end")
    run(ctl, clock, U.SAMPLE_S)
    ctl.end_activation("controller exit", clock.t, "restored")
    records = [json.loads(line) for line in (ctl.bdir / "activations.jsonl").read_text().splitlines()]
    ends = [r for r in records if r["activation_id"] == activation and r["event"] == "end"]
    assert len(ends) == 1 and ends[0]["end_reason"] == "force-restore: test end"
    assert U.activation_records(ctl.bdir)[activation]["end_reason"] == "force-restore: test end"


def test_torn_activation_journal_recovers_and_expired_crash_requires_rest(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    first = ctl.activation["activation_id"]
    with open(ctl.bdir / "activations.jsonl", "a") as f:
        f.write('{"activation_id": "torn')
    clock.t = ctl.activation["start_unix"] + U.MAX_ACTIVATION_S + 1
    ctl2 = U.Controller(state, sampler=FakeSampler(), clock=clock)
    assert U.activation_records(ctl2.bdir)[first]["end_reason"].startswith("startup recovery")
    ctl2.sampler.next = sample(clock)
    assert ctl2.tick() == "expired"
    # Repeated restarts do not reset the rest deadline indefinitely.
    deadline = ctl2.expired_until
    clock.t += U.SAMPLE_S
    ctl3 = U.Controller(state, sampler=FakeSampler(), clock=clock)
    assert ctl3.expired_until == deadline
    run(ctl3, clock, U.EXPIRY_REST_S)
    assert ctl3.activation is not None


def test_missing_srtt_restores_even_when_echo_and_data_remain(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    run_until_change(ctl, clock, max_ticks=int(U.MISSING_S / U.SAMPLE_S) + 4, srtt=None)
    assert ctl.level == 0 and "ControlMaster srtt" in ctl.reason


def test_six_hour_boundary_wins_over_hard_failure(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    clock.t = ctl.activation["start_unix"] + U.MAX_ACTIVATION_S
    ctl.sampler.next = sample(clock)
    ctl.sampler.events = [{"event": "remote_unreachable"}]
    assert ctl.tick() == "expired" and ctl.level == 0
    assert ctl.expired_until >= clock.t + U.EXPIRY_REST_S


@pytest.mark.parametrize("hard", [False, True])
def test_uncapped_deterioration_restores_finite_budget(state, hard):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    for _ in range(3):
        run(ctl, clock, U.DWELL_S + 60, events_every=180)
    assert ctl.level == W.UNCAPPED
    if hard:
        ctl.sampler.events = [{"event": "remote_unreachable"}]
        run(ctl, clock, U.SAMPLE_S)
        assert ctl.level == 0 and ctl.activation is None and ctl.state == "idle"
    else:
        run_until_change(ctl, clock, echo=251.0)
        assert ctl.level == 32768
        assert ctl.blocked(W.UNCAPPED, clock.t)
        # A stale higher blamed cap in the window must never turn another soft breach into an increase.
        ctl.healthy[16384] = clock.t
        ctl.set_level(8192, "test lower cap already in force", {"t": clock.t})
        run(ctl, clock, U.SAMPLE_S, echo=251.0)
        assert ctl.level in (0, 8192)
    pol = json.loads((state / W.BOOST_POLICY).read_text())
    assert pol["aggregate_kib_per_s"] != W.UNCAPPED
