"""Tests for the temporary dataset-upload boost: the worker's fail-safe lease reading (ingest_worker.boost_level /
Worker.upload_budget), reductions vs increases during a transfer, and the controller's decisions (upload_boost.py).
Everything runs in scratch directories under /home/ws/ws/realworld-ops/ingest/scratch/tests/ (removed afterwards).

  cd /home/ws/ws/realworld-ops/ingest/scratch && /home/ws/ws/splatter4d/.venv/bin/python -I -m pytest -q \
      -p no:cacheprovider /home/ws/ws/splatter4d/runs/setup/tools/realworld_ingest_20261106/test_upload_boost.py
"""

from __future__ import annotations

import json
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
def state(request) -> Path:
    d = SCRATCH / ("boost-" + re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)[:70] + f"-{os.getpid()}")
    shutil.rmtree(d, ignore_errors=True)
    (d / "upload-boost").mkdir(parents=True)
    (d / "upload-budget.json").write_text(json.dumps({"kib_per_s": 600}))
    (d / "upload-boost/baseline.json").write_text(json.dumps(BASELINE))
    (d / "upload-boost/original-policy.json").write_text(json.dumps(
        {"original": {"dataset_upload_kib_per_s": 600},
         "files": {str(d / "upload-budget.json"): {"sha256": U.budget_sha(d)}}}))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def policy(d: Path, **kw) -> dict:
    now = kw.pop("now", time.time())
    doc = {"schema": W.BOOST_SCHEMA, "state": "active", "aggregate_kib_per_s": 2048, "original_kib_per_s": 600,
           "lease": {"expires_at_unix": now + 150}}
    doc.update(kw)
    (d / W.BOOST_POLICY).write_text(json.dumps(doc))
    return doc


# --------------------------------------------------------------------------------------------- worker: lease reading


def test_boost_level_is_fail_safe(state):
    now = time.time()
    assert W.boost_level(state, 600, now) == (None, "no boost policy")
    policy(state, now=now)
    assert W.boost_level(state, 600, now) == (2048, "boost lease valid")
    bad = [dict(schema="rw-upload-boost/0"), dict(state="standby"), dict(state="restored"),
           dict(aggregate_kib_per_s=16384), dict(aggregate_kib_per_s=1000), dict(aggregate_kib_per_s=True),
           dict(aggregate_kib_per_s="2048"), dict(original_kib_per_s=300), dict(lease=None),
           dict(lease={"expires_at_unix": now - 1}), dict(lease={"expires_at_unix": now + 3600}),
           dict(lease={"expires_at_unix": True}), dict(lease={"expires_at_unix": "soon"}),
           dict(lease={"expires_at_unix": float("nan")}), dict(lease={"expires_at_unix": float("inf")}),
           dict(lease=[1]), dict(lease="x"), dict(aggregate_kib_per_s=[2048])]
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
    policy(state, aggregate_kib_per_s=4096)
    assert w.upload_budget() == (4096, "boost")
    assert w.upload_budget() == (4096, "boost")
    (state / "upload-budget.json").write_text(json.dumps({"kib_per_s": 0}))
    assert w.upload_budget() == (0, "base")
    (state / "upload-budget.json").write_text(json.dumps({"kib_per_s": 600}))
    policy(state, lease={"expires_at_unix": time.time() - 1})
    assert w.upload_budget() == (600, "base")
    ev = [json.loads(x) for x in (state / "events.jsonl").read_text().splitlines()]
    boost = [(e["applied"], e["kib_per_s"], e["reason"]) for e in ev if e["event"] == "upload_boost"]
    assert boost == [(False, 600, "no boost policy"), (True, 4096, "boost lease valid"),
                     (False, 0, "boost granted for base 600, base is 0"), (False, 600, "boost lease expired")]


def test_reduction_is_enforced_during_a_long_scan_and_mirror_is_never_boosted(state, monkeypatch):
    src = state / "episode_000001.h5"
    src.write_bytes(os.urandom(12 * 1024 * 1024))
    remote = LocalShellRemote(state / "remote")
    part = state / "remote/part/episode_000001.h5.part"
    part.parent.mkdir(parents=True)
    w = W.Worker(state, raw_root=state / "raw", tasks={"drawer-open": {"drawer-opening"}}, remote=remote)
    policy(state, aggregate_kib_per_s=2048)
    entry = {"key": "drawer-open/episode_000001.h5", "size": src.stat().st_size, "transfer": {}}
    seen = {}

    def long_scan(skip_key=None):  # 8 s of "validations"; the lease lapses after 1 s
        t0 = time.time()
        while time.time() - t0 < 8:
            if time.time() - t0 > 1 and "expired" not in seen:
                policy(state, aggregate_kib_per_s=2048, lease={"expires_at_unix": time.time() - 1})
                seen["expired"] = time.time()
            time.sleep(0.5)
            if w._inflight and w._inflight["reduced_to"] is not None and "reduced" not in seen:
                seen["reduced"] = time.time()
        w.last_scan = time.time()
    monkeypatch.setattr(W, "SCAN_DURING_TRANSFER_S", 0.0)
    monkeypatch.setattr(w, "scan", long_scan)
    monkeypatch.setattr(w, "write_status", lambda *a, **k: None)
    rc, err = w._run_rsync(src, str(part), 2048, entry)
    assert rc == 0, err and part.read_bytes() == src.read_bytes()
    assert seen["reduced"] - seen["expired"] < 3.0          # watchdog: enforced during the scan, not after it
    assert [next(a for a in c if a.startswith("--bwlimit")) for c in remote.rsync_calls] == \
        ["--bwlimit=2048", "--bwlimit=600"]
    w._mirror_dirty = True
    policy(state, aggregate_kib_per_s=8192)
    w.mirror({})
    assert remote.rsync_calls[-1][remote.rsync_calls[-1].index("--protect-args") + 2] == "--bwlimit=600"


def test_increase_waits_for_next_transfer_reduction_restarts_at_once(state):
    """Real rsync (local): 12 MiB at 2 MiB/s. A raised lease does not interrupt it; an expired lease restarts it at the
    original 600 KiB/s from the partial bytes, and the copy completes intact."""
    src = state / "episode_000001.h5"
    src.write_bytes(os.urandom(12 * 1024 * 1024))
    remote = LocalShellRemote(state / "remote")
    part = state / "remote/part/episode_000001.h5.part"
    part.parent.mkdir(parents=True)
    w = W.Worker(state, raw_root=state / "raw", tasks={"drawer-open": {"drawer-opening"}}, remote=remote)
    policy(state, aggregate_kib_per_s=2048)
    entry = {"key": "drawer-open/episode_000001.h5", "size": src.stat().st_size, "transfer": {}}

    def later():
        time.sleep(1.0)
        policy(state, aggregate_kib_per_s=4096)                      # increase: must not restart rsync
        time.sleep(2.5)
        policy(state, aggregate_kib_per_s=4096, lease={"expires_at_unix": time.time() - 1})  # lapsed: reduce now
    threading.Thread(target=later).start()
    rc, err = w._run_rsync(src, str(part), 2048, entry)
    assert rc == 0, err
    assert part.read_bytes() == src.read_bytes()
    ev = [json.loads(x) for x in (state / "events.jsonl").read_text().splitlines()]
    changes = [(e["old"], e["new"]) for e in ev if e["event"] == "budget_changed_mid_transfer"]
    assert changes == [(2048, 600)]
    assert [next(a for a in c if a.startswith("--bwlimit")) for c in remote.rsync_calls] == \
        ["--bwlimit=2048", "--bwlimit=600"]
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
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def sample(clock, cap=600, rate=None, echo=50.0, srtt=17.0, **kw):
    s = {"t": clock.t, "alive": True, "version": W.WORKER_VERSION, "base_kib": 600, "remaining": 50,
         "all_done": False, "rsync_pid": 4242 if cap else None, "kib_cap": cap, "key": "drawer-open/e.h5",
         "echo_ms": echo, "ctrl_srtt_ms": srtt, "rate_kib": rate if rate is not None else
         (0.97 * cap if cap else None), "rate_cap": cap, "retrans": 1, "segs": 1000}
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
    assert ctl.level == 2048 and lease_level(state, clock) == 2048
    pol = json.loads((state / W.BOOST_POLICY).read_text())
    assert pol["state"] == "active" and 0 < pol["lease"]["expires_at_unix"] - clock.t <= W.BOOST_MAX_LEASE_S
    run(ctl, clock, U.DWELL_S - 60, events_every=180)
    assert ctl.level == 2048                       # dwell not over yet
    run(ctl, clock, 120, events_every=180)
    assert ctl.level == 4096
    run(ctl, clock, U.DWELL_S + 60, events_every=180)
    assert ctl.level == 8192
    run(ctl, clock, 3 * U.DWELL_S, events_every=180)
    assert ctl.level == 8192 and lease_level(state, clock) == 8192
    assert set(ctl.healthy) == {2048, 4096, 8192}
    assert lease_level(state, Clock(clock.t + U.LEASE_S + 1)) is None   # controller gone -> lease lapses


def test_not_cap_bound_holds_and_reports_bottleneck(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    run(ctl, clock, 2 * U.DWELL_S, events_every=180, rate=1200.0)   # 2 MiB/s cap, only 1.2 MiB/s delivered
    assert ctl.level == 2048
    notes = [json.loads(x) for x in (state / "upload-boost/history.jsonl").read_text().splitlines()]
    assert any(n["event"] == "note" and "not cap-bound" in n["detail"] for n in notes)


def test_soft_breach_drops_to_last_healthy_level(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    run(ctl, clock, U.DWELL_S + 60, events_every=180)
    assert ctl.level == 4096
    run_until_change(ctl, clock, echo=400.0)             # control latency 8x baseline at the 4 MiB/s cap
    assert ctl.level == 2048 and lease_level(state, clock) == 2048
    assert ctl.blocked(4096, clock.t) and not ctl.blocked(2048, clock.t)
    run(ctl, clock, 2 * U.DWELL_S, events_every=180)      # 4096 in cooldown: hold at 2048
    assert ctl.level == 2048
    run_until_change(ctl, clock, srtt=90.0)              # path RTT breach at 2048 -> original
    assert ctl.level == 0 and lease_level(state, clock) is None


def test_hard_failure_restores_original_with_cooldown(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    assert ctl.level == 2048
    ctl.sampler.events.append({"t": iso(clock.t), "event": "transfer_error", "key": "k",
                               "error": "remote part sha256 aaa != local bbb (part removed, full retransfer)"})
    run(ctl, clock, U.SAMPLE_S)
    assert ctl.level == 0 and lease_level(state, clock) is None
    assert "sha256 mismatch" in json.loads((state / W.BOOST_POLICY).read_text())["reason"]
    run(ctl, clock, U.COOLDOWN_S - 120)
    assert ctl.level == 0                                   # cooldown of the failed level
    run(ctl, clock, 240)
    assert ctl.level == 2048


def test_stall_and_echo_failures_are_hard(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    run(ctl, clock, (U.STALL_SAMPLES - 1) * U.SAMPLE_S, rate=0.0)
    assert ctl.level == 2048                                # a resume's prefix verification is not a stall
    run(ctl, clock, U.SAMPLE_S, rate=0.0)
    assert ctl.level == 0 and "stalled" in ctl.reason
    ctl.failures.clear()
    ctl.last_hard_at = 0
    enter(ctl, clock)
    assert ctl.level == 2048
    run(ctl, clock, 2 * U.SAMPLE_S, echo=None)
    assert ctl.level == 2048                                # no master: no sample, not a failure
    for _ in range(2):
        clock.t += U.SAMPLE_S
        ctl.sampler.next = sample(clock, cap=2048, echo=None)
        ctl.sampler.next["echo"] = "exit 255"
        ctl.tick()
    assert ctl.level == 0 and "echo failures" in ctl.reason


def test_isolated_latency_spikes_do_not_trigger(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    for i in range(40):                                     # the baseline's pattern: 1 in 4 samples at 130-250 ms
        run(ctl, clock, U.SAMPLE_S, echo=250.0 if i % 4 == 0 else 50.0, srtt=52.0 if i % 3 == 0 else 20.0)
    assert ctl.level >= 2048 and not ctl.failures
    run_until_change(ctl, clock, echo=180.0)                # sustained: most samples high -> soft
    assert ctl.failures and "control echoes" in ctl.reason


def test_missing_measurements_restore_and_unhealthy_entry_is_refused(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    run(ctl, clock, U.ENTRY_QUIET_S + 120, echo=400.0)      # unhealthy at the original budget: never enter
    assert ctl.level == 0 and not ctl.failures
    enter(ctl, clock)
    run(ctl, clock, U.MISSING_S + 60, echo=None)            # boosted but no control-channel measurement
    assert ctl.level == 0 and "missing evidence" in ctl.reason
    enter(ctl, clock)
    run(ctl, clock, U.MISSING_S + 60, rate=None, **{"rate_kib": None})  # uploading but no data-flow measurement
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
    ctl.fail(8192, clock.t - 2 * U.COOLDOWN_S)
    ctl.fail(8192, clock.t - U.COOLDOWN_S - 10)
    assert ctl.ceiling(clock.t) == 4096 and ctl.blocked(8192, clock.t) and not ctl.blocked(4096, clock.t)


def test_standby_when_idle_worker_gone_or_base_changed_then_reentry(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    run(ctl, clock, U.DWELL_S + 60, events_every=180)
    assert ctl.level == 4096
    run(ctl, clock, U.IDLE_GRACE_S - 60, cap=None, remaining=7)
    assert ctl.level == 4096                                 # gaps between transfers keep the level
    run(ctl, clock, 120, cap=None, remaining=7)
    assert ctl.level == 0 and "no upload for" in ctl.reason  # pending but nothing uploads (e.g. back-off)
    enter(ctl, clock, remaining=7)
    assert ctl.level > 0                                     # the worker uploads again
    run(ctl, clock, U.SAMPLE_S, cap=None, remaining=0)       # nothing in flight, nothing pending: at once
    assert ctl.level == 0 and json.loads((state / W.BOOST_POLICY).read_text())["state"] == "standby"
    enter(ctl, clock, remaining=3)                            # new arrivals: re-enter at the recently healthy level
    assert ctl.level == 2048 or ctl.level == 4096
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
    assert lease_level(state, clock) == 2048
    U.cmd_restore(state, "test")
    assert lease_level(state, clock) is None
    clock.t += U.SAMPLE_S
    ctl.sampler.next = sample(clock, cap=2048)
    assert ctl.tick() == "stop"                               # the loop may not re-activate after a restore
    assert json.loads((state / W.BOOST_POLICY).read_text())["state"] == "restored"


def test_retire_when_both_tasks_complete(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    clock.t += U.SAMPLE_S
    ctl.sampler.next = sample(clock, all_done=True, remaining=0, cap=None)
    assert ctl.tick() == "retire"
    U.retire(state, ctl)
    pol = json.loads((state / W.BOOST_POLICY).read_text())
    ret = json.loads((state / "upload-boost/retired.json").read_text())
    assert pol["state"] == "retired" and lease_level(state, clock) is None and ret["original_restored"] is True
    assert U.main(["run", "--state-dir", str(state)]) == 0    # a retired boost never starts again


def test_run_span_aggregate():
    xs = [{"cap": 2048, "final_cap": 2048, "t_start": 0, "t_end": 50, "bytes": 1},
          {"cap": 2048, "final_cap": 2048, "t_start": 52, "t_end": 100, "bytes": 1},
          {"cap": 4096, "final_cap": 4096, "t_start": 101, "t_end": 120, "bytes": 1},
          {"cap": 2048, "final_cap": 2048, "t_start": 500, "t_end": 550, "bytes": 1}]
    assert U.run_span(xs, 2048) == (150.0, [2.0])
    assert U.run_span(xs, 4096) == (19.0, [])


def test_quiet_period_is_taken_from_the_record(state):
    now = time.time()
    with open(state / "events.jsonl", "w") as f:
        f.write(json.dumps({"t": iso(now - 1200), "event": "transfer_error", "key": "old"}) + "\n")
    assert U.last_hard_failure(state, now) == 0.0                       # older than ENTRY_QUIET_S: quiet shown
    with open(state / "events.jsonl", "a") as f:
        f.write(json.dumps({"t": iso(now - 120), "event": "remote_unreachable"}) + "\n")
    assert abs(U.last_hard_failure(state, now) - (now - 120)) < 2          # recent: wait out the quiet period
    with open(state / "upload-boost/history.jsonl", "w") as f:
        f.write(json.dumps({"t": iso(now - 30), "event": "level", "reason": "soft: x"}) + "\n")
    assert abs(U.last_hard_failure(state, now) - (now - 30)) < 2


def test_hard_failure_wins_over_missing_evidence_and_evidence_survives_step_up(state):
    clock = Clock()
    ctl = U.Controller(state, sampler=FakeSampler(), clock=clock)
    enter(ctl, clock)
    run(ctl, clock, U.MISSING_S - 60, echo=None)
    ctl.sampler.events.append({"t": iso(clock.t), "event": "transfer_error", "key": "k", "error": "rsync exit 12"})
    run(ctl, clock, 120, echo=None)
    assert ctl.level == 0 and ctl.reason.startswith("hard:") and ctl.blocked(2048, clock.t)
    ctl2 = U.Controller(state, sampler=FakeSampler(), clock=clock)
    ctl2.failures.clear()
    ctl2.last_hard_at = 0
    enter(ctl2, clock)
    run(ctl2, clock, U.DWELL_S + 60, events_every=180)
    assert ctl2.level == 4096
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
    marker.write_text(json.dumps({"pid": 20, "kib_per_s": 4096}))
    flow["inode"] = 2
    for _ in range(3):
        sp.poll_flows()
        clock.t += 5
        flow["bytes"] += 4096 * 1024 * 5
    marker.write_text(json.dumps({"pid": 10, "kib_per_s": 8192}))
    flow["inode"] = 3
    sp.poll_flows()
    s = sp.consume_rates({})
    assert s["rate_cap"] == 4096 and abs(s["rate_kib"] - 4096) < 1 and sp.acc == {}


def test_late_watchdog_latch_restarts_instead_of_failing(state, monkeypatch):
    src = state / "episode_000001.h5"
    src.write_bytes(os.urandom(2 * 1024 * 1024))
    remote = LocalShellRemote(state / "remote")
    part = state / "remote/part/episode_000001.h5.part"
    part.parent.mkdir(parents=True)
    w = W.Worker(state, raw_root=state / "raw", tasks={"drawer-open": {"drawer-opening"}}, remote=remote)
    policy(state, aggregate_kib_per_s=2048)
    entry = {"key": "drawer-open/episode_000001.h5", "size": src.stat().st_size, "transfer": {}}
    calls = []

    def racing_watchdog(cur, stop):  # latches only after rsync ended by itself (the race window)
        calls.append(cur["kib"])
        if len(calls) == 1:
            cur["proc"].wait()
            cur["reduced_to"] = 600
    monkeypatch.setattr(w, "_budget_watchdog", racing_watchdog)
    rc, err = w._run_rsync(src, str(part), 2048, entry)
    assert rc == 0, err
    assert [next(a for a in c if a.startswith("--bwlimit")) for c in remote.rsync_calls] == \
        ["--bwlimit=2048", "--bwlimit=600"]
    assert part.read_bytes() == src.read_bytes() and entry["transfer"]["kib_per_s"] == 600
