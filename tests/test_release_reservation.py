"""Release verification priority (user 2026-11-12): ops/jobs/release_reservation.py and its use by the scheduler tick.

While a release is pending, every other remote admission is charged its reserved GPU slot / memory, RAM, disk and CPU
threads; the release's own verification jobs are admitted against it (with a dedicated disk-floor slice); running jobs
are never touched; an expired or orphaned reservation reserves nothing; a malformed one fails closed."""

from __future__ import annotations

import json
import os
import time

import pytest

from ops.jobs import release_reservation as rr
from ops.jobs import scheduler as jobs
from ops.resources.gpu_guard import APPROVED_HOST_GPUS
from tests.test_remote_jobs import GIB, REMOTE, FakeBackend, job, launch_event, tick, write_queue

GPUS = list(APPROVED_HOST_GPUS["remote"])
COMMIT = "e" * 40
NATIVE = {"mem_gb": 32, "load": 1.0, "ram_gb": 96, "cpu_threads": 32, "disk_gb": 1}
ISOLATION = {"mem_gb": 4, "load": 0.0, "ram_gb": 8, "cpu_threads": 8, "disk_gb": 1}


def plan(stage="native", native_gpu=REMOTE, pid=None, now=None, **native):
    return rr.plan(COMMIT, "eeeeeee-cc", stage, native_gpu, GPUS, native={**NATIVE, **native}, isolation=ISOLATION,
                   pid=os.getpid() if pid is None else pid, now=now)


def test_plan_reserves_native_and_isolation_share_then_only_isolation():
    native = plan()
    assert native["jobs"] == ["verify-native-eeeeeee-cc"] + [f"verify-isolation-{i}-eeeeeee-cc" for i in range(4)]
    assert native["gpus"][REMOTE] == {"mem_gb": 36.0, "load": 1.0}
    assert all(native["gpus"][g] == {"mem_gb": 4.0, "load": 0.0} for g in GPUS if g != REMOTE)
    assert (native["ram_gb"], native["cpu_threads"], native["disk_gb"]) == (96 + 4 * 8, 32 + 4 * 8, 1 + 4)
    isolation = plan("isolation")
    assert isolation["jobs"] == [f"verify-isolation-{i}-eeeeeee-cc" for i in range(4)]
    assert all(isolation["gpus"][g] == {"mem_gb": 4.0, "load": 0.0} for g in GPUS)
    assert (isolation["ram_gb"], isolation["cpu_threads"], isolation["disk_gb"]) == (32, 32, 4)
    with pytest.raises(ValueError):
        plan("other")


def test_load_write_clear_and_lifetime(tmp_path):
    path = tmp_path / "r.json"
    assert rr.load(path) is None  # no file: no reservation
    rr.write(plan(), path)
    assert rr.load(path)["commit"] == COMMIT
    assert rr.load(path, now=time.time() + rr.EXPIRY_SECONDS + 1) is None  # not renewed: lapses by itself
    assert rr.load(path, alive=lambda pid: False) is None  # the chain is gone: nothing reserved
    assert not rr.clear("f" * 40, path) and path.exists()  # another release's reservation is kept
    assert not rr.clear(COMMIT, path, pid=os.getpid() + 1) and path.exists()  # another writer's is kept
    assert rr.clear(COMMIT, path) and not path.exists()


@pytest.mark.parametrize("change", [{"schema": 2}, {"commit": "abc"}, {"pid": 0}, {"pid": True}, {"jobs": []},
                                    {"jobs": ["r"]}, {"gpus": {}}, {"gpus": {"x": {"mem_gb": 1, "load": 0}}},
                                    {"gpus": {REMOTE: {"mem_gb": -1, "load": 0}}}, {"ram_gb": "1"},
                                    {"expires": None}, {"expires": float("nan")}, {"expires": 1e309},
                                    {"created": float("inf")}, {"expires": time.time() + 10 * 86400}])
def test_malformed_reservations_fail_closed(tmp_path, change):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({**plan(), **change}))
    with pytest.raises(ValueError):
        rr.load(path)
    path.write_text("{not json")
    with pytest.raises(ValueError):
        rr.load(path)
    with pytest.raises(ValueError):
        rr.reserved_gpus(path)


def test_covers_only_the_reserved_releases_verification_jobs():
    doc = plan()
    assert rr.covers(doc, {"id": "verify-native-eeeeeee-cc", "verification_purpose": "native_suite"})
    assert not rr.covers(doc, {"id": "verify-native-eeeeeee-cc"})  # a campaign row cannot borrow the id
    assert not rr.covers(doc, {"id": "verify-native-fffffff-cc", "verification_purpose": "native_suite"})
    assert not rr.covers(None, {"id": "verify-native-eeeeeee-cc", "verification_purpose": "native_suite"})
    assert rr.charges(None) == {"gpus": {}, "ram_gb": 0.0, "cpu_threads": 0, "disk_gb": 0.0}


def test_default_path_is_resolved_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "PATH", tmp_path / "moved.json")
    rr.write(plan())
    assert (tmp_path / "moved.json").exists() and rr.load()["commit"] == COMMIT and rr.reserved_gpus() == set(GPUS)


# ---- the scheduler tick ----

def reserve(tmp_path, **native):
    path = tmp_path / "reservation.json"
    rr.write(plan(**native), path)
    return path


def launched(backend):
    return [payload["id"] for payload in backend.launches]


def test_no_reservation_admits_as_before(tmp_path):
    queue = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    tick(tmp_path, queue, backend, reservation_path=tmp_path / "absent.json")
    assert launched(backend) == ["r"]


def test_reserved_gpu_memory_and_slot_hold_new_campaign_work(tmp_path):
    queue = write_queue(tmp_path, [job(mem_gb=5)])  # limits: 2 jobs / 40 GB on REMOTE
    backend = FakeBackend(tmp_path / "runs")
    messages = tick(tmp_path, queue, backend, reservation_path=reserve(tmp_path, mem_gb=34))  # 34 + 4 + 5 > 40
    assert launched(backend) == []
    assert any("reserved for release eeeeeee-cc" in m for m in messages)


def test_reserved_ram_holds_new_campaign_work(tmp_path):
    queue = write_queue(tmp_path, [job(ram_gb=10)])  # snapshot 200 GB, reserve 40: 200 - 160 - 10 < 40
    backend = FakeBackend(tmp_path / "runs")
    messages = tick(tmp_path, queue, backend, reservation_path=reserve(tmp_path, mem_gb=1, ram_gb=128))
    assert launched(backend) == [] and any("reserved for release" in m and "host RAM" in m for m in messages)


def test_reserved_cpu_threads_hold_new_campaign_work(tmp_path):
    queue = write_queue(tmp_path, [job()], cpu_thread_budget=64)  # 32 + 32 reserved + 8 > 64
    backend = FakeBackend(tmp_path / "runs")
    messages = tick(tmp_path, queue, backend, reservation_path=reserve(tmp_path, mem_gb=1, ram_gb=1))
    assert launched(backend) == [] and any("CPU threads" in m for m in messages)
    queue = write_queue(tmp_path, [job()], cpu_thread_budget=320)
    tick(tmp_path, queue, backend, reservation_path=reserve(tmp_path, mem_gb=1, ram_gb=1))
    assert launched(backend) == ["r"]


def test_reserved_disk_holds_new_campaign_work(tmp_path):
    queue = write_queue(tmp_path, [job(disk_gb=20)])  # floor 150 GiB of 1000; free 800 - 640 reserved - 20 < 150
    backend = FakeBackend(tmp_path / "runs")
    path = tmp_path / "reservation.json"
    rr.write({**plan(mem_gb=1, ram_gb=1), "disk_gb": 640}, path)
    messages = tick(tmp_path, queue, backend, reservation_path=path)
    assert launched(backend) == [] and any("disk floor" in m for m in messages)


def test_the_releases_own_job_is_admitted_against_the_reservation(tmp_path, monkeypatch):
    # stands in for the release's verification job (its full typed plumbing is covered in test_remote_jobs.py)
    monkeypatch.setattr(rr, "covers", lambda doc, queued: bool(doc) and queued["id"] == "r")
    queue = write_queue(tmp_path, [job(mem_gb=5, disk_gb=1)])
    backend = FakeBackend(tmp_path / "runs")
    tick(tmp_path, queue, backend, reservation_path=reserve(tmp_path, mem_gb=34))
    assert launched(backend) == ["r"]


def test_the_releases_own_job_keeps_the_normal_disk_floor(tmp_path, monkeypatch):
    # Codex review 2026-11-12: the remote launch guard enforces the configured floor, so admission never lowers it
    monkeypatch.setattr(rr, "covers", lambda doc, queued: bool(doc) and queued["id"] == "r")
    queue = write_queue(tmp_path, [job(mem_gb=5, disk_gb=1)])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.disk_free_bytes = int(150 * GIB - 8 * GIB)  # below the 150 GiB floor
    messages = tick(tmp_path, queue, backend, reservation_path=reserve(tmp_path, mem_gb=34))
    assert launched(backend) == [] and any("disk floor" in m for m in messages)


def test_cpu_threads_are_counted_across_launches_of_one_tick(tmp_path):
    # budget 96: 64 reserved; the first 8-thread job fits (8 + 64 <= 96), the next ones must count it
    queue = write_queue(tmp_path, [job("a", mem_gb=1), job("b", mem_gb=1), job("c", mem_gb=1), job("d", mem_gb=1),
                                   job("e", mem_gb=1)], cpu_thread_budget=96,
                        limits={REMOTE: {"max_jobs": 20, "max_mem_gb": 90}})
    backend = FakeBackend(tmp_path / "runs")
    messages = tick(tmp_path, queue, backend, reservation_path=reserve(tmp_path, mem_gb=1, ram_gb=1))
    assert len(launched(backend)) == 4 and any("CPU threads" in m for m in messages)  # 4 x 8 + 64 = 96


def test_expired_or_orphaned_reservations_reserve_nothing(tmp_path):
    path = tmp_path / "reservation.json"
    rr.write(plan(mem_gb=34, now=time.time() - rr.EXPIRY_SECONDS - 5), path)
    queue = write_queue(tmp_path, [job(mem_gb=5)])
    backend = FakeBackend(tmp_path / "runs")
    tick(tmp_path, queue, backend, reservation_path=path)
    assert launched(backend) == ["r"]
    rr.write(plan(mem_gb=34, pid=2**22 + 12345), path)  # no such process
    queue = write_queue(tmp_path, [job("s", mem_gb=5)])
    tick(tmp_path, queue, backend, reservation_path=path)
    assert launched(backend) == ["r", "s"]


def test_malformed_reservation_admits_no_remote_job(tmp_path):
    path = tmp_path / "reservation.json"
    path.write_text("{broken")
    queue = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    messages = tick(tmp_path, queue, backend, reservation_path=path)
    assert launched(backend) == [] and any("release reservation unusable" in m for m in messages)


def test_a_reservation_never_touches_running_jobs(tmp_path):
    queue = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    launch_event(tmp_path)
    from tests.test_remote_jobs import inspected
    backend.value.containers["s4d-r-a1"] = inspected(status="running")
    before = (tmp_path / "registry.jsonl").read_text()
    tick(tmp_path, queue, backend, reservation_path=reserve(tmp_path, mem_gb=34))
    after = (tmp_path / "registry.jsonl").read_text()
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["status"] == "running"
    assert not backend.removals and not backend.launches
    assert all(json.loads(line)["event"] not in ("exited", "failed", "container_removed")
               for line in after[len(before):].splitlines())


def test_scoped_tick_reconciles_only_its_jobs(tmp_path):
    from tests.test_remote_jobs import inspected
    queue = write_queue(tmp_path, [job("r"), job("q")])
    backend = FakeBackend(tmp_path / "runs")
    for run_id in ("r", "q"):
        launch_event(tmp_path, run_id)
        backend.value.containers[f"s4d-{run_id}-a1"] = {**inspected(run_id, status="exited"), "pids": set()}
    tick(tmp_path, queue, backend, scope={"r"})
    events = [json.loads(line) for line in (tmp_path / "registry.jsonl").read_text().splitlines()]
    assert any(e["id"] == "r" and e["event"] in ("remote_exited", "remote_verified", "exited", "results_synced",
                                                   "remote_verification_failed") for e in events)
    assert [e["event"] for e in events if e["id"] == "q"] == ["launched"]  # untouched: left to the daemon
    assert not backend.removals  # no container cleanup inside a scoped tick


# ---- the remote campaign driver's placement (its block run in isolation, as test_remote_jobs does for co-tenants) ----

def _driver_block():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "ops/campaign/remote_campaign_driver.py").read_text()
    start = source.index("# Release verification priority (user 2026-11-12; ops/jobs/release_reservation.py)")
    return source[start:source.index("reviwo = sum(", start)]


def _run_driver_block(path):
    names = {"candidates": [{"id": "x"}], "COTENANT": {"gpus": {}}, "messages": [], "json": json,
             "gpu_load": {g: 3.0 for g in GPUS}, "gpu_mem": {g: 30.0 for g in GPUS}, "reserved": 10 * GIB,
             "release_reservation": rr}
    rr_path = rr.PATH
    try:
        rr.PATH = path
        exec(_driver_block(), names)
    finally:
        rr.PATH = rr_path
    return names


def test_driver_charges_the_reservation_before_any_placement(tmp_path):
    path = tmp_path / "reservation.json"
    names = _run_driver_block(path)  # no reservation: unchanged
    assert names["gpu_load"][REMOTE] == 3.0 and names["gpu_mem"][REMOTE] == 30.0 and names["reserved"] == 10 * GIB
    assert names["candidates"] and names["COTENANT"] is not None and not names["messages"]
    rr.write(plan(), path)
    names = _run_driver_block(path)
    assert names["gpu_load"][REMOTE] == 4.0 and names["gpu_mem"][REMOTE] == 30.0 + 36.0  # native + isolation share
    assert all(names["gpu_load"][g] == 3.0 and names["gpu_mem"][g] == 34.0 for g in GPUS if g != REMOTE)
    assert names["reserved"] == 10 * GIB + int((96 + 32) * GIB) and names["candidates"]
    assert names["messages"][0].startswith("RELEASE_RESERVED eeeeeee-cc (native)")
    path.write_text("{broken")
    names = _run_driver_block(path)  # unusable: no admission and no pool move (COTENANT None blocks the pool)
    assert names["candidates"] == [] and names["COTENANT"] is None
    assert names["messages"][0].startswith("RELEASE_RESERVATION unusable")


# ---- release chain polling (ops/jobs/release_poll.py) ----

from ops.jobs import release_poll  # noqa: E402


def test_next_action_watches_only_when_every_unfinished_job_runs():
    ids = ["a", "b"]
    run = lambda c: {"status": "running", "container": c}  # noqa: E731
    assert release_poll.next_action({"a": {"status": "done"}, "b": {"status": "done"}}, ids) == ("done", {})
    assert release_poll.next_action({"a": {"status": "failed"}, "b": run("cb")}, ids) == ("failed", {})
    assert release_poll.next_action({"a": run("ca"), "b": run("cb")}, ids) == ("watch", {"a": "ca", "b": "cb"})
    assert release_poll.next_action({"a": {"status": "done"}, "b": run("cb")}, ids) == ("watch", {"b": "cb"})
    assert release_poll.next_action({"a": run("ca")}, ids) == ("retry", {})  # b not admitted yet
    assert release_poll.next_action({"a": run("ca"), "b": {"status": "sync_pending"}}, ids) == ("retry", {})
    assert release_poll.next_action({"a": {"status": "running"}, "b": run("cb")}, ids) == ("retry", {})  # no name


class Clock:
    def __init__(self):
        self.now = 0.0

    def sleep(self, seconds):
        self.now += seconds


def test_wait_for_exit_returns_on_stop_quiet_or_error_and_renews_each_poll():
    clock, renewals = Clock(), []
    answers = iter([{"c1": "running", "c2": "running"}, {"c1": "running", "c2": "exited"}])
    outcome = release_poll.wait_for_exit(lambda names: next(answers), ["c1", "c2"], renew=lambda: renewals.append(1),
                                         poll_seconds=10, max_quiet=240, sleep=clock.sleep, clock=lambda: clock.now)
    assert outcome == "stopped" and clock.now == 20 and len(renewals) == 2
    clock = Clock()
    outcome = release_poll.wait_for_exit(lambda names: {"c1": "running"}, ["c1"], poll_seconds=10, max_quiet=60,
                                         sleep=clock.sleep, clock=lambda: clock.now)
    assert outcome == "quiet" and clock.now == 60
    clock = Clock()
    outcome = release_poll.wait_for_exit(lambda names: {}, ["c1"], poll_seconds=10, max_quiet=60, sleep=clock.sleep,
                                         clock=lambda: clock.now)
    assert outcome == "stopped"  # a container Docker no longer knows is not running: tick
    clock, messages = Clock(), []

    def broken(names):
        raise OSError("ssh")
    outcome = release_poll.wait_for_exit(broken, ["c1"], sleep=clock.sleep, clock=lambda: clock.now,
                                         log=messages.append)
    assert outcome == "error" and "container poll failed" in messages[0]


def test_container_states_query_is_read_only_and_parses_names():
    from ops.jobs.remote_backend import RemoteBackend

    class Client:
        root = "/home/compu/kaist/sunho"

        def __init__(self):
            self.commands = []

        def checked_ssh(self, command, **kwargs):
            self.commands.append((command, kwargs))
            from types import SimpleNamespace
            return SimpleNamespace(stdout="/s4d-verify-native-x-a1 running\n/s4d-other-a1 exited\ngarbage\n")
    client = Client()
    backend = RemoteBackend.__new__(RemoteBackend)
    backend.client = client
    assert backend.container_states(["s4d-verify-native-x-a1"]) == {"s4d-verify-native-x-a1": "running"}
    command, kwargs = client.commands[0]
    assert command.startswith("docker inspect --format") and kwargs.get("idempotent") is True
    for bad in ([], ["x; rm -rf /"], ["s4d-a b"], ["../s4d-x"]):
        with pytest.raises(ValueError):
            backend.container_states(bad)


def test_future_timestamps_fail_closed_and_lifetime_is_bounded(tmp_path):
    path = tmp_path / "r.json"
    rr.write(plan(now=time.time() + 3600), path)  # a clock far ahead: malformed for every reader
    with pytest.raises(ValueError):
        rr.load(path)
    created = time.time() - rr.MAX_LIFETIME_SECONDS - 10
    doc = rr.plan(COMMIT, "eeeeeee-cc", "native", REMOTE, GPUS, native=NATIVE, isolation=ISOLATION, created=created)
    doc["expires"] = time.time() + 60  # still renewed, but past its maximum lifetime: reserves nothing
    path.write_text(json.dumps(doc))
    assert rr.load(path) is None
    doc["expires"] = created + rr.MAX_LIFETIME_SECONDS + rr.EXPIRY_SECONDS + 1  # beyond any valid lifetime
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        rr.load(path)


def test_a_live_reservation_of_another_release_or_writer_is_never_overwritten(tmp_path):
    path = tmp_path / "r.json"
    rr.write(plan(), path)
    other = rr.plan("f" * 40, "fffffff-cc", "native", REMOTE, GPUS, native=NATIVE, isolation=ISOLATION)
    with pytest.raises(rr.ReservationConflict):
        rr.write(other, path)
    with pytest.raises(rr.ReservationConflict):
        rr.write(plan(pid=os.getppid()), path)  # same commit, another live writer
    rr.write(plan(), path)  # its own renewal is fine
    rr.write(other, path, alive=lambda pid: False)  # an orphaned reservation is replaced
    assert json.loads(path.read_text())["commit"] == "f" * 40


def test_scoped_reconciliation_owns_only_validated_out_of_scope_containers(tmp_path):
    from tests.test_remote_jobs import inspected
    backend = FakeBackend(tmp_path / "runs")
    launch_event(tmp_path, "q")
    good = {**inspected("q", status="running"), "pids": {101}}
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    snapshot = backend.value
    snapshot.containers["s4d-q-a1"] = good
    owned, _ = jobs._reconcile_remote(backend, snapshot, state, tmp_path / "registry.jsonl", tmp_path / "runs", [],
                                      scope={"verify-x"})
    assert owned == {101}
    snapshot.containers["s4d-q-a1"] = {**good, "Image": "sha256:" + "0" * 64}  # identity mismatch
    owned, _ = jobs._reconcile_remote(backend, snapshot, state, tmp_path / "registry.jsonl", tmp_path / "runs", [],
                                      scope={"verify-x"})
    assert owned == set()
    events = [json.loads(line)["event"] for line in (tmp_path / "registry.jsonl").read_text().splitlines()]
    assert events == ["launched"]  # nothing recorded for out-of-scope jobs
