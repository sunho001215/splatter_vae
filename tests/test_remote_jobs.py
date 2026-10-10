from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from s4d import remote_jobs as remote
from s4d.gpu_guard import APPROVED_HOST_GPUS
from scripts._nvidia_query import QueryUnavailable

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("remote_scheduler", REPO / "scripts/jobs.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
LOCAL = APPROVED_HOST_GPUS["local"][0]
REMOTE = APPROVED_HOST_GPUS["remote"][0]
COMMIT = "a" * 40
IMAGE = "sha256:" + "b" * 64
DATA = "c" * 64
CONTAINER_ID = "d" * 64
SOURCE = {"scripts/worker.py": "e" * 64}
GIB = 2**30


def job(run_id="r", host="remote", **overrides):
    return {"id": run_id, "host": host, "script": "tests/_job_worker.py", "args": [0],
            "gpu": REMOTE if host == "remote" else LOCAL, "mem_gb": 5, "ram_gb": 10,
            "disk_gb": 20, "required_results": ["metrics.jsonl"], **overrides}


def write_queue(tmp_path, job_list, **remote_overrides):
    payload = {"hosts": {
        "local": {"limits": {LOCAL: {"max_jobs": 2, "max_mem_gb": 40}}},
        "remote": {"enabled": True, "commit": COMMIT, "image_digest": IMAGE,
                   "limits": {REMOTE: {"max_jobs": 2, "max_mem_gb": 40}},
                   "host_ram_reserve_gb": 40, "host_ram_ramp_minutes": 10,
                   "disk_floor_gb": 100, "disk_floor_fraction": 0.15, **remote_overrides}},
               "jobs": job_list}
    path = tmp_path / "q.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def inspected(run_id="r", attempt=1, status="running", code=0, gpu=REMOTE, gpu_backend="cdi"):
    device = f"nvidia.com/gpu={gpu}" if gpu_backend == "cdi" else gpu
    visible = "void" if gpu_backend == "cdi" else gpu
    return {"Name": f"/s4d-{run_id}-a{attempt}", "Id": CONTAINER_ID, "Image": IMAGE,
            "State": {"Status": status, "ExitCode": code},
            "Config": {"Labels": remote.labels(run_id, attempt, COMMIT, IMAGE),
                       "Env": [f"CUDA_VISIBLE_DEVICES={gpu}", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics",
                               f"NVIDIA_VISIBLE_DEVICES={visible}"]},
            "HostConfig": {"DeviceRequests": [{"Driver": "cdi" if gpu_backend == "cdi" else "nvidia", "Count": 0,
                                              "DeviceIDs": [device], "Capabilities": None, "Options": None}],
                           "Devices": [], "Privileged": False, "ReadonlyRootfs": True,
                           "PidMode": "host", "ShmSize": 8 * GIB},
            "Mounts": [{"Type": "bind", "Source": source, "Destination": destination, "RW": writable}
                       for source, destination, writable in [
                           (f"{remote.ROOT}/releases/{COMMIT}/code", "/workspace/splatter4d", False),
                           (f"{remote.ROOT}/runs", "/workspace/splatter4d/runs", True),
                           (f"{remote.ROOT}/outputs", "/workspace/splatter4d/outputs", True),
                           (f"{remote.ROOT}/runtime/cache", "/workspace/splatter4d/.cache", True),
                           (f"{remote.ROOT}/runtime", f"{remote.ROOT}/runtime", True),
                           (f"{remote.ROOT}/runtime/acceptance/{COMMIT}-{IMAGE[7:]}",
                            f"{remote.ROOT}/runtime/acceptance/{COMMIT}-{IMAGE[7:]}", False),
                           (f"{remote.ROOT}/releases/{COMMIT}/provenance.json",
                            f"{remote.ROOT}/releases/{COMMIT}/provenance.json", False),
                           (f"{remote.ROOT}/data/metaworld/splatter4d_v1",
                            "/home/ws/data/metaworld/splatter4d_v1", False),
                       ]], "pids": {123}}


class FakeClient:
    root = remote.ROOT


class FakeBackend:
    def __init__(self, runs):
        self.runs = runs
        self.value = remote.RemoteSnapshot({uuid: [] for uuid in APPROVED_HOST_GPUS["remote"]},
                                           200, 1000 * GIB, 800 * GIB, {})
        self.launches, self.syncs = [], []
        self.snapshot_error = self.launch_error = self.sync_error = self.ready_error = None
        self.registry = runs.parent / "registry.jsonl"

    def snapshot(self):
        if self.snapshot_error:
            raise self.snapshot_error
        return self.value

    def ready(self, host):
        if self.ready_error:
            raise self.ready_error
        return {"data_identity": DATA, "manifest_sha256": "f" * 64}

    def prepare(self, queued, gpu, attempt, host, evidence):
        return remote.RemoteBackend(REPO, self.runs, FakeClient()).prepare(queued, gpu, attempt, host, evidence)

    def launch(self, path):
        payload = json.loads(path.read_text())
        current = jobs.read_registry(self.registry)[payload["id"]]
        assert current["status"] == "launching" and current["attempts"] == payload["attempt"]
        self.launches.append(payload)
        item = inspected(payload["id"], payload["attempt"], gpu_backend=payload["gpu_backend"])
        self.value.containers[payload["container"]] = item
        if self.launch_error:
            raise self.launch_error
        return {"container_id": CONTAINER_ID}

    def sync_results(self, run_id, final=False, checkpoints=False):
        self.syncs.append((run_id, final))
        if self.sync_error:
            raise self.sync_error
        current = jobs.read_registry(self.registry)[run_id]
        payload = {"host": "remote", "job_id": run_id, "attempt": current["attempts"],
                   "git_commit": current["commit"], "image_digest": current["image_digest"],
                   "gpu_uuids": [current["gpu"]], "script": current["script"], "arguments": current["args"]}
        folder = self.runs / run_id
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, content in (("exit.json", {**payload, "code": 0}),
                              ("provenance.json", {**payload, "data_id": DATA}),
                              ("metrics.jsonl", {"step": 1, "loss": 0.1})):
            data = json.dumps(content).encode()
            (folder / name).write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
        return {"run_id": run_id, "host": "remote", "final": final, "verified": True, "files": files}


def tick(tmp_path, queue, backend, **kwargs):
    return jobs.tick(queue, tmp_path / "registry.jsonl", tmp_path / "runs", usage={}, available_gb=200,
                     disk_bytes=(2000 * GIB, 1000 * GIB), remote_backend=backend, **kwargs)


def launch_event(tmp_path, run_id="r", kind="launched", **overrides):
    jobs.record({"event": kind, "id": run_id, "host": "remote", "gpu": REMOTE, "attempt": 1,
                 "container": remote.container_name(run_id, 1), "container_id": CONTAINER_ID,
                 "commit": COMMIT, "image_digest": IMAGE, "data_identity": DATA, "gpu_backend": "cdi",
                 "required_results": ["metrics.jsonl"], "script": "tests/_job_worker.py", "args": ["0"],
                 **overrides},
                tmp_path / "registry.jsonl")


def test_host_pools_are_not_unioned_and_legacy_queue_stays_local(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(yaml.safe_dump({"limits": {LOCAL: {"max_jobs": 1, "max_mem_gb": 20}}, "jobs": []}))
    loaded = jobs.load_queue(path)
    assert loaded["hosts"]["local"]["disk_floor_gb"] == 300
    assert "remote" not in loaded["hosts"]
    for queued in (job(gpu=LOCAL), job(host="local", gpu=REMOTE), job(host="elsewhere")):
        with pytest.raises(ValueError, match="host|non-authorized"):
            jobs.load_queue(write_queue(tmp_path, [queued]))
    with pytest.raises(ValueError, match="non-authorized"):
        jobs.load_queue(write_queue(tmp_path, [], limits={LOCAL: {"max_jobs": 1, "max_mem_gb": 20}}))
    queued = job()
    del queued["disk_gb"]
    with pytest.raises(ValueError, match="reservations"):
        jobs.load_queue(write_queue(tmp_path, [queued]))
    for required in (None, [], ["exit.json", "provenance.json"], ["../metrics.jsonl"], ["replay/buffer"], ["*.json"]):
        with pytest.raises(ValueError, match="required_results"):
            jobs.load_queue(write_queue(tmp_path, [job(required_results=required)]))


def test_remote_outage_preserves_jobs_and_local_admission(tmp_path, monkeypatch):
    queued = write_queue(tmp_path, [job(), job("local", "local")])
    backend = FakeBackend(tmp_path / "runs")
    backend.snapshot_error = remote.RemoteUnreachable("unreachable")
    launch_event(tmp_path)
    monkeypatch.setattr(jobs, "launch", lambda *args: 2**30)
    monkeypatch.setattr(jobs, "pid_alive", lambda _: pytest.fail("remote PID must not be inspected locally"))
    monkeypatch.setattr(jobs.os, "killpg", lambda *args: pytest.fail("remote job must not be signalled locally"))
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    assert state["r"]["status"] == "running" and state["r"]["unknown"]
    assert state["local"]["status"] == "running" and state["local"]["host"] == "local"
    assert state[jobs.HOST_EVENT_ID]["status"] == "unreachable" and not backend.launches


def test_launch_response_loss_reconciles_same_container_once(tmp_path):
    queued = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    backend.launch_error = remote.RemoteUnreachable("response lost")
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    assert state["r"]["status"] == "launching" and state["r"]["unknown"]
    assert len(backend.launches) == 1
    backend.launch_error = None
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    assert state["r"]["status"] == "running" and not state["r"]["unknown"]
    assert state["r"]["attempts"] == 1 and len(backend.launches) == 1
    assert state["r"]["gpu_backend"] == backend.launches[0]["gpu_backend"] == "cdi"
    events = [json.loads(line) for line in (tmp_path / "registry.jsonl").read_text().splitlines()]
    reconnected = [event for event in events if event["event"] == "host_reconnected"]
    assert len(reconnected) == 1 and reconnected[0]["duration_seconds"] >= 0


@pytest.mark.parametrize("status", ["running", "exited"])
@pytest.mark.parametrize("field", [
    "privileged", "missing_privileged", "writable_root", "missing_root_policy", "pid_namespace",
    "missing_pid_namespace", "small_shm", "missing_shm", "missing_mount", "extra_mount", "duplicate_mount",
    "destination", "writable_code", "writable_data", "readonly_runs", "wrong_release", "wrong_cache",
    "wrong_acceptance", "wrong_proof", "named_volume", "missing_mount_permissions",
])
def test_lost_response_rejects_incomplete_runtime_isolation(tmp_path, status, field):
    queued = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    backend.launch_error = remote.RemoteUnreachable("response lost")
    tick(tmp_path, queued, backend)
    registry = tmp_path / "registry.jsonl"
    state = jobs.read_registry(registry)["r"]
    item = backend.value.containers[state["container"]]
    item["State"]["Status"] = status
    remote.validate_container(item, "r", state)
    config, mounts = item["HostConfig"], item["Mounts"]
    if field == "privileged":
        config["Privileged"] = True
    elif field == "missing_privileged":
        config.pop("Privileged")
    elif field == "writable_root":
        config["ReadonlyRootfs"] = False
    elif field == "missing_root_policy":
        config.pop("ReadonlyRootfs")
    elif field == "pid_namespace":
        config["PidMode"] = ""
    elif field == "missing_pid_namespace":
        config.pop("PidMode")
    elif field == "small_shm":
        config["ShmSize"] = 64 * 2**20
    elif field == "missing_shm":
        config.pop("ShmSize")
    elif field == "missing_mount":
        mounts.pop()
    elif field == "extra_mount":
        mounts.append({"Type": "bind", "Source": remote.ROOT + "/extra", "Destination": "/extra", "RW": True})
    elif field == "duplicate_mount":
        mounts[-1] = dict(mounts[0])
    elif field == "destination":
        mounts[0]["Destination"] = "/workspace/other"
    elif field == "writable_code":
        mounts[0]["RW"] = True
    elif field == "writable_data":
        mounts[-1]["RW"] = True
    elif field == "readonly_runs":
        mounts[1]["RW"] = False
    elif field == "wrong_release":
        mounts[0]["Source"] = remote.ROOT + "/releases/" + "9" * 40 + "/code"
    elif field == "wrong_cache":
        mounts[3]["Source"] = remote.ROOT + "/runtime/other-cache"
    elif field == "wrong_acceptance":
        mounts[5]["Source"] = remote.ROOT + "/runtime/acceptance/other"
    elif field == "wrong_proof":
        mounts[6]["Source"] = remote.ROOT + "/releases/other/provenance.json"
    elif field == "named_volume":
        mounts[1]["Type"] = "volume"
    else:
        mounts[0].pop("RW")
    with pytest.raises(remote.RemoteJobError, match="mismatch"):
        remote.validate_container(item, "r", state)
    for _ in range(2):
        tick(tmp_path, queued, backend)
    current = jobs.read_registry(registry)["r"]
    assert current["status"] == "launching" and current["unknown"]
    assert current["attempts"] == 1 and current["failures"] == 0
    assert len(backend.launches) == 1 and not backend.syncs


def test_missing_container_never_becomes_failed_or_relaunched(tmp_path):
    queued = write_queue(tmp_path, [job(max_restarts=0)])
    backend = FakeBackend(tmp_path / "runs")
    launch_event(tmp_path, kind="launching")
    for _ in range(2):
        tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert state["status"] == "launching" and state["unknown"] and state["attempts"] == 1
    assert not backend.launches


def test_remote_success_waits_for_checksum_sync_and_exit_identity(tmp_path, monkeypatch):
    queued = write_queue(tmp_path, [job(), job("dependent", deps=["r"])])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    backend.sync_error = remote.RemoteJobError("checksum mismatch")
    launch_event(tmp_path)
    monkeypatch.setattr(jobs, "pid_alive", lambda _: pytest.fail("remote PID checked"))
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    assert state["r"]["status"] == "sync_pending" and not backend.launches
    backend.sync_error = None
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    assert state["r"]["status"] == "done" and state["r"]["sync"]["final"]
    assert [payload["id"] for payload in backend.launches] == ["dependent"]


def test_container_code_zero_without_matching_exit_file_blocks_dependencies(tmp_path, monkeypatch):
    queued = write_queue(tmp_path, [job(), job("dependent", deps=["r"])])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    original = backend.sync_results

    def mismatched(run_id, final=False):
        result = original(run_id, final)
        path = backend.runs / run_id / "exit.json"
        report = json.loads(path.read_text())
        report["attempt"] = 0
        payload = json.dumps(report).encode()
        path.write_bytes(payload)
        result["files"]["exit.json"] = hashlib.sha256(payload).hexdigest()
        return result

    monkeypatch.setattr(backend, "sync_results", mismatched)
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["status"] == "sync_pending"
    assert not backend.launches


def test_wrapper_only_results_do_not_unblock_dependencies(tmp_path, monkeypatch):
    queued = write_queue(tmp_path, [job(), job("dependent", deps=["r"])])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    original = backend.sync_results

    def wrapper_only(run_id, final=False):
        result = original(run_id, final)
        result["files"].pop("metrics.jsonl")
        (backend.runs / run_id / "metrics.jsonl").unlink()
        return result

    monkeypatch.setattr(backend, "sync_results", wrapper_only)
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["status"] == "sync_pending"
    assert not backend.launches


@pytest.mark.parametrize("changed", ["image_digest", "data_id", "arguments", "config_sha256"])
def test_final_numerical_provenance_must_match_declared_identity(tmp_path, monkeypatch, changed):
    queued = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    original = backend.sync_results

    def altered(run_id, final=False):
        result = original(run_id, final)
        path = backend.runs / run_id / "provenance.json"
        report = json.loads(path.read_text())
        report.update(numeric_config={"train": {"seed": 0, "steps": 10}}, config_sha256="0" * 64)
        report[changed] = ["other"] if changed == "arguments" else "not-the-declared-identity"
        payload = json.dumps(report).encode()
        path.write_bytes(payload)
        result["files"]["provenance.json"] = hashlib.sha256(payload).hexdigest()
        return result

    monkeypatch.setattr(backend, "sync_results", altered)
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["status"] == "sync_pending"


def test_explicit_checkpoint_requirement_requests_checkpoint_transfer(tmp_path, monkeypatch):
    required = ["metrics.jsonl", "checkpoints/final.pt"]
    queued = write_queue(tmp_path, [job(required_results=required)])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path, required_results=required)
    original, flags = backend.sync_results, []

    def transfer(run_id, final=False, checkpoints=False):
        flags.append(checkpoints)
        result = original(run_id, final)
        if checkpoints:
            path = backend.runs / run_id / "checkpoints/final.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = b"checksum-transfer-fixture"
            path.write_bytes(payload)
            result["files"]["checkpoints/final.pt"] = hashlib.sha256(payload).hexdigest()
        return result

    monkeypatch.setattr(backend, "sync_results", transfer)
    tick(tmp_path, queued, backend)
    assert flags == [True] and jobs.read_registry(tmp_path / "registry.jsonl")["r"]["status"] == "done"


def test_nonzero_exit_is_failed_only_after_observation(tmp_path):
    queued = write_queue(tmp_path, [job(max_restarts=0)])
    backend = FakeBackend(tmp_path / "runs")
    launch_event(tmp_path)
    backend.value.containers["s4d-r-a1"] = inspected(status="exited", code=7)
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["status"] == "failed"
    assert not backend.syncs and not backend.launches


def test_unknown_remote_never_consumes_local_pid_or_exit_file(tmp_path, monkeypatch):
    queued = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    launch_event(tmp_path, pid=123)
    (tmp_path / "runs/r").mkdir(parents=True)
    (tmp_path / "runs/r/exit_code").write_text("9")
    monkeypatch.setattr(jobs, "pid_alive", lambda _: pytest.fail("remote local PID inspected"))
    monkeypatch.setattr(jobs.os, "killpg", lambda *args: pytest.fail("remote local group signalled"))
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["status"] == "running"


@pytest.mark.parametrize("available,free,launches", [(49, 800, 0), (200, 169, 0), (200, 170, 1)])
def test_remote_ram_and_fifteen_percent_disk_floor(tmp_path, available, free, launches):
    queued = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.available_gb, backend.value.disk_free_bytes = available, free * GIB
    tick(tmp_path, queued, backend)
    assert len(backend.launches) == launches


def test_reserved_future_disk_and_ram_ramp_are_per_host(tmp_path):
    queued = write_queue(tmp_path, [job("first"), job("second")])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.disk_free_bytes = 185 * GIB
    tick(tmp_path, queued, backend)
    assert [payload["id"] for payload in backend.launches] == ["first"]
    backend.value.available_gb = 55
    backend.value.disk_free_bytes = 800 * GIB
    tick(tmp_path, queued, backend)
    assert [payload["id"] for payload in backend.launches] == ["first"]


def test_local_disk_floor_cannot_be_lowered(tmp_path, monkeypatch):
    queued = write_queue(tmp_path, [job("local", "local")])
    payload = yaml.safe_load(queued.read_text())
    payload["hosts"]["local"]["disk_floor_gb"] = 0
    queued.write_text(yaml.safe_dump(payload))
    launched = []
    monkeypatch.setattr(jobs, "launch", lambda *args: launched.append(args))
    jobs.tick(queued, tmp_path / "registry.jsonl", tmp_path / "runs", usage={}, available_gb=200,
              disk_bytes=(2000 * GIB, 319 * GIB), remote_backend=FakeBackend(tmp_path / "runs"))
    assert not launched


@pytest.mark.parametrize("failure", [
    subprocess.CalledProcessError(255, ["nvidia-smi"]),
    subprocess.TimeoutExpired(["nvidia-smi"], 60),
    QueryUnavailable(["nvidia-smi"], 60),
])
@pytest.mark.parametrize("remote_status", ["running", "exited"])
def test_local_nvml_failure_does_not_block_remote(tmp_path, monkeypatch, failure, remote_status):
    dependencies = ["active"] if remote_status == "exited" else []
    queued = write_queue(tmp_path, [job("active"), job(deps=dependencies), job("local", "local")])
    backend = FakeBackend(tmp_path / "runs")
    launch_event(tmp_path, "active")
    backend.value.containers["s4d-active-a1"] = inspected("active", status=remote_status)
    if remote_status == "running":
        backend.value.usage[REMOTE] = [(123, 5)]

    def query(command, state_dir, timeout):
        assert command == ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]
        assert state_dir == REPO / "experiments" and timeout == 60
        raise failure

    monkeypatch.setattr(jobs, "query_output", query)
    monkeypatch.setattr(jobs, "launch", lambda *args: pytest.fail("failed local GPU query admitted a local job"))
    messages = jobs.tick(queued, tmp_path / "registry.jsonl", tmp_path / "runs", available_gb=200,
                         disk_bytes=(2000 * GIB, 1000 * GIB), remote_backend=backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    assert any("GPU process query failed" in message and "on local" in message for message in messages)
    assert state["active"]["status"] == ("done" if remote_status == "exited" else "running")
    assert ("active", remote_status == "exited") in backend.syncs
    assert [payload["id"] for payload in backend.launches] == ["r"]
    assert "local" not in state


def test_disabled_unverified_and_foreign_remote_cannot_launch(tmp_path):
    backend = FakeBackend(tmp_path / "runs")
    for overrides in ({"enabled": False}, {"hold": True}):
        tick(tmp_path, write_queue(tmp_path, [job()], **overrides), backend)
        assert not backend.launches
    queued = write_queue(tmp_path, [job()])
    backend.ready_error = remote.RemoteJobError("not accepted")
    tick(tmp_path, queued, backend)
    assert not backend.launches
    backend.ready_error = None
    backend.value.usage[REMOTE] = [(999, 1)]
    tick(tmp_path, queued, backend)
    assert not backend.launches
    backend.value.usage[REMOTE] = []
    backend.value.resources_ready = False
    tick(tmp_path, queued, backend)
    assert not backend.launches


def test_existing_job_cannot_be_silently_migrated(tmp_path):
    queued = write_queue(tmp_path, [job("r", "local")])
    backend = FakeBackend(tmp_path / "runs")
    launch_event(tmp_path)
    with pytest.raises(ValueError, match="cannot be migrated"):
        tick(tmp_path, queued, backend)


def snapshot_output(containers=(), graphics="", query_error=False):
    rows = ["__S4D_GPUS__", *(f"{index}, {uuid}" for index, uuid in enumerate(APPROVED_HOST_GPUS["remote"])),
            "__S4D_USAGE__", "__S4D_QUERY_ERROR__" if query_error else f"{REMOTE}, 123, 17",
            "__S4D_GRAPHICS__", graphics, "__S4D_MEMORY__", "MemAvailable: 209715200 kB",
            "__S4D_DISK__", "1B-blocks Avail", f"{1000 * GIB} {800 * GIB}", "__S4D_CONTAINERS__"]
    for item in containers:
        rows.extend([json.dumps({key: value for key, value in item.items() if key != "pids"}),
                     f"__S4D_TOP__ {item['Id']}", "PID", "123", "__S4D_END_TOP__"])
    return "\n".join([*rows, "__S4D_END__"]) + "\n"


def test_snapshot_parses_resources_graphics_and_host_pids():
    value = remote.parse_snapshot(snapshot_output([inspected()], "| 0 N/A N/A 456 G graphics 9MiB |"))
    assert value.resources_ready and value.available_gb == 200
    assert value.disk_free_bytes == 800 * GIB
    assert value.usage[REMOTE] == [(123, 17), (456, 0)]
    assert value.containers["s4d-r-a1"]["pids"] == {123}
    assert not remote.parse_snapshot(snapshot_output(query_error=True)).resources_ready
    with pytest.raises(remote.RemoteJobError, match="Incomplete"):
        remote.parse_snapshot("__S4D_MEMORY__\nMemAvailable: 10 kB")
    command = remote.snapshot_command(remote.ROOT)
    assert "python" not in command and "docker inspect" in command and "nvidia-smi" in command
    with pytest.raises(ValueError, match="authorized"):
        remote.snapshot_command("/tmp")


@pytest.mark.parametrize("field", [
    "image", "label", "env", "devices", "mount", "missing_backend", "wrong_backend", "qualified_uuid",
    "extra_request", "ordinary_device", "driver", "count", "visible", "capabilities", "options",
])
def test_container_identity_and_isolation_fail_closed(field):
    item = inspected()
    state = {"container": "s4d-r-a1", "container_id": CONTAINER_ID, "attempts": 1,
             "commit": COMMIT, "image_digest": IMAGE, "gpu": REMOTE, "gpu_backend": "cdi"}
    remote.validate_container(item, "r", state)
    if field == "image":
        item["Image"] = "sha256:" + "9" * 64
    elif field == "label":
        item["Config"]["Labels"]["s4d.commit"] = "9" * 40
    elif field == "env":
        item["Config"]["Env"][0] = f"CUDA_VISIBLE_DEVICES={LOCAL}"
    elif field == "devices":
        item["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = [REMOTE, LOCAL]
    elif field == "mount":
        item["Mounts"][0]["Source"] = "/home/compu/.ssh"
    elif field == "missing_backend":
        state.pop("gpu_backend")
    elif field == "wrong_backend":
        state["gpu_backend"] = "legacy"
    elif field == "qualified_uuid":
        item["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = [f"nvidia.com/gpu={LOCAL}"]
    elif field == "extra_request":
        item["HostConfig"]["DeviceRequests"].append(dict(item["HostConfig"]["DeviceRequests"][0]))
    elif field == "ordinary_device":
        item["HostConfig"]["Devices"] = [{"PathOnHost": "/dev/nvidia0", "PathInContainer": "/dev/nvidia0"}]
    elif field == "driver":
        item["HostConfig"]["DeviceRequests"][0]["Driver"] = "nvidia"
    elif field == "count":
        item["HostConfig"]["DeviceRequests"][0]["Count"] = -1
    elif field == "visible":
        item["Config"]["Env"][2] = "NVIDIA_VISIBLE_DEVICES=all"
    elif field == "capabilities":
        item["HostConfig"]["DeviceRequests"][0]["Capabilities"] = [["gpu"]]
    else:
        item["HostConfig"]["DeviceRequests"][0]["Options"] = {"device": "all"}
    with pytest.raises(remote.RemoteJobError, match="mismatch|outside"):
        remote.validate_container(item, "r", state)


def test_legacy_injection_requires_explicit_legacy_identity():
    item = inspected(gpu_backend="legacy")
    state = {"container": "s4d-r-a1", "container_id": CONTAINER_ID, "attempts": 1,
             "commit": COMMIT, "image_digest": IMAGE, "gpu": REMOTE, "gpu_backend": "legacy"}
    remote.validate_container(item, "r", state)
    state["gpu_backend"] = "cdi"
    with pytest.raises(remote.RemoteJobError, match="mismatch"):
        remote.validate_container(item, "r", state)


def test_audited_remote_failure_never_reuses_container_attempt(tmp_path):
    queued = write_queue(tmp_path, [job(max_restarts=0)])
    backend = FakeBackend(tmp_path / "runs")
    launch_event(tmp_path)
    backend.value.containers["s4d-r-a1"] = inspected(status="exited", code=1)
    tick(tmp_path, queued, backend)
    registry = tmp_path / "registry.jsonl"
    jobs.record({"event": "infrastructure_failure", "id": "r", "attempt": 1,
                 "reason": "cuda_init_unavailable", "incident_started": 0, "incident_ended": None}, registry)
    tick(tmp_path, queued, backend)
    current = jobs.read_registry(registry)["r"]
    assert [payload["container"] for payload in backend.launches] == ["s4d-r-a2"]
    assert current["attempts"] == 2 and current["failures"] == 0 and current["gpu_backend"] == "cdi"
    backend.value.containers["s4d-r-a2"]["State"] = {"Status": "exited", "ExitCode": 7}
    tick(tmp_path, queued, backend)
    current = jobs.read_registry(registry)["r"]
    assert current["attempts"] == 2 and current["failures"] == 1 and current["status"] == "failed"
    assert len(backend.launches) == 1


def evidence_files(tmp_path):
    gitdir = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--absolute-git-dir"],
                            capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / ".git").write_text("gitdir: " + gitdir + "\n")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    source = remote.commit_source_sha256(tmp_path, commit)

    def artifact(name, report):
        path = tmp_path / (name + ".json")
        payload = json.dumps({"commit": commit, "image_digest": IMAGE, "report": report}).encode()
        path.write_bytes(payload)
        return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}

    suite = {"host": "remote", "full_suite_passed": True, "exit_code": 0, "source_sha256": source,
             "source_sha256_before": source, "source_changed_during_suite": False,
             "counts": {"passed": 1, "failed": 0, "error": 0, "skipped": 0}}
    isolation = {}
    for index, uuid in enumerate(APPROVED_HOST_GPUS["remote"]):
        report = {"host": "remote", "passed": True, "source_sha256": source,
                  "source_changed_during_check": False, "parent_mapping": [{"uuid": uuid}],
                  "results": [{"uuid": uuid, "passed": True, "gpus_seen": [uuid],
                               "child": {"mapping": [{"uuid": uuid}], "native_cuda_passed": True,
                                         "native_egl_passed": True}},
                              *({"cuda_visible_devices": invalid, "rejected": True}
                                for invalid in ("0", None, LOCAL))]}
        isolation[uuid] = artifact("isolation" + str(index), report)
    manifest = {"commit": commit, "image_digest": IMAGE, "full_suite": artifact("suite", suite),
                "gpu_isolation": isolation,
                "data": artifact("data", {"identity_sha256": DATA, "files": [{"sha256": "0" * 64}]}),
                "cross_host": artifact("cross", {"passed": True, "step0_identical": True,
                                                "short_run_within_noise": True, "data_identity": DATA,
                                                "protocol": "fixed-seed step0 + short run"})}
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps(manifest))
    return {"commit": commit, "image_digest": IMAGE, "evidence_manifest": str(path)}, manifest


def test_acceptance_requires_hashed_native_isolation_data_and_equivalence(tmp_path, monkeypatch):
    host, manifest = evidence_files(tmp_path)
    accepted = remote.acceptance_manifest(host, tmp_path)
    assert accepted["data_identity"] == DATA and len(accepted["manifest_sha256"]) == 64
    path = Path(manifest["cross_host"]["path"])
    path.write_text(path.read_text() + " ")
    with pytest.raises(remote.RemoteJobError, match="checksum"):
        remote.acceptance_manifest(host, tmp_path)


@pytest.mark.parametrize("part", ["full_suite", "gpu_isolation", "data", "cross_host"])
def test_acceptance_missing_any_gate_holds_admission(tmp_path, monkeypatch, part):
    host, manifest = evidence_files(tmp_path)
    manifest.pop(part)
    Path(host["evidence_manifest"]).write_text(json.dumps(manifest))
    with pytest.raises(remote.RemoteJobError):
        remote.acceptance_manifest(host, tmp_path)


def test_suite_source_identity_is_compared_to_commit(tmp_path, monkeypatch):
    host, _ = evidence_files(tmp_path)
    monkeypatch.setattr(remote, "commit_source_sha256", lambda *_: {"other": "0" * 64})
    with pytest.raises(remote.RemoteJobError, match="source fingerprints"):
        remote.acceptance_manifest(host, tmp_path)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    monkeypatch.undo()
    actual = remote.commit_source_sha256(REPO, commit)
    data = subprocess.run(["git", "-C", str(REPO), "show", commit + ":s4d/gpu_guard.py"],
                          capture_output=True, check=True).stdout
    assert actual["s4d/gpu_guard.py"] == hashlib.sha256(data).hexdigest()


def test_commit_and_local_identity_match_actual_bootstrap_fingerprint_scope(tmp_path, monkeypatch):
    bootstrap_spec = importlib.util.spec_from_file_location("fingerprint_bootstrap", REPO / "scripts/_bootstrap.py")
    bootstrap = importlib.util.module_from_spec(bootstrap_spec)
    bootstrap_spec.loader.exec_module(bootstrap)
    actual = bootstrap.source_fingerprints()
    assert jobs.source_identity()["source_sha256"] == actual
    assert actual["mujoco_mig_setup.py"] == hashlib.sha256((REPO / "mujoco_mig_setup.py").read_bytes()).hexdigest()
    gitdir = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--absolute-git-dir"],
                            capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / ".git").write_text("gitdir: " + gitdir + "\n")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    committed = remote.commit_source_sha256(tmp_path, commit)
    for name in committed:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = subprocess.run(["git", "-C", str(REPO), "show", commit + ":" + name],
                                 capture_output=True, check=True).stdout
        path.write_bytes(payload)
    monkeypatch.setattr(bootstrap, "REPO", tmp_path)
    assert committed == bootstrap.source_fingerprints()
    assert "mujoco_mig_setup.py" in committed


def test_launch_spec_is_safe_immutable_and_exact_uuid(tmp_path):
    backend = remote.RemoteBackend(REPO, tmp_path / "runs", FakeClient())
    host = {"commit": COMMIT, "image_digest": IMAGE, "disk_floor_gb": 100,
            "limits": {REMOTE: {"max_jobs": 2, "max_mem_gb": 40}}}
    evidence = {"data_identity": DATA, "manifest_sha256": "f" * 64}
    path, payload = backend.prepare(job(args=["a; $(false)"]), REMOTE, 1, host, evidence)
    assert payload["gpu"] == REMOTE and payload["args"] == ["a; $(false)"]
    assert payload["ram_gb"] == 10 and payload["disk_gb"] == 20 and payload["container"] == "s4d-r-a1"
    assert backend.prepare(job(args=["a; $(false)"]), REMOTE, 1, host, evidence)[0] == path
    with pytest.raises(remote.RemoteJobError, match="changed"):
        backend.prepare(job(args=[1]), REMOTE, 1, host, evidence)
    with pytest.raises(ValueError, match="authorized"):
        backend.prepare(job(), LOCAL, 1, host, evidence)
    with pytest.raises(ValueError, match="relative"):
        backend.prepare(job(script="../outside.py"), REMOTE, 2, host, evidence)
    with pytest.raises(ValueError, match="identity"):
        remote.container_name("../escape", 1)


def test_backend_snapshot_persists_reachability_without_discarding_last_state(tmp_path):
    class Client(FakeClient):
        error = False

        def ssh(self, command, **kwargs):
            assert kwargs == {"wait": False, "idempotent": True, "timeout": 60}
            if self.error:
                raise remote.RemoteUnreachable("lost")
            return subprocess.CompletedProcess([], 0, snapshot_output([inspected()]), "")

    client = Client()
    backend = remote.RemoteBackend(REPO, tmp_path / "runs", client)
    backend.snapshot()
    path = tmp_path / "runs/remote/host_snapshot.json"
    previous = json.loads(path.read_text())
    assert previous["reachable"] and previous["query_ok"] and previous["containers"]["s4d-r-a1"]["pids"] == [123]
    client.error = True
    with pytest.raises(remote.RemoteUnreachable):
        backend.snapshot()
    current = json.loads(path.read_text())
    assert not current["reachable"] and not current["query_ok"] and current["containers"] == previous["containers"]
    assert current["checked_at"] >= previous["checked_at"]


def test_backend_launcher_response_unknown_is_not_a_second_launch(tmp_path, monkeypatch):
    backend = remote.RemoteBackend(REPO, tmp_path / "runs", FakeClient())
    monkeypatch.setattr(remote.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 75, "", ""))
    with pytest.raises(remote.RemoteUnreachable):
        backend.launch(tmp_path / "job.json")
