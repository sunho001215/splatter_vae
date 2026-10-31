from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from ops.remote import artifact_check as remote_artifact_check
from ops.jobs import remote_backend as remote
from ops.resources.gpu_guard import APPROVED_HOST_GPUS
from ops.resources.nvidia_query import QueryUnavailable

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("remote_scheduler", REPO / "ops/jobs/scheduler.py")
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
        self.launches, self.syncs, self.scopes = [], [], []
        self.snapshot_error = self.launch_error = self.sync_error = self.ready_error = self.remove_error = None
        # 2026-10-27: the tick verifies finished runs in place on the remote (verify_remote_artifacts) and completes them
        # from the bulk worker's local backup record (backup_record). The fake emulates both from the same files its
        # sync_results writes, so tampering with those results fails both checks; backup_ready=False models a backup
        # that the worker has not made yet.
        self.backup_ready, self.inspections, self.backups = True, [], []
        self.removals = []
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

    def remove_container(self, run_id, attempt, container, container_id, expected_code, required):
        self.removals.append((run_id, attempt, container, container_id, expected_code, tuple(required)))
        if self.remove_error:
            raise self.remove_error
        if self.value.containers.pop(container, None) is None:
            return {"absent": True}
        return {"removed": True, "log_sha256": "0" * 64, "inspect_sha256": "1" * 64}

    def sync_results(self, run_id, final=False, checkpoints=False, scope="all", only_checkpoints=None):
        self.syncs.append((run_id, final))
        self.scopes.append(scope)
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

    def verify_remote_artifacts(self, run_id, names):
        self.inspections.append(run_id)
        if self.sync_error:
            raise self.sync_error
        self.sync_results(run_id, True)
        report = remote_artifact_check.inspect_run(str(self.runs / run_id), names)
        report["root"] = f"{remote.ROOT}/runs/{run_id}"
        return report

    def backup_record(self, run_id):
        if not self.backup_ready or self.sync_error:
            return None
        self.backups.append(run_id)
        result = self.sync_results(run_id, True)
        return {**result, "scope": result.get("scope", "essential")}


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
    monkeypatch.setattr(jobs, "REMOTE_RECHECK_SECONDS", 0)  # the failed check is retried on the next tick
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

    def mismatched(run_id, final=False, scope="all"):
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

    def wrapper_only(run_id, final=False, scope="all"):
        result = original(run_id, final)
        result["files"].pop("metrics.jsonl")
        (backend.runs / run_id / "metrics.jsonl").unlink()
        return result

    monkeypatch.setattr(backend, "sync_results", wrapper_only)
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["status"] == "sync_pending"
    assert not backend.launches


@pytest.mark.parametrize("script", ["train_snerl.py", "train_dynamo.py", "train_mvp3d.py"])
def test_new_baseline_completion_requires_numeric_configuration(tmp_path, script):
    entry = f"scripts/{script}"
    queued = write_queue(tmp_path, [job(script=entry)])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path, script=entry)
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["status"] == "sync_pending"


@pytest.mark.parametrize("changed", ["image_digest", "data_id", "arguments", "config_sha256"])
def test_final_numerical_provenance_must_match_declared_identity(tmp_path, monkeypatch, changed):
    queued = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    original = backend.sync_results

    def altered(run_id, final=False, scope="all"):
        result = original(run_id, final, scope=scope)
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


def test_required_checkpoint_is_verified_in_place_and_completed_from_the_backup(tmp_path, monkeypatch):
    """The tick downloads nothing: the required checkpoint's archive is inspected on the remote (integrity, encoder
    tensors), and the job completes from the worker's backup record, which must carry the same checkpoint checksum."""
    import torch

    required = ["metrics.jsonl", "checkpoints/final.pt"]
    queued = write_queue(tmp_path, [job(required_results=required)])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path, required_results=required)
    original = backend.sync_results

    def transfer(run_id, final=False, checkpoints=False, scope="all", only_checkpoints=None):
        result = original(run_id, final)
        path = backend.runs / run_id / "checkpoints/final.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            torch.save({"step": 7, "encoder": {"w": torch.zeros(2, 3)}}, path)
        result["files"]["checkpoints/final.pt"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    monkeypatch.setattr(backend, "sync_results", transfer)
    backend.backup_ready = False
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert state["status"] == "sync_pending" and backend.inspections == ["r"] and not backend.backups
    assert state["remote_verified"]["checks"]["checkpoints"] == {"checkpoints/final.pt": {
        "step": 7, "encoder_tensors": 1, "encoder_signature_sha256": jobs.signature_sha256({
            "w": {"dtype": "FloatStorage", "shape": [2, 3]}})}}
    backend.backup_ready = True
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert state["status"] == "done" and backend.inspections == ["r"] and backend.backups == ["r"]
    assert state["sync"]["files"]["checkpoints/final.pt"] == state["remote_verified"]["files"]["checkpoints/final.pt"]


def test_backup_with_different_checksums_than_the_remote_verification_never_completes(tmp_path, monkeypatch):
    queued = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    backend.backup_ready = False
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["remote_verified"]["attempt"] == 1
    backend.backup_ready = True
    original = backend.backup_record
    monkeypatch.setattr(backend, "backup_record", lambda run_id: {
        **original(run_id), "files": {**original(run_id)["files"], "metrics.jsonl": "0" * 64}})
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["status"] == "sync_pending"


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

    def query(devices, state_dir, timeout):
        assert devices is jobs.MIG_DEVICES
        assert state_dir == REPO / "experiments" and timeout == 180
        raise failure

    monkeypatch.setattr(jobs, "mig_query", query)
    monkeypatch.setattr(jobs, "launch", lambda *args: pytest.fail("failed local GPU query admitted a local job"))
    messages = jobs.tick(queued, tmp_path / "registry.jsonl", tmp_path / "runs", available_gb=200,
                         disk_bytes=(2000 * GIB, 1000 * GIB), remote_backend=backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    assert any("GPU process query failed" in message and "on local" in message for message in messages)
    assert state["active"]["status"] == ("done" if remote_status == "exited" else "running")
    # nothing is downloaded inside the tick; a finished run is inspected in place and completed from its backup record
    assert not [sync for sync in backend.syncs if sync[1] is False]
    assert backend.inspections == (["active"] if remote_status == "exited" else [])
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
    for name, identity in committed.items():
        if name.endswith("@commit"):
            module = name.removesuffix("@commit")
            path = tmp_path / module
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--shared", "--no-checkout", "--quiet", str(REPO / module), str(path)],
                           capture_output=True, check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--quiet", "--detach", identity],
                           capture_output=True, check=True)
            continue
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


def test_refused_launcher_preserves_bounded_redacted_diagnostics(tmp_path, monkeypatch):
    class SecretClient(FakeClient):
        def _redact(self, text):
            return text.replace("test-password", "[REDACTED]")

    backend = remote.RemoteBackend(REPO, tmp_path / "runs", SecretClient())
    monkeypatch.setattr(remote.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        [], 1, "x" * 9000 + "test-password", "GPU admission refused test-password"))
    path = tmp_path / "job.json"
    with pytest.raises(remote.RemoteJobError, match="requires reconciliation"):
        backend.launch(path)
    diagnostic = json.loads(path.with_suffix(".launch-error.json").read_text())
    assert diagnostic["returncode"] == 1 and len(diagnostic["stdout"]) == 8000
    assert diagnostic["stdout"].endswith("[REDACTED]")
    assert diagnostic["stderr"] == "GPU admission refused [REDACTED]"
    assert "test-password" not in json.dumps(diagnostic)


def test_backend_launcher_response_unknown_is_not_a_second_launch(tmp_path, monkeypatch):
    backend = remote.RemoteBackend(REPO, tmp_path / "runs", FakeClient())
    monkeypatch.setattr(remote.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 75, "", ""))
    with pytest.raises(remote.RemoteUnreachable):
        backend.launch(tmp_path / "job.json")

def _rv_setup(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace

    from ops.jobs import remote_backend as remote_jobs
    from ops.resources.gpu_guard import APPROVED_HOST_GPUS
    from ops.containers.verification import POLICIES, ROOT

    gpu, commit, image = next(iter(APPROVED_HOST_GPUS["remote"])), "a" * 40, "sha256:" + "b" * 64
    source = {"s4d/example.py": "c" * 64, "third_party/example": "d" * 64}
    monkeypatch.setattr(remote_jobs, "commit_source_sha256", lambda *_: source)
    files = [{"target": f"{ROOT}/data/metaworld/splatter4d_v1/{relative}", "sha256": "e" * 64, "bytes": 1}
             for task in ("hammer", "pick-place")
             for relative in (f"{task}.hdf5", f"heldout_sets/{task}.hdf5",
                              f"splits/{task}_seed0.json", f"workspace_stats/{task}.json")]
    normalized = [{**item, "target": item["target"][len(ROOT) + 1:]} for item in files]
    normalized.sort(key=lambda item: item["target"])
    identity = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
    binding = tmp_path / "data.json"
    binding.write_text(json.dumps({"commit": commit, "image_digest": image,
                                   "report": {"identity_sha256": identity, "files": files}}))
    manifest = {"schema_version": 1, "verification_purpose": "native_suite", "campaign_eligible": False,
                "commit": commit, "image_digest": image, "source_sha256": source,
                "data": {"path": "data.json", "sha256": hashlib.sha256(binding.read_bytes()).hexdigest()}}
    manifest_path = tmp_path / "verification.json"
    manifest_path.write_text(json.dumps(manifest))
    evidence = {**manifest, "data_identity": identity,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    host = {"commit": commit, "image_digest": image, "gpu_backend": "cdi",
            "verification_manifests": {"native_suite": "verification.json"},
            "host_ram_reserve_gb": 120, "host_ram_ramp_minutes": 10,
            "limits": {gpu: {"max_jobs": 20, "max_mem_gb": 90}}}
    job = {"id": "verify-suite", "host": "remote", "gpu": gpu, "script": POLICIES["native_suite"][0],
           "args": [], "verification_purpose": "native_suite", "campaign_eligible": False,
           "ram_gb": 32, "mem_gb": 12, "disk_gb": 1,
           "required_results": ["evidence/" + name for name in POLICIES["native_suite"][4]]}
    script = tmp_path / job["script"]
    script.parent.mkdir(exist_ok=True)
    script.write_text("pass\n")  # the suite is allowed only through its typed entry
    backend = remote_jobs.RemoteBackend(tmp_path, tmp_path / "runs", SimpleNamespace(root=ROOT))
    return backend, host, job, evidence


def _rv_inspect(spec, identity):
    from ops.containers.verification import ROOT

    runtime = ROOT + "/runtime"
    acceptance = f"{runtime}/acceptance/{spec['commit']}-{spec['image_digest'][7:]}"
    proof = f"{ROOT}/releases/{spec['commit']}/provenance.json"
    repo = "/workspace/splatter4d"
    env = {"CUDA_VISIBLE_DEVICES": spec["gpu"], "NVIDIA_VISIBLE_DEVICES": "void",
           "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
           "S4D_JOB_ID": spec["id"], "S4D_ATTEMPT": str(spec["attempt"]),
           "S4D_EXPECTED_COMMIT": spec["commit"], "S4D_IMAGE_DIGEST": spec["image_digest"],
           "S4D_DATA_ID": identity, "S4D_TEST_EVIDENCE_DIR": spec["args"][1],
           "S4D_VERIFICATION_PURPOSE": "native_suite", "S4D_VERIFICATION_TIMEOUT_SECONDS": "7200",
           "S4D_VERIFICATION_RAM_BYTES": str(32 * 2**30), "S4D_VERIFICATION_GPU_BYTES": str(12 * 2**30),
           "S4D_VERIFICATION_DISK_BYTES": str(2**30), "S4D_VERIFICATION_CPU_THREADS": "8",
           "S4D_VERIFICATION_ACCEPTANCE_DIR": acceptance, "S4D_VERIFICATION_DATA_SHA256": spec["evidence"]["data"]["sha256"]}
    mounts = [(repo, f"{ROOT}/releases/{spec['commit']}/code", False),
              (repo + "/runs", ROOT + "/runs", True), (repo + "/outputs", ROOT + "/outputs", True),
              (repo + "/.cache", runtime + "/cache", True), (runtime, runtime, True),
              (acceptance, acceptance, False), (proof, proof, False),
              ("/home/ws/data/metaworld/splatter4d_v1", ROOT + "/data/metaworld/splatter4d_v1", False)]
    labels = {**spec["labels"], "s4d.ram_bytes": str(32 * 2**30),
              "s4d.mem_bytes": str(12 * 2**30), "s4d.disk_bytes": str(2**30), "s4d.gpu": spec["gpu"]}
    return {"Name": "/" + spec["container"], "Id": "f" * 64, "Image": spec["image_digest"], "pids": {123},
            "State": {"Status": "running", "ExitCode": 0},
            "Config": {"Labels": labels, "Env": [f"{key}={value}" for key, value in env.items()],
                       "Entrypoint": ["python"],
                       "Cmd": ["-I", repo + "/scripts/remote/entry.py", "--script", spec["script"], "--", *spec["args"]]},
            "HostConfig": {"Privileged": False, "ReadonlyRootfs": True, "PidMode": "host", "ShmSize": 8 * 2**30,
                           "Memory": 32 * 2**30, "MemorySwap": 32 * 2**30, "NanoCpus": 8 * 10**9,
                           "Devices": [], "DeviceRequests": [{"Driver": "cdi", "Count": 0,
                               "DeviceIDs": ["nvidia.com/gpu=" + spec["gpu"]], "Capabilities": [], "Options": {}}]},
            "Mounts": [{"Destination": target, "Source": source, "RW": rw, "Type": "bind"}
                       for target, source, rw in mounts]}


def test_verification_ready_without_campaign_acceptance(tmp_path, monkeypatch):
    import subprocess

    from ops.jobs import remote_backend as remote_jobs

    backend, host, job, evidence = _rv_setup(tmp_path, monkeypatch)
    calls = []
    def pushed(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, host["commit"] + "\trefs/heads/splatter4d\n", "")
    monkeypatch.setattr(remote_jobs.subprocess, "run", pushed)
    assert backend.ready_verification(host, job) == evidence
    assert len(calls) == 2 and "ls-remote" in calls[0] and "merge-base" in calls[1]
    with pytest.raises(remote_jobs.RemoteJobError, match="acceptance manifest"):
        backend.ready(host)
    host["evidence_manifest"] = "verification.json"
    with pytest.raises(remote_jobs.RemoteJobError, match="bootstrap"):
        backend.ready(host)


def test_scheduler_bootstrap_does_not_share_campaign_readiness(tmp_path, monkeypatch):
    import json
    import subprocess

    import yaml

    from ops.jobs import remote_backend as remote_jobs
    from ops.resources.gpu_guard import APPROVED_HOST_GPUS
    from scripts import jobs as scheduler

    backend, host, job, evidence = _rv_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(scheduler, "REPO", tmp_path)
    monkeypatch.setattr(remote_jobs.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, host["commit"] + "\trefs/heads/splatter4d\n", ""))
    launches = []
    monkeypatch.setattr(backend, "snapshot", lambda: remote_jobs.RemoteSnapshot(
        {gpu: [] for gpu in APPROVED_HOST_GPUS["remote"]}, 200, 1000 * 2**30, 800 * 2**30, {}))
    monkeypatch.setattr(backend, "launch", lambda path: launches.append(json.loads(path.read_text())) or {"container_id": "f" * 64})
    campaign = {"id": "campaign", "host": "remote", "script": "scripts/worker.py", "gpu": job["gpu"],
                "ram_gb": 10, "mem_gb": 5, "disk_gb": 1, "required_results": ["completion.json"]}
    queue = tmp_path / "queue.yaml"
    queue.write_text(yaml.safe_dump({"hosts": {"remote": {**host, "enabled": True}}, "jobs": [campaign, job]}))
    scheduler.tick(queue, tmp_path / "registry.jsonl", tmp_path / "runs",
                   usage={gpu: [] for gpu in APPROVED_HOST_GPUS["local"]}, available_gb=200,
                   disk_bytes=(1000 * 2**30, 800 * 2**30), remote_backend=backend)
    assert [item["id"] for item in launches] == ["verify-suite"]
    current = scheduler.read_registry(tmp_path / "registry.jsonl")["verify-suite"]
    assert current["campaign_eligible"] is False and current["verification_purpose"] == "native_suite"
    assert current["args"] == launches[0]["args"] and current["source_sha256"] == evidence["source_sha256"]


@pytest.mark.parametrize("direction", ["verification_as_campaign", "campaign_as_verification"])
def test_reconcile_refuses_mixed_job_kinds(tmp_path, monkeypatch, direction):
    from types import SimpleNamespace

    from scripts import jobs as scheduler

    backend, host, job, evidence = _rv_setup(tmp_path, monkeypatch)
    _, spec = backend.prepare(job, job["gpu"], 1, host, evidence)
    current = {**spec, "attempts": 1, "status": "running", "data_identity": evidence["data_identity"]}
    item = _rv_inspect(spec, evidence["data_identity"])
    backend_value = SimpleNamespace(sync_results=lambda *args, **kwargs: pytest.fail("mixed identity was synced"))
    if direction == "verification_as_campaign":
        for key in ("verification_purpose", "campaign_eligible", "verification_timeout_seconds"):
            current.pop(key)
    else:
        for key in ("s4d.verification_purpose", "s4d.campaign_eligible"):
            item["Config"]["Labels"].pop(key)
        item["Config"]["Env"] = [value for value in item["Config"]["Env"] if not value.startswith("S4D_VERIFICATION_")]
    registry = tmp_path / "registry.jsonl"
    owned, reachable = scheduler._reconcile_remote(backend_value, SimpleNamespace(containers={spec["container"]: item}),
                                                   {job["id"]: current}, registry, tmp_path / "runs", [])
    assert not owned and reachable and scheduler.read_registry(registry)[job["id"]]["unknown"]


@pytest.mark.parametrize("fault", [None, "arguments", "memory", "timeout"])
def test_verification_container_bounds_are_reconciled(tmp_path, monkeypatch, fault):
    from ops.jobs import remote_backend as remote_jobs

    backend, host, job, evidence = _rv_setup(tmp_path, monkeypatch)
    _, spec = backend.prepare(job, job["gpu"], 1, host, evidence)
    current = {**spec, "attempts": 1, "data_identity": evidence["data_identity"]}
    item = _rv_inspect(spec, evidence["data_identity"])
    if fault == "arguments":
        item["Config"]["Cmd"].append("--other")
    elif fault == "memory":
        item["HostConfig"]["Memory"] += 1
    elif fault == "timeout":
        item["Config"]["Env"] = [value.replace("TIMEOUT_SECONDS=7200", "TIMEOUT_SECONDS=7199")
                                for value in item["Config"]["Env"]]
    if fault:
        with pytest.raises(remote_jobs.RemoteJobError):
            remote_jobs.validate_container(item, job["id"], current)
    else:
        remote_jobs.validate_container(item, job["id"], current)


@pytest.mark.parametrize("kind", ["matching", "verification_as_campaign", "campaign_as_verification", "failed_report"])
def test_synced_verification_requires_typed_wrappers_and_passing_report(tmp_path, monkeypatch, kind):
    import hashlib
    import json

    from ops.jobs import remote_backend as remote_jobs
    from scripts import jobs as scheduler

    backend, host, job, evidence = _rv_setup(tmp_path, monkeypatch)
    _, spec = backend.prepare(job, job["gpu"], 1, host, evidence)
    current = {**spec, "attempts": 1, "data_identity": evidence["data_identity"],
               "source_sha256": evidence["source_sha256"]}
    root = tmp_path / "runs" / job["id"]
    (root / "evidence").mkdir(parents=True)
    suite = {"host": "remote", "full_suite_passed": kind != "failed_report", "exit_code": 0,
             "counts": {"passed": 1, "failed": 0, "error": 0, "skipped": 0},
             "expected_commit": host["commit"], "expected_image_digest": host["image_digest"],
             "source_sha256": evidence["source_sha256"], "source_sha256_before": evidence["source_sha256"],
             "source_changed_during_suite": False}
    (root / "evidence/tests.json").write_text(json.dumps(suite))
    (root / "evidence/tests.log").write_text("passed\n")
    (root / "evidence/tests.junit.xml").write_text("<testsuite tests='1' failures='0'/>\n")
    identity = {"host": "remote", "job_id": job["id"], "attempt": 1, "git_commit": host["commit"],
                "image_digest": host["image_digest"], "gpu_uuids": [job["gpu"]],
                "verification_purpose": "native_suite", "campaign_eligible": False}
    if kind == "campaign_as_verification":
        identity.pop("verification_purpose")
        identity.pop("campaign_eligible")
    if kind == "verification_as_campaign":
        for key in ("verification_purpose", "campaign_eligible", "verification_timeout_seconds"):
            current.pop(key)
    (root / "exit.json").write_text(json.dumps({**identity, "code": 0}))
    (root / "provenance.json").write_text(json.dumps({**identity, "data_id": evidence["data_identity"],
                                                    "script": job["script"], "arguments": spec["args"]}))
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in [*spec["required_results"], "exit.json", "provenance.json"]}
    if kind == "matching":
        scheduler._verify_synced_exit(job["id"], current, {"files": files}, tmp_path / "runs")
    else:
        with pytest.raises((remote_jobs.RemoteJobError, ValueError)):
            scheduler._verify_synced_exit(job["id"], current, {"files": files}, tmp_path / "runs")


def test_campaign_prepare_and_fingerprint_remain_unchanged(tmp_path, monkeypatch):
    import hashlib
    import json

    from ops.jobs import remote_backend as remote_jobs
    from scripts import jobs as scheduler

    backend, host, _, evidence = _rv_setup(tmp_path, monkeypatch)
    script = tmp_path / "scripts/worker.py"
    script.write_text("guard_gpus()\n")
    job = {"id": "ordinary", "host": "remote", "script": "scripts/worker.py", "args": [0, "a; $(false)"],
           "mem_gb": 5, "ram_gb": 10, "disk_gb": 1, "required_results": ["completion.json"]}
    campaign = {"commit": host["commit"], "image_digest": host["image_digest"], "data_identity": evidence["data_identity"]}
    _, spec = backend.prepare(job, next(iter(host["limits"])), 1, host, campaign)
    assert spec["args"] == ["0", "a; $(false)"] and spec["labels"] == remote_jobs.container_labels(
        job["id"], 1, host["commit"], host["image_digest"])
    assert not {"verification_purpose", "campaign_eligible", "verification_timeout_seconds"} & spec.keys()
    original = {"script": job["script"], "args": ["0", "a; $(false)"],
                "required_results": job["required_results"], "expected_config_sha256": None}
    assert scheduler.job_fingerprint(job) == hashlib.sha256(
        json.dumps(original, sort_keys=True, separators=(",", ":")).encode()).hexdigest()



def test_verification_sync_uses_attempt_output_and_rejects_campaign_state(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from ops.jobs.remote_backend import RemoteJobError
    from ops.containers.verification import verification_output

    backend, host, job, evidence = _rv_setup(tmp_path, monkeypatch)
    _, spec = backend.prepare(job, job["gpu"], 2, host, evidence)
    state = {**spec, "attempts": 2}
    output = verification_output("native_suite", job["id"], 2)
    contents = {f"{output}/{name.split('/')[-1]}": b"original report bytes"
                for name in spec["required_results"]}
    contents.update({f"{backend.client.root}/runs/{job['id']}/{name}": b"wrapper bytes"
                     for name in ("provenance.json", "console.log", "exit.json")})
    def checked(command, **kwargs):
        return SimpleNamespace(stdout="".join(hashlib.sha256(value).hexdigest() + "  " + path + "\n"
                                             for path, value in contents.items()))
    backend.client = SimpleNamespace(checked_ssh=checked,
                                    rsync_download=lambda source, target, **kwargs: target.write_bytes(contents[source]))
    synced = backend.sync_verification_results(job["id"], state, final=True)
    assert synced["campaign_eligible"] is False and synced["verification_purpose"] == "native_suite"
    assert (tmp_path / "runs/verify-suite/evidence/tests.json").read_bytes() == b"original report bytes"
    state = {key: value for key, value in state.items()
             if key not in {"verification_purpose", "campaign_eligible", "verification_timeout_seconds"}}
    with pytest.raises((RemoteJobError, ValueError)):
        backend.sync_verification_results(job["id"], state, final=True)


def test_isolation_bootstrap_requires_passing_suite_for_same_release(tmp_path, monkeypatch):
    import hashlib
    import json

    from ops.containers.verification import POLICIES, verification_artifacts

    backend, host, job, evidence = _rv_setup(tmp_path, monkeypatch)
    suite = {"host": "remote", "full_suite_passed": True, "exit_code": 0,
             "counts": {"passed": 1, "failed": 0, "error": 0, "skipped": 0},
             "expected_commit": host["commit"], "expected_image_digest": host["image_digest"],
             "source_sha256": evidence["source_sha256"], "source_sha256_before": evidence["source_sha256"],
             "source_changed_during_suite": False}
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"commit": host["commit"], "image_digest": host["image_digest"], "report": suite}))
    evidence.update(verification_purpose="gpu_isolation",
                    full_suite={"path": "suite.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    job.update(id="verify-isolation", verification_purpose="gpu_isolation", script=POLICIES["gpu_isolation"][0],
               ram_gb=8, mem_gb=4, required_results=["evidence/gpu_isolation.json"])
    (tmp_path / job["script"]).write_text("pass\n")
    _, spec = backend.prepare(job, job["gpu"], 1, host, evidence)
    assert spec["args"][-3:] == ["--gpu", job["gpu"], "--host-pid-namespace"]
    evidence.pop("full_suite")
    with pytest.raises(ValueError):
        verification_artifacts(evidence, "gpu_isolation", tmp_path, evidence["source_sha256"])


# ---- completed-container cleanup (user instruction 2026-10-21) --------------------------------------------------------
def test_removal_command_never_forces_prunes_or_deletes_volumes_and_rechecks_identity():
    cmd = remote.removal_command(remote.ROOT, "r", 2, "s4d-r-a2", CONTAINER_ID, 0, ["exit.json", "checkpoints/x.pt"])
    lines = cmd.splitlines()
    assert 'docker rm "$id" > /dev/null || exit 6' in lines and lines.count("check") == 3  # identity checked thrice
    assert lines.index("flock -w 60 9 || exit 8") < lines.index("check", lines.index("flock -w 60 9 || exit 8"))
    assert [i for i, line in enumerate(lines) if line == "check"][-1] == lines.index('docker rm "$id" > /dev/null || exit 6') - 1
    assert 'd="$(sha256sum < "$1")" || exit 5' in cmd and "mktemp" in cmd and "removal-receipt" in cmd
    docker = [line for line in lines if "docker " in line]
    assert {line.split("docker ")[1].split()[0] for line in docker} == {"inspect", "logs", "rm"}
    for forbidden in (" -f", "--force", " -v", "--volumes", "prune", "rmi", "volume", "network", "kill", "stop"):
        assert not any(forbidden in line.split("docker ", 1)[1] for line in docker), forbidden
    # the only other deletions are this run's own temporary archive files
    assert [line for line in lines if "rm -f" in line] == ["trap 'rm -f -- \"$tmp_log\" \"$tmp_record\"' EXIT"]
    # incl. read-only root filesystem and no verification labels
    assert f"{CONTAINER_ID}|exited|0|true|remote|r|2|/s4d-r-a2|true||" in cmd
    # optional labels print nothing when missing (Docker prints "<no value>" for a bare index of a missing key)
    assert '{{with index .Config.Labels "s4d.verification_purpose"}}{{.}}{{end}}' in remote._INSPECT_IDENTITY
    assert '{{with index .Config.Labels "s4d.campaign_eligible"}}{{.}}{{end}}' in remote._INSPECT_IDENTITY
    assert f"{remote.ROOT}/runs/r/containers/s4d-r-a2.docker.log" in cmd and "No such object" in cmd
    for bad in (dict(root="/tmp"), dict(container="s4d-other-a2"), dict(container_id="x" * 64),
                dict(required=["../escape"]), dict(run_id="r;rm")):
        args = {"root": remote.ROOT, "run_id": "r", "attempt": 2, "container": "s4d-r-a2", "container_id": CONTAINER_ID,
                "expected_code": 0, "required": ["exit.json"], **bad}
        with pytest.raises(ValueError):
            remote.removal_command(**args)


def test_new_containers_carry_ownership_labels_and_legacy_ones_still_validate():
    assert remote.container_labels("r", 1, COMMIT, IMAGE) == {**remote.labels("r", 1, COMMIT, IMAGE),
                                                              **remote.OWNERSHIP_LABELS}
    state = {"attempts": 1, "commit": COMMIT, "image_digest": IMAGE, "container": "s4d-r-a1", "gpu": REMOTE,
             "gpu_backend": "cdi", "container_id": CONTAINER_ID}
    legacy = inspected(status="running")
    remote.validate_container(legacy, "r", state)  # created before the labels existed
    labelled = inspected(status="running")
    labelled["Config"]["Labels"] = remote.container_labels("r", 1, COMMIT, IMAGE)
    remote.validate_container(labelled, "r", state)
    labelled["Config"]["Labels"]["s4d.owner"] = "someone-else"
    with pytest.raises(remote.RemoteJobError):
        remote.validate_container(labelled, "r", state)


def test_completed_container_is_removed_once_after_verified_sync(tmp_path):
    queued = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    assert state["r"]["status"] == "done"
    assert backend.removals == [("r", 1, "s4d-r-a1", CONTAINER_ID, 0,
                                 ("console.log", "exit.json", "metrics.jsonl", "provenance.json"))]
    assert state["r"]["containers_removed"] == [CONTAINER_ID]
    tick(tmp_path, queued, backend)  # nothing left to do: no second removal
    assert len(backend.removals) == 1


def test_failed_interrupted_or_mismatched_containers_are_kept(tmp_path):
    queued = write_queue(tmp_path, [job(max_restarts=0)])
    backend = FakeBackend(tmp_path / "runs")
    launch_event(tmp_path)
    backend.value.containers["s4d-r-a1"] = inspected(status="exited", code=7)
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["status"] == "failed" and not backend.removals
    # a completed job whose container no longer matches its recorded id is kept and reported
    queued = write_queue(tmp_path, [job("s")])
    launch_event(tmp_path, "s")
    other = inspected("s", status="exited")
    backend.value.containers["s4d-s-a1"] = other
    backend.remove_error = remote.RemoteJobError("transient")
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["s"]["status"] == "done" and len(backend.removals) == 1
    backend.remove_error = None
    backend.removals.clear()
    backend.value.containers["s4d-s-a1"] = {**other, "Id": "e" * 64}
    messages = tick(tmp_path, queued, backend)
    assert not backend.removals and any("kept" in m for m in messages)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["s"]["containers_kept"] == [CONTAINER_ID]
    assert not any("kept" in m for m in tick(tmp_path, queued, backend))  # recorded once, then skipped


def test_cleanup_is_retry_safe_after_refusal_or_lost_connection(tmp_path):
    queued = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    backend.remove_error = remote.RemoteUnreachable("lost")
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    assert state["r"]["status"] == "done" and "containers_removed" not in state["r"]
    backend.remove_error = remote.RemoteJobError("transient")  # e.g. the log could not be saved: retried later
    messages = tick(tmp_path, queued, backend)
    assert "containers_removed" not in jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert any("kept for now" in m for m in messages)
    backend.remove_error = None
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["containers_removed"] == [CONTAINER_ID]
    # an absent container (already removed before a lost response) is confirmed by the command and recorded once
    queued = write_queue(tmp_path, [job("t")])
    launch_event(tmp_path, "t")
    backend.value.containers["s4d-t-a1"] = inspected("t", status="exited")
    backend.remove_error = remote.RemoteUnreachable("lost")
    tick(tmp_path, queued, backend)
    backend.value.containers.pop("s4d-t-a1")
    backend.remove_error = None
    calls = len(backend.removals)
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["t"]
    assert state["containers_absent"] == [CONTAINER_ID] and "containers_removed" not in state
    assert len(backend.removals) == calls + 1
    tick(tmp_path, queued, backend)
    assert len(backend.removals) == calls + 1


def test_cleanup_can_be_disabled_per_host(tmp_path):
    queued = write_queue(tmp_path, [job()], cleanup_completed_containers=False)
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["status"] == "done" and not backend.removals


def test_lasting_refusal_is_recorded_once_and_the_container_kept(tmp_path):
    queued = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    backend.remove_error = remote.ContainerRemovalRefused("required run files are missing")
    messages = tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert state["containers_kept"] == [CONTAINER_ID] and "containers_removed" not in state
    assert any("required run files are missing" in m for m in messages)
    tick(tmp_path, queued, backend)
    assert len(backend.removals) == 1


def test_earlier_attempts_are_removed_only_after_the_job_succeeded(tmp_path):
    queued = write_queue(tmp_path, [job(max_restarts=2)])
    backend = FakeBackend(tmp_path / "runs")
    first, unrecorded = "a" * 64, "b" * 64
    backend.value.containers["s4d-r-a1"] = {**inspected(status="exited", code=137), "Id": first}
    launch_event(tmp_path, container_id=first)
    jobs.record({"event": "exited", "id": "r", "host": "remote", "attempt": 1, "code": 137},
                tmp_path / "registry.jsonl")
    launch_event(tmp_path, attempt=2, container=remote.container_name("r", 2))
    backend.value.containers["s4d-r-a2"] = inspected(attempt=2, status="running")
    tick(tmp_path, queued, backend)
    assert not backend.removals  # interrupted attempt kept while its recovery is still running
    # an attempt whose exit was never recorded is not a target
    backend.value.containers["s4d-r-a3"] = {**inspected(attempt=3, status="exited", code=1), "Id": unrecorded}
    backend.value.containers["s4d-r-a2"] = inspected(attempt=2, status="exited")
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert state["status"] == "done"
    assert [(r[1], r[3], r[4]) for r in backend.removals] == [(1, first, 137), (2, CONTAINER_ID, 0)]
    assert sorted(state["containers_removed"]) == sorted([first, CONTAINER_ID])
    assert "s4d-r-a3" in backend.value.containers


def test_backend_removal_maps_refusals_and_transport_failures(tmp_path):
    class Client:
        root = remote.ROOT
        state_dir = tmp_path

        def __init__(self, code, out=""):
            self.code, self.out, self.commands = code, out, []

        def ssh(self, command, **kwargs):
            self.commands.append((command, kwargs))
            if self.code is None:
                raise remote.RemoteUnreachable("lost")
            return subprocess.CompletedProcess([], self.code, self.out, "")

    args = ("r", 1, "s4d-r-a1", CONTAINER_ID, 0, ["exit.json"])
    ok = Client(0, '{"removed": true, "log_sha256": "%s", "inspect_sha256": "%s"}\n' % ("0" * 64, "1" * 64))
    assert remote.RemoteBackend(REPO, tmp_path / "runs", ok).remove_container(*args)["removed"] is True
    assert ok.commands[0][1]["idempotent"] is True
    gone = Client(0, '{"absent": true}\n')
    assert remote.RemoteBackend(REPO, tmp_path / "runs", gone).remove_container(*args) == {"absent": True}
    for code, error in ((3, remote.ContainerRemovalRefused), (4, remote.ContainerRemovalRefused),
                        (5, remote.RemoteJobError), (7, remote.RemoteJobError), (0, remote.RemoteJobError)):
        with pytest.raises(error):
            remote.RemoteBackend(REPO, tmp_path / "runs", Client(code, "garbage")).remove_container(*args)
    with pytest.raises(remote.RemoteUnreachable):
        remote.RemoteBackend(REPO, tmp_path / "runs", Client(None)).remove_container(*args)


def test_verification_containers_are_never_cleanup_targets(tmp_path):
    """Early verification jobs carry no purpose field in the registry; their container labels identify them."""
    queued = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    backend.remove_error = remote.RemoteJobError("transient")  # the job completes; its container stays for now
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["status"] == "done" and len(backend.removals) == 1
    backend.remove_error = None
    backend.removals.clear()
    item = backend.value.containers["s4d-r-a1"]
    item["Config"]["Labels"] = {**item["Config"]["Labels"], "s4d.verification_purpose": "gpu_isolation",
                                "s4d.campaign_eligible": "false"}
    for _ in range(2):
        tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert not backend.removals and "s4d-r-a1" in backend.value.containers
    assert not {"containers_removed", "containers_kept", "containers_absent"} & set(state)


def test_cleared_keep_records_reenable_cleanup(tmp_path):
    """A refusal traced to a defect of the check itself is released by an explicit container_kept_cleared event."""
    queued = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    backend.remove_error = remote.ContainerRemovalRefused("identity or state differs from the record")
    tick(tmp_path, queued, backend)
    assert jobs.read_registry(tmp_path / "registry.jsonl")["r"]["containers_kept"] == [CONTAINER_ID]
    backend.remove_error = None
    tick(tmp_path, queued, backend)
    assert len(backend.removals) == 1  # kept: not retried
    jobs.record({"event": "container_kept_cleared", "id": "r", "host": "remote", "container_id": CONTAINER_ID},
                tmp_path / "registry.jsonl")
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert len(backend.removals) == 2 and state["containers_removed"] == [CONTAINER_ID]
    assert state["containers_kept"] == []


# ---- lifecycle cleanup (user directive 2026-10-28) ------------------------------------------------------------------
def test_container_is_removed_once_verified_in_place_and_the_job_completes_from_the_later_backup(tmp_path):
    """The local bulk backup no longer delays removal; the removed container of a compute-complete job is then not
    'missing' (unknown) and the job completes from the backup as before."""
    queued = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    backend.backup_ready = False
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert state["status"] == "sync_pending" and state["remote_verified"]["attempt"] == 1
    assert backend.removals == [("r", 1, "s4d-r-a1", CONTAINER_ID, 0,
                                 ("console.log", "exit.json", "metrics.jsonl", "provenance.json"))]
    assert state["containers_removed"] == [CONTAINER_ID] and "s4d-r-a1" not in backend.value.containers
    messages = tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert state["status"] == "sync_pending" and not state["unknown"] and not any("unknown" in m for m in messages)
    backend.backup_ready = True
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert state["status"] == "done" and len(backend.removals) == 1 and backend.inspections == ["r"]


def test_an_unverified_job_whose_container_vanished_stays_unknown(tmp_path):
    queued = write_queue(tmp_path, [job()])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    backend.sync_error = remote.RemoteJobError("not verifiable yet")
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert state["status"] == "sync_pending" and "remote_verified" not in state and not backend.removals
    backend.value.containers.pop("s4d-r-a1")
    backend.sync_error = None
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert state["unknown"] and state["reason"] == "container missing from confirmed snapshot" and not backend.removals


def test_completion_targets_need_the_final_attempt_exited_zero_and_verified_in_place(tmp_path):
    registry = tmp_path / "registry.jsonl"
    launch_event(tmp_path)
    assert jobs._cleanup_targets("r", jobs.read_registry(registry)["r"]) == []  # running
    jobs.record({"event": "remote_exited", "id": "r", "host": "remote", "code": 0, "container_id": CONTAINER_ID}, registry)
    assert jobs._cleanup_targets("r", jobs.read_registry(registry)["r"]) == []  # compute complete, not yet verified
    jobs.record({"event": "remote_verified", "id": "r", "host": "remote", "attempt": 1, "files": {}, "checks": {}},
                registry)
    state = jobs.read_registry(registry)["r"]
    assert [target[:4] for target in jobs._cleanup_targets("r", state)] == [(1, "s4d-r-a1", CONTAINER_ID, 0)]
    for changed in ({"remote_verified": {"attempt": 0}}, {"code": 1}, {"status": "crashed"},
                    {"verification_purpose": "native_suite"}, {"campaign_eligible": False}, {"host": "local"}):
        assert jobs._cleanup_targets("r", {**state, **changed}) == [], changed


PAUSE = {"event": "checkpoint_paused", "id": "r", "host": "remote", "attempt": 1, "intentional_stop": True,
         "container": "s4d-r-a1", "container_id": CONTAINER_ID, "checkpoint_step": 10,
         "checkpoint": "runs/r/checkpoints/step_0000010.pt", "checkpoint_sha256": "0" * 64,
         "container_exit_code": 143, "reason": "paper_ablation_pause"}


@pytest.mark.parametrize("change,targeted", [
    ({}, True), ({"checkpoint": f"{remote.ROOT}/runs/r/checkpoints/step_0000010.pt"}, True),
    ({"container_exit_code": None, "container_id": None}, True),  # pause records without them: the launch record
    ({"container_exit_code": 137}, False), ({"container_id": "e" * 64}, False), ({"checkpoint": None}, False),
    ({"checkpoint": "runs/other/checkpoints/step_0000010.pt"}, False), ({"intentional_stop": False}, False)])
def test_paused_attempt_targets_require_a_recorded_intentional_stop_exit_and_checkpoint(tmp_path, change, targeted):
    registry = tmp_path / "registry.jsonl"
    launch_event(tmp_path)
    jobs.record({key: value for key, value in {**PAUSE, **change}.items() if value is not None}, registry)
    assert jobs._cleanup_targets("r", jobs.read_registry(registry)["r"]) == []  # its exit is not observed yet
    jobs.record({"event": "exited", "id": "r", "host": "remote", "attempt": 1, "code": 143,
                 "container_id": CONTAINER_ID}, registry)
    expected = [(1, "s4d-r-a1", CONTAINER_ID, 143, ("checkpoints/step_0000010.pt", "console.log", "provenance.json"))]
    assert jobs._cleanup_targets("r", jobs.read_registry(registry)["r"]) == (expected if targeted else [])
    # the resumed attempt runs in a new container; the paused one stays a target, the running one never is
    launch_event(tmp_path, attempt=2, container=remote.container_name("r", 2), container_id="f" * 64)
    assert jobs._cleanup_targets("r", jobs.read_registry(registry)["r"]) == (expected if targeted else [])


def test_paused_attempt_container_is_removed_once_by_the_scheduler(tmp_path):
    queued = write_queue(tmp_path, [job()], hold=True)  # admission held: no resume inside this test
    backend = FakeBackend(tmp_path / "runs")
    launch_event(tmp_path)
    backend.value.containers["s4d-r-a1"] = inspected(status="exited", code=143)
    jobs.record(PAUSE, tmp_path / "registry.jsonl")
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert state["status"] == "crashed" and state["failures"] == 0 and not backend.launches
    assert backend.removals == [("r", 1, "s4d-r-a1", CONTAINER_ID, 143,
                                 ("checkpoints/step_0000010.pt", "console.log", "provenance.json"))]
    assert state["containers_removed"] == [CONTAINER_ID]
    tick(tmp_path, queued, backend)
    assert len(backend.removals) == 1


def test_backend_removal_is_deferred_while_a_result_sync_of_the_run_holds_its_lock(tmp_path):
    """The archive files (containers/*.docker.log, *.inspect.json) match the essential sync's *.log / *.json: removal
    takes that sync's per-run lock without blocking, so it never changes a manifest mid-sync nor stalls the tick."""
    import fcntl

    class Client:
        root, state_dir = remote.ROOT, tmp_path

        def __init__(self):
            self.commands = []

        def ssh(self, command, **kwargs):
            self.commands.append(command)
            payload = {"removed": True, "log_sha256": "0" * 64, "inspect_sha256": "1" * 64}
            return subprocess.CompletedProcess([], 0, json.dumps(payload) + "\n", "")

    client = Client()
    backend = remote.RemoteBackend(REPO, tmp_path / "runs", client)
    with (tmp_path / "results-r.lock").open("a") as held:  # e.g. scripts/sync_media.py's final essential copy of r
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(remote.RemoteJobError, match="in progress"):
            backend.remove_container("r", 1, "s4d-r-a1", CONTAINER_ID, 0, ["exit.json"])
        assert not client.commands
        assert backend.remove_container("s", 1, "s4d-s-a1", CONTAINER_ID, 0, ["exit.json"])["removed"] is True
    assert backend.remove_container("r", 1, "s4d-r-a1", CONTAINER_ID, 0, ["exit.json"])["removed"] is True
    assert len(client.commands) == 2


def test_scheduler_downloads_nothing_and_cross_host_dependents_wait_for_the_backup(tmp_path, monkeypatch):
    """2026-10-27: no live or final download inside the tick. A local dependent of a remote job (cross-host) starts only
    once the bulk worker's verified final backup completes the remote job; a remote dependent (same host) starts as soon
    as the finished run is verified in place on the remote."""
    queued = write_queue(tmp_path, [job("active"), job("after", "local", deps=["active"]), job("same", deps=["active"])])
    backend = FakeBackend(tmp_path / "runs")
    launch_event(tmp_path, "active")
    backend.value.containers["s4d-active-a1"] = inspected("active", status="running")
    monkeypatch.setattr(jobs, "mig_query", lambda devices, state_dir, timeout: {
        uuid: {"processes": []} for uuid in devices})
    launched = []
    monkeypatch.setattr(jobs, "launch", lambda job_row, gpu, attempt, runs: launched.append(job_row["id"]) or 4242)
    monkeypatch.setattr(jobs, "source_identity", lambda: {"commit": COMMIT, "source_sha256": {}})
    tick(tmp_path, queued, backend)
    assert backend.syncs == [] and not launched and not backend.launches
    backend.value.containers["s4d-active-a1"] = inspected("active", status="exited")
    backend.backup_ready = False
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    assert state["active"]["status"] == "sync_pending" and state["active"]["remote_verified"]["attempt"] == 1
    assert [payload["id"] for payload in backend.launches] == ["same"] and not launched
    backend.backup_ready = True
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    assert state["active"]["status"] == "done" and launched == ["after"]
    assert [payload["id"] for payload in backend.launches] == ["same"]


def test_compute_complete_remote_jobs_reserve_no_gpu_slot(tmp_path):
    """A container that exited 0 and awaits verification or backup (sync_pending) holds no GPU slot or memory."""
    queued = write_queue(tmp_path, [job("a"), job("b"), job("c")])  # limit: 2 jobs per remote GPU
    backend = FakeBackend(tmp_path / "runs")
    for run_id in ("a", "b"):
        launch_event(tmp_path, run_id)
        backend.value.containers[f"s4d-{run_id}-a1"] = inspected(run_id, status="exited")
    backend.backup_ready = False
    backend.sync_error = remote.RemoteJobError("not verifiable yet")
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    assert state["a"]["status"] == state["b"]["status"] == "sync_pending"
    assert [payload["id"] for payload in backend.launches] == ["c"]


def test_failed_remote_verification_keeps_same_host_dependents_blocked(tmp_path):
    queued = write_queue(tmp_path, [job(), job("dependent", deps=["r"])])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    backend.sync_error = remote.RemoteJobError("inspection failed")
    messages = tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    assert state["r"]["status"] == "sync_pending" and "remote_verified" not in state["r"] and not backend.launches
    assert any("dependents remain blocked" in message for message in messages)


def test_remote_verification_belongs_to_its_attempt(tmp_path):
    registry = tmp_path / "registry.jsonl"
    launch_event(tmp_path)
    jobs.record({"event": "remote_exited", "id": "r", "host": "remote", "code": 0, "container_id": CONTAINER_ID}, registry)
    jobs.record({"event": "remote_verified", "id": "r", "host": "remote", "attempt": 1, "files": {}, "checks": {}}, registry)
    state = jobs.read_registry(registry)["r"]
    assert jobs.dependency_ready({"host": "remote"}, state) and not jobs.dependency_ready({"host": "local"}, state)
    jobs.record({"event": "launching", "id": "r", "host": "remote", "gpu": REMOTE, "attempt": 2}, registry)
    state = jobs.read_registry(registry)["r"]
    assert "remote_verified" not in state and not jobs.dependency_ready({"host": "remote"}, state)


@pytest.mark.parametrize("steps,released", [(9, False), (10, True)])
def test_pretraining_budget_must_be_complete_before_same_host_release(tmp_path, monkeypatch, steps, released):
    args = ["--config", "x.yaml", "--set", "train.steps=10", "train.seed=0"]
    required = ["completion.json", "metrics.jsonl"]
    queued = write_queue(tmp_path, [job(script="scripts/train.py", args=args, required_results=required),
                                    job("dependent", deps=["r"])])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path, script="scripts/train.py", args=args, required_results=required)
    original = backend.sync_results
    numeric = {"train": {"seed": 0, "steps": 10}}
    digest = hashlib.sha256(json.dumps(numeric, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def trained(run_id, final=False, **kwargs):
        result = original(run_id, final)
        folder = backend.runs / run_id
        report = json.loads((folder / "provenance.json").read_text())
        report.update(numeric_config=numeric, config_sha256=digest)
        for name, content in (("provenance.json", report), ("completion.json", {"status": "completed", "steps": steps})):
            data = json.dumps(content).encode()
            (folder / name).write_bytes(data)
            result["files"][name] = hashlib.sha256(data).hexdigest()
        return result

    monkeypatch.setattr(backend, "sync_results", trained)
    backend.backup_ready = False
    tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")
    assert ("remote_verified" in state["r"]) is released
    assert [payload["id"] for payload in backend.launches] == (["dependent"] if released else [])


def test_remote_export_must_be_a_weights_only_export_of_its_source_with_the_configured_signature(tmp_path):
    import dataclasses
    import datetime

    import torch

    from s4d.model.encoder import Encoder, EncoderConfig
    config = EncoderConfig(width=32, depth=1, heads=2)
    encoder = Encoder(config)
    state = {"args": ["--run", "runs/pre", "--checkpoint", "runs/pre/checkpoints/step_0000010.pt",
                      "--out", "runs/exp/encoder.pt"]}
    payload = {"format": "splatter4d-encoder-v1", "encoder_config": dataclasses.asdict(config),
               "state_dict": encoder.state_dict(), "step": 10, "source_checkpoint": state["args"][3], "run_name": "pre"}

    def inspect(**changes):
        folder = tmp_path / str(len(list(tmp_path.iterdir())))
        folder.mkdir()
        torch.save({**payload, **changes}, folder / "encoder.pt")
        return remote_artifact_check.inspect_run(str(folder), ["encoder.pt"])["archives"]["encoder.pt"]
    checks = jobs._verify_remote_export(state, inspect())
    assert checks["step"] == 10 and checks["tensors"] == len(encoder.state_dict())
    wider = dataclasses.asdict(EncoderConfig(width=64, depth=1, heads=2))
    for changes in ({"step": 9}, {"run_name": "other"}, {"source_checkpoint": "runs/pre/checkpoints/latest.pt"},
                    {"format": "other"}, {"encoder_config": wider}, {"extra": datetime.timedelta(days=1)}):
        with pytest.raises(remote.RemoteJobError):
            jobs._verify_remote_export(state, inspect(**changes))


def test_a_backup_never_completes_an_attempt_that_failed_its_in_place_verification(tmp_path, monkeypatch):
    """Codex review 2026-10-27: a failed remote verification must not fall through to completion via the backup."""
    queued = write_queue(tmp_path, [job(), job("local-dependent", "local", deps=["r"])])
    backend = FakeBackend(tmp_path / "runs")
    backend.value.containers["s4d-r-a1"] = inspected(status="exited")
    launch_event(tmp_path)
    monkeypatch.setattr(jobs, "_verify_remote_exit", lambda *args: (_ for _ in ()).throw(
        remote.RemoteJobError("budget shortfall")))
    monkeypatch.setattr(jobs, "launch", lambda *args: pytest.fail("a dependent was released"))
    for _ in range(2):
        tick(tmp_path, queued, backend)
    state = jobs.read_registry(tmp_path / "registry.jsonl")["r"]
    assert state["status"] == "sync_pending" and state["remote_verification_failed"]["attempt"] == 1
    assert backend.inspections == ["r"]  # rechecked at most every REMOTE_RECHECK_SECONDS, not every tick
    events = [json.loads(line)["event"] for line in (tmp_path / "registry.jsonl").read_text().splitlines()]
    assert events.count("remote_verification_failed") == 1 and "results_synced" not in events


def test_remote_export_must_read_its_producers_verified_checkpoint_at_the_producers_budget(tmp_path):
    """Codex review 2026-10-27: producer identity, budget and checkpoint evidence are checked before release."""
    import dataclasses

    import torch

    from s4d.model.encoder import Encoder, EncoderConfig
    config = EncoderConfig(width=32, depth=1, heads=2)
    encoder = Encoder(config)
    state = {"args": ["--run", "runs/pre", "--checkpoint", "runs/pre/checkpoints/step_0000010.pt",
                      "--out", "runs/exp/encoder.pt"]}
    torch.save({"format": "splatter4d-encoder-v1", "encoder_config": dataclasses.asdict(config),
                "state_dict": encoder.state_dict(), "step": 10, "source_checkpoint": state["args"][3],
                "run_name": "pre"}, tmp_path / "encoder.pt")
    archive = remote_artifact_check.inspect_run(str(tmp_path), ["encoder.pt"])["archives"]["encoder.pt"]
    signature = jobs.signature_sha256(archive["state_dict"])

    def producer(steps=10, files=("checkpoints/step_0000010.pt",), digest=signature, attempt=1):
        return {"pre": {"host": "remote", "status": "sync_pending", "attempts": 1, "args": [], "remote_verified": {
            "attempt": attempt, "files": {name: "a" * 64 for name in files}, "checks": {"steps": steps, "checkpoints": {
                "checkpoints/step_0000010.pt": {"step": 10, "encoder_signature_sha256": digest}}}}}}
    assert jobs._verify_remote_export(state, archive, producer())["step"] == 10
    for registry in (producer(steps=20), producer(files=()), producer(digest="0" * 64), producer(attempt=0), {}):
        with pytest.raises(remote.RemoteJobError):
            jobs._verify_remote_export(state, archive, registry)
