"""scripts/sync_media.py: which remote runs the bulk media worker copies (it never writes the registry)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import sync_media  # noqa: E402
from ops.remote.access import RemoteClient, sha256  # noqa: E402


def test_media_worker_selects_deferred_completed_and_active_remote_campaign_runs(tmp_path):
    client = SimpleNamespace(network_dir=tmp_path)
    state = {
        "live": {"host": "remote", "status": "running"},
        "pending-sync": {"host": "remote", "status": "sync_pending"},
        "deferred": {"host": "remote", "status": "done", "sync": {"final": True, "scope": "essential"}},
        "copied": {"host": "remote", "status": "done", "sync": {"final": True, "scope": "essential"}},
        "legacy": {"host": "remote", "status": "done", "sync": {"final": True}},  # synced everything before the split
        "verification": {"host": "remote", "status": "running", "verification_purpose": "native_suite"},
        "ineligible": {"host": "remote", "status": "running", "campaign_eligible": False},
        "local": {"host": "local", "status": "running"},
        "failed": {"host": "remote", "status": "failed"},
    }
    (tmp_path / "results-media").mkdir()
    (tmp_path / "results-media" / "copied.json").write_text(json.dumps(
        {"verified": True, "final": True, "scope": "media"}))
    (tmp_path / "results-media" / "deferred.json").write_text(json.dumps(
        {"verified": False, "final": False, "scope": "media"}))  # an interrupted copy is retried
    final, live = sync_media.candidates(state, client)
    assert final == ["deferred"] and live == ["live", "pending-sync"]


def test_bulk_worker_copies_deferred_checkpoints_of_completed_runs_last(tmp_path):
    client = SimpleNamespace(network_dir=tmp_path)
    state = {
        "required-only": {"host": "remote", "status": "done", "sync": {"final": True, "scope": "essential",
                                                                       "checkpoints": "required"}},
        "all-weights": {"host": "remote", "status": "done", "sync": {"final": True, "scope": "essential"}},
        "copied": {"host": "remote", "status": "done", "sync": {"final": True, "scope": "essential",
                                                                "checkpoints": "required"}},
        "running": {"host": "remote", "status": "running"},
    }
    (tmp_path / "results-checkpoints").mkdir()
    (tmp_path / "results-checkpoints" / "copied.json").write_text(json.dumps(
        {"verified": True, "final": True, "scope": "checkpoints"}))
    assert sync_media.checkpoint_candidates(state, client) == ["required-only"]


def test_backups_cover_compute_complete_runs_with_cross_host_dependencies_first(tmp_path):
    """2026-10-27: the worker makes the final local backups (the tick downloads nothing); runs a pending local job
    depends on come first; a verified final backup of the run's attempt is not repeated."""
    client = SimpleNamespace(network_dir=tmp_path / "net", main_repo=tmp_path / "repo")
    state = {
        "a": {"host": "remote", "status": "sync_pending", "code": 0, "attempts": 1},
        "b": {"host": "remote", "status": "sync_pending", "code": 0, "attempts": 1},
        "copied": {"host": "remote", "status": "sync_pending", "code": 0, "attempts": 2},
        "stale": {"host": "remote", "status": "sync_pending", "code": 0, "attempts": 2},
        "verification": {"host": "remote", "status": "sync_pending", "code": 0, "attempts": 1,
                         "verification_purpose": "native_suite"},
        "running": {"host": "remote", "status": "running", "attempts": 1},
        "done": {"host": "remote", "status": "done", "attempts": 1},
    }
    for run_id, attempt in (("copied", 2), ("stale", 1)):  # "stale": the record is of an earlier attempt
        (client.network_dir / "results").mkdir(parents=True, exist_ok=True)
        (client.main_repo / "runs" / run_id).mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"attempt": attempt}).encode()
        (client.main_repo / "runs" / run_id / "exit.json").write_bytes(payload)
        (client.network_dir / "results" / f"{run_id}.json").write_text(json.dumps(
            {"verified": True, "final": True, "scope": "essential",
             "files": {"exit.json": hashlib.sha256(payload).hexdigest()}}))
    # a record whose checksums differ from the attempt's in-place verification is not a backup of that attempt
    state["b"]["remote_verified"] = {"attempt": 1, "files": {"exit.json": "0" * 64}}
    queue = [{"id": "export-b", "host": "local", "deps": ["b"]}, {"id": "remote-a", "host": "remote", "deps": ["a"]}]
    assert sync_media.backup_candidates(state, client, queue) == ["b", "a", "stale"]
    assert sync_media.running(state) == ["running"]
    assert sync_media.required_checkpoints({"required_results": ["metrics.jsonl", "checkpoints/step_1.pt"]}) == [
        "checkpoints/step_1.pt"]


def test_backup_requirements_include_remote_verified_files_and_declared_results():
    current = {"required_results": ["dataset.h5", "checkpoints/required.pt"],
               "remote_verified": {"files": {"exit.json": "a", "provenance.json": "b",
                                             "checkpoints/verified.pt": "c"}}}
    assert sync_media.required_backup_files(current) == [
        "checkpoints/required.pt", "checkpoints/verified.pt", "dataset.h5", "exit.json", "provenance.json"]
    assert sync_media.required_checkpoints(current) == ["checkpoints/required.pt", "checkpoints/verified.pt"]


def test_dataset_essential_backup_uses_download_budget_and_completes(tmp_path, monkeypatch):
    jobs = sync_media.jobs
    run_id = "rwng-20261107b-fqldata-test"
    registry = tmp_path / "registry.jsonl"
    spec = {"id": run_id, "host": "remote", "attempt": 1, "gpu": "test-gpu", "commit": "a" * 40,
            "image_digest": "sha256:" + "b" * 64, "data_identity": "c" * 64,
            "script": "scripts/realworld_fql_dataset.py", "args": [], "required_results": ["dataset.h5"]}
    identity = {"host": "remote", "job_id": run_id, "attempt": 1, "git_commit": spec["commit"],
                "image_digest": spec["image_digest"], "gpu_uuids": [spec["gpu"]], "script": spec["script"],
                "arguments": [], "data_id": spec["data_identity"]}
    remote = tmp_path / "remote-run"
    remote.mkdir()
    (remote / "dataset.h5").write_bytes(b"required dataset bytes")
    (remote / "exit.json").write_text(json.dumps({**identity, "code": 0}))
    (remote / "provenance.json").write_text(json.dumps(identity))
    (remote / "other.h5").write_bytes(b"unrelated large file")
    (remote / "checkpoint.pt").write_bytes(b"unrelated weights")
    files = {name: sha256(remote / name) for name in ("dataset.h5", "exit.json", "provenance.json")}
    jobs.record({"event": "launched", **spec}, registry)
    jobs.record({"event": "remote_exited", "id": run_id, "host": "remote", "code": 0}, registry)
    jobs.record({"event": "remote_verified", "id": run_id, "host": "remote", "attempt": 1, "files": files}, registry)

    # No real credentials or network: retain the actual manifest selection, staging, budget lock and checksum checks.
    client = RemoteClient.__new__(RemoteClient)
    client.repo = client.main_repo = tmp_path
    client.network_dir = tmp_path / "net"
    client.state_dir = client.network_dir / "access"
    client.state_dir.mkdir(parents=True)
    client.target = "fake-remote"
    (client.state_dir / "download-budget.json").write_text(json.dumps({"kib_per_s": 300}))
    transfers = []

    def manifest(command, **kwargs):
        return subprocess.run(["sh", "-c", command], capture_output=True, text=True, check=True)

    def transfer(source, target, extra, wait, append, accept):
        assert "--bwlimit=300" in extra and "--timeout=300" in extra
        listing = Path(next(arg.split("=", 1)[1] for arg in extra if arg.startswith("--files-from=")))
        transfers.extend(listing.read_text().splitlines())
        for name in listing.read_text().splitlines():
            (Path(target) / name).write_bytes((remote / name).read_bytes())

    read_registry = jobs.read_registry
    monkeypatch.setattr(client, "ensure_remote_path", lambda *args, **kwargs: str(remote))
    monkeypatch.setattr(client, "checked_ssh", manifest)
    monkeypatch.setattr(client, "_legacy_downloads", lambda: [])
    monkeypatch.setattr(client, "_rsync_run", transfer)
    monkeypatch.setattr(sync_media, "RemoteClient", lambda: client)
    monkeypatch.setattr(jobs, "read_registry", lambda *args: read_registry(registry))
    monkeypatch.setattr(jobs, "load_queue", lambda: {"jobs": []})
    monkeypatch.setattr(sys, "argv", ["sync_media.py", "--min-interval", "0"])
    assert sync_media.main() == 0
    assert sorted(transfers) == sorted(files)
    assert (client.state_dir / "download-budget.lock").exists()
    assert (tmp_path / "runs" / run_id / "dataset.h5").read_bytes() == (remote / "dataset.h5").read_bytes()
    current = read_registry(registry)[run_id]
    assert current["status"] == "sync_pending" and sync_media.backup_done(client, run_id, current)
    backup = json.loads((client.network_dir / "results" / (run_id + ".json")).read_text())
    assert backup["files"] == files
    backend = SimpleNamespace(backup_record=lambda _: backup)
    jobs._complete_from_backup(backend, run_id, current, registry, tmp_path / "runs", [])
    completed = read_registry(registry)[run_id]
    assert completed["status"] == "done" and completed["sync"]["files"] == files
