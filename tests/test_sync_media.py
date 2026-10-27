"""scripts/sync_media.py: which remote runs the bulk media worker copies (it never writes the registry)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import sync_media  # noqa: E402


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
        (client.network_dir / "results" / f"{run_id}.json").write_text(json.dumps(
            {"verified": True, "final": True, "scope": "essential"}))
        (client.main_repo / "runs" / run_id).mkdir(parents=True, exist_ok=True)
        (client.main_repo / "runs" / run_id / "exit.json").write_text(json.dumps({"attempt": attempt}))
    queue = [{"id": "export-b", "host": "local", "deps": ["b"]}, {"id": "remote-a", "host": "remote", "deps": ["a"]}]
    assert sync_media.backup_candidates(state, client, queue) == ["b", "a", "stale"]
    assert sync_media.running(state) == ["running"]
    assert sync_media.required_checkpoints({"required_results": ["metrics.jsonl", "checkpoints/step_1.pt"]}) == [
        "checkpoints/step_1.pt"]
