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
