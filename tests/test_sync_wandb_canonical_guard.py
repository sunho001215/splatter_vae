"""The legacy offline `wandb sync` batch never touches runs that have a canonical metrics.jsonl upload."""
import json

from ops.remote import sync_wandb


def test_realworld_prefixes_are_always_skipped(tmp_path):
    missing = tmp_path / "absent.json"
    for run_id in ("rwcal-20261109-snerl-drawer-open-pre", "rwnt-x", "rwnq-x", "rwa100-x", "rwng-x"):
        assert sync_wandb.canonical_owned(run_id, missing)
    assert not sync_wandb.canonical_owned("s4-chlrm-hammer-s0", missing)


def test_ledger_entries_are_skipped_and_bad_ledgers_ignored(tmp_path):
    ledger = tmp_path / "canonical.json"
    ledger.write_text(json.dumps({"runs": {"other-job": {"canonical_id": "abc"}}}))
    assert sync_wandb.canonical_owned("other-job", ledger)
    assert not sync_wandb.canonical_owned("unrelated", ledger)
    ledger.write_text("not json")
    assert not sync_wandb.canonical_owned("other-job", ledger)


def test_main_skips_owned_runs_before_any_remote_access(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sync_wandb, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(sync_wandb.jobs, "read_registry", lambda: {})
    monkeypatch.setattr(sync_wandb, "RemoteClient", lambda: object())
    monkeypatch.setattr(sync_wandb, "remote_listing", lambda client, run_id: calls.append(run_id) or {})
    monkeypatch.setattr("sys.argv", ["sync_wandb.py", "--run", "rwnq-20261111-fql-dynamo-covering-s0", "--run",
                                     "s4-other", "--no-upload"])
    assert sync_wandb.main() == 0
    assert calls == ["s4-other"]
    status = json.loads((tmp_path / "status.json").read_text())
    assert "rwnq-20261111-fql-dynamo-covering-s0" not in status


def test_main_holds_the_shared_status_lock(monkeypatch, tmp_path):
    import fcntl

    monkeypatch.setattr(sync_wandb, "STATUS", tmp_path / "status.json")
    held = []

    def probe():
        with (tmp_path / "status.lock").open("a") as other:
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held.append(False)
            except BlockingIOError:
                held.append(True)
        return 0

    monkeypatch.setattr(sync_wandb, "_main", probe)
    assert sync_wandb.main() == 0
    assert held == [True]
