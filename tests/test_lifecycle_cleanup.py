"""Lifecycle replay cleanup of completed RL runs (user directive 2026-10-28): controller eligibility, on-host re-checks,
the standard-library remote path, the main server's writer check, the ledger and the command-line pass."""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from s4d import lifecycle_cleanup as lifecycle
from s4d import replay_cleanup_host as host_check
from s4d.gpu_guard import APPROVED_HOST_GPUS

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lifecycle_jobs", REPO / "scripts/jobs.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
STEPS = 10
DRM_REPLAY = ["atom", "proprio", "action", "reward", "discount", "state_id", "transition_id", "episode_id",
              "episode_step", "meta"]


def NO_WRITER(run_id, run_dir):  # noqa: N802
    return []


def trainer_row(run_id="r", host="local", steps=STEPS, **overrides):
    return {"id": run_id, "host": host, "script": "scripts/train_rl.py",
            "args": ["--task", "hammer", "--encoder", "cnn", "--seed", "1", "--run-dir", f"runs/{run_id}",
                     "--set", f"train.num_train_steps={steps}"], **overrides}


def eval_row(run_id="r", **overrides):
    return {"id": f"{run_id}-eval", "script": "scripts/eval_rl.py",
            "args": ["--run-dir", f"runs/{run_id}", "--train-job", run_id, "--output-dir", f"runs/{run_id}-eval"],
            **overrides}


def done(host="local", **overrides):
    base = {"status": "done", "host": host, "attempt": 1, "attempts": 1, "code": 0, "unknown": False}
    if host == "remote":
        base.update(remote_verified={"attempt": 1, "files": {}}, sync={"final": True, "files": {}})
    return {**base, **overrides}


def policy(steps=STEPS):
    return {"step": steps, "policy": {"encoder": {"conv.weight": torch.zeros(2, 3)},
                                      "actor": {"fc.weight": torch.ones(4, 2), "fc.bias": torch.zeros(4)}}}


def make_run(runs, run_id="r", steps=STEPS, ram=False):
    run = runs / run_id
    (run / "snapshots").mkdir(parents=True)
    (run / "replay").mkdir()
    (run / "final.json").write_text(json.dumps({"completed": True, "steps": steps, "episodes": 1}))
    (run / "config.yaml").write_text(yaml.safe_dump({"train": {"num_train_steps": steps}}))
    torch.save(policy(steps), run / "snapshots" / f"step_{steps:07d}.pt")
    for name in DRM_REPLAY:
        np.memmap(run / "replay" / f"{name}.memmap", dtype=np.uint8, mode="w+", shape=(64,)).flush()
    if ram:
        for name in ("ram_snapshot.npz", "ram_snapshot.tmp.npz"):
            np.savez(run / "replay" / name, meta=np.zeros(3))
    return run


def spec_for(run_id="r", steps=STEPS, **extra):
    return {"run_id": run_id, "steps": steps, "patterns": list(lifecycle.DEFAULT_REPLAY_PATTERNS),
            "policy_keys": ["encoder", "actor"], **extra}


def memmaps(path):
    return sorted(path.rglob("*.memmap"))


# ---- controller eligibility ----------------------------------------------------------------------------------------
def test_completed_local_run_is_eligible_and_its_evaluation_does_not_delay_it(tmp_path):
    rows = [trainer_row(), eval_row()]
    state = {"r": done(pid=4242), "r-eval": {"status": "running", "host": "local", "attempts": 1}}
    [decision] = lifecycle.plan(rows, state, [], "local", tmp_path)
    assert decision.eligible and decision.session == 4242
    assert decision.spec == {"run_id": "r", "steps": STEPS, "patterns": ["*.memmap", "ram_snapshot*.npz"],
                             "policy_keys": ["encoder", "actor"]}
    assert lifecycle.plan(rows, state, [], "remote", tmp_path) == []  # not executed there


@pytest.mark.parametrize("current,reason", [
    ({"status": "running"}, "registry status running"),
    ({"status": "crashed", "code": -15, "intentional_stops": [{"attempt": 1}]}, "registry status crashed"),  # paused
    ({"status": "failed", "code": 1}, "registry status failed"),
    ({"status": "pending"}, "registry status pending"),  # reset for a rerun
    ({"code": 1}, "registry status done"),
    ({"unknown": True}, "registry state unknown")])
def test_running_paused_failed_reset_or_unknown_runs_are_never_eligible(tmp_path, current, reason):
    [decision] = lifecycle.plan([trainer_row()], {"r": done(**current)}, [], "local", tmp_path)
    assert not decision.eligible and decision.reason == reason


@pytest.mark.parametrize("current,eligible", [
    ({}, True), ({"status": "sync_pending"}, True),  # verified in place; the local backup does not gate replay
    ({"status": "sync_pending", "remote_verified": {"attempt": 0}}, False),
    ({"status": "sync_pending", "remote_verified": None}, False),
    ({"status": "running"}, False), ({"status": "launching"}, False), ({"status": "crashed", "code": 143}, False),
    ({"verification_purpose": "native_suite"}, False), ({"campaign_eligible": False}, False), ({"unknown": True}, False)])
def test_remote_runs_are_eligible_once_compute_complete_and_verified_in_place(tmp_path, current, eligible):
    [decision] = lifecycle.plan([trainer_row(host="remote")], {"r": done("remote", **current)}, [], "remote", tmp_path)
    assert decision.eligible is eligible


def test_remote_spec_carries_the_registry_final_checksum_and_migrated_runs_count_where_they_ran(tmp_path):
    current = done("remote", remote_verified={"attempt": 1, "files": {"final.json": "a" * 64}})
    [decision] = lifecycle.plan([trainer_row(host="remote")], {"r": current}, [], "remote", tmp_path)
    assert decision.spec["final_sha256"] == "a" * 64 and "config_sha256" not in decision.spec
    migrated = {**current, "migrated_from": {"host": "local", "attempt": 1, "checkpoint_step": 5}}
    assert [d.eligible for d in lifecycle.plan([trainer_row(host="remote")], {"r": migrated}, [], "local", tmp_path)] \
        == [True]  # its local leftover replay (the local copy of the final policy is still checked on the host)


def test_admission_paused_runs_are_never_eligible_and_an_unreadable_pause_file_fails_closed(tmp_path):
    path = tmp_path / "pauses.json"
    assert lifecycle.load_pauses(path) == []
    path.write_text(json.dumps({"pauses": [{"active": True, "prefixes": ["enc-abl-"], "ids": ["x"]},
                                           {"active": False, "ids": ["r"]}]}))
    pauses = lifecycle.load_pauses(path)
    rows = [trainer_row("enc-abl-a"), trainer_row("x"), trainer_row("r")]
    decisions = {d.run_id: d for d in lifecycle.plan(rows, {i: done() for i in ("enc-abl-a", "x", "r")}, pauses,
                                                       "local", tmp_path)}
    assert [decisions[i].eligible for i in ("enc-abl-a", "x", "r")] == [False, False, True]
    assert decisions["x"].reason == "admission pause active" and lifecycle.paused("x-eval", pauses)
    for bad in ("{", json.dumps({"pauses": "x"}), json.dumps({}), json.dumps({"pauses": [1]})):
        path.write_text(bad)
        with pytest.raises((ValueError, KeyError)):
            lifecycle.load_pauses(path)


@pytest.mark.parametrize("other,blocks", [
    ({"script": "scripts/train_rl.py", "args": ["--run-dir", "runs/o", "--set", "agent.init=runs/r/checkpoints/latest.pt"]},
     True),
    ({"script": "scripts/train_rl.py", "args": ["--run-dir", "runs/r"]}, True),  # a second writer, e.g. an extension
    ({"script": "scripts/train.py", "args": ["--resume", "/home/compu/kaist/sunho/runs/r"]}, True),
    ({"script": "scripts/train.py", "args": [], "continues": "runs/r/checkpoints/latest.pt"}, True),
    ({"script": "scripts/x.py", "args": ["--replay", "runs/r/replay"]}, True),
    ({"script": "scripts/eval_rl.py", "args": ["--run-dir", "runs/r/checkpoints"]}, True),
    ({"script": "scripts/eval_rl.py", "args": ["--run-dir", "runs/r", "--train-job", "r"]}, False),
    ({"script": "scripts/export_policy.py", "args": ["--snapshot", "runs/r/snapshots/step_0000010.pt"]}, False),
    ({"script": "scripts/train_rl.py", "args": ["--run-dir", "runs/r-eval", "--x", "runs/rr/checkpoints/a.pt"]}, False),
    ({"script": "scripts/train_rl.py", "args": [], "deps": ["r"], "companion_of": "r", "rerun_of": "r"}, False)])
def test_rows_continuing_from_the_run_block_its_cleanup(tmp_path, other, blocks):
    rows = [trainer_row(), {"id": "o", **other}]
    [decision] = [d for d in lifecycle.plan(rows, {"r": done()}, [], "local", tmp_path) if d.run_id == "r"]
    assert decision.eligible is not blocks
    assert not blocks or decision.reason.startswith("continuation references the run: o: ")


def test_references_respect_run_name_boundaries():
    assert lifecycle.references("runs/r", "r") == [""]
    assert lifecycle.references("runs/r/checkpoints/latest.pt", "r") == ["/checkpoints/latest.pt"]
    assert lifecycle.references("x=/home/compu/kaist/sunho/runs/r/replay", "r") == ["/replay"]
    assert lifecycle.references("runs/r-eval runs/rr runs/r.bak myruns/r runs/pretrain/r", "r") == []


def test_budget_comes_from_the_override_or_the_config_and_both_must_agree(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    trainer = lifecycle.TRAINERS["train_rl.py"]
    assert lifecycle.budget(trainer_row(), trainer, run) == (STEPS, None, "queue override")
    (run / "config.yaml").write_text(yaml.safe_dump({"train": {"num_train_steps": 20}}))
    assert lifecycle.budget(trainer_row(), trainer, run)[0] is None  # the run's config differs from the queue
    plain = trainer_row()
    plain["args"] = plain["args"][:-2]
    digest = hashlib.sha256((run / "config.yaml").read_bytes()).hexdigest()
    assert lifecycle.budget(plain, trainer, run) == (20, digest, "config.yaml")
    [decision] = lifecycle.plan([{**plain, "host": "remote"}], {"r": done("remote")}, [], "remote", tmp_path)
    assert decision.spec["config_sha256"] == digest  # compared with the remote config.yaml on the host
    (run / "config.yaml").unlink()
    [decision] = lifecycle.plan([plain], {"r": done()}, [], "local", tmp_path)
    assert not decision.eligible and "budget unknown" in decision.reason


def test_only_run_directory_trainers_are_considered_and_their_allowlists_are_explicit(tmp_path):
    rows = [trainer_row(), eval_row(), {"id": "p", "script": "scripts/train.py", "args": ["--name", "p"]},
            {"id": "q", "script": "scripts/train_sac.py", "args": ["--run-dir", "runs/q"]},
            {"id": "w", "script": "scripts/train_rl.py", "args": ["--run-dir", "runs/elsewhere"]}]
    state = {i: done() for i in ("r", "r-eval", "p", "q", "w")}
    decisions = {d.run_id: d for d in lifecycle.plan(rows, state, [], "local", tmp_path)}
    assert set(decisions) == {"r", "q"}  # a future trainer uses the defaults (here: no budget yet)
    assert lifecycle.TRAINERS.get("train_sac.py", lifecycle.DEFAULT_TRAINER).replay_patterns == ("*.memmap",
                                                                                                 "ram_snapshot*.npz")
    for patterns in (["*"], ["../*.memmap"], ["replay/*.memmap"], ["[a-z]*.memmap"], []):
        with pytest.raises(ValueError):
            host_check.validate_spec({**spec_for(), "patterns": patterns})


# ---- on-host checks and deletion ---------------------------------------------------------------------------------
def test_dry_run_reports_and_a_real_run_deletes_only_allowlisted_replay_files_once(tmp_path):
    runs = tmp_path / "runs"
    run = make_run(runs, ram=True)
    outside = tmp_path / "outside.memmap"
    outside.write_bytes(b"keep")
    os.symlink(outside, run / "replay" / "linked.memmap")
    (run / "replay" / "nested").mkdir()
    (run / "replay" / "nested" / "x.memmap").write_bytes(b"keep")
    (run / "replay" / "notes.txt").write_text("keep")
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "latest.pt").write_bytes(b"checkpoint")
    kept = [outside, run / "replay/nested/x.memmap", run / "replay/notes.txt", run / "checkpoints/latest.pt",
            run / "final.json", run / "config.yaml", run / f"snapshots/step_{STEPS:07d}.pt"]
    before = {path: path.read_bytes() for path in kept}
    calls = []

    def writers(run_id, run_dir):
        calls.append((run_id, run_dir))
        return []

    expected = sorted([f"{name}.memmap" for name in DRM_REPLAY] + ["ram_snapshot.npz", "ram_snapshot.tmp.npz"])
    report = host_check.clean_run(str(runs), spec_for(), dry_run=True, writers=writers,
                                  verify_policy=lifecycle.torch_policy_check)
    assert report["status"] == "would_delete" and calls == [("r", str(run.resolve()))]
    assert [item["name"] for item in report["files"]] == expected
    assert report["kept"] == ["linked.memmap", "nested", "notes.txt"]
    snapshot = report["evidence"]["snapshot"]
    assert snapshot["torch_weights_only_load"] is True and snapshot["tensors"] == 3 and snapshot["step"] == STEPS
    assert report["evidence"]["final"]["steps"] == STEPS and report["evidence"]["writers"] == "none"
    assert all((run / "replay" / name).exists() for name in expected)
    assert not (run / "replay" / host_check.RECEIPTS).exists()  # a dry run writes nothing
    report = host_check.clean_run(str(runs), spec_for(), dry_run=False, writers=writers,
                                  verify_policy=lifecycle.torch_policy_check)
    assert report["status"] == "deleted" and all(item["removed"] for item in report["files"])
    assert sorted(path.name for path in (run / "replay").iterdir()) == sorted(
        [host_check.RECEIPTS, "linked.memmap", "nested", "notes.txt"])
    assert {path: path.read_bytes() for path in kept} == before
    assert [record["state"] for record in report["receipts"]] == ["intent", "done"]
    again = host_check.clean_run(str(runs), spec_for(), dry_run=False, writers=writers)
    assert again["status"] == "noop" and len(again["receipts"]) == 2 and len(calls) == 2  # idempotent


@pytest.mark.parametrize("fault", ["final_incomplete", "final_steps", "snapshot_missing", "snapshot_step",
                                   "snapshot_module", "snapshot_global", "snapshot_corrupt", "snapshot_symlink",
                                   "writer", "replay_symlink", "run_symlink", "config_checksum", "final_checksum"])
def test_every_on_host_condition_refuses_and_deletes_nothing(tmp_path, fault):
    runs = tmp_path / "runs"
    run = make_run(runs)
    spec, writers, snapshot = spec_for(), NO_WRITER, run / f"snapshots/step_{STEPS:07d}.pt"
    if fault == "final_incomplete":
        (run / "final.json").write_text(json.dumps({"completed": False, "steps": STEPS}))
    elif fault == "final_steps":
        (run / "final.json").write_text(json.dumps({"completed": True, "steps": STEPS - 1}))
    elif fault == "snapshot_missing":
        snapshot.unlink()
    elif fault == "snapshot_step":
        torch.save(policy(STEPS - 1), snapshot)
    elif fault == "snapshot_module":
        torch.save({"step": STEPS, "policy": {"actor": policy()["policy"]["actor"]}}, snapshot)
    elif fault == "snapshot_global":
        torch.save({**policy(), "extra": datetime.timedelta(days=1)}, snapshot)
    elif fault == "snapshot_corrupt":
        snapshot.write_bytes(snapshot.read_bytes()[:-200])
    elif fault == "snapshot_symlink":
        snapshot.rename(tmp_path / "real.pt")
        os.symlink(tmp_path / "real.pt", snapshot)
    elif fault == "writer":
        writers = lambda run_id, run_dir: ["pid 7 holds a replay file open or mapped"]  # noqa: E731
    elif fault == "replay_symlink":
        (run / "replay").rename(tmp_path / "real-replay")
        os.symlink(tmp_path / "real-replay", run / "replay")
    elif fault == "run_symlink":
        run.rename(tmp_path / "real-run")
        os.symlink(tmp_path / "real-run", run)
    elif fault == "config_checksum":
        spec["config_sha256"] = "0" * 64
    elif fault == "final_checksum":
        spec["final_sha256"] = "0" * 64
    before = memmaps(tmp_path)
    for dry_run in (True, False):
        report = host_check.clean_run(str(runs), spec, dry_run=dry_run, writers=writers,
                                      verify_policy=lifecycle.torch_policy_check)
        assert report["status"] == "refused" and report["reason"], report
    assert memmaps(tmp_path) == before and len(before) == len(DRM_REPLAY)


def test_torch_check_loads_weights_only_and_checks_step_and_modules(tmp_path):
    path = tmp_path / "s.pt"
    torch.save(policy(), path)
    assert lifecycle.torch_policy_check(str(path), STEPS, ["encoder", "actor"]) == {
        "torch_weights_only_load": True, "torch_tensors": 3}
    for steps, keys in ((STEPS + 1, ["encoder"]), (STEPS, ["critic"])):
        with pytest.raises(ValueError):
            lifecycle.torch_policy_check(str(path), steps, keys)


def test_a_concurrent_execution_holding_the_replay_lock_defers_deletion(tmp_path):
    runs = tmp_path / "runs"
    run = make_run(runs)
    descriptor = os.open(run / "replay", os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        report = host_check.clean_run(str(runs), spec_for(), dry_run=False, writers=NO_WRITER)
        assert report["status"] == "busy" and len(memmaps(run)) == len(DRM_REPLAY)
    finally:
        os.close(descriptor)


def test_an_interrupted_execution_is_closed_and_ledgered_exactly_once(tmp_path):
    runs = tmp_path / "runs"
    run = make_run(runs)
    replay = run / "replay"
    host_check._append(str(replay), {"receipt_id": "a" * 32, "run_id": "r", "state": "intent", "time": 1.0,
                                     "files": [{"name": "atom.memmap", "bytes": 64, "allocated_bytes": 4096}],
                                     "evidence": {"final": {"steps": STEPS}}})
    (replay / "atom.memmap").unlink()  # the interrupted execution removed this file before it ended
    report = host_check.clean_run(str(runs), spec_for(), dry_run=False, writers=NO_WRITER)
    assert report["status"] == "deleted"
    decision = lifecycle.Decision("r", "local", True, evidence={"registry": {"status": "done"}})
    entries = lifecycle.ledger_entries(report, "local", decision, set(), 2.0)
    assert [(entry["receipt_id"] == "a" * 32, entry["interrupted"]) for entry in entries] == [(True, True), (False, False)]
    assert entries[0]["absent_after_interruption"] == 1 and entries[0]["evidence"]["host"] == {"final": {"steps": STEPS}}
    assert entries[1]["removed_files"] == len(DRM_REPLAY) - 1 and entries[1]["evidence"]["host"]["snapshot"]["step"] == STEPS
    ledger = tmp_path / "ledger.jsonl"
    lifecycle.append_ledger(ledger, entries)
    again = host_check.clean_run(str(runs), spec_for(), dry_run=False, writers=NO_WRITER)
    assert again["status"] == "noop"
    assert lifecycle.ledger_entries(again, "local", decision, lifecycle.recorded_receipts(ledger), 3.0) == []


def test_an_intent_is_closed_even_when_nothing_remains_to_delete(tmp_path):
    runs = tmp_path / "runs"
    run = make_run(runs)
    host_check._append(str(run / "replay"), {"receipt_id": "b" * 32, "run_id": "r", "state": "intent", "time": 1.0,
                                             "files": [{"name": f"{name}.memmap", "bytes": 64, "allocated_bytes": 0}
                                                       for name in DRM_REPLAY]})
    for path in memmaps(run):
        path.unlink()
    assert host_check.clean_run(str(runs), spec_for(), dry_run=True, writers=NO_WRITER)["status"] == "noop"
    report = host_check.clean_run(str(runs), spec_for(), dry_run=False, writers=NO_WRITER)
    assert report["status"] == "noop" and [r["state"] for r in report["receipts"]] == ["intent", "done"]
    assert report["receipts"][1]["interrupted"] and all(item["absent_now"] for item in report["receipts"][1]["files"])


# ---- remote path (standard library only) ---------------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("python3") is None, reason="needs a system python3")
def test_remote_command_runs_without_torch_and_rechecks_docker_before_deleting(tmp_path, monkeypatch):
    """The exact SSH command, run by a system python3 without PyTorch or the repository on its path, against a fake
    Docker: refused while a container labelled with the run id is live, deleted once none is, then a no-op."""
    root = tmp_path / "root"
    runs = root / "runs"
    make_run(runs)
    make_run(runs, "s", ram=True)
    monkeypatch.setattr(lifecycle, "REMOTE_ROOT", str(root))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    calls = tmp_path / "docker.calls"

    def fake_docker(body: str):
        docker.write_text(f"#!/bin/sh\necho \"$*\" >> {calls}\n{body}\n")
        docker.chmod(0o755)

    def execute(specs, dry_run):
        command = lifecycle.remote_command(str(root), specs, dry_run)
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, cwd=tmp_path, timeout=120,
                                env={"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path)})
        assert result.returncode == 0, result.stderr
        return lifecycle.parse_remote(result.stdout, specs, dry_run)

    specs = [spec_for(), spec_for("s")]
    fake_docker(f'case "$*" in *s4d.run_id=r\\ *) echo {"c" * 64};; esac')  # a live container of run r only
    reports = execute(specs, False)
    assert [r["status"] for r in reports] == ["refused", "deleted"] and "is not exited" in reports[0]["reason"]
    assert reports[1]["evidence"]["snapshot"]["check"].startswith("zip testzip") and len(memmaps(runs / "r")) == 10
    assert "torch_weights_only_load" not in reports[1]["evidence"]["snapshot"]
    assert "--filter label=s4d.run_id=r --filter status=created" in calls.read_text()
    fake_docker("true")
    assert [r["status"] for r in execute(specs, True)] == ["would_delete", "noop"] and len(memmaps(runs / "r")) == 10
    assert [r["status"] for r in execute(specs, False)] == ["deleted", "noop"] and memmaps(runs) == []
    fake_docker("exit 1")  # Docker unavailable: refused (a no-op needs no Docker query at all)
    make_run(runs, "t")
    assert execute([spec_for("t")], False)[0]["reason"].startswith("a writer may remain: Docker could not be queried")
    for bad in ('{"root": "/elsewhere", "dry_run": false, "reports": []}', "[]", "{}"):
        with pytest.raises(ValueError):
            lifecycle.parse_remote(bad, specs, False)


def test_remote_command_is_bounded_to_the_authorized_root_and_valid_requests():
    command = lifecycle.remote_command(lifecycle.REMOTE_ROOT, [spec_for()], True)
    assert command.startswith(f"set -eu; test -d {lifecycle.REMOTE_ROOT}/runs; test ! -L {lifecycle.REMOTE_ROOT}/runs;")
    assert "exec python3 -I -c" in command
    for bad in (("/tmp", [spec_for()]), (lifecycle.REMOTE_ROOT, [spec_for(), spec_for()]),
                (lifecycle.REMOTE_ROOT, [spec_for("../x")]), (lifecycle.REMOTE_ROOT, []),
                (lifecycle.REMOTE_ROOT, [{**spec_for(), "steps": True}]),
                (lifecycle.REMOTE_ROOT, [{**spec_for(), "extra": 1}])):
        with pytest.raises(ValueError):
            lifecycle.remote_command(*bad, True)


# ---- main-server writer check ------------------------------------------------------------------------------------
def fake_process(proc, pid, argv, session=1, fds=(), maps=(), state="S"):
    base = proc / str(pid)
    (base / "fd").mkdir(parents=True)
    (base / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")
    (base / "stat").write_text(f"{pid} (python) {state} 1 {pid} {session} 0 0\n")
    for index, target in enumerate(fds):
        os.symlink(target, base / "fd" / str(index))
    (base / "maps").write_text("".join(f"7f00-7f01 r--s 00000000 08:01 {1000 + i} {path}\n" for i, path in enumerate(maps)))


def test_local_writer_check_sees_the_session_open_or_mapped_replay_and_run_directory_references(tmp_path):
    run = tmp_path / "runs" / "r"
    (run / "replay").mkdir(parents=True)
    replay = str(run.resolve() / "replay")
    proc = tmp_path / "proc"
    fake_process(proc, 101, ["python", "-I", "scripts/eval_rl.py", "--run-dir", "runs/r", "--train-job", "r"])
    fake_process(proc, 102, ["bash", "-c", "true"], session=777)
    fake_process(proc, 103, ["python", "-I", "scripts/train_rl.py", "--run-dir", "runs/r-eval"])
    fake_process(proc, 104, ["python"], session=4242, state="Z")
    assert lifecycle.local_writers(run, "r", session=4242, proc=proc) == []
    for pid, kwargs in {201: {"argv": ["python", "-I", "scripts/train_rl.py", "--run-dir", "runs/r"]},
                        202: {"argv": ["sleep", "1"], "session": 4242},
                        203: {"argv": ["python"], "fds": [replay + "/atom.memmap"]},
                        204: {"argv": ["python"], "maps": [replay + "/atom.memmap (deleted)"]},
                        205: {"argv": ["python", "scripts/eval_rl.py", "--run-dir", "runs/r/replay"]},
                        206: {"argv": ["bash", "-c", "x", "_", "python", "-I", f"{run.resolve()}"]}}.items():
        fake_process(proc, pid, **kwargs)
        found = lifecycle.local_writers(run, "r", session=4242, proc=proc)
        assert len(found) == 1 and found[0].startswith(f"pid {pid} "), found
        shutil.rmtree(proc / str(pid))


# ---- command-line pass ---------------------------------------------------------------------------------------------
def test_cli_local_pass_dry_run_writes_nothing_then_deletes_ledgers_once_and_is_idempotent(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs"
    make_run(runs)
    make_run(runs, "paused")
    folder = tmp_path / "experiments"
    folder.mkdir()
    queue, registry, ledger = folder / "queue.yaml", folder / "registry.jsonl", tmp_path / "ledger.jsonl"
    queue.write_text(yaml.safe_dump({"jobs": [trainer_row(gpu="any", mem_gb=1), trainer_row("paused", gpu="any", mem_gb=1),
                                              eval_row(gpu="any", mem_gb=1)]}))
    gpu = APPROVED_HOST_GPUS["local"][0]
    for run_id in ("r", "paused"):
        jobs.record({"event": "launched", "id": run_id, "pid": 999_999_999, "gpu": gpu, "attempt": 1}, registry)
    jobs.record({"event": "exited", "id": "r", "attempt": 1, "code": 0}, registry)
    jobs.record({"event": "checkpoint_paused", "id": "paused", "attempt": 1, "intentional_stop": True,
                 "checkpoint_step": 5, "checkpoint_sha256": "0" * 64, "reason": "pause"}, registry)
    jobs.record({"event": "exited", "id": "paused", "attempt": 1, "code": -15}, registry)
    module_spec = importlib.util.spec_from_file_location("cleanup_replay_cli", REPO / "scripts/cleanup_replay.py")
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    argv = ["cleanup_replay.py", "--host", "local", "--queue", str(queue), "--registry", str(registry), "--runs", str(runs),
            "--pauses", str(tmp_path / "absent.json"), "--ledger", str(ledger)]

    def run(*extra):
        monkeypatch.setattr(sys, "argv", argv + list(extra))
        assert cli.main() == 0
        return [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    files = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    out = run("--dry-run")
    assert out[0]["status"] == "would_delete" and len(out[0]["files"]) == len(DRM_REPLAY)
    assert out[-1]["eligible"] == 1 and out[-1]["ineligible_by_reason"] == {"registry status crashed": 1}
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == files  # nothing written or deleted
    out = run()
    assert out[0]["status"] == "deleted" and out[-1]["ledger_entries"] == 1 and memmaps(runs / "r") == []
    assert len(memmaps(runs / "paused")) == len(DRM_REPLAY)  # paused (resumable) work keeps its replay
    [entry] = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert entry["run_id"] == "r" and entry["host"] == "local" and entry["removed_files"] == len(DRM_REPLAY)
    assert entry["evidence"]["controller"]["budget"] == {"steps": STEPS, "source": "queue override"}
    assert entry["evidence"]["host"]["snapshot"]["torch_weights_only_load"] is True
    out = run()
    assert out[0]["status"] == "noop" and out[-1].get("ledger_entries", 0) == 0 and len(ledger.read_text().splitlines()) == 1


def test_traversal_and_noncanonical_references_are_never_harmless():
    from s4d import lifecycle_cleanup as lifecycle
    from scripts import jobs

    assert lifecycle._blocking("/snapshots/../replay/atom.memmap", consumer=True)
    assert lifecycle._blocking("/./replay", consumer=True)
    assert not lifecycle._blocking("/snapshots/step_0001000.pt", consumer=True)
    assert jobs._run_relative("r1", "runs/r1/checkpoints/step_0170000.pt") == "checkpoints/step_0170000.pt"
    for bad in ("runs/r1//etc/passwd", "runs/r1/./checkpoints/x.pt", "runs/r1/checkpoints/../../r2/x.pt"):
        assert jobs._run_relative("r1", bad) is None
