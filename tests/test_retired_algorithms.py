"""Permanent retirement admission policy; CPU-only regressions with no processes or remote operations."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from ops.jobs import host_roles, scheduler
from ops.jobs.retired import retired_reason

RETIRED = [
    ("scripts/train_ppo.py", "ds-ppo-cnn-hammer-s2000"),
    ("scripts/train_dreamer.py", "ds-dreamer-cnn-hammer-s2000"),
    ("scripts/train_flashsac.py", "ds-flashsac-cnn-hammer-s2000"),
]


def retired_rows():
    rows = []
    for index, (script, run_id) in enumerate(RETIRED):
        rows.extend([
            {"id": run_id, "script": "scripts/train_rl.py", "args": []},  # id alone
            {"id": f"renamed-trainer-{index}", "script": script, "args": []},
            {"id": run_id + "-eval", "script": "scripts/eval_rl.py", "args": []},
            {"id": f"train-job-eval-{index}", "script": "scripts/eval_rl.py",
             "args": ["--train-job", run_id]},
            {"id": f"run-dir-eval-{index}", "script": "scripts/eval_rl.py",
             "args": [f"--run-dir=/workspace/splatter4d/runs/{run_id}"]},
            {"id": f"dependency-eval-{index}", "script": "scripts/eval_rl.py", "deps": [run_id + "-done"]},
            {"id": f"reevaluation-{index}", "script": "scripts/eval_rl.py", "reevaluates": run_id},
        ])
    return rows


@pytest.mark.parametrize("previous", ["pending", "crashed", "reset"])
@pytest.mark.parametrize("scoped", [False, True])
def test_retired_rows_never_launch_after_recovery_or_reset(tmp_path, monkeypatch, previous, scoped):
    gpu = scheduler.APPROVED_HOST_GPUS["local"][0]
    retired = retired_rows()
    controls = [
        {"id": "drm-control", "script": "scripts/train_rl.py"},
        {"id": "ds-sac-control", "script": "scripts/train_sac.py"},
        {"id": "sac-eval-control", "script": "scripts/eval_rl.py",
         "args": ["--train-job", "ds-sac-done", "--run-dir=runs/ds-sac-done"], "deps": ["ds-sac-done"]},
    ]
    rows = [{"args": [], "gpu": gpu, "mem_gb": 1, "ram_gb": 1, "max_restarts": 1, **r}
            for r in retired + controls]
    folder = tmp_path / "scoped" if scoped else tmp_path
    folder.mkdir(exist_ok=True)
    queue, registry = folder / "queue.yaml", tmp_path / "registry.jsonl"
    queue.write_text(yaml.safe_dump({"limits": {gpu: {"max_jobs": 4, "max_mem_gb": 20}}, "jobs": rows}))
    deps = {dep for row in rows for dep in row.get("deps", [])}
    for job_id in deps:
        scheduler.record({"id": job_id, "event": "launched", "pid": 2**30, "attempt": 1}, registry)
        scheduler.record({"id": job_id, "event": "exited", "code": 0}, registry)
    if previous != "pending":
        for row in rows:
            scheduler.record({"id": row["id"], "event": "launched", "pid": 2**30, "gpu": gpu,
                              "attempt": 1}, registry)
            scheduler.record({"id": row["id"], "event": "exited", "code": 1}, registry)
            if previous == "reset":
                scheduler.record({"id": row["id"], "event": "reset"}, registry)
    roles = {"hosts": {"local": {"scheduler": True, "admit": ["metaworld"], "unknown": "allow"}}}
    monkeypatch.setattr(host_roles, "load_roles", lambda: roles)
    monkeypatch.setattr(host_roles, "read_ownership", lambda: {})
    monkeypatch.setattr(host_roles, "load_exceptions", lambda: {})
    monkeypatch.setattr(scheduler, "source_identity", lambda: {"commit": "a" * 40, "source_sha256": {}})
    launched = []

    def launch(job, gpu, attempt, runs):
        launched.append((job["id"], attempt))
        return 2**30

    monkeypatch.setattr(scheduler, "launch", launch)
    scheduler.tick(queue, registry, tmp_path / "runs", usage={}, available_gb=200,
                   disk_bytes=(2000 * 2**30, 1000 * 2**30))
    attempt = 1 if previous == "pending" else 2
    assert launched == [(r["id"], attempt) for r in controls]
    state = scheduler.read_registry(registry)
    assert all(state.get(r["id"], {}).get("attempts", 0) == attempt - 1 for r in retired)


def test_retired_remote_only_queue_never_queries_backend(tmp_path):
    gpu = scheduler.APPROVED_HOST_GPUS["remote"][0]
    rows = [{"host": "remote", "gpu": gpu, "mem_gb": 1, "ram_gb": 1, "disk_gb": 1,
             "required_results": ["final.json"], **r} for r in retired_rows()]
    queue = tmp_path / "queue.yaml"
    queue.write_text(yaml.safe_dump({"hosts": {"local": {"enabled": False},
                                              "remote": {"enabled": True}}, "jobs": rows}))

    class Backend:
        def snapshot(self):
            pytest.fail("retired rows must not trigger a remote query")

    registry = tmp_path / "registry.jsonl"
    scheduler.tick(queue, registry, tmp_path / "runs", usage={}, available_gb=200,
                   disk_bytes=(2000 * 2**30, 1000 * 2**30), remote_backend=Backend())
    assert not registry.exists()


@pytest.mark.parametrize("script,run_id", RETIRED)
@pytest.mark.parametrize("reference", [
    {"args": ["--train-job", "renamed"]},
    {"args": ["--train-job=renamed"]},
    {"args": ["--run-dir", "runs/renamed"]},
    {"args": ["--run-dir=/workspace/splatter4d/runs/renamed"]},
    {"deps": ["renamed"]},
    {"reevaluates": "renamed"},
])
def test_renamed_evaluator_targets_use_the_trainer_script(script, run_id, reference):
    trainer = {"id": "renamed", "script": script}
    evaluator = {"id": "unrelated-evaluator", "script": "scripts/eval_rl.py", **reference}
    assert retired_reason(evaluator, {"renamed": trainer}) is not None
    assert scheduler.eligible([trainer, evaluator], {"renamed": {"status": "done"}}) == []
    supported = {**trainer, "script": "scripts/train_sac.py"}
    assert retired_reason(evaluator, {"renamed": supported}) is None
    assert scheduler.eligible([supported, evaluator], {"renamed": {"status": "done"}}) == [evaluator]


@pytest.mark.parametrize("script,run_id", RETIRED)
@pytest.mark.parametrize("exception", ["retry", "drain"])
def test_retirement_precedes_role_retry_and_evaluator_drain(script, run_id, exception):
    job = {"id": run_id + "-eval", "script": "scripts/eval_rl.py",
           "args": ["--train-job", run_id, "--run-dir", "runs/" + run_id]}
    roles = {"hosts": {"remote": {"scheduler": True, "admit": [], "unknown": "deny",
                                   "drain_evaluators": exception == "drain"}}}
    current = {"host": "remote", "attempt": 1, "attempts": 1, "job_sha256": "a" * 64,
               "attempt_containers": {"1": {"container_id": "b" * 64}}}
    grants = {job["id"]: {"host": "remote", "attempt": 1, "job_sha256": "a" * 64}} if exception == "retry" else {}
    trainer = {"host": "remote", "status": "done"} if exception == "drain" else {}
    assert (host_roles.retry_in_place(job["id"], "remote", current, grants) if exception == "retry" else
            host_roles.drain_evaluator(job, "remote", roles, trainer))
    assert "retired" in host_roles.refusal(job, "remote", roles, {}, {}, current, grants, trainer)
    control_id = "ds-sac-control"
    control = {**job, "id": control_id + "-eval",
               "args": ["--train-job", control_id, "--run-dir", "runs/" + control_id]}
    control_grants = {control["id"]: grants[job["id"]]} if grants else {}
    assert host_roles.refusal(control, "remote", roles, {}, {}, current, control_grants, trainer) is None


@pytest.mark.parametrize("disk", [False, True])
def test_default_replay_windows_preserve_returns_rng_and_bootstrap_ids(tmp_path, disk):
    from s4d.rl.replay import Replay

    replay = Replay(tmp_path / "replay" if disk else None, (2,), np.float32, 1, 1, 20, 1, 3)
    # Several episodes overwrite the ring; a terminal discount and a time limit are both represented.
    for episode in range(8):
        replay.add_initial(np.full(2, episode), [episode])
        for step in range(6):
            done = step == 5
            replay.add([step], float(step + 1), 0.0 if done and episode % 2 else 1.0,
                       np.full(2, step + 1), [step + 1], done)
    direct_rng, window_rng = np.random.default_rng(17), np.random.default_rng(17)
    expected = replay.sample_indices(64, direct_rng, 0.97)
    actual = replay.sample_windows(64, window_rng, 0.97)
    for before, after in zip((*expected, expected[0] + replay.nstep), actual):
        assert before.dtype == after.dtype and before.shape == after.shape
        assert before.tobytes() == after.tobytes()
    assert direct_rng.bit_generator.state == window_rng.bit_generator.state
    starts = actual[0] % replay.capacity
    ends = actual[3] % replay.capacity
    assert np.array_equal(replay.episode_id[starts], replay.episode_id[ends])
    # The RAM caller keeps the same next-state atoms and proprioception as these bootstrap ids.
    batch = replay.sample(64, np.random.default_rng(17), 0.97)
    assert np.array_equal(batch[5], replay._stack(actual[3]))
    assert np.array_equal(batch[6], replay.proprio[ends])


@pytest.mark.parametrize("script", [".worktrees/arm4/scripts/train_flashsac.py", "/abs/clone/scripts/train_ppo.py",
                                    ".worktrees/x/scripts/train_dreamer.py"])
def test_retired_trainer_copies_outside_scripts_are_refused(script):
    assert retired_reason({"id": "renamed", "script": script}) is not None


def test_evaluator_copies_outside_scripts_check_their_targets():
    job = {"id": "renamed-eval", "script": ".worktrees/review/scripts/eval_rl.py", "args": ["--train-job", "ds-flashsac-x"]}
    assert retired_reason(job) is not None
    assert retired_reason({**job, "args": ["--train-job", "ds-drm-x"]}) is None
    assert retired_reason({"id": "drm", "script": ".worktrees/review/scripts/train_rl.py"}) is None
