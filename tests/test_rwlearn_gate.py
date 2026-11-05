"""Full-data gate (s4d/rwlearn/gate.py) and launch plans (s4d/rwlearn/launch.py) on synthetic ingestion manifests in
the ``rw-ingest-manifest`` v2 schema of runs/setup/tools/realworld_ingest_20261106 (the live manifest is not read)."""

from __future__ import annotations

import copy
import hashlib

import pytest

from s4d.rwlearn import gate, launch

STAGES = gate.REQUIRED_STAGES


def entry(task: str, i: int, *, status="valid", transfer="verified", pre="done", stages=STAGES, sha=None):
    sha = sha or hashlib.sha256(f"{task}-{i}".encode()).hexdigest()
    return {"task": task, "episode_index": i, "sha256": sha, "flags": {}, "override": None,
            "source": {"present": True}, "validation": {"status": status},
            "transfer": {"status": transfer, "remote_sha256": sha},
            "preprocessing": {"schema": "rw-preprocess", "schema_version": 1, "status": pre,
                              "stages": {s: {"status": "done", "input_sha256": sha, "code_version": "v"}
                                         for s in stages}}}


def manifest(n_drawer=100, n_mug=100, complete=True, extra=None):
    eps = {}
    for task, n in (("drawer-open", n_drawer), ("mug-hanging", n_mug)):
        for i in range(n):
            eps[f"{task}/episode_{i * 2:06d}.h5"] = entry(task, i * 2)  # gaps in the numbering
    eps.update(extra or {})
    return {"schema": "rw-ingest-manifest", "schema_version": 2, "episodes": eps,
            "counts": {t: {"complete": complete, "target": 100} for t in ("drawer-open", "mug-hanging")}}


def test_both_tasks_pass_with_exactly_100_preprocessed_episodes():
    rep = gate.evaluate(manifest(), ["drawer-open", "mug-hanging"])
    assert rep["passed"]
    t = rep["tasks"]["drawer-open"]
    assert t["valid_and_preprocessed"] == 100 and t["store_episodes"][0] == "drawer-open/episode_000000"


@pytest.mark.parametrize("mutate,reason", [
    (lambda m: m["counts"]["drawer-open"].update(complete=False), "ingestion not complete"),
    (lambda m: m["episodes"].pop("drawer-open/episode_000004.h5"), "99 episodes"),
    (lambda m: m["episodes"]["drawer-open/episode_000004.h5"]["preprocessing"].update(status="running"),
     "preprocessing 'running'"),
    (lambda m: m["episodes"]["drawer-open/episode_000004.h5"]["preprocessing"]["stages"].pop("sam3_masks"),
     "required stages"),
    (lambda m: m["episodes"]["drawer-open/episode_000004.h5"]["preprocessing"]["stages"]["calibrate"].update(
        input_sha256="other"), "required stages"),
    (lambda m: m["episodes"]["drawer-open/episode_000004.h5"]["transfer"].update(status="partial"), "transfer"),
    (lambda m: m["episodes"]["drawer-open/episode_000004.h5"].update(flags={"duplicate_content": 1}), "99 episodes"),
    (lambda m: m["episodes"].update({"drawer-open/episode_000999.h5": entry("drawer-open", 999, status="incomplete")}),
     "not final"),
    (lambda m: m["episodes"].update({"drawer-open/episode_000998.h5": entry("drawer-open", 998)}), "101 episodes"),
    (lambda m: m.update(schema_version=1), "schema"),
    (lambda m: m["episodes"]["drawer-open/episode_000004.h5"]["transfer"].update(remote_sha256="0" * 64),
     "remote content"),
    (lambda m: m["episodes"]["drawer-open/episode_000004.h5"].update(sha256=None), "no valid sha256"),
    (lambda m: m["episodes"]["drawer-open/episode_000004.h5"]["validation"].update(status=None), "unknown validation"),
    (lambda m: m["episodes"]["drawer-open/episode_000004.h5"]["source"].update(present=False), "99 episodes"),
    (lambda m: m["episodes"]["drawer-open/episode_000004.h5"].update(
        flags={"source_missing": 1}, override={"decision": "accept"}), "99 episodes"),
    (lambda m: m["episodes"]["drawer-open/episode_000004.h5"].update(flags={"new_kind_of_flag": 1}), "unknown flags"),
    (lambda m: m["episodes"]["drawer-open/episode_000004.h5"].update(override={"decision": "exclude"}), "99 episodes"),
    (lambda m: m["counts"]["drawer-open"].update(complete="yes"), "ingestion not complete"),
])

def test_gate_fails_closed(mutate, reason):
    m = manifest()
    mutate(m)
    rep = gate.evaluate(m, ["drawer-open"])
    assert not rep["passed"]
    text = rep.get("error", "") + " ".join(rep["tasks"].get("drawer-open", {}).get("problems", []))
    assert reason in text, text
    assert not rep["tasks"].get("drawer-open", {}).get("store_episodes")


def test_blocked_records_are_not_counted_like_the_ingestion_worker():
    dup = entry("drawer-open", 4, sha=hashlib.sha256(b"x").hexdigest())
    dup["flags"] = {"duplicate_content": "same content as another recording"}
    m = manifest(extra={"drawer-open/episode_000901.h5": dup})
    rep = gate.evaluate(m, ["drawer-open"])
    assert rep["passed"] and rep["tasks"]["drawer-open"]["excluded"][0]["key"] == "drawer-open/episode_000901.h5"
    assert "drawer-open/episode_000901" not in rep["tasks"]["drawer-open"]["store_episodes"]


def test_accept_override_clears_eligible_flags():
    m = manifest()
    m["episodes"]["drawer-open/episode_000004.h5"].update(flags={"index_conflict": 1}, override={"decision": "accept"})
    assert gate.evaluate(m, ["drawer-open"])["passed"]


def test_invalid_recordings_do_not_count_and_joint_waits_for_both():
    m = manifest(n_mug=60, extra={"drawer-open/episode_000777.h5": entry("drawer-open", 777, status="invalid")})
    rep = gate.evaluate(m, ["drawer-open", "mug-hanging"])
    assert rep["tasks"]["drawer-open"]["passed"] and not rep["tasks"]["mug-hanging"]["passed"]
    assert not rep["passed"]
    plan = launch.plan(rep, ["drawer-open", "mug-hanging", "joint"], "deadbeef")
    ids = [r["id"] for r in plan["rows"]]
    assert set(plan["refused"]) == {"mug-hanging", "joint"}
    assert all("drawer-open" in i for i in ids) and not any("joint" in i or "mug" in i for i in ids)


def test_plan_rows_declare_roles_dependencies_and_are_not_registered():
    rep = gate.evaluate(manifest(), ["drawer-open", "mug-hanging"])
    plan = launch.plan(rep, ["drawer-open", "mug-hanging", "joint"], "abc123", seeds=(0, 1))
    assert plan["registered"] is False and not plan["refused"]
    rows = {r["id"]: r for r in plan["rows"]}
    pre = [r for r in rows.values() if r["role"] == "realworld_pretrain"]
    assert len(pre) == 5 * 3 and all(r["host"] == "remote" for r in rows.values())
    assert all("configs/realworld/baselines/" in " ".join(r["args"]) for r in pre)
    fql = [r for r in rows.values() if r["role"] == "realworld_fql"]
    assert len(fql) == 2 * 6 * (1 + 2)  # per task: 6 conditions x (dataset build + 2 seeds)
    for r in fql:
        for d in r["deps"]:
            assert d in rows
    run = rows["rw-fql-mvp3d_single-mug-hanging-encmug-hanging-s1"]
    assert run["deps"] == ["rw-fqldata-mvp3d_single-mug-hanging-encmug-hanging"]
    assert rows[run["deps"][0]]["deps"] == ["rw-mvp3d-mug-hanging-pre"]
    assert "condition=mvp3d_single" in run["args"] and run["planned_commit"] == "abc123"


def test_role_classification_of_planned_rows_matches_the_declared_roles():
    """Rows must classify as their declared role under the role rules (script / config evidence): the scripts carry
    no Meta-World evidence and the real-world configs live under configs/realworld/."""
    rep = gate.evaluate(manifest(), ["drawer-open", "mug-hanging"])
    rows = launch.plan(copy.deepcopy(rep), ["drawer-open"], "x")["rows"]
    for r in rows:
        script = r["script"].rsplit("/", 1)[-1]
        if r["role"] == "realworld_pretrain":
            assert script.startswith("train_") and any(a.startswith("configs/realworld/") for a in r["args"])
        else:
            assert "realworld" in script and "fql" in script
        assert not any("metaworld" in a or "configs/baselines/" in a or "configs/rl/" in a for a in r["args"])
