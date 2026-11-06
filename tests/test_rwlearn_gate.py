"""Full-data gate (s4d/rwlearn/gate.py), the per-consumer calibration hold, and launch plans (s4d/rwlearn/launch.py) on
synthetic ingestion manifests in the ``rw-ingest-manifest`` v2 schema of runs/setup/tools/realworld_ingest_20261106
(the live manifest is not read; ingest_worker.set_stage is exercised on a temporary copy of the schema)."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from s4d.rwlearn import gate, launch

STAGES = gate.REQUIRED_STAGES


CAMS = ["135122070988", "135122073201", "135122076054"]


def calibration_record(sha, build="cal-build-1", status="done", **metrics):
    """The record agent CAL writes through ingest_worker.set_stage (contract: docs/realworld_fql.md)."""
    return {"status": status, "code_version": "calibration_validation@1", "input_sha256": sha,
            "output_format": "rw-calibration-validation/1", "stage_schema_version": 1,
            "metrics": {"decision": "accepted", "calibrate_build_id": build, "cameras": list(CAMS), **metrics}}


def entry(task: str, i: int, *, status="valid", transfer="verified", pre="done", stages=STAGES, sha=None,
          validated=False):
    sha = sha or hashlib.sha256(f"{task}-{i}".encode()).hexdigest()
    recs = {s: {"status": "done", "input_sha256": sha, "code_version": "v"} for s in stages}
    if "calibrate" in recs:
        recs["calibrate"]["metrics"] = {"build_id": "cal-build-1"}
    if validated:
        recs[gate.CALIBRATION_STAGE] = calibration_record(sha)
    return {"task": task, "episode_index": i, "sha256": sha, "flags": {}, "override": None,
            "source": {"present": True}, "validation": {"status": status},
            "transfer": {"status": transfer, "remote_sha256": sha},
            "preprocessing": {"schema": "rw-preprocess", "schema_version": 1, "status": pre, "stages": recs}}


def manifest(n_drawer=100, n_mug=100, complete=True, extra=None, validated=False):
    eps = {}
    for task, n in (("drawer-open", n_drawer), ("mug-hanging", n_mug)):
        for i in range(n):
            eps[f"{task}/episode_{i * 2:06d}.h5"] = entry(task, i * 2, validated=validated)  # gaps in numbering
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


def test_invalid_recordings_do_not_count_and_each_task_is_gated_alone():
    m = manifest(n_mug=60, extra={"drawer-open/episode_000777.h5": entry("drawer-open", 777, status="invalid")},
                 validated=True)
    rep = gate.evaluate(m, ["drawer-open", "mug-hanging"])
    assert rep["tasks"]["drawer-open"]["passed"] and not rep["tasks"]["mug-hanging"]["passed"]
    assert not rep["passed"]
    plan = launch.plan(rep, ["drawer-open", "mug-hanging"], "deadbeef")
    ids = [r["id"] for r in plan["rows"]]
    assert ids and all("drawer-open" in i for i in ids) and not any("mug" in i for i in ids)
    assert all("mug-hanging" in k for k in plan["refused"])
    with pytest.raises(ValueError, match="task-specific"):
        launch.plan(rep, ["joint"], "x")


# ---------------------------------------------------------------------------------------------- calibration hold
def test_calibration_hold_only_holds_geometry_dependent_consumers():
    rep = gate.evaluate(manifest(), ["drawer-open", "mug-hanging"])  # no calibration_validated record anywhere
    assert rep["passed"] and not rep["geometry_passed"]
    for task in ("drawer-open", "mug-hanging"):
        hold = rep["calibration_hold"][task]
        assert not hold["passed"] and hold["validated"] == 0 and "no calibration_validated record" in hold["problems"][0]
        for consumer in ("reviwo", "dynamo", "e2e_cnn"):
            assert gate.task_passed(rep, task, consumer)
        for consumer in ("snerl", "mvp3d", "sincro", "ours", None, "new-consumer"):
            assert not gate.task_passed(rep, task, consumer)
    rep = gate.evaluate(manifest(validated=True), ["drawer-open", "mug-hanging"])
    assert rep["passed"] and rep["geometry_passed"]
    hold = rep["calibration_hold"]["drawer-open"]
    assert hold["validated"] == 100
    assert hold["episodes"]["drawer-open/episode_000000"] == {"calibrate_build_id": "cal-build-1", "cameras": CAMS}


def test_consumer_classification_is_explicit():
    geo = {k for k, c in gate.CONSUMERS.items() if c.geometry}
    assert geo == {"ours", "snerl", "mvp3d", "sincro"}
    assert {k for k, c in gate.CONSUMERS.items() if not c.geometry} == {"reviwo", "dynamo", "e2e_cnn"}
    from s4d.rwlearn.fql_data import CONDITIONS

    assert set(gate.FQL_CONSUMER) == set(CONDITIONS)  # every FQL condition inherits its encoder's class
    assert {c for c, k in gate.FQL_CONSUMER.items() if gate.CONSUMERS[k].geometry} == {
        "snerl", "sincro", "mvp3d_multiview", "mvp3d_single"}


KEY = "drawer-open/episode_000004.h5"


@pytest.mark.parametrize("mutate,reason", [
    (lambda r, e: r.update(status="pending"), "calibration_validated 'pending'"),
    (lambda r, e: r.update(status="failed", failure={"message": "ambiguous"}), "calibration_validated 'failed'"),
    (lambda r, e: r.update(input_sha256="0" * 64), "input_sha256"),
    (lambda r, e: r.update(output_format="rw-calibration/1"), "output_format"),
    (lambda r, e: r["metrics"].update(decision="rejected"), "decision 'rejected'"),
    (lambda r, e: r["metrics"].update(calibrate_build_id="cal-build-0"), "not the current calibrate build"),
    (lambda r, e: e["calibrate"]["metrics"].update(build_id="cal-build-2"), "not the current calibrate build"),
    (lambda r, e: r["metrics"].pop("cameras"), "metrics.cameras"),
])
def test_calibration_hold_fails_closed_per_episode_without_touching_the_base_gate(mutate, reason):
    m = manifest(validated=True)
    stages = m["episodes"][KEY]["preprocessing"]["stages"]
    mutate(stages[gate.CALIBRATION_STAGE], stages)
    if stages[gate.CALIBRATION_STAGE]["status"] != "done":  # what set_stage's aggregate would say
        m["episodes"][KEY]["preprocessing"]["status"] = stages[gate.CALIBRATION_STAGE]["status"]
    rep = gate.evaluate(m, ["drawer-open"])
    assert rep["passed"] and rep["tasks"]["drawer-open"]["valid_and_preprocessed"] == 100  # 100-episode gate as before
    hold = rep["calibration_hold"]["drawer-open"]
    assert not hold["passed"] and not hold["episodes"] and hold["validated"] == 99
    assert any(reason in p for p in hold["problems"]), hold["problems"]
    assert gate.task_passed(rep, "drawer-open", "e2e_cnn") and not gate.task_passed(rep, "drawer-open", "snerl")


def test_aggregate_status_must_be_explained_by_the_hold_stage():
    m = manifest()
    m["episodes"][KEY]["preprocessing"]["stages"]["lift3d"] = {"status": "running", "code_version": "v"}
    m["episodes"][KEY]["preprocessing"]["status"] = "running"
    assert "stages not done: ['lift3d']" in " ".join(gate.evaluate(m, ["drawer-open"])["tasks"]["drawer-open"]
                                                     ["problems"])


@pytest.mark.parametrize("status", [None, "UNKNOWN", "done", "running"])
def test_open_hold_does_not_excuse_a_missing_or_inconsistent_aggregate(status):
    """Codex review: with calibration_validated pending, set_stage's aggregate is exactly "pending"; anything else
    (missing, unknown, stale) fails the base gate for that episode."""
    m = manifest(validated=True)
    pre = m["episodes"][KEY]["preprocessing"]
    pre["stages"][gate.CALIBRATION_STAGE]["status"] = "pending"
    pre["status"] = status
    rep = gate.evaluate(m, ["drawer-open"])
    assert not rep["passed"] and any(KEY in x for x in rep["tasks"]["drawer-open"]["problems"])
    pre["status"] = "pending"
    assert gate.evaluate(m, ["drawer-open"])["passed"]


def test_ingest_worker_set_stage_accepts_the_calibration_record(tmp_path):
    """ING's set_stage (the committed tool; used read-only, on a temporary manifest) takes the record CAL will write,
    and the gate reads the resulting manifest entry."""
    import importlib.util
    import json

    tool = Path(__file__).resolve().parents[1] / "runs/setup/tools/realworld_ingest_20261106/ingest_worker.py"
    if not tool.is_file():
        pytest.skip("ingest worker tool not present in this checkout")
    import sys

    sys.path.insert(0, str(tool.parent))
    try:
        spec = importlib.util.spec_from_file_location("ingest_worker_under_test", tool)
        iw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(iw)
    finally:
        sys.path.remove(str(tool.parent))
    m = manifest(n_drawer=100, n_mug=0)
    m["counts"] = {"drawer-open": {"complete": True, "target": 100}}
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    sha = m["episodes"][KEY]["sha256"]
    rec = calibration_record(sha)
    rec.update(code_commit="0" * 40, host="192.168.10.22", outputs="/remote/calibration_validation/report.json",
               started_at="2026-11-06T12:00:00Z", finished_at="2026-11-06T12:30:00Z", notes="multistart + mask "
               "sensitivity + independent robot frames")
    iw.set_stage(tmp_path, KEY, gate.CALIBRATION_STAGE, rec)
    doc = json.loads((tmp_path / "manifest.json").read_text())
    stored = doc["episodes"][KEY]["preprocessing"]["stages"][gate.CALIBRATION_STAGE]
    assert stored["status"] == "done" and stored["stage"] == gate.CALIBRATION_STAGE
    rep = gate.evaluate(doc, ["drawer-open"])
    assert rep["calibration_hold"]["drawer-open"]["validated"] == 1
    iw.set_stage(tmp_path, KEY, gate.CALIBRATION_STAGE, {**rec, "status": "pending"})  # flips the aggregate
    doc = json.loads((tmp_path / "manifest.json").read_text())
    assert doc["episodes"][KEY]["preprocessing"]["status"] == "pending"
    rep = gate.evaluate(doc, ["drawer-open"])
    assert rep["passed"] and gate.task_passed(rep, "drawer-open", "dynamo")  # not held by the pending record
    with pytest.raises(ValueError, match="unknown stage fields"):
        iw.set_stage(tmp_path, KEY, gate.CALIBRATION_STAGE, {**rec, "calibrate_build_id": "x"})  # top level: refused


# ---------------------------------------------------------------------------------------------- plans
def test_plan_under_the_calibration_hold_emits_only_non_geometric_rows():
    rep = gate.evaluate(manifest(), ["drawer-open", "mug-hanging"])
    plan = launch.plan(rep, ["drawer-open", "mug-hanging"], "abc123")
    rows = {r["id"]: r for r in plan["rows"]}
    assert {r["baseline"] for r in rows.values() if r["role"] == "realworld_pretrain"} == {"dynamo", "reviwo"}
    assert {r["fql_condition"] for r in rows.values() if r["role"] == "realworld_fql"} == {"dynamo", "reviwo",
                                                                                         "e2e_cnn"}
    assert not any(r["calibration_dependent"] for r in rows.values())
    held = set(plan["refused"])
    assert {"rw-snerl-drawer-open-pre", "rw-fql-snerl-drawer-open", "rw-fql-mvp3d_single-mug-hanging"} <= held
    assert any("calibration_validated" in x for x in plan["refused"]["rw-snerl-drawer-open-pre"])


def test_plan_rows_declare_roles_dependencies_and_are_not_registered():
    rep = gate.evaluate(manifest(validated=True), ["drawer-open", "mug-hanging"])
    plan = launch.plan(rep, ["drawer-open", "mug-hanging"], "abc123")
    assert plan["registered"] is False and not plan["refused"] and plan["remote_host"] == "192.168.10.22"
    rows = {r["id"]: r for r in plan["rows"]}
    pre = [r for r in rows.values() if r["role"] == "realworld_pretrain"]
    assert len(pre) == 5 * 2 and all(r["host"] == "remote" for r in rows.values())  # task-specific only
    assert all("configs/realworld/baselines/" in " ".join(r["args"]) and "joint" not in r["id"] for r in pre)
    fql = [r for r in rows.values() if r["role"] == "realworld_fql"]
    assert len(fql) == 2 * 7 * (1 + 3)  # per task: 7 conditions x (dataset build + 3 seeds)
    assert len(rows) == 66
    for r in fql:
        for d in r["deps"]:
            assert d in rows and rows[d]["priority"] <= r["priority"]
    assert all(r["priority"] < f["priority"] for r in pre for f in fql)  # baselines first
    run = rows["rw-fql-mvp3d_single-mug-hanging-s1"]
    assert run["deps"] == ["rw-fqldata-mvp3d_single-mug-hanging"]
    assert rows[run["deps"][0]]["deps"] == ["rw-mvp3d-mug-hanging-pre"]
    assert "condition=mvp3d_single" in run["args"] and run["planned_commit"] == "abc123"
    assert run["calibration_dependent"] and run["fql_budget"]["alpha"] == 10.0
    e2e = rows["rw-fql-e2e_cnn-drawer-open-s2"]
    assert not e2e["calibration_dependent"] and any(a.startswith("data.store=") for a in e2e["args"])
    e2e_data = rows["rw-fqldata-e2e_cnn-drawer-open"]
    assert "--export" not in e2e_data["args"] and set(e2e_data["deps"]) == {"rw-dynamo-drawer-open-pre",
                                                                          "rw-reviwo-drawer-open-pre"}
    for r in fql:
        if r["script"] == "scripts/realworld_fql_dataset.py":
            assert r["args"][r["args"].index("--outcomes") + 1].endswith("/outcomes.json")
            assert "--assume-final-success" not in r["args"]
    assert {r["fql_seed"] for r in fql if "fql_seed" in r} == {0, 1, 2}


def test_preview_matches_a_fully_passed_plan_and_the_table():
    rep = gate.evaluate(manifest(validated=True), ["drawer-open", "mug-hanging"])
    full = launch.plan(rep, ["drawer-open", "mug-hanging"], "c")["rows"]
    prev = launch.plan_preview("c")["rows"]
    strip = lambda r: {k: v for k, v in r.items() if k != "realworld_gate_sha256"}  # noqa: E731
    assert [strip(r) for r in full] == [strip(r) for r in prev]
    table = launch.plan_table()
    assert len(table) == 2 * 7
    geo = {(t["task"], t["condition"]) for t in table if t["calibration_dependent"]}
    assert geo == {(task, c) for task in launch.TASKS for c in ("snerl", "sincro", "mvp3d_multiview", "mvp3d_single")}
    assert all(t["fql_seeds"] == [0, 1, 2] and t["fql_steps"] == 1_000_000 for t in table)


def test_role_classification_of_planned_rows_matches_the_declared_roles():
    """Rows must classify as their declared role under the role rules (script / config evidence): the scripts carry
    no Meta-World evidence and the real-world configs live under configs/realworld/."""
    rep = gate.evaluate(manifest(validated=True), ["drawer-open", "mug-hanging"])
    rows = launch.plan(copy.deepcopy(rep), ["drawer-open"], "x")["rows"]
    for r in rows:
        script = r["script"].rsplit("/", 1)[-1]
        if r["role"] == "realworld_pretrain":
            assert script.startswith("train_") and any(a.startswith("configs/realworld/") for a in r["args"])
        else:
            assert "realworld" in script and "fql" in script
        assert not any("metaworld" in a or "configs/baselines/" in a or "configs/rl/" in a for a in r["args"])
