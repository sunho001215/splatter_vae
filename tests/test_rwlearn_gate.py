"""Full-data gate (s4d/rwlearn/gate.py), the per-consumer calibration hold, and launch plans (s4d/rwlearn/launch.py) on
synthetic ingestion manifests in the ``rw-ingest-manifest`` v2 schema of runs/setup/tools/realworld_ingest_20261106
(the live manifest is not read; ingest_worker.set_stage or its validation contract uses a temporary manifest)."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from s4d.rwlearn import gate, launch

STAGES = gate.REQUIRED_STAGES


CAMS = ["135122070988", "135122073201", "135122076054"]


def gate_identities(store):
    """Identity section of an already bound synthetic gate report."""
    from s4d.rwlearn.identity import store_identities

    identities = store_identities(store)
    return {"episodes": store.episodes(), "store_episodes": store.episodes(),
            "episode_identities": {ep: {**rec, "store_episode": ep} for ep, rec in identities.items()}}


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
    if "train_data" in recs:
        recs["train_data"]["output_format"] = "s4d.realworld.train/1.1.0"
    if "qa" in recs:
        recs["qa"].update(output_format="rwprep-qa/3", metrics={"families": {
            f: {"passed": True} for f in gate.QA_FAMILIES}})
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
    assert t["valid_and_preprocessed"] == 100 and t["store_episodes"] == []  # bound only when a store is read
    identity = t["episode_identities"]["drawer-open/episode_000000.h5"]
    assert identity["raw_recording"] == "drawer-open/episode_000000.h5" and identity["store_episode"] is None


@pytest.mark.parametrize("duplicate", ["filename", "sha256"])
def test_gate_rejects_duplicate_raw_identity(duplicate):
    m = manifest()
    a, b = (m["episodes"][f"drawer-open/episode_{i:06d}.h5"] for i in (0, 2))
    if duplicate == "filename":
        # An opaque manifest key is valid when the record carries an explicit raw filename.
        m["episodes"]["drawer-open/alias"] = m["episodes"].pop("drawer-open/episode_000002.h5")
        a["filename"] = b["filename"] = "episode_000000.h5"
    else:
        m["episodes"]["drawer-open/episode_000002.h5"] = entry("drawer-open", 2, sha=a["sha256"])
    report = gate.evaluate(m, ["drawer-open"])
    assert not report["passed"]
    assert "duplicate raw identity" in " ".join(report["tasks"]["drawer-open"]["problems"])


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
    assert hold["episodes"]["drawer-open/episode_000000.h5"] == {"calibrate_build_id": "cal-build-1", "cameras": CAMS}


def test_consumer_classification_is_explicit():
    geo = {k for k, c in gate.CONSUMERS.items() if c.geometry}
    assert geo == {"ours", "snerl", "mvp3d", "sincro", "fql_sincro", "fql_ours", "mvp3d_single"}
    assert {k for k, c in gate.CONSUMERS.items() if not c.geometry} == {"reviwo", "dynamo", "e2e_cnn", "fql_reviwo"}
    from s4d.rwlearn.fql_data import CONDITIONS

    assert set(gate.FQL_CONSUMER) == set(CONDITIONS)  # every FQL condition inherits its encoder's class
    assert {c for c, k in gate.FQL_CONSUMER.items() if gate.CONSUMERS[k].geometry} == {
        "snerl", "sincro", "mvp3d_multiview", "mvp3d_single", "ours"}


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


def _set_stage_contract(state_dir, key, stage, record):
    """Validation and aggregation from runs/setup/tools/realworld_ingest_20261106/ingest_worker.py::set_stage
    (rw-ingest-worker/2). Minimal v2-manifest adapter for checkouts without that tool."""
    fields = {"status", "code_version", "stage_schema_version", "code_commit", "config_sha256", "input_sha256",
              "output_format", "outputs", "started_at", "finished_at", "host", "failure", "attempt", "notes", "metrics"}
    statuses = ("pending", "running", "done", "failed", "skipped")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", stage):
        raise ValueError("stage names are lowercase [a-z0-9_.-]")
    unknown = set(record) - fields
    if unknown:
        raise ValueError(f"unknown stage fields {sorted(unknown)} (allowed {sorted(fields)})")
    missing = [k for k in ("status", "code_version") if not record.get(k)]
    if missing:
        raise ValueError(f"stage record needs {missing}")
    if record["status"] not in statuses:
        raise ValueError(f"status must be one of {statuses}")
    if record["status"] == "done" and not record.get("output_format"):
        raise ValueError("a done stage must name its output_format (format name/version)")
    if record["status"] == "failed" and not record.get("failure"):
        raise ValueError("a failed stage must describe its failure")
    manifest_path = state_dir / "manifest.json"
    doc = json.loads(manifest_path.read_text())
    entry = doc["episodes"][key]
    if record.get("input_sha256") and record["input_sha256"] != entry.get("sha256"):
        raise ValueError("input_sha256 does not match the episode's sha256")
    pre = entry.setdefault("preprocessing", {"schema": "rw-preprocess", "schema_version": 1,
                                           "status": "pending", "stages": {}})
    rec = {"stage": stage, "stage_schema_version": int(record.get("stage_schema_version", 1)), "failure": None, **record}
    pre["stages"][stage] = rec
    statuses = [x.get("status") for x in pre["stages"].values()]
    pre["status"] = ("failed" if "failed" in statuses else "running" if "running" in statuses else
                     "done" if statuses and all(x in ("done", "skipped") for x in statuses) else "pending")
    manifest_path.write_text(json.dumps(doc))
    return rec


def test_ingest_worker_set_stage_accepts_the_calibration_record(tmp_path):
    """The real worker or its copied validation contract accepts CAL's record, and the gate reads the result."""

    tool = Path(__file__).resolve().parents[1] / "runs/setup/tools/realworld_ingest_20261106/ingest_worker.py"
    # Real-worker integration runs in the main checkout with /home/ws/data and the URDF.
    if tool.is_file():
        import importlib.util
        import sys

        sys.path.insert(0, str(tool.parent))
        try:
            spec = importlib.util.spec_from_file_location("ingest_worker_under_test", tool)
            iw = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(iw)
        finally:
            sys.path.remove(str(tool.parent))
        set_stage = iw.set_stage
    else:
        set_stage = _set_stage_contract
    m = manifest(n_drawer=100, n_mug=0)
    m["counts"] = {"drawer-open": {"complete": True, "target": 100}}
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    sha = m["episodes"][KEY]["sha256"]
    rec = calibration_record(sha)
    rec.update(code_commit="0" * 40, host="192.168.10.22", outputs="/remote/calibration_validation/report.json",
               started_at="2026-11-06T12:00:00Z", finished_at="2026-11-06T12:30:00Z", notes="multistart + mask "
               "sensitivity + independent robot frames")
    set_stage(tmp_path, KEY, gate.CALIBRATION_STAGE, rec)
    doc = json.loads((tmp_path / "manifest.json").read_text())
    stored = doc["episodes"][KEY]["preprocessing"]["stages"][gate.CALIBRATION_STAGE]
    assert stored["status"] == "done" and stored["stage"] == gate.CALIBRATION_STAGE
    rep = gate.evaluate(doc, ["drawer-open"])
    assert rep["calibration_hold"]["drawer-open"]["validated"] == 1
    set_stage(tmp_path, KEY, gate.CALIBRATION_STAGE, {**rec, "status": "pending"})  # flips the aggregate
    doc = json.loads((tmp_path / "manifest.json").read_text())
    assert doc["episodes"][KEY]["preprocessing"]["status"] == "pending"
    rep = gate.evaluate(doc, ["drawer-open"])
    assert rep["passed"] and gate.task_passed(rep, "drawer-open", "dynamo")  # not held by the pending record
    with pytest.raises(ValueError, match="unknown stage fields"):
        set_stage(tmp_path, KEY, gate.CALIBRATION_STAGE, {**rec, "calibrate_build_id": "x"})  # top level: refused


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


def unmeasured_scenes(monkeypatch, tasks=None):
    """Stand-in for a release whose new-task scene geometry is NOT measured yet (covering's was measured 2026-11-12)."""
    from s4d.rwlearn import configs

    for task in tasks or NEW_TASKS:
        monkeypatch.setitem(configs.SCENE_PARAMETERS, task, {"lookat": None, "pose_scale": None, "bounds": None})


def measured_scenes(monkeypatch):
    """Stand-in for a release whose new-task scene geometry was measured (here: the existing tasks' values)."""
    from s4d.rwlearn import configs

    for task in NEW_TASKS:
        monkeypatch.setitem(configs.SCENE_PARAMETERS, task, dict(configs.SCENE_PARAMETERS["drawer-open"]))


NEW_TASKS = ("covering", "pick-and-place", "drawer-open-v2")
SCENE_IDS = {f"rw-{b}-{{t}}-pre" for b in ("snerl", "mvp3d")} | {f"rw-fql-{c}-{{t}}" for c in
                                                                 ("snerl", "mvp3d_multiview", "mvp3d_single")}


@pytest.mark.parametrize("scenes", ["pending", "measured"])
def test_preview_matches_a_fully_passed_plan_and_the_table(scenes, monkeypatch):
    if scenes == "measured":
        measured_scenes(monkeypatch)
    else:
        unmeasured_scenes(monkeypatch)
    m = manifest(validated=True)
    m["counts"]["pick-and-place"] = {"complete": True, "target": 100}
    m["episodes"].update({f"pick-and-place/episode_{i:06d}.h5": entry("pick-and-place", i, validated=True)
                          for i in range(100)})
    m["counts"]["drawer-open-v2"] = {"complete": True, "target": 100}
    m["episodes"].update({f"drawer-open-v2/episode_{i:06d}.h5": entry("drawer-open-v2", i, validated=True)
                          for i in range(100)})
    covering = amended_manifest()  # covering passes through its 98-episode amendment
    m["counts"].update(covering["counts"])
    m["episodes"].update(covering["episodes"])
    rep = gate.evaluate(m, launch.TASKS, amendments=amendment())
    plan = launch.plan(rep, launch.TASKS, "c")
    full = plan["rows"]
    prev = launch.plan_preview("c")["rows"]
    strip = lambda r: {k: v for k, v in r.items() if k != "realworld_gate_sha256"}  # noqa: E731
    pending = {i.format(t=t) for t in NEW_TASKS for i in SCENE_IDS} if scenes == "pending" else set()
    assert set(plan["refused"]) == pending
    assert all("scene parameters not measured" in " ".join(plan["refused"][i]) for i in pending)
    # the preview is the plan once every gate passed and every scene is measured; pending SNeRL / 3D-MVP rows (and
    # their FQL datasets / runs) are the only difference
    unmeasured = lambda r: (r["realworld_target"] in NEW_TASKS and scenes == "pending"  # noqa: E731
                            and (r.get("baseline") or launch.FQL_ENCODER.get(r.get("fql_condition"))) in ("snerl",
                                                                                                         "mvp3d"))
    assert [strip(r) for r in full] == [strip(r) for r in prev if not unmeasured(r)]
    assert len(prev) == 33 * len(launch.TASKS) and len(full) == len(prev) - (len(NEW_TASKS) * 14 if scenes == "pending" else 0)
    table = launch.plan_table()
    assert len(table) == len(launch.TASKS) * 7
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


def test_only_3dmvp_fql_runs_standardise_their_frozen_features():
    """encoder.feature_norm is opt-in per condition: exactly the 3D-MVP FQL runs set it; every other row (pretraining,
    datasets, the other FQL conditions) keeps its arguments."""
    rows = launch.plan_preview()["rows"]
    flagged = {r["id"] for r in rows if "encoder.feature_norm=train_standardize" in r["args"]}
    assert flagged == {r["id"] for r in rows if r["script"] == "scripts/train_realworld_fql.py"
                       and r["fql_condition"] in ("mvp3d_single", "mvp3d_multiview")}
    assert len(flagged) == 2 * len(launch.SEEDS) * len(launch.TASKS)
    assert not any("feature_norm" in a for r in rows if r["id"] not in flagged for a in r["args"])


def _families(entry_, **passed):
    entry_["preprocessing"]["stages"]["qa"].update(
        output_format="rwprep-qa/3", metrics={"families": {f: {"passed": passed.get(f, True), "failed": []}
                                                           for f in gate.QA_FAMILIES}})


def test_qa_families_gate_each_consumer_on_its_own_families():
    """An episode without a multi-view window (M5) or with a failed geometry family blocks only the consumers that need
    that family; single-view non-geometric consumers keep the base gate (PREP2 2026-11-06)."""
    m = manifest(validated=True)
    for e in m["episodes"].values():
        _families(e)
    _families(m["episodes"]["mug-hanging/episode_000010.h5"], multiview=False)
    _families(m["episodes"]["drawer-open/episode_000004.h5"], geometry=False)
    rep = gate.evaluate(m, ["drawer-open", "mug-hanging"])
    assert rep["passed"] and rep["geometry_passed"]
    assert rep["families"]["mug-hanging"]["multiview"]["failed_episodes"] == ["mug-hanging/episode_000010.h5"]
    for consumer in ("dynamo", "e2e_cnn"):
        assert gate.task_passed(rep, "mug-hanging", consumer) and gate.task_passed(rep, "drawer-open", consumer)
    assert not gate.task_passed(rep, "mug-hanging", "reviwo") and gate.task_passed(rep, "drawer-open", "reviwo")
    for consumer in ("ours", "snerl", "mvp3d", "sincro", None):
        assert not gate.task_passed(rep, "mug-hanging", consumer) and not gate.task_passed(rep, "drawer-open", consumer)
    assert any("QA family multiview" in p for p in gate.task_problems(rep, "mug-hanging", "snerl"))
    assert not any("QA family" in p for p in gate.task_problems(rep, "mug-hanging", "dynamo"))


def test_reviwo_needs_only_the_m5_sets_of_a_calibration_optional_episode():
    """Every camera held (calibration_optional/1): the multi-view contract and geometry families fail, the derived
    ``multiview_sets`` family (M5 sets + robot motion) passes -> ReViWo, DynaMo, e2e_cnn and the single-view FQL
    conditions may train; every geometric consumer stays refused (2026-11-12)."""
    m = manifest()
    for e in m["episodes"].values():
        e["preprocessing"]["stages"]["qa"].update(output_format="rwprep-qa/3", metrics={"families": {
            "singleview": {"passed": True}, "multiview": {"passed": False}, "geometry": {"passed": False},
            "multiview_sets": {"passed": True}}})
    no_sets = m["episodes"]["drawer-open/episode_000004.h5"]["preprocessing"]["stages"]["qa"]["metrics"]["families"]
    no_sets["multiview_sets"]["passed"] = False
    rep = gate.evaluate(m, ["drawer-open", "mug-hanging"])
    assert rep["passed"] and not rep["geometry_passed"]
    assert rep["families"]["mug-hanging"]["multiview_sets"]["passed"]
    assert rep["families"]["drawer-open"]["multiview_sets"]["failed_episodes"] == ["drawer-open/episode_000004.h5"]
    for consumer in ("reviwo", "dynamo", "e2e_cnn", "fql_reviwo"):
        assert gate.task_passed(rep, "mug-hanging", consumer)
    assert not gate.task_passed(rep, "drawer-open", "reviwo") and gate.task_passed(rep, "drawer-open", "dynamo")
    assert any("QA family multiview_sets" in p for p in gate.task_problems(rep, "drawer-open", "reviwo"))
    for consumer in ("ours", "snerl", "mvp3d", "sincro", "fql_sincro", "fql_ours", "mvp3d_single", None, "unknown"):
        assert not gate.task_passed(rep, "mug-hanging", consumer)


def test_multiview_sets_derives_from_a_passed_multiview_family_and_older_reports():
    """Records and reports written before the derived family keep their ReViWo verdict: a passed ``multiview``
    family contains both M5 gates; a report without the key falls back to its ``multiview`` entry."""
    m = manifest(validated=True)
    for e in m["episodes"].values():
        _families(e)  # QA_FAMILIES only, no multiview_sets key (the pre-2026-11-12 QA)
    _families(m["episodes"]["mug-hanging/episode_000010.h5"], multiview=False)
    stages = m["episodes"]["drawer-open/episode_000000.h5"]["preprocessing"]["stages"]
    assert gate.qa_families(stages) == {"singleview": True, "multiview": True, "geometry": True,
                                        "multiview_sets": True}
    rep = gate.evaluate(m, ["drawer-open", "mug-hanging"])
    assert rep["families"]["drawer-open"]["multiview_sets"]["passed"]
    assert not rep["families"]["mug-hanging"]["multiview_sets"]["passed"]
    old = copy.deepcopy(rep)
    for fams in old["families"].values():
        fams.pop("multiview_sets")
    for report in (rep, old):
        assert gate.task_passed(report, "drawer-open", "reviwo") and not gate.task_passed(report, "mug-hanging", "reviwo")
        assert gate.task_passed(report, "drawer-open", "ours") and not gate.task_passed(report, "mug-hanging", "ours")


@pytest.mark.parametrize("entry", [{"passed": False}, None, "passed", {"passed": "true"}])
def test_an_explicit_derived_verdict_is_never_overridden(entry):
    """Only an ABSENT derived family falls back to its source; an explicit failure, null or malformed entry fails
    closed even when the full multi-view family passed (Codex review 2026-11-12, P2)."""
    m = manifest()
    e = m["episodes"]["drawer-open/episode_000004.h5"]
    _families(e)
    e["preprocessing"]["stages"]["qa"]["metrics"]["families"]["multiview_sets"] = entry
    assert gate.qa_families(e["preprocessing"]["stages"])["multiview_sets"] is False
    rep = gate.evaluate(m, ["drawer-open"])
    assert not gate.task_passed(rep, "drawer-open", "reviwo")
    assert gate.task_passed(rep, "drawer-open", "dynamo")
    for fams in rep["families"].values():
        fams["multiview_sets"] = entry
    assert not gate.task_passed(rep, "drawer-open", "reviwo")


@pytest.mark.parametrize("fmt", [None, "rwprep-qa/1", "rwprep-qa/2", "rwprep-qa/3"])
@pytest.mark.parametrize("with_families", [False, True])
def test_only_current_qa_with_explicit_families_can_pass(fmt, with_families):
    m = manifest(validated=True)
    rec = m["episodes"]["mug-hanging/episode_000010.h5"]["preprocessing"]["stages"]["qa"]
    rec.update(output_format=fmt, code_version="qa@2" if fmt == "rwprep-qa/2" else "qa@3")
    if not with_families:
        rec.pop("metrics")
    rep = gate.evaluate(m, ["mug-hanging"])
    expected = fmt == "rwprep-qa/3" and with_families
    for consumer in (*gate.CONSUMERS, None, "unknown"):
        assert gate.task_passed(rep, "mug-hanging", consumer) is expected
    assert gate.consumer_families("unknown") == gate.QA_FAMILIES
    assert set(gate.CONSUMER_FAMILIES) == set(gate.CONSUMERS)  # every classified consumer declares its families


@pytest.mark.parametrize("fmt,passed", [(None, False), ("s4d.realworld.train/1", False),
    ("s4d.realworld.train/1.0.0", False), ("s4d.realworld.train/1.0.99", False), ("unknown/1.1.0", False),
    ("s4d.realworld.train/1.1", False), ("s4d.realworld.train/1.1.0", True),
    ("s4d.realworld.train/1.1.1", True), ("s4d.realworld.train/1.2.0", True)])
def test_every_episode_needs_current_train_data_even_with_current_qa(fmt, passed):
    m = manifest(validated=True)
    m["episodes"][KEY]["preprocessing"]["stages"]["train_data"].update(
        output_format=fmt, code_version="train_data@2" if passed else "train_data@1")
    # Stage-name overrides cannot waive the minimum evidence formats.
    rep = gate.evaluate(m, ["drawer-open"], required_stages=("normalized",))
    for consumer in (*gate.CONSUMERS, None, "unknown"):
        assert gate.task_passed(rep, "drawer-open", consumer) is passed
    if not passed:
        assert KEY in rep["tasks"]["drawer-open"]["problems"][0]
        assert "train_data output_format" in rep["tasks"]["drawer-open"]["problems"][0]


def test_pre_family_gate_reports_cannot_release_consumers():
    rep = gate.evaluate(manifest(validated=True), ["drawer-open"])
    rep.pop("families")
    for consumer in (*gate.CONSUMERS, None):
        assert not gate.task_passed(rep, "drawer-open", consumer)


@pytest.mark.parametrize("families", [[], "passed", {"singleview": True}, {"singleview": {"passed": "true"}}])
def test_malformed_qa_families_fail_closed(families):
    m = manifest(validated=True)
    m["episodes"][KEY]["preprocessing"]["stages"]["qa"].update(
        output_format="rwprep-qa/3", metrics={"families": families})
    rep = gate.evaluate(m, ["drawer-open"])
    assert not gate.task_passed(rep, "drawer-open", "dynamo")


@pytest.mark.parametrize("fmt", ["rwprep-qa/4", "rwprep-qa/bad", "unknown/2", ""])
def test_unknown_qa_schema_is_never_treated_as_legacy(fmt):
    m = manifest(validated=True)
    m["episodes"][KEY]["preprocessing"]["stages"]["qa"].update(output_format=fmt)
    rep = gate.evaluate(m, ["drawer-open"])
    assert not gate.task_passed(rep, "drawer-open", "dynamo")


def test_every_singleview_fql_condition_ignores_cross_camera_family_but_keeps_geometry():
    from s4d.rwlearn.fql_data import CONDITIONS

    m = manifest(validated=True)
    for e in m["episodes"].values():
        _families(e, multiview=False)
    rep = gate.evaluate(m, ["drawer-open"])
    for cond, spec in CONDITIONS.items():
        consumer = gate.FQL_CONSUMER[cond]
        assert gate.task_passed(rep, "drawer-open", consumer) is not spec.multiview
    _families(m["episodes"][KEY], multiview=False, geometry=False)
    rep = gate.evaluate(m, ["drawer-open"])
    assert not gate.task_passed(rep, "drawer-open", gate.FQL_CONSUMER["mvp3d_single"])
    assert not gate.task_passed(rep, "drawer-open", gate.FQL_CONSUMER["sincro"])
    assert gate.task_passed(rep, "drawer-open", gate.FQL_CONSUMER["reviwo"])


@pytest.mark.parametrize("task", NEW_TASKS)
def test_new_tasks_have_an_independent_100_episode_gate(task, tmp_path, monkeypatch):
    from s4d.rwlearn.tasks import TARGETS

    unmeasured_scenes(monkeypatch, [task])

    m = {"schema": gate.SCHEMA, "schema_version": gate.SCHEMA_VERSION,
         "counts": {task: {"complete": True, "target": 100}},
         "episodes": {f"{task}/episode_{i * 2:06d}.h5": entry(task, i * 2, validated=True) for i in range(100)}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m))
    # the default 100-episode rule (covering's 98-episode amendment is tested below)
    default = gate.evaluate_file(path, amendments={})  # the default enumerates every registered target, each alone
    assert tuple(default["tasks"]) == TARGETS and not default["passed"]
    assert default["tasks"][task]["passed"] and default["calibration_hold"][task]["passed"]
    assert all(not default["tasks"][t]["passed"] for t in TARGETS if t != task)
    plan = launch.plan(default, [task], "new-pin")  # scene geometry still unmeasured: SNeRL / 3D-MVP refused
    assert {r["realworld_target"] for r in plan["rows"]} == {task} and len(plan["rows"]) == 33 - 14
    assert set(plan["refused"]) == {i.format(t=task) for i in SCENE_IDS}
    assert {r.get("baseline") for r in plan["rows"] if r["role"] == "realworld_pretrain"} == {"sincro", "reviwo",
                                                                                            "dynamo"}
    measured_scenes(monkeypatch)
    plan = launch.plan(default, [task], "new-pin")
    assert len(plan["rows"]) == 33 and not plan["refused"]
    m["episodes"].pop(next(iter(m["episodes"])))
    rep = gate.evaluate(m, [task], amendments={})
    assert not rep["tasks"][task]["passed"] and not launch.plan(rep, [task], "new-pin")["rows"]


def test_documented_training_plan_is_the_preview():
    """docs/realworld_training_plan.yaml is the plan preview of every target. Only the FQL rows' stale consumer labels
    (written before the FQL-specific QA family aliases) may differ; the frozen non-geometric registrar
    (runs/setup/tools/register_realworld_nongeometric_20261106.py) compares the two the same way."""
    import yaml

    doc = yaml.safe_load((Path(__file__).resolve().parents[1] / "docs/realworld_training_plan.yaml").read_text())
    rows = copy.deepcopy(doc["rows"])
    for row in rows:
        cond = row.get("fql_condition")
        if cond is not None:
            assert row["realworld_consumer"] in {launch.FQL_CONSUMER[cond], launch.FQL_ENCODER[cond] or cond}
            row["realworld_consumer"] = launch.FQL_CONSUMER[cond]
    assert rows == launch.plan_preview()["rows"]
    assert doc["table"] == launch.plan_table()


# ---------------------------------------------------------------------------------------------- task amendments
EXCLUDED = (57, 61)  # covering: device timestamps not strictly increasing (AMEND-covering-98.md)


def amendment(task="covering", excl=EXCLUDED, **over):
    rec = {"format": gate.AMENDMENT_FORMAT, "task": task, "version": 1, "target": 100 - len(excl),
           "base_target": 100, "decision": "permit the valid episodes", "decided_by": "user",
           "decided_at": "2026-11-10T13:00:00+09:00", "document": {"path": "AMEND.md", "sha256": "d" * 64},
           "episodes": {f"{task}/episode_{i:06d}.h5": hashlib.sha256(f"{task}-{i}".encode()).hexdigest()
                        for i in range(100) if i not in excl},
           "excluded": {f"{task}/episode_{i:06d}.h5": {"reason": "timestamps not strictly increasing"}
                        for i in excl}}
    rec.update(over)
    return {task: {"record": rec, "path": f"/cfg/{task}-amend.json", "sha256": "e" * 64}}


def amended_manifest(task="covering", excluded=EXCLUDED, validated=True):
    eps = {}
    for i in range(100):
        e = entry(task, i, validated=validated)
        if i in excluded:  # as the ingest worker records an invalid recording: no sha256, never transferred
            e.update(sha256=None, validation={"status": "invalid"}, transfer={"status": "pending"}, preprocessing={})
        eps[f"{task}/episode_{i:06d}.h5"] = e
    return {"schema": gate.SCHEMA, "schema_version": gate.SCHEMA_VERSION, "episodes": eps,
            "counts": {task: {"complete": False, "target": 100}}}  # the ingest worker's 100-rule stays unmet


def test_amendment_releases_exactly_its_episodes_and_keeps_every_other_check():
    rep = gate.evaluate(amended_manifest(), ["covering"], amendments=amendment())
    t = rep["tasks"]["covering"]
    assert rep["passed"] and rep["geometry_passed"], t["problems"]
    assert t["valid_and_preprocessed"] == 98 and t["target"] == 98 and len(t["episodes"]) == 98
    assert t["amendment"]["target"] == 98 and t["amendment"]["excluded"] == [
        "covering/episode_000057.h5", "covering/episode_000061.h5"] and t["amendment"]["sha256"] == "e" * 64
    assert rep["calibration_hold"]["covering"]["validated"] == 98
    assert gate.task_passed(rep, "covering", "snerl") and gate.task_passed(rep, "covering", "e2e_cnn")
    # without the amendment the same manifest fails the default rule
    rep = gate.evaluate(amended_manifest(), ["covering"], amendments={})
    text = " ".join(rep["tasks"]["covering"]["problems"])
    assert not rep["passed"] and "ingestion not complete" in text and "need exactly 100" in text
    # every other check is unchanged: one episode's open stage fails the amended task as before
    m = amended_manifest()
    m["episodes"]["covering/episode_000004.h5"]["preprocessing"]["stages"].pop("sam3_masks")
    rep = gate.evaluate(m, ["covering"], amendments=amendment())
    assert not rep["passed"] and "need exactly 98" in " ".join(rep["tasks"]["covering"]["problems"])
    # the calibration hold needs all 98
    m = amended_manifest()
    m["episodes"]["covering/episode_000004.h5"]["preprocessing"]["stages"].pop(gate.CALIBRATION_STAGE)
    rep = gate.evaluate(m, ["covering"], amendments=amendment())
    assert rep["passed"] and not rep["geometry_passed"] and rep["calibration_hold"]["covering"]["validated"] == 97


def test_amendment_is_task_specific_and_the_default_stays_100():
    m = manifest(n_drawer=98)
    m["episodes"].update(amended_manifest()["episodes"])
    m["counts"]["covering"] = {"complete": False, "target": 100}
    rep = gate.evaluate(m, ["drawer-open", "mug-hanging", "covering"], amendments=amendment())
    assert rep["tasks"]["covering"]["passed"] and rep["tasks"]["mug-hanging"]["passed"]
    assert not rep["tasks"]["drawer-open"]["passed"] and rep["tasks"]["drawer-open"]["target"] == 100
    assert rep["tasks"]["drawer-open"]["amendment"] is None
    assert "need exactly 100" in " ".join(rep["tasks"]["drawer-open"]["problems"])


@pytest.mark.parametrize("mutate,reason", [
    (lambda m: m["episodes"].update({"covering/episode_000100.h5": entry("covering", 100)}),
     "not in the gate amendment"),
    (lambda m: m["episodes"].update({"covering/episode_000057.h5": entry("covering", 57)}),
     "excluded by the amendment but validation is 'valid'"),
    (lambda m: m["episodes"].pop("covering/episode_000061.h5"), "amendment episode missing from the manifest"),
    (lambda m: m["episodes"].pop("covering/episode_000005.h5"), "amendment episode missing from the manifest"),
    (lambda m: m["episodes"].update({"covering/episode_000005.h5": entry("covering", 5, sha="f" * 64)}),
     "sha256 differs from the gate amendment"),
    (lambda m: m["counts"]["covering"].update(target=98), "amendment base_target 100"),
])
def test_amendment_fails_closed_on_any_other_episode_set(mutate, reason):
    m = amended_manifest()
    mutate(m)
    rep = gate.evaluate(m, ["covering"], amendments=amendment())
    text = " ".join(rep["tasks"]["covering"]["problems"])
    assert not rep["passed"] and reason in text, text


@pytest.mark.parametrize("over,reason", [
    ({"format": "x"}, "format"),
    ({"target": 99}, "target 99 != 98"),
    ({"base_target": 98}, "base_target"),
    ({"document": {"path": "AMEND.md"}}, "document"),
    ({"decided_by": ""}, "provenance"),
    ({"excluded": {"covering/episode_000057.h5": {}}}, "reason"),
    ({"excluded": {"drawer-open/episode_000057.h5": {"reason": "x"}}}, "manifest keys of the task"),
])
def test_malformed_amendment_records_are_refused(over, reason, tmp_path):
    rec = amendment(**over)["covering"]["record"]
    assert reason in gate.amendment_problem(rec)
    (tmp_path / "a.json").write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="gate amendment"):
        gate.load_amendments(tmp_path)


def test_amendment_records_never_duplicate_content_or_tasks(tmp_path):
    rec = amendment()["covering"]["record"]
    dup = copy.deepcopy(rec)
    dup["episodes"]["covering/episode_000001.h5"] = dup["episodes"]["covering/episode_000000.h5"]
    assert "duplicated content" in gate.amendment_problem(dup)
    overlap = copy.deepcopy(rec)
    overlap["excluded"]["covering/episode_000000.h5"] = {"reason": "x"}
    assert gate.amendment_problem(overlap)
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="two gate amendments"):
        gate.load_amendments(tmp_path)


# the versioned amendment records (user decisions 2026-11-10): target, excluded indices, document, index range and
RECORDS = {
    # the sha256 of the record file (immutable: changing an approved key or checksum needs a new dated decision)
    "covering": (98, (57, 61), "AMEND-covering-98.md", (0, 106),
                 "685fdd216aad8f029a7c584fb9ceb68fc470b32efb93ed29a816af58de980566"),
    "pick-and-place": (97, (14, 80, 101), "AMEND-pick-and-place-97.md", (2, 114),
                       "098231dc642ab448b4c6f40ee63d0baf67d5dca8a72c7dc73541804259a353a5"),
    "drawer-open-v2": (98, (77, 79), "AMEND-drawer-open-v2-98.md", (1, 117),
                       "041042f1ff98d220b20dfd5ee159ed113eb79ad29a357134f246e8e5514ccbe7"),
}


def test_the_versioned_amendment_records_are_exactly_the_user_decisions():
    """Exactly the dated, hash-bound decisions, including the binding drawer-v2 raw rejection amendment."""
    assert set(gate.load_amendments()) == set(RECORDS)


@pytest.mark.parametrize("task", sorted(RECORDS))
def test_the_versioned_amendment_record(task):
    target, excluded, doc, (first, last), digest = RECORDS[task]
    loaded = gate.load_amendments()[task]
    assert loaded["sha256"] == digest  # the approved episode list and checksums, byte for byte
    rec = loaded["record"]
    assert rec["target"] == len(rec["episodes"]) == target and rec["base_target"] == gate.TARGET
    assert sorted(rec["excluded"]) == [f"{task}/episode_{i:06d}.h5" for i in excluded]
    validator = "dsreset inspect_raw/1" if task == "drawer-open-v2" else "rw-ingest-validator/1"
    assert all((v.get("validation_reasons") or v.get("reason")) and v["validator"] == validator
               and len(v["raw_sha256"]) == 64 for v in rec["excluded"].values())
    assert rec["document"]["path"].endswith(doc)
    indices = sorted(int(k[-9:-3]) for k in rec["episodes"])
    assert indices[0] == first and indices[-1] == last and len(set(indices)) == target
    assert not set(indices) & set(excluded) and len(set(rec["episodes"].values())) == target


def test_a_supplied_amendment_map_is_validated_like_the_versioned_records():
    bad = amendment()
    bad["covering"]["record"]["target"] = 0
    rep = gate.evaluate(amended_manifest(), ["covering"], amendments=bad)
    assert not rep["passed"] and "gate amendment" in rep["tasks"]["covering"]["problems"][0]
    other = amendment(task="pick-and-place")
    rep = gate.evaluate(amended_manifest(), ["covering"], amendments={"covering": other["pick-and-place"]})
    assert not rep["passed"] and "record task" in rep["tasks"]["covering"]["problems"][0]
    assert not gate.task_passed(rep, "covering", "e2e_cnn")
