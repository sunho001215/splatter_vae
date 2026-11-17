"""Task-specific calibration acceptance override (user decision 2026-11-17, drawer-open-v2 only).

Covers the committed record, the exact-evidence acceptance rule with its negative cases (other task / camera / gate /
pair, extra failing gate or sub-check, value / threshold / cohort drift, tampered blocks), the publisher (original
records never rewritten, separate override decision, automatic verdict kept), the unchanged robot path, the offline
consumers (route gate, native-depth session, calibration_validated records) and the process-local phase B rig mode.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from s4d.realworld import io as rio
from s4d.realworld.calib import acceptance_override as AO
from s4d.realworld.calib import criteria as C
from s4d.realworld.calib.dw.publish import CAMS
from tests.test_realworld_calibration_release import _calibrate_with, _release_doc
from tests.test_realworld_calibration_release import rig_episode as rig_episode_fixture

rig_episode = rig_episode_fixture  # shared module-scoped synthetic rig-mode episode

REPO = Path(__file__).resolve().parents[1]
PAIR = "135122070988-135122073201"
G1R = "G1: centre radius 8.8 mm > 8 (loco-x)"
DTP = f"{PAIR}: long-offset median difference +0.014 px (95 % CI upper +0.099) > 0.05"
G5R = f"G5b: dt extension not validated ({DTP})"
GATES = ("G1", "G2", "G3", "G4", "G5a", "G5b", "G6")
COMMITTED = AO.RECORDS_DIR / AO.RECORDS["drawer-open-v2"]  # captured before any test redirects the record directory


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _v2(g1_fail):
    return {"g1": {"radius_centre_m": 0.008770722160007146 if g1_fail else 0.005, "worst": {"centre_m": "loco-x"},
                   "thresholds": {"radius_centre_m": 0.008}, "reasons": [G1R] if g1_fail else []},
            "g5b": {"reasons": [G5R], "moved": [], "missing": [], "verifiable_fraction": 1.0,
                    "dt_extension": {"max_diff_px": 0.05, "problems": [DTP], "pairs": {PAIR: {
                        "n_short": 89, "n_long": 9, "median_diff_px": 0.014269243648465879,
                        "median_diff_ci95_px": [-0.0098, 0.09885288634467633], "p90_diff_px": 0.07094970276816659,
                        "long_violations": []}}}}}


def _held_doc(cams, episodes, g1cam, task="synthetic"):
    doc = _release_doc(cams, episodes[0], task=task)
    doc["episodes"] = list(episodes)
    doc["fixedness_evidence"] = {c: {e: {"registration": "fixed", "epipolar": "ok"} for e in episodes} for c in cams}
    for c in cams:
        g1 = c == g1cam
        doc["cameras"][c].update(status="held", held_by=["G1", "G5b"] if g1 else ["G5b"],
                                 reasons=([G1R] if g1 else []) + [G5R],
                                 checks={g: not (g == "G5b" or (g1 and g == "G1")) for g in GATES},
                                 diagnostics={"criteria_v2": _v2(g1)})
    return doc


def _record(cams, g1cam, episodes, cohort, shas, task="synthetic"):
    rec = json.loads(COMMITTED.read_text())
    rec.update(task=task, id=f"acceptance-override-{task}")
    rec["cameras"] = {c: {"gates": ["G1", "G5b"] if c == g1cam else ["G5b"]} for c in cams}
    g1, g5 = rec["exceptions"]
    g1.update(cameras=[g1cam], worst_solve="loco-x", reason=G1R)
    g5.update(cameras=list(cams), long_offset_cohort=cohort, problem=DTP, reason=G5R)
    auto = rec["automatic_acceptance"]
    auto.update(decision={"cameras": {c: "held" for c in cams}, "accepted": False},
                decision_sha256=shas["auto"], verdict_release1_sha256=shas["verdict"])
    rec["provenance"].update(spec_sha256=shas["spec"], freeze_sha256=shas["freeze"])
    rec["retained"] = {"episodes": len(episodes), "cameras": len(cams), "excluded_episodes": [],
                       "excluded_cameras": []}
    return rec


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A held synthetic task with the exact recorded evidence, its committed record and publisher inputs."""
    def make(cams=tuple(CAMS), episodes=tuple(f"synthetic-{n:06d}" for n in range(12)), g1cam=CAMS[1],
             cohort_n=3, task="synthetic"):
        root = tmp_path / f"w{len(list(tmp_path.iterdir()))}"
        work, bundle, recs = root / "work", root / "bundle", root / "overrides"
        for d in (work, bundle, recs):
            d.mkdir(parents=True)
        automatic = {"cameras": {c: "held" for c in cams}, "accepted": False}
        AO.write_once(work / "automatic-decision.json", automatic)
        AO.write_once(work / "verdict-release1.json", {"rig_dw_gates": {}, "samfree": {}})
        (bundle / "spec.json").write_text(json.dumps({"task": task}))
        (bundle / "FREEZE.json").write_text(json.dumps({"protocol": "test"}))
        cohort = sorted(episodes[:cohort_n])
        static = {"dt": {"rows": [{"episode": e, "pair": 1, "file": 0, "abs_dt_ms": 21.5 if e in cohort else 5.0}
                                  for e in episodes]}}
        shas = {"auto": _sha(work / "automatic-decision.json"), "verdict": _sha(work / "verdict-release1.json"),
                "spec": _sha(bundle / "spec.json"), "freeze": _sha(bundle / "FREEZE.json")}
        rec = _record(list(cams), g1cam, episodes, cohort, shas, task)
        (recs / "rec.json").write_text(json.dumps(rec, indent=1))
        monkeypatch.setattr(AO, "RECORDS_DIR", recs)
        monkeypatch.setattr(AO, "RECORDS", {task: "rec.json"})
        doc = _held_doc(list(cams), list(episodes), g1cam, task)
        kw = dict(task=task, work=work, specpath=bundle / "spec.json", static=static,
                  aliases={e: e for e in episodes}, automatic=automatic, commit=rec["automatic_acceptance"][
                      "criteria_commit"])
        return {"doc": doc, "kw": kw, "rec": rec, "work": work, "static": static, "cams": list(cams)}
    return make


def _script(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _published(world_state):
    return AO.apply_publisher_override(copy.deepcopy(world_state["doc"]), **world_state["kw"])


# ----------------------------------------------------------------------------------------------- committed record
def test_committed_record_names_exactly_the_authorized_exceptions_and_provenance():
    rec, sha, path = AO.record_for("drawer-open-v2")
    assert path == "s4d/realworld/calib/overrides/drawer-open-v2-20261117.json" and sha == _sha(REPO / path)
    assert AO.record_for("drawer-open") is None and AO.record_for("covering") is None and AO.record_for(None) is None
    assert rec["task"] == "drawer-open-v2" and rec["status"] == AO.STATUS == "validated_by_override"
    assert rec["cameras"] == {"135122070988/color": {"gates": ["G5b"]}, "135122073201/color": {"gates": ["G1", "G5b"]},
                              "135122076054/color": {"gates": ["G5b"]}}
    g1, g5 = rec["exceptions"]
    assert g1["cameras"] == ["135122073201/color"] and g1["gate"] == "G1" and g1["metric"] == "radius_centre_m"
    assert round(g1["measured"] * 1e3, 6) == 8.770722 and g1["threshold"] == C.G1_V2["radius_centre_m"] == 0.008
    assert g5["pair"] == PAIR and round(g5["measured"], 6) == 0.098853
    assert g5["threshold"] == C.G5B_DT_MAX_DIFF_PX == 0.05 and g5["cameras"] == list(CAMS)
    assert g5["long_offset_cohort"] == [f"drawer-open-v2-success-episode_{n:06d}" for n in (1, 2, 3, 4, 6, 7, 8, 9, 10)]
    auto, prov = rec["automatic_acceptance"], rec["provenance"]
    assert auto["passed"] is False and auto["decision"]["accepted"] is False and auto["thresholds_changed"] is False
    assert auto["decision_sha256"].startswith("64343dd6") and auto["verdict_release1_sha256"].startswith("1b0f5514")
    assert prov["criteria_commit"].startswith("38a337e") and prov["queue_sha256"].startswith("f0fcafb5")
    assert prov["spec_sha256"].startswith("91e74365") and prov["freeze_sha256"].startswith("1680e1e0")
    texts = [d["text"] for d in rec["user_decisions"]]
    assert texts[0].startswith("Drawer-open-v2 calibration - choose Option 1") and texts[1].startswith(
        "Proceed with Option 2: override drawer-open-v2 only.")
    sens = rec["sensitivity_check"]
    assert sens["result"] == "FAILED" and sens["accepted_by_user"] is True and len(sens["failed_criteria"]) == 2
    assert "4.881430x" in sens["failed_criteria"][0]["measured"] and "3 of 9" in sens["failed_criteria"][1]["measured"]
    assert rec["scope"]["robot_execution_authorized"] is False and rec["retained"]["episodes"] == 98
    assert rec["retained"]["cameras"] == 3 and not rec["retained"]["excluded_episodes"]


@pytest.mark.parametrize("mutate,why", [
    (lambda r: r["scope"].update(robot_execution_authorized=True), "robot"),
    (lambda r: r["automatic_acceptance"].update(passed=True), "FAILED automatic"),
    (lambda r: r["automatic_acceptance"].update(thresholds_changed=True), "thresholds"),
    (lambda r: r["value_tolerance"].update(relative=1e-3), "tolerance"),
    (lambda r: r["exceptions"][0].update(measured=0.0079), "within its threshold"),
    (lambda r: r["exceptions"][0].update(check="any"), "malformed"),
    (lambda r: r["exceptions"][0].update(cameras=["135122070988/color"]), "does not name"),
    (lambda r: r["sensitivity_check"].update(accepted_by_user=False), "sensitivity"),
    (lambda r: r.update(user_decisions=[]), "verbatim"),
    (lambda r: r.update(task="covering"), "task"),
])
def test_malformed_records_fail_closed(mutate, why):
    rec = json.loads(COMMITTED.read_text())
    assert AO.record_problem(rec, "drawer-open-v2") is None
    mutate(rec)
    assert why in AO.record_problem(rec, "drawer-open-v2")


# ---------------------------------------------------------------------------------------------------- publisher
def test_publisher_override_publishes_effective_status_and_keeps_automatic_verdict(world):
    w = world()
    before = {n: _sha(w["work"] / n) for n in ("automatic-decision.json", "verdict-release1.json")}
    doc = _published(w)
    assert {n: _sha(w["work"] / n) for n in before} == before  # original records untouched
    assert json.loads((w["work"] / "automatic-decision.json").read_text())["accepted"] is False
    assert all(doc["cameras"][c]["status"] == "held" for c in CAMS)  # automatic verdicts kept
    assert doc["name"].endswith("-ao20261117") and all(doc["cameras"][c]["provenance"]["rig"] == doc["name"]
                                                        for c in CAMS)
    block = doc["acceptance_override"]
    assert block["automatic_decision"]["accepted"] is False and block["automatic_acceptance_passed"] is False
    assert block["robot_execution_authorized"] is False and block["record"] == w["rec"]
    decision = json.loads((w["work"] / "override-decision.json").read_text())
    assert decision["effective_cameras"] == {c: AO.STATUS for c in CAMS} and decision["record"]["sha256"] == \
        block["record_sha256"] and decision["automatic_decision"]["accepted"] is False
    assert decision["user_decisions"] == w["rec"]["user_decisions"]
    assert block["override_decision"]["sha256"] == _sha(w["work"] / "override-decision.json")
    for c in CAMS:
        status, why, ref = AO.effective_status(doc, c)
        assert status == AO.STATUS and why == [] and ref["robot_execution_authorized"] is False
    # deterministic: a re-run with the same evidence verifies the same decision file (write-once, equal bytes)
    assert _published(w) == doc
    # the automatic decision is never rewritten: different content refuses
    with pytest.raises(ValueError, match="never rewritten"):
        AO.write_once(w["work"] / "automatic-decision.json", {"cameras": {}, "accepted": True})


@pytest.mark.parametrize("mutate,expect", [
    (lambda w: (w["work"] / "verdict-release1.json").write_text("{}"), "verdict-release1.json differs"),
    (lambda w: w["kw"].update(automatic={**w["kw"]["automatic"], "accepted": True}), "automatic-decision.json"),
    (lambda w: (w["kw"]["specpath"]).write_text("{}"), "spec.json differs"),
    (lambda w: (w["kw"]["specpath"].parent / "FREEZE.json").write_text("{}"), "FREEZE.json differs"),
    (lambda w: w["kw"].update(commit="0" * 40), "criteria commit"),
    (lambda w: w["static"]["dt"]["rows"][5].update(abs_dt_ms=22.0), "cohort"),
    (lambda w: w["doc"].update(episodes=w["doc"]["episodes"][:-1]), "retains"),
    (lambda w: w["doc"]["cameras"][CAMS[0]]["diagnostics"]["criteria_v2"]["g5b"]["dt_extension"]["pairs"][PAIR]
     .update(median_diff_ci95_px=[0.0, 0.0989]), "median upper bound"),
])
def test_publisher_refuses_any_mismatch_and_keeps_the_rig_held(world, mutate, expect):
    w = world()
    mutate(w)
    with pytest.raises(ValueError, match="rig held by established automatic criteria") as err:
        _published(w)
    assert expect in str(err.value)
    assert not (w["work"] / "override-decision.json").exists()


def test_publisher_without_a_task_record_keeps_the_original_failure(world, monkeypatch):
    w = world()
    monkeypatch.setattr(AO, "RECORDS", {})
    with pytest.raises(ValueError, match="see automatic-decision.json"):
        _published(w)


# ------------------------------------------------------------------------------------- consumer-side acceptance
def _g1_extra(doc, cam, reason="G1: rotation radius 0.400 deg > 0.35 (loco-x)"):
    e = doc["cameras"][cam]
    e["reasons"].insert(0, reason)
    e["diagnostics"]["criteria_v2"]["g1"]["reasons"].append(reason)


@pytest.mark.parametrize("mutate,cam,expect", [
    # an extra failing gate
    (lambda d: d["cameras"][CAMS[0]].update(held_by=["G2", "G5b"], reasons=["G2: x", G5R],
                                            checks={**d["cameras"][CAMS[0]]["checks"], "G2": False}),
     CAMS[0], "does not name"),
    # an extra failing G1 sub-check on the covered camera
    (lambda d: _g1_extra(d, CAMS[1]), CAMS[1], "unrecorded reasons"),
    # G1 failing on a camera whose override names only G5b
    (lambda d: d["cameras"][CAMS[0]].update(held_by=["G1", "G5b"], reasons=[G1R, G5R]), CAMS[0], "does not name"),
    # value drift
    (lambda d: d["cameras"][CAMS[1]]["diagnostics"]["criteria_v2"]["g1"].update(radius_centre_m=0.0087708), CAMS[1],
     "centre radius"),
    (lambda d: d["cameras"][CAMS[2]]["diagnostics"]["criteria_v2"]["g5b"]["dt_extension"]["pairs"][PAIR]
     .update(n_long=10), CAMS[2], "n_long"),
    (lambda d: d["cameras"][CAMS[2]]["diagnostics"]["criteria_v2"]["g5b"]["dt_extension"]["pairs"][PAIR]
     .update(median_diff_px=0.0143), CAMS[2], "median_diff_px"),
    # another pair failing as well
    (lambda d: d["cameras"][CAMS[0]]["diagnostics"]["criteria_v2"]["g5b"]["dt_extension"]["problems"]
     .append("135122070988-135122076054: long-offset p90 +0.2 px > 0.1"), CAMS[0], "single problem"),
    # a changed threshold in the evidence
    (lambda d: d["cameras"][CAMS[1]]["diagnostics"]["criteria_v2"]["g1"]["thresholds"].update(radius_centre_m=0.009),
     CAMS[1], "threshold"),
    # per-episode G5b flags
    (lambda d: d["cameras"][CAMS[0]]["diagnostics"]["criteria_v2"]["g5b"].update(moved=["e"]), CAMS[0], "per-episode"),
    # a tampered embedded record / automatic decision / robot flag
    (lambda d: d["acceptance_override"]["record"]["exceptions"][0].update(threshold=0.009), CAMS[1], "committed"),
    (lambda d: d["acceptance_override"]["automatic_decision"].update(accepted=True), CAMS[0], "failed automatic"),
    (lambda d: d["acceptance_override"].update(robot_execution_authorized=True), CAMS[0], "robot"),
    # other task
    (lambda d: d.update(task="covering"), CAMS[0], "no committed acceptance override"),
    # an automatically validated camera is simply validated (no override needed)
])
def test_consumer_keeps_camera_held_on_any_deviation(world, mutate, cam, expect):
    doc = _published(world())
    mutate(doc)
    status, why, ref = AO.effective_status(doc, cam)
    assert status == "held" and ref is None and expect in why[-1], why
    assert AO.offline_acceptance(doc)["accepted"] is False


def test_override_covers_only_named_cameras_and_binds_to_the_committed_record(world):
    w = world()
    rec = copy.deepcopy(w["rec"])
    del rec["cameras"][CAMS[2]]
    for e in rec["exceptions"]:
        e["cameras"] = [c for c in e["cameras"] if c != CAMS[2]]
    assert AO.camera_problems(rec, CAMS[2], w["doc"]["cameras"][CAMS[2]]) == [f"{CAMS[2]} is not named by the override"]
    doc = _published(w)
    assert AO.offline_acceptance(doc)["accepted"]
    # changing the committed record after publication invalidates the document for every camera
    (AO.RECORDS_DIR / "rec.json").write_text(json.dumps(rec))
    assert all(AO.effective_status(doc, c)[0] == "held" for c in CAMS)
    (w["work"] / "override-decision.json").unlink()
    with pytest.raises(ValueError, match="cameras differ from the record"):
        _published(w)


# --------------------------------------------------------------------------------------------- robot unchanged
def test_robot_and_deployment_paths_never_accept_the_override(world, tmp_path):
    from s4d.realworld.calib.rig import load_rig, publish_rig, rig_camera_status
    from s4d.rwlearn.deploy.realsense import load_rig_session

    w = world()
    doc = _published(w)
    ref = {f"{c.replace('/', '__')}__{k}": np.zeros((2, 2)) for c in CAMS for k in ("grey", "dynamic")}
    path = w["work"] / "rig-synthetic-accepted.json"
    publish_rig(path, doc, ref)
    loaded, _, ident = load_rig(path)
    assert {c: rig_camera_status(loaded, c)[0] for c in CAMS} == {c: "held" for c in CAMS}
    cams = [c.split("/")[0] for c in CAMS]
    with pytest.raises(ValueError, match="not validated"):
        load_rig_session(path, cams, {c: np.eye(3) for c in cams})
    session = AO.load_rig_session_offline(path, cams, {c: np.eye(3) for c in cams})
    ev = session["native_depth_correction"][cams[0]]["evidence"]
    assert ev["effective_rig_status"] == AO.STATUS and ev["acceptance_override"]["robot_execution_authorized"] is False
    assert ev["rig_calibration"] == ident["rig_calibration"]
    # without an override block the offline reader IS the unchanged deployment reader
    plain = _release_doc(list(CAMS), "e", task="synthetic")
    plain_path = tmp_path / "plain.json"
    publish_rig(plain_path, plain, ref)
    assert AO.load_rig_session_offline(plain_path, cams, {c: np.eye(3) for c in cams}) == load_rig_session(
        plain_path, cams, {c: np.eye(3) for c in cams})
    # route gate: offline only, bound to both decision files
    gate = AO.offline_gate(path, "synthetic")
    assert gate["mode"] == "acceptance_override" and gate["robot_execution_authorized"] is False
    (w["work"] / "override-decision.json").write_text("{}")
    with pytest.raises(ValueError, match="override-decision.json"):
        AO.offline_gate(path, "synthetic")
    with pytest.raises(ValueError, match="task"):
        AO.offline_gate(path, "covering")


# ------------------------------------------------------------------------------------------------- phase B
def test_phase_b_rig_mode_accepts_only_with_the_installed_override_and_records_it(world, rig_episode):
    from s4d.realworld.calib import rig
    from s4d.realworld.calib.rig import load_rig

    cams, ep = rig_episode["cams"], rig_episode["ep"]
    w = world(cams=tuple(cams), episodes=(ep,), g1cam=cams[1], cohort_n=1)
    doc = _published(w)
    ref = rig_episode["reference"]()
    _, plain, _ = _calibrate_with(rig_episode, copy.deepcopy(doc), ref, "ao-plain")
    assert all(plain[c]["status"] == "rejected" and any("not validated" in r for r in plain[c]["reasons"])
               for c in cams)  # the unchanged rig mode never accepts the override
    original = (rig.rig_camera_status, rig.validate_episode)
    restore = AO.install_rig_override()
    try:
        _, outs, path = _calibrate_with(rig_episode, copy.deepcopy(doc), ref, "ao-installed")
    finally:
        restore()
    assert (rig.rig_camera_status, rig.validate_episode) == original
    published, _, _ = load_rig(path)
    summary = AO.summary(published)
    for c in cams:  # every per-episode check except the rig status still applies (synthetic visibility may reject)
        out = outs[c]
        assert not any("not validated" in r or "rig calibration of" in r for r in out["reasons"]), out["reasons"]
        assert out["diagnostics"]["release"]["effective_rig_status"] == AO.STATUS
        assert out["diagnostics"]["acceptance_override"] == summary
    assert any(outs[c]["status"] == "accepted" for c in cams)


def _stage_outputs(tmp_path, doc, fp, ep, override):
    root = tmp_path / "root"

    def build(tmp):
        for cam, c in doc["cameras"].items():
            diag = {"release": {"rig": doc["name"], "effective_rig_status": AO.STATUS if override else "validated"},
                    "fixedness": {"fixed": True, "rule": "g5ab/1", "registration": {"moved": False}, "exception": None}}
            if override:
                diag["acceptance_override"] = override
            rio.atomic_write_json(tmp / f"{rio.safe_name(cam)}.json", {
                "status": "accepted", "reasons": [], "T_cam_base": c["T_cam_base"], "init_method": "rig_shared",
                "depth_scale": c["depth_scale"], "robot_model": doc["robot_model"], "diagnostics": diag})
        return {}

    rec = rio.run_stage(rio.stage_dir(root, ep, "calibrate"), stage="calibrate", version="1",
                        config={"rig_calibration": {"name": doc["name"]}},
                        inputs={"rig_calibration": fp["rig_calibration"], "rig_reference": doc["reference_sha256"]},
                        build=build)
    return {"sha256": "a" * 64, "preprocessing": {"stages": {"calibrate": {
        "status": "done", "input_sha256": "a" * 64, "metrics": {"build_id": rec["build_id"]},
        "outputs": str(rio.episode_dir(root, ep))}}}}


def test_calibration_validated_records_accept_the_override_and_carry_it(world, tmp_path):
    from s4d.realworld.calib.release import check_episode
    from s4d.realworld.calib.rig import load_rig, publish_rig

    w = world()
    path = tmp_path / "rig.json"
    publish_rig(path, _published(w), {f"{c.replace('/', '__')}__{k}": np.zeros((2, 2)) for c in CAMS
                                      for k in ("grey", "dynamic")})
    doc, _, fp = load_rig(path)
    summary = AO.summary(doc)
    entry = _stage_outputs(tmp_path / "a", doc, fp, "synthetic-000000", summary)
    record, problems = check_episode("synthetic/episode.h5", entry, doc, fp["rig_calibration"])
    assert problems == [] and record["metrics"]["decision"] == "accepted"
    assert record["metrics"]["acceptance_override"] == summary
    assert record["metrics"]["evidence"]["acceptance_override"]["id"] == w["rec"]["id"]
    assert all(v["effective_rig_status"] == AO.STATUS for v in record["metrics"]["per_camera"].values())
    # outputs that do not carry the document's override (e.g. built without the phase B installation): refused
    entry = _stage_outputs(tmp_path / "b", doc, fp, "synthetic-000000", None)
    record, problems = check_episode("synthetic/episode.h5", entry, doc, fp["rig_calibration"])
    assert record is None and all("does not carry the document's acceptance override" in p for p in problems)
    # a camera whose override no longer applies: refused like any held camera
    drift = copy.deepcopy(doc)
    drift["cameras"][CAMS[1]]["diagnostics"]["criteria_v2"]["g1"]["radius_centre_m"] = 0.0095
    record, problems = check_episode("synthetic/episode.h5", _stage_outputs(tmp_path / "c", drift, fp,
                                                                            "synthetic-000000", summary),
                                     drift, fp["rig_calibration"])
    assert record is None and any(f"camera {CAMS[1]} is held" in p for p in problems)


def test_compile_phase_b_marks_jobs_and_routes_the_episode_step_through_the_override(world, tmp_path):
    from s4d.realworld.calib.rig import publish_rig

    w = world()
    doc = _published(w)
    rig = tmp_path / "rig.json"
    publish_rig(rig, doc, {f"{c.replace('/', '__')}__{k}": np.zeros((2, 2)) for c in CAMS for k in ("grey", "dynamic")})
    rows, rigmap = [], {}
    for n in range(2):
        ep = f"drawer-open-v2-success-episode_{n:06d}"
        cfg = tmp_path / f"{n}.yaml"
        cfg.write_text(yaml.safe_dump({"output_root": str(tmp_path / f"p{n}"), "adapter": {"config": {"episodes": {
            ep: f"/home/ws/data/real_world/drawer-open-v2/success/episode_{n:06d}.h5"}}},
            "stages": {"calibrate": {"rig_calibration": "/pending"}}}))
        rows.append({"episode": ep, "config": str(cfg), "raw_sha256": "c" * 64})
        rigmap[ep] = str(rig)
    plan, rmap, tools = tmp_path / "plan.json", tmp_path / "rig-map.json", tmp_path / "tools"
    plan.write_text(json.dumps(rows))
    rmap.write_text(json.dumps(rigmap))
    (tools / "configs").mkdir(parents=True)
    (tools / "configs/rwprep_server.yaml").write_text(yaml.safe_dump({"gpu_allowlist": []}))
    out = tmp_path / "queue"
    argv = ["--phase", "B", "--plan", str(plan), "--out", str(out), "--processed", str(tmp_path / "processed"),
            "--host", "remote", "--image", "sha256:" + "a" * 64, "--code", str(REPO), "--weights", str(tmp_path),
            "--tools", str(tools), "--control-repo", str(tmp_path), "--rig-map", str(rmap)]
    compiler = _script("realworld_prep_register_ao")
    assert compiler.compile_phase_b(argv) == 0
    jobs = json.loads((out / "jobs.json").read_text())
    assert len(jobs) == 2
    for job in jobs:
        step = job["steps"][0]
        assert step["kind"] == "python" and step["argv"][:5] == ["-m", "s4d.realworld.calib.acceptance_override",
                                                                 "rwprep", "--", "episode"]
        assert step["argv"][-1] == "calibrate,track_everything,track_motion,train_data"
        assert job["acceptance_override"]["rig_sha256"] == _sha(rig) and job["acceptance_override"][
            "robot_execution_authorized"] is False
        assert [s["kind"] for s in job["steps"][1:]] == ["rwprep", "rwprep"]
    from s4d.realworld.calib import rig as rig_module
    assert rig_module.rig_camera_status.__module__ == "s4d.realworld.calib.rig" and not AO._ORIGINAL
    # a rig whose override does not apply: the established compiler refusal stands
    held = copy.deepcopy(doc)
    held["cameras"][CAMS[0]]["diagnostics"]["criteria_v2"]["g5b"]["moved"] = ["x"]
    bad = tmp_path / "bad.json"
    publish_rig(bad, held, {f"{c.replace('/', '__')}__{k}": np.zeros((2, 2)) for c in CAMS for k in ("grey", "dynamic")})
    rmap.write_text(json.dumps({ep: str(bad) for ep in rigmap}))
    with pytest.raises(ValueError, match="must be accepted"):
        compiler.compile_phase_b([*argv[:5], str(tmp_path / "queue2"), *argv[6:]])


def test_runtime_wrapper_refuses_a_rig_not_accepted_for_offline_use(world, tmp_path, capsys):
    from s4d.realworld.calib.rig import publish_rig

    w = world()
    doc = _published(w)
    doc["cameras"][CAMS[1]]["diagnostics"]["criteria_v2"]["g1"]["radius_centre_m"] = 0.0095
    rig = tmp_path / "rig.json"
    publish_rig(rig, doc, {f"{c.replace('/', '__')}__{k}": np.zeros((2, 2)) for c in CAMS for k in ("grey", "dynamic")})
    cfg = tmp_path / "p.yaml"
    cfg.write_text(yaml.safe_dump({"stages": {"calibrate": {"rig_calibration": str(rig)}}}))
    marker = tmp_path / "rwprep.py"
    marker.write_text("raise SystemExit(99)")
    with pytest.raises(SystemExit) as code:
        AO.main(["rwprep", "--rwprep", str(marker), "--", "episode", "--pipeline", str(cfg)])
    assert code.value.code == 20 and "refused" in capsys.readouterr().out
    assert not AO._ORIGINAL
    # an accepted rig: the established rwprep script runs in-process with the override installed, then restored
    publish_rig(rig, _published(w), {f"{c.replace('/', '__')}__{k}": np.zeros((2, 2)) for c in CAMS
                                     for k in ("grey", "dynamic")})
    marker.write_text("import sys\nfrom s4d.realworld.calib import rig\n"
                      "ok = rig.rig_camera_status.__module__.endswith('acceptance_override') and sys.argv[1:3] == "
                      "['episode', '--pipeline']\nraise SystemExit(0 if ok else 3)\n")
    with pytest.raises(SystemExit) as code:
        AO.main(["rwprep", "--rwprep", str(marker), "--", "episode", "--pipeline", str(cfg)])
    from s4d.realworld.calib import rig as rig_module
    assert code.value.code == 0 and not AO._ORIGINAL
    assert rig_module.rig_camera_status.__module__ == "s4d.realworld.calib.rig"


# ------------------------------------------------------------------------------------------ native depth (mvp3d)
def test_native_depth_override_build_equals_the_plain_build_and_names_the_override(world, tmp_path, monkeypatch):
    """The override build reads the held rig only through the offline reader (here also through the raw mapping);
    overlays equal a validated-rig build of the same data. The pool runs in-process so the test record is visible."""
    import concurrent.futures
    import os
    import shutil

    from s4d.realworld.calib.rig import load_rig, publish_rig
    from s4d.rwlearn import native_depth as ND
    from s4d.rwlearn.deploy.realsense import load_rig_session
    from s4d.rwlearn.native_depth_override import build_native_depth_override
    from tests.test_rwlearn_native_depth import _moved_raw, _overlay_arrays, build, make_store

    class InProcess(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, max_workers=None, mp_context=None):
            super().__init__(max_workers=1)

    monkeypatch.setattr(ND, "ProcessPoolExecutor", InProcess)
    reference = make_store(tmp_path / "ref", count=2)
    build(reference)
    fixture = make_store(tmp_path / "store", count=2)
    plain, _, _ = load_rig(fixture.rig)
    with np.load(fixture.rig.with_suffix(".reference.npz")) as z:
        arrays = {k: z[k] for k in z.files}
    cams = sorted(plain["cameras"])
    w = world(cams=tuple(cams), episodes=("e",), g1cam=cams[1], cohort_n=1)
    for c in cams:
        w["doc"]["cameras"][c].update(T_cam_base=plain["cameras"][c]["T_cam_base"],
                                      depth_scale=plain["cameras"][c]["depth_scale"])
    held = _published(w)
    publish_rig(fixture.rig, held, arrays)
    with pytest.raises(ValueError, match="not validated"):  # the ordinary (deployment-strict) build refuses
        build(fixture)
    shutil.rmtree(fixture.overlays, ignore_errors=True)
    environment = dict(os.environ)
    local = _moved_raw(fixture, tmp_path)
    result = build_native_depth_override(fixture.locator, fixture.rig, fixture.overlays, fixture.out,
                                         raw_root=f"{fixture.root}={local}")
    assert result["built"] == 2 and result["descriptor_written"]
    got, want = _overlay_arrays(fixture.out), _overlay_arrays(reference.out)
    for ep in want:
        for cam in want[ep][1]:
            assert np.array_equal(got[ep][1][cam], want[ep][1][cam])
        for cam, corr in got[ep][0]["native_depth_correction"].items():
            assert corr["factor"] == want[ep][0]["native_depth_correction"][cam]["factor"]
            assert corr["evidence"]["effective_rig_status"] == AO.STATUS
            assert corr["evidence"]["acceptance_override"]["id"] == w["rec"]["id"]
    assert ND.load_rig_session is load_rig_session and ND._build_episode.__module__ == "s4d.rwlearn.native_depth"
    assert ND.CompactStore.__module__ == "s4d.realworld.train_store" and dict(os.environ) == environment
    with pytest.raises(ValueError, match="not a rig accepted by a valid acceptance override"):
        build_native_depth_override(reference.locator, reference.rig, reference.overlays, reference.out)
