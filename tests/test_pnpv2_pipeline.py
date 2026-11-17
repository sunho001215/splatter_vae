"""pick-and-place-v2 pipeline: arrival/check-data, evidence-gated waves, registrar dry run/execute, v1 isolation.

Everything runs on a synthetic 100-episode fixture in tmp_path; no network, Robot PC, queue or registry access.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from ops.jobs import pnpv2_registration as reg
from ops.jobs.pnpv2_plan import STAGES, build_rows, wave_of
from s4d.rwlearn import gate
from s4d.rwlearn.pnpv2_arrival import arrival, check_data, v1_collisions
from s4d.rwlearn.pnpv2_spec import (
    GEOMETRY_METHODS,
    METHODS,
    PAUSE,
    RGB_METHODS,
    SPEC,
    STEM,
    TASK,
    V1_PAUSE,
    WRAPPERS,
    config,
    dispatcher_stem,
    load_spec,
    row_id,
    values,
)

ROOT = Path(__file__).resolve().parents[1]
V1 = json.loads((ROOT / "tests/fixtures/pnpv2-v1-pause.json").read_text())
RELEASE = {"commit": "a" * 40, "image": "sha256:" + "b" * 64, "source": "local"}
SCENE = {"lookat": [0.4, 0.0, 0.1], "pose_scale": 2.5, "bounds": [-0.5, -0.6, -0.05, 0.9, 0.6, 0.7]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fake_inspect(path):
    return {"cameras": {}, "robot_samples": 10, "label_fields": [], "action_contract": "synthetic"}


def make_raw(raw, n=100, marker=True, age=3600):
    d = raw / TASK / "success"
    d.mkdir(parents=True)
    old = time.time() - age
    for i in range(1, n + 1):
        p = d / f"episode_{i:06d}.h5"
        p.write_bytes(f"pnpv2 synthetic recording {i}".encode())
        os.utime(p, (old, old))
    if marker:
        text = "".join(f"{sha(p)}  {p.name}\n" for p in sorted(d.iterdir()))
        for name in ("success.local.sha256", "success.robotpc.sha256"):
            (raw / TASK / name).write_text(text)
        (raw / TASK / "success.VERIFIED").touch()
    return d


@pytest.fixture
def spec(tmp_path):
    s = load_spec(ROOT / SPEC)
    for key in ("raw", "processed", "work"):
        s["roots"][key] = str(tmp_path / key)
    return s


def check(spec, **kw):
    return check_data(spec, inspect=fake_inspect, settle_seconds=0, sleep=lambda s: None, **kw)


# ----------------------------------------------------------------------------------------------- arrival
def test_arrival_tolerates_partial_uploads_and_needs_marker_or_stable_inventory(spec, tmp_path):
    raw = Path(spec["roots"]["raw"])
    assert arrival(spec, settle_seconds=0, sleep=lambda s: None)["state"] == "absent"
    d = make_raw(raw, n=99, marker=False)
    assert arrival(spec, settle_seconds=0, sleep=lambda s: None)["state"] == "partial"  # 99/100
    (d / "episode_000100.h5.part").write_bytes(b"in flight")
    state = arrival(spec, settle_seconds=0, sleep=lambda s: None)
    assert state["state"] == "partial" and state["partial"] == ["episode_000100.h5.part"]
    (d / "episode_000100.h5.part").rename(d / "episode_000100.h5")
    # No marker: a freshly finished file is younger than the settle window -> unstable, never complete.
    assert arrival(spec, settle_seconds=600, sleep=lambda s: None)["state"] == "unstable"
    old = time.time() - 3600
    os.utime(d / "episode_000100.h5", (old, old))
    grew = {"n": 0}

    def growing(_):
        grew["n"] += 1
        (d / "episode_000101.h5").write_bytes(b"late arrival")

    assert arrival(spec, settle_seconds=600, sleep=growing)["state"] == "unstable"
    (d / "episode_000101.h5").unlink()
    stable = arrival(spec, settle_seconds=600, sleep=lambda s: None)
    assert (stable["state"], stable["basis"]) == ("complete", "stable")


def test_marker_requires_equal_robot_pc_and_local_sha_lists(spec):
    raw = Path(spec["roots"]["raw"])
    make_raw(raw)
    assert arrival(spec, settle_seconds=0)["basis"] == "marker"
    lst = raw / TASK / "success.robotpc.sha256"
    lst.write_text(lst.read_text().replace(lst.read_text()[:64], "0" * 64, 1))
    state = arrival(spec, settle_seconds=0)
    assert state["state"] == "partial" and "differ" in " ".join(state["problems"])


def test_check_data_validates_the_synthetic_dataset_and_its_split(spec):
    make_raw(Path(spec["roots"]["raw"]))
    report = check(spec)
    assert report["passed"], report["problems"]
    assert report["episodes"] == 100 and report["split"] == {"train": 90, "validation": 10}
    m = report["manifest"]
    assert m["task"] == TASK and m["variant"] == "S" and {r["label"] for r in m["episodes"]} == {"success"}
    assert all(r["id"].startswith(f"{TASK}-success-episode_") for r in m["episodes"])
    assert report["writes"] is False and report["robot_pc_io"] is False
    assert check(spec)["manifest_sha256"] == report["manifest_sha256"]  # deterministic seed-0 split


def test_check_data_refuses_integrity_and_duplicate_problems(spec):
    d = make_raw(Path(spec["roots"]["raw"]))
    (d / "episode_000007.h5").write_bytes((d / "episode_000008.h5").read_bytes())
    report = check(spec)
    assert not report["passed"]
    text = " ".join(report["problems"])
    assert "Robot-PC list" in text or "differ" in text


def test_check_data_never_touches_network_or_writes(spec, monkeypatch, tmp_path):
    make_raw(Path(spec["roots"]["raw"]))
    import socket

    def boom(*a, **k):
        raise AssertionError("network/subprocess used")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    assert check(spec)["passed"]
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_robot_pc_paths_are_refused(spec):
    spec["roots"]["raw"] = "/mnt/robotpc/teleop_collector/data"
    with pytest.raises(ValueError, match="Robot-PC"):
        arrival(spec, settle_seconds=0)


# ----------------------------------------------------------------------------------- v1 non-collision
def test_v2_row_and_dispatcher_ids_never_match_v1_discontinuation():
    s = load_spec(ROOT / SPEC)
    rows = build_rows(s, RELEASE, scene=SCENE)
    ep = f"{TASK}-success-episode_000001"
    dispatcher = [f"{dispatcher_stem('prepA')}-{ep}", f"{dispatcher_stem('prepB')}-{ep}",
                  f"{dispatcher_stem('calib')}-{TASK}-view", f"{dispatcher_stem('qa')}-{TASK}-records"]
    ids = [r["id"] for r in rows] + dispatcher
    hits, other, problems = v1_collisions(ids, V1)
    assert hits == [] and other == [] and problems == []
    # The hazard is real: the established producers' default ids for this task WOULD be held/stopped as v1.
    legacy = [f"rwv2-20261113-prep-ours-{ep}-phaseA", f"rwv2-20261113-calib-ours-{TASK}-view",
              f"rwv2-20261113-qa-ours-{TASK}-records", f"rwv2-20261113-pre-dynamo-{TASK}-S"]
    assert v1_collisions(legacy, V1)[0] == sorted(legacy)
    # No v2 row depends on, reads or resumes a v1 run.
    for row in rows:
        assert all(d.startswith(STEM + "-") for d in row["deps"])
        assert not any("runs/rw" in str(a) for a in row["args"])
        if "--name" in row["args"]:
            assert row["args"][row["args"].index("--name") + 1] == row["id"]


def test_v1_pause_must_stay_active():
    lifted = json.loads(json.dumps(V1))
    lifted["pauses"][0]["active"] = False
    assert v1_collisions([row_id("gate", "rgb")], lifted)[2]
    assert v1_collisions([row_id("gate", "rgb")], {"pauses": []})[2]


def test_prep_compilers_use_the_v2_stem(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from realworld_prep_register import compile_plans

    raw = tmp_path / "raw/real_world" / TASK / "success"
    raw.mkdir(parents=True)
    (raw / "episode_000001.h5").write_bytes(b"x")
    cfg = tmp_path / "ep.yaml"
    ep = f"{TASK}-success-episode_000001"
    cfg.write_text(yaml.safe_dump({"adapter": {"config": {"episodes": {ep: str(raw / "episode_000001.h5")}}},
                                   "stages": {"calibrate": {}}, "output_root": "x"}))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([{"episode": ep, "config": str(cfg), "raw_sha256": "c" * 64}]))
    jobs = compile_plans([plan], tmp_path / "bundle", tmp_path / "processed", tmp_path / "raw/real_world",
                         stem=dispatcher_stem("prepA"))
    assert [j["id"] for j in jobs] == [f"{dispatcher_stem('prepA')}-{ep}"]
    assert v1_collisions([j["id"] for j in jobs], V1)[0] == []
    from ops.jobs.rwprep_mig import validate

    with pytest.raises(ValueError, match="invalid MIG job"):
        validate(dict(host="local", image="sha256:" + "0" * 64, commit="0" * 40), [{**jobs[0], "id": "v1-x"}])


# ------------------------------------------------------------------------------------ waves and graph
def ancestors(byid, rid):
    return {d for dep in byid[rid]["deps"] for d in ({dep} | ancestors(byid, dep))}


def test_calibration_independent_wave_has_its_own_dependency_path():
    s = load_spec(ROOT / SPEC)
    rows = build_rows(s, RELEASE, scene=SCENE)
    byid = {r["id"]: r for r in rows}
    assert len(rows) == 90 and sum(r["pnpv2_wave"] == "rgb" for r in rows) == 33
    for row in rows:
        method = row["realworld_method"]
        if method in ("rgb", "geometry"):
            continue
        assert row["pnpv2_wave"] == wave_of(method)
        up = ancestors(byid, row["id"])
        if row["pnpv2_wave"] == "rgb":
            assert all(byid[a]["pnpv2_wave"] == "rgb" for a in up)
            assert row_id("gate", "geometry") not in up and row_id("native", "mvp3d") not in up
            assert row["calibration_dependent"] is False
        else:
            assert row_id("gate", "geometry") in up | {row["id"]} and row_id("gate", "rgb") not in up
    # The wave split is exactly the gate's consumer geometry classification.
    for method in METHODS:
        consumer = gate.FQL_CONSUMER["ours" if method.startswith("ours") else method.replace("_noaug", "")]
        assert gate.CONSUMERS[consumer].geometry == (method in GEOMETRY_METHODS)
    # RGB stays registrable without rig/Phase B; geometry needs them.
    assert set(STAGES["wave:rgb"][1]) == {"check-data", "ingest", "phaseA", "rgb-publication"}
    assert {"rig", "phaseB", "geometry-publication", "scene"} <= set(STAGES["wave:geometry"][1])
    rgb_only = build_rows(s, RELEASE, waves=("rgb",))
    assert {r["pnpv2_wave"] for r in rgb_only} == {"rgb"} and len(rgb_only) == 33


def test_rows_use_established_settings_and_resolvable_entrypoints():
    s = load_spec(ROOT / SPEC)
    rows = build_rows(s, RELEASE, scene=SCENE)
    from ops.jobs.pnpv2_registration import _merged_config, placement_problems
    from s4d.rwlearn.pnpv2_spec import leftover_tokens

    assert placement_problems(rows, {}) == []
    for row in rows:
        assert row["script"] in WRAPPERS and (ROOT / row["script"]).is_file()
        merged = _merged_config(row)
        if merged:
            assert leftover_tokens(merged[1]) == [], row["id"]
        if row["id"].split("-")[2] == "fql":
            cfg = merged[1]
            pixel = row["realworld_method"].startswith("e2e")
            assert cfg["agent"]["alpha"] == 10.0 and cfg["agent"]["normalize_q_loss"] is (not pixel)
            assert cfg["train"]["offline_steps"] == 1_000_000 and row["fql_seed"] in (0, 1, 2)
            if row["realworld_method"] == "e2e_cnn_noaug":
                assert cfg["pixel"]["p_aug"] == 0.0
    ours = [r for r in rows if r["id"].startswith(f"{STEM}-pre-ours_")]
    assert {r["args"][1] for r in ours} == {f"configs/realworld/ours_matrix/{TASK}-S-{a}-scratch400k.yaml"
                                            for a in ("orig", "ch01")}
    snerl = next(r for r in rows if r["id"] == row_id("pre", "snerl"))
    assert "data_interface.pose_scale=2.5" in snerl["args"]


def test_resolver_alias_maps_every_entrypoint():
    from ops.jobs import host_roles  # noqa: F401  (installs the coordinator aliases)
    from s4d.rwlearn import dsreset_entrypoints

    for wrapper, original in WRAPPERS.items():
        # The established dsreset aliases still apply after the pnpv2 one (task-aware evaluation body).
        expected = dsreset_entrypoints.ENTRYPOINTS.get(original, original)
        assert dsreset_entrypoints.legacy_row({"script": wrapper})["script"] == expected
        text = (ROOT / wrapper).read_text()
        assert "\nguard_gpus()\n" in text
    assert dsreset_entrypoints.legacy_row({"script": "scripts/train.py"})["script"] == "scripts/train.py"


# --------------------------------------------------------------------------------- registrar fixture
class FakeStore:
    def __init__(self, manifest):
        self.rows = {f"store-{r['id']}": r for r in manifest["episodes"]}

    def episodes(self):
        return list(self.rows)

    def task(self, ep):
        return TASK

    def raw_recording(self, ep):
        return self.rows[ep]["path"]

    def raw_sha256(self, ep):
        return self.rows[ep]["sha256"]


def stage_bundle(spec, view, m, *, geometry=False):
    from ops.jobs.pnpv2_rgb import local_report

    p = values(spec)
    store = FakeStore(m)
    evidence = {r["id"]: dict(singleview={"passed": True}, m5={"accepted": 4}, raw_validation={"ok": True})
                for r in m["episodes"]}
    report = local_report(m, store, evidence, {"synthetic": True})
    if geometry:
        report["calibration_hold"][TASK].update(passed=True, problems=[], problem_count=0)
        for fam in report["families"][TASK].values():
            fam.update(passed=True, failed_count=0, failed_episodes=[])
    report["dataset_manifest_sha256"] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    keys = {r["sha256"]: k for k, r in store.rows.items()}
    docs = {"manifest.json": m, "gate_report.json": report,
            "split_seed0.json": {s: [keys[r["sha256"]] for r in m["episodes"] if r["split"] == s]
                                 for s in ("train", "validation")},
            "outcomes.json": {k: True for k in store.rows}}
    d = Path(p[f"{view}_dir_local"])
    d.mkdir(parents=True)
    for name, doc in docs.items():
        (d / name).write_text(json.dumps(doc))
    base = p["remote_processed"] if geometry else p[f"{view}_dir"] + "/episodes"
    (d / "store.yaml").write_text(yaml.safe_dump({"kind": "compact", "paths": [f"{base}/{k}" for k in store.rows]}))
    files = {f"{p[f'{view}_dir']}/{n}": sha(d / n) for n in [*docs, "store.yaml"]}
    (d.parent / f"{view}.UPLOADED.json").write_text(json.dumps({"schema": "pick-and-place-v2-upload/1",
                                                                "files": files}))


def stage_dispatcher(bundle, stem, m):
    bundle = Path(bundle)
    (bundle / "state").mkdir(parents=True)
    jobs = [{"id": f"{stem}-{r['id']}", "episode": r["id"]} for r in m["episodes"]]
    (bundle / "jobs.json").write_text(json.dumps(jobs))
    (bundle / "state/state.json").write_text(json.dumps({"jobs": {j["id"]: {"status": "done"} for j in jobs}}))


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    (repo / "runs/setup/ownership").mkdir(parents=True)
    (repo / "experiments/queue.yaml").write_text(yaml.safe_dump({"jobs": [{"id": "existing-row", "script": "x"}]},
                                                                sort_keys=False))
    (repo / "experiments/registry.jsonl").write_text(json.dumps({"id": "existing-row"}) + "\n")
    (repo / "runs/setup/admission-pauses.json").write_text(json.dumps(V1, indent=2))
    return repo


def snapshot(repo):
    return {p: p.read_bytes() for p in sorted(repo.rglob("*")) if p.is_file()}


@pytest.fixture
def arrived(spec, tmp_path):
    """Dataset arrived and validated, ingest + Phase A + RGB publication staged (rig/Phase B not)."""
    from s4d.rwlearn.dsreset import prep_bundle, write_bundle

    make_raw(Path(spec["roots"]["raw"]))
    report = check(spec)
    m = report["manifest"]
    p = values(spec)
    template = config(ROOT / "configs/realworld/prep/pick-and-place-v2.yaml", spec)
    write_bundle(p["ingest"], {"S": m}, prep_bundle({"S": m}, template, p["remote_processed"]))
    stage_dispatcher(p["phase_a"], reg.STAGE_IDS["phaseA"], m)
    stage_bundle(spec, "rgb", m)
    return report


def test_dry_run_registers_only_the_calibration_independent_wave_first(spec, arrived, tmp_path):
    repo = make_repo(tmp_path)
    before = snapshot(repo)
    ev = reg.evidence(spec, arrived)
    assert reg.ready_waves(ev) == ["rgb"] and ev["rig"]["status"] == "pending"
    record = reg.prepare(repo, spec, arrived, RELEASE)
    assert record["status"] == "dry-run" and record["registered"] is False and record["waves"] == ["rgb"]
    assert len(record["rows"]) == 33 and {r["pnpv2_wave"] for r in record["rows"]} == {"rgb"}
    assert {r["realworld_method"] for r in record["rows"]} == {"rgb", *RGB_METHODS}
    assert snapshot(repo) == before  # dry run writes nothing
    with pytest.raises(reg.Refused, match="not ready"):
        reg.prepare(repo, spec, arrived, RELEASE, waves=["geometry"])


def test_geometry_wave_waits_for_rig_phase_b_publication_and_scene(spec, arrived, tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    m = arrived["manifest"]
    p = values(spec)
    monkeypatch.setattr(reg, "rig_evidence", lambda s: reg._stage("done", "synthetic accepted rig"))
    assert "geometry" not in reg.ready_waves(reg.evidence(spec, arrived))  # Phase B missing
    stage_dispatcher(p["phase_b"], reg.STAGE_IDS["phaseB"], m)
    assert reg.evidence(spec, arrived)["geometry-publication"]["status"] == "pending"
    stage_bundle(spec, "geometry", m, geometry=True)
    assert reg.evidence(spec, arrived)["scene"]["status"] == "pending"
    monkeypatch.setattr(reg, "scene_evidence", lambda s, m, r: reg._stage("done", "synthetic", scene=SCENE))
    record = reg.prepare(repo, spec, arrived, RELEASE, waves=["geometry"])
    assert len(record["rows"]) == 57 and {r["pnpv2_wave"] for r in record["rows"]} == {"geometry"}
    mvp = next(r for r in record["rows"] if r["id"] == row_id("pre", "mvp3d_multiview"))
    assert "render.bounds=" + json.dumps(SCENE["bounds"], separators=(",", ":")) in mvp["args"]


def test_geometry_bundle_with_unaccepted_calibration_is_blocked(spec, arrived, monkeypatch):
    monkeypatch.setattr(reg, "rig_evidence", lambda s: reg._stage("done", "synthetic"))
    stage_dispatcher(values(spec)["phase_b"], reg.STAGE_IDS["phaseB"], arrived["manifest"])
    stage_bundle(spec, "geometry", arrived["manifest"], geometry=False)  # calibration hold not passed
    ev = reg.evidence(spec, arrived)
    assert ev["geometry-publication"]["status"] == "blocked" and "geometry" not in reg.ready_waves(ev)


def test_register_refuses_without_arrived_data(spec, tmp_path):
    repo = make_repo(tmp_path)
    before = snapshot(repo)
    with pytest.raises(reg.Refused, match="check-data must pass"):
        reg.prepare(repo, spec, check(spec), RELEASE)
    spec_file = tmp_path / "spec.yaml"
    doc = yaml.safe_load((ROOT / SPEC).read_text())
    doc["roots"].update(raw=str(tmp_path / "raw"), processed=str(tmp_path / "processed"), work=str(tmp_path / "w"))
    spec_file.write_text(yaml.safe_dump(doc))
    for extra in ([], ["--execute"]):
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/pnpv2.py"), "--task-spec", str(spec_file),
                            "--repo", str(repo), "--settle-seconds", "0", "register", "--commit", "a" * 40,
                            "--image", "sha256:" + "b" * 64, *extra], capture_output=True, text=True, timeout=120)
        assert r.returncode == 2 and "REFUSED" in r.stderr and not r.stdout
    status = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/pnpv2.py"), "--task-spec", str(spec_file),
                             "--repo", str(repo), "status"], capture_output=True, text=True, timeout=120)
    out = json.loads(status.stdout)
    assert out["arrived"] is False and out["would_register"] == {} and out["writes"] is False
    assert out["waves"] == {"rgb": {"rows": 33, "ready_after_check_data": False, "registered": 0},
                            "geometry": {"rows": 57, "ready_after_check_data": False, "registered": 0}}
    assert snapshot(repo) == before and not (tmp_path / "w").exists()


def test_execute_appends_rows_behind_an_exact_pause_and_is_not_repeatable(spec, arrived, tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    record = reg.prepare(repo, spec, arrived, RELEASE)
    monkeypatch.setattr(reg, "require_main", lambda r: None)
    monkeypatch.setattr(reg._reviewed(), "current_pin", lambda r, rel: rel)
    with pytest.raises(reg.Refused, match="check-data"):
        reg.execute(repo, record, recheck=lambda: {**arrived, "passed": False})
    assert not list((repo / "runs/setup").glob("*.pending.json"))
    done = reg.execute(repo, record, recheck=lambda: arrived)
    assert done["status"] == "published" and done["registered"] is True
    queue = yaml.safe_load((repo / "experiments/queue.yaml").read_text())["jobs"]
    assert [r["id"] for r in queue] == ["existing-row", *[r["id"] for r in record["rows"]]]
    pauses = json.loads((repo / "runs/setup/admission-pauses.json").read_text())["pauses"]
    hold = next(h for h in pauses if h["name"] == PAUSE)
    assert hold["active"] is True and hold["prefixes"] == [] and sorted(hold["ids"]) == sorted(
        r["id"] for r in record["rows"])
    v1 = next(h for h in pauses if h["name"] == V1_PAUSE)
    assert v1 == V1["pauses"][0]
    assert len(list((repo / "runs/setup").glob(f"{STEM}-rgb-*.registration.json"))) == 1
    with pytest.raises(reg.Refused):
        reg.execute(repo, record, recheck=lambda: arrived)
    with pytest.raises(reg.Refused, match="already registered|nothing to register"):
        reg.prepare(repo, spec, arrived, RELEASE, waves=["rgb"])


def test_execute_refuses_outside_the_adopted_main(spec, arrived, tmp_path):
    repo = make_repo(tmp_path)
    record = reg.prepare(repo, spec, arrived, RELEASE)
    before = snapshot(repo)
    with pytest.raises(reg.Refused, match="adopted main"):
        reg.execute(repo, record, recheck=lambda: arrived)
    assert snapshot(repo) == before


def test_status_reports_progress_and_next_commands(spec, arrived):
    ev = reg.evidence(spec, None, arrival_state=arrived["arrival"])
    nxt = reg.next_steps(spec, ev)
    assert ev["phaseA"]["status"] == "done" and ev["rgb-publication"]["status"] == "done"
    assert "rig" in nxt and "calib-remote" in nxt["rig"][0]  # remote route; job stem pnpv2-20261117-calib
    assert "wave:rgb" not in nxt  # check-data has not run in this evidence view


def test_register_refuses_while_any_v1_row_is_resumable(spec, arrived, tmp_path):
    repo = make_repo(tmp_path)
    q = repo / "experiments/queue.yaml"
    rows = yaml.safe_load(q.read_text())["jobs"] + [
        {"id": "rwv2-20261113-pre-dynamo-pick-and-place-S", "realworld_target": "pick-and-place"},
        {"id": "rwz-20261118-pre-dynamo-pick-and-place-S", "realworld_target": "pick-and-place"}]
    q.write_text(yaml.safe_dump({"jobs": rows}, sort_keys=False))
    with pytest.raises(reg.Refused, match="rwz-20261118-pre-dynamo-pick-and-place-S"):
        reg.prepare(repo, spec, arrived, RELEASE)
    q.write_text(yaml.safe_dump({"jobs": rows[:2]}, sort_keys=False))  # the paused v1 row alone is fine
    assert reg.prepare(repo, spec, arrived, RELEASE)["waves"] == ["rgb"]


def test_live_v1_pause_still_covers_every_v1_row_and_no_v2_row():
    """Read-only check against the coordinator's live files when present (skipped elsewhere)."""
    main = Path("/home/ws/ws/splatter4d")
    pauses, queue = main / "runs/setup/admission-pauses.json", main / "experiments/queue.yaml"
    if not pauses.is_file() or not queue.is_file():
        pytest.skip("live coordinator files not present")
    from s4d.rwlearn.pnpv2_arrival import v1_unpaused

    doc = json.loads(pauses.read_text())
    if not any(p.get("name") == V1_PAUSE for p in doc["pauses"]):
        pytest.skip("live pauses predate the v1 discontinuation")
    rows = yaml.safe_load(queue.read_text())["jobs"]
    assert v1_unpaused(rows, doc) == []
    ids = [r["id"] for r in build_rows(load_spec(ROOT / SPEC), RELEASE, scene=SCENE)]
    hits, _, problems = v1_collisions(ids, doc)
    assert hits == [] and problems == []
    assert not any(r["id"].startswith(STEM + "-") for r in rows)  # nothing registered yet


def test_calibration_and_followup_producers_use_v2_stems(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from realworld_dsreset_calibration import plan

    ep = f"{TASK}-success-episode_000001"
    cfg_yaml = tmp_path / "ep.yaml"
    cfg_yaml.write_text(yaml.safe_dump({"output_root": str(tmp_path / "processed/x")}))
    phase_jobs = tmp_path / "phaseA-jobs.json"
    phase_jobs.write_text(json.dumps([{"id": f"{dispatcher_stem('prepA')}-{ep}", "episode": ep,
                                       "inputs": {str(cfg_yaml): "0" * 64}}]))
    (tmp_path / "phaseA-state").mkdir()
    phase = tmp_path / "phaseA.json"
    base = dict(host="local", image="sha256:" + "0" * 64, commit="0" * 40, jobs=str(phase_jobs),
                state_dir=str(tmp_path / "phaseA-state"), priority_file=str(tmp_path / "prio/a.json"),
                processed=str(tmp_path / "processed"))
    phase.write_text(json.dumps(base))
    plan(phase, TASK, tmp_path / "calib", tmp_path / "work", tmp_path / "upstream", stem=dispatcher_stem("calib"))
    jobs = json.loads((tmp_path / "calib/jobs.json").read_text())
    ids = [j["id"] for j in jobs]
    assert ids and all(i.startswith(f"{dispatcher_stem('calib')}-{TASK}-") for i in ids)
    assert v1_collisions(ids, V1)[0] == []
    # Scene measurement follow-up: v2 stem and the process-local pnpv2 wrapper (dsreset TASKS lacks v2).
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("store.yaml", "manifest.json", "split_seed0.json", "outcomes.json", "gate_report.json"):
        (inputs / name).write_text("{}")
    rig = tmp_path / "rig.json"
    rig.write_text("{}")
    rig.with_suffix(".reference.npz").write_bytes(b"x")
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/realworld_prep_followup.py"), "measure",
                        "--base-config", str(phase), "--task", TASK, "--rig", str(rig), "--inputs", str(inputs),
                        "--out", str(tmp_path / "measure"), "--result", str(tmp_path / "processed/measurements"),
                        "--id-stem", dispatcher_stem("qa")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    (job,) = json.loads((tmp_path / "measure/jobs.json").read_text())
    assert job["id"] == f"{dispatcher_stem('qa')}-{TASK}-measure" and v1_collisions([job["id"]], V1)[0] == []
    assert job["steps"][0]["argv"][0] == "scripts/pnpv2_realworld_dsreset_measure.py"
