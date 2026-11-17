"""Preparation-only new dataset: science parity, fail-closed gates and no side effects."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ops.jobs.pnpv2_plan import SCRIPT, build_rows
from ops.jobs.pnpv2_registration import proposal, registration_request
from s4d.rwlearn.pnpv2_spec import METHODS, RGB_METHODS, SPEC, TASK, config, load_spec, resolve, row_id, values

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def spec():
    return load_spec(ROOT / SPEC)


def raw(path):
    return yaml.safe_load((ROOT / path).read_text())


def run_python(code):
    return subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r});\n" + code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_ours_science_is_exact_drawer_s_scratch400k(spec):
    for arm in ("orig", "ch01"):
        expected = raw(f"configs/realworld/ours_matrix/drawer-open-v2-S-{arm}-scratch400k.yaml")
        expected["data"].update(
            task=TASK,
            target=TASK,
            store=values(spec)["geometry_store"],
            split_manifest=values(spec)["split"],
            gate_report=values(spec)["geometry_gate"],
        )
        expected["model"]["anchor_stats"] = values(spec)["workspace_stats"]
        actual = config(ROOT / f"configs/realworld/ours_matrix/{TASK}-S-{arm}-scratch400k.yaml", spec)
        assert actual == expected
        assert actual["train"]["steps"] == 400000 and "restarts" not in actual["train"]
        assert actual["model"]["encoder"]["backbone"] == "scratch"
        assert actual["loss"].get("chamfer", 0) == (0.1 if arm == "ch01" else 0)


@pytest.mark.parametrize("method", ["dynamo", "reviwo", "sincro", "snerl", "mvp3d"])
def test_baseline_science_and_unmeasured_geometry(spec, method):
    expected = raw(f"configs/realworld/baselines/{method}/drawer-open-v2.yaml")
    p = values(spec)
    view = "rgb" if method in RGB_METHODS else "geometry"
    expected["dataset"]["hdf5_path"] = p[view + "_store"]
    expected["data_interface"].update(target=TASK, split_manifest=p["split"], gate_report=p[view + "_gate"])
    assert config(ROOT / f"configs/realworld/baselines/{method}/{TASK}.yaml", spec) == expected
    if method == "snerl":
        assert expected["data_interface"]["lookat"] is None and expected["data_interface"]["pose_scale"] is None
    if method == "mvp3d":
        assert expected["render"]["bounds"] is None


def test_one_spec_resolves_every_dataset_path_without_changing_science(spec):
    spec["roots"] = {key: f"/new-collection/{key}" for key in spec["roots"]}
    spec["scene"] = {"lookat": [1, 2, 3], "pose_scale": 4.0, "bounds": [0, 0, 0, 1, 1, 1]}
    for path in (ROOT / "configs/realworld").rglob(f"{TASK}*.yaml"):
        if path == ROOT / SPEC:
            continue
        cfg = config(path, spec)
        text = json.dumps(cfg)
        assert "${" not in text and "__TODO_" not in text and "drawer-open-v2" not in text
        assert "/home/compu/" not in text
    assert values(spec)["raw_success"] == f"/new-collection/raw/{TASK}/success"
    assert resolve("${pnpv2.lookat}", spec) == [1, 2, 3]
    with pytest.raises(ValueError, match="unknown"):
        resolve("${pnpv2.typo}", spec)
    with pytest.raises(ValueError, match="embedded"):
        resolve("prefix/${pnpv2.split}", spec)


def test_fql_and_eval_cover_all_retained_conditions(spec):
    for method in METHODS:
        cfg = config(ROOT / f"configs/realworld/fql/{TASK}-{method}.yaml", spec)
        assert cfg["agent"]["alpha"] == 10.0 and cfg["agent"]["normalize_q_loss"] is True
        assert cfg["train"]["offline_steps"] == 1000000
        assert cfg["data"]["smoke_without_calibration"] is False
        assert cfg["encoder"]["feature_norm"] == ("train_standardize" if method.startswith("mvp3d") else "none")
        assert cfg["data"]["dataset"] == f"runs/{row_id('fqldata', method)}/dataset.h5"
        if method.startswith("e2e"):
            assert cfg["condition"] == "e2e_cnn" and cfg["data"]["store"] == values(spec)["rgb_store"]
            assert cfg["pixel"]["p_aug"] == (0.0 if method.endswith("noaug") else 0.5)
        ev = config(ROOT / f"configs/realworld/eval/{TASK}-{method}.yaml", spec)
        assert ev["method"] == method and ev["kind"] == "policy" and ev["required_steps"] == 1000000
        assert ev["require_final_validation"] is True and ev["robot_io"] is False and ev["actuation_eligible"] is False


def test_complete_s_graph_and_calibration_independent_routes(spec):
    rows = build_rows(spec, ROOT)
    byid = {r["id"]: r for r in rows}
    assert len(rows) == len(byid) == 88
    assert {r["realworld_method"] for r in rows if "--entry" in r["args"] and r["args"][1] == "fql"} == set(METHODS)
    for row in rows:
        assert all(dep in byid for dep in row["deps"])
        assert row["script"] == SCRIPT and row["preparation_only"] is True
        assert TASK in row["id"] and "-SF" not in row["id"] and "mug-hanging" not in json.dumps(row)
        assert row["planned_commit"] is None and row["planned_image"] is None

    def ancestors(rid):
        return {dep for d in byid[rid]["deps"] for dep in ({d} | ancestors(d))}

    for method in METHODS:
        for seed in (0, 1, 2):
            chain = ancestors(row_id("eval", method, seed))
            view = "rgb" if method in RGB_METHODS else "geometry"
            assert row_id("gate", view) in chain
            assert row_id("gate", "geometry" if view == "rgb" else "rgb") not in chain
            assert row_id("fql", method, seed) in chain
    # Single/multiview MVP share one fused representation, distinct FQL conditions.
    assert row_id("export", "mvp3d_multiview") in ancestors(row_id("fql", "mvp3d_single", 0))
    assert row_id("native", "mvp3d") in ancestors(row_id("fql", "mvp3d_single", 0))
    assert not any("-pre-e2e_" in row["id"] for row in rows)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(task="pick-and-place"),
        lambda s: s.update(variants=["S", "SF"]),
        lambda s: s["collection"].update(failures_enabled=True),
        lambda s: s["collection"].update(expected_success=98),
        lambda s: s["calibration"].update(acceptance_override="drawer-open-v2-20261117"),
        lambda s: s["roots"].update(raw="/data/../legacy"),
    ],
)
def test_no_v1_sf_count_amendment_or_override(tmp_path, spec, mutate):
    mutate(spec)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):
        load_spec(path)


def test_registrar_cli_is_stdout_only_and_all_registration_requests_refuse(tmp_path, spec):
    queue = ROOT / "experiments/queue.yaml"
    registry = ROOT / "experiments/registry.jsonl"
    before = (queue.read_bytes(), registry.read_bytes())
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/register_pick_and_place_v2.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["status"] == "dry-run" and record["registered"] is False and record["launch_authorized"] is False
    assert record["rows"] == proposal(ROOT / SPEC)["rows"]
    auth = tmp_path / "authorization.json"
    with pytest.raises(ValueError, match="explicit regular authorization"):
        registration_request(record, None)
    with pytest.raises(ValueError, match="explicit regular authorization"):
        registration_request(record, auth)
    auth.write_text("{}")
    with pytest.raises(ValueError, match="explicitly bind"):
        registration_request(record, auth)
    auth.write_text(
        json.dumps(
            dict(
                schema="pick-and-place-v2-registration-authorization/1",
                task=TASK,
                authorize_registration=True,
                authorized_by="test only",
                spec_sha256=record["spec_sha256"],
                rows_sha256=record["rows_sha256"],
            )
        )
    )
    with pytest.raises(ValueError, match="queue publication is not implemented"):
        registration_request(record, auth)
    for flags in (["--register"], ["--authorization-file", str(auth)], ["--register", "--authorization-file", str(auth)]):
        r = subprocess.run(
            [sys.executable, "-I", str(ROOT / "scripts/register_pick_and_place_v2.py"), *flags],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert r.returncode == 2 and "REFUSED" in r.stderr and not r.stdout
    assert (queue.read_bytes(), registry.read_bytes()) == before
    assert set(p.name for p in tmp_path.iterdir()) == {"authorization.json"}


def test_execution_refuses_uncollected_spec_before_any_runtime_imports():
    code = """
from scripts.pick_and_place_v2 import main
try: main(['--entry','train'])
except ValueError as error: assert 'not collected' in str(error)
else: raise AssertionError('uncollected dataset admitted')
assert 's4d.rwlearn.pnpv2_runtime' not in sys.modules
assert 'scripts.train' not in sys.modules
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr


def test_opt_in_runtime_leaves_legacy_registries_unchanged():
    code = """
from s4d.rwlearn import tasks, configs, dsreset
legacy_tasks, legacy_scene = tasks.TARGETS, dict(configs.SCENE_PARAMETERS)
from s4d.rwlearn.pnpv2_runtime import install
install()
from s4d.rwlearn import tasks as new_tasks, configs as new_configs
assert 'pick-and-place-v2' in new_tasks.TARGETS
assert 'pick-and-place-v2' in new_configs.TARGET_TASKS
assert 'pick-and-place-v2' in dsreset.TASKS
assert tasks.TARGETS == legacy_tasks
assert configs.SCENE_PARAMETERS == legacy_scene
assert 'pick-and-place-v2' not in legacy_scene
assert new_configs.SCENE_PARAMETERS['pick-and-place-v2'] == dict(lookat=None,pose_scale=None,bounds=None)
install()
assert new_tasks.TARGETS.count('pick-and-place-v2') == 1
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr


def test_no_task_override_can_be_carried_into_publication():
    from s4d.rwlearn.pnpv2_publication import no_override

    no_override({"acceptance_override": None})
    with pytest.raises(ValueError, match="no calibration acceptance override"):
        no_override(
            {
                "tasks": {
                    "pick-and-place-v2": {"episodes": [{"metrics": {"acceptance_override": {"task": "drawer-open-v2"}}}]}
                }
            }
        )


def test_new_files_do_not_change_placed_execution_scopes():
    from ops.jobs.dsreset_scope import audit

    snapshot = ROOT / "tests/fixtures/dsreset-live-scopes.yaml"
    result = audit(snapshot, ROOT)
    assert result["zero_release_changes"], [
        (r["job_id"], r["release_changes"]) for r in result["rows"] if r["release_changes"]
    ]
    assert result["same_scope_differences_as_main"]


def test_publication_uses_real_full_data_and_calibration_gates(tmp_path):
    code = """
import copy, json
from pathlib import Path
from s4d.rwlearn.pnpv2_runtime import install
install()
from s4d.rwlearn import gate, store as stores
from s4d.rwlearn.pnpv2_spec import TASK, SPEC, load_spec, values
from s4d.rwlearn.pnpv2_publication import check_publication
from tests.test_rwlearn_gate import entry, CAMS
spec=load_spec(SPEC)
root=Path(TEMP)
spec['roots']={k:str(root/k) for k in spec['roots']}
p=values(spec)
episodes=[]
ingest={}
for i in range(100):
    name=f'episode_{i:06d}.h5'
    ep=f'{TASK}-success-{name[:-3]}'
    rec=entry(TASK,i,validated=False)
    rec['filename']=f'/raw/{TASK}/success/{name}'
    rec['episode_label']='success'
    ingest[ep]=rec
    episodes.append(dict(id=ep,task=TASK,label='success',filename=name,path=rec['filename'],
                         sha256=rec['sha256'],split='train' if i<90 else 'validation',qa={}))
split={s:[r['id'] for r in episodes if r['split']==s] for s in ('train','validation')}
outcomes={r['id']:True for r in episodes}
manifest=dict(schema='dsreset-episodes/v1',task=TASK,variant='S',seed=0,validation_fraction=.1,
              episodes=episodes,excluded=[],amendment=None,split=split,outcomes=outcomes)
byid={r['id']:r for r in episodes}
class Store:
    build='cal-build-1'
    def episodes(self):return list(byid)
    def task(self,ep):return TASK
    def raw_recording(self,ep):return byid[ep]['path']
    def raw_sha256(self,ep):return byid[ep]['sha256']
    def cameras(self,ep):return CAMS
    def upstream(self,ep):return {'calibrate':self.build}
store=Store()
stores.open_store=lambda path:store

def write(key,doc):
    path=Path(p[key]);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(doc))
def report(validated):
    rows=copy.deepcopy(ingest)
    if validated:
        for ep,r in rows.items():
            r['preprocessing']['stages']=entry(TASK,r['episode_index'],validated=True)['preprocessing']['stages']
    return gate.evaluate(dict(schema=gate.SCHEMA,schema_version=gate.SCHEMA_VERSION,episodes=rows,
                              counts={TASK:dict(target=100,complete=True)}),[TASK],amendments={})
for key,doc in [('manifest',manifest),('split',split),('outcomes',outcomes)]:write(key,doc)
unaccepted=report(False)
write('rgb_gate',unaccepted)
write('geometry_gate',unaccepted)
assert check_publication(spec,'rgb')['passed']
try:check_publication(spec,'geometry')
except RuntimeError as e:assert 'calibration' in str(e)
else:raise AssertionError('geometry admitted without calibration')
accepted=report(True)
write('geometry_gate',accepted)
assert check_publication(spec,'geometry')['passed']
store.build='stale-calibration'
try:check_publication(spec,'geometry')
except RuntimeError as e:assert 'store built from calibrate' in str(e)
else:raise AssertionError('stale calibration admitted')
store.build='cal-build-1'
write('outcomes',{**outcomes,episodes[0]['id']:False})
try:check_publication(spec,'rgb')
except ValueError as e:assert 'outcomes' in str(e)
else:raise AssertionError('wrong S outcome admitted')
write('outcomes',outcomes)
byid[episodes[0]['id']]={**episodes[0],'sha256':'f'*64}
try:check_publication(spec,'rgb')
except ValueError as e:assert 'raw hashes' in str(e)
else:raise AssertionError('different raw identity admitted')
""".replace("TEMP", repr(str(tmp_path)))
    result = run_python(code)
    assert result.returncode == 0, result.stderr
