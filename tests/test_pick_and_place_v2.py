"""pick-and-place-v2 science parity, spec validation, process-local task support and runtime readiness."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from s4d.rwlearn.pnpv2_spec import (
    METHODS,
    RGB_METHODS,
    SPEC,
    TASK,
    apply_arrival,
    config,
    load_spec,
    resolve,
    row_id,
    values,
)

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
        timeout=120,
    )


def test_ours_science_is_exact_drawer_s_scratch400k(spec):
    for arm in ("orig", "ch01"):
        expected = raw(f"configs/realworld/ours_matrix/drawer-open-v2-S-{arm}-scratch400k.yaml")
        expected["data"].update(
            task=TASK,
            target=TASK,
            store=values(spec)["geometry_store"],
            split_manifest=values(spec)["geometry_split"],
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
    expected["data_interface"].update(target=TASK, split_manifest=p[view + "_split"], gate_report=p[view + "_gate"])
    assert config(ROOT / f"configs/realworld/baselines/{method}/{TASK}.yaml", spec) == expected
    if method == "snerl":
        assert expected["data_interface"]["lookat"] is None and expected["data_interface"]["pose_scale"] is None
    if method == "mvp3d":
        assert expected["render"]["bounds"] is None


def test_one_spec_resolves_every_dataset_path_without_changing_science(spec):
    spec["roots"] = {key: f"/new-collection/{key}" for key in spec["roots"]}
    spec["scene"] = {"lookat": [1, 2, 3], "pose_scale": 4.0, "bounds": [0, 0, 0, 1, 1, 1]}
    spec["sam3_prompts"] = {"robot arm": "robot", "cup": "object"}
    for path in (ROOT / "configs/realworld").rglob(f"{TASK}*.yaml"):
        if path == ROOT / SPEC:
            continue
        cfg = config(path, spec)
        text = json.dumps(cfg)
        assert "${" not in text and "__TODO_" not in text and "drawer-open-v2" not in text
        assert "/home/compu/" not in text
    assert values(spec)["raw_success"] == f"/new-collection/raw/{TASK}/success"
    assert values(spec)["rgb_store"] == f"/new-collection/publication/{TASK}/S/rgb/store.yaml"
    assert resolve("${pnpv2.lookat}", spec) == [1, 2, 3]
    with pytest.raises(ValueError, match="unknown"):
        resolve("${pnpv2.typo}", spec)
    with pytest.raises(ValueError, match="embedded"):
        resolve("prefix/${pnpv2.rgb_split}", spec)


def test_fql_configs_cover_all_retained_conditions(spec):
    for method in METHODS:
        cfg = config(ROOT / f"configs/realworld/fql/{TASK}-{method}.yaml", spec)
        pixel = method.startswith("e2e")
        # frozen encoders: a10nq; E2E CNN: alpha 10 with normalize_q_loss false (established E2E setting)
        assert cfg["agent"]["alpha"] == 10.0 and cfg["agent"]["normalize_q_loss"] is (not pixel)
        assert cfg["train"]["offline_steps"] == 1000000
        assert cfg["data"]["smoke_without_calibration"] is False
        assert cfg["encoder"]["feature_norm"] == ("train_standardize" if method.startswith("mvp3d") else "none")
        assert cfg["data"]["dataset"] == f"runs/{row_id('fqldata', method)}/dataset.h5"
        view = "rgb" if method in RGB_METHODS else "geometry"
        assert cfg["data"]["split_manifest"] == values(spec)[view + "_split"]
        if pixel:
            assert cfg["condition"] == "e2e_cnn" and cfg["data"]["store"] == values(spec)["rgb_store"]
            assert cfg["pixel"]["p_aug"] == (0.0 if method.endswith("noaug") else 0.5)
    assert not list((ROOT / "configs/realworld/eval").glob(f"{TASK}-*.yaml"))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(task="pick-and-place"),
        lambda s: s.update(variants=["S", "SF"]),
        lambda s: s["collection"].update(failures_enabled=True),
        lambda s: s["collection"].update(expected_success=98),
        lambda s: s["calibration"].update(acceptance_override="drawer-open-v2-20261117"),
        lambda s: s["roots"].update(raw="/data/../legacy"),
        lambda s: s["roots"].update(publication="/home/ws/local-only"),
        lambda s: s.update(sam3_prompts={"robot arm": "robot"}),
        lambda s: s["arrival"].update(stable_seconds=0),
    ],
)
def test_no_v1_sf_count_amendment_or_override(tmp_path, spec, mutate):
    mutate(spec)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):
        load_spec(path)


def test_arrival_confirmation_is_narrow(spec):
    good = dict(schema="pick-and-place-v2-arrival/1", task=TASK, sam3_prompts={"robot arm": "robot", "cup": "object"},
                robot_urdf_sha256="c" * 64, confirmed_by="operator")
    assert apply_arrival(spec, good)["sam3_prompts"] == good["sam3_prompts"]
    for bad in ({**good, "roots": {}}, {**good, "sam3_prompts": {"cup": "object"}}, {**good, "confirmed_by": " "},
                {**good, "task": "pick-and-place"}):
        with pytest.raises(ValueError):
            apply_arrival(spec, bad)


def test_producers_refuse_without_data_before_trainer_imports(tmp_path):
    doc = yaml.safe_load((ROOT / SPEC).read_text())
    doc["roots"].update(raw=str(tmp_path / "raw"), work=str(tmp_path / "work"), processed=str(tmp_path / "p"))
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(doc))
    code = f"""
from scripts.pick_and_place_v2 import main
for entry in ('manifest', 'publish-rgb', 'rig-map'):
    try: main(['--entry', entry, '--task-spec', {str(path)!r}, '--settle-seconds', '0'])
    except (ValueError, OSError) as error: pass
    else: raise AssertionError(entry + ' admitted without data')
assert 'scripts.train' not in sys.modules
"""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "work").exists()


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


def test_readiness_barrier_uses_real_full_data_and_calibration_gates(tmp_path):
    code = """
import copy, json
from pathlib import Path
from s4d.rwlearn.pnpv2_runtime import install
install()
from s4d.rwlearn import gate
from s4d.rwlearn.pnpv2_spec import TASK, SPEC, load_spec
from s4d.rwlearn.pnpv2_publication import readiness
from tests.test_rwlearn_gate import entry, CAMS
spec=load_spec(SPEC)
root=Path(TEMP)
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
def write(name,doc):
    path=root/name;path.write_text(json.dumps(doc));return path
def report(validated):
    rows=copy.deepcopy(ingest)
    if validated:
        for ep,r in rows.items():
            r['preprocessing']['stages']=entry(TASK,r['episode_index'],validated=True)['preprocessing']['stages']
    return gate.evaluate(dict(schema=gate.SCHEMA,schema_version=gate.SCHEMA_VERSION,episodes=rows,
                              counts={TASK:dict(target=100,complete=True)}),[TASK],amendments={})
m,s,o=write('manifest.json',manifest),write('split.json',split),write('outcomes.json',outcomes)
def ready(stage,gate_path):
    return readiness(stage,m,s,o,gate_path,'store.yaml',spec,open_store=lambda path:store)
unaccepted=write('unaccepted.json',report(False))
assert ready('prep',unaccepted)['consumers']==['dynamo','reviwo','e2e_cnn']
try:ready('calib',unaccepted)
except RuntimeError as e:assert 'calibration' in str(e)
else:raise AssertionError('geometry admitted without calibration')
accepted=write('accepted.json',report(True))
assert ready('calib',accepted)['view']=='geometry'
store.build='stale-calibration'
try:ready('calib',accepted)
except RuntimeError as e:assert 'store built from calibrate' in str(e)
else:raise AssertionError('stale calibration admitted')
store.build='cal-build-1'
write('outcomes.json',{**outcomes,episodes[0]['id']:False})
try:ready('prep',unaccepted)
except ValueError as e:assert 'outcomes' in str(e)
else:raise AssertionError('wrong S outcome admitted')
write('outcomes.json',outcomes)
byid[episodes[0]['id']]={**episodes[0],'sha256':'f'*64}
try:ready('prep',unaccepted)
except ValueError as e:assert 'raw hashes' in str(e)
else:raise AssertionError('different raw identity admitted')
""".replace("TEMP", repr(str(tmp_path)))
    result = run_python(code)
    assert result.returncode == 0, result.stderr
