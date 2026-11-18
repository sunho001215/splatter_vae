"""Exact held PNP evidence and opt-in offline consumers; robot and placed RGB bytes remain unchanged."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from s4d.realworld.calib import pnpv2_acceptance_override as A
from s4d.realworld.calib import rig
from s4d.rwlearn import pnpv2_offline as P
from s4d.rwlearn.pnpv2_spec import SPEC, TASK, load_spec, values
from tests.test_pnpv2_geometry_route import _hermetic_imports as _hermetic_imports

ROOT = Path(__file__).resolve().parents[1]
CAMS = ('135122070988/color', '135122073201/color', '135122076054/color')


@pytest.fixture
def world(tmp_path):
    doc = json.loads((ROOT / 'tests/fixtures/pnpv2-calibration-held.json').read_text())
    evidence = json.loads((ROOT / 'tests/fixtures/pnpv2-calibration-evidence.json').read_text())
    for n, text in evidence.items():
        (tmp_path / n).write_text(text)
    import shutil
    shutil.copyfile(ROOT / 'tests/fixtures/pnpv2-calibration.reference.npz', tmp_path / 'fixedness.reference.npz')
    rec, _, _ = A.record_for(TASK)
    kw = dict(task=TASK, work=tmp_path, specpath=tmp_path / 'spec.json', static={}, aliases={},
              automatic=json.loads(evidence['automatic-decision.json']), commit=rec['automatic_acceptance'][
                  'criteria_commit'])
    return doc, kw, rec


def published(world):
    doc, kw, _ = world
    out = A.apply_publisher_override(doc, **kw)
    path = kw['work'] / 'rig-pick-and-place-v2-accepted.json'
    with np.load(kw['work'] / 'fixedness.reference.npz') as refs:
        rig.publish_rig(path, out, {k:refs[k] for k in refs.files})
    return rig.load_rig(path)[0], path


def test_committed_record_and_publisher_exact_evidence(world):
    doc, kw, rec = world
    assert rec['retained'] == dict(episodes=100, cameras=3, excluded_episodes=[], excluded_cameras=[])
    rot, centre, motion1, motion2 = rec['exceptions']
    assert (rot['measured'], rot['threshold']) == (0.3594761331966316, 0.35)
    assert (centre['measured'], centre['threshold']) == (0.009473656243675717, 0.008)
    assert [int(e[-6:]) for e in motion1['registration']['moved']] == [56,58,63,66,67,68,71,73,75,77,81,86]
    assert motion2['registration']['moved'] == ['pick-and-place-v2-000003']
    assert len(motion2['registration']['not_verifiable']) == 96
    before = {n: A.sha256_file(kw['work'] / n) for n in ('automatic-decision.json', 'verdict-release1.json')}
    effective = A.apply_publisher_override(doc, **kw)
    assert A.apply_publisher_override(doc, **kw) == effective
    assert before == {n: A.sha256_file(kw['work'] / n) for n in before}
    assert A.offline_acceptance(effective)['statuses'] == {CAMS[0]: 'validated', CAMS[1]: A.STATUS, CAMS[2]: A.STATUS}
    assert [effective['cameras'][c]['status'] for c in CAMS] == ['validated', 'held', 'held']
    assert '100 episodes and 3 cameras retained' in json.loads((kw['work'] / 'override-decision.json').read_text())['verified']


@pytest.mark.parametrize('mutate', [
    lambda d: d['cameras'][CAMS[1]]['empirical_spread'].update(max_pair_rot_deg=0.36),
    lambda d: d['cameras'][CAMS[1]]['empirical_spread'].update(max_pair_centre_m=0.0095),
    lambda d: d['cameras'][CAMS[2]]['validation']['fixedness']['registration']['moved'].append('other'),
    lambda d: d['cameras'][CAMS[0]].update(status='held', reasons=['G1: unexpected']),
    lambda d: d['cameras'][CAMS[1]]['diagnostics']['criteria_v2']['g1']['thresholds'].update(radius_rot_deg=0.36),
    lambda d: d['cameras'][CAMS[1]]['reasons'].append('G2: unexpected'),
    lambda d: d['cameras'].update(other=d['cameras'].pop(CAMS[2])),
    lambda d: d.update(task='drawer-open-v2'),
    lambda d: d['episodes'].pop(),
])
def test_any_evidence_drift_is_held(world, mutate):
    doc, kw, _ = world
    mutate(doc)
    with pytest.raises(ValueError, match='held'):
        A.apply_publisher_override(doc, **kw)
    assert not (kw['work'] / 'override-decision.json').exists()


@pytest.mark.parametrize('mutate', [
    lambda d: d['acceptance_override']['record']['scope'].update(robot_execution_authorized=True),
    lambda d: d['acceptance_override'].update(robot_execution_authorized=True),
    lambda d: d['cameras'][CAMS[2]]['validation']['fixedness']['registration']['not_verifiable'].pop(),
    lambda d: d['cameras'][CAMS[0]].update(T_cam_base=np.eye(4).tolist()),
    lambda d: d.update(reference_sha256='0' * 64),
    lambda d: d.update(task='other'),
])
def test_tampered_consumers_fail_closed(world, mutate):
    doc, _ = published(world)
    mutate(doc)
    assert not A.offline_acceptance(doc)['accepted']


def test_missing_committed_record_keeps_consumers_held(world, monkeypatch):
    doc, _ = published(world)
    monkeypatch.setattr(A, 'RECORDS', {})
    assert not A.offline_acceptance(doc)['accepted']
    with pytest.raises(ValueError, match='committed'):
        P.check_reference(A.summary(doc))


def test_offline_gate_and_robot_reader(world):
    from s4d.rwlearn.deploy.realsense import load_rig_session
    doc, path = published(world)
    assert P.rig_gate(path)['mode'] == 'acceptance_override'
    cams = [c.split('/')[0] for c in CAMS]
    K = {c: np.eye(3) for c in cams}
    with pytest.raises(ValueError, match='not validated'):
        load_rig_session(path, cams, K)
    assert [rig.rig_camera_status(doc, c)[0] for c in CAMS] == ['validated', 'held', 'held']
    offline = A.load_rig_session_offline(path, cams, K)
    assert offline['native_depth_correction'][cams[1]]['evidence']['effective_rig_status'] == A.STATUS
    (path.parent / 'override-decision.json').write_text('{}')
    with pytest.raises(ValueError, match='override-decision'):
        P.rig_gate(path)


def test_g5a_episode_override_is_exact_and_process_local(world):
    doc, _ = published(world)
    rec = world[2]
    original = rig.release_fixedness
    restore = A.install_rig_override()
    try:
        for exc in rec['exceptions'][2:]:
            cam = exc['cameras'][0]
            for alias, motion in exc['motions'].items():
                episode = next(e for e, a in rec['episode_aliases'].items() if a == alias)
                cfg = rig.FixednessConfig(**rec['fixedness_config'])
                result = rig.release_fixedness(doc, cam, episode, motion, cfg)
                assert result['fixed'] and result['registration']['moved'] and result['strict_segment_test_fixed'] is False
                drift = copy.deepcopy(motion)
                drift['segments'][0]['shift_px'] += 1e-6
                assert not rig.release_fixedness(doc, cam, episode, drift, cfg)['fixed']
                assert not rig.release_fixedness(doc, cam, 'other-episode', motion, cfg)['fixed']
                assert not rig.release_fixedness(doc, CAMS[0], episode, motion, cfg)['fixed']
                moved_epipolar = copy.deepcopy(doc)
                moved_epipolar['fixedness_evidence'][cam][episode]['epipolar'] = 'moved'
                assert not rig.release_fixedness(moved_epipolar, cam, episode, motion, cfg)['fixed']
        # Unflagged episodes must still pass G5a; the camera-level held status alone must not hold them.
        cam = CAMS[1]
        ep = doc['episodes'][0]
        motion = dict(fixed=True, segments=[dict(patches=10, inlier_patches=10, shift_px=0, fixed=True)])
        assert rig.release_fixedness(doc, cam, ep, motion, cfg)['fixed']
    finally:
        restore()
    assert rig.release_fixedness is original


def test_publication_adapter_only_accepts_committed_references(world):
    from s4d.rwlearn import pnpv2_publication as legacy
    doc, _ = published(world)
    ref = A.summary(doc)
    P.no_override({'acceptance_override': [ref]})
    with pytest.raises(ValueError):
        legacy.no_override({'acceptance_override': [ref]})
    for changes in ({'task': 'other'}, {'record_sha256': '0' * 64}, {'robot_execution_authorized': True},
                    {'cameras': ['other']}, {'status': 'validated'}, {'exceptions': []}):
        with pytest.raises(ValueError):
            P.no_override({'nested': [{'acceptance_overrides': [{**ref, **changes}]}]})
    original = legacy.no_override
    with P.publication_readers():
        legacy.no_override({'acceptance_override': [ref]})
    assert legacy.no_override is original


def test_registration_rig_and_new_geometry_resolvers(world, tmp_path):
    from ops.jobs import pnpv2_registration as reg, host_roles  # noqa: F401
    from s4d.rwlearn import dsreset_entrypoints
    doc, path = published(world)
    spec = load_spec(ROOT / SPEC)
    spec['roots']['processed'] = str(tmp_path / 'processed')
    local = Path(values(spec)['rig_local'])
    local.parent.mkdir(parents=True)
    import shutil
    for f in (path, path.with_suffix('.reference.npz'), path.parent / 'automatic-decision.json',
              path.parent / 'override-decision.json'):
        shutil.copyfile(f, local.parent / f.name)
    stage = reg.rig_evidence(spec)
    assert stage['status'] == 'done' and stage['acceptance_override'] == A.summary(doc)
    assert len(stage['inputs']) == 4
    for wrapper, original in P.OFFLINE_WRAPPERS.items():
        assert dsreset_entrypoints.legacy_row({'script': wrapper})['script'] == original
    (local.parent / 'override-decision.json').unlink()
    assert reg.rig_evidence(spec)['status'] == 'blocked'


def test_phase_b_planner_selects_the_override_compiler(world, tmp_path, monkeypatch):
    import subprocess
    import sys
    import scripts.pick_and_place_v2 as producer
    _, path = published(world)
    base = tmp_path / 'A'
    base.mkdir()
    (base / 'dispatcher.json').write_text(json.dumps({k: '/' + k for k in
                                                    ('processed', 'raw', 'weights', 'image', 'tools', 'assets')}))
    (base / 'prep-plan.json').write_text(json.dumps([{'episode': 'one'}, {'episode': 'two'}]))
    out = tmp_path / 'B'
    calls = []
    def compile(args, check):
        calls.append(args)
        out.mkdir()
        (out / 'jobs.json').write_text(json.dumps([{'disk_gib': 32}]))
    monkeypatch.setattr(subprocess, 'run', compile)
    monkeypatch.setattr(sys, 'argv', ['plan', str(base), str(out), str(path), 'pnpv2-20261117-prepB',
                                     str(path.parent), '4', 'acceptance_override'])
    exec(producer._PHASE_B_PLAN, {})
    assert calls[0][1] == 'scripts/realworld_prep_register_ao.py'
    assert json.loads((out / 'jobs.json').read_text())[0]['disk_gib'] == 4


def test_published_geometry_propagates_committed_reference(world, tmp_path, monkeypatch):
    # Exercise the unchanged learning-inputs producer through the opt-in reader, with actual gate inputs.
    from tests.test_pnpv2_geometry_route import ingest_for, FakeStore
    from s4d.rwlearn.pnpv2_arrival import task_known
    from s4d.rwlearn import dsreset, dsreset_gate, configs, store
    doc, _ = published(world)
    spec = load_spec(ROOT / SPEC)
    raw = tmp_path / 'data/real_world' / TASK / 'success'
    raw.mkdir(parents=True)
    for i in range(100):
        (raw / f'episode_{i:06d}.h5').write_text(str(i))
    with task_known() as D:
        manifest = D.paired_manifests(TASK, {'success': raw}, variants=('S',), require_verified=False,
                                      inspect=lambda p: {}, expected={'success': 100})['S']
    ingest = ingest_for(manifest, validated=True)
    ref = A.summary(doc)
    for row in ingest['episodes'].values():
        row['preprocessing']['stages']['calibration_validated']['metrics']['acceptance_override'] = ref
    monkeypatch.setattr(dsreset_gate._impl, 'TARGETS', (*dsreset_gate._impl.TARGETS, TASK))
    monkeypatch.setattr(configs, 'TARGET_TASKS', {**configs.TARGET_TASKS, TASK: (TASK,)})
    monkeypatch.setattr(store, 'open_store', lambda *a: FakeStore(manifest))
    m, i = tmp_path / 'manifest.json', tmp_path / 'ingest.json'
    m.write_text(json.dumps(manifest)); i.write_text(json.dumps(ingest))
    before = dsreset.learning_inputs
    with task_known():
        P.build_geometry(spec, m, i, tmp_path / 'publication')
    assert dsreset.learning_inputs is before
    report = json.loads((tmp_path / 'publication/gate_report.json').read_text())
    assert report['acceptance_override'] == [ref]
    result = P.readiness('calib', tmp_path / 'publication/manifest.json', tmp_path / 'publication/split_seed0.json',
                         tmp_path / 'publication/outcomes.json', tmp_path / 'publication/gate_report.json',
                         tmp_path / 'publication/store.yaml', spec)
    assert result['passed'] and result['effective_rig_status'] == A.STATUS


@pytest.mark.parametrize('name', ['fixedness.json', 'static.json', 'fixedness.reference.npz'])
def test_publisher_preserves_original_source_hashes(world, name):
    doc, kw, _ = world
    source = kw['work'] / name
    source.write_bytes(source.read_bytes() + b' ')
    with pytest.raises(ValueError, match='recorded evidence'):
        A.apply_publisher_override(doc, **kw)
    assert not (kw['work'] / 'override-decision.json').exists()


def test_scene_refuses_an_uncommitted_override_reference(world, tmp_path, monkeypatch):
    from ops.jobs import pnpv2_registration as reg
    from s4d.rwlearn import dsreset_measure
    doc, path = published(world)
    spec = load_spec(ROOT / SPEC)
    spec['roots']['processed'] = str(tmp_path / 'processed')
    scene = Path(values(spec)['scene_report'])
    scene.parent.mkdir(parents=True)
    scene.write_text(json.dumps(dict(native_provenance=dict(acceptance_override=[
        {**A.summary(doc), 'task': 'other'}]))))
    monkeypatch.setattr(dsreset_measure, 'validate_report', lambda *a: pytest.fail('invalid reference admitted'))
    assert reg.scene_evidence(spec, {}, {})['status'] == 'blocked'



def test_pnp_installation_is_reversible_and_does_not_change_the_shared_drawer_reader(world):
    from s4d.realworld.calib import acceptance_override as shared
    before = dict(vars(shared))
    assert shared.record_for(TASK) is None
    with A.installed():
        assert shared.record_for(TASK)[0]['task'] == TASK
        with A.installed():
            assert shared.record_for(TASK)[0]['task'] == TASK
    assert all(vars(shared).get(k) is v for k, v in before.items())
    assert shared.record_for(TASK) is None


def test_actual_phase_b_compiler_uses_the_isolated_worker_and_restores_shared_ao(world, tmp_path):
    import yaml
    from tests.test_calibration_acceptance_override import _script
    from s4d.realworld.calib import acceptance_override as shared

    doc, path = published(world)
    rows = []
    for n, ep in enumerate(doc['episodes'][:2]):
        cfg = tmp_path / f'pipeline{n}.yaml'
        cfg.write_text(yaml.safe_dump(dict(output_root=str(tmp_path / f'p{n}'),
            adapter={'config': {'episodes': {ep: f'/home/ws/data/real_world/{TASK}/success/episode_{n:06d}.h5'}}},
            stages={'calibrate': {'rig_calibration': '/pending'}})))
        rows.append(dict(episode=ep, config=str(cfg), raw_sha256='c' * 64))
    plan, rmap, tools = tmp_path / 'plan.json', tmp_path / 'map.json', tmp_path / 'tools'
    plan.write_text(json.dumps(rows)); rmap.write_text(json.dumps({r['episode']: str(path) for r in rows}))
    (tools / 'configs').mkdir(parents=True)
    (tools / 'configs/rwprep_server.yaml').write_text('gpu_allowlist: []\n')
    out = tmp_path / 'compiled'
    argv = ['--phase', 'B', '--plan', str(plan), '--out', str(out), '--processed', str(tmp_path / 'processed'),
            '--host', 'remote', '--image', 'sha256:' + 'a' * 64, '--code', str(ROOT), '--weights', str(tmp_path),
            '--tools', str(tools), '--control-repo', str(tmp_path), '--rig-map', str(rmap)]
    assert shared.record_for(TASK) is None
    assert _script('realworld_prep_register_ao').compile_phase_b(argv) == 0
    jobs = json.loads((out / 'jobs.json').read_text())
    assert len(jobs) == 2
    for job in jobs:
        assert job['acceptance_override']['record_sha256'] == A.record_for(TASK)[1]
        assert job['steps'][0]['argv'][:5] == ['-m', 's4d.realworld.calib.pnpv2_acceptance_override',
                                             'rwprep', '--', 'episode']
    assert shared.record_for(TASK) is None and not shared._ORIGINAL


def test_native_episode_worker_installs_the_isolated_reader(world, monkeypatch):
    from s4d.rwlearn import pnpv2_native_depth_override as native
    from s4d.realworld.calib import acceptance_override as shared
    doc, path = published(world)
    cams = [c.split('/')[0] for c in CAMS]
    original = native._nd.load_rig_session
    monkeypatch.setenv(native.ENV, '1')
    monkeypatch.delenv(native._mapped.ENV, raising=False)
    def episode(job):
        return native._nd.load_rig_session(path, cams, {c: np.eye(3) for c in cams})
    monkeypatch.setattr(native, '_ORIGINAL_BUILD', episode)
    result = native._build_episode_override(None)
    assert result['native_depth_correction'][cams[1]]['evidence']['effective_rig_status'] == A.STATUS
    assert native._nd.load_rig_session is original and shared.record_for(TASK) is None
