"""Pure checks of the proposed registrar register_realworld_newtasks_20261110.py: synthetic manifest / inputs only,
no scheduler import, GPU, network, queue or registry. Lives next to the registrar (n01_out now; runs/setup/tools once
installed) and runs against the checkout on sys.path:

    cd <checkout> && CUDA_VISIBLE_DEVICES=<one GPU or ""> .venv/bin/python -m pytest -q -p no:cacheprovider \
        --basetemp=outputs/pytest-rwnt-registrar <dir>/test_register_realworld_newtasks_20261110.py
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

HERE = Path(__file__).resolve()
REPO = next((p for p in (*HERE.parents, Path.cwd()) if (p / 's4d/rwlearn/tasks.py').is_file()), Path.cwd())
sys.path.insert(0, str(REPO))

from s4d.rwlearn import configs, gate, launch, outcomes  # noqa: E402
from s4d.rwlearn.store import split_episodes  # noqa: E402

PROPOSAL = HERE.with_name('register_realworld_newtasks_20261110.py')
SPEC = importlib.util.spec_from_file_location('newtask_registrar', PROPOSAL)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
PIN = {'commit': '1' * 40, 'image': 'sha256:' + '2' * 64}
IMAGE_COUNTS = {'image-pretrain': 2, 'image-dataset': 3, 'image-fql': 18}  # + 3 noaug seeds + 6 a100 (FQLNQ)
GEOMETRY_COUNTS = {'geometry-stage': 1, 'geometry-pretrain': 3, 'geometry-dataset': 4, 'geometry-fql': 24}  # + a100
RIG = {'remote': '/home/compu/kaist/sunho/realworld/configs/calrel-6c3b9ec/rig/rig-loma-t.json', 'sha256': 'a' * 64}


@pytest.fixture(autouse=True)
def no_amendments(monkeypatch):
    """Synthetic manifests: the default 100-episode rule unless a test installs an amendment (the versioned covering
    record would otherwise apply to the synthetic covering episodes)."""
    monkeypatch.setattr(gate, 'load_amendments', lambda *a, **k: {})


def amendment(task, excluded):
    rec = {'format': gate.AMENDMENT_FORMAT, 'task': task, 'version': 1, 'target': 100 - len(excluded),
           'base_target': 100, 'decision': 'permit the valid episodes', 'decided_by': 'user',
           'decided_at': '2026-11-10T13:00:00+09:00', 'document': {'path': 'AMEND.md', 'sha256': 'd' * 64},
           'episodes': {f'{task}/episode_{i:06d}.h5': R.sha(f'{task}-{i}'.encode()) for i in range(100)
                        if i not in excluded},
           'excluded': {f'{task}/episode_{i:06d}.h5': {'reason': 'timestamps'} for i in excluded}}
    return {task: {'record': rec, 'path': f'/cfg/{task}.json', 'sha256': 'e' * 64}}


def measured_scenes(monkeypatch, task):
    """A release whose scene geometry for ``task`` was measured (stand-in values: the existing tasks')."""
    monkeypatch.setitem(configs.SCENE_PARAMETERS, task, dict(configs.SCENE_PARAMETERS['drawer-open']))


def staged_inputs(directory, task='covering', *, geometry=True, excluded=()):
    """100 compact episodes (or 100 minus ``excluded``, invalid recordings, with the task's gate amendment installed)
    with a live gate, bound provenance, exact seed-0 split and successful outcomes."""
    entries = {}
    for i in range(100):
        if i in excluded:
            entries[f'{task}/episode_{i:06d}.h5'] = {'task': task, 'episode_index': None, 'sha256': None,
                                                     'validation': {'status': 'invalid'}, 'transfer': {'status': 'pending'}}
            continue
        digest = R.sha(f'{task}-{i}'.encode())
        stages = {s: {'status': 'done', 'input_sha256': digest} for s in gate.REQUIRED_STAGES}
        stages['calibrate']['metrics'] = {'build_id': 'cal-1'}
        stages['train_data'].update(output_format='s4d.realworld.train/1.1.0',
                                    outputs=f'/remote/processed/{task}/{i}/train.h5')
        stages['qa'].update(output_format='rwprep-qa/3', metrics={'families': {
            f: {'passed': True} for f in gate.QA_FAMILIES}})
        if geometry:
            stages[gate.CALIBRATION_STAGE] = {'status': 'done', 'input_sha256': digest,
                'output_format': 'rw-calibration-validation/1', 'metrics': {'decision': 'accepted',
                'calibrate_build_id': 'cal-1', 'cameras': ['camera-0', 'camera-1', 'camera-2']}}
        entries[f'{task}/episode_{i:06d}.h5'] = {'task': task, 'episode_index': i, 'sha256': digest,
            'validation': {'status': 'valid'}, 'transfer': {'status': 'verified', 'remote_sha256': digest},
            'preprocessing': {'schema': gate.STAGE_SCHEMA, 'schema_version': 1, 'status': 'done', 'stages': stages}}
    manifest = {'schema': gate.SCHEMA, 'schema_version': gate.SCHEMA_VERSION, 'episodes': entries,
                'counts': {task: {'complete': not excluded, 'target': 100}}}
    raw = json.dumps(manifest).encode()
    report = gate.evaluate(manifest, [task], amendments=amendment(task, excluded) if excluded else {})
    report['manifest'] = {'path': '/synthetic/manifest.json', 'sha256': R.sha(raw)}
    identities = {f'{task}/compact_{i:06d}': rec
                  for i, rec in enumerate(report['tasks'][task]['episode_identities'].values())}
    eps = list(identities)
    bound = gate.bind_store_episodes(report, {task: eps}, identities)
    store = SimpleNamespace(episodes=lambda: eps, task=lambda ep: task,
                            raw_recording=lambda ep: identities[ep]['raw_recording'],
                            raw_sha256=lambda ep: identities[ep]['raw_sha256'])
    labels = outcomes.outcome_labels(eps)
    for rec in labels['episodes'].values():
        rec.update(ending_decision='no_reversal', ending_flags=[])
    labels['episodes'][eps[0]].update(ending_decision='needs_image_review', ending_flags=['multiple_grasps'])
    files = {'store.yaml': yaml.safe_dump({'kind': 'compact', 'paths': [
        r['preprocessing']['stages']['train_data']['outputs'] for r in entries.values() if 'preprocessing' in r]}),
        'gate_report.json': json.dumps(bound), 'split_seed0.json': json.dumps(split_episodes(store)),
        'outcomes.json': json.dumps(labels)}
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / name).write_text(text)
    return manifest, report


@pytest.mark.parametrize('task', ['covering', 'pick-and-place'])
def test_proposal_has_exact_families_resources_seeds_and_artifact_dependencies(task, tmp_path, monkeypatch):
    measured_scenes(monkeypatch, task)
    manifest, report = staged_inputs(tmp_path, task)
    remote = f'/remote/learning_inputs/newtasks/{task}'
    result = R.plan_task(task, report, PIN, remote, rig=RIG)
    assert result['counts'] == {**IMAGE_COUNTS, **GEOMETRY_COUNTS}
    rows = result['jobs']
    assert len(rows) == 33 + 3 + 1 + 18 and not result['plan_refused']  # + noaug + native stage + 18 a100 (FQLNQ)
    templates = {r['id']: r for r in launch.plan(report, [task], PIN['commit'])['rows']}
    by_id = {r['id']: r for r in rows}
    assert len(by_id) == len(rows)
    directive_r_checks(task, rows, by_id, remote)
    for row in rows:
        if R.kind(row) == 'stage' or row.get('fql_variant') == 'noaug':
            continue  # checked by directive_r_checks
        setting = row.get('fql_setting')
        assert row['id'] == (R.SETTINGS[setting][0] if setting else R.PREFIX) + row['plan_row_id'][3:]
        assert (setting is not None) == (R.kind(row) == 'fql' and row['fql_condition'] != 'e2e_cnn')
        template = templates[row['plan_row_id']]
        assert all(row[k] == template[k] for k in ('mem_gb', 'ram_gb', 'gpu', 'max_restarts', 'role'))
        assert row['priority'] == (R.A100_PRIORITY if setting == 'a100' else template['priority'])
        assert row['planned_commit'] == PIN['commit'] and row['planned_image'] == PIN['image']
        assert row['pin_host'] == 'remote' and row['realworld_target'] == task
        assert all(d in by_id for d in row['deps'])
        args = row['args']
        if R.kind(row) == 'pretrain':
            assert args[args.index('--output-root') + 1] == 'runs'
            assert all(f'{key}={remote}/{name}' in args for key, name in R.OVERRIDE_KEYS.items()
                       if not (key == 'dataset.hdf5_path' and row.get('depth_source') == R.NATIVE_DEPTH))
        elif R.kind(row) == 'dataset':
            assert f"runs/{row['id']}/dataset.h5" in args and f'{remote}/outcomes.json' in args
            if '--export' in args:
                assert f"runs/{row['deps'][0]}/encoder.pt" in args
        else:
            alpha = 'agent.alpha=100.0' if setting == 'a100' else 'agent.alpha=10.0'
            assert alpha in args and row['fql_seed'] in (0, 1, 2)
            flags = [a for a in args if a.startswith('agent.normalize_q_loss')]
            assert flags == ([] if setting is None else
                             [f"agent.normalize_q_loss={'true' if setting == 'a10nq' else 'false'}"])
            assert f"data.dataset=runs/{row['deps'][0]}/dataset.h5" in args
    hashes = R.verify_inputs(task, manifest, report, tmp_path)
    assert set(hashes) == set(R.INPUT_NAMES) and all(len(v) == 64 for v in hashes.values())


def directive_r_checks(task, rows, by_id, remote):
    """Directive R additions: e2e_cnn_noaug = its e2e_cnn row + pixel.p_aug=0.0 only; the native 3D-MVP chain; the
    frozen-encoder FQL hold (e2e_cnn / noaug excluded)."""
    noaug = [r for r in rows if r.get('fql_variant') == 'noaug']
    assert len(noaug) == 3
    for r in noaug:
        orig = by_id[r['derived_from']]
        assert r['args'] == [r['id'] if a == orig['id'] else a for a in orig['args']] + ['pixel.p_aug=0.0']
        same = {k for k in orig if k not in ('id', 'args', 'plan_row_id')}
        assert all(r[k] == orig[k] for k in same)
        assert r['plan_row_id'] == orig['plan_row_id'].replace('e2e_cnn', 'e2e_cnn_noaug')
    stage = by_id[f'{R.PREFIX}nativedepth-mvp3d-{task}']
    store = f"runs/{stage['id']}/store-native.yaml"
    assert stage['args'][:5] == ['build', '--store', f'{remote}/store.yaml', '--rig', RIG['remote']]
    assert stage['rig_document_sha256'] == RIG['sha256'] and R.kind(stage) == 'stage' and not stage['deps']
    mvp = [r for r in rows if r.get('baseline') == 'mvp3d' or str(r.get('fql_condition', '')).startswith('mvp3d')]
    assert len(mvp) == 3 + 12 and all(r['depth_source'] == R.NATIVE_DEPTH for r in mvp)  # 6 nq + 6 a100 FQL
    pre = next(r for r in mvp if R.kind(r) == 'pretrain')
    assert f'dataset.hdf5_path={store}' in pre['args'] and stage['id'] in pre['deps']
    assert all(r['args'][r['args'].index('--store') + 1] == store for r in mvp if R.kind(r) == 'dataset')
    assert all(r['args'][-2] == 'encoder.feature_norm=train_standardize' and 'feature check' in r['launch_gate']
               and r['args'][-1].startswith('agent.normalize_q_loss=') for r in mvp if R.kind(r) == 'fql')
    nq, a100, feature = R.holds(rows, task)
    fql = [r for r in rows if R.kind(r) == 'fql']
    assert set(nq['rows']) == {r['id'] for r in fql if r.get('fql_setting') == 'a10nq'} and len(nq['rows']) == 18
    assert set(a100['rows']) == {r['id'] for r in fql if r.get('fql_setting') == 'a100'} and len(a100['rows']) == 18
    held = set(nq['rows']) | set(a100['rows'])
    assert held == {r['id'] for r in fql if r['fql_condition'] != 'e2e_cnn'}
    assert all(r['id'] not in held and r['id'].startswith(R.PREFIX) and 'agent.alpha=10.0' in r['args']
               and not any(a.startswith('agent.normalize_q_loss') for a in r['args'])
               for r in fql if r['fql_condition'] == 'e2e_cnn')  # e2e_cnn + noaug unchanged, may run
    assert set(feature['rows']) == {r['id'] for r in fql if r.get('depth_source') == R.NATIVE_DEPTH} \
        and len(feature['rows']) == 12


def test_proposal_respects_each_consumer_gate_and_old_tasks_cannot_be_registered(tmp_path, monkeypatch):
    measured_scenes(monkeypatch, 'covering')
    _, report = staged_inputs(tmp_path, geometry=False)  # calibration hold closed
    image = R.plan_task('covering', report, PIN, '/remote/covering', family='image', rig=RIG)
    assert len(image['jobs']) == 23 and not any(r['calibration_dependent'] for r in image['jobs'])
    assert len(image['holds']) == 2 and set(image['holds'][0]['rows']) | set(image['holds'][1]['rows']) == {
        r['id'] for r in image['jobs'] if R.kind(r) == 'fql' and r['fql_condition'] in ('reviwo', 'dynamo')}
    assert image['counts'] == IMAGE_COUNTS
    for family in ('geometry', 'all'):
        with pytest.raises(R.Refused, match='incomplete consumer family'):
            R.plan_task('covering', report, PIN, '/remote/covering', family=family, rig=RIG)
    with pytest.raises(R.Refused, match='one new'):
        R.plan_task('drawer-open', report, PIN, '/remote/drawer-open', rig=RIG)
    report['tasks']['covering']['passed'] = False
    with pytest.raises(R.Refused, match='100-episode'):
        R.plan_task('covering', report, PIN, '/remote/covering', rig=RIG)


# synthetic manifests number episodes 0..99 (the excluded indices stand in for the real ones)
@pytest.mark.parametrize('task,excluded,n_train', [('covering', (57, 61), 88), ('pick-and-place', (14, 80, 99), 87)])
def test_an_amended_task_registers_on_exactly_its_amendment_episodes(task, excluded, n_train, tmp_path, monkeypatch):
    """The 2026-11-10 amendments (covering 98, pick-and-place 97): the same rows, inputs checked against exactly the
    amendment's episodes and their seed-0 split (10 validation); a report whose target does not match its amendment is
    refused."""
    measured_scenes(monkeypatch, task)
    manifest, report = staged_inputs(tmp_path, task, excluded=excluded)
    n = 100 - len(excluded)
    assert report['tasks'][task]['passed'] and report['tasks'][task]['target'] == n
    assert R.plan_task(task, report, PIN, f'/remote/{task}', rig=RIG)['counts'] == {**IMAGE_COUNTS, **GEOMETRY_COUNTS}
    assert set(R.verify_inputs(task, manifest, report, tmp_path)) == set(R.INPUT_NAMES)
    split = json.loads((tmp_path / 'split_seed0.json').read_text())
    assert len(split['validation']) == 10 and len(split['train']) == n_train
    report['tasks'][task]['target'] = 100
    with pytest.raises(R.Refused, match='gate target'):
        R.plan_task(task, report, PIN, f'/remote/{task}', rig=RIG)
    # the same episodes without the amendment never pass the gate
    plain = gate.evaluate(manifest, [task], amendments={})
    assert not plain['tasks'][task]['passed']


@pytest.mark.parametrize('task', ['covering', 'pick-and-place'])
def test_unmeasured_scene_never_registers_a_partial_geometry_family(task, tmp_path):
    """Before the scene geometry is measured the planner refuses SNeRL / 3D-MVP (and their FQL conditions); SinCro
    alone would pass its gates, but a partial family would collide with the rest later, so it is refused too."""
    assert configs.pending_scene_parameters('snerl', task) and configs.pending_scene_parameters('mvp3d', task)
    _, report = staged_inputs(tmp_path, task)
    plan = launch.plan(report, [task], PIN['commit'])
    assert {r.get('baseline') for r in plan['rows'] if r['role'] == 'realworld_pretrain'} == {'sincro', 'reviwo',
                                                                                            'dynamo'}
    assert all('scene parameters not measured' in ' '.join(v) for v in plan['refused'].values())
    for family in ('geometry', 'all'):
        with pytest.raises(R.Refused, match='incomplete consumer family'):
            R.plan_task(task, report, PIN, f'/remote/{task}', family=family, rig=RIG)
    assert R.plan_task(task, report, PIN, f'/remote/{task}', family='image', rig=RIG)['counts'] == IMAGE_COUNTS


@pytest.mark.parametrize('change', ['output', 'split', 'split_seed', 'outcome', 'ending', 'calibration', 'identity'])
def test_staged_inputs_cannot_silently_differ_from_live_evidence(tmp_path, change):
    manifest, report = staged_inputs(tmp_path)
    if change == 'output':
        path = tmp_path / 'store.yaml'
        doc = yaml.safe_load(path.read_text())
        doc['paths'][0] = '/wrong/train.h5'
        path.write_text(yaml.safe_dump(doc))
    else:
        name = ('split_seed0.json' if change in ('split', 'split_seed') else
                'outcomes.json' if change in ('outcome', 'ending') else 'gate_report.json')
        path = tmp_path / name
        doc = json.loads(path.read_text())
        if change == 'split':
            doc['validation'][0] = doc['train'][0]
        elif change == 'split_seed':
            doc['seed'] = 1
        elif change == 'outcome':
            next(iter(doc['episodes'].values()))['outcome'] = 'failure'
        elif change == 'ending':
            next(iter(doc['episodes'].values()))['ending_decision'] = None
        elif change == 'calibration':
            next(iter(doc['calibration_hold']['covering']['episodes'].values()))['calibrate_build_id'] = 'stale'
        else:
            next(iter(doc['tasks']['covering']['episode_identities'].values()))['raw_sha256'] = '0' * 64
        path.write_text(json.dumps(doc))
    with pytest.raises((R.Refused, RuntimeError)):
        R.verify_inputs('covering', manifest, report, tmp_path)


def test_staged_inputs_survive_manifest_writes_for_other_tasks(tmp_path):
    """Ingestion keeps rewriting the live manifest (other tasks arriving); only this task's sections must agree."""
    manifest, report = staged_inputs(tmp_path)
    path = tmp_path / 'gate_report.json'
    doc = json.loads(path.read_text())
    doc['manifest']['sha256'] = '0' * 64
    path.write_text(json.dumps(doc))
    assert set(R.verify_inputs('covering', manifest, report, tmp_path)) == set(R.INPUT_NAMES)


def test_role_release_is_read_at_registration_time(tmp_path):
    path = tmp_path / 'runs/setup/role-release.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(PIN))
    assert R.release_pin(tmp_path) == PIN
    changed = {**PIN, 'commit': '3' * 40}
    path.write_text(json.dumps(changed))
    assert R.release_pin(tmp_path) == changed
    path.write_text(json.dumps({'commit': 'main', 'image': PIN['image']}))
    with pytest.raises(R.Refused, match='full commit'):
        R.release_pin(tmp_path)


def test_prepare_refuses_unmeasured_scene_but_can_prepare_image_rows(tmp_path, monkeypatch):
    directory = tmp_path / 'inputs'
    manifest, _ = staged_inputs(directory)
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    setup = tmp_path / 'runs/setup'
    setup.mkdir(parents=True)
    (setup / 'role-release.json').write_text(json.dumps(PIN))
    for b in ('snerl', 'mvp3d'):
        path = tmp_path / f'configs/realworld/baselines/{b}/covering.yaml'
        path.parent.mkdir(parents=True)
        path.write_text(configs.render(b, 'covering'))  # the generated configs: scene parameters null
    with pytest.raises(R.Refused, match='incomplete consumer family'):
        R.prepare(tmp_path, 'covering', manifest_path, directory, '/remote/covering')
    with monkeypatch.context() as m:  # planner says measured, but the released configs still hold nulls
        measured_scenes(m, 'covering')
        with pytest.raises(R.Refused, match='native 3D-MVP chain needs'):
            R.prepare(tmp_path, 'covering', manifest_path, directory, '/remote/covering')
        with pytest.raises(R.Refused, match='released config has unmeasured scene parameters'):
            R.prepare(tmp_path, 'covering', manifest_path, directory, '/remote/covering', rig=RIG)
    roles = tmp_path / 'configs/hosts/roles.yaml'
    roles.parent.mkdir(parents=True)
    roles.write_text((configs.REPO / 'configs/hosts/roles.yaml').read_text())
    monkeypatch.setattr(R, 'release_evidence', lambda *args: {'source_sha256': {}})
    prepared = R.prepare(tmp_path, 'covering', manifest_path, directory, '/remote/covering', family='image')
    assert len(prepared['jobs']) == 23 and prepared['status'] == 'dry-run' and not prepared['registered']
    assert prepared['holds'][0]['rows'] and prepared['holds'][0]['active'] is True
    assert prepared['release'] == PIN and not (tmp_path / 'experiments').exists()
    assert set(prepared['learning_inputs']['sha256']) == set(R.INPUT_NAMES)
    assert {r['planned_commit'] for r in prepared['jobs']} == {PIN['commit']}
    assert all(r['id'].startswith((R.SETTINGS[r['fql_setting']][0] if r.get('fql_setting') else R.PREFIX)
                                  + ('fql' if R.kind(r) != 'pretrain' else '')) for r in prepared['jobs'])


@pytest.mark.parametrize('appended', [False, True])
def test_recovery_only_aborts_staging_or_finishes_the_saved_proof(tmp_path, monkeypatch, appended):
    setup, experiments = tmp_path / 'runs/setup', tmp_path / 'experiments'
    setup.mkdir(parents=True)
    experiments.mkdir()
    before = b'jobs: []\n'
    rows = [{'id': R.PREFIX + 'test'}]
    after = R.queue_candidate(before, rows)
    record = {'jobs': rows, 'jobs_sha256': R.json_sha(rows), 'queue_sha256_before': R.sha(before),
              'queue_sha256_after': R.sha(after), 'proof_file': 'proof.json'}
    pending = setup / f'{R.STEM}.pending.json'
    pending.write_text(json.dumps(record))
    queue = experiments / 'queue.yaml'
    queue.write_bytes(after if appended else before)
    monkeypatch.setattr(R, 'check_unchanged', lambda *args: None)
    result = R.recover(tmp_path)
    assert result['registered'] == appended and not pending.exists()
    assert queue.read_bytes() == (after if appended else before)
    assert (setup / 'proof.json').exists() == appended


@pytest.mark.parametrize('jobs', ['[]', '\n- id: old\n  deps: []'])
def test_queue_candidate_preserves_existing_rows_and_comments(jobs):
    before = f'# comment\nhosts: {{local: {{}}, remote: {{}}}}\njobs: {jobs}\n# tail\n'.encode()
    rows = [{'id': 'new', 'deps': []}]
    saved = copy.deepcopy(rows)
    candidate = R.queue_candidate(before, rows)
    parsed = yaml.safe_load(before)
    assert yaml.safe_load(candidate) == {**parsed, 'jobs': parsed['jobs'] + rows}
    assert b'# comment' in candidate and b'# tail' in candidate and rows == saved


def test_rig_document_must_be_this_tasks_validated_release(tmp_path):
    """The native stage's rig document: a release of THIS task, every camera validated, with sensor_depth scales."""
    from types import SimpleNamespace

    import numpy as np

    from s4d.realworld.calib.rig import FIXEDNESS_RULE_G5AB, FORMAT, publish_rig

    cams = ['c0/color', 'c1/color']

    def doc(task, status='validated', scales=('foundation_stereo_depth', 'sensor_depth')):
        return {'format': FORMAT, 'name': f'rig-{task}', 'task': task, 'gate_version': 'g/1',
                'fixedness_rule': FIXEDNESS_RULE_G5AB, 'fixedness_exception': {}, 'episodes': [],
                'cameras': {c: {'status': status, 'held_by': [] if status == 'validated' else ['G1'],
                                'depth_scale': dict.fromkeys(scales, 1.0), 'T_cam_base': np.eye(4).tolist()}
                            for c in cams}}

    arrays = {f"{c.replace('/', '__')}__{k}": np.zeros((2, 2)) for c in cams for k in ('grey', 'dynamic')}

    def args(name, d, task='covering'):
        path = tmp_path / name
        publish_rig(path, d, arrays)
        return SimpleNamespace(rig_document=path, rig_remote=f'/remote/rig/{name}', task=task)

    good = R.rig_arg(args('a.json', doc('covering')))
    assert good['task'] == 'covering' and len(good['sha256']) == 64
    for name, d in (('b.json', doc('drawer-open')), ('c.json', doc('covering', status='held')),
                    ('d.json', doc('covering', scales=('foundation_stereo_depth',)))):
        with pytest.raises(R.Refused):
            R.rig_arg(args(name, d))
    with pytest.raises(R.Refused, match='same document'):
        R.rig_arg(SimpleNamespace(rig_document=tmp_path / 'a.json', rig_remote='/remote/rig/x.json', task='covering'))


def test_setting_rows_differ_from_the_planned_alpha10_row_only_by_the_setting():
    """FQLNQ: a frozen-encoder FQL plan row -> its a10nq row + a100 comparison (lowest priority); e2e rows untouched."""
    base = {'id': f'{R.PREFIX}fql-sincro-covering-s1', 'script': 'scripts/train_realworld_fql.py', 'priority': 2,
            'role': 'realworld_fql',
            'fql_condition': 'sincro', 'args': ['--name', f'{R.PREFIX}fql-sincro-covering-s1', '--set', 'condition=sincro',
                                                'train.seed=1', 'agent.alpha=10.0', 'train.offline_steps=1000000'],
            'fql_budget': {'alpha': 10.0, 'offline_steps': 1000000}, 'deps': ['d'], 'mem_gb': 6}
    e2e = {**base, 'id': f'{R.PREFIX}fql-e2e_cnn-covering-s1', 'fql_condition': 'e2e_cnn'}
    nq, a100, kept = R.setting_rows([base, e2e])[0], R.setting_rows([base, e2e])[1], R.setting_rows([base, e2e])[2]
    assert kept == e2e
    assert nq['id'] == 'rwnq-20261111-fql-sincro-covering-s1' and a100['id'] == 'rwa100-20261111-fql-sincro-covering-s1'
    assert nq['args'] == ['--name', nq['id'], '--set', 'condition=sincro', 'train.seed=1', 'agent.alpha=10.0',
                          'train.offline_steps=1000000', 'agent.normalize_q_loss=true']
    assert a100['args'] == ['--name', a100['id'], '--set', 'condition=sincro', 'train.seed=1', 'agent.alpha=100.0',
                            'train.offline_steps=1000000', 'agent.normalize_q_loss=false']
    assert (nq['priority'], a100['priority']) == (2, R.A100_PRIORITY) and R.A100_PRIORITY > 2
    assert nq['fql_budget'] == {'alpha': 10.0, 'offline_steps': 1000000, 'normalize_q_loss': True}
    assert a100['fql_budget'] == {'alpha': 100.0, 'offline_steps': 1000000, 'normalize_q_loss': False}
    same = set(base) - {'id', 'args', 'priority', 'fql_budget'}
    assert all(nq[k] == base[k] == a100[k] for k in same) and {nq['fql_setting'], a100['fql_setting']} == {'a10nq', 'a100'}
    with pytest.raises(R.Refused, match='unexpected FQL row'):
        R.setting_rows([{**base, 'args': base['args'] + ['agent.normalize_q_loss=true']}])
