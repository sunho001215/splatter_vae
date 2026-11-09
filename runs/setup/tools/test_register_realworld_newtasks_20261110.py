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
IMAGE_COUNTS = {'image-pretrain': 2, 'image-dataset': 3, 'image-fql': 9}
GEOMETRY_COUNTS = {'geometry-pretrain': 3, 'geometry-dataset': 4, 'geometry-fql': 12}


def measured_scenes(monkeypatch, task):
    """A release whose scene geometry for ``task`` was measured (stand-in values: the existing tasks')."""
    monkeypatch.setitem(configs.SCENE_PARAMETERS, task, dict(configs.SCENE_PARAMETERS['drawer-open']))


def staged_inputs(directory, task='covering', *, geometry=True):
    """100 compact episodes with a live gate, bound provenance, exact seed-0 split and successful outcomes."""
    entries = {}
    for i in range(100):
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
                'counts': {task: {'complete': True, 'target': 100}}}
    raw = json.dumps(manifest).encode()
    report = gate.evaluate(manifest, [task])
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
        r['preprocessing']['stages']['train_data']['outputs'] for r in entries.values()]}),
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
    result = R.plan_task(task, report, PIN, remote)
    assert result['counts'] == {**IMAGE_COUNTS, **GEOMETRY_COUNTS}
    rows = result['jobs']
    assert len(rows) == 33 and not result['plan_refused']
    templates = {r['id']: r for r in launch.plan(report, [task], PIN['commit'])['rows']}
    by_id = {r['id']: r for r in rows}
    assert len(by_id) == len(rows)
    for row in rows:
        assert row['id'] == R.PREFIX + row['plan_row_id'][3:]
        template = templates[row['plan_row_id']]
        assert all(row[k] == template[k] for k in ('priority', 'mem_gb', 'ram_gb', 'gpu', 'max_restarts', 'role'))
        assert row['planned_commit'] == PIN['commit'] and row['planned_image'] == PIN['image']
        assert row['pin_host'] == 'remote' and row['realworld_target'] == task
        assert all(d in by_id for d in row['deps'])
        args = row['args']
        if R.kind(row) == 'pretrain':
            assert args[args.index('--output-root') + 1] == 'runs'
            assert all(f'{key}={remote}/{name}' in args for key, name in R.OVERRIDE_KEYS.items())
        elif R.kind(row) == 'dataset':
            assert f"runs/{row['id']}/dataset.h5" in args and f'{remote}/outcomes.json' in args
            if '--export' in args:
                assert f"runs/{row['deps'][0]}/encoder.pt" in args
        else:
            assert 'agent.alpha=10.0' in args and row['fql_seed'] in (0, 1, 2)
            assert f"data.dataset=runs/{row['deps'][0]}/dataset.h5" in args
    hashes = R.verify_inputs(task, manifest, report, tmp_path)
    assert set(hashes) == set(R.INPUT_NAMES) and all(len(v) == 64 for v in hashes.values())


def test_proposal_respects_each_consumer_gate_and_old_tasks_cannot_be_registered(tmp_path, monkeypatch):
    measured_scenes(monkeypatch, 'covering')
    _, report = staged_inputs(tmp_path, geometry=False)  # calibration hold closed
    image = R.plan_task('covering', report, PIN, '/remote/covering', family='image')
    assert len(image['jobs']) == 14 and not any(r['calibration_dependent'] for r in image['jobs'])
    assert image['counts'] == IMAGE_COUNTS
    for family in ('geometry', 'all'):
        with pytest.raises(R.Refused, match='incomplete consumer family'):
            R.plan_task('covering', report, PIN, '/remote/covering', family=family)
    with pytest.raises(R.Refused, match='one new'):
        R.plan_task('drawer-open', report, PIN, '/remote/drawer-open')
    report['tasks']['covering']['passed'] = False
    with pytest.raises(R.Refused, match='100-episode'):
        R.plan_task('covering', report, PIN, '/remote/covering')


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
            R.plan_task(task, report, PIN, f'/remote/{task}', family=family)
    assert R.plan_task(task, report, PIN, f'/remote/{task}', family='image')['counts'] == IMAGE_COUNTS


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
        with pytest.raises(R.Refused, match='released config has unmeasured scene parameters'):
            R.prepare(tmp_path, 'covering', manifest_path, directory, '/remote/covering')
    roles = tmp_path / 'configs/hosts/roles.yaml'
    roles.parent.mkdir(parents=True)
    roles.write_text((configs.REPO / 'configs/hosts/roles.yaml').read_text())
    monkeypatch.setattr(R, 'release_evidence', lambda *args: {'source_sha256': {}})
    prepared = R.prepare(tmp_path, 'covering', manifest_path, directory, '/remote/covering', family='image')
    assert len(prepared['jobs']) == 14 and prepared['status'] == 'dry-run' and not prepared['registered']
    assert prepared['release'] == PIN and not (tmp_path / 'experiments').exists()
    assert set(prepared['learning_inputs']['sha256']) == set(R.INPUT_NAMES)
    assert {r['planned_commit'] for r in prepared['jobs']} == {PIN['commit']}
    assert all(r['id'].startswith(R.PREFIX + ('fql' if R.kind(r) != 'pretrain' else '')) for r in prepared['jobs'])


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
