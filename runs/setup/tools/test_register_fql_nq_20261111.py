"""Synthetic FQLNQ registrar checks: no live scheduler, remote, pause or operational writes (tmp repos only)."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SPEC = importlib.util.spec_from_file_location('fqlnq_registrar', Path(__file__).with_name('register_fql_nq_20261111.py'))
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
SOURCE_TREE = Path(__file__).resolve().parents[3]
COMMIT, IMAGE = '8' * 40, 'sha256:' + '5' * 64
EXPECTED_SOURCES = {  # the brief's table, written out independently of R.SOURCE_PREFIX
    'reviwo': {'drawer-open': 'rwng-20261107b', 'mug-hanging': 'rwcal-20261109'},
    'sincro': {'drawer-open': 'rwcal-20261109', 'mug-hanging': 'rwcal-20261109'},
    'dynamo': {'drawer-open': 'rwng-20261107b', 'mug-hanging': 'rwng-20261107b'},
    'mvp3d_single': {'drawer-open': 'rwcal-20261110', 'mug-hanging': 'rwcal-20261110'},
    'mvp3d_multiview': {'drawer-open': 'rwcal-20261110', 'mug-hanging': 'rwcal-20261110'}}
PAUSE_NAMES = {'rwnq-20261111-fql-': 'fqlnq-await-placement-20261111', 'rwa100-20261111-fql-': 'fqla100-after-nq-20261111'}


def dep_of(prefix, condition, task):
    return f"{prefix}-fqldata-{condition}-{task}{'-native' if prefix == 'rwcal-20261110' else ''}"


def source_row(prefix, condition, task, seed, alpha='10.0'):
    job, dep = f'{prefix}-fql-{condition}-{task}-s{seed}', dep_of(prefix, condition, task)
    args = ['--config', 'configs/realworld/fql/base.yaml', '--name', job, '--output-root', 'runs', '--set',
            f'condition={condition}', f'data.dataset=runs/{dep}/dataset.h5',
            f'data.split_manifest=/home/compu/kaist/sunho/realworld/learning_inputs/{task}/split_seed0.json',
            f'data.gate_report=/home/compu/kaist/sunho/realworld/learning_inputs/{task}/gate_report.json',
            f'train.seed={seed}', f'agent.alpha={alpha}', 'agent.batch_size=256', 'train.offline_steps=1000000',
            'train.log_interval=5000', 'train.eval_interval=100000']
    row = {'id': job, 'script': 'scripts/train_realworld_fql.py', 'args': args, 'role': 'realworld_fql',
           'host': 'remote', 'gpu': 'any', 'mem_gb': 6, 'ram_gb': 12, 'priority': 2, 'deps': [dep], 'max_restarts': 2,
           'oom_score_adj': 0, 'realworld_target': task, 'fql_condition': condition, 'fql_seed': seed,
           'fql_budget': {'offline_steps': 1000000, 'batch_size': 256, 'alpha': float(alpha), 'log_interval': 5000,
                          'eval_interval': 100000},
           'realworld_consumer': f'fql_{condition}', 'calibration_dependent': False, 'gate_level': 'image',
           'realworld_gate_sha256': 'a' * 64, 'planned_commit': 'e' * 40, 'remote_host': '192.168.10.22',
           'pin_host': 'remote', 'planned_image': 'sha256:' + 'f' * 64,
           'evidence_manifest': 'runs/remote/acceptance/eeeeeee-cc/manifest.json', 'disk_gb': 10,
           'realworld_manifest_sha256': 'b' * 64, 'registration_tool': 'runs/setup/tools/old.py',
           'plan_row_id': f'rw-fql-{condition}-{task}-s{seed}', 'required_results': ['policy.pt', 'metrics.jsonl',
                                                                                     'completion.json']}
    if prefix == 'rwcal-20261110':
        args.append('encoder.feature_norm=train_standardize')
        row.update(derived_from=f'rwcal-20261109-fql-{condition}-{task}-s{seed}',
                   supersedes=f'rwcal-20261109-fql-{condition}-{task}-s{seed}', depth_source='native_realsense',
                   launch_gate='pre-launch feature check must pass', calibration_dependent=True)
    return row


def queue_rows():
    rows = []
    for condition, tasks in EXPECTED_SOURCES.items():
        for task, prefix in tasks.items():
            rows.append({'id': dep_of(prefix, condition, task), 'script': 'scripts/realworld_fql_dataset.py',
                         'args': ['build'], 'host': 'remote'})
            rows += [source_row(prefix, condition, task, seed) for seed in range(3)]
            if condition.startswith('mvp3d'):  # quarantined predecessors with the same shape must never be sources
                rows += [source_row('rwcal-20261109', condition, task, seed) for seed in range(3)]
    # E2E rows must stay untouched (and are never sources)
    rows.append(source_row('rwng-20261107b', 'e2e_cnn', 'drawer-open', 0))
    return rows


def write_repo(root, rows):
    (root / 'experiments').mkdir(parents=True)
    (root / 'runs/setup').mkdir(parents=True)
    (root / R.QUEUE).write_text('# keep this comment\nhosts: {remote: {enabled: false}}\njobs:\n'
                                + yaml.safe_dump(rows, sort_keys=False) + 'limits: {}\n')
    (root / R.REGISTRY).write_text(json.dumps({'id': 'unrelated', 'event': 'launched'}) + '\n')
    (root / 'runs/setup/role-release.json').write_text(json.dumps({'commit': COMMIT, 'image': IMAGE}))
    manifest = root / f'runs/remote/acceptance/{COMMIT[:7]}-cc/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    for row in rows:
        if row['script'] == 'scripts/realworld_fql_dataset.py':
            (root / 'runs' / row['id']).mkdir(parents=True)
            (root / 'runs' / row['id'] / 'dataset.h5').write_bytes(b'h5')


def fake_pauses(active=True, skip=()):
    def active_pauses(job_id, *, fresh=False):
        assert fresh
        return [name for prefix, name in PAUSE_NAMES.items()
                if active and job_id.startswith(prefix) and job_id not in skip]
    return active_pauses


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'execution_allowed', lambda root: None)  # synthetic repos are not the main checkout
    root = tmp_path / 'repo'
    write_repo(root, queue_rows())
    return root


def prepare(root, **kw):
    return R.prepare(root, COMMIT, IMAGE, active_pauses=kw.pop('active_pauses', fake_pauses()), **kw)


def expected_row(source, setting, job):
    """Independent restatement of the allowed change set."""
    alpha, normalize, priority = {'a10nq': ('10.0', 'true', 2), 'a100': ('100.0', 'false', 3)}[setting]
    row = copy.deepcopy(source)
    row.pop('supersedes', None)
    row['args'] = [job if a == source['id'] else ('agent.alpha=' + alpha if a.startswith('agent.alpha=') else a)
                   for a in source['args']] + ['agent.normalize_q_loss=' + normalize]
    row['fql_budget'] = {**source['fql_budget'], 'alpha': float(alpha), 'normalize_q_loss': normalize == 'true'}
    return row


def test_sixty_rows_equal_their_source_field_by_field(repo):
    record = prepare(repo)
    queue = yaml.safe_load((repo / R.QUEUE).read_text())
    by_id = {r['id']: r for r in queue['jobs']}
    rows = record['jobs']
    assert len(rows) == 60 and len({r['id'] for r in rows}) == 60
    assert [r['fql_setting'] for r in rows] == ['a10nq'] * 30 + ['a100'] * 30  # nq first in the queue
    for row in rows:
        setting = row['fql_setting']
        prefix = {'a10nq': 'rwnq-20261111', 'a100': 'rwa100-20261111'}[setting]
        condition, task, seed = row['fql_condition'], row['realworld_target'], row['fql_seed']
        assert row['id'] == f'{prefix}-fql-{condition}-{task}-s{seed}'
        source = by_id[f'{EXPECTED_SOURCES[condition][task]}-fql-{condition}-{task}-s{seed}']
        assert row['derived_from'] == source['id'] and not source['id'].startswith('rwcal-20261109-fql-mvp3d_')
        want = expected_row(source, setting, row['id'])
        bookkeeping = {'id', 'derived_from', 'fql_setting', 'plan_row_id', 'purpose', 'priority', 'planned_commit',
                       'planned_image', 'evidence_manifest', 'registration_tool'}
        for key in sorted((set(row) | set(want)) - bookkeeping):  # field by field, type-sensitive
            assert key in row and key in want, (row['id'], key)
            assert json.dumps(row[key], sort_keys=True) == json.dumps(want[key], sort_keys=True), (row['id'], key)
        assert row['priority'] == (2 if setting == 'a10nq' else 3)
        assert row['args'].count('--name') == 1 and row['args'][row['args'].index('--name') + 1] == row['id']
        assert [a for a in row['args'] if a.startswith('agent.')] == (
            ['agent.alpha=10.0', 'agent.batch_size=256', 'agent.normalize_q_loss=true'] if setting == 'a10nq' else
            ['agent.alpha=100.0', 'agent.batch_size=256', 'agent.normalize_q_loss=false'])
        assert 'train.offline_steps=1000000' in row['args'] and f'train.seed={seed}' in row['args']
        assert 'supersedes' not in row and row['fql_setting'] in row['plan_row_id']
        assert (row['planned_commit'], row['planned_image'], row['evidence_manifest'], row['registration_tool']) == (
            COMMIT, IMAGE, f'runs/remote/acceptance/{COMMIT[:7]}-cc/manifest.json',
            'runs/setup/tools/register_fql_nq_20261111.py')
        assert '2026-11-11' in row['purpose']
    assert record['queue_diff']['counts'] == {'a10nq': 30, 'a100': 30}
    assert record['queue_diff']['conditions'] == {c: 12 for c in EXPECTED_SOURCES}
    assert record['pauses_active_now'] == {'a10nq': 30, 'a100': 30}


def test_dry_run_writes_nothing_and_candidate_only_appends(repo):
    before = {p: p.read_bytes() for p in repo.rglob('*') if p.is_file()}
    record = prepare(repo)
    assert {p: p.read_bytes() for p in repo.rglob('*') if p.is_file()} == before
    candidate = R.queue_candidate((repo / R.QUEUE).read_bytes(), record['jobs'])
    old, new = yaml.safe_load(before[repo / R.QUEUE]), yaml.safe_load(candidate)
    assert new['jobs'][:len(old['jobs'])] == old['jobs'] and new['jobs'][len(old['jobs']):] == record['jobs']
    assert {k: v for k, v in new.items() if k != 'jobs'} == {k: v for k, v in old.items() if k != 'jobs'}
    assert candidate.decode().startswith('# keep this comment\n')
    e2e = [r for r in new['jobs'] if r.get('fql_condition') == 'e2e_cnn']
    assert e2e == [r for r in old['jobs'] if r.get('fql_condition') == 'e2e_cnn'] and len(e2e) == 1


def test_execute_publishes_once_with_pauses(repo):
    record = prepare(repo)
    R.execute(repo, record, active_pauses=fake_pauses())
    proof = json.loads((repo / f'runs/setup/{R.STEM}.json').read_text())
    assert proof['status'] == 'published' and proof['registered'] is True
    assert R.sha((repo / R.QUEUE).read_bytes()) == record['queue_sha256_after']
    assert not (repo / f'runs/setup/{R.STEM}.pending.json').exists()
    with pytest.raises(R.Refused, match='pending transaction/proof exists'):
        prepare(repo)


@pytest.mark.parametrize('active, skip', [(False, ()), (True, ('rwa100-20261111-fql-dynamo-mug-hanging-s2',))])
def test_execute_refuses_without_the_pauses(repo, active, skip):
    record = prepare(repo)
    before = (repo / R.QUEUE).read_bytes()
    with pytest.raises(R.Refused, match='lack their active pause'):
        R.execute(repo, record, active_pauses=fake_pauses(active=active, skip=skip))
    assert (repo / R.QUEUE).read_bytes() == before
    assert not (repo / f'runs/setup/{R.STEM}.json').exists()


def test_execute_refuses_a_changed_queue(repo):
    record = prepare(repo)
    (repo / R.QUEUE).write_bytes((repo / R.QUEUE).read_bytes() + b'# edited\n')
    with pytest.raises(R.Refused, match='queue changed under locks'):
        R.execute(repo, record, active_pauses=fake_pauses())


def mutate(repo, job, change):
    doc = yaml.safe_load((repo / R.QUEUE).read_text())
    for row in doc['jobs']:
        if row['id'] == job:
            change(row)
    (repo / R.QUEUE).write_text(yaml.safe_dump(doc, sort_keys=False))


@pytest.mark.parametrize('change, message', [
    (lambda r: r['args'].__setitem__(r['args'].index('agent.alpha=10.0'), 'agent.alpha=100.0'), 'agent.alpha=10.0'),
    (lambda r: r['args'].append('agent.normalize_q_loss=true'), 'normalize_q_loss'),
    (lambda r: r['args'].__setitem__(r['args'].index('train.offline_steps=1000000'), 'train.offline_steps=200000'),
     'train.offline_steps'),
    (lambda r: r.update(host='local', pin_host='local'), 'host'),
    (lambda r: r['fql_budget'].update(normalize_q_loss=False), 'fql_budget'),
    (lambda r: r.update(deps=['other']), 'data.dataset'),
])
def test_refuses_unexpected_sources(repo, change, message):
    mutate(repo, 'rwcal-20261109-fql-sincro-mug-hanging-s1', change)
    with pytest.raises(R.Refused, match=message):
        prepare(repo)


def test_refuses_missing_source_and_quarantined_prefix(repo, monkeypatch):
    doc = yaml.safe_load((repo / R.QUEUE).read_text())
    doc['jobs'] = [r for r in doc['jobs'] if r['id'] != 'rwng-20261107b-fql-dynamo-drawer-open-s2']
    (repo / R.QUEUE).write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(R.Refused, match='derived_from missing'):
        prepare(repo)
    write_repo(repo.parent / 'q', queue_rows())
    monkeypatch.setitem(R.SOURCE_PREFIX, ('mvp3d_single', 'drawer-open'), 'rwcal-20261109')
    with pytest.raises(R.Refused, match='quarantined source'):
        prepare(repo.parent / 'q')


@pytest.mark.parametrize('case', ['release', 'manifest', 'existing_id', 'registry_id', 'run_dir', 'dataset'])
def test_refusals(repo, case):
    if case == 'release':
        (repo / 'runs/setup/role-release.json').write_text(json.dumps({'commit': '9' * 40, 'image': IMAGE}))
        match = 'role-release'
    elif case == 'manifest':
        (repo / f'runs/remote/acceptance/{COMMIT[:7]}-cc/manifest.json').unlink()
        match = 'acceptance manifest'
    elif case == 'existing_id':
        doc = yaml.safe_load((repo / R.QUEUE).read_text())
        doc['jobs'].append({'id': 'rwnq-20261111-fql-reviwo-drawer-open-s0', 'script': 'x'})
        (repo / R.QUEUE).write_text(yaml.safe_dump(doc, sort_keys=False))
        match = 'already exists'
    elif case == 'registry_id':
        with (repo / R.REGISTRY).open('a') as stream:
            stream.write(json.dumps({'id': 'rwa100-20261111-fql-sincro-drawer-open-s2'}) + '\n')
        match = 'already exists'
    elif case == 'run_dir':
        (repo / 'runs/rwa100-20261111-fql-reviwo-mug-hanging-s1').mkdir()
        match = 'run directory already exists'
    else:
        (repo / 'runs/rwcal-20261110-fqldata-mvp3d_single-mug-hanging-native/dataset.h5').unlink()
        match = 'missing dependency dataset'
    with pytest.raises(R.Refused, match=match):
        prepare(repo)


def test_cli_dry_run_and_execute_guard(repo, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(R, 'default_active_pauses', lambda root: fake_pauses())
    out = tmp_path / 'dryrun.json'
    assert R.main(['--repo', str(repo), '--commit', COMMIT, '--image', IMAGE, '--out', str(out)]) == 0
    assert json.loads(out.read_text())['status'] == 'dry-run'
    assert json.loads(capsys.readouterr().out)['queue_diff']['added_rows'] == 60
    with pytest.raises(R.Refused, match='exists'):
        R.main(['--repo', str(repo), '--commit', COMMIT, '--image', IMAGE, '--out', str(out)])
    with pytest.raises(R.Refused, match='operational'):
        R.main(['--repo', str(repo), '--commit', COMMIT, '--image', IMAGE, '--out', str(repo / 'runs/setup/x.json')])
    dangling = tmp_path / 'dangling.json'
    dangling.symlink_to(tmp_path / 'elsewhere' / 'target.json')
    with pytest.raises(R.Refused, match='exists'):
        R.main(['--repo', str(repo), '--commit', COMMIT, '--image', IMAGE, '--out', str(dangling)])
    assert not (tmp_path / 'elsewhere').exists()
    monkeypatch.undo()  # the real execution guard from here on
    with pytest.raises(R.Refused, match="coordinator's step"):
        R.main(['--repo', str(repo), '--commit', COMMIT, '--image', IMAGE, '--execute'])
    record = prepare(repo)
    for publish in (lambda: R.execute(repo, record, active_pauses=fake_pauses()),
                    lambda: R.check_publication(repo, record, fake_pauses())):
        with pytest.raises(R.Refused, match="coordinator's step"):
            publish()
    assert not (repo / f'runs/setup/{R.STEM}.json').exists()


def test_base_config_keeps_e2e_setting():
    """The setting rides on --set overrides; base.yaml (what E2E rows inherit) keeps normalize_q_loss false."""
    agent = yaml.safe_load((SOURCE_TREE / 'configs/realworld/fql/base.yaml').read_text())['agent']
    assert agent['normalize_q_loss'] is False


def test_live_queue_rows_derive_cleanly():
    """Read-only: the live sources (if present) satisfy every source/derivation check."""
    live = R.MAIN / R.QUEUE
    if not live.is_file():
        pytest.skip('no live queue')
    rows = R.build_rows(R.queue_document(live.read_bytes()),
                        {'commit': COMMIT, 'image': IMAGE, 'evidence_manifest': 'x.json'})
    assert len(rows) == 60 and {r['derived_from'].split('-fql-')[0] for r in rows} == {
        'rwng-20261107b', 'rwcal-20261109', 'rwcal-20261110'}
