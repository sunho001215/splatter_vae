"""Synthetic registrar checks: no live manifest, scheduler, remote, or operational writes."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SPEC = importlib.util.spec_from_file_location('ours_registrar', Path(__file__).with_name(
    'register_ours_realworld_20261111.py'))
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
PIN = {'commit': '1' * 40, 'image': 'sha256:' + '2' * 64}


def template(task='drawer-open', seed=0, alpha='10.0'):
    job = f'rwcal-20261109-fql-sincro-{task}-s{seed}'
    return {'id': job, 'script': 'scripts/train_realworld_fql.py', 'args': [
        '--config', 'configs/realworld/fql/base.yaml', '--name', job, '--output-root', 'runs', '--set',
        'condition=sincro', f'data.dataset=runs/rwcal-20261109-fqldata-sincro-{task}/dataset.h5',
        f'data.split_manifest=/old/{task}/split_seed0.json', f'data.gate_report=/old/{task}/gate_report.json',
        f'train.seed={seed}', f'agent.alpha={alpha}', 'agent.batch_size=256', 'train.offline_steps=1000000',
        'train.log_interval=5000', 'train.eval_interval=100000']}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    (repo / 'experiments').mkdir(parents=True)
    (repo / R.QUEUE).write_text('# keep this comment\njobs:\n' + yaml.safe_dump([template()], sort_keys=False)
                              + '\nmetadata: {preserved: true}\n')
    (repo / R.REGISTRY).write_bytes(b'')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{"episodes": {}}')
    copies, reports = {}, {}
    for task in R.TASKS:
        config = repo / f'configs/realworld/ours/{task}.yaml'
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_bytes((R.SOURCE / f'configs/realworld/ours/{task}.yaml').read_bytes())
        directory = tmp_path / 'inputs' / task
        directory.mkdir(parents=True)
        for name in R.INPUT_NAMES:
            (directory / name).write_bytes(b'synthetic input')
        copies[task] = str(directory)
        reports[task] = {'tasks': {task: {'passed': True, 'target': 100, 'amendment': None}},
                         'families': {task: {f: {'passed': True} for f in R.gate.QA_FAMILIES}},
                         'calibration_hold': {task: {'passed': True, 'problems': []}},
                         'manifest': {'path': str(manifest), 'sha256': R.sha(manifest.read_bytes())}}

    def evaluate(path, tasks):
        assert path == manifest and len(tasks) == 1
        return copy.deepcopy(reports[tasks[0]])

    monkeypatch.setattr(R.gate, 'evaluate_file', evaluate)
    monkeypatch.setattr(R, 'verify_inputs', lambda t, m, r, d: {
        n: R.sha((Path(d) / n).read_bytes()) for n in R.INPUT_NAMES})
    return SimpleNamespace(repo=repo, manifest=manifest, copies=copies, reports=reports)


def plan(fake, tasks=('drawer-open',), inputs=None, copies=None):
    return R.prepare(fake.repo, list(tasks), fake.manifest, inputs or R.DEFAULT_INPUTS,
                     fake.copies if copies is None else copies, PIN)


def cli(fake, tasks=('drawer-open',)):
    return ['--repo', str(fake.repo), '--manifest', str(fake.manifest), '--commit', PIN['commit'],
            '--image', PIN['image'], '--tasks', *tasks,
            *(arg for t in tasks for arg in ('--inputs-copy', f'{t}={fake.copies[t]}'))]


def test_stdout_dry_run_exact_rows_resources_and_proof(fake, capsys):
    before = {p: p.read_bytes() for p in fake.repo.rglob('*') if p.is_file()}
    assert R.main(cli(fake)) == 0
    record = json.loads(capsys.readouterr().out)
    assert record['status'] == 'dry-run' and record['registered'] is False
    assert set(record['admitted']) == {'drawer-open'} and record['refused'] == {}
    rows = record['rows']
    task, inputs = 'drawer-open', R.DEFAULT_INPUTS['drawer-open']
    stats, pre, export = [f'rwcal-20261111-ours-{task}-{k}' for k in ('stats', 'pre', 'export')]
    assert [r['id'] for r in rows] == [stats, pre, export]
    assert [r['deps'] for r in rows] == [[], [stats], [pre]]
    assert [r['script'] for r in rows] == ['scripts/compute_workspace_stats.py', 'scripts/train.py',
                                         'scripts/export_encoder.py']
    assert rows[0]['args'] == ['--regime', 'realworld', '--path', f'{inputs}/store.yaml', '--split',
                               f'{inputs}/split_seed0.json', '--out', f'runs/{stats}/workspace_stats.json']
    assert rows[1]['args'] == ['--config', f'configs/realworld/ours/{task}.yaml', '--name', pre,
                               '--output-root', 'runs', '--resume', 'auto', '--set',
                               f'data.store={inputs}/store.yaml', f'data.split_manifest={inputs}/split_seed0.json',
                               f'data.gate_report={inputs}/gate_report.json',
                               f'model.anchor_stats=runs/{stats}/workspace_stats.json']
    assert rows[2]['args'] == ['--run', f'runs/{pre}', '--checkpoint',
                               f'runs/{pre}/checkpoints/step_0200000.pt', '--out', f'runs/{export}/encoder.pt']
    assert [r['required_results'] for r in rows] == [['workspace_stats.json'],
        ['completion.json', 'metrics.jsonl', 'checkpoints/step_0200000.pt', 'config.yaml'], ['encoder.pt']]
    for kind, row in zip(('stats', 'pre', 'export'), rows):
        assert {k: row[k] for k in R.FOOTPRINT[kind]} == R.FOOTPRINT[kind]
        assert row['max_restarts'] == (2 if kind == 'pre' else 1)
        assert all(row[k] == v for k, v in dict(role='realworld_pretrain', host='remote', pin_host='remote',
            remote_host='192.168.10.22', gpu='any', priority=1, oom_score_adj=0, realworld_target=task,
            realworld_consumer='ours', calibration_dependent=True,
            gate_level='geometry (full-data gate + calibration_validated)', planned_commit=PIN['commit'],
            planned_image=PIN['image'], registration_tool=R.TOOL).items())
        assert row['inputs'] == [f'{inputs}/{name}' for name in R.INPUT_NAMES[:3]]
        assert row['realworld_gate_sha256'] == R.json_sha(fake.reports[task])
        assert row['realworld_manifest_sha256'] == R.sha(fake.manifest.read_bytes())
        assert row['evidence_manifest'] == f"runs/remote/acceptance/{PIN['commit'][:7]}-cc/manifest.json"
    candidate = R.queue_candidate(before[fake.repo / R.QUEUE], rows)
    assert record['queue_sha256_before'] == R.sha(before[fake.repo / R.QUEUE])
    assert record['queue_sha256_after'] == R.sha(candidate)
    assert record['tool_sha256'] == R.sha(Path(R.__file__).read_bytes())
    assert record['counts'] == dict(requested_tasks=1, admitted_tasks=1, refused_tasks=0,
                                    pretraining_rows=3, held_fql_specs=4)
    assert before == {p: p.read_bytes() for p in fake.repo.rglob('*') if p.is_file()}


@pytest.mark.parametrize('failure', ['base', 'singleview', 'multiview', 'geometry', 'calibration'])
def test_ours_gate_failure_refuses_whole_task_and_continues(fake, failure, capsys):
    report = fake.reports['drawer-open']
    if failure == 'base':
        report['tasks']['drawer-open'].update(passed=False, problems=['not complete'])
    elif failure == 'calibration':
        report['calibration_hold']['drawer-open'].update(passed=False, problems=['calibration held'])
    else:
        report['families']['drawer-open'][failure].update(passed=False, failed_count=1)
    assert R.main(cli(fake, ('drawer-open', 'mug-hanging'))) == 0
    record = json.loads(capsys.readouterr().out)
    assert record['refused']['drawer-open'] == R.gate.task_problems(report, 'drawer-open', 'ours')
    assert set(record['admitted']) == {'mug-hanging'}
    assert len(record['rows']) == 3 and all(r['realworld_target'] == 'mug-hanging' for r in record['rows'])


@pytest.mark.parametrize('collision', ['queue', 'registry', 'run', 'symlink', 'other-id'])
def test_collisions_refuse_entire_task(fake, collision):
    job = 'rwcal-20261111-ours-drawer-open-pre'
    if collision in ('queue', 'other-id'):
        row = {'id': job} if collision == 'queue' else dict(id='different-id', realworld_target='drawer-open',
                realworld_consumer='ours', role='realworld_pretrain')
        text = (fake.repo / R.QUEUE).read_text()
        (fake.repo / R.QUEUE).write_text(text.replace('\nmetadata:', yaml.safe_dump([row]) + '\nmetadata:'))
    elif collision == 'registry':
        (fake.repo / R.REGISTRY).write_text(json.dumps({'id': job}) + '\n')
    else:
        run = fake.repo / 'runs' / job
        run.parent.mkdir()
        run.mkdir() if collision == 'run' else run.symlink_to('/nonexistent-synthetic-run')
    record = plan(fake, ('drawer-open', 'mug-hanging'))
    assert 'already exists' in ' '.join(record['refused']['drawer-open'])
    assert set(record['admitted']) == {'mug-hanging'} and len(record['rows']) == 3
    assert all(s['realworld_target'] == 'mug-hanging' for s in record['held_spec'])


def test_config_drift_refuses_and_all_refused_exit_two(fake, capsys):
    config = fake.repo / 'configs/realworld/ours/drawer-open.yaml'
    config.write_text(config.read_text() + '# drift\n')
    assert R.main(cli(fake)) == 2
    record = json.loads(capsys.readouterr().out)
    assert record['refused'] == {'drawer-open': ['config differs from ours.render(task)']}
    assert record['rows'] == record['held_spec'] == []
    assert record['queue_sha256_before'] == record['queue_sha256_after']


def test_execute_refuses_outside_main_before_any_writes(fake):
    before = sorted(fake.repo.rglob('*'))
    with pytest.raises(R.Refused, match="execution is the coordinator's step"):
        R.main(cli(fake) + ['--execute'])
    assert sorted(fake.repo.rglob('*')) == before


def test_worktree_tool_cannot_execute_even_with_main_repo(fake, monkeypatch):
    monkeypatch.setattr(R, 'release_pin', lambda repo: PIN)
    with pytest.raises(R.Refused, match="execution is the coordinator's step"):
        R.execution_allowed(R.MAIN, PIN)


def test_main_copy_still_requires_matching_role_release(fake, monkeypatch):
    tool = fake.repo / R.TOOL
    monkeypatch.setattr(R, 'MAIN', fake.repo)
    monkeypatch.setattr(R, '__file__', str(tool))
    monkeypatch.setattr(R, 'release_pin', lambda repo: {**PIN, 'commit': '3' * 40})
    with pytest.raises(R.Refused, match='role release differs'):
        R.execution_allowed(fake.repo, PIN)
    monkeypatch.setattr(R, 'release_pin', lambda repo: PIN)
    R.execution_allowed(fake.repo, PIN)


def test_publication_rechecks_gate_without_comparing_clock_time(fake, monkeypatch):
    record = plan(fake)
    monkeypatch.setattr(R, 'execution_allowed', lambda repo, release: None)
    fake.reports['drawer-open']['checked_at_unix'] = 456
    R.check_publication(fake.repo, record)
    fake.reports['drawer-open']['families']['drawer-open']['geometry']['passed'] = False
    with pytest.raises(R.Refused, match='live gate changed'):
        R.check_publication(fake.repo, record)


def test_locked_publication_journals_before_queue_replace(fake, monkeypatch):
    # Only a synthetic tmp_path repository is mutated; exercise coordinator publication locally.
    record = plan(fake)
    setup = fake.repo / 'runs/setup'
    setup.mkdir(parents=True)
    monkeypatch.setattr(R, 'execution_allowed', lambda repo, release: None)
    original_replace = R.os.replace
    original_flock = R.fcntl.flock
    lock_names = []

    def flock(fd, operation):
        lock_names.append(Path(fd.name).name)
        return original_flock(fd, operation)

    def replace(source, destination):
        if destination == fake.repo / R.QUEUE:
            pending = json.loads((setup / f'{R.STEM}.pending.json').read_bytes())
            assert pending['status'] == 'pending' and pending['registered'] is False
            assert R.sha(Path(source).read_bytes()) == pending['queue_sha256_after']
            assert R.sha(Path(destination).read_bytes()) == pending['queue_sha256_before']
        return original_replace(source, destination)

    monkeypatch.setattr(R.fcntl, 'flock', flock)
    monkeypatch.setattr(R.os, 'replace', replace)
    result = R.execute(fake.repo, record)
    assert lock_names == ['stage2-driver.lock', 'registry.lock']
    assert result['registered'] is True and result['status'] == 'published'
    assert json.loads(R.proof_path(setup, result).read_bytes()) == result
    assert not (setup / f'{R.STEM}.pending.json').exists()
    assert not (setup / f'{R.STEM}.queue-candidate.yaml').exists()
    assert R.sha((fake.repo / R.QUEUE).read_bytes()) == result['queue_sha256_after']
    assert (fake.repo / R.REGISTRY).read_bytes() == b''


@pytest.mark.parametrize('local_template', [False, True])
def test_held_fql_specs_mirror_queue_args_and_never_enter_candidate(fake, local_template):
    task, remote = 'covering', '/remote/release/covering'
    if local_template:
        text = (fake.repo / R.QUEUE).read_text()
        extra = yaml.safe_dump([template(task, s, alpha=str(20 + s)) for s in range(3)])
        (fake.repo / R.QUEUE).write_text(text.replace('\nmetadata:', extra + '\nmetadata:'))
    record = plan(fake, (task,), inputs={task: remote})
    assert set(record['admitted']) == {task}
    held = record['held_spec']
    dataset, export = f'rwcal-20261111-fqldata-ours-{task}', f'rwcal-20261111-ours-{task}-export'
    assert [s['id'] for s in held] == [dataset] + [f'rwcal-20261111-fql-ours-{task}-s{s}' for s in range(3)]
    assert all(s['registrable'] is False and s['held_by'] == R.HELD_BY for s in held)
    assert held[0]['deps'] == [export]
    assert held[0]['args'] == ['build', '--store', f'{remote}/store.yaml', '--condition', 'ours',
        '--out', f'runs/{dataset}/dataset.h5', '--gate-report', f'{remote}/gate_report.json',
        '--outcomes', f'{remote}/outcomes.json', '--export', f'runs/{export}/encoder.pt', '--device', 'cuda']
    for seed, spec in enumerate(held[1:]):
        source = template(task if local_template else 'drawer-open', seed if local_template else 0,
                          alpha=str(20 + seed) if local_template else '10.0')
        assert spec['derived_from'] == source['id'] and spec['deps'] == [dataset]
        assert spec['args'] == ['--config', 'configs/realworld/fql/base.yaml', '--name', spec['id'],
            '--output-root', 'runs', '--set', 'condition=ours', f'data.dataset=runs/{dataset}/dataset.h5',
            f'data.split_manifest={remote}/split_seed0.json', f'data.gate_report={remote}/gate_report.json',
            f'train.seed={seed}', f"agent.alpha={20 + seed if local_template else '10.0'}", 'agent.batch_size=256',
            'train.offline_steps=1000000', 'train.log_interval=5000', 'train.eval_interval=100000']
    before = (fake.repo / R.QUEUE).read_bytes()
    candidate = R.queue_candidate(before, record['rows'])
    assert b'# keep this comment' in candidate
    assert yaml.safe_load(candidate) == {**yaml.safe_load(before),
                                        'jobs': yaml.safe_load(before)['jobs'] + record['rows']}
    assert not {s['id'] for s in held} & {s['id'] for s in yaml.safe_load(candidate)['jobs']}


def test_inputs_not_staged_and_optional_out(fake, tmp_path, capsys):
    record = plan(fake, ('covering', 'pick-and-place'))
    assert record['refused'] == {t: ['inputs not staged'] for t in ('covering', 'pick-and-place')}
    assert plan(fake, copies={})['refused'] == {'drawer-open': ['inputs not staged']}
    out = tmp_path / 'proof.json'
    assert R.main(cli(fake) + ['--out', str(out)]) == 0
    assert json.loads(out.read_bytes()) == json.loads(capsys.readouterr().out)


@pytest.mark.parametrize('n', [100, 98, 97])
def test_verify_inputs_uses_amendment_target_and_hashes_outcomes_without_parsing(tmp_path, n):
    task = 'drawer-open' if n == 100 else 'covering' if n == 98 else 'pick-and-place'
    identities = {f'manifest-{i}': dict(R.gate.raw_identity(task, f'episode_{i:06d}.h5', R.sha(str(i).encode())),
                                     store_episode=None) for i in range(n)}
    report = {'tasks': {task: dict(target=n, amendment={'target': n} if n != 100 else None,
               episodes=list(identities), episode_identities=identities, store_episodes=[])},
               'calibration_hold': {task: {'passed': True, 'episodes': {
                   ep: {'calibrate_build_id': 'cal-1', 'cameras': ['cam']} for ep in identities}}},
               'families': {task: {f: {'passed': True} for f in R.gate.QA_FAMILIES}},
               'manifest': {'sha256': 'a' * 64}}
    actual = {f'compact-{i}': rec for i, rec in enumerate(identities.values())}
    eps = list(actual)
    store = SimpleNamespace(episodes=lambda: eps, task=lambda ep: task,
                            raw_recording=lambda ep: actual[ep]['raw_recording'],
                            raw_sha256=lambda ep: actual[ep]['raw_sha256'])
    manifest = {'episodes': {ep: {'preprocessing': {'stages': {'train_data': {
        'outputs': f'/remote/{ep}/train.h5'}}}} for ep in identities}}
    descriptor = {'kind': 'compact', 'paths': [f'/remote/{ep}/train.h5' for ep in identities]}
    bound = R.gate.bind_store_episodes(report, {task: eps}, actual)
    (tmp_path / 'store.yaml').write_text(yaml.safe_dump(descriptor))
    (tmp_path / 'gate_report.json').write_text(json.dumps(bound))
    (tmp_path / 'split_seed0.json').write_text(json.dumps(R.split_episodes(store)))
    (tmp_path / 'outcomes.json').write_bytes(b'not JSON: pretraining never reads labels')
    hashes = R.verify_inputs(task, manifest, report, tmp_path)
    assert hashes == {name: R.sha((tmp_path / name).read_bytes()) for name in R.INPUT_NAMES}
    descriptor['paths'][0] = '/wrong/train.h5'
    (tmp_path / 'store.yaml').write_text(yaml.safe_dump(descriptor))
    with pytest.raises(R.Refused, match='live .* train_data outputs'):
        R.verify_inputs(task, manifest, report, tmp_path)


def test_config_root_is_a_dry_run_only_source_for_the_generated_configs(fake, tmp_path, capsys):
    """Before the release the main checkout lacks configs/realworld/ours: --config-root points the dry run at the
    source checkout; --execute never accepts it."""
    for task in R.TASKS:
        (fake.repo / f'configs/realworld/ours/{task}.yaml').unlink()
    assert R.main(cli(fake)) == 2  # the repo itself has no generated configs: refused
    capsys.readouterr()
    assert R.main([*cli(fake), '--config-root', str(R.SOURCE)]) == 0
    assert set(json.loads(capsys.readouterr().out)['admitted']) == {'drawer-open'}
    with pytest.raises(R.Refused):
        R.main([*cli(fake), '--config-root', str(R.SOURCE), '--execute'])


def _staged_inputs(tmp_path, task, live_task_section, *, staged_target):
    """A verify_inputs-consistent learning-inputs copy whose staged gate report has (or lacks) 'target'."""
    eps = [f'{task}/{task}-{i:06d}' for i in range(R.gate.TARGET)]
    identities = {f'{task}/episode_{i:06d}.h5': {'task': task, 'raw_recording': f'{task}/episode_{i:06d}.h5',
                                                  'raw_sha256': f'{i:064x}', 'store_episode': ep}
                  for i, ep in enumerate(eps)}
    staged_tasks = {k: v for k, v in live_task_section.items() if staged_target or k not in ('target', 'amendment')}
    staged_tasks.update(store_episodes=eps, episode_identities=identities)
    report = {'tasks': {task: staged_tasks}, 'calibration_hold': {task: {'passed': True}},
              'families': {task: {'singleview': {'passed': True}}}, 'manifest': {'sha256': '0' * 64}}
    directory = tmp_path / f'staged-{staged_target}' / task
    directory.mkdir(parents=True)
    outputs = [f'/remote/processed/{task}/{i:06d}' for i in range(R.gate.TARGET)]
    (directory / 'store.yaml').write_text(yaml.safe_dump({'kind': 'compact', 'paths': outputs}))
    (directory / 'gate_report.json').write_text(json.dumps(report))
    store = SimpleNamespace(episodes=lambda: eps, task=lambda ep: task,
                            raw_recording=lambda ep: identities[f'{task}/episode_{eps.index(ep):06d}.h5']['raw_recording'],
                            raw_sha256=lambda ep: identities[f'{task}/episode_{eps.index(ep):06d}.h5']['raw_sha256'])
    (directory / 'split_seed0.json').write_text(json.dumps(R.split_episodes(store)))
    (directory / 'outcomes.json').write_text('{}')
    manifest = {'episodes': {key: {'preprocessing': {'stages': {'train_data': {'outputs': out}}}}
                             for key, out in zip(identities, outputs)}}
    return directory, manifest, report


@pytest.mark.parametrize('staged_target', [True, False])
def test_reports_staged_before_the_amendment_schema_verify_only_for_the_unamended_default(
        tmp_path, monkeypatch, staged_target):
    task = 'drawer-open'
    live_section = {'passed': True, 'target': R.gate.TARGET, 'amendment': None, 'episodes': []}
    directory, manifest, staged = _staged_inputs(tmp_path, task, live_section, staged_target=staged_target)
    live = copy.deepcopy(staged)
    live['tasks'][task].update(target=R.gate.TARGET, amendment=None)
    live['tasks'][task]['episodes'] = list(manifest['episodes'])
    staged_disk = json.loads((directory / 'gate_report.json').read_text())
    staged_disk['tasks'][task]['episodes'] = list(manifest['episodes'])
    (directory / 'gate_report.json').write_text(json.dumps(staged_disk))
    monkeypatch.setattr(R.gate, 'bind_store_episodes', lambda report, task_eps, ids: report)
    assert set(R.verify_inputs(task, manifest, live, directory)) == set(R.INPUT_NAMES)
    amended = copy.deepcopy(live)
    amended['tasks'][task].update(target=98, amendment={'target': 98})
    with pytest.raises(R.Refused):  # an amended live gate never matches a report without the amendment fields
        R.verify_inputs(task, manifest, amended, directory)


def test_a_published_task_set_does_not_block_later_tasks_but_a_pending_journal_does(fake):
    setup = fake.repo / 'runs/setup'
    setup.mkdir(parents=True, exist_ok=True)
    (setup / f'{R.STEM}-drawer-open+mug-hanging.json').write_text('{}')
    assert set(plan(fake, tasks=('covering',), inputs={**R.DEFAULT_INPUTS, 'covering':
                '/home/compu/kaist/sunho/realworld/learning_inputs/calrel-x/covering'})['admitted']) == {'covering'}
    assert R.proof_path(setup, {'admitted': {'pick-and-place': {}, 'covering': {}}}).name == \
        f'{R.STEM}-covering+pick-and-place.json'
    (setup / f'{R.STEM}.pending.json').write_text('{}')
    with pytest.raises(R.Refused, match='pending'):
        plan(fake, tasks=('covering',))
