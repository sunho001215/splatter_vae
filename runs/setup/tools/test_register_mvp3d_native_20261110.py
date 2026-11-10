"""CPU-only tests; every mutation stays in tmp_path beneath this scratch directory.

No real queue/registry/pause writes, scheduler/campaign imports, network or GPU.
The native worktree's actual trainer API is used until that release is installed
in the main checkout; the check tool itself always imports from the main repo.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

sys.dont_write_bytecode = True
import h5py
import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = Path('/home/ws/ws/splatter4d')
SPEC_PATH = Path('/home/ws/ws/realworld-ops/serving/evidence/mvpfix/native-rows.json')
PIN = {'commit': '1' * 40, 'image': 'sha256:' + '2' * 64}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def sandbox_repo(tmp_path):
    # The basetemp in the requested command keeps even temporary repos HERE.
    assert tmp_path.resolve().is_relative_to(HERE)
    repo = tmp_path / 'repo'
    setup = repo / 'runs/setup'
    tools = setup / 'tools'
    tools.mkdir(parents=True)
    (repo / 'experiments').mkdir()
    for name in ('register_mvp3d_native_20261110.py', 'check_mvp3d_native_fqldata_20261110.py'):
        (tools / name).write_bytes((HERE / name).read_bytes())
    rows = repo / 'native-rows.json'
    rows.write_bytes(SPEC_PATH.read_bytes())
    (repo / 'experiments/queue.yaml').write_text(
        '# existing queue comment\nhosts: {local: {}, remote: {}}\njobs:\n'
        '- id: rwcal-20261109-old\n  deps: []\n# tail comment\n')
    (repo / 'experiments/registry.jsonl').write_text('{"id":"old-done","event":"exited","code":0}\n')
    (setup / 'admission-pauses.json').write_text(json.dumps({'doc': 'Existing holds', 'pauses': [
        {'name': 'older-hold', 'active': True, 'prefixes': ['rwcal-20261109-'],
         'since': '2026-11-09', 'by': 'coordinator', 'reason': 'keep this entry'}]}))
    r = load('native_registrar_test_copy', tools / 'register_mvp3d_native_20261110.py')
    c = load('native_checker_test_copy', tools / 'check_mvp3d_native_fqldata_20261110.py')
    return repo, rows, r, c


def cli_args(repo, rows):
    return ['--repo', str(repo), '--rows', str(rows), '--commit', PIN['commit'], '--image', PIN['image']]


def plan(fixture):
    repo, rows, r, _ = fixture
    return r.prepare(repo, rows, **PIN)


def test_dry_run_prints_exact_plan_and_writes_nothing(sandbox_repo, capsys):
    repo, rows, r, _ = sandbox_repo
    before = snapshot(repo)
    assert r.main(cli_args(repo, rows)) == 0
    result = json.loads(capsys.readouterr().out)
    assert snapshot(repo) == before
    assert result['status'] == 'dry-run' and result['registered'] is False
    assert len(result['jobs']) == 20
    assert result['queue_diff']['counts'] == {'native-depth': 2, 'pretraining': 2, 'fqldata': 4, 'FQL': 12}
    assert result['queue_diff']['modified_rows'] == result['queue_diff']['removed_rows'] == 0
    source = json.loads(rows.read_text())['rows']
    for raw, row in zip(source, result['jobs']):
        expected = {**raw, 'planned_commit': PIN['commit'], 'planned_image': PIN['image'],
                    'evidence_manifest': 'runs/remote/acceptance/1111111-cc/manifest.json',
                    'registration_tool': 'runs/setup/tools/register_mvp3d_native_20261110.py'}
        assert row == expected
    assert '<release containing' not in json.dumps(result)
    assert result['pause_entry']['name'] == r.PAUSE_NAME
    assert result['pause_entry']['active'] is True
    assert result['pause_entry']['reason'] == r.PAUSE_REASON
    assert result['pause_entry']['prefixes'] == r.PAUSE_PREFIXES


def test_execute_only_temp_copy_pins_appends_and_adds_pause(sandbox_repo, capsys, monkeypatch):
    repo, rows, r, _ = sandbox_repo
    queue, registry, pauses = (repo / r.QUEUE), (repo / r.REGISTRY), (repo / r.PAUSES)
    before, reg_before, pauses_before = queue.read_bytes(), registry.read_bytes(), json.loads(pauses.read_bytes())
    write_order = []
    original_write, original_replace = r.atomic_write, r.os.replace

    def confined_write(path, raw):
        assert path.resolve().is_relative_to(repo)
        write_order.append(path)
        original_write(path, raw)

    def confined_replace(source, destination):
        assert Path(source).resolve().is_relative_to(repo)
        assert Path(destination).resolve().is_relative_to(repo)
        write_order.append(Path(destination))
        return original_replace(source, destination)

    monkeypatch.setattr(r, 'atomic_write', confined_write)
    monkeypatch.setattr(r.os, 'replace', confined_replace)
    # --execute is exercised only inside this imported, copied module and temp repo.
    assert r.main(cli_args(repo, rows) + ['--execute']) == 0
    result = json.loads(capsys.readouterr().out)
    jobs = yaml.safe_load(queue.read_bytes())['jobs']
    assert jobs == yaml.safe_load(before)['jobs'] + result['jobs']
    assert len(jobs) == 21 and len({j['id'] for j in jobs}) == 21
    assert b'# existing queue comment' in queue.read_bytes() and b'# tail comment' in queue.read_bytes()
    assert registry.read_bytes() == reg_before
    assert all(j['planned_commit'] == PIN['commit'] and j['planned_image'] == PIN['image'] for j in jobs[1:])
    pause_doc = json.loads(pauses.read_bytes())
    assert pause_doc == {**pauses_before, 'pauses': pauses_before['pauses'] + [result['pause_entry']]}
    for prefix in r.PAUSE_PREFIXES:
        assert sum(j['id'].startswith(prefix) for j in jobs) == 3
    assert write_order.index(pauses) < write_order.index(queue)
    assert result['registered'] and result['status'] == 'published'
    setup = repo / 'runs/setup'
    proof = json.loads((setup / f'{r.STEM}.json').read_bytes())
    assert proof == result
    assert proof['queue_sha256_before'] == r.sha(before)
    assert proof['queue_sha256_after'] == r.sha(queue.read_bytes())
    assert proof['pauses_sha256_after'] == r.sha(pauses.read_bytes())
    assert proof['jobs_sha256'] == r.json_sha(result['jobs'])
    assert not (setup / f'{r.STEM}.pending.json').exists()
    assert not (setup / f'{r.STEM}.queue-candidate.yaml').exists()
    assert (setup / 'stage2-driver.lock').exists() and (repo / 'experiments/registry.lock').exists()


@pytest.mark.parametrize('where', ['queue', 'registry'])
def test_existing_id_refused_without_writes(sandbox_repo, where):
    repo, rows, r, _ = sandbox_repo
    job = json.loads(rows.read_text())['rows'][0]['id']
    if where == 'queue':
        (repo / r.QUEUE).write_text(yaml.safe_dump({'jobs': [{'id': job}]}))
    else:
        (repo / r.REGISTRY).write_text(json.dumps({'id': job, 'event': 'reset'}) + '\n')
    before = snapshot(repo)
    with pytest.raises(r.Refused, match='ID already exists'):
        r.main(cli_args(repo, rows) + ['--execute'])
    assert snapshot(repo) == before


@pytest.mark.parametrize('commit,image', [
    ('4bb3bc0', PIN['image']), ('g' * 40, PIN['image']), ('A' * 40, PIN['image']),
    (PIN['commit'], 'latest'), (PIN['commit'], 'sha256:' + 'a' * 63),
    (PIN['commit'], 'sha256:' + 'z' * 64)])
def test_malformed_release_refused(sandbox_repo, commit, image):
    repo, rows, r, _ = sandbox_repo
    before = snapshot(repo)
    with pytest.raises(r.Refused, match='--commit|--image'):
        r.prepare(repo, rows, commit, image)
    assert snapshot(repo) == before


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unexpected', 'deps', 'norm', 'placeholder', 'shape'])
def test_wrong_row_set_or_chain_refused(sandbox_repo, change):
    repo, path, r, _ = sandbox_repo
    doc = json.loads(path.read_text())
    if change == 'missing':
        doc['rows'].pop()
    elif change == 'duplicate':
        doc['rows'][-1] = copy.deepcopy(doc['rows'][0])
    elif change == 'unexpected':
        doc['rows'][-1]['id'] += '-oops'
    elif change == 'deps':
        doc['rows'][1]['deps'] = ['rwcal-20261109-old']
    elif change == 'norm':
        next(row for row in doc['rows'] if r.kind(row) == 'FQL')['args'].remove('encoder.feature_norm=train_standardize')
    elif change == 'placeholder':
        doc['rows'][0]['unexpected'] = '<release containing unknown>'
    else:
        doc = []
    path.write_text(json.dumps(doc))
    before = snapshot(repo)
    with pytest.raises(r.Refused):
        r.main(cli_args(repo, path) + ['--execute'])
    assert snapshot(repo) == before


@pytest.mark.parametrize('active', [True, False])
def test_existing_pause_name_refused_even_if_inactive(sandbox_repo, active):
    repo, rows, r, _ = sandbox_repo
    path = repo / r.PAUSES
    doc = json.loads(path.read_bytes())
    doc['pauses'].append({'name': r.PAUSE_NAME, 'active': active, 'prefixes': []})
    path.write_text(json.dumps(doc))
    before = snapshot(repo)
    with pytest.raises(r.Refused, match='pause already exists'):
        r.main(cli_args(repo, rows) + ['--execute'])
    assert snapshot(repo) == before


@pytest.mark.parametrize('when', ['acquiring', 'staging'])
def test_queue_changed_under_lock_refused(sandbox_repo, monkeypatch, when):
    repo, _, r, _ = sandbox_repo
    record = plan(sandbox_repo)
    queue, pauses = repo / r.QUEUE, repo / r.PAUSES
    before, pause_before = queue.read_bytes(), pauses.read_bytes()
    changed = before + b'# external concurrent edit\n'
    if when == 'acquiring':
        original = r.locks

        @contextmanager
        def racing_locks(repo):
            with original(repo):
                queue.write_bytes(changed)
                yield

        monkeypatch.setattr(r, 'locks', racing_locks)
    else:
        original_write = r.atomic_write

        def race_during_staging(path, raw):
            original_write(path, raw)
            if path.name.endswith('.pending.json'):
                queue.write_bytes(changed)

        monkeypatch.setattr(r, 'atomic_write', race_during_staging)
    with pytest.raises(r.Refused, match='queue changed under locks'):
        r.execute(repo, record)
    assert queue.read_bytes() == changed and pauses.read_bytes() == pause_before
    assert not (repo / 'runs/setup' / f'{r.STEM}.json').exists()


def test_pause_precedes_queue_even_when_queue_replace_is_interrupted(sandbox_repo, monkeypatch):
    repo, _, r, _ = sandbox_repo
    record = plan(sandbox_repo)
    before = (repo / r.QUEUE).read_bytes()
    original = r.os.replace

    def interrupted(source, destination):
        if Path(destination) == repo / r.QUEUE:
            raise OSError('synthetic interrupted queue replace')
        return original(source, destination)

    monkeypatch.setattr(r.os, 'replace', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        r.execute(repo, record)
    assert (repo / r.QUEUE).read_bytes() == before
    assert json.loads((repo / r.PAUSES).read_bytes())['pauses'][-1] == record['pause_entry']
    assert (repo / 'runs/setup' / f'{r.STEM}.pending.json').exists()


@pytest.mark.parametrize('jobs', ['[]', '[{id: old, deps: []}]', '\n- id: old\n  deps: []'])
def test_queue_append_preserves_existing_data_and_comments(sandbox_repo, jobs):
    _, _, r, _ = sandbox_repo
    before = f'# comment\nhosts: {{local: {{}}, remote: {{}}}}\njobs: {jobs}\n# tail\n'.encode()
    rows = [{'id': 'new', 'deps': []}]
    candidate = r.queue_candidate(before, rows)
    assert yaml.safe_load(candidate) == {**yaml.safe_load(before), 'jobs': yaml.safe_load(before)['jobs'] + rows}
    assert b'# comment' in candidate and b'# tail' in candidate


def test_cli_refusal_style_and_no_runtime_imports(sandbox_repo):
    repo, rows, r, _ = sandbox_repo
    args = cli_args(repo, rows)
    args[args.index('--commit') + 1] = 'main'
    proc = subprocess.run([sys.executable, '-I', '-B', r.__file__, *args], capture_output=True, text=True)
    assert proc.returncode == 2 and proc.stderr.startswith('REFUSED: --commit') and not proc.stdout
    assert 'ops.campaign' not in sys.modules and 'scripts.jobs' not in sys.modules


@pytest.fixture
def trainer(sandbox_repo, monkeypatch):
    repo, _, _, checker = sandbox_repo
    sys.path.insert(0, str(REPO))
    try:
        # Copy the real released implementation, with dependencies read from the
        # main checkout. Never import a trainer script or scheduler.
        source = REPO / 's4d/rwlearn/fql_data.py'
        if 'def fit_feature_standardization(' not in source.read_text():
            source = REPO / '.worktrees/mvpnative/s4d/rwlearn/fql_data.py'
        copied = repo / 'trainer_fql_data.py'
        copied.write_bytes(source.read_bytes())
        api = load('native_trainer_api_test_copy', copied)
    finally:
        sys.path.remove(str(REPO))
    monkeypatch.setattr(checker, 'trainer_api', lambda: (api.OfflineTransitions, api.fit_feature_standardization))
    return api


def dataset_fixture(directory, api, mode='pass'):
    directory.mkdir(parents=True, exist_ok=True)
    path, split = directory / 'dataset.h5', directory / 'split.json'
    split.write_text(json.dumps({'train': ['train_ep'], 'validation': ['val_ep']}))
    rng = np.random.default_rng(7)
    x = rng.normal(size=(96, 16)).astype(np.float16)
    val = x.copy()
    if mode == 'shift':
        val = (val.astype(np.float32) + 10).astype(np.float16)
    elif mode == 'low_ln':
        val = np.tile(x[0], (96, 1))
        val[-1] += np.float16(0.001)
    contract = {'pos_scale_m': 0.1, 'rot_scale_rad': 0.2, 'nominal_period_s': 1 / 15, 'max_step_s': 0.1}
    with h5py.File(path, 'w') as f:
        f.attrs.update(format=api.FORMAT, training_depth='native_realsense', condition='mvp3d_single',
                       sequence_feature=False, contract=json.dumps(contract), episodes=json.dumps(['train_ep', 'val_ep']))
        for ep, features in [('train_ep', x), ('val_ep', val)]:
            # Two timelines exercise per-timeline sampling, not a global 20-row sample.
            for timeline, features_part in [('cam0', features[:48]), ('cam1', features[48:])]:
                g = f.create_group(f'episodes/{ep}/{timeline}')
                n = len(features_part)
                g.attrs['task'] = 'drawer-open'
                for name, data in {'features': features_part, 'run': np.zeros(n, np.int64),
                                   'transition_ok': np.arange(n) < n - 1, 'action': np.zeros((n, 7), np.float32),
                                   'reward': np.zeros(n, np.float32), 'mask': np.ones(n, np.float32),
                                   'proprio': np.zeros((n, 10), np.float32)}.items():
                    g.create_dataset(name, data=data)
    return path, split


@pytest.mark.parametrize('mode,passed', [('pass', True), ('shift', False), ('low_ln', False)])
def test_checker_real_trainer_fit_and_pass_fail_output(sandbox_repo, trainer, mode, passed, capsys):
    repo, _, _, c = sandbox_repo
    dataset, split = dataset_fixture(repo / 'data', trainer, mode)
    report = repo / 'report.json'
    before = snapshot(repo)
    status = c.main(['drawer-open', 'mvp3d_single', '--dataset', str(dataset), '--split', str(split), '--out', str(report)])
    result = json.loads(capsys.readouterr().out)
    assert status == (0 if passed else 1)
    assert result == json.loads(report.read_text())
    after = snapshot(repo)
    assert {k: v for k, v in after.items() if k != 'report.json'} == before
    assert result['pass'] is passed and result['status'] == ('PASS' if passed else 'FAIL')
    assert ('pause_lift' in result) is passed
    assert result['norm_train']['sampled_rows'] == result['norm_validation']['sampled_rows'] == 40
    expected = trainer.fit_feature_standardization(trainer.OfflineTransitions(dataset, ['train_ep'], 'cpu').features)
    assert result['fit']['sha256'] == expected['sha256']
    assert result['V1_pass']['a_train']
    if mode == 'shift':
        assert result['norm_validation']['offset_to_signal'] > 1.0
    elif mode == 'low_ln':
        assert result['norm_validation']['ln_unit_std'] < 0.5
    if passed:
        assert result['pause_lift']['remove_prefix'] == 'rwcal-20261110-fql-mvp3d_single-drawer-open-'
        assert 'train offset_to_signal=' in result['pause_lift']['append_reason']


def test_checker_default_dry_run_writes_nothing(sandbox_repo, trainer, capsys):
    repo, _, _, c = sandbox_repo
    dataset, split = dataset_fixture(repo / 'data', trainer)
    before = snapshot(repo)
    assert c.main(['drawer-open', 'mvp3d_single', '--dataset', str(dataset), '--split', str(split)]) == 0
    assert json.loads(capsys.readouterr().out)['pass']
    assert snapshot(repo) == before


def test_statistics_match_the_coordinators_actual_check(sandbox_repo, trainer):
    repo, _, _, c = sandbox_repo
    dataset, _ = dataset_fixture(repo / 'data', trainer, 'shift')
    reference = Path('/home/ws/ws/realworld-ops/serving/tools/mvpfix/featcheck_norm.py')
    # Extract only the pure check function. Never execute the reference's CLI or
    # its output-file writer. This compares against the existing implementation.
    node = next(n for n in ast.parse(reference.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'check')
    namespace = {'np': np, 'h5py': h5py}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(reference), 'exec'), namespace)
    train = trainer.OfflineTransitions(dataset, ['train_ep'], 'cpu')
    stats = trainer.fit_feature_standardization(train.features)
    mean, std = stats['mean'].astype(np.float64), stats['std'].astype(np.float64)
    for episode in ('train_ep', 'val_ep'):
        actual = c.check(dataset, [episode], mean, std)
        expected = namespace['check'](dataset, [episode], mean, std)
        for name in ('rows', 'offset_to_signal', 'ln_unit_std', 'per_dim_sd_median', 'mean_abs_median'):
            assert actual[name] == expected[name]


@pytest.mark.parametrize('change', ['depth', 'missing_depth', 'condition', 'task', 'overlap', 'empty_val', 'nonfinite'])
def test_checker_refuses_wrong_provenance_or_split(sandbox_repo, trainer, change):
    repo, _, _, c = sandbox_repo
    dataset, split = dataset_fixture(repo / 'data', trainer)
    if change in ('depth', 'missing_depth', 'condition', 'task', 'nonfinite'):
        with h5py.File(dataset, 'r+') as f:
            if change == 'depth':
                f.attrs['training_depth'] = 'registered_depth'
            elif change == 'missing_depth':
                del f.attrs['training_depth']
            elif change == 'condition':
                f.attrs['condition'] = 'mvp3d_multiview'
            elif change == 'task':
                f['episodes/train_ep/cam0'].attrs['task'] = 'mug-hanging'
            else:
                f['episodes/val_ep/cam0/features'][0, 0] = np.nan
    else:
        doc = json.loads(split.read_text())
        doc['validation'] = doc['train'] if change == 'overlap' else []
        split.write_text(json.dumps(doc))
    before = snapshot(repo)
    with pytest.raises(c.Refused):
        c.evaluate('drawer-open', 'mvp3d_single', dataset, split)
    assert snapshot(repo) == before


@pytest.mark.parametrize('offset,ln_std,passed', [(0.01, 0.5, True), (0.010001, 0.5, False),
                                                (0.0, 0.49999, False), (None, 1.0, False)])
def test_exact_thresholds_and_undefined_statistics_fail_closed(sandbox_repo, offset, ln_std, passed):
    _, _, _, c = sandbox_repo
    train = {'offset_to_signal': offset, 'ln_unit_std': ln_std}
    validation = {'offset_to_signal': 1.0, 'ln_unit_std': 0.5}
    assert c.verdict(train, validation)[1] is passed
    validation['offset_to_signal'] = 1.000001
    assert not c.verdict({'offset_to_signal': 0, 'ln_unit_std': 1}, validation)[1]


@pytest.mark.parametrize('last', [True, False])
def test_pass_lift_patch_narrows_only_one_cell_and_deactivates_last(sandbox_repo, last):
    repo, _, r, c = sandbox_repo
    record = plan(sandbox_repo)
    doc = json.loads((repo / r.PAUSES).read_bytes())
    pause = record['pause_entry']
    prefix = 'rwcal-20261110-fql-mvp3d_single-drawer-open-'
    if last:
        pause['prefixes'] = [prefix]
    doc['pauses'].append(pause)
    stats = {'offset_to_signal': 0.001, 'ln_unit_std': 0.8}
    proposal = c.lift_proposal('drawer-open', 'mvp3d_single', stats, stats)
    saved = copy.deepcopy(doc)
    writes = []

    # Evaluate the printed patch with an in-memory pause path/writer; no pause-file
    # write is performed by this test or by the checker.
    class MemoryPath:
        def __truediv__(self, name):
            return self

        def read_bytes(self):
            return r.json_bytes(doc)

    @contextmanager
    def memory_locks(root):
        yield

    context = {'ROOT': MemoryPath(), 'locks': memory_locks, 'require': r.require, 'json': json,
               'json_bytes': r.json_bytes, 'atomic_write': lambda path, raw: writes.append(json.loads(raw))}
    exec(proposal['python_patch'], context)
    assert len(writes) == 1
    narrowed = writes[0]['pauses'][-1]
    assert narrowed['prefixes'] == [p for p in saved['pauses'][-1]['prefixes'] if p != prefix]
    assert narrowed['active'] is (not last)
    assert narrowed['reason'] == saved['pauses'][-1]['reason'] + ' | ' + proposal['append_reason']
    assert writes[0]['pauses'][:-1] == saved['pauses'][:-1]


@pytest.mark.parametrize('symlink', [False, True])
def test_checker_out_cannot_overwrite_pause(sandbox_repo, symlink):
    repo, _, _, c = sandbox_repo
    out = repo / 'runs/setup/admission-pauses.json'
    if symlink:
        link = repo / 'looks-like-a-report.json'
        link.symlink_to(out)
        out = link
    before = snapshot(repo)
    with pytest.raises(c.Refused, match='--out must be a report'):
        c.main(['drawer-open', 'mvp3d_single', '--out', str(out)])
    assert snapshot(repo) == before
