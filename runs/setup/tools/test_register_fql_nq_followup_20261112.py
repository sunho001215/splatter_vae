"""Synthetic checks of the FQLNQ follow-up registrar (SNeRL + ours rows): tmp repos only, no live writes."""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

SPEC = importlib.util.spec_from_file_location('fqlnq_followup', Path(__file__).with_name(
    'register_fql_nq_followup_20261112.py'))
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
COMMIT, IMAGE = '7' * 40, 'sha256:' + '6' * 64
INPUTS = '/home/compu/kaist/sunho/realworld/learning_inputs/calrel-6c3b9ec/{task}'
PAUSES = {'rwnq-20261111-fql-snerl-': [R.SNERL_HOLD], 'rwa100-20261111-fql-snerl-': [R.SNERL_HOLD, R.A100_PAUSE],
          'rwnq-20261111-fql-ours-': [R.OURS_HOLD], 'rwa100-20261111-fql-ours-': [R.OURS_HOLD, R.A100_PAUSE]}


def fql_row(cond, task, seed, prefix='rwcal-20261109'):
    job, dep, inputs = f'{prefix}-fql-{cond}-{task}-s{seed}', f'{prefix}-fqldata-{cond}-{task}', INPUTS.format(task=task)
    return {'id': job, 'script': 'scripts/train_realworld_fql.py', 'args': [
        '--config', 'configs/realworld/fql/base.yaml', '--name', job, '--output-root', 'runs', '--set',
        f'condition={cond}', f'data.dataset=runs/{dep}/dataset.h5', f'data.split_manifest={inputs}/split_seed0.json',
        f'data.gate_report={inputs}/gate_report.json', f'train.seed={seed}', 'agent.alpha=10.0', 'agent.batch_size=256',
        'train.offline_steps=1000000', 'train.log_interval=5000', 'train.eval_interval=100000'],
        'role': 'realworld_fql', 'host': 'remote', 'gpu': 'any', 'mem_gb': 6, 'ram_gb': 12, 'priority': 2,
        'deps': [dep], 'max_restarts': 2, 'oom_score_adj': 0, 'realworld_target': task, 'fql_condition': cond,
        'fql_seed': seed, 'fql_budget': {'offline_steps': 1000000, 'batch_size': 256, 'alpha': 10.0,
                                         'log_interval': 5000, 'eval_interval': 100000},
        'realworld_consumer': {'snerl': 'snerl', 'sincro': 'fql_sincro'}[cond], 'calibration_dependent': True,
        'gate_level': 'geometry', 'realworld_gate_sha256': 'a' * 64, 'planned_commit': '6c' * 20,
        'remote_host': '192.168.10.22', 'pin_host': 'remote', 'planned_image': 'sha256:' + 'f' * 64,
        'evidence_manifest': 'runs/remote/acceptance/6c6c6c6-cc/manifest.json', 'disk_gb': 10,
        'realworld_manifest_sha256': 'b' * 64, 'realworld_inputs': inputs, 'calibration_release': 'calrel',
        'registration_tool': 'runs/setup/tools/old.py', 'plan_row_id': f'rw-fql-{cond}-{task}-s{seed}',
        'required_results': ['policy.pt', 'metrics.jsonl', 'completion.json']}


def data_row(cond, task, export):
    job, inputs = f'rwcal-20261109-fqldata-{cond}-{task}', INPUTS.format(task=task)
    return {'id': job, 'script': 'scripts/realworld_fql_dataset.py', 'args': [
        'build', '--store', f'{inputs}/store.yaml', '--condition', cond, '--out', f'runs/{job}/dataset.h5',
        '--gate-report', f'{inputs}/gate_report.json', '--outcomes', f'{inputs}/outcomes.json', '--export', export,
        '--device', 'cuda'], 'role': 'realworld_fql', 'host': 'remote', 'gpu': 'any', 'mem_gb': 8, 'ram_gb': 16,
        'priority': 2, 'deps': [export.split('/')[1]], 'max_restarts': 1, 'oom_score_adj': 0,
        'realworld_target': task, 'fql_condition': cond, 'realworld_consumer': f'fql_{cond}',
        'calibration_dependent': True, 'gate_level': 'geometry', 'realworld_gate_sha256': 'a' * 64,
        'planned_commit': '6c' * 20, 'remote_host': '192.168.10.22', 'pin_host': 'remote',
        'planned_image': 'sha256:' + 'f' * 64, 'evidence_manifest': 'x.json', 'disk_gb': 20,
        'realworld_manifest_sha256': 'b' * 64, 'realworld_inputs': inputs, 'calibration_release': 'calrel',
        'registration_tool': 'old.py', 'plan_row_id': f'rw-fqldata-{cond}-{task}', 'required_results': ['dataset.h5']}


def queue_rows():
    rows = []
    for task in R.TASKS:
        pre, export = f'rwcal-20261111-ours-{task}-pre', f'rwcal-20261111-ours-{task}-export'
        rows += [{'id': pre, 'script': 'scripts/train.py', 'args': []},
                 {'id': export, 'script': 'scripts/export_encoder.py', 'args': [], 'deps': [pre],
                  'realworld_consumer': 'ours', 'realworld_target': task, 'realworld_inputs': INPUTS.format(task=task),
                  'realworld_gate_sha256': 'c' * 64, 'realworld_manifest_sha256': 'd' * 64},
                 data_row('sincro', task, f'runs/rwcal-20261109-sincro-{task}-pre/encoder.pt'),
                 data_row('snerl', task, f'runs/rwcal-20261109-snerl-{task}-pre/encoder.pt')]
        rows += [fql_row(c, task, s) for c in ('sincro', 'snerl') for s in R.SEEDS]
    rows.append({'id': 'rwng-20261107b-fql-e2e_cnn-drawer-open-s0', 'script': 'x', 'args': []})  # untouched
    return rows


def git_release(root, with_ours=True):
    """A tiny git repository whose commit has (or lacks) the ours FQL condition."""
    for rel, text in (('s4d/rwlearn/observation.py', '"ours": Condition("splatter4d", 1, ...)' if with_ours else ''),
                      ('s4d/rwlearn/gate.py', '"ours": "fql_ours"' if with_ours else '')):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    env = ['-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false']
    subprocess.run(['git', '-C', str(root), 'init', '-q'], check=True)
    subprocess.run(['git', '-C', str(root), 'add', 's4d'], check=True)
    subprocess.run(['git', '-C', str(root), *env, 'commit', '-q', '-m', 'r'], check=True)
    return subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'execution_allowed', lambda root: None)
    root = tmp_path / 'repo'
    (root / 'experiments').mkdir(parents=True)
    (root / 'runs/setup').mkdir(parents=True)
    commit = git_release(root)
    (root / R.QUEUE).write_text('# comment\njobs:\n' + yaml.safe_dump(queue_rows(), sort_keys=False))
    (root / R.REGISTRY).write_text('')
    (root / 'runs/setup/role-release.json').write_text(json.dumps({'commit': commit, 'image': IMAGE}))
    manifest = root / f'runs/remote/acceptance/{commit[:7]}-cc/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    return root


def commit_of(root):
    return json.loads((root / 'runs/setup/role-release.json').read_text())['commit']


def pauses(active=True, skip=()):
    def active_pauses(job_id, *, fresh=False):
        assert fresh
        return [] if not active or job_id in skip else next((v for k, v in PAUSES.items() if job_id.startswith(k)), [])
    return active_pauses


def prepare(root, group, **kw):
    return R.prepare(root, group, commit_of(root), IMAGE, active_pauses=kw.pop('active_pauses', pauses()), **kw)


def test_snerl_rows_equal_their_paused_alpha10_sources(repo):
    record = prepare(repo, 'snerl')
    by_id = {r['id']: r for r in yaml.safe_load((repo / R.QUEUE).read_text())['jobs']}
    assert len(record['jobs']) == 12 and record['pauses_active_now'] == '12/12'
    for row in record['jobs']:
        source = by_id[row['derived_from']]
        assert source['id'] == f"rwcal-20261109-fql-snerl-{row['realworld_target']}-s{row['fql_seed']}"
        alpha, flag, prio = ('10.0', 'true', 2) if row['fql_setting'] == 'a10nq' else ('100.0', 'false', R.A100_PRIORITY)
        want = [row['id'] if a == source['id'] else ('agent.alpha=' + alpha if a.startswith('agent.alpha=') else a)
                for a in source['args']] + ['agent.normalize_q_loss=' + flag]
        assert row['args'] == want and row['priority'] == prio and row['deps'] == source['deps']
        assert R.A100_PRIORITY == 4  # the lowest priority used in the queue (user 2026-11-12 ~00:20)
        same = set(source) - R.P1.CHANGED
        assert {k: row[k] for k in same} == {k: source[k] for k in same}
        assert R.required_pauses(row) == PAUSES[row['id'].rsplit('-snerl-', 1)[0] + '-snerl-']


def test_ours_rows_follow_the_sincro_protocol_template(repo):
    record = prepare(repo, 'ours')
    rows = {r['id']: r for r in record['jobs']}
    by_id = {r['id']: r for r in yaml.safe_load((repo / R.QUEUE).read_text())['jobs']}
    assert len(rows) == 14 and record['pauses_active_now'] == '12/12'
    for task in R.TASKS:
        data, template = rows[f'rwcal-20261111-fqldata-ours-{task}'], by_id[f'rwcal-20261109-fqldata-sincro-{task}']
        assert data['deps'] == [f'rwcal-20261111-ours-{task}-export'] and R.required_pauses(data) == []
        opts = dict(zip(data['args'][1::2], data['args'][2::2]))
        assert opts['--condition'] == 'ours' and opts['--out'] == f'runs/rwcal-20261111-fqldata-ours-{task}/dataset.h5'
        assert opts['--export'] == f'runs/rwcal-20261111-ours-{task}-export/encoder.pt'
        assert opts['--store'] == INPUTS.format(task=task) + '/store.yaml'
        assert (data['mem_gb'], data['ram_gb'], data['disk_gb']) == (template['mem_gb'], template['ram_gb'], 20)
        assert data['realworld_consumer'] == 'fql_ours' and data['realworld_gate_sha256'] == 'c' * 64
        for setting, prefix in (('a10nq', 'rwnq'), ('a100', 'rwa100')):
            for seed in R.SEEDS:
                row = rows[f'{prefix}-20261111-fql-ours-{task}-s{seed}']
                tmpl = by_id[f'rwcal-20261109-fql-sincro-{task}-s{seed}']
                assert row['protocol_template'] == tmpl['id'] and 'derived_from' not in row
                assert row['deps'] == [data['id']] and row['fql_condition'] == 'ours'
                assert 'condition=ours' in row['args'] and f'data.dataset=runs/{data["id"]}/dataset.h5' in row['args']
                assert [a for a in row['args'] if a.startswith(('data.split', 'data.gate', 'train.'))] == \
                    [a for a in tmpl['args'] if a.startswith(('data.split', 'data.gate', 'train.'))]
                assert row['args'][-1] == f"agent.normalize_q_loss={'true' if setting == 'a10nq' else 'false'}"
                assert row['priority'] == (2 if setting == 'a10nq' else R.A100_PRIORITY) and row['fql_setting'] == setting
                assert (row['mem_gb'], row['ram_gb'], row['disk_gb']) == (tmpl['mem_gb'], tmpl['ram_gb'], tmpl['disk_gb'])


def test_ours_refused_without_a_release_that_defines_it(tmp_path, repo):
    old = tmp_path / 'old'
    old.mkdir()
    commit = git_release(old, with_ours=False)
    with pytest.raises(R.Refused, match='has no FQL condition ours'):
        R.release_has_ours(old, commit)
    R.release_has_ours(repo, commit_of(repo))


def test_ours_template_inputs_must_be_the_ours_inputs(repo):
    doc = yaml.safe_load((repo / R.QUEUE).read_text())
    for row in doc['jobs']:
        if row['id'] == 'rwcal-20261111-ours-mug-hanging-export':
            row['realworld_inputs'] = '/other/mug-hanging'
    (repo / R.QUEUE).write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(R.Refused, match='other inputs than ours'):
        prepare(repo, 'ours')


@pytest.mark.parametrize('group', R.GROUPS)
def test_publication_needs_the_holds_and_preview_never_publishes(repo, group):
    record = prepare(repo, group)
    before = (repo / R.QUEUE).read_bytes()
    with pytest.raises(R.Refused, match='lack their active pauses'):
        R.execute(repo, copy.deepcopy(record), active_pauses=pauses(skip=(record['jobs'][-1]['id'],)))
    assert (repo / R.QUEUE).read_bytes() == before
    preview = R.prepare(repo, group, commit_of(repo), IMAGE, active_pauses=pauses(), preview=True)
    with pytest.raises(R.Refused, match='never published'):
        R.execute(repo, preview, active_pauses=pauses())
    R.execute(repo, record, active_pauses=pauses())
    queue = yaml.safe_load((repo / R.QUEUE).read_text())
    old = yaml.safe_load(before)
    assert queue['jobs'][:len(old['jobs'])] == old['jobs'] and queue['jobs'][len(old['jobs']):] == record['jobs']
    assert json.loads((repo / f'runs/setup/{R.STEM.format(group=group)}.json').read_text())['registered'] is True
    with pytest.raises(R.Refused, match='pending transaction/proof exists'):
        prepare(repo, group)


def test_cli_guards(repo, monkeypatch):
    with pytest.raises(R.Refused, match='only for dry runs'):
        R.main(['--repo', str(repo), '--group', 'ours', '--commit', commit_of(repo), '--image', IMAGE, '--preview',
                '--execute'])
    monkeypatch.undo()
    with pytest.raises(R.Refused, match="coordinator's step"):
        R.main(['--repo', str(repo), '--group', 'snerl', '--commit', commit_of(repo), '--image', IMAGE, '--execute'])
    with pytest.raises(R.Refused, match="coordinator's step"):
        R.execute(repo, {'group': 'snerl'})


def test_live_queue_snerl_rows_derive_cleanly():
    live = R.MAIN / R.QUEUE
    if not live.is_file():
        pytest.skip('no live queue')
    by_id = {r['id']: r for r in R.P1.queue_document(live.read_bytes())['jobs']}
    rows = R.snerl_rows(by_id, {'commit': COMMIT, 'image': IMAGE, 'evidence_manifest': 'x.json'})
    assert len(rows) == 12 and all(r['derived_from'].startswith('rwcal-20261109-fql-snerl-') for r in rows)
