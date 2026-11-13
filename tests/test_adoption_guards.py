"""CPU regressions for live admission evidence and read-only adoption audits."""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from ops.campaign.local_readiness import LocalReadiness
from ops.jobs import admission_pause, host_roles, scheduler
from ops.jobs.closure_impact import audit

ROOT = Path(__file__).resolve().parents[1]


def test_daemon_refreshes_pause_evidence_once_per_tick(tmp_path, monkeypatch):
    pause_file = tmp_path / 'pauses.json'
    monkeypatch.setattr(admission_pause, 'FILE', pause_file)
    gpu = scheduler.APPROVED_HOST_GPUS['local'][0]
    rows = [dict(id=i, script='scripts/train_rl.py', args=[], mem_gb=1, ram_gb=1, gpu=gpu)
            for i in ('first', 'second')]
    queue = tmp_path / 'queue.yaml'
    queue.write_text(yaml.safe_dump(dict(limits={gpu: dict(max_jobs=4, max_mem_gb=20)}, jobs=rows)))
    monkeypatch.setattr(host_roles, 'read_ownership', lambda: {})
    monkeypatch.setattr(host_roles, 'load_exceptions', lambda: {})
    monkeypatch.setattr(scheduler, 'source_identity', lambda: dict(commit='a' * 40, source_sha256={}))
    launched, reads = [], []
    monkeypatch.setattr(scheduler, 'launch', lambda job, *args: launched.append(job['id']) or 2**30)
    snapshot = admission_pause.snapshot
    monkeypatch.setattr(admission_pause, 'snapshot', lambda: reads.append(True) or snapshot())
    for index, held in enumerate(('first', 'second', None)):
        pause_file.write_text(json.dumps({'pauses': [dict(name='live', active=True, ids=[held])] if held else []}))
        launched.clear()
        scheduler.tick(queue, tmp_path / f'registry-{index}.jsonl', tmp_path / 'runs', usage={}, available_gb=200,
                       disk_bytes=(2000 * 2**30, 1000 * 2**30))
        assert launched == [j['id'] for j in rows if j['id'] != held]
        assert len(reads) == index + 1


def test_other_pause_consumers_see_additions_lifts_and_invalid_evidence(tmp_path, monkeypatch):
    path = tmp_path / 'pauses.json'
    monkeypatch.setattr(admission_pause, 'FILE', path)
    # This reference predates every file update, as in a long-lived imported consumer.
    paused = admission_pause.paused
    assert not paused('row-eval')
    path.write_text(json.dumps({'pauses': [dict(name='hold', active=True, prefixes=['row'])]}))
    assert paused('row-eval')
    path.write_text('{"pauses": []}')
    assert not paused('row-eval')
    path.write_text('invalid')
    with pytest.raises(ValueError):
        paused('row-eval')


def readiness_fixture(tmp_path):
    setup = tmp_path / 'runs/setup'
    setup.mkdir(parents=True)
    rows = [dict(id='original-rl', script='scripts/train_rl.py', stage=2, deps=['export']),
            dict(id='middle', script='scripts/train.py', stage=2, stage2_admission_rank='middle'),
            dict(id='later', script='scripts/train.py', stage=2, priority=2),
            dict(id='eval', script='scripts/evaluate.py', stage=2),
            dict(id='baseline', script='scripts/train_rl.py'),
            dict(id='baseline-eval', script='scripts/eval_rl.py'),
            dict(id='export', script='scripts/export_encoder.py', stage=2)]
    (setup / 'stage2-registration-20261013.json').write_text(json.dumps({'jobs': [rows[0], rows[-1]]}))
    return setup, rows


@pytest.mark.parametrize('gate', ['config', 'original_rl', 'middle_pretrain', 'proof', 'hold', 'frozen', 'companion', 'pin'])
def test_cross_driver_reservations_require_all_owning_driver_gates(tmp_path, gate):
    setup, rows = readiness_fixture(tmp_path)
    state = {}
    target = {'config': 'eval', 'original_rl': 'middle', 'middle_pretrain': 'later', 'proof': 'original-rl',
              'hold': 'export', 'frozen': 'baseline', 'companion': 'baseline-eval', 'pin': 'baseline'}[gate]
    if gate == 'hold':
        (setup / 'stage2-admission-hold.json').write_text(json.dumps(dict(
            allow=[], admit_until='2099-01-01T00:00:00+00:00', reason='test')))
    if gate == 'frozen':
        (setup / 'cnn-freeze-20261024.json').write_text('{"frozen": ["baseline"]}')
    if gate == 'pin':
        next(j for j in rows if j['id'] == target)['pin_host'] = 'remote'
    ready = LocalReadiness(dict(jobs=rows), state, tmp_path)
    job = next(j for j in rows if j['id'] == target)
    assert not ready(job)
    from ops.jobs.priority import PriorityReservations
    policy = PriorityReservations({'mig': dict(max_jobs=2, max_mem_gb=10)},
                                  dict(ram_gb=20, cpu_threads=20, disk_gb=20), lambda: {}, lambda: {})
    assert policy.reserve({**job, 'priority': -1, 'mem_gb': 1}, ready=ready(job)) is None
    assert not policy.reservations
    # Execute the real baseline external-row selection against the owning gates.
    import ast
    source = ast.parse((ROOT / 'ops/campaign/baseline_local_driver.py').read_text())
    selection = next(n for n in source.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'external' for t in n.targets))
    ns = dict(jobs=type('Jobs', (), {'eligible': staticmethod(lambda *a: [job])}),
              queue=dict(jobs=rows), _owner_ready=ready, by_id={}, paused=lambda _: False,
              host_roles=type('Roles', (), {'refusal': staticmethod(lambda *a: None)}),
              _roles={}, _ownership={}, _context={})
    exec(compile(ast.Module(body=[selection], type_ignores=[]), '<external selection>', 'exec'), ns)
    assert ns['external'] == []


def test_stage2_does_not_reintroduce_rejected_own_rows_as_external():
    import ast
    source = ast.parse((ROOT / 'ops/campaign/stage2_local_driver.py').read_text())
    start = next(i for i, n in enumerate(source.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '_own' for t in n.targets))
    bad = dict(id='missing-eval-config', stage=2, script='scripts/evaluate.py')
    ns = dict(stage2=[bad], plain_extra=[], candidates=[], queue=dict(jobs=[bad]), eligible_view={},
              jobs=type('Jobs', (), {'eligible': staticmethod(lambda *a: [bad])}), paused=lambda _: False,
              _owner_ready=lambda _: False)
    exec(compile(ast.Module(body=source.body[start:start + 2], type_ignores=[]), '<selection>', 'exec'), ns)
    assert ns['_other_ready'] == []


def test_closure_audit_detects_changed_imports_and_recorded_argument_files_read_only(tmp_path):
    repo, live = tmp_path / 'repo', tmp_path / 'live'
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    git('init', '-q')
    (repo / 'scripts').mkdir()
    (repo / 'scripts/train.py').write_text('import old\n')
    (repo / 'scripts/old.py').write_text('value = 1\n')
    (repo / 'science.yaml').write_text('fixed: true\n')
    git('add', '.')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.org', 'commit', '-qm', 'source')
    commit = git('rev-parse', 'HEAD')
    scope = dict(source_commit=commit, files={
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in ('scripts/train.py', 'scripts/old.py', 'science.yaml')})
    (live / 'experiments').mkdir(parents=True)
    (live / 'runs/setup/ownership').mkdir(parents=True)
    (live / 'experiments/queue.yaml').write_text(yaml.safe_dump(dict(jobs=[dict(id='placed', script='scripts/train.py')])))
    (live / 'experiments/registry.jsonl').write_text('')
    record = live / 'runs/setup/ownership/placed.json'
    record.write_text(json.dumps(dict(execution_scope=scope)))
    assert audit(repo, live, base=commit)['result'] == 'no existing row affected'
    (repo / 'scripts/train.py').write_text('import new\n')
    (repo / 'scripts/new.py').write_text('value = 2\n')
    (repo / 'science.yaml').write_text('fixed: false\n')
    git('add', '.')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.org', 'commit', '-qm', 'adoption')
    before = {p: p.read_bytes() for p in live.rglob('*') if p.is_file()}
    report = audit(repo, live, base=commit)
    assert report['affected_rows'] == 1
    assert set(report['rows'][0]['differences']) == {'scripts/train.py', 'scripts/new.py', 'science.yaml'}
    assert before == {p: p.read_bytes() for p in live.rglob('*') if p.is_file()}
    assert git('status', '--porcelain') == ''

    # Separately preserved files are excluded only while byte-identical to that commit.
    preserved = git('rev-parse', 'HEAD')
    assert audit(repo, live, base=commit, preserved_commit=preserved)['affected_rows'] == 0
    (repo / 'science.yaml').write_text('changed again: true\n')
    git('add', '.')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.org', 'commit', '-qm', 'changed protected file')
    with pytest.raises(ValueError, match='preserved'):
        audit(repo, live, base=commit, preserved_commit=preserved)


def test_remote_closeout_always_requires_continuation_metadata():
    state = dict(script='scripts/train_ours_continuation.py', required_results=['metrics.jsonl'])
    assert {'completion.json', 'continuation.json'} <= set(scheduler.remote_check_names(state))


def test_scheduler_withholds_inadmissible_stage2_before_launch_or_reservation(tmp_path, monkeypatch):
    pause_file = tmp_path / 'pauses.json'
    pause_file.write_text('{"pauses": []}')
    monkeypatch.setattr(admission_pause, 'FILE', pause_file)
    monkeypatch.setattr(scheduler, 'REPO', tmp_path)
    gpu = scheduler.APPROVED_HOST_GPUS['local'][0]
    rows = [dict(id='missing-config', stage=2, priority=-1, script='scripts/evaluate.py'),
            dict(id='low', priority=2, script='scripts/train_rl.py')]
    rows = [dict(args=[], mem_gb=1, ram_gb=1, gpu=gpu, **r) for r in rows]
    queue = tmp_path / 'queue.yaml'
    queue.write_text(yaml.safe_dump(dict(limits={gpu: dict(max_jobs=4, max_mem_gb=20)}, jobs=rows)))
    monkeypatch.setattr(host_roles, 'read_ownership', lambda: {})
    monkeypatch.setattr(host_roles, 'load_exceptions', lambda: {})
    monkeypatch.setattr(scheduler, 'source_identity', lambda: dict(commit='a' * 40, source_sha256={}))
    launched = []
    monkeypatch.setattr(scheduler, 'launch', lambda job, *args: launched.append(job['id']) or 2**30)
    messages = scheduler.tick(queue, tmp_path / 'registry.jsonl', tmp_path / 'runs', usage={}, available_gb=200,
                              disk_bytes=(2000 * 2**30, 1000 * 2**30))
    assert launched == ['low']
    assert any('missing-config: not ready under owning driver gates' in m for m in messages)
