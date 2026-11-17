"""Refusal-only deferrals and a serialized multi-batch launch cap."""
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops.jobs import release_reservation
from ops.jobs import rwprep_remote as R
from ops.jobs.rwprep_mig import atomic, digest
from ops.resources.gpu_guard import REMOTE_MIG_DEVICES


@pytest.fixture
def route(tmp_path, monkeypatch):
    mirrors = []
    def bundle(name, statuses=('pending', 'pending')):
        p = tmp_path/name
        jobs = [dict(id=f'rwv2-20261113-prep-{name}-{i}', gpu_profile='1g.24gb',
                     gpu_gib=6, ram_gib=16, cpus=4, inputs={'/raw': 'a'*64},
                     steps=[{'kind': 'rwprep', 'argv': []}]) for i in range(len(statuses))]
        config = dict(host='remote', image='sha256:'+'a'*64, commit='b'*40,
                      jobs=str(p/'jobs.json'), state_dir=str(p/'state'),
                      priority_file=str(p/'priority.json'), remote='/remote/'+name)
        atomic(config['jobs'], jobs)
        states = {j['id']: dict(status=status, attempt=0 if status == 'pending' else 3)
                  for j, status in zip(jobs, statuses)}
        atomic(p/'state/state.json', dict(queue_sha256=digest(config['jobs']), jobs=states))
        return config, jobs
    def ssh(command, **kwargs):
        payload = json.loads(shlex.split(command)[-1])
        if isinstance(payload, list):
            return SimpleNamespace(stdout=json.dumps([j['id'] for j in payload]))
        mirrors.append(payload)
        return SimpleNamespace(stdout='')
    parallel = SimpleNamespace(load=lambda *_: ({}, ''), snapshot=lambda _: {},
                               decide=lambda *a, **kw: {'candidates': list(REMOTE_MIG_DEVICES), 'host': []})
    monkeypatch.setattr(R, 'established', lambda: (None, parallel))
    monkeypatch.setattr(R, 'remote_host_policy', lambda *_: {'cpu_thread_budget': 320})
    monkeypatch.setattr(release_reservation, 'reserved_gpus', lambda: set())
    return bundle, SimpleNamespace(checked_ssh=ssh), mirrors


def state(config):
    return json.loads((Path(config['state_dir'])/'state.json').read_text())


def policy(tmp_path, configs, cap=1, per_tick=1):
    p = tmp_path/'launch-policy.json'
    atomic(p, dict(schema='rwprep-launch-policy/1', lock=str(tmp_path/'fleet.lock'),
                   max_active=cap, max_launch_per_tick=per_tick,
                   state_dirs=[c['state_dir'] for c in configs]))
    for c in configs:
        c['launch_policy'] = str(p)
    return p


@pytest.mark.parametrize('status', ['pending', 'retry'])
@pytest.mark.parametrize('reason', sorted(R.RESOURCE_REFUSALS))
def test_confirmed_refusal_preserves_attempt_history_and_mirror(route, monkeypatch, status, reason):
    bundle, client, mirrors = route
    config, jobs = bundle('batch', (status,))
    before = state(config)
    previous = before['jobs'][jobs[0]['id']]
    previous.update(exit_code=1, operator_recovery='/operator-proof')
    atomic(Path(config['state_dir'])/'state.json', before)
    attempts = []
    def refuse(_client, _cfg, _job, intent):
        attempts.append(intent['attempt'])
        raise SystemExit('admission refused (locked guard): '+repr([reason]))
    monkeypatch.setattr(R, 'launch_job', refuse)
    for count in range(1, 4):
        result = R.tick(config, client, apply=True)
        saved = state(config)
        row = saved['jobs'][jobs[0]['id']]
        assert {k: v for k, v in row.items() if k != 'admission_deferral'} == previous
        assert row['admission_deferral']['count'] == count
        assert row['admission_deferral']['reasons'] == [reason]
        assert result['counts'][status] == 1 and not result['counts']['failed']
        assert mirrors[-1] == saved
    assert attempts == [previous['attempt']+1]*3
    monkeypatch.setattr(R, 'launch_job', lambda *_: None)
    R.tick(config, client, apply=True)
    assert state(config)['jobs'][jobs[0]['id']]['attempt'] == previous['attempt']+1
    assert state(config)['jobs'][jobs[0]['id']]['status'] == 'running'


@pytest.mark.parametrize('exc', [
    RuntimeError("admission refused (locked guard): ['disk floor']"),
    SystemExit("admission refused (locked guard): ['container name already exists']"),
    SystemExit("admission refused (locked guard): ['disk floor', 'input changed']"),
    SystemExit("admission refused (locked guard): ['disk floor'"),
    SystemExit("admission refused (locked guard): 'disk floor'"),
    SystemExit("admission refused (locked guard): []"),
    SystemExit("admission refused (locked guard): [{}]"),
    SystemExit('admission refused: scheduler/rwprep launch lock busy for 120 s'),
    RuntimeError('lost SSH reply'), ValueError('dirty source'),
])
def test_ambiguous_or_nonresource_errors_keep_intent(route, monkeypatch, exc):
    bundle, client, _ = route
    config, jobs = bundle('batch', ('retry',))
    def fail(*_):
        raise exc
    monkeypatch.setattr(R, 'launch_job', fail)
    R.tick(config, client, apply=True)
    row = state(config)['jobs'][jobs[0]['id']]
    assert row['status'] == 'launching' and row['attempt'] == 4
    assert row['launch_error'] == type(exc).__name__
    R.reconcile({jobs[0]['id']: row}, {row['container']: None}, config, '', row['deadline']+299)
    assert row['status'] == 'launching'
    R.reconcile({jobs[0]['id']: row}, {row['container']: None}, config, '', row['deadline']+301)
    assert row['status'] == 'failed'


def test_controllers_share_lock_cap_and_continue_reconciliation(route, monkeypatch, tmp_path):
    bundle, client, mirrors = route
    first, jobs = bundle('first')
    second, _ = bundle('second')
    policy(tmp_path, [first, second])
    nested = []
    def launch(*_):
        nested.append(R.tick(second, client, apply=True))
    monkeypatch.setattr(R, 'launch_job', launch)
    result = R.tick(first, client, apply=True)
    assert len(result['launched']) == 1
    assert nested[0]['blocked'] == {'launch_gate': 'route launch lock busy'}
    assert all(v['attempt'] == 0 for v in state(second)['jobs'].values())
    assert R.tick(second, client, apply=True)['blocked'] == {'launch_gate': 'route active cap'}
    row = state(first)['jobs'][jobs[0]['id']]
    item = dict(labels={'s4d.commit': first['commit'], 's4d.rwprep.job': jobs[0]['id'],
                        's4d.gpu': row['gpu'], 's4d.role': 'rwprep', 's4d.owner': 'kaist-sunho'},
                state={'Running': False, 'ExitCode': 0},
                proof={'status': 'ok', 'queue_sha256': digest(first['jobs'])})
    monkeypatch.setattr(R, 'remote_observe', lambda *_: {row['container']: item})
    Path(first['state_dir'], 'STOP').touch()
    drained = R.tick(first, client, apply=True)
    assert drained['counts']['done'] == 1 and not drained['launched']
    assert mirrors[-1]['jobs'][jobs[0]['id']]['status'] == 'done'
    monkeypatch.setattr(R, 'launch_job', lambda *_: None)
    assert len(R.tick(second, client, apply=True)['launched']) == 1


@pytest.mark.parametrize('status', ['launching', 'running', 'unknown'])
def test_cap_counts_unreconciled_intents(route, tmp_path, status):
    bundle, _, _ = route
    config, _ = bundle('batch')
    other, _ = bundle('other', (status,))
    policy(tmp_path, [config, other])
    with R.launch_gate(config) as gate:
        assert gate['active'] == 1


def test_policy_fails_closed_on_corrupt_member(route, tmp_path):
    bundle, _, _ = route
    config, _ = bundle('batch')
    other, _ = bundle('other')
    policy(tmp_path, [config, other])
    (Path(other['state_dir'])/'state.json').write_text('{}')
    with pytest.raises(KeyError), R.launch_gate(config):
        pytest.fail('unreadable fleet member admitted')


@pytest.mark.parametrize('cap,per_tick,expected', [(8, 1, 1), (1, 8, 1), (8, 8, 2)])
def test_per_tick_and_fleet_limits_both_apply(route, monkeypatch, tmp_path, cap, per_tick, expected):
    bundle, client, _ = route
    config, _ = bundle('batch')
    policy(tmp_path, [config], cap=cap, per_tick=per_tick)
    monkeypatch.setattr(R, 'launch_job', lambda *_: None)
    assert len(R.tick(config, client, apply=True)['launched']) == expected


@pytest.mark.parametrize('field,value', [
    ('max_active', 0), ('max_active', True), ('max_launch_per_tick', -1),
    ('max_launch_per_tick', True), ('state_dirs', []), ('state_dirs', ['relative']),
    ('lock', 'relative'), ('schema', 'unknown'),
])
def test_invalid_policy_does_not_launch(route, monkeypatch, tmp_path, field, value):
    bundle, client, _ = route
    config, _ = bundle('batch')
    p = policy(tmp_path, [config])
    doc = json.loads(p.read_text())
    doc[field] = value
    atomic(p, doc)
    monkeypatch.setattr(R, 'launch_job', lambda *_: pytest.fail('invalid policy launched'))
    with pytest.raises(ValueError):
        R.tick(config, client, apply=True)
    assert all(v['attempt'] == 0 for v in state(config)['jobs'].values())
