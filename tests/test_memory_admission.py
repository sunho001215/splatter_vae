"""Local staging reserve and current-queue growth policy; never import campaign drivers."""
import ast
import json
import sys
from pathlib import Path

import pytest

from ops.jobs.footprints import local_footprint
from ops.resources import memory_budget as mb
from ops.resources import memory_watch as watch


@pytest.fixture(autouse=True)
def memory_control(tmp_path, monkeypatch):
    control = tmp_path / 'local-memory-budget.json'
    control.write_text(json.dumps({'ceiling_bytes': 400_000_000_000}))
    monkeypatch.setattr(mb, 'CONTROL', control)
    return control


@pytest.mark.parametrize('mode,usage,growth,extra,host,expected', [
    ('fits', 64e9, 30e9, 294e9, 500e9, True),
    ('fits', 64e9, 30e9, 294e9 + 1, 500e9, False),
    ('fits-staging', 388e9, 100e9, 2e9, 500e9, True),
    ('fits-staging', 388e9 + 1, 0, 2e9, 500e9, False),
    ('fits-staging', 64e9, 100e9, 2e9 + 1, 500e9, False),
    ('fits-staging', 64e9, 100e9, 2e9, mb.HOST_RESERVE + 2e9 - 1, False),
    ('fits-staging', 64e9, 100e9, 2e9, mb.HOST_RESERVE + 2e9, True),
    ('fits', 64e9, 0, 2e9, mb.HOST_RESERVE + 2e9 - 1, False),
    ('fits-staging', 1e9, 0, -1, 500e9, False),
])
def test_modes(monkeypatch, capsys, mode, usage, growth, extra, host, expected):
    monkeypatch.setattr(mb, 'snapshot', lambda: dict(budget_usage=int(usage), host_mem_available=int(host),
                                                   pressure={}, io_pressure={}))
    monkeypatch.setattr(watch, 'running_local', lambda: {})
    def remaining(_):
        assert mode == 'fits', 'staging must never query training growth'
        return int(growth)
    monkeypatch.setattr(watch, 'remaining_growth', remaining)
    monkeypatch.setattr(sys, 'argv', ['memory_watch', mode, str(int(extra))])
    assert watch.main() == (0 if expected else 1)
    data = json.loads(capsys.readouterr().out)
    assert data['fits'] is expected and data['mode'] == mode
    assert data['staging_reserve'] == 2_000_000_000 and data['ceiling'] == 400_000_000_000
    assert data['remaining_growth'] == (0 if mode == 'fits-staging' else growth)


def test_current_queue_growth_matches_admitting_driver(monkeypatch):
    queue = {
        'frozen': dict(script='scripts/train_realworld_fql.py', fql_condition='ours', ram_gb=4),
        'cnn': dict(script='scripts/train_realworld_fql.py', fql_condition='e2e_cnn', ram_gb=8),
        'declared': dict(script='scripts/train_snerl.py', ram_gb=4),
        'drm': dict(script='scripts/train_rl.py', args=[], ram_gb=24),
        'replay': dict(script='scripts/train_rl.py', args=['--encoder', 'cnn'], ram_gb=48),
        'sac': dict(script='scripts/train_sac.py', ram_gb=24),
    }
    states = {key: dict(pid=i + 100, ram_gb=24) for i, key in enumerate(queue)}
    pss = {i + 100: 2**30 for i in range(len(queue))}
    monkeypatch.setattr(mb, 'session_nonfile_pss', lambda: pss)
    source = Path(__file__).resolve().parents[1] / 'ops/campaign/baseline_local_driver.py'
    tree = ast.parse(source.read_text())
    names = {'by_queue': queue}
    for name in ('footprint', 'active_footprint'):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), names)
    expected = sum(max(0, names['active_footprint'](key, states[key]) - pss[states[key]['pid']]) for key in queue)
    assert watch.remaining_growth({key: (states[key], job) for key, job in queue.items()}) == expected
    assert local_footprint(queue['declared']) == 4 * 2**30
    assert local_footprint(queue['frozen']) == 3 * 2**30
    assert local_footprint(queue['cnn']) == 8 * 2**30
    assert local_footprint(queue['drm']) == 3.5e9
    assert watch.remaining_growth({'orphan': ({'ram_gb': 4}, {})}) == 4 * 2**30


@pytest.mark.parametrize('filename', ['baseline_local_driver.py', 'stage2_local_driver.py'])
def test_driver_cap_arithmetic(filename):
    source = Path(__file__).resolve().parents[1] / 'ops/campaign' / filename
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == 'CAP' for t in n.targets))
    names = {'mb': mb, 'VALIDATION_RESERVATION': 7_000_000_000}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), names)
    assert mb.CEILING == 400_000_000_000
    assert mb.TRAINING_CEILING == 388_000_000_000
    assert names['CAP'] == 381_000_000_000


def test_realworld_capacity_withholds_staging(monkeypatch):
    from ops.jobs import realworld_local
    monkeypatch.setattr(mb, 'snapshot', lambda: {'budget_usage': 64_000_000_000})
    monkeypatch.setattr(mb, 'session_nonfile_pss', lambda: {})
    assert realworld_local._capacity({'jobs': []}, {}) == {'ram_free_bytes': 324_000_000_000}


@pytest.mark.parametrize('job,recorded_ram,footprint', [
    ({'script': 'scripts/train_snerl.py', 'ram_gb': 4}, 24, 4 * 2**30),
    ({'script': 'scripts/train_snerl.py', 'ram_gb': 24}, 4, 24 * 2**30),
    ({'script': 'scripts/train_realworld_fql.py', 'fql_condition': 'ours', 'ram_gb': 24}, 24, 3 * 2**30),
    ({'script': 'scripts/train_realworld_fql.py', 'fql_condition': 'e2e_cnn', 'ram_gb': 24}, 24, 8 * 2**30),
    ({'script': 'scripts/train_rl.py', 'ram_gb': 24}, 24, int(3.5e9)),
    ({}, 4, 4 * 2**30),
])
def test_realworld_capacity_matches_current_queue_growth(monkeypatch, job, recorded_ram, footprint):
    from ops.jobs import realworld_local

    monkeypatch.setattr(mb, 'snapshot', lambda: {'budget_usage': 64_000_000_000})
    monkeypatch.setattr(mb, 'session_nonfile_pss', lambda: {123: 2**30})
    current = {'host': 'local', 'status': 'running', 'pid': 123, 'ram_gb': recorded_ram}
    queue = {'jobs': [{'id': 'active', **job}] if job else []}
    growth = watch.remaining_growth({'active': (current, job)})
    assert growth == footprint - 2**30
    assert realworld_local._capacity(queue, {'active': current}) == {
        'ram_free_bytes': mb.TRAINING_CEILING - 64_000_000_000 - growth}


@pytest.mark.parametrize('contents', [None, '', '{', 'null', '[]', '400000000000',
    '{"ceiling_bytes": true}', '{"ceiling_bytes": "400000000000"}',
    '{"ceiling_bytes": 400000000000.0}', '{"ceiling_bytes": 149999999999}',
    '{"ceiling_bytes": 400000000001}', '{"ceiling_bytes": -1}',
    '{"ceiling_bytes": 400000000000, "unexpected": 1}', '{}'])
def test_invalid_control_fails_safe(memory_control, contents):
    if contents is None:
        memory_control.unlink()
    else:
        memory_control.write_text(contents)
    assert mb.CEILING == 150_000_000_000
    assert mb.TRAINING_CEILING == 138_000_000_000


@pytest.mark.parametrize('ceiling', [150_000_000_000, 225_000_000_000, 400_000_000_000])
def test_control_bounds(memory_control, ceiling):
    memory_control.write_text(json.dumps({'ceiling_bytes': ceiling}))
    assert mb.CEILING == ceiling
    assert mb.TRAINING_CEILING == ceiling - 12_000_000_000
    assert mb.HOST_RESERVE == 60 * 2**30


def test_unreadable_control_fails_safe(memory_control, monkeypatch):
    def refused(*args, **kwargs):
        raise PermissionError('unreadable')
    monkeypatch.setattr(Path, 'read_text', refused)
    assert mb.CEILING == 150_000_000_000


def test_alert_rereads_control(memory_control, monkeypatch, capsys):
    monkeypatch.setattr(mb, 'snapshot', lambda: dict(budget_usage=200_000_000_000,
        host_mem_available=500_000_000_000, current=200_000_000_000, pressure={}, io_pressure={}))
    monkeypatch.setattr(sys, 'argv', ['memory_watch', 'alert'])
    assert watch.main() == 1
    memory_control.write_text('{"ceiling_bytes": 150000000000}')
    assert watch.main() == 0
    assert 'MEMORY_ALERT' in capsys.readouterr().out


def test_fe462a1_execution_scope_excludes_scheduler_policy():
    from ops.jobs import local_devices, migration, realworld_local
    from ops.resources.gpu_guard import APPROVED_HOST_GPUS
    root = Path(__file__).resolve().parents[1]
    pin = migration._git(root, 'rev-parse', 'fe462a1').decode().strip()
    scripts = ['scripts/train.py', 'scripts/train_rl.py', 'scripts/train_realworld_fql.py',
               'scripts/train_snerl.py', 'scripts/eval_realworld_fql.py']
    scope = migration.execution_scope(scripts, pin, root)
    assert scope['identical'], scope['differences']
    for path in ['ops/resources/memory_budget.py', 'ops/resources/memory_watch.py',
                 'ops/jobs/realworld_local.py', 'ops/jobs/rwprep_mig.py']:
        assert path not in scope['files']
    # Exercise the actual pin admission predicate and placement proof, not just a hash comparison.
    row = dict(id='ram400-pin-proof', host='local', planned_commit=pin,
               script='scripts/train_realworld_fql.py', args=[])
    proof = realworld_local.placement_scope(row, root=root)
    assert proof['source_commit'] == pin
    assert 'ops/resources/memory_budget.py' not in proof['files']
    assert all(local_devices.allows(row, gpu, root) for gpu in APPROVED_HOST_GPUS['local'])
