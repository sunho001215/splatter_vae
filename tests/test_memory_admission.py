"""Local staging reserve and current-queue growth policy; never import campaign drivers."""
import ast
import json
import sys
from pathlib import Path

import pytest

from ops.resources import memory_budget as mb, memory_watch as watch
from ops.jobs.footprints import local_footprint


@pytest.mark.parametrize('mode,usage,growth,extra,host,expected', [
    ('fits', 64e9, 30e9, 44e9, 200e9, True),
    ('fits', 64e9, 30e9, 44e9 + 1, 200e9, False),
    ('fits-staging', 138e9, 100e9, 2e9, 200e9, True),
    ('fits-staging', 138e9 + 1, 0, 2e9, 200e9, False),
    ('fits-staging', 64e9, 100e9, 2e9 + 1, 200e9, False),
    ('fits-staging', 64e9, 100e9, 2e9, mb.HOST_RESERVE + 2e9 - 1, False),
    ('fits-staging', 64e9, 100e9, 2e9, mb.HOST_RESERVE + 2e9, True),
    ('fits', 64e9, 0, 2e9, mb.HOST_RESERVE + 2e9 - 1, False),
    ('fits-staging', 1e9, 0, -1, 200e9, False),
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
    assert data['staging_reserve'] == 2_000_000_000 and data['ceiling'] == 150_000_000_000
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
    assert mb.CEILING == 150_000_000_000
    assert mb.TRAINING_CEILING == 138_000_000_000
    assert names['CAP'] == 131_000_000_000


def test_realworld_capacity_withholds_staging(monkeypatch):
    from ops.jobs import realworld_local
    monkeypatch.setattr(mb, 'snapshot', lambda: {'budget_usage': 64_000_000_000})
    monkeypatch.setattr(mb, 'session_nonfile_pss', lambda: {})
    assert realworld_local._capacity({'jobs': []}, {}) == {'ram_free_bytes': 74_000_000_000}


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
