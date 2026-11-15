"""Pinned local guard compatibility; no old code is executed and no job starts."""
import copy
import subprocess
from pathlib import Path

import pytest

from ops.jobs import fair_share, local_devices, realworld_local, scheduler
from ops.jobs.priority import PriorityReservations
from ops.resources.gpu_guard import APPROVED_HOST_GPUS

pytestmark = pytest.mark.local_host_data

GPUS = APPROVED_HOST_GPUS['local']
ROOT = Path(__file__).resolve().parents[1]


def job(**extra):
    return dict(id='pinned', script='scripts/train.py', mem_gb=1, ram_bytes=1,
                family='metaworld', **extra)


def test_old_release_and_new_release_use_their_own_allowlists():
    old = job(planned_commit='ee85ad9')
    assert all(local_devices.allows(old, g) for g in GPUS[:8])
    assert not any(local_devices.allows(old, g) for g in GPUS[8:])
    # Read the candidate's current source in lieu of a self-referential commit SHA.
    approved = local_devices._allowlist((ROOT / 'ops/resources/gpu_guard.py').read_text(),
                                       (ROOT / 'configs/hosts/local.yaml').read_text())
    assert approved == frozenset(GPUS)
    assert all(local_devices.allows(job(), g) for g in GPUS)
    assert not local_devices.allows(job(planned_commit='f' * 40), GPUS[-1])


def test_old_pin_never_admitted_when_only_new_devices_free():
    limits = {g: dict(max_jobs=1, max_mem_gb=22) for g in GPUS}
    running = [dict(id=str(i), host='local', gpu=g, mem_gb=22) for i, g in enumerate(GPUS[:8])]
    old = job(planned_commit='ee85ad9')
    assert scheduler.choose_gpu(old, limits, running, set()) is None
    slots = {g: dict(**limit, jobs=int(g in GPUS[:8]), mem_gb=22 if g in GPUS[:8] else 0)
             for g, limit in limits.items()}
    before = copy.deepcopy(slots)
    assert realworld_local._pack(old, slots) is None
    assert slots == before
    active = [{**r, 'family': 'metaworld', 'ram_bytes': 1} for r in running]
    result = fair_share.select([old], active, limits, 100, 8)
    assert not result['admitted']
    assert all(g in GPUS[:8] for g in result['reserved'])
    policy = PriorityReservations(limits, dict(ram_gb=100, cpu_threads=100, disk_gb=100),
                                  lambda: slots, lambda: {})
    assert policy.devices(old) == list(GPUS[:8])
    assert scheduler.choose_gpu(job(), limits, running, set()) == GPUS[8]
    assert realworld_local._pack(job(), slots) == GPUS[8]


def test_pinned_entry_point_uses_its_guard_and_config(tmp_path, monkeypatch):
    checkout = tmp_path / '.worktrees/old'
    for name in ('ops/resources/gpu_guard.py', 'configs/hosts/local.yaml'):
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.run(['git', '-C', str(ROOT), 'show', f'ee85ad9:{name}'],
                                       capture_output=True, check=True).stdout)
    script = checkout / 'scripts/train.py'
    script.parent.mkdir()
    script.write_text('guard_gpus()\n')
    row = job()
    row['script'] = str(script)
    monkeypatch.setattr(scheduler, 'REPO', tmp_path)
    assert local_devices.execution_root(row, tmp_path) == checkout
    assert local_devices.allows(row, GPUS[0], tmp_path)
    assert not local_devices.allows(row, GPUS[-1], tmp_path)
    calls = []
    class Process:
        pid = 123
    monkeypatch.setattr(scheduler.subprocess, 'Popen', lambda *a, **kw: calls.append(kw) or Process())
    assert scheduler.launch(row, GPUS[0], 1, runs=tmp_path / 'runs') == 123
    assert calls[0]['env']['S4D_HOST_CONFIG'] == str(checkout / 'configs/hosts/local.yaml')
    with pytest.raises(ValueError, match='device refused'):
        scheduler.launch(row, GPUS[-1], 2, runs=tmp_path / 'runs')
    assert len(calls) == 1


def test_new_pinned_release_can_use_all_sixteen(tmp_path):
    """An actual new commit, with guard/config committed; no special-case HEAD authorization."""
    for name in ('ops/resources/gpu_guard.py', 'configs/hosts/local.yaml'):
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], check=True, capture_output=True, text=True).stdout.strip()
    git('init', '-q')
    git('add', '.')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'MIG16 fixture')
    pinned = job(planned_commit=git('rev-parse', 'HEAD'))
    assert all(local_devices.allows(pinned, gpu, tmp_path) for gpu in GPUS)
    slots = {g: dict(max_jobs=1, max_mem_gb=22, jobs=int(g in GPUS[:8]),
                    mem_gb=22 if g in GPUS[:8] else 0) for g in GPUS}
    assert realworld_local._pack(pinned, slots, root=tmp_path) == GPUS[8]


@pytest.mark.parametrize('config', [None, '{broken',
    'name: local\ngpu_uuids: [MIG-a, MIG-a]\n',
    'name: local\ngpu_uuids: [MIG-a]\nruntime_root: /other\n'])
def test_malformed_pinned_config_fails_closed(tmp_path, config):
    checkout = tmp_path / '.worktrees/old'
    guard = checkout / 'ops/resources/gpu_guard.py'
    guard.parent.mkdir(parents=True)
    guard.write_text("APPROVED_HOST_GPUS = {'local': ('MIG-a',)}")
    host = checkout / 'configs/hosts/local.yaml'
    host.parent.mkdir(parents=True)
    host.write_text(config or 'null')
    row = job()
    row['script'] = str(checkout / 'scripts/train.py')
    # The actual authorization must include this fixture UUID to reach the pinned config check.
    from ops.resources import gpu_guard
    original = gpu_guard.APPROVED_HOST_GPUS
    from unittest.mock import patch
    with patch.object(gpu_guard, 'APPROVED_HOST_GPUS', {**original, 'local': ('MIG-a',)}):
        assert not local_devices.allows(row, 'MIG-a', tmp_path)



def test_planner_without_script_uses_current_checkout():
    assert local_devices.execution_root({}, ROOT) == ROOT
    assert all(local_devices.allows({}, gpu) for gpu in GPUS)


@pytest.fixture
def placed_identical(tmp_path, monkeypatch):
    import hashlib
    import json
    from ops.provenance import upstreams
    row = dict(id='placed', host='local', script='scripts/train.py', mem_gb=1, planned_commit='a'*40,
               args=['--config', 'configs/train.yaml'])
    paths = {'scripts/train.py': 'import s4d.model\n', 's4d/model.py': 'VALUE = 1\n',
             'configs/train.yaml': 'steps: 400000\n', 'uv.lock': 'environment\n'}
    for name, data in paths.items():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    record = dict(job_id='placed', kind='role_placement', state='placed', owner_host='local',
                  execution_scope=dict(source_commit='a'*40, files={
                      p: hashlib.sha256(v.encode()).hexdigest() for p,v in paths.items()}))
    ownership = tmp_path/'runs/setup/ownership/placed.json'
    ownership.parent.mkdir(parents=True)
    ownership.write_text(json.dumps(record))
    monkeypatch.setattr(local_devices, '_release_devices', lambda *a: frozenset(GPUS[:8]))
    monkeypatch.setattr(upstreams, 'working_tree_identity', lambda root: {})
    return row, record, ownership, tmp_path


def test_identical_owned_old_pin_uses_new_migs_without_mutation(placed_identical, monkeypatch):
    row, record, path, root = placed_identical
    monkeypatch.setattr(scheduler, 'REPO', root)
    before = path.read_bytes(), copy.deepcopy(row)
    limits = {g: dict(max_jobs=1, max_mem_gb=22) for g in GPUS}
    running = [dict(id=str(i), host='local', gpu=g, mem_gb=22) for i,g in enumerate(GPUS[:8])]
    assert all(local_devices.allows(row, gpu, root) for gpu in GPUS)
    assert scheduler.choose_gpu(row, limits, running, set()) == GPUS[8]
    assert (path.read_bytes(), row) == before


@pytest.mark.parametrize('change', ['module', 'script', 'config', 'environment', 'argv', 'upstream',
                                    'deleted', 'missing-proof', 'pin', 'owner', 'state', 'kind'])
def test_old_pin_new_migs_refuse_incomplete_or_changed_scope(placed_identical, monkeypatch, change):
    import json
    from ops.provenance import upstreams
    row, record, path, root = placed_identical
    if change in ('module', 'script', 'config', 'environment'):
        name = {'module':'s4d/model.py', 'script':'scripts/train.py',
                'config':'configs/train.yaml', 'environment':'uv.lock'}[change]
        (root/name).write_text('changed\n')
    elif change == 'argv':
        row['args'] = ['--config', 'configs/other.yaml']
    elif change == 'upstream':
        monkeypatch.setattr(upstreams, 'working_tree_identity', lambda root: {'third_party/other@commit': 'b'*40})
    elif change == 'deleted':
        (root/'s4d/model.py').unlink()
    elif change == 'missing-proof':
        record['execution_scope']['files'].pop(row['script'])
    elif change == 'pin':
        row['planned_commit'] = 'b'*40
    else:
        record[{'owner':'owner_host', 'state':'state', 'kind':'kind'}[change]] = 'other'
    path.write_text(json.dumps(record))
    assert not local_devices.allows(row, GPUS[-1], root)
    # Existing device eligibility and all state/ownership enforcement are unchanged.
    assert local_devices.allows(row, GPUS[0], root)


def test_missing_ownership_invalid_pin_and_external_checkout_fail_closed(placed_identical, monkeypatch):
    row, record, path, root = placed_identical
    path.unlink()
    assert not local_devices.allows(row, GPUS[-1], root)
    path.write_text('{broken')
    assert not local_devices.allows(row, GPUS[-1], root)
    monkeypatch.setattr(local_devices, '_release_devices', lambda *a: frozenset())
    assert not local_devices.allows(row, GPUS[-1], root)
