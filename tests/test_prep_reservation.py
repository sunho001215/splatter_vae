"""Offline prep priority/backfill checks; no live queues, holds, tools or SSH."""
import json
from types import SimpleNamespace

import pytest

from ops.jobs import admission_pause as pauses
from ops.jobs import prep_reservation as prep
from ops.jobs import remote_offload
from ops.jobs.priority import PriorityReservations
from ops.jobs.rwprep_mig import atomic, digest
from ops.resources.gpu_guard import APPROVED_HOST_GPUS

GPUS = list(APPROVED_HOST_GPUS['remote'])


def bundle(tmp_path, name='batch', statuses=('pending',), **overrides):
    root = tmp_path / name
    root.mkdir()
    jobs = [dict(id=f'rwv2-20261113-prep-{name}-{i}', gpu_profile='1g.24gb', gpu_gib=16,
                 ram_gib=48, cpus=4, disk_gib=32, inputs={'/raw': 'a'*64},
                 steps=[{'kind': 'rwprep', 'argv': []}]) for i in range(len(statuses))]
    for job in jobs:
        job.update(overrides)
    config = dict(host='remote', image='sha256:'+'b'*64, commit='c'*40,
                  jobs=str(root/'jobs.json'), state_dir=str(root/'state'))
    atomic(config['jobs'], jobs)
    atomic(root/'dispatcher.json', config)
    states = {j['id']: dict(status=s, gpu=GPUS[i % len(GPUS)], container=f'prep-{name}-{i}')
              for i, (j, s) in enumerate(zip(jobs, statuses))}
    atomic(root/'state/state.json', dict(queue_sha256=digest(config['jobs']), jobs=states))
    atomic(pauses.PREP_PRIORITY_DIR/(name+'.json'), dict(active=True, host='remote', state_dir=config['state_dir']))
    return root, jobs, states


def policy(*, slots=None, free=None, gpus=None):
    limits = {g: dict(max_jobs=4, max_mem_gb=21.5) for g in (gpus or GPUS)}
    free = dict(ram_gb=1000, cpu_threads=320, disk_gb=1000) if free is None else free
    return PriorityReservations(limits, dict(ram_gb=1000, cpu_threads=320, disk_gb=1000),
                                lambda: slots or {}, lambda: free)


def campaign(gpu=None, **overrides):
    return dict(id='snerl-covering-resume', priority=-100, host='remote', gpu=gpu or GPUS[-1],
                mem_gb=8, ram_gb=16, cpu_threads=8, disk_gb=10, **overrides)


def snapshot(**kw):
    return SimpleNamespace(cotenants={}, **kw)


def test_remote_hold_no_longer_pauses_any_campaign_or_pool_row(tmp_path):
    bundle(tmp_path)
    assert not pauses.paused('rwnt-20261110-snerl-covering-pre')
    assert not pauses.paused('some-fql-eval')
    hold = json.loads(next(pauses.PREP_PRIORITY_DIR.glob('*.json')).read_text())
    atomic(tmp_path/'runs/setup/rwprep-mig-priority.d/remote.json', hold)
    assert not remote_offload.pauses(tmp_path)
    atomic(pauses.FILE, dict(pauses=[dict(active=True, name='explicit', ids=['some-fql'])]))
    assert pauses.paused('some-fql-eval')


def test_three_running_batches_leave_idle_migs_open(tmp_path):
    live = {}
    for name in ('b09', 'b10', 'b11'):
        _, _, states = bundle(tmp_path, name, ('running',))
        live[next(iter(states.values()))['container']] = {}
    p = prep.reserve(policy(), SimpleNamespace(cotenants=live))
    assert not p.reservations
    assert all(p.allows(campaign(g), g) for g in GPUS)


def test_pending_retry_launching_reserve_distinct_migs_without_double_count(tmp_path):
    _, _, states = bundle(tmp_path, statuses=('pending', 'retry', 'launching', 'running'))
    live = {list(states.values())[-1]['container']: {}}
    slots = {GPUS[-1]: dict(jobs=1, mem_gb=8)}
    p = prep.reserve(policy(slots=slots), SimpleNamespace(cotenants=live))
    assert len(p.reservations) == 3
    held = {r['gpu'] for r in p.reservations.values()}
    assert len(held) == 3 and GPUS[2] in held and GPUS[-1] not in held
    assert not any(p.allows(campaign(g), g) for g in held)
    assert p.allows(campaign(), GPUS[-1])  # healthy occupancy survives; backfill can share elsewhere
    assert sum(r['need']['ram_gb'] for r in p.reservations.values()) == 144
    assert slots == {GPUS[-1]: dict(jobs=1, mem_gb=8)}
    # Reservations are ephemeral; duplicate discovery of one batch cannot double charge it.
    atomic(pauses.PREP_PRIORITY, json.loads(next(pauses.PREP_PRIORITY_DIR.glob('*.json')).read_text()))
    assert len(prep.reserve(p, SimpleNamespace(cotenants=live)).reservations) == 3


@pytest.mark.parametrize('resource,held', [('ram_gb',48), ('cpu_threads',4), ('disk_gb',32)])
def test_host_resources_preserved_cumulatively(tmp_path, resource, held):
    bundle(tmp_path)
    free = dict(ram_gb=1000, cpu_threads=320, disk_gb=1000)
    row = campaign()
    free[resource] = held + row[resource]
    p = prep.reserve(policy(free=free), snapshot())
    assert p.backfill([row, {**row, 'id':'second'}]) == [row]
    free[resource] -= .1
    assert not p.allows(row, GPUS[-1])


def test_dependencies_remote_artifacts_external_states_stop_and_transitions(tmp_path):
    root, jobs, states = bundle(tmp_path, statuses=('pending', 'pending', 'retry'))
    jobs[1]['deps'] = [jobs[0]['id']]
    jobs[2].update(requires=['/remote/rig.json'], external_states={'/remote/other.json':['producer']})
    atomic(root/'jobs.json', jobs)
    atomic(root/'state/state.json', dict(queue_sha256=digest(root/'jobs.json'), jobs=states))
    calls = []
    class Client:
        def checked_ssh(self, command, **kw):
            import shlex
            args = shlex.split(command)
            assert args[:3] == ['python3', '-I', '-c'] and args[3] == prep.GATES
            gates = json.loads(args[4])
            calls.append(gates)
            assert gates[0]['requires'] == ['/remote/rig.json']
            assert gates[0]['external_states'] == {'/remote/other.json':['producer']}
            return SimpleNamespace(stdout=json.dumps([g['id'] for g in gates]) if len(calls)>1 else '[]')
    client = Client()
    assert len(prep.ready_rows(snapshot(), client)) == 1
    assert len(prep.ready_rows(snapshot(), client)) == 2
    (root/'state/STOP').touch()
    assert not prep.ready_rows(snapshot(), client)
    (root/'state/STOP').unlink()
    states[jobs[0]['id']]['status'] = 'done'
    states[jobs[2]['id']]['status'] = 'done'
    atomic(root/'state/state.json', dict(queue_sha256=digest(root/'jobs.json'), jobs=states))
    p = prep.reserve(policy(), snapshot(), client)
    assert len(p.reservations) == 1
    assert next(iter(p.reservations)).endswith(jobs[1]['id'])
    states[jobs[1]['id']]['status'] = 'done'
    atomic(root/'state/state.json', dict(queue_sha256=digest(root/'jobs.json'), jobs=states))
    assert not prep.reserve(p, snapshot(), client).reservations


def test_ambiguous_prep_evidence_fails_closed(tmp_path):
    bundle(tmp_path, statuses=('unknown',))
    with pytest.raises(ValueError):
        prep.reserve(policy(), snapshot())


@pytest.mark.parametrize('active', [True, False])
def test_finished_running_row_does_not_crash_or_reserve_while_state_lags(tmp_path, active):
    root, _, states = bundle(tmp_path, statuses=('running', 'pending', 'launching', 'running'))
    state_bytes = (root/'state/state.json').read_bytes()
    hold_path = next(pauses.PREP_PRIORITY_DIR.glob('*.json'))
    atomic(hold_path, dict(json.loads(hold_path.read_text()), active=active))
    hold_bytes = hold_path.read_bytes()
    calls = []
    class Client:
        def checked_ssh(self, command, **kw):
            calls.append(command)
            assert command == "docker ps --format '{{.Names}}'" and kw == {'idempotent': True}
            return SimpleNamespace(stdout='unrelated-container\n')  # stale running row has exited
    live = {list(states.values())[-1]['container']: {}}
    p = prep.reserve(policy(), SimpleNamespace(cotenants=live), Client())
    held = list(p.reservations.values())
    assert len(held) == 2  # only pending and launching; live cotenant counted once
    assert GPUS[2] in {r['gpu'] for r in held}
    assert sum(r['need']['ram_gb'] for r in held) == 96
    assert p.allows(campaign(), GPUS[-1])
    assert calls == ["docker ps --format '{{.Names}}'"]
    assert (root/'state/state.json').read_bytes() == state_bytes
    assert hold_path.read_bytes() == hold_bytes


def test_running_after_snapshot_reserves_pinned_quota_until_cotenant_is_charged(tmp_path):
    # Already running jobs bypass artifact/dependency/STOP gates just like launch intents.
    root, _, states = bundle(tmp_path, statuses=('running', 'running'), requires=['/not-present'])
    (root/'state/STOP').touch()
    names = [s['container'] for s in states.values()]
    calls = []
    class Client:
        def checked_ssh(self, command, **kw):
            calls.append(command)
            assert command == "docker ps --format '{{.Names}}'" and kw == {'idempotent': True}
            return SimpleNamespace(stdout='\n'.join(names)+'\n')
    p = prep.reserve(policy(), snapshot(), Client())
    assert len(calls) == 1  # one unfiltered, read-only listing for all missing rows
    assert {r['gpu'] for r in p.reservations.values()} == set(GPUS[:2])
    assert sum(r['need']['ram_gb'] for r in p.reservations.values()) == 96
    assert not any(p.allows(campaign(g), g) for g in GPUS[:2])
    assert not prep.reserve(p, SimpleNamespace(cotenants=dict.fromkeys(names)), Client()).reservations
    assert len(calls) == 1  # cotenant snapshot already charges them; no SSH or double count


def test_later_batch_running_intent_is_checked_after_reading_its_state(tmp_path):
    bundle(tmp_path, name='first', statuses=('running',))
    root, _, states = bundle(tmp_path, name='second')
    state = json.loads((root/'state/state.json').read_text())
    calls = []
    class Client:
        def checked_ssh(self, command, **kw):
            calls.append(command)
            if len(calls) == 1:
                # Another dispatcher launches after the first batch's Docker query.
                next(iter(state['jobs'].values()))['status'] = 'running'
                atomic(root/'state/state.json', state)
                return SimpleNamespace(stdout='')
            return SimpleNamespace(stdout=next(iter(states.values()))['container']+'\n')
    p = prep.reserve(policy(), snapshot(), Client())
    assert len(calls) == 2
    assert len(p.reservations) == 1
    assert next(iter(p.reservations.values()))['gpu'] == GPUS[0]


def test_missing_running_row_remote_query_failure_fails_closed(tmp_path):
    bundle(tmp_path, statuses=('running',))
    class Client:
        def checked_ssh(self, command, **kw):
            raise RuntimeError('Docker query failed')
    with pytest.raises(RuntimeError, match='Docker query failed'):
        prep.reserve(policy(), snapshot(), Client())


def test_changed_queue_or_missing_state_fails_closed(tmp_path):
    root, jobs, _ = bundle(tmp_path)
    jobs[0]['gpu_gib'] = 18
    atomic(root/'jobs.json', jobs)
    with pytest.raises(ValueError, match='queue changed'):
        prep.reserve(policy(), snapshot())
    (root/'state/state.json').unlink()
    with pytest.raises(OSError):
        prep.reserve(policy(), snapshot())


def test_more_ready_than_migs_reserves_only_device_placed_host_demand(tmp_path):
    bundle(tmp_path, statuses=('pending',)*3)
    p = prep.reserve(policy(gpus=GPUS[:2]), snapshot())
    assert len(p.reservations) == 2
    assert {r['gpu'] for r in p.reservations.values()} == set(GPUS[:2])
    assert not any(p.allows(campaign(g), g) for g in GPUS[:2])
    assert sum(r['need']['ram_gb'] for r in p.reservations.values()) == 96


def test_large_prep_backlog_leaves_campaign_device_and_host_capacity(tmp_path):
    bundle(tmp_path, statuses=('pending',)*31)
    slots = {GPUS[-1]: dict(jobs=1, mem_gb=8)}
    free = dict(ram_gb=703, cpu_threads=320, disk_gb=1000)
    p = prep.reserve(policy(gpus=[*GPUS[:2], GPUS[-1]], slots=slots, free=free), snapshot())
    assert list(p.reservations.values()) == [
        dict(gpu=g, priority=float('-inf'), need=dict(ram_gb=48, cpu_threads=4, disk_gb=32))
        for g in GPUS[:2]]
    small = {**campaign(), 'ram_gb': 8, 'mem_gb': 2}
    assert p.allows(small, GPUS[-1])
    assert not any(p.allows(small, g) for g in GPUS[:2])


@pytest.mark.parametrize('resource,demand', [('ram_gb', 48), ('cpu_threads', 4), ('disk_gb', 32)])
@pytest.mark.parametrize('fitting', [0, 1])
def test_host_shortfall_reserves_first_overflow_and_stops_later_rows(tmp_path, resource, demand, fitting):
    _, jobs, _ = bundle(tmp_path, statuses=('pending',)*4)
    free = dict(ram_gb=1000, cpu_threads=320, disk_gb=1000)
    free[resource] = (fitting + .5) * demand
    p = prep.reserve(policy(gpus=GPUS[:4], free=free), snapshot())
    held = list(p.reservations.values())
    assert len(held) == fitting + 1
    assert all(key.endswith(job['id']) for key, job in zip(p.reservations, jobs))
    assert sum(r['need'][resource] for r in held) == (fitting + 1) * demand
    assert not p.allows(campaign(GPUS[-1]), GPUS[-1])


@pytest.mark.parametrize('mem_gb,max_jobs,max_mem_gb,per_device', [(7, 4, 21, 3), (5, 2, 21.5, 2)])
def test_small_prep_rows_pack_until_device_memory_or_job_limit(tmp_path, mem_gb, max_jobs, max_mem_gb, per_device):
    bundle(tmp_path, statuses=('pending',)*7, gpu_gib=mem_gb, ram_gib=8, cpus=1)
    p = policy(gpus=GPUS[:2])
    for limit in p.limits.values():
        limit.update(max_jobs=max_jobs, max_mem_gb=max_mem_gb)
    prep.reserve(p, snapshot())
    held = list(p.reservations.values())
    assert [r['gpu'] for r in held] == GPUS[:2] * per_device
    assert sum(r['need']['ram_gb'] for r in held) == 8 * 2 * per_device


def test_live_device_memory_and_job_occupancy_are_respected(tmp_path):
    bundle(tmp_path, statuses=('pending',)*2)
    slots = {GPUS[0]: dict(jobs=1, mem_gb=8), GPUS[1]: dict(jobs=4, mem_gb=1),
             GPUS[2]: dict(jobs=1, mem_gb=1)}
    p = prep.reserve(policy(gpus=GPUS[:3], slots=slots), snapshot())
    assert [r['gpu'] for r in p.reservations.values()] == [GPUS[2]]
    assert slots == {GPUS[0]: dict(jobs=1, mem_gb=8), GPUS[1]: dict(jobs=4, mem_gb=1),
                     GPUS[2]: dict(jobs=1, mem_gb=1)}


def test_pinned_launching_rows_always_reserve_full_demand(tmp_path):
    root, _, states = bundle(tmp_path, statuses=('pending', 'launching', 'launching'))
    for state in states.values():
        state['gpu'] = GPUS[0]
    atomic(root/'state/state.json', dict(queue_sha256=digest(root/'jobs.json'), jobs=states))
    slots = {GPUS[0]: dict(jobs=4, mem_gb=21.5)}
    free = dict(ram_gb=24, cpu_threads=320, disk_gb=1000)
    p = prep.reserve(policy(gpus=GPUS[:2], slots=slots, free=free), snapshot())
    assert list(p.reservations.values()) == [
        dict(gpu=GPUS[0], priority=float('-inf'), need=dict(ram_gb=48, cpu_threads=4, disk_gb=32))] * 2


def test_pinned_host_demand_counts_toward_first_unpinned_overflow(tmp_path):
    _, jobs, _ = bundle(tmp_path, statuses=('pending', 'pending', 'launching', 'pending'))
    free = dict(ram_gb=72, cpu_threads=320, disk_gb=1000)
    p = prep.reserve(policy(gpus=GPUS[:4], free=free), snapshot())
    assert [key.rsplit(':', 1)[-1] for key in p.reservations] == [jobs[2]['id'], jobs[0]['id']]
    assert [r['gpu'] for r in p.reservations.values()] == [GPUS[2], GPUS[0]]


def test_incompatible_ready_row_still_raises_after_host_overflow(tmp_path):
    bundle(tmp_path, name='first')
    bundle(tmp_path, name='second', gpu_gib=21)
    free = dict(ram_gb=24, cpu_threads=320, disk_gb=1000)
    p = policy(free=free)
    for limit in p.limits.values():
        limit['max_mem_gb'] = 20
    with pytest.raises(ValueError, match='^ready prep has no compatible remote MIG$'):
        prep.reserve(p, snapshot())


def test_inactive_batch_can_become_ready_before_next_prep_writer_tick(tmp_path):
    bundle(tmp_path)
    path = next(pauses.PREP_PRIORITY_DIR.glob('*.json'))
    hold = json.loads(path.read_text())
    atomic(path, dict(hold, active=False))
    assert not pauses.paused('backfill')
    assert len(prep.reserve(policy(), snapshot()).reservations) == 1


def test_campaign_pool_replay_and_dispatch_consume_prep_reservations(tmp_path, monkeypatch):
    from tests.test_ours_priority import functions, remote_context
    bundle(tmp_path)
    actual = prep.ready_rows
    # The existing driver harness uses two abstract 10 GiB slots and a 10 GiB
    # host. Scale the real ready demand to that fixture, retaining discovery.
    monkeypatch.setattr(prep, 'ready_rows', lambda *a, **kw: [
        {**r, 'gpu_profile':None, 'mem_gb':8, 'ram_gb':1, 'cpu_threads':1, 'disk_gb':1}
        for r in actual(*a, **kw)])
    ns = remote_context()
    assert len([k for k in ns['priority_reservations'].reservations if k.startswith('prep:')]) == 1
    assert not ns['chosen']  # prep + waiting pretrain reserve both synthetic devices
    ns = remote_context(pool=True)
    assert not ns['chosen'] and not ns['plan']
    ns.update(cotenant_charges=lambda *a: dict(unbounded=[], cpus=0, gpus={}, ram_pending_bytes=0),
              owned_cpu_threads=lambda *a: 0, RELEASE_HOLD=None, _disk_floor=0)
    ns['jobs'].RESERVING = {'running'}
    ns['jobs'].reservation = lambda s,r,k: max(s.get(k,0),r.get(k,0))
    functions('ops/campaign/remote_campaign_driver.py', ns, 'dispatch_policy')
    fresh = snapshot(resources_ready=True,available_gb=1000,disk_free_bytes=1000*2**30)
    p = ns['dispatch_policy']({}, SimpleNamespace(snapshot=lambda:fresh))
    assert any(k.startswith('prep:') for k in p.reservations)


def test_scheduler_backfills_without_consuming_prep_mig(tmp_path):
    from tests.test_remote_jobs import FakeBackend, job, launched, tick, write_queue
    bundle(tmp_path)
    backend = FakeBackend(tmp_path/'runs')
    backend.value.admission_records = []
    backend.value.container_names = set()
    backend.value.gpu_memory = {g:(24192,67) for g in GPUS}
    queue = write_queue(tmp_path, [job('backfill', gpu='any')],
                        limits={g:dict(max_jobs=4,max_mem_gb=21.5) for g in GPUS[:2]})
    messages = tick(tmp_path, queue, backend)
    assert launched(backend) == [('backfill',GPUS[1])], messages
    assert not backend.removals


def test_prep_becoming_ready_between_batch_admissions_gets_next_mig(tmp_path, monkeypatch):
    from tests.test_remote_jobs import FakeBackend, job, launched, tick, write_queue
    root, _, _ = bundle(tmp_path)
    stop = root/'state/STOP'
    stop.touch()
    backend = FakeBackend(tmp_path/'runs')
    backend.value.admission_records = []
    backend.value.container_names = set()
    backend.value.gpu_memory = {g:(24192,67) for g in GPUS}
    launch = backend.launch
    def first_launch(path):
        result = launch(path)
        stop.unlink()  # prep becomes ready after the batch's initial snapshot
        return result
    monkeypatch.setattr(backend, 'launch', first_launch)
    queue = write_queue(tmp_path, [job('first',gpu=GPUS[0]), job('second',gpu=GPUS[1])],
                        limits={g:dict(max_jobs=4,max_mem_gb=21.5) for g in GPUS[:2]})
    messages = tick(tmp_path, queue, backend)
    assert launched(backend) == [('first',GPUS[0])], messages
    assert not backend.removals
