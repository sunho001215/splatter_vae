"""Offline regressions for the release/prep priority inversion of 2026-11-17."""

import json
import shlex
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ops.containers import rwprep_capacity
from ops.containers.verification import CPU_THREADS, POLICIES
from ops.jobs import release_reservation as rr
from ops.jobs import rwprep_remote as prep
from ops.jobs import scheduler
from ops.jobs.remote_backend import RemoteSnapshot
from ops.jobs.rwprep_mig import atomic, digest
from ops.resources.gpu_guard import APPROVED_HOST_GPUS
from tests.test_prep_reservation import bundle
from tests.test_remote_jobs import _rv_setup
from tests.test_remote_mig import slot_container

REPO = Path(__file__).resolve().parents[1]
GPUS = list(APPROVED_HOST_GPUS['remote'])
GIB = 2**30


@pytest.mark.parametrize('all_reserved', [False, True])
def test_reserved_prep_mig_does_not_stop_reconciliation_or_other_migs(tmp_path, monkeypatch, all_reserved):
    jobs = [dict(id=f'rwv2-20261113-prep-test-{i}', gpu_profile='1g.24gb', gpu_gib=16,
                 ram_gib=48, cpus=4, inputs={'/raw': 'a'*64}, steps=[{'kind': 'rwprep', 'argv': []}])
            for i in range(3)]
    config = dict(host='remote', image='sha256:'+'b'*64, commit='c'*40,
                  jobs=str(tmp_path/'jobs.json'), state_dir=str(tmp_path/'state'),
                  priority_file=str(tmp_path/'priority.json'), remote='/remote')
    atomic(config['jobs'], jobs)
    states = {j['id']: dict(status='running', attempt=1, gpu=GPUS[i], container=f'prep-{i}')
              for i, j in enumerate(jobs[:2])}
    atomic(tmp_path/'state/state.json', dict(queue_sha256=digest(config['jobs']), jobs=states))
    queue = tmp_path/'queue.yaml'
    queue.write_text(yaml.safe_dump({'hosts': {'remote': {'cpu_thread_budget': 368}}}))
    monkeypatch.setattr(prep, 'LIVE_QUEUE', queue)
    reserved = set(GPUS if all_reserved else GPUS[:1])
    monkeypatch.setattr(rr, 'reserved_gpus', lambda: reserved)
    observed = {
        f'prep-{i}': dict(labels={'s4d.commit': config['commit'], 's4d.rwprep.job': jobs[i]['id'],
                                's4d.gpu': GPUS[i], 's4d.role': 'rwprep', 's4d.owner': 'kaist-sunho'},
                         state={'Running': i == 0, 'ExitCode': 0},
                         proof={'status': 'ok', 'queue_sha256': digest(config['jobs'])})
        for i in range(2)}
    monkeypatch.setattr(prep, 'remote_observe', lambda *_: observed)

    def decide(*args, **kwargs):
        assert kwargs['cpu_budget_threads'] == 368
        return {'candidates': GPUS, 'host': []}

    parallel = SimpleNamespace(load=lambda *_: ({}, ''), snapshot=lambda _: {}, decide=decide)
    monkeypatch.setattr(prep, 'established', lambda: (None, parallel))
    launches, mirrors = [], []

    def launch(client, cfg, job, intent):
        # Keep the real launch-time reservation guard: filtering is per device.
        rwprep_capacity.guard_payload(dict(gpu=intent['gpu'], gpu_need_bytes=16*GIB,
                                           ram_bytes=48*GIB, cpus=4), queue_path=queue)
        launches.append(intent['gpu'])

    monkeypatch.setattr(prep, 'launch_job', launch)

    def ssh(command, **kwargs):
        payload = json.loads(shlex.split(command)[-1])
        if isinstance(payload, list):
            return SimpleNamespace(stdout=json.dumps([j['id'] for j in payload]))
        mirrors.append(payload)
        return SimpleNamespace(stdout='')

    result = prep.tick(config, SimpleNamespace(checked_ssh=ssh), apply=True)
    assert launches == ([] if all_reserved else [GPUS[1]])
    saved = json.loads((tmp_path/'state/state.json').read_text())
    assert saved['jobs'][jobs[0]['id']]['status'] == 'running'  # reserved MIG's existing job still reconciles
    assert saved['jobs'][jobs[1]['id']]['status'] == 'done'
    assert mirrors[-1] == saved  # external dependency gates receive the finished row
    assert result['counts']['done'] == 1
    assert result['counts']['pending'] == int(all_reserved)


@pytest.mark.parametrize('constraint', ['none', 'live_prep', 'ram', 'cpu', 'disk', 'unreserved', 'campaign_first'])
def test_release_outranks_pending_prep_but_preserves_live_capacity(tmp_path, monkeypatch, constraint):
    bundle(tmp_path, statuses=('pending',)*len(GPUS))  # enough waiting prep to hold every MIG
    backend, host, native, evidence = _rv_setup(tmp_path, monkeypatch)
    native.update(id='verify-native-aaaaaaa', mem_gb=POLICIES['native_suite'][3],
                  cpu_threads=CPU_THREADS['native_suite'])
    host.update(enabled=True, cpu_thread_budget=368,
                limits={g: dict(max_jobs=4, max_mem_gb=21.5) for g in GPUS})
    snapshot = RemoteSnapshot({g: [] for g in GPUS}, 1000, 1000*GIB, 800*GIB, {},
                              gpu_memory={g: (24192, 0) for g in GPUS}, admission_records=[],
                              container_names=set(), total_ram_gb=1200)
    if constraint == 'live_prep':
        item = slot_container(tmp_path, gpu=native['gpu'])
        snapshot.admission_records = [(item, set(), 16*GIB)]
    elif constraint == 'ram':
        snapshot.available_gb = 200  # native needs 96 + 120 GiB reserve
    elif constraint == 'cpu':
        host['cpu_thread_budget'] = 16
    elif constraint == 'disk':
        snapshot.disk_free_bytes = 150*GIB  # no room for native's 1 GiB above the floor
    monkeypatch.setattr(scheduler, 'REPO', tmp_path)
    monkeypatch.setattr(backend, 'snapshot', lambda: snapshot)
    monkeypatch.setattr(backend, 'ready_verification', lambda *_: evidence)
    launches = []
    monkeypatch.setattr(backend, 'launch', lambda p: launches.append(json.loads(p.read_text()))
                        or {'container_id': 'f'*64})
    resource = dict(mem_gb=native['mem_gb'], ram_gb=96, cpu_threads=32, disk_gb=1, load=1)
    rr.write(rr.plan(host['commit'], 'aaaaaaa', 'native', native['gpu'], GPUS,
                     native=resource, isolation=dict(resource, mem_gb=4, ram_gb=8, cpu_threads=8, load=0)))
    rows = [native]
    if constraint == 'unreserved':
        rr.PATH.unlink()  # unrelated verification cannot bypass prep priority
    elif constraint == 'campaign_first':
        monkeypatch.setattr(backend, 'ready', lambda *_: evidence)
        monkeypatch.setattr(scheduler, 'check_guarded', lambda _: tmp_path/'worker.py')
        rows.insert(0, dict(id='campaign', host='remote', gpu=GPUS[0], role='realworld_pretrain',
                            script='worker.py', args=[], mem_gb=8, ram_gb=10, disk_gb=1, priority=-100,
                            required_results=['metrics.jsonl']))
    queue = tmp_path/'queue.yaml'
    queue.write_text(yaml.safe_dump({'hosts': {'remote': host}, 'jobs': rows}))
    messages = scheduler.tick(queue, tmp_path/'registry.jsonl', tmp_path/'runs', usage={}, remote_backend=backend)
    expected = [native['id']] if constraint in ('none', 'campaign_first') else []
    assert [j['id'] for j in launches] == expected, messages
    assert not any(m.startswith(native['id']) and 'higher-priority' in m for m in messages)


def chain_module(root):
    # Match the isolated release-chain harness in test_isolation_reuse; no live ROOT paths.
    source = (REPO/'ops/campaign/release_chain.py').read_text().replace(
        "ROOT = Path('/home/ws/ws/splatter4d')", f'ROOT = Path({str(root)!r})')
    names = {'__name__': 'test_release_prep_chain'}
    exec(compile(source, 'release_chain.py', 'exec'), names)
    return names


@pytest.mark.parametrize('failures', [0, 1, 2, 3])
def test_native_gpu_choice_never_pins_busy_first_mig_after_inventory_error(tmp_path, monkeypatch, failures):
    from ops.jobs import remote_backend

    names = chain_module(tmp_path)
    release = names['Release'](['release_chain', 'a'*40])
    first, idle = GPUS[:2]
    release.gpus = [first, idle]
    item = slot_container(tmp_path, gpu=first, managed='true')
    item['Name'] = '/s4d-live'
    snapshot = RemoteSnapshot({first: [(777, 8740)], idle: []}, 1468, 10000*GIB, 8000*GIB,
                             {'s4d-live': {**item, 'pids': {777}}},
                             gpu_memory={first: (24192, 8810), idle: (24192, 64)},
                             admission_records=[(item, {777}, 2*GIB)])
    reads = []

    def fresh():
        reads.append(None)
        if len(reads) <= failures:
            raise ValueError('unreadable owned host resource evidence')
        return snapshot

    monkeypatch.setitem(names, 'RemoteBackend', lambda *_: SimpleNamespace(snapshot=fresh))
    monkeypatch.setattr(names['jobs'], 'read_registry', lambda _: {})
    monkeypatch.setattr(names['jobs'], 'load_queue', lambda _: {'hosts': {'remote': {'limits': {
        g: dict(max_jobs=4, max_mem_gb=21.5) for g in release.gpus}}}})
    if failures == 3:
        with pytest.raises(SystemExit, match='no GPU pinned'):
            release.pick_native_gpu()
        assert release.native_gpu is None and len(reads) == 3
    else:
        release.pick_native_gpu()
        assert release.native_gpu == idle and len(reads) == failures + 1


@pytest.mark.parametrize('status', [None, 'pending', 'crashed'])
def test_capacity_timeout_fails_main_and_clears_reservation(tmp_path, monkeypatch, status):
    names = chain_module(tmp_path)
    release = names['Release'](['release_chain', 'a'*40, '--native-gpu', GPUS[0], '--skip-sync'])
    monkeypatch.setitem(names, 'commit_reader', lambda *_: (None, None))
    monkeypatch.setattr(names['suite_partition'], 'release_support', lambda _: {'compatible': True})
    queue = tmp_path/'native.yaml'
    queue.write_text(yaml.safe_dump({'jobs': [{'gpu': GPUS[0], 'mem_gb': 12, 'ram_gb': 96}]}))
    monkeypatch.setitem(names, 'run', lambda _: str(queue))
    monkeypatch.setattr(release, 'start_local_partition', lambda: None)
    (tmp_path/'experiments').mkdir()
    clock = [0]

    def tick(**kwargs):
        assert rr.load() is not None
        clock[0] += 300
        return []

    fake_jobs = SimpleNamespace(tick=tick, read_registry=lambda _: {'verify-native-aaaaaaa': {'status': status}},
                                RUNNING='running', DONE='done', PENDING='pending')
    monkeypatch.setitem(names, 'jobs', fake_jobs)
    monkeypatch.setitem(names, 'RemoteBackend', lambda *_: None)
    monkeypatch.setitem(names, 'time', SimpleNamespace(time=time.time, strftime=time.strftime,
                                                     monotonic=lambda: clock[0], sleep=lambda _: None))
    with pytest.raises(SystemExit, match='capacity admission timeout'):
        release.main()
    assert clock[0] == 600
    assert not rr.PATH.exists() and release._stop.is_set()
    assert json.loads((tmp_path/'runs/setup/release-timing/aaaaaaa.json').read_text())['outcome'] == 'failed'


def test_running_suite_is_not_subject_to_capacity_timeout(tmp_path, monkeypatch):
    names = chain_module(tmp_path)
    release = names['Release'](['release_chain', 'a'*40])
    clock = [0]
    registry = {'verify-native-aaaaaaa': {'status': 'running', 'container': 'native'}}

    def watch(*args, **kwargs):
        clock[0] += names['ADMISSION_TIMEOUT_SECONDS'] + 1
        registry['verify-native-aaaaaaa']['status'] = 'done'

    monkeypatch.setitem(names, 'jobs', SimpleNamespace(tick=lambda **_: [], read_registry=lambda _: registry,
                                                     RUNNING='running', DONE='done', PENDING='pending'))
    monkeypatch.setitem(names, 'RemoteBackend', lambda *_: SimpleNamespace(container_states=None))
    monkeypatch.setattr(names['release_poll'], 'wait_for_exit', watch)
    monkeypatch.setitem(names, 'time', SimpleNamespace(time=time.time, strftime=time.strftime,
                                                     monotonic=lambda: clock[0]))
    release.drive(tmp_path/'queue.yaml', list(registry))
    assert clock[0] > names['ADMISSION_TIMEOUT_SECONDS']
