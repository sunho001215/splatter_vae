"""Verified guard refusal/retry paths in the current MIG regime, using test fakes only."""
import json
from pathlib import Path

import pytest

from tests.test_remote_jobs import (jobs, remote, REPO, REMOTE, COMMIT, IMAGE, CONTAINER_ID, GIB,
                                    FakeClient, FakeBackend, job, write_queue, tick)


@pytest.fixture
def empty_mig_backend(tmp_path):
    backend = FakeBackend(tmp_path / 'runs')
    backend.value.admission_records = []
    backend.value.container_names = set()
    backend.value.gpu_memory = {REMOTE: (24192, 67)}
    return backend


def refusal_proof(container='s4d-r-a1', gpu=REMOTE):
    return {'version': 1, 'container': container, 'guard': {
        'admit': False, 'uuid': gpu, 'reasons': ['measured free GPU memory/headroom'],
        'containers': 2, 'used_mib': 961, 'gpu_reserved_bytes': 12 * GIB,
        'gpu_pending_bytes': 11950000000, 'pending_ram_bytes': 0, 'reserved_disk_bytes': 0}}


def automatic_refusal_event(attempt=1):
    return {'event': 'admission_refused', 'id': 'r', 'host': 'remote', 'gpu': REMOTE,
            'attempt': attempt, 'container': f's4d-r-a{attempt}', 'time': 110,
            'checked_at': 109, 'snapshot_confirmed': True, 'containers_with_name': 0,
            'proof': refusal_proof(f's4d-r-a{attempt}')}


@pytest.mark.parametrize('executed', [False, True])
def test_automatic_refusal_preserves_resume_and_never_charges_failure(tmp_path, executed):
    registry = tmp_path / 'registry.jsonl'
    base = dict(id='r', host='remote', gpu=REMOTE, commit=COMMIT, image_digest=IMAGE,
                job_sha256='f' * 64, container='s4d-r-a1')
    events = []
    if executed:
        events += [dict(base, event='launched', attempt=1, time=1, container_id=CONTAINER_ID),
                   dict(id='r', event='checkpoint_paused', intentional_stop=True, attempt=1,
                        checkpoint='runs/r/checkpoints/step_10.pt', checkpoint_step=10, time=2),
                   dict(id='r', event='exited', attempt=1, code=143, time=3)]
    attempt = 2 if executed else 1
    events += [dict(base, event='launching', attempt=attempt, time=100, container=f's4d-r-a{attempt}'),
               automatic_refusal_event(attempt)]
    registry.write_text(''.join(json.dumps(e) + '\n' for e in events))
    result = jobs.read_registry(registry)['r']
    assert result['status'] == 'pending' and result['unknown'] is False
    assert result['failures'] == 0 and result['failed_attempts'] == []
    assert result['placement_backoff'][REMOTE] == 710
    if executed:
        assert result['commit'] == COMMIT and result['image_digest'] == IMAGE
        assert result['job_sha256'] == 'f' * 64 and result['intentional_stops'][0]['checkpoint_step'] == 10
    else:
        assert all(key not in result for key in ('commit', 'image_digest', 'job_sha256'))


@pytest.mark.parametrize('change', [
    {'attempt': True}, {'attempt': 2}, {'gpu': 'MIG-bad'}, {'container': 's4d-r-a2'},
    {'host': 'local'}, {'snapshot_confirmed': False}, {'snapshot_confirmed': 1},
    {'containers_with_name': 1}, {'containers_with_name': False},
    {'checked_at': 99}, {'checked_at': 111}, {'time': 500}, {'checked_at': float('nan')},
    {'proof': {}}, {'proof': {'version': 1, 'container': 's4d-r-a1', 'guard': {'admit': False}}},
])
def test_automatic_refusal_rejects_invalid_proof(tmp_path, change):
    registry = tmp_path / 'registry.jsonl'
    base = dict(event='launching', id='r', host='remote', gpu=REMOTE, attempt=1,
                container='s4d-r-a1', time=100)
    registry.write_text(json.dumps(base) + '\n' + json.dumps({**automatic_refusal_event(), **change}) + '\n')
    with pytest.raises(ValueError):
        jobs.read_registry(registry)


@pytest.mark.parametrize('confirmed,present', [(True, False), (True, True), (False, False), (None, False)])
def test_tick_guard_refusal_requires_fresh_absence_then_backs_off(tmp_path, monkeypatch, empty_mig_backend,
                                                               confirmed, present):
    queued = job()
    queue = write_queue(tmp_path, [queued], limits={REMOTE: {'max_jobs': 4, 'max_mem_gb': 21.5}})
    backend = empty_mig_backend
    calls = []
    def refused(path):
        calls.append(path)
        if confirmed is None:
            backend.value.container_names = None
        else:
            backend.value.resources_ready = confirmed
            backend.value.container_names = {'s4d-r-a1'} if present else set()
        raise remote.AdmissionRefused(refusal_proof())
    backend.launch = refused
    tick(tmp_path, queue, backend)
    current = jobs.read_registry(tmp_path / 'registry.jsonl')['r']
    assert current['failures'] == 0
    if confirmed is True and not present:
        assert current['status'] == 'pending' and not current['unknown']
        # A second tick may run, but cannot choose this job/GPU again until expiry.
        tick(tmp_path, queue, backend)
        assert len(calls) == 1
        monkeypatch.setattr(jobs.time, 'time', lambda: current['placement_backoff'][REMOTE] + 1)
        tick(tmp_path, queue, backend)
        assert len(calls) == 2
    else:
        assert current['status'] == 'launching' and current['unknown']


def test_backend_distinguishes_only_valid_guard_refusal(tmp_path, monkeypatch):
    import subprocess
    path = tmp_path / 'attempt.json'
    path.write_text(json.dumps({'container': 's4d-r-a1', 'gpu': REMOTE}))
    client = FakeClient()
    client._redact = lambda text: text
    backend = remote.RemoteBackend(REPO, tmp_path / 'runs', client)
    for code, stdout, expected in [(20, json.dumps(refusal_proof()), remote.AdmissionRefused),
                                   (20, 'not-json', remote.RemoteJobError),
                                   (20, '{}', remote.RemoteJobError),
                                   (1, json.dumps(refusal_proof()), remote.RemoteJobError),
                                   (75, json.dumps(refusal_proof()), remote.RemoteUnreachable)]:
        monkeypatch.setattr(remote.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess([], code, stdout, ''))
        with pytest.raises(expected) as caught:
            backend.launch(path)
        assert (isinstance(caught.value, remote.AdmissionRefused)) is (expected is remote.AdmissionRefused)


@pytest.mark.local_host_data
def test_real_registry_replay_unchanged_for_every_job(tmp_path):
    """Only read main's registry; replay one captured byte sequence with both reducers."""
    import ast
    import subprocess
    source = subprocess.run(['git', 'show', '1b1e228:ops/jobs/scheduler.py'], cwd=REPO,
                            check=True, capture_output=True, text=True).stdout
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'read_registry')
    original = dict(jobs.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<original-reducer>', 'exec'), original)
    registry = Path('/home/ws/ws/splatter4d/experiments/registry.jsonl')
    captured = tmp_path / 'registry.jsonl'
    captured.write_bytes(registry.read_bytes())
    before = original['read_registry'](captured)
    after = jobs.read_registry(captured)
    assert before and before == after


@pytest.mark.parametrize('state', ['running', 'paused', 'restarting', 'exited', 'created', 'dead'])
def test_refusal_never_resets_with_container_in_any_state(tmp_path, empty_mig_backend, state):
    backend = empty_mig_backend
    queue = write_queue(tmp_path, [job()], limits={REMOTE: {'max_jobs': 4, 'max_mem_gb': 21.5}})
    def refused(path):
        spec = json.loads(path.read_text())
        backend.value.container_names = {spec['container']}
        backend.value.containers[spec['container']] = {'State': {'Status': state}}
        raise remote.AdmissionRefused(refusal_proof())
    backend.launch = refused
    tick(tmp_path, queue, backend)
    current = jobs.read_registry(tmp_path / 'registry.jsonl')['r']
    assert current['status'] == 'launching' and current['unknown']
    assert current['failures'] == 0


@pytest.mark.parametrize('mutate', [
    lambda p: p['guard'].update(admit=True),
    lambda p: p['guard'].update(reasons=[]),
    lambda p: p['guard'].update(reasons=['unknown reason']),
    lambda p: p['guard'].update(used_mib=True),
    lambda p: p['guard'].update(error='not an admission refusal'),
    lambda p: p.update(container='s4d-wrong-a1'),
    lambda p: p.update(version=True),
])
def test_refusal_verdict_validation_rejects_ambiguous_shapes(mutate):
    proof = refusal_proof()
    mutate(proof)
    with pytest.raises(ValueError):
        remote.validate_guard_refusal(proof, 's4d-r-a1', REMOTE)
