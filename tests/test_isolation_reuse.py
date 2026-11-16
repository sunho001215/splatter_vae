"""CPU-only release reuse/gate tests. All records and queue writes are under tmp_path."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ops.containers import isolation_cache as cache
from ops.containers.launch import evidence_artifacts
from ops.containers.verification import POLICIES, canonical_arguments
from ops.jobs import release_reservation as reservation
from ops.jobs import remote_backend as remote
from ops.resources.gpu_guard import APPROVED_HOST_GPUS

REPO = Path(__file__).resolve().parents[1]
OLD, NEW = 'a' * 40, 'b' * 40
IMAGE = 'sha256:' + 'c' * 64
GPUS = list(APPROVED_HOST_GPUS['remote'])
SOURCE = {cache.HOST_CONFIG: 'd' * 64, cache.MIG_CONFIG: 'e' * 64, 'scripts/worker.py': 'f' * 64}
NEW_SOURCE = {**SOURCE, 'scripts/worker.py': '0' * 64}


def write(repo, relative, payload):
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return cache.reference(path, repo)


@pytest.fixture
def history(tmp_path, monkeypatch):
    reads = []
    def source(repo, commit):
        reads.append(commit)
        return SOURCE if commit == OLD else NEW_SOURCE
    monkeypatch.setattr(remote, 'commit_source_sha256', source)
    now = time.time()
    suite = {'host': 'remote', 'full_suite_passed': True, 'exit_code': 0,
             'source_sha256': NEW_SOURCE, 'source_sha256_before': NEW_SOURCE,
             'source_changed_during_suite': False, 'expected_commit': NEW, 'expected_image_digest': IMAGE,
             'counts': {'passed': 5, 'failed': 0, 'error': 0, 'skipped': 0},
             'isolation_host_identity': cache.host_identity(NEW_SOURCE, '580.95.05', 'pid:[4026531836]', 'cdi')}
    state = {}
    for i, uuid in enumerate(GPUS):
        job_id = f'verify-isolation-{i}-{OLD[:7]}-cc'
        mapping = [{'uuid': uuid, 'index': 0, 'name': 'NVIDIA MIG 1g.24gb'}]
        report = {'host': 'remote', 'passed': True, 'source_sha256': SOURCE,
                  'source_changed_during_check': False, 'parent_mapping': mapping,
                  'pid_matching': {'host_pid_namespace_asserted': True, 'pid_namespace': 'pid:[4026531836]'},
                  'results': [{'uuid': uuid, 'passed': True, 'gpus_seen': [uuid],
                               'full_process_table': '| NVIDIA-SMI 580.95.05 Driver Version: 580.95.05 CUDA Version: 13.0 |',
                               'child': {'mapping': mapping, 'egl_device': 0, 'native_cuda_passed': True,
                                         'native_egl_passed': True}},
                              *({'cuda_visible_devices': bad, 'rejected': True}
                                for bad in ('0', None, APPROVED_HOST_GPUS['local'][0]))]}
        args = canonical_arguments('gpu_isolation', job_id, 1, uuid)
        spec = {'host': 'remote', 'id': job_id, 'attempt': 1, 'commit': OLD, 'image_digest': IMAGE,
                'gpu': uuid, 'gpu_backend': 'cdi', 'exclusive_gpu': True, 'script': POLICIES['gpu_isolation'][0],
                'verification_purpose': 'gpu_isolation', 'campaign_eligible': False, 'args': args,
                'ram_gb': 8, 'mem_gb': 4, 'disk_gb': 1, 'verification_timeout_seconds': 1200,
                'required_results': ['evidence/gpu_isolation.json']}
        identity = {'host': 'remote', 'git_commit': OLD, 'image_digest': IMAGE, 'job_id': job_id, 'attempt': 1,
                    'gpu_uuids': [uuid], 'verification_purpose': 'gpu_isolation', 'campaign_eligible': False}
        files = {}
        for name, payload in [('evidence/gpu_isolation.json', report),
                              ('provenance.json', {**identity, 'source_sha256': SOURCE,
                                                   'script': spec['script'], 'arguments': args}),
                              ('exit.json', {**identity, 'finished_at': now - 3600, 'code': 0})]:
            files[name] = write(tmp_path, f'runs/{job_id}/{name}', payload)['sha256']
        spec_ref = write(tmp_path, f'runs/remote/jobs/{job_id}-a1.json', spec)
        state[job_id] = {'status': 'done', 'commit': OLD, 'image_digest': IMAGE, 'gpu': uuid,
                         'code': 0, 'gpu_backend': 'cdi', 'verification_purpose': 'gpu_isolation',
                         'campaign_eligible': False, 'exclusive_gpu': True,
                         'spec_path': str(tmp_path / spec_ref['path']),
                         'sync': {'final': True, 'verified': True, 'files': files}}
    return SimpleNamespace(repo=tmp_path, state=state, suite=suite, now=now, reads=reads)


def selected(h):
    return cache.select_reused(h.repo, IMAGE, h.suite, h.state, now=h.now)


def manifest(h):
    def bound(name, report):
        return write(h.repo, name, {'commit': NEW, 'image_digest': IMAGE, 'report': report})
    return {'commit': NEW, 'image_digest': IMAGE, 'gpu_isolation': {}, 'reused_isolation': selected(h),
            'full_suite': bound('suite.json', h.suite),
            'data': bound('data.json', {'identity_sha256': '1' * 64, 'files': [{'sha256': '2' * 64,
                                'target': remote.ROOT + '/data/metaworld/splatter4d_v1/hammer.hdf5'}]}),
            'cross_host': bound('cross.json', {'passed': True, 'step0_identical': True,
                'short_run_within_noise': True, 'data_identity': '1' * 64, 'protocol': 'fixed-seed'})}


def accepted(h, m):
    write(h.repo, 'manifest.json', m)
    return remote.acceptance_manifest({'commit': NEW, 'image_digest': IMAGE, 'evidence_manifest': 'manifest.json'}, h.repo)


def test_all_devices_reuse_old_commit_with_original_checksums_and_cached_source_reads(history):
    h = history
    reused = selected(h)
    assert set(reused) == set(GPUS)
    assert h.reads == [OLD]  # one original-commit integrity check for the whole set
    entry = reused[GPUS[0]]
    assert entry['source_release'] == OLD[:7] + '-cc' and entry['source_commit'] == OLD
    assert entry['verified_at'] == h.now - 3600
    assert entry['device_identity']['child_mapping'][0]['uuid'] == GPUS[0]
    assert cache.read_reference(entry['evidence'], h.repo)[1]['source_sha256'] == SOURCE
    assert set(accepted(h, manifest(h))['reused_isolation']) == set(GPUS)


def test_mixed_fresh_reused_acceptance_and_launcher_package_original_bytes(history):
    h = history
    m = manifest(h)
    uuid = GPUS[-1]
    entry = m['reused_isolation'].pop(uuid)
    report = cache.read_reference(entry['evidence'], h.repo)[1]
    report['source_sha256'] = NEW_SOURCE
    m['gpu_isolation'][uuid] = write(h.repo, 'fresh.json', {'commit': NEW, 'image_digest': IMAGE, 'report': report})
    accepted(h, m)
    artifacts, _, payloads = evidence_artifacts({'id': 'training', 'commit': NEW, 'image_digest': IMAGE,
                                                 'evidence': m}, h.repo)
    assert len(artifacts) == 3 + len(GPUS)
    assert payloads[f'isolation-{GPUS[0]}.json']['source_sha256'] == SOURCE
    assert payloads[f'isolation-{uuid}.json']['source_sha256'] == NEW_SOURCE
    # The launcher independently rejects a forged cache binding, even after admission.
    m['reused_isolation'][GPUS[0]]['evidence']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checksum'):
        evidence_artifacts({'id': 'training', 'commit': NEW, 'image_digest': IMAGE, 'evidence': m}, h.repo)


@pytest.mark.parametrize('field', ['driver_version', 'pid_namespace', 'gpu_backend', 'host_config_sha256',
                                  'mig_config_sha256', 'host'])
def test_changed_current_host_configuration_is_a_cache_miss_and_gate_failure(history, field):
    h = history
    m = manifest(h)
    h.suite['isolation_host_identity'][field] = 'changed'
    assert selected(h) == {}
    m['full_suite'] = write(h.repo, 'suite.json', {'commit': NEW, 'image_digest': IMAGE, 'report': h.suite})
    with pytest.raises(remote.RemoteJobError, match='configuration'):
        accepted(h, m)


@pytest.mark.parametrize('age', [cache.MAX_AGE_SECONDS, cache.MAX_AGE_SECONDS + 1, -1, float('nan'), float('inf')])
def test_age_bound_and_future_or_nonfinite_timestamps_fail_closed(history, age):
    h = history
    m = manifest(h)
    entry = m['reused_isolation'][GPUS[0]]
    path, closeout = cache.read_reference(entry['closeout'], h.repo)
    closeout['finished_at'] = h.now - age
    entry['closeout'] = write(h.repo, path.relative_to(h.repo), closeout)
    entry['verified_at'] = closeout['finished_at']
    with pytest.raises(ValueError, match='timestamp|expired'):
        cache.validate_reuse_set(m, h.repo, h.suite, now=h.now)


@pytest.mark.parametrize('key', ['evidence', 'provenance', 'closeout', 'launch_spec', 'source_commit',
                                'source_release', 'device_uuid', 'image_digest', 'host_identity',
                                'device_identity', 'verified_at'])
def test_missing_any_reuse_binding_fails_closed(history, key):
    h = history
    m = manifest(h)
    m['reused_isolation'][GPUS[0]].pop(key)
    with pytest.raises(remote.RemoteJobError, match='bindings'):
        accepted(h, m)


@pytest.mark.parametrize('key', ['evidence', 'provenance', 'closeout', 'launch_spec'])
def test_mutated_original_files_fail_checksum_even_after_selection(history, key):
    h = history
    m = manifest(h)
    path = h.repo / m['reused_isolation'][GPUS[0]][key]['path']
    path.write_text(path.read_text() + ' ')
    with pytest.raises(remote.RemoteJobError, match='checksum'):
        accepted(h, m)


@pytest.mark.parametrize('fault', ['failed', 'image', 'device', 'native', 'rejection', 'driver', 'namespace', 'source'])
def test_invalid_report_with_recomputed_checksum_still_fails_gate(history, fault):
    h = history
    m = manifest(h)
    entry = m['reused_isolation'][GPUS[0]]
    path, report = cache.read_reference(entry['evidence'], h.repo)
    if fault == 'failed':
        report['passed'] = False
    elif fault == 'image':
        entry['image_digest'] = 'sha256:' + '0' * 64
    elif fault == 'device':
        report['parent_mapping'][0]['uuid'] = GPUS[1]
    elif fault == 'native':
        report['results'][0]['child']['native_egl_passed'] = False
    elif fault == 'rejection':
        report['results'].pop()
    elif fault == 'driver':
        report['results'][0]['full_process_table'] = 'Driver Version: 581.0.0'
    elif fault == 'namespace':
        report['pid_matching']['pid_namespace'] = 'pid:[1]'
    else:
        report['source_sha256']['scripts/worker.py'] = '3' * 64
    entry['evidence'] = write(h.repo, path.relative_to(h.repo), report)
    with pytest.raises(remote.RemoteJobError):
        accepted(h, m)


@pytest.mark.parametrize('fault', ['missing', 'overlap', 'unknown'])
def test_allowlist_must_be_covered_exactly_once(history, fault):
    h = history
    m = manifest(h)
    if fault == 'missing':
        m['reused_isolation'].pop(GPUS[0])
    elif fault == 'overlap':
        m['gpu_isolation'][GPUS[0]] = m['reused_isolation'][GPUS[0]]['evidence']
    else:
        m['reused_isolation']['MIG-unknown'] = m['reused_isolation'][GPUS[0]]
    with pytest.raises(remote.RemoteJobError, match='every approved GPU'):
        accepted(h, m)


def test_unverified_history_and_missing_context_are_cache_misses(history):
    h = history
    first = next(iter(h.state))
    h.state[first]['sync']['verified'] = False
    assert set(selected(h)) == set(GPUS[1:])
    h.suite.pop('isolation_host_identity')
    assert selected(h) == {}


@pytest.mark.parametrize('header', ['NVIDIA UNIX x86_64 Kernel Module  580.95.05',
                                   'NVIDIA UNIX Open Kernel Module for x86_64  580.95.05'])
def test_runtime_identity_reads_open_and_proprietary_driver_headers(monkeypatch, header):
    def read(path):
        assert path == Path('/proc/driver/nvidia/version')
        return header
    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(cache.os, 'readlink', lambda _: 'pid:[4026531836]')
    identity = cache.runtime_identity(NEW_SOURCE, {'NVIDIA_VISIBLE_DEVICES': 'void'})
    assert identity == cache.host_identity(NEW_SOURCE, '580.95.05', 'pid:[4026531836]', 'cdi')


@pytest.mark.parametrize('fault', ['checksum', 'missing', 'driver', 'legacy', 'failed', 'outside', 'symlink'])
def test_invalid_history_becomes_per_device_miss(history, fault):
    h = history
    job_id, current = next(iter(h.state.items()))
    report_path = h.repo / 'runs' / job_id / 'evidence/gpu_isolation.json'
    if fault == 'checksum':
        report_path.write_text('{}')
    elif fault == 'missing':
        report_path.unlink()
    elif fault == 'driver':
        report = json.loads(report_path.read_text())
        report['results'][0]['full_process_table'] = ''
        ref = write(h.repo, report_path.relative_to(h.repo), report)
        current['sync']['files']['evidence/gpu_isolation.json'] = ref['sha256']
    elif fault == 'legacy':
        current['gpu_backend'] = 'legacy'
    elif fault == 'failed':
        current['code'] = 1
    elif fault == 'outside':
        current['spec_path'] = '/etc/passwd'
    else:
        original = report_path.with_suffix('.original')
        report_path.rename(original)
        report_path.symlink_to(original)
    assert set(selected(h)) == set(GPUS[1:])


def test_reused_isolation_does_not_replace_new_native_suite(history):
    h = history
    m = manifest(h)
    suite = copy.deepcopy(h.suite)
    suite['source_sha256'] = SOURCE
    suite['source_sha256_before'] = SOURCE
    m['full_suite'] = write(h.repo, 'suite.json', {'commit': NEW, 'image_digest': IMAGE, 'report': suite})
    with pytest.raises(remote.RemoteJobError, match='source fingerprints'):
        accepted(h, m)


def run_tool(name, root, argv, monkeypatch):
    # Execute the actual CLI body with only ROOT replaced; never import its live entry point.
    source = (REPO / 'ops/campaign' / name).read_text().replace(
        "ROOT = Path('/home/ws/ws/splatter4d')", f'ROOT = Path({str(root)!r})')
    monkeypatch.setattr(sys, 'argv', [name, *argv])
    exec(compile(source, name, 'exec'), {'__name__': 'test_release_cli'})


def test_preparation_schedules_only_cache_misses_and_always_runs_new_native(history, monkeypatch):
    h = history
    from ops.jobs import scheduler
    monkeypatch.setattr(scheduler, 'read_registry', lambda *_: h.state)
    monkeypatch.setattr('ops.containers.verification.validate_development_data', lambda _: None)
    write(h.repo, 'runs/remote/acceptance/226d8ae/data.json', {})
    qpath = h.repo / 'experiments/queue.yaml'
    qpath.parent.mkdir()
    qpath.write_text(yaml.safe_dump({'hosts': {'remote': {'gpu_backend': 'cdi'}}}))
    (h.repo / 'runs/remote/verification').mkdir(parents=True)
    suite = h.repo / 'runs/verify-native-current/evidence/tests.json'
    write(h.repo, suite.relative_to(h.repo), h.suite)
    run_tool('prepare_remote_verification.py', h.repo, [NEW, '--image', IMAGE], monkeypatch)
    native = yaml.safe_load((h.repo / f'runs/remote/verification/native-{NEW[:7]}-queue.yaml').read_text())
    assert len(native['jobs']) == 1 and native['jobs'][0]['verification_purpose'] == 'native_suite'
    next(iter(h.state.values()))['sync']['verified'] = False
    run_tool('prepare_remote_verification.py', h.repo, [NEW, '--image', IMAGE, '--suite', str(suite)], monkeypatch)
    queue = yaml.safe_load((h.repo / f'runs/remote/verification/isolation-{NEW[:7]}-queue.yaml').read_text())
    assert [j['gpu'] for j in queue['jobs']] == [GPUS[0]]
    assert queue['jobs'][0]['id'] == f'verify-isolation-0-{NEW[:7]}'
    plan = json.loads((h.repo / f'runs/remote/acceptance/{NEW[:7]}/isolation-plan.json').read_text())
    assert set(plan['reused_isolation']) == set(GPUS[1:])
    # Restore the final device: an entirely cached isolation stage has no jobs.
    next(iter(h.state.values()))['sync']['verified'] = True
    run_tool('prepare_remote_verification.py', h.repo, [NEW, '--image', IMAGE, '--suite', str(suite)], monkeypatch)
    assert yaml.safe_load((h.repo / f'runs/remote/verification/isolation-{NEW[:7]}-queue.yaml').read_text())['jobs'] == []


def test_reservation_keeps_original_job_indices_for_cache_misses():
    resources = {'mem_gb': 4, 'load': 0, 'ram_gb': 8, 'cpu_threads': 8, 'disk_gb': 1}
    doc = reservation.plan(NEW, NEW[:7], 'isolation', GPUS[0], GPUS, native=resources, isolation=resources,
                           isolation_gpus=[GPUS[2]])
    reservation.validate(doc)
    assert set(doc['gpus']) == {GPUS[2]} and doc['jobs'] == [f'verify-isolation-2-{NEW[:7]}']
    assert doc['ram_gb'] == 8 and doc['cpu_threads'] == 8


def test_actual_assembly_combines_fresh_and_reused_evidence(history, monkeypatch):
    h = history
    from ops.jobs import scheduler
    monkeypatch.setattr(scheduler, 'read_registry', lambda *_: h.state)
    monkeypatch.setattr('ops.containers.verification.validate_development_data', lambda _: None)
    base = manifest(h)
    for name, key in [('data.json', 'data'), ('cross_host.json', 'cross_host')]:
        _, payload = cache.read_reference(base[key], h.repo)
        write(h.repo, 'runs/remote/acceptance/226d8ae/' + name, payload)
    queue = h.repo / 'experiments/queue.yaml'
    queue.parent.mkdir()
    queue.write_text(yaml.safe_dump({'hosts': {'remote': {'gpu_backend': 'cdi'}}}))
    (h.repo / 'runs/remote/verification').mkdir(parents=True)
    native = f'verify-native-{NEW[:7]}'
    suite_ref = write(h.repo, f'runs/{native}/evidence/tests.json', h.suite)
    h.state[native] = {'status': 'done', 'commit': NEW, 'image_digest': IMAGE,
                       'sync': {'final': True, 'verified': True, 'files': {'evidence/tests.json': suite_ref['sha256']}}}
    # One old device has no reusable closeout. Only it needs a new exclusive check.
    old_id = next(iter(h.state))
    h.state[old_id]['sync']['verified'] = False
    run_tool('prepare_remote_verification.py', h.repo, [NEW, '--image', IMAGE,
             '--suite', str(h.repo / suite_ref['path'])], monkeypatch)
    fresh_id = f'verify-isolation-0-{NEW[:7]}'
    report = json.loads((h.repo / f'runs/{old_id}/evidence/gpu_isolation.json').read_text())
    report['source_sha256'] = NEW_SOURCE
    fresh_ref = write(h.repo, f'runs/{fresh_id}/evidence/gpu_isolation.json', report)
    h.state[fresh_id] = {'status': 'done', 'commit': NEW, 'image_digest': IMAGE,
                         'sync': {'final': True, 'verified': True,
                                  'files': {'evidence/gpu_isolation.json': fresh_ref['sha256']}}}
    run_tool('assemble_remote_acceptance.py', h.repo, [NEW, '--image', IMAGE], monkeypatch)
    result = json.loads((h.repo / f'runs/remote/acceptance/{NEW[:7]}/manifest.json').read_text())
    assert set(result['gpu_isolation']) == {GPUS[0]}
    assert set(result['reused_isolation']) == set(GPUS[1:])
    assert result['full_suite']['path'].startswith(f'runs/remote/acceptance/{NEW[:7]}/')
    accepted(h, result)


def test_release_chain_with_all_reused_devices_drives_only_the_new_native_job(history, monkeypatch):
    h = history
    source = (REPO / 'ops/campaign/release_chain.py').read_text().replace(
        "ROOT = Path('/home/ws/ws/splatter4d')", f'ROOT = Path({str(h.repo)!r})')
    names = {'__name__': 'test_release_chain'}
    exec(compile(source, 'release_chain.py', 'exec'), names)
    monkeypatch.setitem(names, 'commit_reader', lambda *_: (None, None))
    monkeypatch.setattr(names['suite_partition'], 'release_support', lambda _: {'compatible': True})
    native_id = f'verify-native-{NEW[:7]}'
    write(h.repo, f'runs/{native_id}/evidence/tests.json', h.suite)
    native_q, isolation_q = h.repo / 'native.yaml', h.repo / 'isolation.yaml'
    native_q.write_text(yaml.safe_dump({'jobs': [{'gpu': GPUS[0], 'mem_gb': 12, 'ram_gb': 96}]}))
    isolation_q.write_text(yaml.safe_dump({'jobs': []}))
    calls = []
    def run(cmd):
        calls.append(cmd)
        if 'prepare_remote_verification.py' in cmd[2]:
            return str(isolation_q if '--suite' in cmd else native_q)
        return 'ACCEPTED'
    monkeypatch.setitem(names, 'run', run)
    release = names['Release'](['release_chain', NEW, '--skip-sync', '--native-gpu', GPUS[0]])
    driven, stages = [], []
    release.drive = lambda q, ids: driven.append(ids)
    release.publish = stages.append
    release.start_local_partition = lambda: None
    release.finish_local_partition = lambda: None
    release.write_timing = lambda _: None
    release.clear = release._stop.set
    release.main()
    assert driven == [[native_id]] and stages == ['native', 'native', 'isolation']
    assert release.isolation_gpus == []
    assert any('assemble_remote_acceptance.py' in cmd[2] for cmd in calls)
