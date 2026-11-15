"""Integrity, science gate, scheduler integration and atomic publication refusals."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from ops.jobs import local_phase_a as a
from ops.jobs import local_phase_a_gate as gate
from ops.jobs import host_roles, realworld_local

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 's4d').is_dir())


def tool(name):
    p = ROOT / 'runs/setup/tools' / name
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(p.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(p.parent))
    return mod


@pytest.mark.parametrize('bad', ['covering-000000', 'covering-success-../../foo', 'mug-hanging-success-episode_000000',
                                 'drawer-open-success-episode_000000', 'covering-discarded-episode_000000'])
def test_identity_rejects_old_ambiguous_or_unsafe_names(bad):
    with pytest.raises(ValueError):
        a.identity(bad)


def test_success_and_failure_same_number_have_distinct_paths():
    assert a.identity('drawer-open-v2-success-episode_000002') != a.identity('drawer-open-v2-failure-episode_000002')


def test_depth_absolute_relative_validity_and_nonfinite():
    r = np.array([[[1., 2., 0.]]], np.float32)
    c = np.array([[[1.001, 2., 0.]]], np.float32)
    m = gate.metrics_depth(r, c)
    assert .00049 < m['mean_abs_m'] < .00051
    assert .00098 < m['p99_rel'] < .00101
    assert m['valid_iou'] == 1 and not m['bitwise']
    assert gate.metrics_depth(r, r)['bitwise']
    with pytest.raises(ValueError, match='nonfinite'):
        gate.metrics_depth(r, np.full_like(r, np.nan))


def test_mask_gate_exposes_missing_foreground_without_empty_class_inflation():
    r = np.array([[[1, 1], [0, 0]]], np.uint8)
    c = np.array([[[1, 0], [0, 0]]], np.uint8)
    m = gate.metrics_masks(r, c)
    assert m['mean_iou'] == .5 and m['min_iou'] == .5 and m['absent_both'] == 2


def test_readonly_inventory_detects_content_changes_and_symlinks(tmp_path):
    frozen = tmp_path / 'frozen'
    (frozen/'source').mkdir(parents=True)
    f = frozen/'source/code.py'; f.write_text('first')
    doc = {'files': a.inventory(frozen), 'frozen': str(frozen), 'commit': 'pin', 'upstream_pins': {}, 'environment_files': {}}
    a.atomic(tmp_path/'envelope.json', doc)
    checksum = a.sha(tmp_path/'envelope.json')
    a.readonly(frozen)
    from unittest.mock import patch
    with patch.object(a, 'git', return_value='pin'):
        assert a.verify_envelope(tmp_path, checksum) == doc
        f.chmod(0o644); f.write_text('second'); f.chmod(0o444)
        with pytest.raises(ValueError, match='SHA drift'):
            a.verify_envelope(tmp_path, checksum)
    frozen.chmod(0o755)
    (frozen/'link').symlink_to(f)
    with pytest.raises(ValueError, match='symlink'):
        a.inventory(frozen)


@pytest.mark.parametrize('coverage,qualified', [('cpu-only', True), ('full-two-episodes', False), ('short-smoke', True)])
def test_partial_or_failed_gate_never_authorizes_bulk(tmp_path, coverage, qualified):
    (tmp_path/'envelope.json').write_text('{}')
    d = {'qualified': qualified, 'coverage': coverage, 'envelope_sha256': a.sha(tmp_path/'envelope.json'), 'episodes': []}
    a.atomic(tmp_path/'equivalence.json', d)
    with pytest.raises(ValueError):
        a.require_gate(tmp_path, a.sha(tmp_path/'equivalence.json'))


def test_local_preprocess_exception_still_requires_registered_verified_inputs(tmp_path):
    row = {'id': 'rwv2-20261113-prep-ours-covering-failure-episode_000001-localA', 'script': a.SCRIPT,
           'args': [], 'role': 'realworld_preprocess'}
    roles = host_roles.load_roles()
    assert realworld_local.has_resolver(row)
    assert host_roles.refusal(row, 'local', roles, {}) is None
    with pytest.raises(realworld_local.LocalInputError, match='invalid phase-A argv'):
        realworld_local._inputs(row, {}, tmp_path, tmp_path/'missing.json', tmp_path)
    assert host_roles.refusal({**row, 'script': 'scripts/realworld_ingest.py'}, 'local', roles, {})


def test_rows_preserve_priority_profile_labels_and_never_launch(tmp_path):
    reg = tool('register_local_phase_a_20261115.py')
    ep = 'covering-failure-episode_000001'
    payload = {'episode': ep, 'envelope_sha256': 'e'*64}
    rows = reg.build_rows([payload], tmp_path, {'commit': 'a'*40, 'image': 'sha256:test', 'evidence_manifest': 'suite'}, 'f'*64, {})
    r = rows[0]
    assert r['priority'] == -4 and r['gpu_profile'] == '1g.24gb' and r['host'] == 'remote'
    assert r['local_placement_pause'] == reg.PAUSE and r['max_restarts'] == 0
    assert r['args'][-1] == 'f'*64 and r['cpu_threads'] == 4


def archive(path, data, unsafe=False):
    with tarfile.open(path, 'w') as tar:
        for name, body in data.items():
            info = tarfile.TarInfo('../escape' if unsafe else name); info.size = len(body)
            tar.addfile(info, io.BytesIO(body))


def installer(tmp_path):
    mod = tool('upload_local_phase_a_20261115.py')
    return mod.REMOTE_INSTALL.replace('/home/compu/kaist/sunho/realworld/processed/dsreset-20261113/', str(tmp_path/'remote') + '/')


def test_atomic_publication_checks_every_file_and_refuses_existing_conflict(tmp_path):
    body = b'output'
    manifest = {'files': {'data': hashlib.sha256(body).hexdigest()}}
    packed = tmp_path/'episode.tar'
    archive(packed, {'data': body, '.publication.json': json.dumps(manifest).encode()})
    target = tmp_path/'remote/covering/success/episode_000000'
    cmd = [sys.executable, '-I', '-c', installer(tmp_path), str(packed), str(target), a.sha(packed)]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    assert (target/'data').read_bytes() == body
    assert subprocess.run(cmd, capture_output=True).returncode == 0  # idempotent
    (target/'data').write_bytes(b'canonical-conflict')
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    assert (target/'data').read_bytes() == b'canonical-conflict'


@pytest.mark.parametrize('unsafe,wrong_digest', [(True, False), (False, True)])
def test_publication_rejects_escape_and_wrong_payload_sha(tmp_path, unsafe, wrong_digest):
    packed = tmp_path/'bad.tar'
    manifest = {'files': {'data': '0'*64}}
    archive(packed, {'data': b'data', '.publication.json': json.dumps(manifest).encode()}, unsafe)
    target = tmp_path/'remote/covering/success/episode_000000'
    result = subprocess.run([sys.executable, '-I', '-c', installer(tmp_path), str(packed), str(target), a.sha(packed)], capture_output=True)
    assert result.returncode != 0 and not target.exists() and not (tmp_path/'escape').exists()


def test_sam_raw_instance_counts_and_scores_are_part_of_gate(tmp_path):
    import h5py
    paths = [tmp_path/'a.h5', tmp_path/'b.h5']
    for path in paths:
        with h5py.File(path, 'w') as f:
            group = f.create_group('chunks/000000')
            for key in ('frame', 'prompt_index', 'instance_id', 'count'):
                group[key] = np.array([0], np.int32)
            group['masks'] = np.ones((1, 2, 2), bool)
            group['scores'] = np.array([.9], np.float32)
            group['boxes_xyxy'] = np.array([[0., 0., 2., 2.]], np.float32)
    with h5py.File(paths[0], 'r') as r, h5py.File(paths[1], 'r+') as c:
        assert gate.compare_instances(r, c)['min_iou'] == 1
        c['chunks/000000/scores'][0] = .5
        with pytest.raises(ValueError, match='scores tolerance'):
            gate.compare_instances(r, c)


@pytest.mark.parametrize('returncode', [0, 7])
def test_gate_real_child_heartbeat_and_quiet_exit_are_persisted(tmp_path, capsys, returncode):
    frozen = tmp_path/'source'
    package = frozen/'ops/jobs'
    package.mkdir(parents=True)
    (frozen/'ops/__init__.py').write_text('')
    (package/'__init__.py').write_text('')
    (package/'local_phase_a.py').write_text(
        'import time\ndef execute(*args, **kwargs):\n'
        f'    time.sleep(.1)\n    raise SystemExit({returncode})\n')
    job = tmp_path/'gate-test.json'
    job.write_text('{}')
    result = gate.run_job(tmp_path, job, frozen, a.STAGES, heartbeat_s=.02)
    persisted = json.loads((tmp_path/'execution/gate-test.json').read_text())
    assert persisted == result
    assert result['returncode'] == returncode
    assert result['status'] == ('ok' if returncode == 0 else 'failed')
    assert result['finished_unix'] >= result['started_unix']
    output = capsys.readouterr().out
    assert 'gate_episode_started' in output and 'gate_episode_running' in output
    assert f'"returncode": {returncode}' in output


@pytest.mark.parametrize('returncode', [0, 7, -9])
def test_gate_child_failure_cannot_qualify_cached_success(tmp_path, monkeypatch, returncode):
    (tmp_path/'envelope.json').write_text('{}')
    monkeypatch.setattr(a, 'verify_envelope', lambda *args: {'frozen': str(tmp_path), 'files': {'source/ops/jobs/local_phase_a.py': a.sha(a.__file__)}})
    jobs = [tmp_path/'one.json', tmp_path/'two.json']
    monkeypatch.setattr(gate, 'gate_jobs', lambda _: jobs)
    observed = []
    def run(*args):
        observed.append(json.loads((tmp_path/'equivalence.json').read_text()))
        return {'returncode': returncode}
    monkeypatch.setattr(gate, 'run_job', run)
    monkeypatch.setattr(gate, 'compare_episode', lambda task, *args: {
        'task': task, 'episode': task, 'passed': True, 'stages': {}})
    assert gate.main(['--bundle', str(tmp_path)]) == (0 if returncode == 0 else 3)
    result = json.loads((tmp_path/'equivalence.json').read_text())
    assert result['qualified'] is (returncode == 0)
    assert observed[0]['coverage'] == 'in-progress' and not observed[0]['qualified']
    assert not observed[1]['qualified']  # one completed episode is insufficient
    assert all(row['execution']['returncode'] == returncode for row in result['episodes'])


def test_gate_launch_failure_is_recorded(tmp_path, monkeypatch):
    job = tmp_path/'job.json'
    job.write_text('{}')
    def fail(*args, **kwargs):
        raise FileNotFoundError('missing interpreter')
    monkeypatch.setattr(gate.subprocess, 'Popen', fail)
    with pytest.raises(FileNotFoundError, match='missing interpreter'):
        gate.run_job(tmp_path, job, tmp_path, a.STAGES)
    record = json.loads((tmp_path/'execution/job.json').read_text())
    assert record['status'] == 'launch-failed' and 'missing interpreter' in record['error']


def test_native_cache_loads_compiled_library_from_private_executable_ram():
    import ctypes
    import os
    with a.native_code_cache() as (cache, report):
        path = cache/'smoke.so'
        result = subprocess.run(['/usr/bin/gcc', '-shared', '-fPIC', '-x', 'c', '-o', str(path), '-'],
                                input='int local_phase_a_answer(void) { return 42; }',
                                text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert ctypes.CDLL(str(path)).local_phase_a_answer() == 42
        assert report['ok'] and report['executable']
        assert cache.stat().st_mode & 0o777 == 0o700
        assert not os.statvfs(cache).f_flag & os.ST_NOEXEC
    assert not cache.exists()


def test_native_cache_rejects_noexec_shared_memory():
    with pytest.raises(ValueError, match='noexec'):
        a.executable_ram('/dev/shm')


def test_gate_refuses_stale_frozen_runner_before_any_spawn(tmp_path, monkeypatch):
    (tmp_path/'envelope.json').write_text('{}')
    monkeypatch.setattr(a, 'verify_envelope', lambda *args: {
        'files': {'source/ops/jobs/local_phase_a.py': '0'*64}})
    def unexpected(*args):
        pytest.fail('must refuse before creating jobs or starting any process')
    monkeypatch.setattr(gate, 'gate_jobs', unexpected)
    with pytest.raises(ValueError, match='runner differs from frozen source'):
        gate.main(['--bundle', str(tmp_path)])


def test_prompt_order_survives_sorted_job_json_and_pipeline_yaml(tmp_path):
    import yaml
    order = ['robot arm', 'end effector', 'blue towel', 'tin can']
    prompts = dict(zip(order, ['robot', 'robot', 'object', 'fixture']))
    cfg = {'adapter': {'config': {'episodes': {
        'covering-success-episode_000000': '/home/ws/data/real_world/covering/episode_000000.h5'}}},
        'stages': {'sam3_masks': {'prompts': prompts, 'chunk': 228, 'mode': 'segment_video'}}}
    config = tmp_path/'pipeline.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (tmp_path/'envelope.json').write_text('{}')
    job = a.make_job({'episode': 'covering-success-episode_000000', 'config': str(config),
                      'raw_sha256': 'a'*64}, tmp_path)
    path = tmp_path/'job.json'
    a.atomic(path, job)
    decoded = json.loads(path.read_text())
    assert list(decoded['pipeline']['stages']['sam3_masks']['prompts']) != order
    restored = yaml.safe_load(yaml.safe_dump(a.execution_pipeline(decoded), sort_keys=False))
    assert list(restored['stages']['sam3_masks']['prompts']) == order
    assert restored['stages']['sam3_masks']['chunk'] == 228
    assert restored['stages']['sam3_masks']['mode'] == 'segment_video'
    assert list(decoded['pipeline']['stages']['sam3_masks']['prompts']) == sorted(order)


@pytest.mark.parametrize('order', [None, ['robot arm'], ['robot arm', 'robot arm'], ['other', 'robot arm']])
def test_missing_duplicate_or_wrong_prompt_order_is_rejected(order):
    job = {'pipeline': {'stages': {'sam3_masks': {'prompts': {'robot arm': 'robot', 'blue towel': 'object'}}}},
           'sam3_prompt_order': order}
    with pytest.raises(ValueError, match='prompt order'):
        a.execution_pipeline(job)
