"""Bounded reuse of original, checksum-verified one-device CUDA/EGL evidence.

The old report, launch spec, provenance and exit record stay immutable. In particular
the exit timestamp, rather than the time a release reuses it, starts the seven days.
Legacy reports carry the driver in full_process_table and the host PID namespace;
their source fingerprints bind the remote host/MIG configuration, and the launch
spec binds CDI. Missing identities are cache misses, never permission to skip a check.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from functools import lru_cache
from pathlib import Path

from ops.containers.verification import validate_report, validate_verification
from ops.resources.gpu_guard import APPROVED_HOST_GPUS

MAX_AGE_SECONDS = 7 * 24 * 3600
HOST_CONFIG = 'configs/hosts/remote.yaml'
MIG_CONFIG = 'configs/hosts/remote-mig.yaml'
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
COMMIT = re.compile(r'[0-9a-f]{40}\Z')


def _path(path: Path, repo: Path) -> Path:
    path = (repo / path).absolute()
    if repo.resolve() not in path.parents or path.resolve() != path or not path.is_file():
        raise ValueError('Isolation evidence must be a symlink-free repository file')
    return path


def reference(path: Path, repo: Path) -> dict:
    path = _path(path, repo)
    return {'path': str(path.relative_to(repo.resolve())),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read_reference(ref: dict, repo: Path) -> tuple[Path, dict]:
    if (not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}
            or not isinstance(ref['path'], str) or not ref['path']
            or not isinstance(ref['sha256'], str) or not HEX64.fullmatch(ref['sha256'])):
        raise ValueError('Isolation evidence requires path and SHA256 bindings')
    path = _path(Path(ref['path']), repo)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref['sha256']:
        raise ValueError('Isolation evidence checksum mismatch')
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError('Isolation evidence must be a JSON object')
    return path, data


def host_identity(source: dict, driver: str, namespace: str, backend: str) -> dict:
    if (not isinstance(driver, str) or not re.fullmatch(r'\d+(?:\.\d+)+', driver)
            or not isinstance(namespace, str) or not re.fullmatch(r'pid:\[\d+\]', namespace)
            or backend != 'cdi' or not HEX64.fullmatch(str(source.get(HOST_CONFIG, '')))
            or (MIG_CONFIG in source and not HEX64.fullmatch(str(source[MIG_CONFIG])))):
        raise ValueError('Isolation host/driver/CDI identity is incomplete')
    return {'host': 'remote', 'driver_version': driver, 'pid_namespace': namespace, 'gpu_backend': backend,
            'host_config_sha256': source[HOST_CONFIG], 'mig_config_sha256': source.get(MIG_CONFIG)}


def runtime_identity(source: dict, env: dict) -> dict:
    """Read-only identity inside the new native verification container; no GPU work."""
    version = Path('/proc/driver/nvidia/version').read_text()
    driver = re.search(r'Kernel Module(?:\s+for\s+\S+)?\s+(\d+(?:\.\d+)+)', version)
    return host_identity(source, driver[1] if driver else '', os.readlink('/proc/self/ns/pid'),
                         'cdi' if env.get('NVIDIA_VISIBLE_DEVICES') == 'void' else '')


def report_identity(report: dict, spec: dict) -> dict:
    positives = [item for item in report.get('results', []) if item.get('uuid') == spec['gpu']]
    table = positives[0].get('full_process_table', '') if len(positives) == 1 else ''
    driver = re.search(r'Driver Version:\s*(\d+(?:\.\d+)+)', table)
    return host_identity(report['source_sha256'], driver[1] if driver else '',
                         report.get('pid_matching', {}).get('pid_namespace', ''), spec.get('gpu_backend'))


def _source_reader(repo):
    from ops.jobs.remote_backend import commit_source_sha256
    # Recheck each original commit once per gate invocation, not once per device.
    return lru_cache(maxsize=None)(lambda commit: commit_source_sha256(repo, commit))


def validate_reused(entry: dict, repo: Path, uuid: str, image: str, suite: dict, *, now=None, source_reader=None) -> dict:
    """Validate every binding without consulting mutable registry state. Raises ValueError."""
    required = {'source_commit', 'source_release', 'image_digest', 'device_uuid', 'verified_at', 'host_identity',
                'device_identity', 'evidence', 'provenance', 'closeout', 'launch_spec'}
    if (not isinstance(entry, dict) or set(entry) != required or entry['device_uuid'] != uuid
            or entry['image_digest'] != image or not COMMIT.fullmatch(str(entry['source_commit']))):
        raise ValueError('Reused isolation identity/bindings are incomplete')
    _, report = read_reference(entry['evidence'], repo)
    _, provenance = read_reference(entry['provenance'], repo)
    _, closeout = read_reference(entry['closeout'], repo)
    _, spec = read_reference(entry['launch_spec'], repo)
    source_commit = entry['source_commit']
    job_id = spec.get('id', '')
    if (not re.fullmatch(r'verify-isolation-\d+-' + re.escape(str(entry['source_release'])), job_id)
            or not str(entry['source_release']).startswith(source_commit[:7])
            or spec.get('commit') != source_commit or spec.get('image_digest') != image or spec.get('gpu') != uuid
            or spec.get('verification_purpose') != 'gpu_isolation' or spec.get('exclusive_gpu') is not True):
        raise ValueError('Reused isolation source release/launch identity differs')
    validate_verification(spec)
    for record in (provenance, closeout):
        if (record.get('host') != 'remote' or record.get('git_commit') != source_commit
                or record.get('image_digest') != image or record.get('gpu_uuids') != [uuid]
                or record.get('job_id') != job_id or record.get('attempt') != spec.get('attempt')
                or record.get('verification_purpose') != 'gpu_isolation' or record.get('campaign_eligible') is not False):
            raise ValueError('Reused isolation closeout provenance differs')
    if (type(closeout.get('code')) is not int or closeout['code'] != 0
            or provenance.get('script') != spec['script'] or provenance.get('arguments') != spec['args']):
        raise ValueError('Reused isolation did not close successfully')
    finished = closeout.get('finished_at')
    now = time.time() if now is None else now
    if (type(finished) not in (int, float) or not math.isfinite(finished)
            or not 0 <= now - finished < MAX_AGE_SECONDS or entry['verified_at'] != finished):
        raise ValueError('Reused isolation is expired or has an invalid timestamp')
    source = (source_reader or _source_reader(repo))(source_commit)
    if provenance.get('source_sha256') != source:
        raise ValueError('Reused isolation sources differ from the original commit')
    validate_report('gpu_isolation', report, source, source_commit, image, uuid)
    identity = report_identity(report, spec)
    current = suite.get('isolation_host_identity')
    # Verify the current identity is also bound to the new suite's source configuration.
    if (not isinstance(current, dict) or current != identity or entry['host_identity'] != identity
            or current != host_identity(suite['source_sha256'], current.get('driver_version'),
                                        current.get('pid_namespace'), current.get('gpu_backend'))):
        raise ValueError('Reused isolation host/MIG/driver/CDI configuration differs')
    positive = next(item for item in report['results'] if item.get('uuid') == uuid)
    device = {'parent_mapping': report['parent_mapping'], 'child_mapping': positive['child']['mapping'],
              'egl_device': positive['child'].get('egl_device')}
    if entry['device_identity'] != device:
        raise ValueError('Reused isolation device mapping differs')
    return report


def validate_reuse_set(manifest: dict, repo: Path, suite: dict, *, now=None) -> dict:
    """Fresh and reused sets must partition the allowlist; never hide an unbound reuse."""
    fresh, reused = manifest.get('gpu_isolation', {}), manifest.get('reused_isolation', {})
    if (not isinstance(fresh, dict) or not isinstance(reused, dict) or set(fresh) & set(reused)
            or set(fresh) | set(reused) != set(APPROVED_HOST_GPUS['remote'])):
        raise ValueError('Isolation evidence must cover every approved GPU exactly once')
    reader = _source_reader(repo)
    return {uuid: validate_reused(entry, repo, uuid, manifest['image_digest'], suite, now=now, source_reader=reader)
            for uuid, entry in reused.items()}


def select_reused(repo: Path, image: str, suite: dict, state: dict, *, now=None) -> dict:
    """Only successful scheduler closeouts can enter the cache. Missing/old history is a miss."""
    if not suite.get('isolation_host_identity'):
        return {}
    selected = {}
    reader = _source_reader(repo)
    for job_id, current in state.items():
        sync = current.get('sync', {})
        uuid = current.get('gpu')
        if (not re.fullmatch(r'verify-isolation-\d+-.+', job_id) or uuid not in APPROVED_HOST_GPUS['remote']
                or current.get('status') != 'done' or current.get('image_digest') != image
                or current.get('code') != 0 or current.get('gpu_backend') != 'cdi'
                or current.get('verification_purpose') != 'gpu_isolation'
                or current.get('campaign_eligible') is not False or current.get('exclusive_gpu') is not True
                or sync.get('final') is not True or sync.get('verified') is not True):
            continue
        try:
            paths = {'evidence': repo / 'runs' / job_id / 'evidence/gpu_isolation.json',
                     'provenance': repo / 'runs' / job_id / 'provenance.json',
                     'closeout': repo / 'runs' / job_id / 'exit.json', 'launch_spec': Path(current['spec_path'])}
            refs = {key: reference(path, repo) for key, path in paths.items()}
            for key in ('evidence', 'provenance', 'closeout'):
                name = str(paths[key].relative_to(repo / 'runs' / job_id))
                if sync.get('files', {}).get(name) != refs[key]['sha256']:
                    raise ValueError('Isolation closeout checksum differs from scheduler sync')
            _, report = read_reference(refs['evidence'], repo)
            _, closeout = read_reference(refs['closeout'], repo)
            _, spec = read_reference(refs['launch_spec'], repo)
            validate_report('gpu_isolation', report, reader(current['commit']), current['commit'], image, uuid)
            positive = next(item for item in report['results'] if item.get('uuid') == uuid)
            entry = {**refs, 'source_commit': current['commit'], 'source_release': job_id.split('-', 3)[3],
                     'image_digest': image, 'device_uuid': uuid, 'verified_at': closeout['finished_at'],
                     'host_identity': report_identity(report, spec),
                     'device_identity': {'parent_mapping': report['parent_mapping'],
                                         'child_mapping': positive['child']['mapping'],
                                         'egl_device': positive['child'].get('egl_device')}}
            validate_reused(entry, repo, uuid, image, suite, now=now, source_reader=reader)
            if uuid not in selected or entry['verified_at'] > selected[uuid]['verified_at']:
                selected[uuid] = entry
        except (OSError, ValueError, KeyError, TypeError, StopIteration, subprocess.CalledProcessError):
            continue
    return selected
