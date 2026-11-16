"""Coordinator preparation for immutable runtimes with a one-repin guard.

Executed on the host only, after admission and deterministic-container checks.
A durable receipt preserves the exact old provenance and authorized target view.
Preparing a view records authorization, never claims that target code has run.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def prepare(path: Path, events: list[dict], data_identity: str, job_sha256: str, invocation: dict | None = None) -> None:
    """Advance only release fields; leave data/config identity for the runtime to verify.

    The registry chain is validated by the coordinator before shipping this code.
    Every historical runtime transition must match a prefix of that chain. Missing
    intermediate launches are allowed; their authorizations remain in the receipt.
    """
    if len(events) < 2:
        raise ValueError("provenance preparation requires a chained re-pin")
    last = events[-1]
    job_id, image = last['id'], last['image_digest']
    if last.get('job_sha256') != job_sha256:
        raise ValueError('registry fingerprint differs')
    # Refuse symlinks in the run directory, including receipt storage.
    for node in [path, *path.parents]:
        if node.is_symlink():
            raise ValueError("symlink in provenance path")
    previous_bytes = path.read_bytes()
    previous = json.loads(previous_bytes)
    if (previous.get('host') != 'remote' or previous.get('campaign_eligible') is not True
            or previous.get('job_id') != job_id or previous.get('image_digest') != image
            or previous.get('data_id') != data_identity
            or any(e.get('job_sha256', job_sha256) != job_sha256 for e in events)):
        raise ValueError("provenance identity differs from recorded run")
    if (previous.get('job_id') != job_id or previous.get('image_digest') != image
            or previous.get('data_id') != data_identity or previous.get('host') != 'remote'
            or previous.get('campaign_eligible') is not True):
        raise ValueError('runtime provenance identity differs')
    if invocation and any(previous.get(k) != v for k, v in invocation.items()):
        raise ValueError('runtime invocation differs')
    history = previous.get('release_history', [])
    expected = [{'from_commit': e['from_commit'], 'from_image_digest': e['from_image_digest'],
                 'to_commit': e['commit'], 'to_image_digest': e['image_digest']} for e in events]
    if not isinstance(history, list) or len(history) > len(events):
        raise ValueError("invalid runtime release history")
    for actual, transition in zip(history, expected):
        if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in transition.items()):
            raise ValueError("runtime release history differs from registry")
    current = events[len(history)-1]['commit'] if history else events[0]['from_commit']
    if previous.get('git_commit') != current:
        raise ValueError("runtime release history tail differs")
    receipts = path.parent / '.release-repins'
    if receipts.is_symlink():
        raise ValueError("symlink in receipt path")
    receipts.mkdir(exist_ok=True)
    binding = {'events': events, 'data_identity': data_identity, 'job_sha256': job_sha256}
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    receipt = receipts / (digest + '.json')
    if receipt.is_symlink():
        raise ValueError("symlink receipt")
    if receipt.exists():
        saved = json.loads(receipt.read_bytes())
        if saved['binding'] == binding and len(history) == len(events):
            prepared = json.loads(saved['prepared'])
            keys = ('host', 'campaign_eligible', 'git_commit', 'image_digest', 'data_id', 'config_sha256',
                    'job_id', 'release_history', 'release_preparation')
            if any(previous.get(k) != prepared.get(k) for k in keys):
                raise ValueError('prepared provenance identity changed')
            return  # Entry/trainer may have updated attempt/start metadata after preparation.
        if saved['binding'] != binding or previous_bytes.decode() not in (saved['before'], saved['prepared']):
            raise ValueError("provenance changed after preparation")
        prepared_bytes = saved['prepared'].encode()
    else:
        if len(history) == len(events):
            raise ValueError("prepared provenance has no receipt")
        prepared = {**previous, 'git_commit': last['commit'],
                    'release_history': [*history, *[{**item, 'prepared_by': 'coordinator',
                                                   'registry_attempt': event.get('attempt')}
                                                  for item, event in zip(expected[len(history):], events[len(history):])]],
                    'release_preparation': {'receipt': str(receipt), 'binding_sha256': digest,
                                            'status': 'authorized; runtime start not asserted'}}
        prepared_bytes = (json.dumps(prepared, indent=1, sort_keys=True) + '\n').encode()
        payload = {'binding': binding, 'before': previous_bytes.decode(), 'prepared': prepared_bytes.decode()}
        # Receipt is durable before replacing provenance; a retry can finish an interrupted preparation.
        with receipt.open('x') as stream:
            stream.write(json.dumps(payload, indent=1, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(receipts, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    if previous_bytes != prepared_bytes:
        fd, name = tempfile.mkstemp(prefix='.repin-', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(prepared_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(name):
                os.unlink(name)
