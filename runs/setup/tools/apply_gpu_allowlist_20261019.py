"""Local GPU authorization change (user, 2026-10-19): local schedulable devices are the eight MIG 1g.24gb instances of
GPU 4 (GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce) and GPU 5 (GPU-d09f0338-71b9-d915-3c7f-e99754a3b639); GPU 6
(GPU-6b33eef2-80c5-547e-6a61-7ab81c14d63b) and its four MIG instances are no longer authorized. Remote unchanged.

Immediate admission stop: removes the GPU 6 MIG limits from experiments/queue.yaml (the scheduler only places local
jobs on devices with a limit entry; jobs.load_queue accepts a subset of the guard's approved set). No job definition
changes. The guard/host-config/watcher allowlist change follows as a verified source merge.

    python -I apply_gpu_allowlist_20261019.py [--publish]
"""
import fcntl
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

PUBLISH = '--publish' in sys.argv
SETUP = ROOT / 'runs/setup'
MANIFEST = SETUP / 'gpu-allowlist-20261019.json'
AUTHORIZED = {
    'GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce': ('MIG-8da55219-bd6b-5260-8be1-3b2a76ac733c',
                                                 'MIG-c4ec2a95-30cc-5398-82bf-d9b11787fc70',
                                                 'MIG-423203d9-5801-5892-a489-30f16a5a2d4a',
                                                 'MIG-02b55a66-ed41-53c2-9004-c3f5619d6369'),
    'GPU-d09f0338-71b9-d915-3c7f-e99754a3b639': ('MIG-05a151e7-c6bd-5344-91cb-8bfb3a342631',
                                                 'MIG-b50e85c3-8498-5ef0-906e-5eda9db34b90',
                                                 'MIG-2ebd9e11-f2e0-5e22-870a-7779d44ec10a',
                                                 'MIG-c3511b41-5f78-52f1-9b93-9c6794cdbbdd'),
}
REVOKED = {'GPU-6b33eef2-80c5-547e-6a61-7ab81c14d63b': ('MIG-f30be823-f40d-5292-9474-427746dd3703',
                                                        'MIG-908fda99-f524-5319-ae5a-9f534a881ee1',
                                                        'MIG-3b813dc6-21e9-5e2e-bc3c-4aa8659b51a8',
                                                        'MIG-42dd67ec-ecc1-5c8f-b9a2-22e9a48a8c29')}
KEEP = [m for migs in AUTHORIZED.values() for m in migs]
DROP = [m for migs in REVOKED.values() for m in migs]

with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    state = jobs.read_registry()
    occupying = {k: v.get('gpu') for k, v in state.items()
                 if v.get('host', 'local') == 'local' and v['status'] in jobs.ACTIVE and v.get('gpu') in DROP}
    before = jobs.QUEUE.read_bytes()
    lines = before.decode().splitlines(keepends=True)
    out, skip = [], 0
    for line in lines:
        if skip:
            skip -= 1
            continue
        if line.startswith('  ') and line.strip().rstrip(':') in DROP:
            skip = 2  # max_jobs, max_mem_gb
            continue
        out.append(line)
    text = ''.join(out)
    new, old = yaml.safe_load(text), yaml.safe_load(before)
    assert list(new['limits']) == KEEP and set(old['limits']) == set(KEEP) | set(DROP)
    assert {k: v for k, v in new.items() if k != 'limits'} == {k: v for k, v in old.items() if k != 'limits'}
    record = {'recorded_local': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
              'authorized_local': AUTHORIZED, 'revoked_local': REVOKED,
              'remote_unchanged': ['GPU-4391fcee-537f-d408-d138-10d8a3866eea', 'GPU-6122c7a9-eaa6-b539-6005-f20e339da4ea',
                                   'GPU-9e9f1e97-b2ca-04e0-eb2a-8035398daa79', 'GPU-ca65e1b7-c0ac-6757-7d3f-816c3119a4c1'],
              'campaign_jobs_on_revoked_devices_at_change': occupying, 'migrations': []}
    if not PUBLISH:
        print(json.dumps(record, indent=1))
        raise SystemExit(0)
    assert not MANIFEST.exists()
    (SETUP / 'queue-before-gpu-allowlist-20261019.yaml').write_bytes(before)
    tmp = jobs.QUEUE.with_suffix('.allow.tmp')
    tmp.write_text(text)
    os.replace(tmp, jobs.QUEUE)
    assert list(jobs.load_queue()['hosts']['local']['limits']) == KEEP
    record.update(queue_sha256_before=hashlib.sha256(before).hexdigest(),
                  queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
    MANIFEST.write_text(json.dumps(record, indent=2) + '\n')
    print('APPLIED', MANIFEST, 'occupying', occupying)
