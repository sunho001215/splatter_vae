"""Local devices became 12 MIG 1g.24gb instances (user, 2026-10-16): replace the local GPU limits in the queue and
unpin the historical (non-active) jobs that named an old full-GPU UUID (`gpu: any`; job fingerprints do not include
the GPU). Atomic, under the registry lock, old queue kept beside the proof.

    python -I queue_mig_limits.py [--publish]
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
from s4d.gpu_guard import APPROVED_HOST_GPUS, LOCAL_MIG_PARENTS  # noqa: E402
from scripts import jobs  # noqa: E402

OLD = {'GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce', 'GPU-d09f0338-71b9-d915-3c7f-e99754a3b639'}
MAX_JOBS, MAX_MEM_GB = 3, 22  # 24 GB per instance (24192 MiB), declared job GPU memory; jobs per instance by measure
PUBLISH = '--publish' in sys.argv

text = jobs.QUEUE.read_text()
queue = yaml.safe_load(text)
lines = text.splitlines(keepends=True)
assert lines[0] == 'limits:\n' and set(queue['limits']) == OLD
end = next(i for i, line in enumerate(lines[1:], 1) if not line.startswith(' '))
block = ['limits:\n'] + [f'  {uuid}:\n    max_jobs: {MAX_JOBS}\n    max_mem_gb: {MAX_MEM_GB}\n'
                         for uuid in APPROVED_HOST_GPUS['local']]
pinned = [i for i, line in enumerate(lines) if line.startswith('  gpu: ') and line.split()[1] in OLD]
state = jobs.read_registry()
ids = {}
for i in pinned:
    job_line = next(j for j in range(i, -1, -1) if lines[j].startswith('- id: ') or lines[j].startswith('  id: '))
    ids[i] = lines[job_line].split()[-1]
statuses = {job_id: state.get(job_id, {}).get('status', jobs.PENDING) for job_id in ids.values()}
assert not any(status in jobs.ACTIVE for status in statuses.values()), statuses
new_lines = list(lines)
for i in pinned:
    new_lines[i] = '  gpu: any\n'
new_text = ''.join(block + new_lines[end:])
new = yaml.safe_load(new_text)
assert set(new['limits']) == set(APPROVED_HOST_GPUS['local']) and not set(new['limits']) & set(LOCAL_MIG_PARENTS)
old_jobs, new_jobs = queue['jobs'], new['jobs']
assert [j['id'] for j in old_jobs] == [j['id'] for j in new_jobs]
for a, b in zip(old_jobs, new_jobs):
    assert a == b or ({k: v for k, v in a.items() if k != 'gpu'} == {k: v for k, v in b.items() if k != 'gpu'}
                      and a['gpu'] in OLD and b['gpu'] == 'any' and jobs.job_fingerprint(a) == jobs.job_fingerprint(b))
assert {k: v for k, v in new.items() if k not in ('jobs', 'limits')} == \
    {k: v for k, v in queue.items() if k not in ('jobs', 'limits')}
print(json.dumps({'unpinned': len(ids), 'not_done': {k: v for k, v in statuses.items() if v != jobs.DONE},
                  'limits': len(new['limits'])}))
if PUBLISH:
    with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert jobs.QUEUE.read_text() == text
        tag = datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')
        before = ROOT / f'runs/setup/queue-before-mig-{tag}.yaml'
        before.write_text(text)
        staged = jobs.QUEUE.with_suffix('.mig.tmp')
        staged.write_text(new_text)
        jobs.load_queue(staged)  # validates against the MIG allowlist
        os.replace(staged, jobs.QUEUE)
        proof = {'published': datetime.now().astimezone().isoformat(), 'reason': 'local MIG reconfiguration (user)',
                 'queue_sha256_before': hashlib.sha256(text.encode()).hexdigest(),
                 'queue_sha256_after': hashlib.sha256(new_text.encode()).hexdigest(), 'before_copy': str(before),
                 'max_jobs': MAX_JOBS, 'max_mem_gb': MAX_MEM_GB, 'unpinned_jobs_status': statuses}
        (ROOT / f'runs/setup/queue-mig-limits-{tag}.json').write_text(json.dumps(proof, indent=2) + '\n')
        print('PUBLISHED', proof['queue_sha256_after'])
