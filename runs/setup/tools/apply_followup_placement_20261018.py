"""Host-neutral placement overlay (2026-10-18) for unstarted RL pairs of followup-registration-20261017.

The local MIG instances are freeing up while the remote queue is long. The scientific registration stays immutable;
this manifest stores the original and effective definition of each moved job. Only host-specific execution fields
change (host; cpu_threads, disk_gb, required_results dropped; local ram_gb as the other local DrM rows). Arguments
(seed, steps, export, evaluator 3 x 8 and --output-dir) are identical. Moved: the E1 crop and E2 shift DrM pairs
(`s2rl-crop-*`, `s2rl-shift-*`) that have not launched. The CNN no-shift rows stay remote (48 GB RAM each).

    python -I apply_followup_placement_20261018.py [--publish]
"""
import copy
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
REGISTRATION = SETUP / 'followup-registration-20261017.json'
MANIFEST = SETUP / 'followup-placement-20261018.json'
REMOTE_ONLY = ('cpu_threads', 'disk_gb', 'required_results')

registration = json.loads(REGISTRATION.read_text())
by_queue = {j['id']: j for j in jobs.load_queue()['jobs']}
state = jobs.read_registry()
pairs = {}
for job in registration['jobs']:
    if job['script'] in ('scripts/train_rl.py', 'scripts/eval_rl.py') and job['id'].startswith(('s2rl-crop-', 's2rl-shift-')):
        pairs.setdefault(job['id'].removesuffix('-eval'), []).append(job)
moved = [j for trainer, both in pairs.items() for j in both
         if all(x['id'] not in state and not (ROOT / 'runs' / x['id']).exists() for x in both)]
for job in moved:
    assert by_queue[job['id']] == job, job['id']
effective = {}
for job in moved:
    new = {k: copy.deepcopy(v) for k, v in job.items() if k not in REMOTE_ONLY}
    new.update(host='local', ram_gb=6, placement='host-neutral overlay 2026-10-18 (followup-placement-20261018.json)')
    assert {k: v for k, v in new.items() if k not in ('host', 'ram_gb', 'placement')} == \
        {k: v for k, v in job.items() if k not in ('host', 'ram_gb', *REMOTE_ONLY)}
    effective[job['id']] = new
record = {'recorded_local': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'registration': str(REGISTRATION.relative_to(ROOT)), 'original_jobs': {j['id']: j for j in moved},
          'effective_jobs': effective, 'scientific_fields_unchanged': True}
if not PUBLISH:
    print(len(moved), sorted(effective)[:6])
    raise SystemExit(0)
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not MANIFEST.exists()
    state = jobs.read_registry()
    assert not any(i in state for i in effective)
    before = jobs.QUEUE.read_bytes()
    text = before.decode()
    current = yaml.safe_load(before)['jobs']
    node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    staged = SETUP / 'followup-placement-queue-candidate-20261018.yaml'
    staged.write_text(text[:node.start_mark.index] + yaml.safe_dump(
        json.loads(json.dumps([effective.get(j['id'], j) for j in current])), sort_keys=False) + text[node.end_mark.index:])
    assert jobs.load_queue(staged)['jobs'] == [effective.get(j['id'], j) for j in jobs.load_queue()['jobs']]
    (SETUP / 'followup-placement-queue-before-20261018.yaml').write_bytes(before)
    tmp = jobs.QUEUE.with_suffix('.fup.tmp')
    tmp.write_bytes(staged.read_bytes())
    os.replace(tmp, jobs.QUEUE)
    record.update(queue_sha256_before=hashlib.sha256(before).hexdigest(),
                  queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
    MANIFEST.write_text(json.dumps(record, indent=2) + '\n')
    print('PLACED', len(effective), MANIFEST)
