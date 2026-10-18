"""Host-neutral placement overlay (2026-10-18) for unstarted remote jobs of baselines-eight-task-registration-20261017.

The local MIG instances are idling while the remote queue is long. The scientific registration stays immutable; this
manifest stores the original and the effective definition of each moved job. Only the host and host-specific
execution fields change (host, cpu_threads, disk_gb, required_results; local footprint ram_gb as the other local DrM
rows): arguments, seeds, steps, exports and evaluation settings are identical. Moved: the 3D-MVP condition A eight-task
DrM pairs that never launched, and the DynaMo epoch-9 diagnostic pairs. Host is recorded per run as always.

    python -I apply_baseline_placement_20261018.py [--publish]
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
REGISTRATION = SETUP / 'baselines-eight-task-registration-20261017.json'
MANIFEST = SETUP / 'baselines-placement-20261018.json'
REMOTE_ONLY = ('cpu_threads', 'disk_gb', 'required_results')

registration = json.loads(REGISTRATION.read_text())
queue = jobs.load_queue()
by_queue = {j['id']: j for j in queue['jobs']}
state = jobs.read_registry()
moved = [j for j in registration['jobs']
         if j.get('host') == 'remote' and (j['id'].startswith('s4-drm-mvp3d-multiview-') or 'ep09' in j['id'])
         and j['id'] not in state and not (ROOT / 'runs' / j['id']).exists()]
for job in moved:  # both halves of every pair move together
    partner = job['id'].removesuffix('-eval') if job['id'].endswith('-eval') else job['id'] + '-eval'
    assert partner in {m['id'] for m in moved}, partner
    assert by_queue[job['id']] == job, job['id']
effective = {}
for job in moved:
    new = {k: copy.deepcopy(v) for k, v in job.items() if k not in REMOTE_ONLY}
    new['host'] = 'local'
    new['ram_gb'] = 8 if job['script'] == 'scripts/eval_rl.py' else 6
    new['placement'] = 'host-neutral overlay 2026-10-18 (baselines-placement-20261018.json)'
    assert {k: v for k, v in new.items() if k not in ('host', 'ram_gb', 'placement')} == \
        {k: v for k, v in job.items() if k not in ('host', 'ram_gb', *REMOTE_ONLY)}
    effective[job['id']] = new
record = {'recorded_local': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'registration': str(REGISTRATION.relative_to(ROOT)),
          'original_jobs': {j['id']: j for j in moved}, 'effective_jobs': effective,
          'scientific_fields_unchanged': True}
if not PUBLISH:
    print(len(moved), sorted(effective)[:4])
    raise SystemExit(0)
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not MANIFEST.exists()
    state = jobs.read_registry()
    assert not any(i in state for i in effective)
    before = jobs.QUEUE.read_bytes()
    current = yaml.safe_load(before)
    current['jobs'] = [effective.get(j['id'], j) for j in current['jobs']]
    text = before.decode()
    node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    staged = SETUP / 'baselines-placement-queue-candidate-20261018.yaml'
    staged.write_text(text[:node.start_mark.index] + yaml.safe_dump(json.loads(json.dumps(current['jobs'])),
                                                                     sort_keys=False) + text[node.end_mark.index:])
    assert jobs.load_queue(staged)['jobs'] == [effective.get(j['id'], j) for j in jobs.load_queue()['jobs']]
    (SETUP / 'baselines-placement-queue-before-20261018.yaml').write_bytes(before)
    tmp = jobs.QUEUE.with_suffix('.bxp.tmp')
    tmp.write_bytes(staged.read_bytes())
    os.replace(tmp, jobs.QUEUE)
    record.update(queue_sha256_before=hashlib.sha256(before).hexdigest(),
                  queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
    MANIFEST.write_text(json.dumps(record, indent=2) + '\n')
    print('PLACED', len(effective), MANIFEST)
