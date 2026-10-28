"""Baseline encoder pretraining for the new tasks (user directive 2026-10-29: register the missing new-task dependencies
explicitly, never substitute another task's encoder).

For each new task T in drawer-open, window-open, soccer, sweep-into and each baseline, one row ds-pre-<baseline>-T that
equals the baseline's door-open pretraining row (same script, recipe and resources) with only the task config
(configs/baselines/<baseline>/T.yaml, identical to door-open apart from data paths) and the name changed. These scripts
write encoder.pt themselves (no separate export row); 3D-MVP's export serves both of its RL conditions (A and B), as on
the original tasks. Remote rows (data uploaded with upload_task_data_20261028.py), priority 3, ds- prefix (downstream
release). Measured costs on the original tasks: 3D-MVP ~3 h, DynaMo ~5 h, ReViWo ~6 h, SinCro ~21 h, SNeRL ~103 h.

    python -I register_newtask_baselines_20261029.py --tasks T [...] [--publish]
"""
import argparse
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
sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
from scripts import jobs  # noqa: E402
import pool_placement as pool  # noqa: E402

REFERENCE = {'sincro': 's4-sincro-door-open-b', 'reviwo': 's4-reviwo-door-open-b', 'dynamo': 's4-dynamo-door-open-rem-b',
             'snerl': 's4-snerl-door-open-rem-b', 'mvp3d': 's4-mvp3d-door-open-rem-b'}
NEW_TASKS = ('drawer-open', 'window-open', 'soccer', 'sweep-into')
STAMP = datetime.now().astimezone().isoformat(timespec='seconds')
ap = argparse.ArgumentParser()
ap.add_argument('--tasks', nargs='+', default=list(NEW_TASKS))
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
assert set(args.tasks) <= set(NEW_TASKS)
queue = jobs.load_queue()
in_queue = {j['id']: j for j in queue['jobs']}
new_jobs = []
for task in args.tasks:
    for base, ref_id in REFERENCE.items():
        ref = in_queue[ref_id]
        rid = f'ds-pre-{base}-{task}'
        row = copy.deepcopy(ref)
        row['args'] = [a.replace('door-open', task) if a.startswith('configs/baselines/') else (rid if a == ref_id else a)
                       for a in ref['args']]
        assert row['args'].count(rid) == 1 and f'configs/baselines/{base}/{task}.yaml' in row['args'], row['args']
        row.update(id=rid, host='remote', priority=3, stage=6, deps=[],
                   variant=f'baseline_{base}', variant_definition='runs/setup/downstream-selection-rule-20261028-a1.json',
                   placement=f'pool {STAMP} (registered on the remote: new-task baseline pretraining)')
        row.pop('continues', None)
        changed = {k for k in set(row) | set(ref) if row.get(k) != ref.get(k)}
        assert changed <= {'id', 'args', 'priority', 'stage', 'variant', 'variant_definition', 'placement', 'host'}, changed
        new_jobs.append(row)
ids = [j['id'] for j in new_jobs]
assert len(ids) == len(set(ids)) and not set(ids) & set(in_queue), 'duplicate or already registered'
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0], 'tasks': args.tasks,
          'references': REFERENCE, 'counts': {s: sum(j['script'] == s for j in new_jobs) for s in sorted({j['script'] for j in new_jobs})},
          'jobs': new_jobs}
proof = ROOT / f"runs/setup/newtask-baselines-registration-{'-'.join(args.tasks)}-20261029.json"
if not args.publish:
    print(json.dumps(record['counts']), proof.name)
    print(json.dumps(new_jobs[0]))
    raise SystemExit(0)
with open(pool.ADMISSION_LOCK, 'w') as admission:
    fcntl.flock(admission, fcntl.LOCK_EX)
    with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not proof.exists()
        before = jobs.QUEUE.read_bytes()
        text = before.decode()
        node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
        staged = proof.with_suffix('.queue-candidate.yaml')
        staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                          + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n'
                          + text[node.end_mark.index:])
        assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
        tmp = jobs.QUEUE.with_suffix('.newbase.tmp')
        tmp.write_bytes(staged.read_bytes())
        os.replace(tmp, jobs.QUEUE)
        record.update(registered_at=datetime.now().astimezone().isoformat(),
                      queue_sha256_before=hashlib.sha256(before).hexdigest(),
                      queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
        proof.write_text(json.dumps(record, indent=2) + '\n')
        print('REGISTERED', len(new_jobs), proof)
