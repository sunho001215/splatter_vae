"""Rerun of the SNeRL drawer-open pretraining under a new name on the fixed release (user approval 2026-10-31).

ds-pre-snerl-drawer-open (release b9196a8) crashed at data loading before any training step (KeyError: 'drawer-open'
in the SNeRL class table; empty checkpoints directory). The fix (TASK_BODIES for the new tasks) is release dd86d78. A
retry of the same row would have to keep b9196a8 (scripts/jobs.py holds a retry whose code changed), so the row is
rerun under a new name, with its six dependent FlashSAC rows (never started) copied to read the new export:

  ds-pre-snerl-drawer-open-r2                      = ds-pre-snerl-drawer-open with --name changed
  ds-flashsac-snerl-drawer-open-r2-s200{0,1,2}     = ds-flashsac-snerl-drawer-open-s200X with the export path, run
  ds-flashsac-snerl-drawer-open-r2-s200X-eval        dirs, train-job and dependency changed

Every other field is the original row's. The seven original rows get terminal cancelled events (superseded); their
definitions in the queue are unchanged.

    python -I register_snerl_drawer_open_r2_20261031.py [--publish]
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
from scripts import jobs  # noqa: E402
import pool_placement as pool  # noqa: E402

OLD, NEW = 'ds-pre-snerl-drawer-open', 'ds-pre-snerl-drawer-open-r2'
FS_OLD, FS_NEW = 'ds-flashsac-snerl-drawer-open-', 'ds-flashsac-snerl-drawer-open-r2-'
REASON = ('cancelled: superseded by a rerun under a new name on release dd86d78 (SNeRL class table for the new tasks; '
          'user approval 2026-10-31); attempt 1 crashed at data loading before any training step')
STAMP = datetime.now().astimezone().isoformat(timespec='seconds')
PROOF = ROOT / 'runs/setup/snerl-drawer-open-r2-registration-20261031.json'
ap = argparse.ArgumentParser()
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
queue = jobs.load_queue()
in_queue = {j['id']: j for j in queue['jobs']}
state = jobs.read_registry()

pre = copy.deepcopy(in_queue[OLD])
row = copy.deepcopy(pre)
row['id'] = NEW
row['args'][row['args'].index('--name') + 1] = NEW
row['placement'] = f'pool {STAMP} (rerun of {OLD} under a new name on release dd86d78; remote)'
assert {k: v for k, v in row.items() if k not in ('id', 'args', 'placement')} == \
    {k: v for k, v in pre.items() if k not in ('id', 'args', 'placement')}
new_jobs, cancelled = [row], [OLD]
for seed in ('2000', '2001', '2002'):
    for suffix in ('', '-eval'):
        old_id = f'{FS_OLD}s{seed}{suffix}'
        ref = in_queue[old_id]
        assert ref['deps'] == [OLD], (old_id, ref['deps'])
        text = json.dumps(ref).replace(f'runs/{OLD}/encoder.pt', f'runs/{NEW}/encoder.pt').replace(FS_OLD, FS_NEW)
        r = json.loads(text)
        r['deps'] = [NEW]
        r['placement'] = f'pool {STAMP} (copy of {old_id} reading the {NEW} export; local frozen-encoder FlashSAC)'
        assert r['id'] == f'{FS_NEW}s{seed}{suffix}' and OLD + '/' not in json.dumps(r)
        assert {k: v for k, v in r.items() if k not in ('id', 'args', 'deps', 'placement')} == \
            {k: v for k, v in ref.items() if k not in ('id', 'args', 'deps', 'placement')}
        new_jobs.append(r)
        cancelled.append(old_id)
ids = [j['id'] for j in new_jobs]
assert len(ids) == 7 and not set(ids) & set(in_queue) and not any((ROOT / 'runs' / i).exists() for i in ids)
st = state.get(OLD, {})
assert st.get('status') == 'crashed' and st.get('attempts') == 1, st
for c in cancelled[1:]:
    assert state.get(c, {}).get('status', jobs.PENDING) == jobs.PENDING and not state.get(c, {}).get('attempts'), c
if not args.publish:
    for j in new_jobs[:3]:
        print(json.dumps(j))
    raise SystemExit(0)


def record(event):
    with jobs.REGISTRY.open('a') as stream:
        stream.write(json.dumps({**event, 'time': time.time()}) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


with open(pool.ADMISSION_LOCK, 'w') as admission:
    fcntl.flock(admission, fcntl.LOCK_EX)
    with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not PROOF.exists()
        before = jobs.QUEUE.read_bytes()
        text = before.decode()
        node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
        staged = PROOF.with_suffix('.queue-candidate.yaml')
        staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                          + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n'
                          + text[node.end_mark.index:])
        assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
        tmp = jobs.QUEUE.with_suffix('.snerlr2.tmp')
        tmp.write_bytes(staged.read_bytes())
        os.replace(tmp, jobs.QUEUE)
        for c in cancelled:
            record({'event': 'failed', 'id': c, 'host': in_queue[c].get('host', 'local'), 'cancelled': True,
                    'superseded': True, 'superseded_by': c.replace(OLD, NEW) if c == OLD else c.replace(FS_OLD, FS_NEW),
                    'reason': REASON})
        PROOF.write_text(json.dumps({
            'registered_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
            'release': 'dd86d78', 'cancelled': cancelled, 'jobs': new_jobs,
            'queue_sha256_before': hashlib.sha256(before).hexdigest(),
            'queue_sha256_after': hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest()}, indent=2) + '\n')
        print('REGISTERED', len(new_jobs), 'CANCELLED', len(cancelled), PROOF)
