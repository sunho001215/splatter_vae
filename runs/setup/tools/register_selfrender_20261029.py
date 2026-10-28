"""Self-render replacement pretraining on csw4 (user directive 2026-10-29 items 4-5, amendment 2026-10-29 ~00:40):
the combined warm-up + expanded-viewpoint self-render recipe (configs/metaworld/variants/selfrender_replace_v2.yaml:
no self-render contribution 0-20k, linear ramp 20k-40k, full 40k-200k; cameras from the frozen training-supported domain
configs/metaworld/selfrender_domain_v1.yaml) on hammer, pick-place and bin-picking, 200k steps, pretraining seed 0.

Rows sr-csw4-seed0-<task>-200k and sr-csw4-export-seed0-<task>-200k are the prepared rows
(runs/setup/selfrender-rows-prepared-20261029.yaml): each equals its enc-csw4 row apart from the
id, the appended variant file, the run/export paths and the variant name. Placement as the new-task csw4 rows
(ds-csw4-*): pretraining and export on the remote (export where the checkpoint is; remote-first), priority 2, released
on DS_COMMIT (sr- prefix). The FlashSAC rows ds-flashsac-sr-csw4-* depend on the exports.

    python -I register_selfrender_20261029.py [--publish]
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

PREPARED = ROOT / 'runs/setup/selfrender-rows-prepared-20261029.yaml'  # merged with release 8ad7734
TASKS = ('hammer', 'pick-place', 'bin-picking')
VARIANT_FILE = 'configs/metaworld/variants/selfrender_replace_v2.yaml'
DOMAIN = 'configs/metaworld/selfrender_domain_v1.yaml'
STAMP = datetime.now().astimezone().isoformat(timespec='seconds')
ap = argparse.ArgumentParser()
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
queue = jobs.load_queue()
in_queue = {j['id']: j for j in queue['jobs']}
prepared = {j['id']: j for j in yaml.safe_load(PREPARED.read_text())}
assert (ROOT / VARIANT_FILE).is_file() and (ROOT / DOMAIN).is_file(), 'variant not merged into the main checkout'
new_jobs = []
for task in TASKS:
    for kind, ref_id in (('pre', f'enc-csw4-seed0-{task}-200k'), ('export', f'enc-csw4-export-seed0-{task}-200k')):
        rid = f'sr-csw4-seed0-{task}-200k' if kind == 'pre' else f'sr-csw4-export-seed0-{task}-200k'
        row, ref = copy.deepcopy(prepared[rid]), in_queue[ref_id]
        if kind == 'pre':  # the enc-csw4 row plus the appended variant file, with this row's name
            expected = list(ref['args'])
            expected.insert(expected.index('--name'), VARIANT_FILE)
            expected[expected.index('--name') + 1] = rid
            assert row['args'] == expected, (row['args'], expected)
        row.update(host='remote', priority=2, stage=7, variant='selfrender_replace_v2', variant_definition=VARIANT_FILE,
                   placement=f'pool {STAMP} (registered on the remote: self-render replacement pretraining)')
        new_jobs.append(row)
ids = [j['id'] for j in new_jobs]
assert len(ids) == len(set(ids)) and not set(ids) & set(in_queue), 'duplicate or already registered'
assert not any((ROOT / 'runs' / i).exists() for i in ids), 'run dir exists'
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'variant_file': VARIANT_FILE, 'viewpoint_domain': DOMAIN,
          'label': 'combined warm-up + expanded-viewpoint self-render recipe', 'tasks': TASKS, 'jobs': new_jobs}
proof = ROOT / 'runs/setup/selfrender-registration-20261029.json'
if not args.publish:
    print(len(new_jobs), proof.name)
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
        tmp = jobs.QUEUE.with_suffix('.sr.tmp')
        tmp.write_bytes(staged.read_bytes())
        os.replace(tmp, jobs.QUEUE)
        record.update(registered_at=datetime.now().astimezone().isoformat(),
                      queue_sha256_before=hashlib.sha256(before).hexdigest(),
                      queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
        proof.write_text(json.dumps(record, indent=2) + '\n')
        print('REGISTERED', len(new_jobs), proof)
