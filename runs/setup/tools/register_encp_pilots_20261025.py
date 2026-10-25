"""Encoder pilots on top of chamfer_lean_randmask (chlrm): hard negatives (hn), patch-local conv stem (cs), head dim 64 at
fixed size (h2), SinCro-shaped ViT body (w4) and width 256 at depth 8 (w8), on hammer, pick-place and bin-picking (user
directive 2026-10-25; registered before any result of these runs). Definition: runs/setup/encp-definition-20261025.json
(written here). Source: branch encoder-pilots-20261025, merged after the full suite and review.

Rows per arm and task (pretraining seed 0, 200k steps; the established protocol): enc-<arm>-seed0-<task>-200k
(base + task + chamfer_lean.yaml + chamfer_lean_randmask.yaml + configs/metaworld/variants/enc_<arm>.yaml), its export
enc-<arm>-export-seed0-<task>-200k and DrM seeds 2000-2002 (1M agent steps) with evaluators enc-<arm>-drm-<task>-s200x
[-eval]. Every row equals its chlrm reference row as registered (hammer, pick-place: mpf-registration-20261023.json;
bin-picking: chlrm-registration-eight-task-20261024.json) apart from names, the variant file, the priority and the
bookkeeping fields. 5 arms x 3 tasks x 8 = 120 rows.

Scheduling (not protocol): priority 4 for every row, strictly below every row registered before (maximum 3); the
drivers additionally admit enc rows only when no earlier-registered row waits for capacity (local: baseline driver;
remote: the pool pass places enc units last and none after a skipped earlier unit). Hosts: registered local; the pool
may move unstarted rows to the remote, where they run on the encoder-pilot release only (remote driver ENC_COMMIT).

    python -I register_encp_pilots_20261025.py [--publish]
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

ap = argparse.ArgumentParser()
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
TASKS = ('hammer', 'pick-place', 'bin-picking')
SEEDS = (2000, 2001, 2002)
ARMS = ('hn', 'cs', 'h2', 'w4', 'w8')
PRIORITY = 4
RANDMASK = 'configs/metaworld/variants/chamfer_lean_randmask.yaml'
DEFINITION = 'runs/setup/encp-definition-20261025.json'
BOOKKEEPING = {'id', 'args', 'variant', 'variant_definition', 'continues', 'deps', 'priority'}

queue = jobs.load_queue()
by_queue = {j['id']: j for j in queue['jobs']}
# every row registered before (the encoder-pilot smoke rows encsmoke-* are this campaign's own release checks)
assert max(j.get('priority', 5) for j in queue['jobs'] if not j['id'].startswith('encsmoke-')) < PRIORITY, \
    'a registered row already has priority >= 4'
REFS = {j['id']: j for name in ('mpf-registration-20261023.json', 'chlrm-registration-eight-task-20261024.json')
        for j in json.loads((ROOT / 'runs/setup' / name).read_text())['jobs'] if j['id'].startswith('chlrm-')}
overlay = pool.effective_rows()
for i, r in REFS.items():  # the queue equals the registrations up to the recorded pool placement (host fields, paths)
    assert by_queue[i] == overlay.get(i, r), i
for arm in ARMS:
    assert (ROOT / f'configs/metaworld/variants/enc_{arm}.yaml').is_file(), arm


def strip(row):
    return {k: v for k, v in row.items() if k not in ('variant', 'variant_definition', 'continues')}


new_jobs = []
for arm in ARMS:
    vfile = f'configs/metaworld/variants/enc_{arm}.yaml'
    meta = {'stage': 5, 'variant': f'enc_{arm}', 'variant_definition': DEFINITION}
    for task in TASKS:
        pid, eid = f'enc-{arm}-seed0-{task}-200k', f'enc-{arm}-export-seed0-{task}-200k'
        rp, re_ = f'chlrm-seed0-{task}-200k', f'chlrm-export-seed0-{task}-200k'
        pre = copy.deepcopy(strip(REFS[rp]))
        a = pre['args']
        a.insert(a.index(RANDMASK) + 1, vfile)  # on top of chamfer_lean + chamfer_lean_randmask
        a[a.index('--name') + 1] = pid
        pre.update(id=pid, priority=PRIORITY, **meta)
        new_jobs.append(pre)
        exp = copy.deepcopy(strip(REFS[re_]))
        exp['args'] = [x.replace(rp, pid).replace(re_, eid) for x in exp['args']]
        exp.update(id=eid, deps=[pid], priority=PRIORITY, **meta)
        new_jobs.append(exp)
        for seed in SEEDS:
            for suffix in ('', '-eval'):
                ref = f'chlrm-drm-{task}-s{seed}'
                rid = f'enc-{arm}-drm-{task}-s{seed}'
                row = copy.deepcopy(strip(REFS[ref + suffix]))
                row['args'] = [x.replace(ref, rid).replace(re_, eid) for x in row['args']]
                row.update(id=rid + suffix, deps=[eid], priority=PRIORITY, **meta)
                new_jobs.append(row)

for j in new_jobs:  # each row equals its chlrm reference apart from names, the variant file, priority and bookkeeping
    arm = j['id'].split('-')[1]
    ref_id = j['id'].replace(f'enc-{arm}-', 'chlrm-', 1)
    ref = REFS[ref_id]
    back = [x.replace(f'enc-{arm}-', 'chlrm-') for x in j['args'] if x != f'configs/metaworld/variants/enc_{arm}.yaml']
    assert back == ref['args'], (j['id'], back, ref['args'])
    assert not {k for k in set(j) | set(ref) if j.get(k) != ref.get(k)} - BOOKKEEPING, j['id']
    assert j['priority'] == PRIORITY and j['host'] == 'local'
assert len(new_jobs) == 120 and len({j['id'] for j in new_jobs}) == 120
assert not set(by_queue) & {j['id'] for j in new_jobs}, 'already registered'

definition = json.loads((ROOT / 'runs/setup/encp-definition-20261025.draft.json').read_text())
definition['recorded'] = datetime.now().astimezone().isoformat()
counts = {}
for j in new_jobs:
    counts.setdefault(j['variant'], {}).setdefault(j['script'].split('/')[-1], 0)
    counts[j['variant']][j['script'].split('/')[-1]] += 1
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0], 'tasks': list(TASKS),
          'arms': list(ARMS), 'definition': DEFINITION, 'priority': PRIORITY,
          'variant_files_sha256': {a: hashlib.sha256((ROOT / f'configs/metaworld/variants/enc_{a}.yaml').read_bytes()).hexdigest()
                                   for a in ARMS},
          'counts': counts, 'jobs': new_jobs}
proof = ROOT / 'runs/setup/encp-registration-20261025.json'
if not args.publish:
    print(json.dumps({'counts': counts, 'n': len(new_jobs)}, indent=1))
    for j in new_jobs[:8]:
        print(json.dumps(j))
    raise SystemExit(0)
with open(pool.ADMISSION_LOCK, 'w') as admission:
    fcntl.flock(admission, fcntl.LOCK_EX)
    with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not proof.exists() and not (ROOT / DEFINITION).exists()
        (ROOT / DEFINITION).write_text(json.dumps(definition, indent=1) + '\n')
        record['definition_sha256'] = hashlib.sha256((ROOT / DEFINITION).read_bytes()).hexdigest()
        before = jobs.QUEUE.read_bytes()
        text = before.decode()
        node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
        staged = ROOT / 'runs/setup/encp-queue-candidate-20261025.yaml'
        staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                          + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n' + text[node.end_mark.index:])
        assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
        assert jobs.load_queue(staged)['hosts'] == queue['hosts']
        tmp = jobs.QUEUE.with_suffix('.encp.tmp')
        tmp.write_bytes(staged.read_bytes())
        os.replace(tmp, jobs.QUEUE)
        record.update(registered_at=datetime.now().astimezone().isoformat(),
                      queue_sha256_before=hashlib.sha256(before).hexdigest(),
                      queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
        proof.write_text(json.dumps(record, indent=2) + '\n')
        print('REGISTERED', len(new_jobs), proof)
