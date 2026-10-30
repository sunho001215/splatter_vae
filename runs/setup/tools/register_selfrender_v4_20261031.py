"""Self-render variant selfrender_replace_v4 (user instruction 2026-10-31): selfrender_replace_v2 + the +-5 % radial
camera jitter (r = 1.0 x (1 + eps), eps ~ U(-0.05, 0.05) per synthetic camera) + the viewpoint domain with the coverage
and ws_support2 minimums relaxed by 0.95 (configs/metaworld/selfrender_domain_v1_relaxed5.yaml), trained from scratch
under new names; the v2 runs were stopped (stop_selfrender_v2_20261031.py). Everything else unchanged.

Rows (24), each a copy of its v2 row in the current queue (sr-csw4-*, ds-flashsac-sr-csw4-*) that differs ONLY in:
- pretraining srv4-csw4-seed0-<task>-200k: configs/metaworld/variants/selfrender_replace_v4.yaml appended after the v2
  variant file, the run name, variant and variant_definition, placement text;
- export srv4-csw4-export-seed0-<task>-200k: run/checkpoint/export paths, the dependency, variant fields, placement;
- FlashSAC trainer + evaluator ds-flashsac-srv4-csw4-<task>-s200{0,1,2}[-eval]: the export path, run/output dirs,
  train-job, dependencies and the variant name (flashsac_srv4-csw4), placement.
Host, priority (pretraining/export 0, FlashSAC 1: the self-render pilot priority of 2026-10-30), resources, seeds,
steps (--resume auto only resumes a run of the same new name; none exists) and every other field are the v2 rows'.
Remote pool launches of srv4-* rows run on SRV4_COMMIT (remote_campaign_driver.py); local ones on the main checkout,
which must contain the v4 variant (asserted).

    python -I register_selfrender_v4_20261031.py [--publish]
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

TASKS = ('hammer', 'pick-place', 'bin-picking')
SEEDS = ('2000', '2001', '2002')
V2_FILE = 'configs/metaworld/variants/selfrender_replace_v2.yaml'
V4_FILE = 'configs/metaworld/variants/selfrender_replace_v4.yaml'
DOMAIN = 'configs/metaworld/selfrender_domain_v1_relaxed5.yaml'
STAMP = datetime.now().astimezone().isoformat(timespec='seconds')
PLACEMENT = (f'pool {STAMP} (registered local like the v2 rows after their 2026-10-30 move; the remote driver pulls '
             'pool rows if remote capacity frees first)')
ap = argparse.ArgumentParser()
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
merged = ((ROOT / V4_FILE).is_file() and (ROOT / DOMAIN).is_file()
          and 'synth_radius_jitter' in (ROOT / 's4d/train/loop.py').read_text()
          and yaml.safe_load((ROOT / V4_FILE).read_text()) == {'aug': {'synth_radius_jitter': 0.05,
                                                                      'synth_camera_domain': DOMAIN}}
          and yaml.safe_load((ROOT / DOMAIN).read_text())['version'] == 'selfrender_domain_v1_relaxed5')
assert merged or not args.publish, 'the main checkout lacks the v4 variant (variant file, domain or jitter code)'
queue = jobs.load_queue()
in_queue = {j['id']: j for j in queue['jobs']}


def rename(value):
    return json.loads(json.dumps(value).replace('sr-csw4-', 'srv4-csw4-'))


new_jobs = []
for task in TASKS:
    pre_id, exp_id = f'sr-csw4-seed0-{task}-200k', f'sr-csw4-export-seed0-{task}-200k'
    pre = copy.deepcopy(in_queue[pre_id])
    row = rename(pre)
    a = row['args']
    a.insert(a.index(V2_FILE) + 1, V4_FILE)
    assert a[a.index('--name') + 1] == f'srv4-csw4-seed0-{task}-200k'
    row.update(variant='selfrender_replace_v4', variant_definition=V4_FILE, placement=PLACEMENT)
    expected = list(pre['args'])
    expected.insert(expected.index(V2_FILE) + 1, V4_FILE)
    expected[expected.index('--name') + 1] = row['id']
    assert row['args'] == expected and row['id'] == f'srv4-csw4-seed0-{task}-200k'
    assert {k: v for k, v in row.items() if k not in ('id', 'args', 'variant', 'variant_definition', 'placement')} == \
        {k: v for k, v in pre.items() if k not in ('id', 'args', 'variant', 'variant_definition', 'placement')}
    new_jobs.append(row)
    ex = copy.deepcopy(in_queue[exp_id])
    row = rename(ex)
    row.update(variant='selfrender_replace_v4', variant_definition=V4_FILE, placement=PLACEMENT)
    assert row['deps'] == [f'srv4-csw4-seed0-{task}-200k'] and row['id'] == f'srv4-csw4-export-seed0-{task}-200k'
    assert {k: v for k, v in row.items() if k not in ('id', 'args', 'deps', 'variant', 'variant_definition',
                                                       'placement')} == \
        {k: v for k, v in ex.items() if k not in ('id', 'args', 'deps', 'variant', 'variant_definition', 'placement')}
    new_jobs.append(row)
for task in TASKS:
    for seed in SEEDS:
        for suffix in ('', '-eval'):
            ref = copy.deepcopy(in_queue[f'ds-flashsac-sr-csw4-{task}-s{seed}{suffix}'])
            row = rename(ref)
            assert row['id'] == f'ds-flashsac-srv4-csw4-{task}-s{seed}{suffix}'
            assert row['deps'] == [f'srv4-csw4-export-seed0-{task}-200k'], row['deps']
            row.update(variant='flashsac_srv4-csw4', placement=f'pool {STAMP} (registered local as the v2 rows: '
                       'frozen-encoder FlashSAC, local MIG; not a pool-movable script)')
            assert json.dumps(row).count('sr-csw4') == 0
            assert {k: v for k, v in row.items() if k not in ('id', 'args', 'deps', 'variant', 'placement')} == \
                {k: v for k, v in ref.items() if k not in ('id', 'args', 'deps', 'variant', 'placement')}
            new_jobs.append(row)
ids = [j['id'] for j in new_jobs]
assert len(new_jobs) == 24 and len(ids) == len(set(ids)) and not set(ids) & set(in_queue), 'duplicate or registered'
assert not any((ROOT / 'runs' / i).exists() for i in ids), 'run dir exists'
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'variant_file': V4_FILE, 'base_variant_file': V2_FILE, 'viewpoint_domain': DOMAIN,
          'label': 'combined warm-up + expanded-viewpoint self-render recipe, +-5 % radial camera jitter, 5 %-relaxed domain',
          'tasks': TASKS, 'jobs': new_jobs}
proof = ROOT / 'runs/setup/selfrender-v4-registration-20261031.json'
if not args.publish:
    print(len(new_jobs), proof.name, 'main checkout has v4:', merged)
    for j in new_jobs[:2] + new_jobs[6:8]:
        print(json.dumps(j))
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
        tmp = jobs.QUEUE.with_suffix('.srv4.tmp')
        tmp.write_bytes(staged.read_bytes())
        os.replace(tmp, jobs.QUEUE)
        record.update(registered_at=datetime.now().astimezone().isoformat(),
                      queue_sha256_before=hashlib.sha256(before).hexdigest(),
                      queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
        proof.write_text(json.dumps(record, indent=2) + '\n')
        print('REGISTERED', len(new_jobs), proof)
