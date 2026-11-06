"""Variant future_k2 (definition and gate: runs/setup/afk2-definition-20261020.json; replaces the cancelled action_future), registered before any
result of these runs. Mirrors the frozen method's registered jobs (s3-m3d-seed0-<task>-200k, its export and
s3-drm-splatter4d-<task>-s200x + evaluator) with only the variant config added and new names/paths.

    python -I register_actfut_20261020.py --tasks hammer pick-place [--publish]
"""
import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument('--tasks', nargs='+', required=True)
ap.add_argument('--tag', default='20261020')
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
SEEDS = (2000, 2001, 2002)
meta = {'stage': 5, 'variant': 'future_k2', 'variant_definition': 'runs/setup/afk2-definition-20261020.json'}

queue = jobs.load_queue()
by_queue = {j['id']: j for j in queue['jobs']}
ref = by_queue['s3-m3d-seed0-door-open-200k']
new_jobs = []
for task in args.tasks:
    assert (ROOT / f'configs/metaworld/tasks/{task}.yaml').is_file(), task
    pid, eid = f'afk2-seed0-{task}-200k', f'afk2-export-seed0-{task}-200k'
    new_jobs.append({'id': pid, 'host': 'local', 'script': 'scripts/train.py',
                     'args': ['--config', 'configs/metaworld/base.yaml', f'configs/metaworld/tasks/{task}.yaml',
                              'configs/metaworld/variants/future_k2.yaml', '--name', pid, '--output-root',
                              'runs/pretrain', '--resume', 'auto', '--set', 'train.steps=200000', 'train.seed=0'],
                     'gpu': 'any', 'mem_gb': 10, 'ram_gb': 13, 'disk_gb': 30, 'priority': 1, 'deps': [],
                     'max_restarts': 3, **meta,
                     'required_results': ['completion.json', 'metrics.jsonl', 'checkpoints/step_0200000.pt']})
    new_jobs.append({'id': eid, 'host': 'local', 'script': 'scripts/export_encoder.py',
                     'args': ['--run', f'runs/pretrain/{pid}', '--checkpoint', f'runs/pretrain/{pid}/checkpoints/step_0200000.pt',
                              '--out', f'runs/{eid}/encoder.pt'],
                     'gpu': 'any', 'mem_gb': 2, 'ram_gb': 4, 'disk_gb': 1, 'priority': 0, 'deps': [pid], 'max_restarts': 1,
                     **meta, 'required_results': ['encoder.pt']})
    for seed in SEEDS:
        rid = f'afk2-drm-{task}-s{seed}'
        common = {'host': 'local', 'gpu': 'any', 'priority': 2, 'deps': [eid], 'max_restarts': 1, **meta}
        new_jobs.append({**common, 'id': rid, 'script': 'scripts/train_rl.py',
                         'args': ['--task', task, '--encoder', 'splatter4d', '--seed', str(seed), '--run-dir', f'runs/{rid}',
                                  '--set', 'train.num_train_steps=1000000', f'vision.export_path=runs/{eid}/encoder.pt'],
                         'mem_gb': 4, 'ram_gb': 6, 'disk_gb': 10, 'required_results': ['final.json', 'train.jsonl']})
        new_jobs.append({**common, 'id': f'{rid}-eval', 'script': 'scripts/eval_rl.py',
                         'args': ['--run-dir', f'runs/{rid}', '--train-job', rid, '--workers', '6', '--envs-per-worker', '8',
                                  '--output-dir', f'runs/{rid}-eval'],
                         'mem_gb': 3, 'ram_gb': 8, 'disk_gb': 1, 'required_results': ['eval.jsonl']})
# the protocol must equal the frozen method's registered rows apart from the variant config and names
assert ref['args'][-6:] == ['train.steps=200000', 'train.seed=0', 'aug.synth_views=2', 'aug.synth_render=true',
                            'loss.depth_valid_far=true', 'loss.motion_space=gaussian']
variant = yaml.safe_load((ROOT / 'configs/metaworld/variants/future_k2.yaml').read_text())
assert variant['data'] == {'objective': 'future', 'history_stride': 2, 'future_strides': [1, 2, 3, 4, 5, 6],
                           'val_future_strides': [2, 6]}
assert not (ROOT / 'configs/metaworld/variants/action_future.yaml').exists(), 'the superseded variant must stay retired'
assert variant['aug'] == {'synth_views': 2, 'synth_render': True}
assert variant['loss'] == {'depth_valid_far': True, 'motion_space': 'gaussian'}
assert not set(by_queue) & {j['id'] for j in new_jobs}, 'already registered'
counts = {}
for j in new_jobs:
    counts[j['script'].split('/')[-1]] = counts.get(j['script'].split('/')[-1], 0) + 1
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'tasks': args.tasks, 'counts': counts, 'jobs': new_jobs}
proof = ROOT / f'runs/setup/afk2-registration-{"-".join(args.tasks)}-{args.tag}.json'
if not args.publish:
    print(json.dumps({'counts': counts, 'ids': [j['id'] for j in new_jobs]}, indent=1))
    raise SystemExit(0)
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not proof.exists()
    before = jobs.QUEUE.read_bytes()
    text = before.decode()
    node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    staged = ROOT / f'runs/setup/afk2-queue-candidate-{args.tag}-{len(args.tasks)}.yaml'
    staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                      + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n' + text[node.end_mark.index:])
    assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
    tmp = jobs.QUEUE.with_suffix('.afk2.tmp')
    tmp.write_bytes(staged.read_bytes())
    os.replace(tmp, jobs.QUEUE)
    record.update(registered_at=datetime.now().astimezone().isoformat(),
                  queue_sha256_before=hashlib.sha256(before).hexdigest(),
                  queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
    proof.write_text(json.dumps(record, indent=2) + '\n')
    print('REGISTERED', len(new_jobs), proof)
