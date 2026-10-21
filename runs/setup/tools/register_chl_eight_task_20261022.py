"""chamfer_lean: 200k pretraining, RL evaluation and eight-task expansion (user instruction 2026-10-22, after the
50k pilot results; definition runs/setup/chl-definition-20261021.json, unchanged). Registered before any result of
these runs. Mirrors the frozen method's eight-task protocol (pretraining s3-m3d-seed0-<task>-200k / i2-m3d-seed0, its
export, s3-drm-splatter4d-<task>-s200x + evaluator) and the future_k2 rows, with only the variant config and names:
- chl-seed0-<task>-200k: base + task + variants/chamfer_lean.yaml, train.steps=200000, train.seed=0. hammer and
  pick-place CONTINUE the 50k pilots (which ran the 200k learning-rate schedule): their step-50000 checkpoints are
  copied into the new run directories as latest.pt before admission (model, optimiser, scheduler, RNG and data
  position restored by --resume auto), so the 0-50k segment is not repeated and the pilot directories stay intact;
- chl-export-seed0-<task>-200k, chl-drm-<task>-s2000..2002 (DrM, 1M agent steps) and their evaluators.
Scheduling only (not protocol): pretraining priority 2, so the already registered future_k2 RL rows (priority 2,
registered earlier) are admitted first; exports keep priority 0.

    python -I register_chl_eight_task_20261022.py [--publish]
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
TASKS = ('hammer', 'pick-place', 'bin-picking', 'coffee-push', 'door-open', 'peg-insert-side', 'peg-unplug-side', 'stick-push')
CONTINUED = ('hammer', 'pick-place')
SEEDS = (2000, 2001, 2002)
meta = {'stage': 5, 'variant': 'chamfer_lean', 'variant_definition': 'runs/setup/chl-definition-20261021.json'}

queue = jobs.load_queue()
by_queue = {j['id']: j for j in queue['jobs']}
state = jobs.read_registry()
for t in CONTINUED:
    assert state[f'chl-seed0-{t}-50k']['status'] == jobs.DONE, t
new_jobs = []
for task in TASKS:
    assert (ROOT / f'configs/metaworld/tasks/{task}.yaml').is_file(), task
    pid, eid = f'chl-seed0-{task}-200k', f'chl-export-seed0-{task}-200k'
    row = {'id': pid, 'host': 'local', 'script': 'scripts/train.py',
           'args': ['--config', 'configs/metaworld/base.yaml', f'configs/metaworld/tasks/{task}.yaml',
                    'configs/metaworld/variants/chamfer_lean.yaml', '--name', pid, '--output-root', 'runs/pretrain',
                    '--resume', 'auto', '--set', 'train.steps=200000', 'train.seed=0'],
           'gpu': 'any', 'mem_gb': 10, 'ram_gb': 13, 'disk_gb': 30, 'priority': 2, 'deps': [], 'max_restarts': 3, **meta,
           'required_results': ['completion.json', 'metrics.jsonl', 'checkpoints/step_0200000.pt']}
    if task in CONTINUED:
        row['continues'] = f'runs/pretrain/chl-seed0-{task}-50k/checkpoints/step_0050000.pt'
    new_jobs.append(row)
    new_jobs.append({'id': eid, 'host': 'local', 'script': 'scripts/export_encoder.py',
                     'args': ['--run', f'runs/pretrain/{pid}', '--checkpoint', f'runs/pretrain/{pid}/checkpoints/step_0200000.pt',
                              '--out', f'runs/{eid}/encoder.pt'],
                     'gpu': 'any', 'mem_gb': 2, 'ram_gb': 4, 'disk_gb': 1, 'priority': 0, 'deps': [pid], 'max_restarts': 1,
                     **meta, 'required_results': ['encoder.pt']})
    for seed in SEEDS:
        rid = f'chl-drm-{task}-s{seed}'
        common = {'host': 'local', 'gpu': 'any', 'priority': 2, 'deps': [eid], 'max_restarts': 1, **meta}
        new_jobs.append({**common, 'id': rid, 'script': 'scripts/train_rl.py',
                         'args': ['--task', task, '--encoder', 'splatter4d', '--seed', str(seed), '--run-dir', f'runs/{rid}',
                                  '--set', 'train.num_train_steps=1000000', f'vision.export_path=runs/{eid}/encoder.pt'],
                         'mem_gb': 4, 'ram_gb': 6, 'disk_gb': 10, 'required_results': ['final.json', 'train.jsonl']})
        new_jobs.append({**common, 'id': f'{rid}-eval', 'script': 'scripts/eval_rl.py',
                         'args': ['--run-dir', f'runs/{rid}', '--train-job', rid, '--workers', '6', '--envs-per-worker', '8',
                                  '--output-dir', f'runs/{rid}-eval'],
                         'mem_gb': 3, 'ram_gb': 8, 'disk_gb': 1, 'required_results': ['eval.jsonl']})
# protocol equals the registered future_k2 rows (which mirror the frozen method) apart from the variant config,
# names and the scheduling priority of the pretraining rows
for j in new_jobs:
    ref_id = j['id'].replace('chl-', 'afk2-', 1)
    ref = by_queue[ref_id]
    a = [x.replace('chamfer_lean', 'future_k2').replace('chl-', 'afk2-') for x in j['args']]
    assert a == ref['args'], (j['id'], a, ref['args'])
    diff = {k for k in set(j) | set(ref) if j.get(k) != ref.get(k)} - {'id', 'args', 'variant', 'variant_definition', 'continues'}
    assert diff <= {'priority', 'deps'}, (j['id'], diff)
variant = yaml.safe_load((ROOT / 'configs/metaworld/variants/chamfer_lean.yaml').read_text())
assert variant['loss']['chamfer'] == 5.0 and variant['aug'] == {'synth_views': 2, 'synth_render': True}
assert not set(by_queue) & {j['id'] for j in new_jobs}, 'already registered'
counts = {}
for j in new_jobs:
    counts[j['script'].split('/')[-1]] = counts.get(j['script'].split('/')[-1], 0) + 1
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0], 'tasks': list(TASKS),
          'counts': counts, 'jobs': new_jobs}
proof = ROOT / 'runs/setup/chl-registration-eight-task-20261022.json'
if not args.publish:
    print(json.dumps({'counts': counts, 'ids': [j['id'] for j in new_jobs][:12]}, indent=1))
    raise SystemExit(0)
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not proof.exists()
    seeded = {}
    for task in CONTINUED:  # seed the continuation runs with the pilots' 50k checkpoints (copies; pilots untouched)
        src = ROOT / f'runs/pretrain/chl-seed0-{task}-50k/checkpoints/step_0050000.pt'
        dst_dir = ROOT / f'runs/pretrain/chl-seed0-{task}-200k/checkpoints'
        assert not (ROOT / f'runs/pretrain/chl-seed0-{task}-200k').exists()
        dst_dir.mkdir(parents=True)
        for name in ('step_0050000.pt', 'latest.pt'):
            shutil.copy2(src, dst_dir / name)
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        assert all(hashlib.sha256((dst_dir / n).read_bytes()).hexdigest() == digest for n in ('step_0050000.pt', 'latest.pt'))
        seeded[task] = {'source': str(src.relative_to(ROOT)), 'sha256': digest}
    before = jobs.QUEUE.read_bytes()
    text = before.decode()
    node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    staged = ROOT / 'runs/setup/chl-queue-candidate-eight-task-20261022.yaml'
    staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                      + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n' + text[node.end_mark.index:])
    assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
    tmp = jobs.QUEUE.with_suffix('.chl8.tmp')
    tmp.write_bytes(staged.read_bytes())
    os.replace(tmp, jobs.QUEUE)
    record.update(registered_at=datetime.now().astimezone().isoformat(), continuation_checkpoints=seeded,
                  queue_sha256_before=hashlib.sha256(before).hexdigest(),
                  queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
    proof.write_text(json.dumps(record, indent=2) + '\n')
    print('REGISTERED', len(new_jobs), proof)
