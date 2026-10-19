"""New baselines, acceptance step 5 for SNeRL (user go-ahead 2026-10-20, after the SNeRL hammer acceptance report in
docs/BASELINES.md, sanity passed 0.589 vs 0.013): the eight-task SNeRL runs. Registered before any result of these
runs exists, with the same protocol as the DynaMo / 3D-MVP eight-task registration (baselines-eight-task-registration-
20261017.json).

Eight-task runs (the registered final-comparison protocol, docs/BASELINES.md "Expansion policy"):
- pretraining `s4-snerl-<task>-rem-b` for the seven non-hammer tasks: official recipe configs unchanged (2.4M steps,
  seed 0), remote release e4206f1, same resources as the hammer run. Hammer reuses the accepted `s4-snerl-hammer-rem-b`
  final encoder (the hammer encoder of the comparison, as registered for step 3);
- frozen-encoder DrM `s4-drm-snerl-<task>-s<seed>` + evaluator, eight tasks x seeds 2000-2002 x 1M agent steps, default
  RL protocol with the `snerl` encoder config (three training-camera RGB views + poses, privileged; multi-view columns
  as in s4d/rl/evaluate.py), evaluator 6 workers x 8 envs (as every other final-comparison row). Reported as the row
  "SNeRL, multi-view RGB + poses (privileged)".
- Placement (host-neutral, recorded per run): pretraining on the remote, under the remote driver's existing load rule;
  DrM on local MIG (allowlisted GPU 4/5 instances), reading the remote pretraining export synced back with checksums,
  as for the DynaMo and 3D-MVP condition B rows.

    python -I register_snerl_eight_task_20261020.py [--publish]
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
from scripts import jobs  # noqa: E402

PUBLISH = '--publish' in sys.argv
TAG = '20261020'
TASKS = ('bin-picking', 'coffee-push', 'door-open', 'hammer', 'peg-insert-side', 'peg-unplug-side', 'pick-place',
         'stick-push')
SEEDS = (2000, 2001, 2002)
meta = {'stage': 4, 'baseline_expansion': f'snerl-eight-task-{TAG}'}
assert sorted(p.stem for p in (ROOT / 'configs/baselines/snerl').glob('*.yaml')) == list(TASKS)

queue = jobs.load_queue()
by_queue = {j['id']: j for j in queue['jobs']}
hammer = by_queue['s4-snerl-hammer-rem-b']
new_jobs = []
for task in TASKS:
    if task == 'hammer':
        continue
    pid = f's4-snerl-{task}-rem-b'
    new_jobs.append({'id': pid, 'host': 'remote', 'script': hammer['script'],
                     'args': ['--config', f'configs/baselines/snerl/{task}.yaml', '--name', pid,
                              '--output-root', 'runs', '--resume', 'auto'],
                     'gpu': 'any', 'mem_gb': hammer['mem_gb'], 'ram_gb': hammer['ram_gb'], 'disk_gb': hammer['disk_gb'],
                     'priority': 1, 'deps': [], 'max_restarts': 2, 'baseline': 'snerl', **meta,
                     'required_results': hammer['required_results']})
for task in TASKS:
    pretrain = f's4-snerl-{task}-rem-b'
    export = f'runs/{pretrain}/encoder.pt'
    for seed in SEEDS:
        rid = f's4-drm-snerl-{task}-s{seed}'
        common = {'host': 'local', 'gpu': 'any', 'priority': 2, 'deps': [pretrain], 'max_restarts': 1,
                  'baseline': 'snerl', 'baseline_condition': 'snerl', **meta}
        # local declared footprints as the DynaMo / 3D-MVP B local rows (the local driver uses measured DrM footprints)
        new_jobs.append({**common, 'id': rid, 'script': 'scripts/train_rl.py',
                         'args': ['--task', task, '--encoder', 'snerl', '--seed', str(seed), '--run-dir', f'runs/{rid}',
                                  '--set', f'vision.export_path={export}'],
                         'mem_gb': 4, 'ram_gb': 6, 'disk_gb': 10, 'baseline_export': export})
        new_jobs.append({**common, 'id': f'{rid}-eval', 'script': 'scripts/eval_rl.py',
                         'args': ['--run-dir', f'runs/{rid}', '--train-job', rid, '--workers', '6',
                                  '--envs-per-worker', '8', '--output-dir', f'runs/{rid}-eval'],
                         'mem_gb': 3, 'ram_gb': 8, 'disk_gb': 1})

assert not set(by_queue) & {j['id'] for j in new_jobs}
state = jobs.read_registry()
assert state['s4-snerl-hammer-rem-b']['status'] == jobs.DONE
hammer_sha = hashlib.sha256((ROOT / 'runs/s4-snerl-hammer-rem-b/encoder.pt').read_bytes()).hexdigest()
assert state['s4-snerl-hammer-rem-b']['sync']['files']['encoder.pt'] == hammer_sha
counts = {}
for j in new_jobs:
    key = f"{j['host']}/{j['script'].split('/')[-1]}"
    counts[key] = counts.get(key, 0) + 1
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'protocol': __doc__.split('\n\n')[1], 'remote_release': 'e4206f1e3106cfd115517d4e33fb967199f2e74a',
          'reused_hammer_pretraining': 's4-snerl-hammer-rem-b', 'hammer_encoder_sha256': hammer_sha,
          'sanity': 'docs/BASELINES.md "Hammer acceptance report: SNeRL" (pretrained 0.589 vs random 0.013, passed)',
          'counts': counts, 'jobs': new_jobs}
proof = ROOT / f'runs/setup/baselines-snerl-eight-task-registration-{TAG}.json'
if not PUBLISH:
    print(json.dumps({'counts': counts, 'ids': [j['id'] for j in new_jobs][::6]}, indent=1))
    raise SystemExit(0)
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not proof.exists()
    before = jobs.QUEUE.read_bytes()
    text = before.decode()
    node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    staged = ROOT / f'runs/setup/baselines-snerl-eight-task-queue-candidate-{TAG}.yaml'
    staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                      + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n' + text[node.end_mark.index:])
    assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
    (ROOT / f'runs/setup/baselines-snerl-eight-task-queue-before-{TAG}.yaml').write_bytes(before)
    tmp = jobs.QUEUE.with_suffix('.snerl8.tmp')
    tmp.write_bytes(staged.read_bytes())
    os.replace(tmp, jobs.QUEUE)
    record.update(registered_at=datetime.now().astimezone().isoformat(),
                  queue_sha256_before=hashlib.sha256(before).hexdigest(),
                  queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
    proof.write_text(json.dumps(record, indent=2) + '\n')
    print('REGISTERED', len(new_jobs), proof)
