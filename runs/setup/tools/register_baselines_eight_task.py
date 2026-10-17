"""New baselines, acceptance step 5 onward (user go-ahead 2026-10-17 08:4x, after the hammer acceptance report in
docs/BASELINES.md): the eight-task runs of DynaMo and 3D-MVP, plus the DynaMo epoch-9 diagnostic. Registered before
any result of these runs exists. SNeRL follows once its hammer acceptance completes.

Eight-task runs (the registered final-comparison protocol, docs/BASELINES.md "Expansion policy"):
- pretraining `s4-{dynamo,mvp3d}-<task>-rem-b` for the seven non-hammer tasks: recipe configs unchanged, seed 0,
  remote release e4206f1, same resources as the hammer runs. Hammer reuses the accepted `s4-{dynamo,mvp3d}-hammer-rem-b`
  final encoders (the hammer encoders of the comparison, as registered for step 3);
- frozen-encoder DrM `s4-drm-<method>-<task>-s<seed>` + evaluator, eight tasks x seeds 2000-2002 x 1M agent steps,
  default RL protocol, evaluator 6 workers x 8 envs (as every other final-comparison row). DynaMo: `dynamo`;
  3D-MVP: condition A `mvp3d-multiview` and condition B `mvp3d-single`, reported as separate rows.
- Placement (host-neutral, recorded per run): pretraining and 3D-MVP condition A RL on the remote; DynaMo and 3D-MVP
  condition B RL on local MIG (local RL reads the remote pretraining export synced back with checksums).

DynaMo epoch-9 diagnostic (user request 2026-10-17): `bxs-dynamo-hammer-ep09-s100{0,1,2}-rem-b`, the hammer sanity
protocol (DrM, 400k agent steps, development seeds 1000-1002, evaluator 3 x 8) with the epoch-9 export
`runs/s4-dynamo-hammer-rem-b/exports/encoder_epoch09.pt`. Purpose: test the over-training explanation of the failed
sanity check. Rule fixed now: report its mean last-5 training-camera success next to the final-encoder (0.349) and
random-init (0.516) arms, descriptively; it never replaces the final encoder and no checkpoint choice follows from it.

    python -I register_baselines_eight_task.py [--publish]
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
TAG = '20261017'
TASKS = ('bin-picking', 'coffee-push', 'door-open', 'hammer', 'peg-insert-side', 'peg-unplug-side', 'pick-place',
         'stick-push')
SEEDS = (2000, 2001, 2002)
DIAG_SEEDS = (1000, 1001, 1002)
PRETRAIN = {'dynamo': ('scripts/train_dynamo.py', 8, 48), 'mvp3d': ('scripts/train_mvp3d.py', 12, 32)}
# RL rows: (job-id method, RL encoder config, pretraining method, host)
ROWS = (('dynamo', 'dynamo', 'dynamo', 'local'), ('mvp3d-multiview', 'mvp3d_multiview', 'mvp3d', 'remote'),
        ('mvp3d-single', 'mvp3d_single', 'mvp3d', 'local'))
EPOCH9 = 'runs/s4-dynamo-hammer-rem-b/exports/encoder_epoch09.pt'
meta = {'stage': 4, 'baseline_expansion': f'eight-task-{TAG}'}
for method in PRETRAIN:
    assert sorted(p.stem for p in (ROOT / f'configs/baselines/{method}').glob('*.yaml')) == list(TASKS), method

new_jobs = []
for method, (script, mem, ram) in PRETRAIN.items():
    for task in TASKS:
        if task == 'hammer':
            continue
        pid = f's4-{method}-{task}-rem-b'
        new_jobs.append({'id': pid, 'host': 'remote', 'script': script,
                         'args': ['--config', f'configs/baselines/{method}/{task}.yaml', '--name', pid,
                                  '--output-root', 'runs', '--resume', 'auto'],
                         'gpu': 'any', 'mem_gb': mem, 'ram_gb': ram, 'disk_gb': 20, 'priority': 1, 'deps': [],
                         'max_restarts': 2, 'baseline': method, **meta,
                         'required_results': ['encoder.pt', 'encoder_init.pt', 'metrics.jsonl', 'completion.json']})
for name, encoder, method, host in ROWS:
    for task in TASKS:
        pretrain = f's4-{method}-{task}-rem-b'
        export = f'runs/{pretrain}/encoder.pt'
        for seed in SEEDS:
            rid = f's4-drm-{name}-{task}-s{seed}'
            common = {'host': host, 'gpu': 'any', 'priority': 2, 'deps': [pretrain], 'max_restarts': 1,
                      'baseline': method, 'baseline_condition': encoder, **meta}
            trainer = {**common, 'id': rid, 'script': 'scripts/train_rl.py',
                       'args': ['--task', task, '--encoder', encoder, '--seed', str(seed), '--run-dir', f'runs/{rid}',
                                '--set', f'vision.export_path={export}'],
                       'mem_gb': 4, 'disk_gb': 10, 'baseline_export': export}
            evaluator = {**common, 'id': f'{rid}-eval', 'script': 'scripts/eval_rl.py',
                         'args': ['--run-dir', f'runs/{rid}', '--train-job', rid, '--workers', '6',
                                  '--envs-per-worker', '8', '--output-dir', f'runs/{rid}-eval'],
                         'mem_gb': 3, 'disk_gb': 1}
            if host == 'remote':
                trainer.update(ram_gb=16, required_results=['final.json', 'train.jsonl'])
                evaluator.update(ram_gb=8, required_results=['eval.jsonl'])
            else:  # local declared footprints: measured DrM trainer 3.5 GB / evaluator 4.5 GB (Stage 2 driver table)
                trainer.update(ram_gb=6)
                evaluator.update(ram_gb=8)
            new_jobs += [trainer, evaluator]
for seed in DIAG_SEEDS:
    rid = f'bxs-dynamo-hammer-ep09-s{seed}-rem-b'
    common = {'host': 'remote', 'gpu': 'any', 'priority': 2, 'deps': ['s4-dynamo-hammer-rem-b'], 'max_restarts': 1,
              'baseline': 'dynamo', 'baseline_arm': 'ep09-diagnostic', 'stage': 4,
              'baseline_expansion': f'dynamo-ep09-diagnostic-{TAG}'}
    new_jobs.append({**common, 'id': rid, 'script': 'scripts/train_rl.py',
                     'args': ['--task', 'hammer', '--encoder', 'dynamo', '--seed', str(seed), '--run-dir', f'runs/{rid}',
                              '--set', 'train.num_train_steps=400000', f'vision.export_path={EPOCH9}'],
                     'mem_gb': 4, 'ram_gb': 16, 'disk_gb': 5, 'required_results': ['final.json', 'train.jsonl'],
                     'baseline_export': EPOCH9})
    new_jobs.append({**common, 'id': f'{rid}-eval', 'script': 'scripts/eval_rl.py',
                     'args': ['--run-dir', f'runs/{rid}', '--train-job', rid, '--workers', '3', '--envs-per-worker', '8',
                              '--output-dir', f'runs/{rid}-eval'],
                     'mem_gb': 4, 'ram_gb': 12, 'disk_gb': 1, 'required_results': ['eval.jsonl']})

queue = jobs.load_queue()
assert not {j['id'] for j in queue['jobs']} & {j['id'] for j in new_jobs}
state = jobs.read_registry()
for method in PRETRAIN:
    assert state[f's4-{method}-hammer-rem-b']['status'] == jobs.DONE
epoch9_sha = hashlib.sha256((ROOT / EPOCH9).read_bytes()).hexdigest()
assert state['s4-dynamo-hammer-rem-b']['sync']['files']['exports/encoder_epoch09.pt'] == epoch9_sha
counts = {}
for j in new_jobs:
    key = f"{j['host']}/{j['script'].split('/')[-1]}"
    counts[key] = counts.get(key, 0) + 1
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'protocol': __doc__.split('\n\n')[1], 'diagnostic_rule': __doc__.split('\n\n')[2],
          'remote_release': 'e4206f1e3106cfd115517d4e33fb967199f2e74a', 'epoch9_export_sha256': epoch9_sha,
          'reused_hammer_pretraining': [f's4-{m}-hammer-rem-b' for m in PRETRAIN], 'counts': counts, 'jobs': new_jobs}
proof = ROOT / f'runs/setup/baselines-eight-task-registration-{TAG}.json'
if not PUBLISH:
    print(json.dumps({'counts': counts, 'ids': [j['id'] for j in new_jobs][::12]}, indent=1))
    raise SystemExit(0)
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not proof.exists()
    before = jobs.QUEUE.read_bytes()
    text = before.decode()
    node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    staged = ROOT / f'runs/setup/baselines-eight-task-queue-candidate-{TAG}.yaml'
    staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                      + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n' + text[node.end_mark.index:])
    assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
    (ROOT / f'runs/setup/baselines-eight-task-queue-before-{TAG}.yaml').write_bytes(before)
    tmp = jobs.QUEUE.with_suffix('.bx8.tmp')
    tmp.write_bytes(staged.read_bytes())
    os.replace(tmp, jobs.QUEUE)
    record.update(registered_at=datetime.now().astimezone().isoformat(),
                  queue_sha256_before=hashlib.sha256(before).hexdigest(),
                  queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
    proof.write_text(json.dumps(record, indent=2) + '\n')
    print('REGISTERED', len(new_jobs), proof)
