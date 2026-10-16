"""Remote placement of the new-baseline hammer sanity set (host-neutral, user 2026-10-15; registered 2026-10-16 before
any result). Same scientific definitions as runs/setup/baselines-sanity-registration-20261016.json (recipes, seeds,
400k agent steps, development RL seeds 1000-1002, criterion), executed on the remote release accepted as cffa071:

- pretraining `s4-{dynamo,snerl,mvp3d}-hammer-rem`, a fresh remote run with the same config and seed. The local
  `s4-snerl-hammer` start (~15k of 2.4M steps on a 1/4-GPU MIG instance, ~6 days) is stopped as superseded; local
  `s4-{dynamo,mvp3d}-hammer` never started;
- random arms use the seeded init exports already written locally by `s4-{dynamo,snerl}-hammer-init` (byte-identical
  to the pretrain run's own encoder_init.pt, checked for SNeRL), uploaded with checksums; 3D-MVP's random arm uses its
  remote pretraining run's encoder_init.pt;
- the completed local `bxs-dynamo-hammer-rand-s1000` pair is kept (arms may span hosts; host recorded), so DynaMo's
  random arm adds only seeds 1001-1002 remotely.

    python -I register_baselines_sanity_remote.py [--publish]
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
TAG = '20261016'
SEEDS = (1000, 1001, 1002)
STEPS = 400000
LOCAL = json.loads((ROOT / f'runs/setup/baselines-sanity-registration-{TAG}.json').read_text())
PRETRAIN = {'dynamo': ('scripts/train_dynamo.py', 8, 48), 'snerl': ('scripts/train_snerl.py', 16, 32),
            'mvp3d': ('scripts/train_mvp3d.py', 12, 32)}
INIT = {'dynamo': 'runs/pretrain/s4-dynamo-hammer-init/encoder_init.pt',
        'snerl': 'runs/pretrain/s4-snerl-hammer-init/encoder_init.pt'}
KEEP_LOCAL = {'bxs-dynamo-hammer-rand-s1000'}
meta = {'stage': 4, 'baseline_expansion': f'hammer-sanity-remote-{TAG}'}

new_jobs = []
for method, (script, mem, ram) in PRETRAIN.items():
    pid = f's4-{method}-hammer-rem'
    new_jobs.append({'id': pid, 'host': 'remote', 'script': script,
                     'args': ['--config', f'configs/baselines/{method}/hammer.yaml', '--name', pid, '--output-root', 'runs',
                              '--resume', 'auto'],
                     'gpu': 'any', 'mem_gb': mem, 'ram_gb': ram, 'disk_gb': 20, 'priority': 1, 'deps': [],
                     'max_restarts': 2, 'baseline': method, **meta,
                     'required_results': ['encoder.pt', 'encoder_init.pt', 'metrics.jsonl', 'completion.json']})
for local in LOCAL['jobs']:
    if local['script'] != 'scripts/train_rl.py' or local['id'] in KEEP_LOCAL:
        continue
    method, arm = local['baseline'], local['baseline_arm']
    rid = local['id'] + '-rem'
    args = list(local['args'])
    k = next(i for i, a in enumerate(args) if a.startswith('vision.export_path='))
    if arm == 'pre':
        export, deps = f'runs/s4-{method}-hammer-rem/encoder.pt', [f's4-{method}-hammer-rem']
    elif method in INIT:
        export, deps = INIT[method], []
    else:
        export, deps = f'runs/s4-{method}-hammer-rem/encoder_init.pt', [f's4-{method}-hammer-rem']
    args[k] = f'vision.export_path={export}'
    args[args.index('--run-dir') + 1] = f'runs/{rid}'
    common = {'host': 'remote', 'gpu': 'any', 'priority': 2, 'deps': deps, 'max_restarts': 1, 'baseline': method,
              'baseline_arm': arm, **meta}
    new_jobs.append({**common, 'id': rid, 'script': 'scripts/train_rl.py', 'args': args, 'mem_gb': 4, 'ram_gb': 16,
                     'disk_gb': 5, 'required_results': ['final.json', 'train.jsonl'], 'baseline_export': export})
    new_jobs.append({**common, 'id': rid + '-eval', 'script': 'scripts/eval_rl.py',
                     'args': ['--run-dir', f'runs/{rid}', '--train-job', rid, '--workers', '3', '--envs-per-worker', '8',
                              '--output-dir', f'runs/{rid}-eval'],
                     'mem_gb': 4, 'ram_gb': 12, 'disk_gb': 1, 'required_results': ['eval.jsonl']})
uploads = {method: {'path': path, 'sha256': hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
           for method, path in INIT.items()}
queue = jobs.load_queue()
assert not {j['id'] for j in queue['jobs']} & {j['id'] for j in new_jobs}
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'same_scientific_definitions_as': f'runs/setup/baselines-sanity-registration-{TAG}.json',
          'criterion': LOCAL['criterion'], 'release': 'cffa071d2db27f507b56a3847454067fb24e6c30',
          'init_uploads': uploads, 'kept_local': sorted(KEEP_LOCAL), 'jobs': new_jobs,
          'superseded_local': sorted(j['id'] for j in LOCAL['jobs'] if j['id'] not in KEEP_LOCAL
                                     and not j['id'].endswith('-init'))}
proof = ROOT / f'runs/setup/baselines-sanity-remote-registration-{TAG}.json'
if not PUBLISH:
    print(json.dumps({'jobs': len(new_jobs), 'ids': [j['id'] for j in new_jobs][:8]}, indent=1))
    raise SystemExit(0)
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not proof.exists()
    before = jobs.QUEUE.read_bytes()
    text = before.decode()
    node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    staged = ROOT / f'runs/setup/baselines-sanity-remote-queue-candidate-{TAG}.yaml'
    staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                      + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n' + text[node.end_mark.index:])
    assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
    (ROOT / f'runs/setup/baselines-sanity-remote-queue-before-{TAG}.yaml').write_bytes(before)
    tmp = jobs.QUEUE.with_suffix('.bxsr.tmp')
    tmp.write_bytes(staged.read_bytes())
    os.replace(tmp, jobs.QUEUE)
    record.update(registered_at=datetime.now().astimezone().isoformat(),
                  queue_sha256_before=hashlib.sha256(before).hexdigest(),
                  queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
    proof.write_text(json.dumps(record, indent=2) + '\n')
    print('REGISTERED', len(new_jobs), proof)
