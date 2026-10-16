"""New baselines, acceptance step 3 (docs/BASELINES.md "Acceptance and scientific order"; user 2026-10-13/14):
hammer pretraining at the unchanged recipes plus the pretrained-frozen versus random-frozen DrM sanity check.

- Pretraining `s4-{dynamo,snerl,mvp3d}-hammer` (seed 0, recipe configs unchanged). These are the final hammer encoders
  of the comparison, not side runs.
- Random arm: the same seeded initial encoder (`encoder_init.pt`). DynaMo/SNeRL write it with their init-only flag in a
  separate run (`<id>-init`, same seed; its hash is checked against the pretrain's own encoder_init.pt before
  reporting); 3D-MVP writes it at the start of its pretraining run (its random arm waits for that run).
- DrM on hammer, development seeds 1000-1002 (final seeds 2000-2002 stay untouched), 400k agent steps, full evaluation
  protocol: pretrained vs random for DynaMo, SNeRL and 3D-MVP condition A; condition B pretrained, one seed.
Sanity criterion (fixed now, before any result): pretrained-frozen should exceed random-frozen in mean last-5
training-camera success on hammer; lack of benefit is investigated (panels, losses, adapters) before the eight-task
runs, never fixed by retuning recipes. Reported once with sizes/costs/panels (success and return, last-5/final/AUC).

    python -I register_baselines_sanity.py [--publish]
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
# Declared reservations until measured (local admission uses them as footprints); GPU memory per MIG instance <= 22 GB.
PRETRAIN = {
    'dynamo': {'script': 'scripts/train_dynamo.py', 'init_flag': '--init-export-only', 'mem_gb': 8, 'ram_gb': 30},
    'snerl': {'script': 'scripts/train_snerl.py', 'init_flag': '--export-init-only', 'mem_gb': 16, 'ram_gb': 20},
    'mvp3d': {'script': 'scripts/train_mvp3d.py', 'init_flag': None, 'mem_gb': 12, 'ram_gb': 20},
}
RL_ARMS = [  # (method, rl encoder config, arm, seeds)
    ('dynamo', 'dynamo', 'pre', SEEDS), ('dynamo', 'dynamo', 'rand', SEEDS),
    ('snerl', 'snerl', 'pre', SEEDS), ('snerl', 'snerl', 'rand', SEEDS),
    ('mvp3d', 'mvp3d_multiview', 'pre', SEEDS), ('mvp3d', 'mvp3d_multiview', 'rand', SEEDS),
    ('mvp3d', 'mvp3d_single', 'pre', SEEDS[:1]),
]
meta = {'stage': 4, 'baseline_expansion': f'hammer-sanity-{TAG}'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


new_jobs = []
for method, spec in PRETRAIN.items():
    pid = f's4-{method}-hammer'
    config = f'configs/baselines/{method}/hammer.yaml'
    assert (ROOT / config).is_file()
    new_jobs.append({'id': pid, 'script': spec['script'],
                     'args': ['--config', config, '--name', pid, '--output-root', 'runs/pretrain', '--resume', 'auto'],
                     'gpu': 'any', 'mem_gb': spec['mem_gb'], 'ram_gb': spec['ram_gb'], 'priority': 1, 'deps': [],
                     'max_restarts': 2, 'oom_score_adj': 0, 'baseline': method, **meta})
    if spec['init_flag']:
        new_jobs.append({'id': f'{pid}-init', 'script': spec['script'],
                         'args': ['--config', config, '--name', f'{pid}-init', '--output-root', 'runs/pretrain',
                                  spec['init_flag']],
                         'gpu': 'any', 'mem_gb': 4, 'ram_gb': 8, 'priority': 0, 'deps': [], 'max_restarts': 1,
                         'oom_score_adj': 300, 'baseline': method, **meta})

for method, encoder, arm, seeds in RL_ARMS:
    pid = f's4-{method}-hammer'
    if arm == 'pre':
        export, dep = f'runs/pretrain/{pid}/encoder.pt', pid
    else:
        init = f'{pid}-init' if PRETRAIN[method]['init_flag'] else pid
        export, dep = f'runs/pretrain/{init}/encoder_init.pt', init
    for seed in seeds:
        rid = f'bxs-{encoder.replace("_", "-")}-hammer-{arm}-s{seed}'
        common = {'gpu': 'any', 'priority': 2, 'deps': [dep], 'max_restarts': 1, 'baseline': method,
                  'baseline_arm': arm, **meta}
        new_jobs.append({**common, 'id': rid, 'script': 'scripts/train_rl.py',
                         'args': ['--task', 'hammer', '--encoder', encoder, '--seed', str(seed), '--run-dir', f'runs/{rid}',
                                  '--set', f'train.num_train_steps={STEPS}', f'vision.export_path={export}'],
                         'mem_gb': 4, 'ram_gb': 8, 'oom_score_adj': 300})
        new_jobs.append({**common, 'id': f'{rid}-eval', 'script': 'scripts/eval_rl.py',
                         'args': ['--run-dir', f'runs/{rid}', '--train-job', rid, '--workers', '3',
                                  '--envs-per-worker', '8'],
                         'mem_gb': 4, 'ram_gb': 10, 'oom_score_adj': 400})

queue = jobs.load_queue()
existing = {j['id'] for j in queue['jobs']}
assert not existing & {j['id'] for j in new_jobs}, existing & {j['id'] for j in new_jobs}
assert len({j['id'] for j in new_jobs}) == len(new_jobs)
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'criterion': 'pretrained-frozen mean last-5 training-camera success above random-frozen on hammer; '
                       'otherwise investigate before the eight-task runs; no recipe retuning',
          'rl_seeds': list(SEEDS), 'rl_agent_steps': STEPS, 'jobs': new_jobs,
          'source_commit': os.popen(f'git -C {ROOT} rev-parse HEAD').read().strip(),
          'admission': 'runs/setup/tools/baseline_local_driver.py, job by job under the local full-charge cap'}
proof_path = ROOT / f'runs/setup/baselines-sanity-registration-{TAG}.json'
if not PUBLISH:
    print(json.dumps([j['id'] for j in new_jobs], indent=0))
    raise SystemExit(0)
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not proof_path.exists()
    before = jobs.QUEUE.read_bytes()
    text = before.decode()
    jobs_node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    boundary = jobs_node.end_mark.index
    staged = ROOT / f'runs/setup/baselines-sanity-queue-candidate-{TAG}.yaml'
    staged.write_text(text[:boundary].rstrip() + '\n' + yaml.safe_dump(json.loads(json.dumps(new_jobs)),
                                                                       sort_keys=False) + '\n' + text[boundary:])
    combined = jobs.load_queue(staged)
    assert combined['jobs'] == jobs.load_queue()['jobs'] + new_jobs
    (ROOT / f'runs/setup/baselines-sanity-queue-before-{TAG}.yaml').write_bytes(before)
    temporary = jobs.QUEUE.with_suffix('.bxs.tmp')
    temporary.write_bytes(staged.read_bytes())
    os.replace(temporary, jobs.QUEUE)
    record.update(registered_at=datetime.now().astimezone().isoformat(),
                  queue_sha256_before=hashlib.sha256(before).hexdigest(), queue_sha256_after=sha(jobs.QUEUE))
    proof_path.write_text(json.dumps(record, indent=2) + '\n')
    print('REGISTERED', len(new_jobs), proof_path)
