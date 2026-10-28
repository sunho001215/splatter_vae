"""SNeRL-code SAC comparison groups (user directive 2026-10-28; scripts/train_sac.py, s4d/rl/snerl_sac.py).

One trainer + evaluator pair per (family, task, seed): ds-sac-<family>-<task>-s<seed>[-eval]. Every family uses exactly
the encoder configuration and export of its registered DrM runs (the s2000 DrM row of the same task: --encoder and
vision.export_path; the CNN bin-picking reference is the -r4 rerun row); nothing is pretrained. Budget 1M agent steps.
Evaluators are the DrM evaluator rows' settings (scripts/eval_rl.py evaluates either algorithm). Placement: frozen-
encoder families start as local pool rows (movable to the remote by pool_placement.py); the end-to-end CNN and SNeRL
(pixel replay: 4.6 / 13.8 GB, per-update convolutions over 128-px frames, about 5 agent steps/s on a 1g MIG slice) are
remote rows. Priority 2: ahead of the encoder pilots and the paused paper ablations (user 2026-10-28).

    python -I register_downstream_sac_20261028.py --seeds 2000 [--tasks ...] [--families ...] [--publish]
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

TASKS = ('door-open', 'peg-unplug-side', 'hammer', 'peg-insert-side', 'bin-picking', 'pick-place', 'stick-push',
         'coffee-push')
FAMILIES = {  # family -> DrM reference row of a task (seed 2000)
    'csw4': 'enc-csw4-drm-{t}-s2000', 'cnn': 's4-drm-cnn-{t}-s2000', 'sincro': 's4-drm-sincro-{t}-s2000',
    'reviwo': 's4-drm-reviwo-{t}-s2000', 'dynamo': 's4-drm-dynamo-{t}-s2000', 'snerl': 's4-drm-snerl-{t}-s2000',
    'mvp3d-multiview': 's4-drm-mvp3d-multiview-{t}-s2000', 'mvp3d-single': 's4-drm-mvp3d-single-{t}-s2000'}
RERUN = {('cnn', 'bin-picking'): '-r4'}
REMOTE = {'cnn', 'snerl'}
RESOURCES = {'cnn': {'mem_gb': 6, 'ram_gb': 12, 'disk_gb': 10}, 'snerl': {'mem_gb': 6, 'ram_gb': 24, 'disk_gb': 20}}
PRIORITY, STAGE = 2, 6
DEFINITION = 'runs/setup/downstream-selection-rule-20261028-a1.json'
STAMP = datetime.now().astimezone().isoformat(timespec='seconds')

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', nargs='+', type=int, required=True)
ap.add_argument('--tasks', nargs='+', default=list(TASKS))
ap.add_argument('--families', nargs='+', default=list(FAMILIES))
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
assert set(args.tasks) <= set(TASKS) and set(args.families) <= set(FAMILIES)
assert set(args.seeds) <= {2000, 2001, 2002}

queue = jobs.load_queue()
in_queue = {j['id']: j for j in queue['jobs']}
state = jobs.read_registry()
new_jobs, references = [], {}
for task in args.tasks:
    for fam in args.families:
        ref_id = FAMILIES[fam].format(t=task) + RERUN.get((fam, task), '')
        ref = in_queue[ref_id]
        assert state.get(ref_id, {}).get('status') == jobs.DONE, f'reference {ref_id} not complete'
        a = ref['args']
        encoder = a[a.index('--encoder') + 1]
        sets = a[a.index('--set') + 1:] if '--set' in a else []
        exports = [s for s in sets if s.startswith('vision.export_path=')]
        assert {s.split('=')[0] for s in sets} <= {'vision.export_path', 'train.num_train_steps'}, (ref_id, sets)
        assert (fam == 'cnn') == (not exports), ref_id
        for dep in ref.get('deps', []):
            assert state.get(dep, {}).get('status') == jobs.DONE, f'{ref_id}: dependency {dep} not complete'
        references[f'{fam}/{task}'] = {'drm_reference': ref_id, 'encoder': encoder, 'export': exports[0].split('=', 1)[1]
                                       if exports else None, 'deps': ref.get('deps', [])}
        for seed in args.seeds:
            rid = f'ds-sac-{fam}-{task}-s{seed}'
            remote = fam in REMOTE
            res = RESOURCES.get(fam, {'mem_gb': ref['mem_gb'], 'ram_gb': ref['ram_gb'], 'disk_gb': ref.get('disk_gb', 10)})
            placement = (f'pool {STAMP} (registered on the remote: pixel replay, per-update convolutions)' if remote
                         else f'pool {STAMP} (registered local; movable by pool_placement.py)')
            meta = {'stage': STAGE, 'variant': f'snerl_sac_{fam}', 'variant_definition': DEFINITION,
                    'placement': placement}
            trainer = {'host': 'remote' if remote else 'local', 'gpu': 'any', 'priority': PRIORITY,
                       'deps': list(ref.get('deps', [])), 'max_restarts': 1, 'id': rid, 'script': 'scripts/train_sac.py',
                       'args': ['--task', task, '--encoder', encoder, '--seed', str(seed), '--run-dir', f'runs/{rid}',
                                '--set', 'train.num_train_steps=1000000', *exports],
                       **res, 'required_results': ['final.json', 'train.jsonl'], **meta}
            # evaluator: the DrM evaluator settings (6 workers x 8 envs, 240 episodes), always with its own output
            # directory and explicit resources (required for remote rows and pool moves)
            evaluator = {'host': trainer['host'], 'gpu': 'any', 'priority': PRIORITY, 'deps': list(ref.get('deps', [])),
                         'max_restarts': 1, 'id': rid + '-eval', 'script': 'scripts/eval_rl.py',
                         'args': ['--run-dir', f'runs/{rid}', '--train-job', rid, '--workers', '6', '--envs-per-worker',
                                  '8', '--output-dir', f'runs/{rid}-eval'],
                         'mem_gb': 3, 'ram_gb': 8, 'disk_gb': 1, 'required_results': ['eval.jsonl'], **meta}
            new_jobs += [trainer, evaluator]
ids = [j['id'] for j in new_jobs]
assert len(ids) == len(set(ids)) and not set(ids) & set(in_queue), 'duplicate or already registered'
assert not any((ROOT / 'runs' / i).exists() for i in ids), 'run dir exists'
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'algorithm': 'snerl_sac', 'algorithm_config': 'configs/rl/algorithms/snerl_sac.yaml',
          'seeds': args.seeds, 'tasks': args.tasks, 'families': args.families, 'references': references,
          'definition': DEFINITION, 'definition_sha256': hashlib.sha256((ROOT / DEFINITION).read_bytes()).hexdigest(),
          'counts': {s: sum(j['script'] == s for j in new_jobs) for s in sorted({j['script'] for j in new_jobs})},
          'jobs': new_jobs}
tag = '-'.join(str(s) for s in args.seeds) + ('' if set(args.tasks) == set(TASKS) else '-' + '-'.join(args.tasks))
proof = ROOT / f'runs/setup/downstream-sac-registration-s{tag}-20261028.json'
if not args.publish:
    print(json.dumps({k: record[k] for k in ('seeds', 'tasks', 'families', 'counts')}, indent=1), proof.name)
    print(json.dumps(new_jobs[:2], indent=1))
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
        assert jobs.load_queue(staged)['hosts'] == queue['hosts']
        tmp = jobs.QUEUE.with_suffix('.downstream.tmp')
        tmp.write_bytes(staged.read_bytes())
        os.replace(tmp, jobs.QUEUE)
        record.update(registered_at=datetime.now().astimezone().isoformat(),
                      queue_sha256_before=hashlib.sha256(before).hexdigest(),
                      queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
        proof.write_text(json.dumps(record, indent=2) + '\n')
        print('REGISTERED', len(new_jobs), proof)
