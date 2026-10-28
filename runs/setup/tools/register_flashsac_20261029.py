"""FlashSAC (no RL image augmentation) comparison groups (user directive 2026-10-29; scripts/train_flashsac.py,
s4d/rl/flashsac.py, configs/rl/algorithms/flashsac.yaml): the eight existing tasks plus drawer-open, window-open,
soccer, sweep-into x {csw4, CNN, SinCro, ReViWo, DynaMo, SNeRL, 3D-MVP A, 3D-MVP B} x seeds 2000-2002, 1M agent steps,
plus the self-render replacement encoders (sr-csw4) on hammer, pick-place, bin-picking with the same seeds.

Exports are always the task's own (never another task's encoder): existing tasks reuse the export of the family's
registered DrM row (as register_downstream_sac_20261028.py); new tasks use ds-csw4-export-seed0-<task>-200k (csw4) and
ds-pre-<baseline>-<task>/encoder.pt (register_newtask_baselines_20261029.py; 3D-MVP's export serves A and B), as
dependencies. Rows ds-flashsac-<family>-<task>-s<seed>[-eval]; ordered seed 2000 first, csw4 and CNN first within a seed.
Placement by measured cost (FlashSAC port report 2026-10-29): frozen-encoder families are local rows (not pool-movable; 1g
MIG: csw4 ~5.4 h, SNeRL ~7.2 h, 3D-MVP A ~11 h per 1M; RSS ~2.4 GB, replay 0.2-2.5 GB RAM), the end-to-end CNN is a
remote row (~18 h per 1M on a 1g MIG, GPU-bound, 46 GB disk replay; charged a full GPU unit by the remote driver).
Evaluators: eval_rl defaults (6 workers x 8 envs; peak 4-6 GB GPU, ~1 GB RAM per worker).

    python -I register_flashsac_20261029.py [--seeds ...] [--tasks ...] [--families ...] [--publish]
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
sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
from scripts import jobs  # noqa: E402
import pool_placement as pool  # noqa: E402

OLD_TASKS = ('hammer', 'bin-picking', 'door-open', 'peg-insert-side', 'peg-unplug-side', 'pick-place', 'stick-push',
             'coffee-push')
NEW_TASKS = ('drawer-open', 'window-open', 'soccer', 'sweep-into')
FAMILIES = ('csw4', 'cnn', 'sincro', 'reviwo', 'dynamo', 'snerl', 'mvp3d-multiview', 'mvp3d-single', 'sr-csw4')
SR_TASKS = ('hammer', 'pick-place', 'bin-picking')
DRM_REF = {'csw4': 'enc-csw4-drm-{t}-s2000', 'cnn': 's4-drm-cnn-{t}-s2000', 'sincro': 's4-drm-sincro-{t}-s2000',
           'reviwo': 's4-drm-reviwo-{t}-s2000', 'dynamo': 's4-drm-dynamo-{t}-s2000', 'snerl': 's4-drm-snerl-{t}-s2000',
           'mvp3d-multiview': 's4-drm-mvp3d-multiview-{t}-s2000', 'mvp3d-single': 's4-drm-mvp3d-single-{t}-s2000'}
RERUN = {('cnn', 'bin-picking'): '-r4'}
ENCODER = {'csw4': 'splatter4d', 'cnn': 'cnn', 'sincro': 'sincro', 'reviwo': 'reviwo', 'dynamo': 'dynamo',
           'snerl': 'snerl', 'mvp3d-multiview': 'mvp3d_multiview', 'mvp3d-single': 'mvp3d_single', 'sr-csw4': 'splatter4d'}
NEW_PRE = {'sincro': 'sincro', 'reviwo': 'reviwo', 'dynamo': 'dynamo', 'snerl': 'snerl', 'mvp3d-multiview': 'mvp3d',
           'mvp3d-single': 'mvp3d'}
RES = {'cnn': {'mem_gb': 6, 'ram_gb': 48, 'disk_gb': 50}, 'mvp3d-multiview': {'mem_gb': 4, 'ram_gb': 8, 'disk_gb': 10}}
DEFAULT_RES = {'mem_gb': 4, 'ram_gb': 6, 'disk_gb': 10}
EVAL_RES = {'mem_gb': 6, 'ram_gb': 8, 'disk_gb': 1}
PRIORITY, STAGE = 2, 7
DEFINITION = 'runs/setup/downstream-selection-rule-20261028-a1.json'
STAMP = datetime.now().astimezone().isoformat(timespec='seconds')

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', nargs='+', type=int, default=[2000, 2001, 2002])
ap.add_argument('--tasks', nargs='+', default=list(OLD_TASKS + NEW_TASKS))
ap.add_argument('--families', nargs='+', default=list(FAMILIES))
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
queue = jobs.load_queue()
in_queue = {j['id']: j for j in queue['jobs']}
state = jobs.read_registry()


def source(fam, task):
    """(encoder yaml, export path or None, deps) of a family on a task: always the task's own encoder."""
    if fam == 'sr-csw4':
        eid = f'sr-csw4-export-seed0-{task}-200k'
        return ENCODER[fam], f'runs/{eid}/encoder.pt', [eid]
    if task in OLD_TASKS:
        ref = in_queue[DRM_REF[fam].format(t=task) + RERUN.get((fam, task), '')]
        assert state.get(ref['id'], {}).get('status') == jobs.DONE, ref['id']
        a = ref['args']
        exports = [s.split('=', 1)[1] for s in a[a.index('--set') + 1:] if s.startswith('vision.export_path=')] \
            if '--set' in a else []
        assert a[a.index('--encoder') + 1] == ENCODER[fam] and (fam == 'cnn') == (not exports), ref['id']
        return ENCODER[fam], exports[0] if exports else None, list(ref.get('deps', []))
    if fam == 'cnn':
        return 'cnn', None, []
    if fam == 'csw4':
        eid = f'ds-csw4-export-seed0-{task}-200k'
        return 'splatter4d', f'runs/{eid}/encoder.pt', [eid]
    pid = f'ds-pre-{NEW_PRE[fam]}-{task}'
    return ENCODER[fam], f'runs/{pid}/encoder.pt', [pid]


new_jobs = []
order = sorted(args.families, key=lambda f: (f not in ('csw4', 'cnn'), FAMILIES.index(f)))
for seed in sorted(args.seeds):
    for fam in order:
        for task in args.tasks:
            if fam == 'sr-csw4' and task not in SR_TASKS:
                continue
            encoder, export, deps = source(fam, task)
            for dep in deps:
                assert dep in in_queue, f'{fam}/{task}: dependency {dep} is not registered'
            rid = f'ds-flashsac-{fam}-{task}-s{seed}'
            remote = fam == 'cnn'
            placement = (f'pool {STAMP} (registered on the remote: end-to-end CNN, GPU-bound, disk replay)' if remote
                         else f'pool {STAMP} (registered local: frozen-encoder FlashSAC, local MIG; not a pool-movable script)')
            meta = {'stage': STAGE, 'variant': f'flashsac_{fam}', 'variant_definition': DEFINITION, 'placement': placement,
                    'priority': PRIORITY}
            sets = ['train.num_train_steps=1000000'] + ([f'vision.export_path={export}'] if export else [])
            trainer = {'host': 'remote' if remote else 'local', 'gpu': 'any', 'deps': list(deps), 'max_restarts': 1,
                       'id': rid, 'script': 'scripts/train_flashsac.py',
                       'args': ['--task', task, '--encoder', encoder, '--seed', str(seed), '--run-dir', f'runs/{rid}',
                                '--set', *sets],
                       **RES.get(fam, DEFAULT_RES), 'required_results': ['final.json', 'train.jsonl'], **meta}
            evaluator = {'host': trainer['host'], 'gpu': 'any', 'deps': list(deps), 'max_restarts': 1, 'id': rid + '-eval',
                         'script': 'scripts/eval_rl.py',
                         'args': ['--run-dir', f'runs/{rid}', '--train-job', rid, '--workers', '6', '--envs-per-worker',
                                  '8', '--output-dir', f'runs/{rid}-eval'],
                         **EVAL_RES, 'required_results': ['eval.jsonl'], **meta}
            new_jobs += [trainer, evaluator]
ids = [j['id'] for j in new_jobs]
assert len(ids) == len(set(ids)) and not set(ids) & set(in_queue), 'duplicate or already registered'
assert not any((ROOT / 'runs' / i).exists() for i in ids), 'run dir exists'
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'algorithm': 'flashsac', 'algorithm_config': 'configs/rl/algorithms/flashsac.yaml',
          'label': 'FlashSAC, no-image-augmentation adaptation (official state networks + our visual adapter)',
          'seeds': args.seeds, 'tasks': args.tasks, 'families': args.families, 'definition': DEFINITION,
          'counts': {s: sum(j['script'] == s for j in new_jobs) for s in sorted({j['script'] for j in new_jobs})},
          'jobs': new_jobs}
proof = ROOT / 'runs/setup/flashsac-registration-20261029.json'
if not args.publish:
    print(json.dumps(record['counts']), len(new_jobs), proof.name)
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
        tmp = jobs.QUEUE.with_suffix('.flashsac.tmp')
        tmp.write_bytes(staged.read_bytes())
        os.replace(tmp, jobs.QUEUE)
        record.update(registered_at=datetime.now().astimezone().isoformat(),
                      queue_sha256_before=hashlib.sha256(before).hexdigest(),
                      queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
        proof.write_text(json.dumps(record, indent=2) + '\n')
        print('REGISTERED', len(new_jobs), proof)
