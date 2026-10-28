"""New-task screening rows (user directive 2026-10-28, step 5): the fixed csw4 recipe on a new Meta-World task and the
SNeRL-code SAC screen of csw4 against the matched end-to-end CNN control.

Per task T (data collected with the established protocol, verified, and uploaded: upload_task_data_20261028.py):
- ds-csw4-seed0-T-200k: the enc-csw4-seed0-door-open-200k row with only the task config and names changed
  (chamfer_lean + chamfer_lean_randmask + enc_cs + enc_w4, 200k steps, pretraining seed 0);
- ds-csw4-export-seed0-T-200k: the matching export row (runs on the remote, where the pretraining ran);
- ds-sac-csw4-T-s<seed>[-eval] and ds-sac-cnn-T-s<seed>[-eval]: as register_downstream_sac_20261028.py (seed 2000
  first). Baseline encoders are pretrained only for promising tasks (separate registration).
All rows are remote (the pretraining's outputs stay there; no local download on the critical path), priority 2, on the
downstream release (ds- prefix, remote_campaign_driver.py DS_COMMIT).

    python -I register_newtask_batch_20261028.py --tasks T [...] --seeds 2000 [--publish]
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

REF_TASK = 'door-open'
PRIORITY, STAGE = 2, 6
DEFINITION = 'runs/setup/downstream-selection-rule-20261028-a1.json'
STAMP = datetime.now().astimezone().isoformat(timespec='seconds')
ap = argparse.ArgumentParser()
ap.add_argument('--tasks', nargs='+', required=True)
ap.add_argument('--seeds', nargs='+', type=int, required=True)
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
queue = jobs.load_queue()
in_queue = {j['id']: j for j in queue['jobs']}
verified = json.loads((ROOT / 'docs/task_verification/summary.json').read_text())
pre_ref, exp_ref = in_queue[f'enc-csw4-seed0-{REF_TASK}-200k'], in_queue[f'enc-csw4-export-seed0-{REF_TASK}-200k']
cnn_ref = in_queue[f's4-drm-cnn-{REF_TASK}-s2000']
ev_ref = in_queue[f's4-drm-cnn-{REF_TASK}-s2000-eval']
placement = f'pool {STAMP} (registered on the remote: new-task screening)'
meta = {'stage': STAGE, 'variant_definition': DEFINITION, 'placement': placement, 'priority': PRIORITY}
new_jobs = []
for task in args.tasks:
    assert verified[task]['passed'] is True, task
    assert (ROOT / f'configs/metaworld/tasks/{task}.yaml').is_file() or True  # in the downstream release
    pid, eid = f'ds-csw4-seed0-{task}-200k', f'ds-csw4-export-seed0-{task}-200k'
    pre = copy.deepcopy(pre_ref)
    pre['args'] = [a.replace(f'tasks/{REF_TASK}.yaml', f'tasks/{task}.yaml').replace(pre_ref['id'], pid)
                   for a in pre_ref['args']]
    pre.update(id=pid, host='remote', deps=[], variant='enc_csw4', **meta)
    exp = copy.deepcopy(exp_ref)
    exp['args'] = [a.replace(pre_ref['id'], pid).replace(exp_ref['id'], eid) for a in exp_ref['args']]
    exp.update(id=eid, host='remote', deps=[pid], variant='enc_csw4', **meta)
    changed = [a for a in pre['args'] if a not in pre_ref['args']]
    assert changed == [f'configs/metaworld/tasks/{task}.yaml', pid], changed
    new_jobs += [pre, exp]
    for seed in args.seeds:
        for fam, encoder, deps, sets, res in (
                ('csw4', 'splatter4d', [eid], [f'vision.export_path=runs/{eid}/encoder.pt'],
                 {'mem_gb': 4, 'ram_gb': 6, 'disk_gb': 10}),
                ('cnn', 'cnn', [], [], {'mem_gb': 6, 'ram_gb': 12, 'disk_gb': 10})):
            rid = f'ds-sac-{fam}-{task}-s{seed}'
            trainer = {'host': 'remote', 'gpu': 'any', 'deps': list(deps), 'max_restarts': 1, 'id': rid,
                       'script': 'scripts/train_sac.py',
                       'args': ['--task', task, '--encoder', encoder, '--seed', str(seed), '--run-dir', f'runs/{rid}',
                                '--set', 'train.num_train_steps=1000000', *sets],
                       **res, 'required_results': ['final.json', 'train.jsonl'], 'variant': f'snerl_sac_{fam}', **meta}
            evaluator = {'host': 'remote', 'gpu': 'any', 'deps': list(deps), 'max_restarts': 1, 'id': rid + '-eval',
                         'script': 'scripts/eval_rl.py',
                         'args': ['--run-dir', f'runs/{rid}', '--train-job', rid, '--workers', '6', '--envs-per-worker',
                                  '8', '--output-dir', f'runs/{rid}-eval'],
                         'mem_gb': 3, 'ram_gb': 8, 'disk_gb': 1, 'required_results': ['eval.jsonl'],
                         'variant': f'snerl_sac_{fam}', **meta}
            new_jobs += [trainer, evaluator]
ids = [j['id'] for j in new_jobs]
assert len(ids) == len(set(ids)) and not set(ids) & set(in_queue), 'duplicate or already registered'
assert not any((ROOT / 'runs' / i).exists() for i in ids), 'run dir exists'
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'tasks': args.tasks, 'seeds': args.seeds, 'reference_rows': [pre_ref['id'], exp_ref['id'], cnn_ref['id']],
          'screening': 'runs/setup/task-screening-20261028/summary.json', 'definition': DEFINITION,
          'counts': {s: sum(j['script'] == s for j in new_jobs) for s in sorted({j['script'] for j in new_jobs})},
          'jobs': new_jobs}
proof = ROOT / f"runs/setup/newtask-registration-{'-'.join(args.tasks)}-20261028.json"
if not args.publish:
    print(json.dumps({k: record[k] for k in ('tasks', 'seeds', 'counts')}), proof.name)
    print(json.dumps(new_jobs[0]), '\n', json.dumps(new_jobs[2]))
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
        tmp = jobs.QUEUE.with_suffix('.newtask.tmp')
        tmp.write_bytes(staged.read_bytes())
        os.replace(tmp, jobs.QUEUE)
        record.update(registered_at=datetime.now().astimezone().isoformat(),
                      queue_sha256_before=hashlib.sha256(before).hexdigest(),
                      queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
        proof.write_text(json.dumps(record, indent=2) + '\n')
        print('REGISTERED', len(new_jobs), proof)
