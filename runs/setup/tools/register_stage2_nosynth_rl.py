"""Stage 2 RL arm for the already-completed no-synthetic-view pretrains (registered 2026-10-16, before any RL result).

The pretrain-only variant `s2ab-nosynth-seed0-{hammer,pick-place}` (aug.synth_views=0, aug.synth_render=false; no
crop) finished 200k with driver closeouts. This adds its guarded 200k exports and DrM pairs (seeds 1000-1002, 1M agent
steps, standard evaluation companion), mirroring register_stage2.py / register_stage2_crop.py job definitions exactly.
Comparison: against the existing full-method references s2rl-full-<task>-s100{0,1,2}, with the existing Stage 2 RL
rule (seed-mean last-5 success difference, margin max(0.10, 2 x SE of the difference), per camera condition) and the
RL reporting standard (success and return, last-5/final/AUC, 95% bootstrap CIs). Attribution only; the frozen method
does not change. Host-neutral: admitted locally by the Stage 2 local driver.

    python -I register_stage2_nosynth_rl.py [--publish]
"""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

PUBLISH = '--publish' in sys.argv
TAG = '20261016'
MARGINS_SHA = '5fadd04fd5050a440f13d5adad540e424a0f78257ed3f82eacec36df31db5677'
TASKS = ('hammer', 'pick-place')
ARM = 'nosynth'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, check=True).stdout


freeze_path = ROOT / 'runs/setup/method-frozen-v1.json'
margins_path = ROOT / 'runs/setup/stage2-reporting-margins-20261013.json'
freeze = json.loads(freeze_path.read_text())
freeze_sha = sha(freeze_path)
assert sha(margins_path) == MARGINS_SHA
commit = freeze['source_commit']
assert freeze['ablation_rl_seeds'] == [1000, 1001, 1002]
# Scientific sources (model, losses, data, RL, evaluation, configs other than the local GPU host list) are unchanged
# since the frozen commit; only GPU selection/scheduling files may differ.
changed = git('diff', '--name-only', commit, 'HEAD', '--', 's4d', 'scripts', 'configs', 'pyproject.toml',
              'uv.lock').split()
allowed = {'s4d/gpu_guard.py', 's4d/remote_jobs.py', 's4d/remote_launch.py', 'scripts/jobs.py',
           'scripts/_nvidia_query.py', 'scripts/watch_events.py', 'scripts/check_gpu_isolation.py',
           'scripts/gpu_env.sh', 'configs/hosts/local.yaml'}
assert set(changed) <= allowed, sorted(set(changed) - allowed)

registration = json.loads((ROOT / 'runs/setup/stage2-registration-20261013.json').read_text())
meta = {'stage': 2, 'frozen_method': 'method-frozen-v1', 'freeze_manifest_sha256': freeze_sha,
        'stage2_margins_sha256': MARGINS_SHA, 'stage2_extension': f'nosynth-rl-{TAG}'}
queue = jobs.load_queue()
by_id = {j['id']: j for j in queue['jobs']}
state = jobs.read_registry()
template = {j['id']: j for j in registration['jobs']}

new_jobs, exports = [], {}
for task in TASKS:
    trainer = f's2ab-{ARM}-seed0-{task}'
    assert trainer in registration['configs'] and state[trainer]['status'] == jobs.DONE
    assert (ROOT / 'runs/setup' / f's2-{trainer}-completion.json').is_file()
    checkpoint = f'runs/pretrain/{trainer}/checkpoints/step_0200000.pt'
    exporter = f'export-{trainer}-200k'
    out = f'runs/exports/{trainer}-200k.pt'
    reference = template[f'export-s2ab-t1-seed0-{task}-200k']  # the registered export definition, other variant
    job = {**reference, 'id': exporter,
           'args': ['--run', f'runs/pretrain/{trainer}', '--checkpoint', checkpoint, '--out', out, '--wait'],
           'deps': [trainer], **meta}
    job.pop('stage2_variant', None)
    job['stage2_variant'] = ARM
    new_jobs.append(job)
    exports[task] = (out, exporter)

for seed in freeze['ablation_rl_seeds']:
    for task in TASKS:
        export, dependency = exports[task]
        trainer = f's2rl-{ARM}-{task}-s{seed}'
        reference = template[f's2rl-t1-{task}-s{seed}']
        args = list(reference['args'])
        index = next(i for i, a in enumerate(args) if a.startswith('vision.export_path='))
        args[index] = f'vision.export_path={export}'
        args[args.index('--run-dir') + 1] = f'runs/{trainer}'
        new_jobs.append({**reference, 'id': trainer, 'args': args, 'deps': [dependency], **meta,
                         'stage2_variant': ARM})
        evaluator = template[f's2rl-t1-{task}-s{seed}-eval']
        eargs = list(evaluator['args'])
        eargs[eargs.index('--run-dir') + 1] = f'runs/{trainer}'
        eargs[eargs.index('--train-job') + 1] = trainer
        new_jobs.append({**evaluator, 'id': trainer + '-eval', 'args': eargs, 'deps': [dependency], **meta,
                         'stage2_variant': ARM})

for job in new_jobs:
    assert 'host' not in job and 'stage2_admission_rank' not in job
    assert job['id'] not in by_id
counts = {s: sum(j['script'] == s for j in new_jobs) for s in
          ('scripts/export_encoder.py', 'scripts/train_rl.py', 'scripts/eval_rl.py')}
assert counts == {'scripts/export_encoder.py': 2, 'scripts/train_rl.py': 6, 'scripts/eval_rl.py': 6}, counts
rule = {
    'reference_arm': 'full (s2rl-full-<task>-s1000..s1002, existing completed runs on the reused A200 encoders)',
    'decision_quantity': 'per task and camera condition (training cameras, held-out cameras, trajectories): mean over '
                         'RL seeds 1000-1002 of the last-5 success; difference variant minus full',
    'margin': 'max(0.10, 2 x standard error of the difference of seed means), the existing Stage 2 RL rule '
              '(runs/setup/tools/stage2_report.py)',
    'reporting': 'success and return, last-5 / final (1M) / AUC (100 points), per task seed values, mean and 95% '
                 'bootstrap CI (10,000 resamples, RNG 0), via runs/setup/tools/rl_reporting.py; camera conditions '
                 'separate',
    'interpretation': 'attribution of the synthetic-view component (C) for RL, including held-out-camera transfer; '
                      'within-margin is not evidence of no effect; no method selection, reseeding or retuning',
}
record = {'prepared_at': datetime.now().astimezone().isoformat(),
          'request': 'host-neutral delegated registration (user 2026-10-15/16): RL arm for the completed no-synth '
                     'pretrains, registered before any RL result of this arm exists',
          'source_commit_frozen': commit, 'source_commit_head': git('rev-parse', 'HEAD').strip(),
          'method_tag': 'method-frozen-v1', 'freeze_manifest_sha256': freeze_sha,
          'reporting_margins_sha256': MARGINS_SHA, 'rl_seeds': freeze['ablation_rl_seeds'], 'rl_agent_steps': 1000000,
          'rule': rule, 'counts': counts, 'configs': {}, 'jobs': new_jobs,
          'templates': 'export-s2ab-t1-seed0-<task>-200k and s2rl-t1-<task>-s<seed>[-eval] of the 2026-10-13 '
                       'registration, with only id, export path, run dir, train job and dependency changed',
          'admission': 'Stage 2 local driver, job by job under the local full-charge cap (host-neutral)'}
proof_path = ROOT / f'runs/setup/stage2-nosynth-rl-registration-{TAG}.json'
if not PUBLISH:
    print(json.dumps({'counts': counts, 'jobs': [j['id'] for j in new_jobs]}, indent=1))
    raise SystemExit(0)

queue_path = jobs.QUEUE
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = jobs.load_queue()
    assert not proof_path.exists()
    assert not any(j['id'] in {q['id'] for q in queue['jobs']} for j in new_jobs)
    assert not any((ROOT / 'runs' / j['id']).exists() for j in new_jobs)
    before = queue_path.read_bytes()
    text = before.decode()
    jobs_node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    boundary = jobs_node.end_mark.index
    staged = ROOT / f'runs/setup/stage2-nosynth-rl-queue-candidate-{TAG}.yaml'
    assert not staged.exists()
    staged.write_text(text[:boundary].rstrip() + '\n' + yaml.safe_dump(json.loads(json.dumps(new_jobs)),
                                                                       sort_keys=False) + '\n' + text[boundary:])
    combined = jobs.load_queue(staged)
    assert combined['jobs'] == queue['jobs'] + new_jobs
    assert {k: v for k, v in combined.items() if k != 'jobs'} == {k: v for k, v in queue.items() if k != 'jobs'}
    snapshot = ROOT / f'runs/setup/stage2-nosynth-rl-queue-before-{TAG}.yaml'
    snapshot.write_bytes(before)
    temporary = queue_path.with_suffix('.nosynth-rl.tmp')
    temporary.write_bytes(staged.read_bytes())
    os.replace(temporary, queue_path)
    record.update(registered_at=datetime.now().astimezone().isoformat(),
                  queue_sha256_before=hashlib.sha256(before).hexdigest(), queue_sha256_after=sha(queue_path))
    proof_path.write_text(json.dumps(record, indent=2) + '\n')
    print('REGISTERED', len(new_jobs), proof_path, sha(proof_path))
