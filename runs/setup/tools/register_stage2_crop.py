"""Crop follow-up step 2 (user request 2026-10-15): two further Stage 2 attribution variants on the frozen method
(C + D0 + M3D, 200k, seed 0), hammer and pick-place, local (same host as the other Stage 2 runs, so the Stage 2
reporting margins apply):
  (a) crop        - the frozen configuration plus aug.crop as defined on 2026-10-08 (prob 0.5, scale [0.8, 1.0],
                    aspect [0.95, 1.05]; encoder input only, training only, never at RL time);
  (b) cropnosynth - aug.synth_views=0 and aug.synth_render=false, plus the same crop.
Each: 200k pretraining, full-split evaluation, guarded export, DrM RL seeds 1000-1002 at 1M agent steps with the
standard evaluation companion. Job definitions mirror register_stage2.py exactly. Attribution only; the frozen
method does not change.

Default: write the candidate registration for review and change nothing else.
`--publish`: append the jobs to the queue (only after the user has approved the shown registration).

    python -I register_stage2_crop.py [--publish]
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
from s4d.config import load_config  # noqa: E402
from scripts import jobs  # noqa: E402

PUBLISH = '--publish' in sys.argv
TAG = '20261015'
MARGINS_SHA = '5fadd04fd5050a440f13d5adad540e424a0f78257ed3f82eacec36df31db5677'
TASKS = ('hammer', 'pick-place')
CROP_SETS = ['aug.crop.prob=0.5', 'aug.crop.scale=[0.8,1.0]', 'aug.crop.ratio=[0.95,1.05]']
CROP_DIFF = {'aug.crop.prob': 0.5, 'aug.crop.scale': [0.8, 1.0], 'aug.crop.ratio': [0.95, 1.05]}
# name: (extra overrides, expected flattened difference from the frozen config)
VARIANTS = {
    'crop': (CROP_SETS, CROP_DIFF),
    'cropnosynth': (['aug.synth_views=0', 'aug.synth_render=false', *CROP_SETS],
                    {'aug.synth_views': 0, 'aug.synth_render': False, **CROP_DIFF}),
}
# measured on this host (registry launch-to-exit, completed Stage 2 jobs, GPUs shared as now): job wall-clock hours
MEASURED_HOURS = {'pretrain_200k': (17.65, 20.73), 'fulleval': (0.08, 0.23), 'export': (0.02, 0.04),
                  'rl_trainer_1M': (4.78, 6.69), 'rl_evaluator_1M': (1.57, 6.48)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def flat(tree, prefix=''):
    out = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(flat(value, f'{prefix}{key}.'))
        else:
            out[prefix + key] = value
    return out


freeze_path = ROOT / 'runs/setup/method-frozen-v1.json'
margins_path = ROOT / 'runs/setup/stage2-reporting-margins-20261013.json'
freeze = json.loads(freeze_path.read_text())
freeze_sha = sha(freeze_path)
assert sha(margins_path) == MARGINS_SHA
commit = freeze['source_commit']
assert freeze['pretraining_steps'] == 200000 and freeze['ablation_rl_seeds'] == [1000, 1001, 1002]
assert git('rev-parse', 'method-frozen-v1^{commit}').strip() == commit
assert not git('diff', commit, 'HEAD', '--', 's4d', 'scripts', 'tests', 'configs', 'pyproject.toml', 'uv.lock')
meta = {'stage': 2, 'frozen_method': 'method-frozen-v1', 'freeze_manifest_sha256': freeze_sha,
        'stage2_margins_sha256': MARGINS_SHA, 'stage2_extension': f'crop-{TAG}'}

new_jobs, configs, exports = [], {}, {}
for variant, (sets, expected) in VARIANTS.items():
    for task in TASKS:
        cfg = load_config([ROOT / 'configs/metaworld/base.yaml', ROOT / f'configs/metaworld/tasks/{task}.yaml'],
                          [*freeze['overrides'], *sets])
        frozen = flat(freeze['task_configs'][task])
        got = flat(cfg)
        diff = {k: got.get(k) for k in set(frozen) | set(got) if got.get(k, KeyError) != frozen.get(k, KeyError)}
        assert diff == expected, (variant, task, diff)
        trainer = f's2ab-{variant}-seed0-{task}'
        configs[trainer] = cfg
        checkpoint = f'runs/pretrain/{trainer}/checkpoints/step_0200000.pt'
        new_jobs.append({
            'id': trainer, 'script': 'scripts/train.py',
            'args': ['--config', 'configs/metaworld/base.yaml', f'configs/metaworld/tasks/{task}.yaml',
                     '--name', trainer, '--output-root', 'runs/pretrain', '--resume', 'auto',
                     '--set', *freeze['overrides'], *sets],
            'gpu': 'any', 'mem_gb': 10, 'ram_gb': 13, 'priority': 0, 'deps': [], 'max_restarts': 6,
            'oom_score_adj': 0, 'stage2_variant': variant, **meta})
        evaluation = f'fulleval-{trainer}-200k'
        new_jobs.append({
            'id': evaluation, 'script': 'scripts/evaluate.py',
            'args': ['--run', f'runs/{evaluation}', '--checkpoint', checkpoint],
            'gpu': 'any', 'mem_gb': 5, 'ram_gb': 12, 'priority': 1, 'deps': [trainer], 'max_restarts': 1,
            'oom_score_adj': 300, 'stage2_variant': variant, **meta})
        exporter = f'export-{trainer}-200k'
        out = f'runs/exports/{trainer}-200k.pt'
        new_jobs.append({
            'id': exporter, 'script': 'scripts/export_encoder.py',
            'args': ['--run', f'runs/pretrain/{trainer}', '--checkpoint', checkpoint, '--out', out, '--wait'],
            'gpu': 'any', 'mem_gb': 1, 'ram_gb': 4, 'priority': 0, 'deps': [trainer], 'max_restarts': 1,
            'oom_score_adj': 300, 'stage2_variant': variant, **meta})
        exports[(variant, task)] = (out, exporter)

for seed in freeze['ablation_rl_seeds']:
    for task in TASKS:
        for arm in VARIANTS:
            export, dependency = exports[(arm, task)]
            trainer = f's2rl-{arm}-{task}-s{seed}'
            common = {'gpu': 'any', 'priority': 1, 'deps': [dependency], 'max_restarts': 1, 'stage2_variant': arm,
                      **meta}
            new_jobs.append({**common, 'id': trainer, 'script': 'scripts/train_rl.py',
                             'args': ['--task', task, '--encoder', 'splatter4d', '--seed', str(seed),
                                      '--run-dir', f'runs/{trainer}', '--set', 'train.num_train_steps=1000000',
                                      f'vision.export_path={export}'],
                             'mem_gb': 4, 'ram_gb': 6, 'oom_score_adj': 300})
            new_jobs.append({**common, 'id': trainer + '-eval', 'script': 'scripts/eval_rl.py',
                             'args': ['--run-dir', f'runs/{trainer}', '--train-job', trainer,
                                      '--workers', '3', '--envs-per-worker', '8'],
                             'mem_gb': 3, 'ram_gb': 6, 'oom_score_adj': 400})

# User approval 2026-10-15: middle admission rank, i.e. after every pending original Stage 2 RL pair (so the 3-seed
# Stage 2 comparison finishes first) and ahead of the 12 pending pretrain-only variants; all local.
RANK_NOTE = ('middle: admitted after every original Stage 2 RL job has been admitted, and before the pretrain-only '
             'Stage 2 variants (user approval 2026-10-15); local, same host as the Stage 2 references')
for job in new_jobs:
    assert 'host' not in job  # local (the Stage 2 local driver admits local jobs only)
    job['stage2_admission_rank'] = 'middle'

counts = {s: sum(j['script'] == s for j in new_jobs) for s in
          ('scripts/train.py', 'scripts/evaluate.py', 'scripts/export_encoder.py', 'scripts/train_rl.py',
           'scripts/eval_rl.py')}
assert counts == {'scripts/train.py': 4, 'scripts/evaluate.py': 4, 'scripts/export_encoder.py': 4,
                  'scripts/train_rl.py': 12, 'scripts/eval_rl.py': 12}, counts
assert len({j['id'] for j in new_jobs}) == len(new_jobs) == 36
low, high = (sum(MEASURED_HOURS[k][i] * n for k, n in (('pretrain_200k', 4), ('fulleval', 4), ('export', 4),
                                                       ('rl_trainer_1M', 12))) for i in (0, 1))
estimate = {'basis': 'measured job wall-clock hours of completed Stage 2 jobs on this host, GPUs shared as now; '
                     'cropnosynth pretraining ~3-8 % faster per step (10-09 crop vs synth screens, run side by side)',
            'per_job_hours': MEASURED_HOURS, 'gpu_job_hours_excluding_evaluators': [round(low, 1), round(high, 1)],
            'rl_evaluator_companion_hours': [round(12 * MEASURED_HOURS['rl_evaluator_1M'][0], 1),
                                             round(12 * MEASURED_HOURS['rl_evaluator_1M'][1], 1)]}
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'request': 'crop follow-up step 2 (user, 2026-10-15)',
          'source_commit': commit, 'method_tag': 'method-frozen-v1', 'freeze_manifest_sha256': freeze_sha,
          'reporting_margins': str(margins_path.relative_to(ROOT)), 'reporting_margins_sha256': MARGINS_SHA,
          'host': 'local (same host as the other Stage 2 runs)', 'pretraining_steps': 200000, 'pretraining_seed': 0,
          'rl_seeds': freeze['ablation_rl_seeds'], 'rl_agent_steps': 1000000,
          'variants': {v: {'overrides': s, 'expected_difference': e} for v, (s, e) in VARIANTS.items()},
          'rl_arms': list(VARIANTS), 'encoders': {f'{a}/{t}': {'export': e, 'dependency': d}
                                                  for (a, t), (e, d) in exports.items()},
          'counts': counts, 'estimate': estimate, 'configs': configs, 'jobs': new_jobs,
          'purpose': 'attribution only; not a method-selection gate; the frozen method and its hyperparameters do '
                     'not change',
          'admission': 'Stage 2 local driver, job by job under the local full-charge cap', 'admission_rank': RANK_NOTE}
shown_path = ROOT / f'runs/setup/stage2-crop-registration-candidate-{TAG}.json'  # the version shown to the user
shown = json.loads(shown_path.read_text())
assert shown['configs'] == configs
assert shown['jobs'] == [{k: v for k, v in j.items() if k != 'stage2_admission_rank'} for j in new_jobs]
candidate_path = ROOT / f'runs/setup/stage2-crop-registration-candidate-v2-{TAG}.json'
proof_path = ROOT / f'runs/setup/stage2-crop-registration-{TAG}.json'
if not PUBLISH:
    if candidate_path.exists():
        old = json.loads(candidate_path.read_text())
        assert old['jobs'] == new_jobs and old['configs'] == configs, 'candidate changed'
    else:
        candidate_path.write_text(json.dumps(record, indent=2) + '\n')
    print('CANDIDATE', candidate_path, sha(candidate_path))
    raise SystemExit(0)

candidate = json.loads(candidate_path.read_text())
assert candidate['jobs'] == new_jobs and candidate['configs'] == configs, 'jobs differ from the shown candidate'
queue_path = jobs.QUEUE
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = jobs.load_queue()
    old = {j['id']: j for j in queue['jobs']}
    existing = [j for j in new_jobs if j['id'] in old]
    if existing:
        assert len(existing) == len(new_jobs) and all(old[j['id']] == j for j in new_jobs) and proof_path.is_file()
        print('ALREADY_REGISTERED', len(existing))
        raise SystemExit(0)
    assert not proof_path.exists()
    assert not any((ROOT / 'runs/pretrain' / j['id']).exists() or (ROOT / 'runs' / j['id']).exists() for j in new_jobs)
    before = queue_path.read_bytes()
    staged = ROOT / f'runs/setup/stage2-crop-queue-candidate-{TAG}.yaml'
    assert not staged.exists()
    text = before.decode()
    jobs_node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    boundary = jobs_node.end_mark.index
    staged.write_text(text[:boundary].rstrip() + '\n' + yaml.safe_dump(json.loads(json.dumps(new_jobs)),
                                                                       sort_keys=False) + '\n' + text[boundary:])
    combined = jobs.load_queue(staged)
    assert combined['jobs'] == queue['jobs'] + new_jobs
    assert {k: v for k, v in combined.items() if k != 'jobs'} == {k: v for k, v in queue.items() if k != 'jobs'}
    snapshot = ROOT / f'runs/setup/stage2-crop-queue-before-{TAG}.yaml'
    assert not snapshot.exists()
    snapshot.write_bytes(before)
    temporary = queue_path.with_suffix('.stage2-crop.tmp')
    assert not temporary.exists()
    temporary.write_bytes(staged.read_bytes())
    os.replace(temporary, queue_path)
    proof = {**candidate, 'registered_at': datetime.now().astimezone().isoformat(),
             'candidate_sha256': sha(candidate_path), 'shown_candidate_sha256': sha(shown_path),
             'difference_from_shown': 'stage2_admission_rank=middle on every job (user approval 2026-10-15)',
             'queue_sha256_before': hashlib.sha256(before).hexdigest(),
             'queue_sha256_after': sha(queue_path)}
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    print('REGISTERED', len(new_jobs), proof_path, sha(proof_path))
