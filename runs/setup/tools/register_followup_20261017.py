"""Meta-World follow-up experiments (user request 2026-10-17 17:4x), registered before any of their results exists.

The frozen method stays C + D0 + M3D, 200k pretraining steps, pretraining seed 0 (method-frozen-v1). All RL is DrM at
1M agent steps with the existing evaluation protocol (240 episodes every 10k steps; training, held-out and trajectory
(moving) cameras separately). Matched seeds within every comparison; completed runs are reused only when their
configuration, seeds, budget and evaluation protocol are compatible; missing matched controls are run here.

E1 crop on the remaining six tasks (bin-picking, coffee-push, door-open, peg-insert-side, peg-unplug-side, stick-push):
   the frozen configuration plus the crop recipe of the completed hammer/pick-place runs (aug.crop prob 0.5, scale
   [0.8, 1.0], ratio [0.95, 1.05], one shared transformation over the history frames; synthetic views and rendering
   kept). Pretraining seed 0, full-split evaluation, guarded export, DrM seeds 2000-2002 (the final-evaluation seed
   policy; matched to the full method's completed eight-task runs `s3-drm-splatter4d-<task>-s200x`). Missing matched
   controls: full-split evaluations of the full method's six final encoders (`s3-m3d-seed0-<task>-200k`), and crop RL
   on hammer/pick-place at seeds 2000-2002 from the existing crop encoders, so the crop-vs-full comparison exists on
   all eight tasks with the same seeds (the dev-seed hammer/pick-place results stay reported as they are).
   H1: crop changes training-camera RL success, novel-view success, or learning efficiency relative to the full method.
E2 pretraining random shift on hammer and pick-place: the resized crop replaced by DrM-style random shift (replicate
   padding of 4 px, integer crop back to the original resolution; aug.shift prob 0.5 as the crop; one shift per row,
   shared by all history frames and the aligned motion-score input map). Encoder input during pretraining only;
   rendering targets, synthetic views, losses, architecture and the RL setup unchanged. Stage 2 policy: pretraining
   seed 0, DrM development seeds 1000-1002 (matched to the completed `s2rl-full-*` and `s2rl-crop-*` runs).
   H2: shift and crop differ in their effect on representation and RL relative to the full method.
E3 DrM CNN without RL random shift on all eight tasks: fresh end-to-end CNN policies with `agent.random_shift=false`
   (RandomShiftsAug disabled in every update), everything else identical to the completed CNN baseline
   (`s4-drm-cnn-*`, seeds 2000-2002; final-evaluation seed policy). Evaluating existing policies without augmentation
   does not replace this training ablation. H3: removing RL shift lowers performance; the analysis separates general
   learning degradation (training cameras) from loss of viewpoint robustness (held-out / trajectory cameras).

    python -I register_followup_20261017.py [--publish]
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
from s4d.config import load_config  # noqa: E402
from scripts import jobs  # noqa: E402

PUBLISH = '--publish' in sys.argv
TAG = '20261017'
MARGINS_SHA = '5fadd04fd5050a440f13d5adad540e424a0f78257ed3f82eacec36df31db5677'
DEV = ('hammer', 'pick-place')
SIX = ('bin-picking', 'coffee-push', 'door-open', 'peg-insert-side', 'peg-unplug-side', 'stick-push')
EIGHT = ('bin-picking', 'coffee-push', 'door-open', 'hammer', 'peg-insert-side', 'peg-unplug-side', 'pick-place',
         'stick-push')
FINAL_SEEDS = (2000, 2001, 2002)
CROP_SETS = ['aug.crop.prob=0.5', 'aug.crop.scale=[0.8,1.0]', 'aug.crop.ratio=[0.95,1.05]']
CROP_DIFF = {'aug.crop.prob': 0.5, 'aug.crop.scale': [0.8, 1.0], 'aug.crop.ratio': [0.95, 1.05]}
SHIFT_SETS = ['aug.shift.prob=0.5', 'aug.shift.pad=4']
SHIFT_DIFF = {'aug.shift.prob': 0.5, 'aug.shift.pad': 4}
REMOTE = {'host': 'remote', 'cpu_threads': 8}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flat(tree, prefix=''):
    out = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(flat(value, f'{prefix}{key}.'))
        else:
            out[prefix + key] = value
    return out


freeze_path = ROOT / 'runs/setup/method-frozen-v1.json'
freeze = json.loads(freeze_path.read_text())
freeze_sha = sha(freeze_path)
assert sha(ROOT / 'runs/setup/stage2-reporting-margins-20261013.json') == MARGINS_SHA
assert freeze['pretraining_steps'] == 200000 and freeze['ablation_rl_seeds'] == [1000, 1001, 1002]
queue = jobs.load_queue()
by_queue = {j['id']: j for j in queue['jobs']}
state = jobs.read_registry()
meta = {'stage': 2, 'frozen_method': 'method-frozen-v1', 'freeze_manifest_sha256': freeze_sha,
        'stage2_margins_sha256': MARGINS_SHA, 'stage2_extension': f'followup-{TAG}'}

new_jobs, configs, exports, reused = [], {}, {}, {}


def pretrain(variant, task, sets, expected):
    cfg = load_config([ROOT / 'configs/metaworld/base.yaml', ROOT / f'configs/metaworld/tasks/{task}.yaml'],
                      [*freeze['overrides'], *sets])
    frozen, got = flat(freeze['task_configs'][task]), flat(cfg)
    diff = {k: got.get(k) for k in set(frozen) | set(got) if got.get(k, KeyError) != frozen.get(k, KeyError)}
    assert diff == expected, (variant, task, diff)
    trainer = f's2ab-{variant}-seed0-{task}'
    configs[trainer] = cfg
    checkpoint = f'runs/{trainer}/checkpoints/step_0200000.pt'
    new_jobs.append({'id': trainer, 'script': 'scripts/train.py',
                     'args': ['--config', 'configs/metaworld/base.yaml', f'configs/metaworld/tasks/{task}.yaml',
                              '--name', trainer, '--output-root', 'runs', '--resume', 'auto',
                              '--set', *freeze['overrides'], *sets],
                     'gpu': 'any', 'mem_gb': 10, 'ram_gb': 16, 'priority': 0, 'deps': [], 'max_restarts': 6,
                     'oom_score_adj': 0, 'stage2_variant': variant, **meta, **REMOTE, 'disk_gb': 30,
                     'required_results': ['completion.json', 'metrics.jsonl', 'checkpoints/step_0200000.pt'],
                     **({'requires_feature': 'aug-shift'} if variant == 'shift' else {})})
    fulleval(f'fulleval-{trainer}-200k', trainer, checkpoint, variant)
    out = f'runs/exports/{trainer}-200k.pt'
    new_jobs.append({'id': f'export-{trainer}-200k', 'script': 'scripts/export_encoder.py',
                     'args': ['--run', f'runs/{trainer}', '--checkpoint', checkpoint, '--out', out, '--wait'],
                     'gpu': 'any', 'mem_gb': 1, 'ram_gb': 4, 'priority': 0, 'deps': [trainer], 'max_restarts': 1,
                     'oom_score_adj': 300, 'stage2_variant': variant, **meta})
    exports[(variant, task)] = (out, f'export-{trainer}-200k')


def fulleval(eid, trainer, checkpoint, variant):
    new_jobs.append({'id': eid, 'script': 'scripts/evaluate.py',
                     'args': ['--run', f'runs/{eid}', '--checkpoint', checkpoint],
                     'gpu': 'any', 'mem_gb': 5, 'ram_gb': 12, 'priority': 1, 'deps': [trainer], 'max_restarts': 1,
                     'oom_score_adj': 300, 'stage2_variant': variant, **meta, **REMOTE, 'disk_gb': 3,
                     'required_results': ['eval/step_0200000/summary.json']})


def rl_pair(rid, task, seed, export, dependency, variant):
    common = {'gpu': 'any', 'priority': 1, 'deps': [dependency], 'max_restarts': 1, 'stage2_variant': variant,
              **meta, **REMOTE}
    new_jobs.append({**common, 'id': rid, 'script': 'scripts/train_rl.py',
                     'args': ['--task', task, '--encoder', 'splatter4d', '--seed', str(seed), '--run-dir', f'runs/{rid}',
                              '--set', 'train.num_train_steps=1000000', f'vision.export_path={export}'],
                     'mem_gb': 4, 'ram_gb': 8, 'oom_score_adj': 300, 'disk_gb': 5,
                     'required_results': ['final.json', 'train.jsonl']})
    new_jobs.append({**common, 'id': f'{rid}-eval', 'script': 'scripts/eval_rl.py',
                     'args': ['--run-dir', f'runs/{rid}', '--train-job', rid, '--workers', '3', '--envs-per-worker',
                              '8', '--output-dir', f'runs/{rid}-eval'],
                     'mem_gb': 3, 'ram_gb': 8, 'oom_score_adj': 400, 'disk_gb': 3, 'required_results': ['eval.jsonl']})


# E1: crop on the six remaining tasks; missing full-method full-split evaluations (reused final encoders)
for task in SIX:
    pretrain('crop', task, CROP_SETS, CROP_DIFF)
    source = f's3-m3d-seed0-{task}-200k'
    assert state[source]['status'] == jobs.DONE and by_queue[source]['args'][-len(freeze['overrides']):] == \
        freeze['overrides']
    checkpoint = f'runs/{source}/checkpoints/step_0200000.pt'
    expected = state[source]['sync']['files']['checkpoints/step_0200000.pt']
    assert sha(ROOT / checkpoint) == expected, source
    reused[source] = {'checkpoint': checkpoint, 'sha256': expected, 'config': freeze['task_configs'][task],
                      'host': 'remote', 'role': 'full-method final encoder (E1 control)'}
    fulleval(f'fulleval-{source}', source, checkpoint, 'full')
for task in EIGHT:
    if task in DEV:
        export, dependency = f'runs/exports/s2ab-crop-seed0-{task}-200k.pt', f'export-s2ab-crop-seed0-{task}-200k'
        assert (ROOT / f'runs/setup/s2-{dependency}-completion.json').is_file()
    else:
        export, dependency = exports[('crop', task)]
    for seed in FINAL_SEEDS:
        rl_pair(f's2rl-crop-{task}-s{seed}', task, seed, export, dependency, 'crop')
# E2: shift on the development tasks, Stage 2 seeds
for task in DEV:
    pretrain('shift', task, SHIFT_SETS, SHIFT_DIFF)
for seed in freeze['ablation_rl_seeds']:
    for task in DEV:
        export, dependency = exports[('shift', task)]
        rl_pair(f's2rl-shift-{task}-s{seed}', task, seed, export, dependency, 'shift')

# E3: CNN without RL random shift, matched to the completed final CNN runs (definitions copied, one override added)
def final_run(method, task, seed):  # rl_reporting.py's final-comparison run names (that module runs on import)
    if method == 'cnn' and (task, seed) == ('bin-picking', 2000):
        return 's4-drm-cnn-bin-picking-s2000-r4'
    return f's4-drm-{method}-{task}-s{seed}'


cnn_controls = {}
for task in EIGHT:
    for seed in FINAL_SEEDS:
        control = final_run('cnn', task, seed)
        trainer, evaluator = by_queue[control], by_queue[f'{control}-eval']
        assert state[control]['status'] == jobs.DONE and state[f'{control}-eval']['status'] == jobs.DONE
        assert '--set' not in trainer['args'] and trainer['args'][:6] == ['--task', task, '--encoder', 'cnn', '--seed',
                                                                          str(seed)]
        rid = f's4-drm-cnnnoshift-{task}-s{seed}'
        cnn_controls[rid] = control
        # resources/sync fields (not scientific) from a remote control; the scientific arguments mirror this control
        template = trainer if trainer.get('host') == 'remote' else by_queue['s4-drm-cnn-hammer-s2000']
        etemplate = evaluator if trainer.get('host') == 'remote' else by_queue['s4-drm-cnn-hammer-s2000-eval']
        fields = {k: v for k, v in template.items() if k not in ('id', 'args')}
        new_jobs.append({**fields, 'id': rid, 'args': ['--task', task, '--encoder', 'cnn', '--seed', str(seed),
                                                       '--run-dir', f'runs/{rid}', '--set', 'agent.random_shift=false'],
                         'host': 'remote', 'deps': [], 'cnn_ablation': f'noshift-{TAG}', 'requires_feature': 'aug-shift'})
        eargs = list(evaluator['args'])
        eargs[eargs.index('--run-dir') + 1] = f'runs/{rid}'
        eargs[eargs.index('--train-job') + 1] = rid
        if '--output-dir' in eargs:
            eargs[eargs.index('--output-dir') + 1] = f'runs/{rid}-eval'
        else:  # a local control wrote eval.jsonl into its run directory; the remote companion needs its own output
            eargs += ['--output-dir', f'runs/{rid}-eval']
        efields = {k: v for k, v in etemplate.items() if k not in ('id', 'args', 'companion_of')}
        if 'companion_of' in etemplate:
            efields['companion_of'] = rid
        new_jobs.append({**efields, 'id': f'{rid}-eval', 'args': eargs, 'host': 'remote', 'deps': [],
                         'cnn_ablation': f'noshift-{TAG}'})

ANALYSIS = {
    'primary_rule_unchanged': 'Stage 2 RL rule per task and camera condition (training / held-out / trajectory): '
                              'seed-mean last-5 success, margin max(0.10, 2 x SE); historical margin verdicts are kept. '
                              'A difference within the margin is never read as evidence of no effect.',
    'effect_sizes': 'per task and condition: difference of seed means (variant - control) for success and return, '
                    'last-5 / final / AUC (mean over the 100 evaluation points), with 95% percentile-bootstrap CIs '
                    '(10,000 resamples, fixed RNG seed 0) that resample matched seed IDs jointly across the two arms '
                    '(paired by RL seed; the unpaired per-arm bootstrap reported alongside); standardised difference '
                    '(Hedges g over seeds) as a secondary statistic, NA when the pooled SD is 0 (e.g. both arms at the '
                    'success ceiling); raw differences always reported.',
    'learning_efficiency': 'explicit for every comparison (hammer is near the success ceiling): AUC over 1M, early AUC '
                           '(mean success over the first 30 points, 10k-300k), and time to reach 0.5 and 0.8 '
                           'training-camera success as the restricted mean through 1M (first evaluation point reaching '
                           'it; a run that never reaches it counts as 1M) together with the reach rate (runs reaching / '
                           'runs); differences with the same paired bootstrap CIs.',
    'aggregates': 'E1: per-task differences averaged over the eight tasks (and the six non-development tasks), with a '
                  'task-stratified bootstrap CI (seeds resampled within task and arm); IQM of per-run scores per arm '
                  'with CI as in the final comparison. E3: the same over eight tasks. E2: dev tasks only, per task.',
    'representation': 'E1: the 32 Stage 2 representation metrics of each crop encoder against the full-method '
                      'encoder of the same task (registered dev-task margins applied as a descriptive yardstick; '
                      'raw deltas reported). E2: shift vs full and shift vs crop on the same metrics and margins.',
    'cnn_viewpoint': 'E3: per task and pooled, report the change in training-camera success alongside the changes '
                     'in held-out and trajectory success, the novel/training success ratio per run (NA when training '
                     'success is 0; such runs are counted and listed), and the difference-in-differences (delta novel - '
                     'delta training, raw success units) with the paired bootstrap CI; a general learning '
                     'degradation shows as a training-camera drop, a loss of viewpoint robustness as a novel-view drop '
                     'beyond the training-camera change.',
    'matched_controls': 'E1 crop vs full on seeds 2000-2002 for all eight tasks (hammer/pick-place crop at 2000-2002 '
                        'registered here; their dev-seed 1000-1002 crop results stay a separate, already-reported '
                        'comparison). E2 shift vs full and vs crop on seeds 1000-1002 only. E3 noshift vs CNN on '
                        '2000-2002. Evaluator settings match within each comparison (E1/E2: 3 x 8; E3: as each CNN control).',
    'reuse': {'E1_full_rl': 's3-drm-splatter4d-<task>-s200{0,1,2} (final, remote, 226d8ae; eval 3 x 8)',
              'E2_controls': 's2rl-full-<task>-s100x (local, 6a48200) and s2rl-crop-<task>-s100x (remote, 226d8ae)',
              'E3_controls': cnn_controls},
    'decisions': 'attribution only; the frozen method, its hyperparameters and seeds do not change; non-development '
                 'task outcomes are never used for method decisions.',
}
counts = {}
for j in new_jobs:
    key = f"{j.get('host', 'local')}/{j['script'].split('/')[-1]}"
    counts[key] = counts.get(key, 0) + 1
assert len({j['id'] for j in new_jobs}) == len(new_jobs)
assert not {j['id'] for j in new_jobs} & set(by_queue)
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'request': 'user 2026-10-17: three follow-up experiments',
          'design': __doc__.split('\n\n', 1)[1].rsplit('\n\n', 1)[0], 'method_tag': 'method-frozen-v1',
          'freeze_manifest_sha256': freeze_sha, 'reporting_margins_sha256': MARGINS_SHA,
          'pretraining_steps': 200000, 'pretraining_seed': 0, 'rl_agent_steps': 1000000,
          'seeds': {'E1': list(FINAL_SEEDS), 'E2': freeze['ablation_rl_seeds'], 'E3': list(FINAL_SEEDS)},
          'variants': {'crop': {'overrides': CROP_SETS, 'expected_difference': CROP_DIFF},
                       'shift': {'overrides': SHIFT_SETS, 'expected_difference': SHIFT_DIFF},
                       'cnnnoshift': {'overrides': ['agent.random_shift=false']}},
          'reused_pretraining': reused, 'analysis_plan': ANALYSIS, 'counts': counts, 'configs': configs,
          'jobs': new_jobs, 'source_requirement': 'aug.shift and agent.random_shift need the accepted aug-shift source '
                                                  'merge and a remote release accepted with it before launch'}
proof = ROOT / f'runs/setup/followup-registration-{TAG}.json'
if not PUBLISH:
    print(json.dumps({'counts': counts, 'ids': [j['id'] for j in new_jobs][::9]}, indent=1))
    raise SystemExit(0)
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not proof.exists()
    before = jobs.QUEUE.read_bytes()
    text = before.decode()
    node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    staged = ROOT / f'runs/setup/followup-queue-candidate-{TAG}.yaml'
    staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                      + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n' + text[node.end_mark.index:])
    assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
    (ROOT / f'runs/setup/followup-queue-before-{TAG}.yaml').write_bytes(before)
    tmp = jobs.QUEUE.with_suffix('.fu.tmp')
    tmp.write_bytes(staged.read_bytes())
    os.replace(tmp, jobs.QUEUE)
    record.update(registered_at=datetime.now().astimezone().isoformat(),
                  queue_sha256_before=hashlib.sha256(before).hexdigest(),
                  queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
    proof.write_text(json.dumps(record, indent=2) + '\n')
    print('REGISTERED', len(new_jobs), proof)
