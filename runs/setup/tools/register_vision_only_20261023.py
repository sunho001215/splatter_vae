"""Vision-only DrM campaign (user instruction 2026-10-23; registered before any result of these runs).

Arms, eight tasks (hammer, pick-place, bin-picking, door-open, coffee-push, peg-insert-side, peg-unplug-side,
stick-push), RL seeds 2000-2002, 1M agent steps, each with its evaluator (48 trainers + 48 evaluators):
- vo-drm-splatter4d-<task>-s<seed>: the ORIGINAL frozen encoder = the exports used by the completed eight-task comparison
  (method-frozen-v1, C + D0 + M3D, 200k steps, pretraining seed 0; resolved below with paths, hashes and configs);
- vo-drm-cnn-<task>-s<seed>: the end-to-end DrM CNN with its standard random-shift augmentation.
Each row mirrors its matched WITH-proprio row (s3-drm-splatter4d-<task>-s<seed>, s4-drm-cnn-<task>-s<seed>; bin-picking
s2000: the -r4 row of the final comparison) with only `env.proprio_indices=[]` (no proprioception anywhere in the policy;
input widths follow) and `wandb.tags=[vision_only,no_proprio]` added, plus names / run directories and the bookkeeping
fields of this campaign. Policies train from scratch. Code: branch vision-only-20261023 (zero-width proprio replay, W&B
tags; tests/test_vision_only.py). Hosts: registered local; the host-neutral pool moves unstarted pairs to the remote.

    python -I register_vision_only_20261023.py [--publish]
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

import torch
import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
TASKS = ('hammer', 'pick-place', 'bin-picking', 'door-open', 'coffee-push', 'peg-insert-side', 'peg-unplug-side', 'stick-push')
SEEDS = (2000, 2001, 2002)
EXTRA = ['env.proprio_indices=[]', 'wandb.tags=[vision_only,no_proprio]']
DEFINITION = 'runs/setup/vision-only-definition-20261023.json'
BOOKKEEPING = {'stage', 'frozen_method', 'freeze_manifest_sha256', 'stage2_variant', 'stage2_margins_sha256', 'host',
               'placement', 'baseline', 'baseline_condition', 'baseline_expansion'}
queue = jobs.load_queue()
by_queue = {j['id']: j for j in queue['jobs']}
state = jobs.read_registry()
freeze = json.loads((ROOT / 'runs/setup/method-frozen-v1.json').read_text())
assert freeze['tag'] == 'method-frozen-v1' and freeze['pretraining_steps'] == 200000


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def reference(arm, task, seed):
    if arm == 'splatter4d':
        return f's3-drm-splatter4d-{task}-s{seed}'
    return f's4-drm-cnn-{task}-s{seed}' + ('-r4' if (task, seed) == ('bin-picking', 2000) else '')


encoders = {}
for task in TASKS:
    row = by_queue[reference('splatter4d', task, 2000)]
    export = next(a.split('=', 1)[1] for a in row['args'] if a.startswith('vision.export_path='))
    payload = torch.load(ROOT / export, map_location='cpu', weights_only=True)
    source = payload['source_checkpoint']
    run = Path(source).parents[1]
    assert payload['step'] == 200000 and payload['encoder_config']['num_frames'] == 3 and Path(source).name == 'step_0200000.pt'
    assert 'seed0' in payload['run_name'] and 'm3d' in payload['run_name']
    for seed in SEEDS:  # every matched with-proprio run used this export
        assert f'vision.export_path={export}' in by_queue[reference('splatter4d', task, seed)]['args']
    cfg = yaml.safe_load((ROOT / run / 'config.yaml').read_text())
    assert cfg['train']['seed'] == 0 and cfg['model']['encoder']['mask_ratio'] == 0.5
    assert cfg['loss'].get('chamfer', 0.0) == 0.0 and cfg['data'].get('objective', 'reconstruct') == 'reconstruct'
    encoders[task] = {'export': export, 'export_sha256': sha(export), 'run_name': payload['run_name'],
                      'source_checkpoint': source, 'source_checkpoint_sha256': sha(source),
                      'pretraining_config': str(run / 'config.yaml'), 'pretraining_config_sha256': sha(run / 'config.yaml'),
                      'encoder_config': payload['encoder_config'], 'frame_strides': payload['frame_strides']}
new_jobs = []
for arm in ('splatter4d', 'cnn'):
    for task in TASKS:
        for seed in SEEDS:
            ref_id = reference(arm, task, seed)
            assert state[ref_id]['status'] == jobs.DONE and state[f'{ref_id}-eval']['status'] == jobs.DONE, ref_id
            rid = f'vo-drm-{arm}-{task}-s{seed}'
            for suffix in ('', '-eval'):
                ref = by_queue[ref_id + suffix]
                row = {k: copy.deepcopy(v) for k, v in ref.items() if k not in BOOKKEEPING}
                row['args'] = [x.replace(ref_id, rid) for x in ref['args']]
                if 'companion_of' in row:  # bookkeeping pointer of the reference evaluator: point at this trainer
                    row['companion_of'] = rid
                if not suffix:
                    if '--set' not in row['args']:
                        row['args'].append('--set')
                    row['args'] += EXTRA
                    row.setdefault('required_results', ['final.json', 'train.jsonl'])
                else:
                    if '--output-dir' not in row['args']:
                        row['args'] += ['--output-dir', f'runs/{rid}-eval']
                    row.setdefault('required_results', ['eval.jsonl'])
                row.update(id=rid + suffix, host='local', priority=2, deps=[], campaign='vision_only',
                           observation='vision_only (no proprioception)', matched_with_proprio=ref_id + suffix,
                           variant_definition=DEFINITION)
                if arm == 'splatter4d' and not suffix:
                    row['original_encoder'] = encoders[task]
                new_jobs.append(row)
# protocol: each row equals its matched with-proprio row apart from names, the two added overrides, an explicit
# evaluator output directory, declared results and bookkeeping
for j in new_jobs:
    ref = by_queue[j['matched_with_proprio']]
    back = [x for x in j['args'] if x not in EXTRA]
    if j['id'].endswith('-eval') and '--output-dir' not in ref['args']:
        back = back[:-2]
    rid = j['id'].removesuffix('-eval')
    back = [x.replace(rid, j['matched_with_proprio'].removesuffix('-eval')) for x in back]
    if not j['id'].endswith('-eval') and '--set' not in ref['args']:
        back = [x for x in back if x != '--set']
    assert back == ref['args'], (j['id'], back, ref['args'])
assert not set(by_queue) & {j['id'] for j in new_jobs}, 'already registered'
definition = {
    'recorded': datetime.now().astimezone().isoformat(),
    'instruction': 'user 2026-10-23: vision-only DrM without proprioception, original frozen encoder vs end-to-end CNN, '
                   'eight tasks, seeds 2000-2002, 1M agent steps; preregistered protocol and metrics',
    'observation': 'RGB history only (3 frames, 2 simulator steps apart; 128x128; one training camera per episode as in '
                   'the standard protocol); env.proprio_indices=[] so actor, critics, target critic, value network and '
                   'dormant-ratio passes receive no hand position, gripper state or other simulator-state feature. '
                   'Simulator state remains internal for rewards, termination and success.',
    'unchanged': 'rewards, termination, episode length, camera sampling, evaluation schedule (every 10k agent steps, '
                 '20 episodes per training camera / held-out camera / trajectory), frozen encoder, CNN random shift, '
                 'DrM hyperparameters (only the trunks\' input width changes by -4)',
    'original_encoders': encoders,
    'matched_with_proprio': 's3-drm-splatter4d-<task>-s<seed> (ours) and s4-drm-cnn-<task>-s<seed> (CNN; bin-picking '
                            's2000: -r4), completed',
    'metrics': {
        'per_run': 'success and episode return on the training cameras (train0..5), held-out cameras (eval0..3) and moving '
                   'cameras (traj_lateral, traj_circular): learning curves vs agent steps; last-5 (mean of the final five '
                   'evaluation points), final, full-budget AUC (mean over the 100 points 10k..1M) and early AUC (mean over '
                   'the 20 points 10k..200k)',
        'primary': 'training-camera success, full-budget AUC (learning efficiency): IQM over the 24 runs per arm of ours '
                   'minus CNN, task-stratified bootstrap 95% interval (B = 2000, seeds resampled within task, arms '
                   'independently)',
        'early_learning': 'training-camera success, early AUC over the first 200k agent steps, same aggregate (preregistered '
                          'secondary)',
        'secondary': 'last-5 and final success; held-out and moving cameras separately; per-task seed means; returns per '
                     'task only (scales differ), no pooled return IQM',
        'proprio_effect': 'the same quantities for the matched with-proprio runs; change of our advantage: '
                          '(ours - CNN)_vision_only - (ours - CNN)_with_proprio on the IQM aggregate, bootstrap over seeds '
                          'within task for all four arms independently; also each arm\'s own vision-only minus '
                          'with-proprio difference',
        'efficiency': 'sample efficiency (per agent step, the quantities above) reported separately from wall-clock '
                      'efficiency (training wall time per run from final.json / registry, by host and GPU type); the '
                      'original pretraining cost (GPU-hours of the 200k pretraining runs) reported separately',
    },
    'decision_rule': 'descriptive campaign; no adoption decision; no other baselines or pretraining variants',
}
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'definition': DEFINITION, 'count': len(new_jobs), 'jobs': new_jobs}
proof = ROOT / 'runs/setup/vision-only-registration-20261023.json'
if not args.publish:
    print(json.dumps({'n': len(new_jobs), 'encoders': {t: (e['export'], e['export_sha256'][:12]) for t, e in encoders.items()}}, indent=1))
    for j in (new_jobs[0], new_jobs[1], new_jobs[48], new_jobs[49]):
        print(json.dumps({k: v for k, v in j.items() if k != 'original_encoder'}))
    raise SystemExit(0)
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not proof.exists()
    (ROOT / DEFINITION).write_text(json.dumps(definition, indent=2) + '\n')
    before = jobs.QUEUE.read_bytes()
    text = before.decode()
    node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    staged = ROOT / 'runs/setup/vision-only-queue-candidate-20261023.yaml'
    staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                      + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n' + text[node.end_mark.index:])
    assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
    tmp = jobs.QUEUE.with_suffix('.vo.tmp')
    tmp.write_bytes(staged.read_bytes())
    os.replace(tmp, jobs.QUEUE)
    record.update(registered_at=datetime.now().astimezone().isoformat(),
                  definition_sha256=hashlib.sha256((ROOT / DEFINITION).read_bytes()).hexdigest(),
                  queue_sha256_before=hashlib.sha256(before).hexdigest(),
                  queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
    proof.write_text(json.dumps(record, indent=2) + '\n')
    print('REGISTERED', len(new_jobs), proof)
