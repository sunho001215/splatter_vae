"""Variant chamfer_lean (definition: runs/setup/chl-definition-20261021.json, written here): pilot pretrains on hammer
and pick-place, seed 0, 200k LR schedule stopped at 50k, registered before any result. Mirrors the reference
i2-m3d-seed0-<task> rows (base + task + the four reference overrides, now in the variant file) with the variant config.

    python -I register_chl_20261021.py [--publish]
"""
import argparse
import fcntl
import os

import yaml

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
variant = ROOT / 'configs/metaworld/variants/chamfer_lean.yaml'
check = json.loads((ROOT / 'runs/setup/chamfer-weight-check-20261021.json').read_text())
now = datetime.datetime.now().astimezone().isoformat()
definition = {
    'prepared_at': now,
    'name': 'chamfer_lean',
    'description': 'Pretraining candidate: the completed reference C + D0 + M3D (i2-m3d) with the hard-opacity depth '
                   'pass, depth-gradient loss and visibility regulariser removed (occlusion/free-space already 0) and '
                   'a bidirectional 3D Chamfer loss between all decoded Gaussian centres and the training cameras\' '
                   'fused depth cloud added at each reconstructed timestep. Not "D2" (historical D2 also had occlusion).',
    'commit': commit,
    'config': {'files': ['configs/metaworld/base.yaml', 'configs/metaworld/tasks/<task>.yaml',
                         'configs/metaworld/variants/chamfer_lean.yaml'],
               'variant_sha256': hashlib.sha256(variant.read_bytes()).hexdigest(),
               'variant_text': variant.read_text()},
    'reference': {'runs': 'runs/pretrain/i2-m3d-seed{0,1}-<task>', 'overrides': [
        'aug.synth_views=2', 'aug.synth_render=true', 'loss.depth_valid_far=true', 'loss.motion_space=gaussian']},
    'loss': {
        'formula': 'L = 0.5 * mean_g min_p ||g - p||^2 + 0.5 * mean_p min_g ||p - g||^2 (metres^2), per sample and '
                   'reconstructed timestep, averaged over valid samples; added to the render loss with weight 5',
        'gaussians': 'all decoded Gaussian centres (static and dynamic) at that timestep; no opacity or contribution '
                     'filter; no alignment',
        'targets': 'training cameras of the batch only (no held-out or moving-camera data): every pixel lifted with its '
                   'stored depth and calibration; valid if depth > 0 and < far (3.0 m, loss.depth_valid_far); 1 cm '
                   'voxel means (one point per occupied voxel, across views); 4096 points per sample and timestep '
                   'sampled uniformly over voxels (without replacement when enough voxels exist)',
        'nearest_neighbours': 'exact fp32, fused Triton kernel (bounded memory, no distance matrix), argmin without '
                              'gradient, squared distance recomputed with gradient at the argmin',
        'weight': {'value': 5, 'source': 'runs/setup/chamfer-weight-check-20261021.json', 'rule': check['rule'],
                   'median_ratio': check['median_ratio'], 'ratio_range': check['ratio_range'],
                   'note': 'rule v1 (including the visibility term) gave ~108 because of the visibility '
                           'initialisation transient; v2 matches the gradient scale of the removed depth terms. '
                           'Fixed before any candidate training; no sweep.'},
    },
    'computation': 'disabled terms are skipped (no hard-opacity render, no depth-gradient or visibility computation); '
                   'the diagnostic motion-feature render runs only on logging steps (loss.motion_feature_render: '
                   'log_steps), with identical logged metrics',
    'unchanged': ['RGB L1 + SSIM', 'M3D Gaussian-space motion', 'expected-depth L1', 'alpha coverage', 'InfoNCE',
                  'slot consistency', 'synthetic views (2, render targets)', 'encoder', 'decoder', 'Gaussian counts',
                  'anchors', 'temporal sampling', 'splits', 'rendering settings', 'reconstruction timing'],
    'pilot': {'tasks': ['hammer', 'pick-place'], 'seed': 0, 'schedule': 'train.steps=200000 (LR schedule), '
              'train.stop_step=50000', 'checkpoints': 'every 10k (resumable)', 'comparison': 'i2-m3d-seed0 in-training '
              'evaluation at 50k (same seed); i2-m3d-seed1 50k checkpoints (full evaluation and diagnostics); '
              'i2-m3d-seed0 100k for context', 'not_yet': ['200k', 'RL', 'eight-task expansion']},
}
jobs_new = []
for task in ('hammer', 'pick-place'):
    jid = f'chl-seed0-{task}-50k'
    jobs_new.append({
        'id': jid, 'host': 'local', 'script': 'scripts/train.py',
        'args': ['--config', 'configs/metaworld/base.yaml', f'configs/metaworld/tasks/{task}.yaml',
                 'configs/metaworld/variants/chamfer_lean.yaml', '--name', jid, '--output-root', 'runs/pretrain',
                 '--resume', 'auto', '--set', 'train.steps=200000', 'train.stop_step=50000', 'train.seed=0'],
        'gpu': 'any', 'mem_gb': 10, 'ram_gb': 13, 'disk_gb': 10, 'priority': 1, 'deps': [], 'max_restarts': 3,
        'stage': 5, 'variant': 'chamfer_lean', 'variant_definition': 'runs/setup/chl-definition-20261021.json',
        'required_results': ['completion.json', 'metrics.jsonl', 'checkpoints/step_0050000.pt']})
registration = {
    'prepared_at': now, 'commit': commit,
    'purpose': 'chamfer_lean pilot pretrains (definition runs/setup/chl-definition-20261021.json), registered before '
               'any result: hammer and pick-place, seed 0, 200k LR schedule stopped at 50k. No 200k, RL or eight-task '
               'expansion yet.',
    'tasks': ['hammer', 'pick-place'], 'counts': {'train.py': 2}, 'jobs': jobs_new}
queue = jobs.load_queue()
by_queue = {j['id']: j for j in queue['jobs']}
for task in ('hammer', 'pick-place'):  # the protocol equals the reference rows apart from the variant file and stop step
    ref = by_queue[f'i2-m3d-seed0-{task}']
    assert ref['args'][-7:] == ['train.steps=200000', 'train.stop_step=100000', 'train.seed=0', 'aug.synth_views=2',
                                'aug.synth_render=true', 'loss.depth_valid_far=true', 'loss.motion_space=gaussian']
variant_cfg = yaml.safe_load(variant.read_text())
assert variant_cfg['aug'] == {'synth_views': 2, 'synth_render': True}
assert {k: variant_cfg['loss'][k] for k in ('depth_valid_far', 'motion_space')} == {'depth_valid_far': True,
                                                                                   'motion_space': 'gaussian'}
assert set(variant_cfg) == {'aug', 'loss'}
assert not set(by_queue) & {j['id'] for j in jobs_new}, 'already registered'
out = ROOT / 'runs/setup/chl-definition-20261021.json'
reg = ROOT / 'runs/setup/chl-registration-hammer-pick-place-20261021.json'
if not args.publish:
    print(json.dumps({'commit': commit, 'ids': [j['id'] for j in jobs_new]}, indent=1))
    raise SystemExit(0)
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not reg.exists() and not out.exists()
    out.write_text(json.dumps(definition, indent=1) + '\n')
    before = jobs.QUEUE.read_bytes()
    text = before.decode()
    node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    staged = ROOT / 'runs/setup/chl-queue-candidate-20261021.yaml'
    staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                      + yaml.safe_dump(json.loads(json.dumps(jobs_new)), sort_keys=False) + '\n' + text[node.end_mark.index:])
    assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + jobs_new
    tmp = jobs.QUEUE.with_suffix('.chl.tmp')
    tmp.write_bytes(staged.read_bytes())
    os.replace(tmp, jobs.QUEUE)
    registration.update(registered_at=datetime.datetime.now().astimezone().isoformat(),
                        queue_sha256_before=hashlib.sha256(before).hexdigest(),
                        queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
    reg.write_text(json.dumps(registration, indent=1) + '\n')
    print('REGISTERED', len(jobs_new), reg, out)
