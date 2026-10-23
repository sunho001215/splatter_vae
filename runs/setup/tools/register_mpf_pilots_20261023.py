"""Masking and per-frame pilots on hammer and pick-place (user instructions 2026-10-23; registered before any result of
these runs). Definitions: runs/setup/mpf-definition-20261023.json (written here). Source: branch masking-perframe-20261023
(s4d/model/encoder.py mask_mode, s4d/train/perframe.py, s4d/rl/encoders.py frame_features), merged before publication.

Rows (each task; pretraining seed 0, 200k steps, the established protocol):
- chlrm-seed0-<task>-200k: SinCro-style masking ablation = base + task + chamfer_lean.yaml + chamfer_lean_randmask.yaml;
  its export and DrM seeds 2000-2002 (1M agent steps) with evaluators, mirroring the chamfer_lean control rows
  (chl-seed0-<task>-200k, chl-drm-<task>-s200x, completed) apart from names, the variant file and the export;
- pfs-seed0-<task>-200k: per-frame spatial reconstruction (chamfer_lean.yaml + perframe_spatial.yaml), export, DrM with the
  stacked policy input (vision.frame_features=stack);
- pft-seed0-<task>-200k: the same plus the pair-conditioned temporal branch, cross-time reconstruction and M3D
  (perframe_temporal.yaml; 75 % source / 75 % target masking, the user's revision before launch), export, stacked DrM;
- t1stack-drm-<task>-s1000..1002: the existing Stage 2 T=1 exports (runs/exports/s2ab-t1-seed0-<task>-200k.pt; no new
  pretraining) with chronological feature stacking [E(I_t-4), E(I_t-2), E(I_t)], mirroring the T=1 latest-frame rows
  s2rl-t1-<task>-s100x (seeds 1000-1002, completed at 6a48200) apart from names and vision.frame_features=stack.
Scheduling only (not protocol): stacked-input DrM rows declare ram_gb 7 (768-d replay features instead of 256-d);
pretraining and RL priority 2 (after the registered chamfer_lean rows), exports 0. Hosts: registered local; the
host-neutral pool placement may move unstarted rows to the remote (scientific fields unchanged, host recorded per run).

    python -I register_mpf_pilots_20261023.py [--publish]
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
from scripts import jobs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
TASKS = ('hammer', 'pick-place')
SEEDS = (2000, 2001, 2002)
T1_SEEDS = (1000, 1001, 1002)
VARIANTS = {  # prefix -> (variant name, variant file, stacked policy input)
    'chlrm': ('chamfer_lean_randmask', 'configs/metaworld/variants/chamfer_lean_randmask.yaml', False),
    'pfs': ('perframe_spatial', 'configs/metaworld/variants/perframe_spatial.yaml', True),
    'pft': ('perframe_temporal', 'configs/metaworld/variants/perframe_temporal.yaml', True),
}
DEFINITION = 'runs/setup/mpf-definition-20261023.json'
queue = jobs.load_queue()
by_queue = {j['id']: j for j in queue['jobs']}
state = jobs.read_registry()
for t in TASKS:
    for ref in (f'chl-seed0-{t}-200k', f'export-s2ab-t1-seed0-{t}-200k'):
        assert state[ref]['status'] == jobs.DONE, ref
    for s in SEEDS:
        assert state[f'chl-drm-{t}-s{s}-eval']['status'] == jobs.DONE
    for s in T1_SEEDS:
        assert state[f's2rl-t1-{t}-s{s}-eval']['status'] == jobs.DONE
for _, path, _ in VARIANTS.values():
    assert (ROOT / path).is_file(), path


def strip(row):
    return {k: v for k, v in row.items() if k not in ('variant', 'variant_definition', 'continues')}


new_jobs = []
for prefix, (variant, path, stack) in VARIANTS.items():
    meta = {'stage': 5, 'variant': variant, 'variant_definition': DEFINITION}
    for task in TASKS:
        pre_ref, exp_ref = by_queue[f'chl-seed0-{task}-200k'], by_queue[f'chl-export-seed0-{task}-200k']
        pid, eid = f'{prefix}-seed0-{task}-200k', f'{prefix}-export-seed0-{task}-200k'
        pre = copy.deepcopy(strip(pre_ref))
        a = pre['args']
        a.insert(a.index('configs/metaworld/variants/chamfer_lean.yaml') + 1, path)  # applied on top of chamfer_lean
        a[a.index('--name') + 1] = pid
        pre.update(id=pid, **meta)
        new_jobs.append(pre)
        exp = copy.deepcopy(strip(exp_ref))
        exp['args'] = [x.replace(f'chl-seed0-{task}-200k', pid).replace(f'chl-export-seed0-{task}-200k', eid) for x in exp['args']]
        exp.update(id=eid, deps=[pid], **meta)
        new_jobs.append(exp)
        for seed in SEEDS:
            for suffix in ('', '-eval'):
                ref_row = by_queue[f'chl-drm-{task}-s{seed}{suffix}']
                rid = f'{prefix}-drm-{task}-s{seed}'
                row = copy.deepcopy(strip(ref_row))
                row['args'] = [x.replace(f'chl-drm-{task}-s{seed}', rid).replace(f'chl-export-seed0-{task}-200k', eid)
                               for x in row['args']]
                if stack and not suffix:
                    row['args'].append('vision.frame_features=stack')
                if stack:
                    row['ram_gb'] = 7 if not suffix else row['ram_gb']
                row.update(id=rid + suffix, deps=[eid], **meta)
                new_jobs.append(row)
for task in TASKS:
    meta = {'stage': 5, 'variant': 't1_stack', 'variant_definition': DEFINITION}
    for seed in T1_SEEDS:
        for suffix in ('', '-eval'):
            ref_row = by_queue[f's2rl-t1-{task}-s{seed}{suffix}']
            rid = f't1stack-drm-{task}-s{seed}'
            row = {k: copy.deepcopy(v) for k, v in ref_row.items()
                   if k not in ('stage2_variant', 'stage', 'frozen_method', 'freeze_manifest_sha256', 'stage2_margins_sha256')}
            row['args'] = [x.replace(f's2rl-t1-{task}-s{seed}', rid) for x in row['args']]
            if not suffix:
                row['args'].append('vision.frame_features=stack')
                row['ram_gb'] = 7
            row.update(id=rid + suffix, host='local', priority=2, **meta)
            new_jobs.append(row)

# protocol checks: every row equals its reference apart from names, the variant file, the export, the stacked input,
# the declared RAM of stacked-input trainers and the bookkeeping fields
for j in new_jobs:
    if j['id'].startswith('t1stack-'):
        ref = by_queue[j['id'].replace('t1stack-drm-', 's2rl-t1-')]
        back = [x for x in j['args'] if x != 'vision.frame_features=stack']
        back = [x.replace('t1stack-drm-', 's2rl-t1-') for x in back]
        assert back == ref['args'], j['id']
        continue
    prefix = j['id'].split('-')[0]
    ref = by_queue[j['id'].replace(f'{prefix}-', 'chl-', 1)]
    back = [x for x in j['args'] if x not in (VARIANTS[prefix][1], 'vision.frame_features=stack')]
    back = [x.replace(f'{prefix}-', 'chl-') for x in back]
    assert back == ref['args'], (j['id'], back, ref['args'])
    diff = {k for k in set(j) | set(ref) if j.get(k) != ref.get(k)} - {'id', 'args', 'variant', 'variant_definition',
                                                                        'continues', 'deps'}
    assert diff <= {'ram_gb'}, (j['id'], diff)
assert not set(by_queue) & {j['id'] for j in new_jobs}, 'already registered'
counts = {}
for j in new_jobs:
    counts[j['variant']] = counts.get(j['variant'], 0) + 1
definition = {
    'recorded': datetime.now().astimezone().isoformat(),
    'instructions': '2026-10-23: masking ablation, per-frame STP-inspired pilots and T=1 stacking on hammer and pick-place; '
                    'later revision (same day, before launch): temporal pilot 75 % source / 75 % target masking reusing '
                    "each image's masked encoding; the superseded 75 % / 90 % definition (never registered or run) is "
                    'commit 5fd623e of branch masking-perframe-20261023.',
    'source_branch': 'masking-perframe-20261023',
    'variants': {
        'chamfer_lean_randmask': 'chamfer_lean with 75 % uniform random masking over the 3 x 64 concatenated temporal tokens '
                                 '(48 of 192 kept, no per-frame quota, no shared spatial mask, no motion priority; '
                                 'synthetic inputs alike among covered tokens). Everything else identical to chamfer_lean. '
                                 'Tests the combined masking choice, not its parts.',
        'perframe_spatial': 'single-image encoder, every image masked 75 % uniform random (16 of 64, independent per image '
                            'and timestep); static per-frame decoding from the source camera; chamfer_lean reconstruction '
                            'terms per frame; per-frame invariance; stacked policy input.',
        'perframe_temporal': 'perframe_spatial + pair-conditioned temporal branch (pairs (0,1), (1,2), (0,2); conditioned '
                             "on the source camera's 75 %-masked encodings of both frames, the same outputs the spatial "
                             'reconstructions use), cross-time reconstruction and M3D on per-pair source sets (identities '
                             'preserved); no action, timestep or horizon input. Labelled pair-conditioned temporal '
                             'reconstruction, not causal action-conditioned future prediction.',
        't1_stack': 'the Stage 2 T=1 exports, policy input = chronological stack of independently encoded frames; no new '
                    'pretraining; compared with the T=1 latest-frame rows (seeds 1000-1002), which stay reported separately.',
    },
    'comparisons': {
        'masking': 'chamfer_lean_randmask vs chamfer_lean (chl-drm-<task>-s2000..2002, completed): same encoder, decoder, '
                   'losses, sampling, augmentation, RL adapter; seeds 2000-2002.',
        'temporal_supervision': 'perframe_temporal vs perframe_spatial: identical encoder, masking, spatial objectives, image '
                                'sampling/exposure, three-frame stacking, policy capacity and evaluation; seeds 2000-2002.',
        'stacking': 't1_stack vs T=1 latest frame (s2rl-t1-<task>-s1000..1002), full method (s2rl-full) as context; '
                    'the stacked policy head has a 768-d input (263,680 head parameters vs 132,608), same 256-d output.',
        'context_only': 'per-frame arms vs joint three-frame encoders (chamfer_lean, original) differ in the policy-input '
                        'structure; differences there are not attributed to temporal supervision alone.',
    },
    'reporting': 'descriptive pilots (no adoption rule): DrM success and episodic return, last-5 / final / AUC, training, '
                 'held-out and moving-camera groups; per task seed means; pooled IQM over the 6 runs with the '
                 'task-stratified bootstrap 95 % interval (B = 2000) of each direct difference; targeted rendering and '
                 'representation diagnostics (runs/setup/tools/variant_repr_probes.py and the 200k render analysis). '
                 'Reported before any new eight-task expansion; chamfer_lean stays a candidate, not adopted.',
    'compute_note': 'perframe_temporal over perframe_spatial: the branch, three renders of the moved sets in the training '
                    'cameras and M3D; no extra encoding (measured step times reported).',
}
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0], 'tasks': list(TASKS),
          'counts': counts, 'definition': DEFINITION, 'jobs': new_jobs}
proof = ROOT / 'runs/setup/mpf-registration-20261023.json'
if not args.publish:
    print(json.dumps({'counts': counts, 'n': len(new_jobs)}, indent=1))
    for j in new_jobs[:3] + new_jobs[-2:]:
        print(json.dumps(j))
    raise SystemExit(0)
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not proof.exists()
    (ROOT / DEFINITION).write_text(json.dumps(definition, indent=2) + '\n')
    before = jobs.QUEUE.read_bytes()
    text = before.decode()
    node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    staged = ROOT / 'runs/setup/mpf-queue-candidate-20261023.yaml'
    staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                      + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n' + text[node.end_mark.index:])
    assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
    tmp = jobs.QUEUE.with_suffix('.mpf.tmp')
    tmp.write_bytes(staged.read_bytes())
    os.replace(tmp, jobs.QUEUE)
    record.update(registered_at=datetime.now().astimezone().isoformat(),
                  definition_sha256=hashlib.sha256((ROOT / DEFINITION).read_bytes()).hexdigest(),
                  queue_sha256_before=hashlib.sha256(before).hexdigest(),
                  queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
    proof.write_text(json.dumps(record, indent=2) + '\n')
    print('REGISTERED', len(new_jobs), proof)
