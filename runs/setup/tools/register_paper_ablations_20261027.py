"""Five matched ablations of the selected paper architecture on one environment (user directive 2026-10-27; definition
runs/setup/paper-definition-20261027.json "ablations"). Run only after the architecture and the environment set are
frozen (paper/selection.json) and the ablation environment is recorded in docs/EXPERIMENT_LOG.md; registered before any
ablation result.

Reference: the selected architecture's existing rows on the ablation task (pretraining seed 0, 200k; export; DrM seeds
2000-2002, 1M agent steps, proprioception; evaluators) - reused, never re-registered. Arms change exactly one component:
- noinfonce: --set loss.invariance=0.0 (slot consistency, synthetic views, reconstruction and every other term kept);
- nomotion:  --set loss.motion=0.0 (explicit M3D weight only; three-frame encoder, displacement heads and temporal
             rendering kept);
- t1spatial: configs/metaworld/variants/perframe_spatial.yaml appended (one shared single-image encoder, every frame
             reconstructed independently, no temporal reconstruction or motion supervision); RL with
             vision.frame_features=stack ([E(I_t-4), E(I_t-2), E(I_t)], oldest first; policy feature 768) and ram_gb 7 as
             the registered per-frame rows;
- nodepth:   --set loss.depth_l1=0.0 (expected-depth L1 removed from real and synthetic views; Chamfer and coverage kept);
- nochamfer: --set loss.chamfer=0.0 (expected-depth L1, coverage and every other term kept).
Row names: enc-abl-<arm>-<arch>-seed0-<task>-200k, enc-abl-<arm>-<arch>-export-seed0-<task>-200k,
enc-abl-<arm>-<arch>-drm-<task>-s200x[-eval] (the enc- prefix keeps them on the encoder release, d281258, that has the
stem / body variants). A genuinely equivalent completed run (identical resolved configuration, seeds and protocol) is
reused instead: the per-frame spatial pilot pfs-* on hammer / pick-place is equivalent to t1spatial for arch chlrm.
Every new row equals its reference row apart from names, the single change, priority and bookkeeping (asserted), and the
resolved configuration of each arm differs from the reference only in the listed keys (asserted with s4d.config).

    python -I register_paper_ablations_20261027.py --arch A --task T [--publish]
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
from s4d.config import load_config  # noqa: E402
from scripts import jobs  # noqa: E402
import pool_placement as pool  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--arch', required=True, choices=('chlrm', 'cs', 'w4', 'csw4'))
ap.add_argument('--task', required=True)
ap.add_argument('--publish', action='store_true')
ap.add_argument('--dry-run-before-freeze', action='store_true', help='check the row construction only (never registers)')
args = ap.parse_args()
assert not (args.publish and args.dry_run_before_freeze)
ARCH, TASK = args.arch, args.task
SEEDS = (2000, 2001, 2002)
V = 'configs/metaworld/variants/'
PERFRAME = V + 'perframe_spatial.yaml'
ARMS = {'noinfonce': {'set': ['loss.invariance=0.0'], 'diff': {'loss.invariance': (1.0, 0.0)}},
        'nomotion': {'set': ['loss.motion=0.0'], 'diff': {'loss.motion': (5.0, 0.0)}},
        't1spatial': {'files': [PERFRAME], 'rl_set': ['vision.frame_features=stack'], 'ram_gb': 7,
                      'diff': {'model.encoder.num_frames': (3, 1), 'model.decoder.motion_head': (None, False),
                               'model.decoder.temporal_branch': (None, False), 'model.perframe.temporal': (None, False)}},
        'nodepth': {'set': ['loss.depth_l1=0.0'], 'diff': {'loss.depth_l1': (1.0, 0.0)}},
        'nochamfer': {'set': ['loss.chamfer=0.0'], 'diff': {'loss.chamfer': (5.0, 0.0)}}}
PRIORITY = 4
DEFINITION = 'runs/setup/paper-definition-20261027.json'
BOOKKEEPING = {'id', 'args', 'variant', 'variant_definition', 'continues', 'deps', 'priority', 'ram_gb'}
REF = {'chlrm': 'chlrm', 'cs': 'enc-cs', 'w4': 'enc-w4', 'csw4': 'enc-csw4'}[ARCH]

if not args.dry_run_before_freeze:
    selection = json.loads((ROOT / 'reports/results-artifact/paper/selection.json').read_text())
    assert selection.get('frozen') is True and selection['chosen_arch'] == ARCH, 'architecture not frozen to this arm'
    assert selection.get('ablation_task') == TASK, 'ablation environment not recorded in paper/selection.json'
queue = jobs.load_queue()
in_queue = {j['id']: j for j in queue['jobs']}
state = jobs.read_registry()
# reference rows as registered (before any pool placement rewrote host fields / output paths): the new rows are offered
# to both hosts by the drivers exactly like the 2x2 rows
by_id = {}
for name in ('mpf-registration-20261023.json', 'chlrm-registration-eight-task-20261024.json',
             'encp-registration-20261025.json', 'encx-registration-20261026.json'):
    for j in json.loads((ROOT / 'runs/setup' / name).read_text())['jobs']:
        by_id.setdefault(j['id'], j)
overlay = pool.effective_rows()
for i, r in by_id.items():  # the queue equals the registrations up to the recorded pool placement
    if i in in_queue:
        assert in_queue[i] == overlay.get(i, r), i
pre_ref, exp_ref = f'{REF}-seed0-{TASK}-200k', f'{REF}-export-seed0-{TASK}-200k'
drm_ref = [f'{REF}-drm-{TASK}-s{s}' for s in SEEDS]
for i in [pre_ref, exp_ref] + drm_ref + [d + '-eval' for d in drm_ref]:
    assert i in by_id and state.get(i, {}).get('status') == jobs.DONE, f'reference {i} not complete'


def flat(d, p=''):
    out = {}
    for k, v in d.items():
        out.update(flat(v, p + k + '.') if isinstance(v, dict) else {p + k: v})
    return out


def config_of(row):
    a = row['args']
    files = a[a.index('--config') + 1:a.index('--name')]
    sets = a[a.index('--set') + 1:] if '--set' in a else []
    return flat(load_config([ROOT / f for f in files], overrides=sets))


reference_cfg = config_of(by_id[pre_ref])


def strip(row):
    return {k: v for k, v in row.items() if k not in ('variant', 'variant_definition', 'continues', 'placement')}


def build(arm):
    spec = ARMS[arm]
    pid = f'enc-abl-{arm}-{ARCH}-seed0-{TASK}-200k'
    eid = f'enc-abl-{arm}-{ARCH}-export-seed0-{TASK}-200k'
    meta = {'stage': 5, 'variant': f'abl_{arm}_{ARCH}', 'variant_definition': DEFINITION}
    pre = copy.deepcopy(strip(by_id[pre_ref]))
    a = pre['args']
    if spec.get('files'):
        k = a.index('--name')
        a[k:k] = spec['files']
    a[a.index('--name') + 1] = pid
    a += spec.get('set', [])
    pre.update(id=pid, priority=PRIORITY, **meta)
    exp = copy.deepcopy(strip(by_id[exp_ref]))
    exp['args'] = [x.replace(pre_ref, pid).replace(exp_ref, eid) for x in exp['args']]
    exp.update(id=eid, deps=[pid], priority=PRIORITY, **meta)
    rows = [pre, exp]
    for ref in drm_ref:
        rid = ref.replace(f'{REF}-drm-', f'enc-abl-{arm}-{ARCH}-drm-', 1)
        for suffix in ('', '-eval'):
            row = copy.deepcopy(strip(by_id[ref + suffix]))
            row['args'] = [x.replace(ref, rid).replace(exp_ref, eid) for x in row['args']]
            if suffix == '' and spec.get('rl_set'):
                row['args'] += spec['rl_set']
                row['ram_gb'] = spec['ram_gb']
            row.update(id=rid + suffix, deps=[eid], priority=PRIORITY, **meta)
            rows.append(row)
    return rows


def equivalent_completed(arm):
    """An existing completed run with the identical resolved configuration and protocol, or None."""
    if arm != 't1spatial' or ARCH != 'chlrm' or TASK not in ('hammer', 'pick-place'):
        return None
    pfs = f'pfs-seed0-{TASK}-200k'
    if pfs not in by_id or state.get(pfs, {}).get('status') != jobs.DONE:  # pfs rows: mpf-registration-20261023
        return None
    built = build(arm)[0]
    if config_of(by_id[pfs]) != config_of(built):
        return None
    for s in SEEDS:
        for suffix in ('', '-eval'):
            if state.get(f'pfs-drm-{TASK}-s{s}{suffix}', {}).get('status') != jobs.DONE:
                return None
    return {'pretraining': pfs, 'rl': [f'pfs-drm-{TASK}-s{s}' for s in SEEDS]}


new_jobs, reused, checks = [], {}, {}
for arm, spec in ARMS.items():
    rows = build(arm)
    cfg = config_of(rows[0])
    diff = {k: (reference_cfg.get(k), cfg.get(k)) for k in set(reference_cfg) | set(cfg) if reference_cfg.get(k) != cfg.get(k)}
    assert diff == spec['diff'], (arm, diff)
    checks[arm] = {k: list(v) for k, v in diff.items()}
    for row in rows:  # equal to the reference row apart from names, the single change, priority and bookkeeping
        ref_id = row['id'].replace(f'enc-abl-{arm}-{ARCH}-', f'{REF}-', 1)
        ref = strip(by_id[ref_id])
        extra = [x for x in row['args'] if x not in ref['args'] and x.replace(f'enc-abl-{arm}-{ARCH}-', f'{REF}-') not in
                 ref['args']]
        assert set(extra) <= set(spec.get('set', []) + spec.get('files', []) + spec.get('rl_set', [])), (row['id'], extra)
        assert not {k for k in set(row) | set(ref) if row.get(k) != ref.get(k)} - BOOKKEEPING - {'stage'}, row['id']
    same = equivalent_completed(arm)
    if same:
        reused[arm] = same
    else:
        new_jobs += rows
ids = [j['id'] for j in new_jobs]
assert len(ids) == len(set(ids)) and not set(ids) & set(in_queue), 'duplicate or already registered'
assert not any((ROOT / 'runs' / i).exists() or (ROOT / 'runs/pretrain' / i).exists() for i in ids), 'run dir exists'
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0], 'arch': ARCH,
          'task': TASK, 'reference': {'pretraining': pre_ref, 'export': exp_ref, 'rl': drm_ref},
          'resolved_config_differences': checks, 'reused_equivalent': reused, 'definition': DEFINITION,
          'definition_sha256': hashlib.sha256((ROOT / DEFINITION).read_bytes()).hexdigest(),
          'counts': {s: sum(j['script'] == s for j in new_jobs) for s in sorted({j['script'] for j in new_jobs})},
          'jobs': new_jobs}
proof = ROOT / f'runs/setup/paper-ablation-registration-{ARCH}-{TASK}-20261027.json'
if not args.publish:
    print(json.dumps({k: record[k] for k in ('arch', 'task', 'reference', 'resolved_config_differences',
                                             'reused_equivalent', 'counts')}, indent=1))
    raise SystemExit(0)
with open(pool.ADMISSION_LOCK, 'w') as admission:
    fcntl.flock(admission, fcntl.LOCK_EX)
    with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not proof.exists()
        before = jobs.QUEUE.read_bytes()
        text = before.decode()
        node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
        staged = ROOT / 'runs/setup/paper-ablation-queue-candidate-20261027.yaml'
        staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                          + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n'
                          + text[node.end_mark.index:])
        assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
        assert jobs.load_queue(staged)['hosts'] == queue['hosts']
        tmp = jobs.QUEUE.with_suffix('.ablation.tmp')
        tmp.write_bytes(staged.read_bytes())
        os.replace(tmp, jobs.QUEUE)
        record.update(registered_at=datetime.now().astimezone().isoformat(),
                      queue_sha256_before=hashlib.sha256(before).hexdigest(),
                      queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
        proof.write_text(json.dumps(record, indent=2) + '\n')
        print('REGISTERED', len(new_jobs), proof)
