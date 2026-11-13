"""Episode manifests for the paired S/SF real-world experiment.

This module writes no raw data and never launches preprocessing. The manifest is the
single source of episode identities, directory-derived outcomes, and paired splits.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import yaml

TASKS = ('drawer-open-v2', 'pick-and-place', 'covering')
LABELS = ('success', 'failure')
EPISODE = re.compile(r'episode_[0-9]{6}\.h5\Z')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def raw_roots(root, task):
    if task not in TASKS:
        raise ValueError(f'unsupported task: {task}')
    root = Path(root)
    return ({label: root / task / label for label in LABELS} if task == 'drawer-open-v2' else
            {'success': root / task, 'failure': root / f'{task}-failure'})


def inspect_raw(path):
    """Read lightweight timing/calibration QA; full image/geometry QA remains the pipeline gate."""
    from s4d.rwlearn.actions import contract_for, read_raw_episode, step_table
    path = Path(path)
    if not EPISODE.fullmatch(path.name) or path.is_symlink() or 'discarded' in path.parts:
        raise ValueError(f'not a complete episode path: {path}')
    before = path.stat()
    raw = read_raw_episode(path)
    if raw.status != 'complete':
        raise ValueError(f'{path}: recorder has not completed')
    def monotonic(t, label):
        if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError(f'{path}: nonmonotonic/empty {label}')
    monotonic(raw.robot_t, 'robot')
    monotonic(raw.gripper_t, 'gripper')
    cameras = {}
    contract = contract_for(raw)
    with h5py.File(path, 'r') as f:
        if 'end_unix' not in f.attrs or len(raw.cameras) != 3:
            raise ValueError(f'{path}: missing completion signal or three cameras')
        for serial, cam in raw.cameras.items():
            monotonic(cam['t'], serial)
            g = f[f'cameras/cam_{serial}']
            if len(g['color']) != len(cam['t']) or len(cam['segment']) != len(cam['t']):
                raise ValueError(f'{path}: camera rows disagree')
            cal = json.loads(g.attrs['calibration_json'])
            k = np.asarray(cal['streams']['color']['intrinsics']['K'], float)
            if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
                raise ValueError(f'{path}: invalid intrinsics')
            tab = step_table(raw, cam['t'], cam['segment'], nominal_period_s=contract.nominal_period_s)
            if not tab.transition_ok.any():
                raise ValueError(f'{path}: no synchronized sent/engaged action transitions')
            cameras[serial] = dict(frames=len(cam['t']), valid_transitions=int(tab.transition_ok.sum()),
                                   median_hz=float(1e9 / np.median(np.diff(cam['t']))),
                                   intrinsics=k.tolist(), calibration_id=cal.get('calibration_id'),
                                   extrinsics=list(cal['extrinsics']))
        label_fields = [k for k in f.attrs if any(s in k.lower() for s in ('success', 'outcome', 'label'))]
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f'{path}: changed during QA')
    return dict(cameras=cameras, robot_samples=len(raw.robot_t), label_fields=label_fields,
                action_contract=contract.__class__.__name__)


def paired_manifests(task, roots, *, seed=0, validation_fraction=.1, expected=None,
                     require_verified=True, inspect=inspect_raw, success_split=None):
    """Split each label independently; adding failures cannot change success membership.

    ``success_split`` can preserve a validated legacy split, keyed by filenames.
    Missing files/markers or wrong counts fail closed, including while transfers arrive.
    """
    if task not in TASKS or set(roots) != set(LABELS):
        raise ValueError('task/label roots must be explicit')
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must lie in (0,1)')
    rows = []
    digests = set()
    for label in LABELS:
        root = Path(roots[label])
        if root.is_symlink() or 'discarded' in root.parts or not root.is_dir():
            raise ValueError(f'missing/unsafe {label} root: {root}')
        if require_verified and not Path(str(root) + '.VERIFIED').is_file():
            raise ValueError(f'missing transfer verification marker: {root}.VERIFIED')
        files = sorted(p for p in root.iterdir() if EPISODE.fullmatch(p.name))
        if not files or (expected and len(files) != expected[label]):
            raise ValueError(f'{label}: expected {expected and expected[label]} complete episodes, found {len(files)}')
        label_rows = []
        for p in files:
            if p.is_symlink() or not p.is_file():
                raise ValueError(f'unsafe recording: {p}')
            stat = p.stat()
            qa = inspect(p)
            digest = sha256(p)
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
                    p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns):
                raise ValueError(f'recording changed during hashing: {p}')
            if digest in digests:
                raise ValueError(f'duplicate recording content across episodes: {p}')
            digests.add(digest)
            # Label stays in identity even when success/failure counters overlap.
            label_rows.append(dict(id=f'{task}-{label}-{p.stem}', task=task, label=label,
                                   filename=p.name, path=str(p.resolve()), sha256=digest, qa=qa))
        if label == 'success' and success_split is not None:
            train, val = set(success_split['train']), set(success_split['validation'])
            if train & val or train | val != {p.name for p in files}:
                raise ValueError('legacy success split is not an exact disjoint partition')
        else:
            # Per-label PRNG: SF order/count does not influence S.
            order = np.random.default_rng(seed).permutation(len(files))
            nval = max(1, round(len(files) * validation_fraction)) if len(files) > 1 else 0
            val = {files[int(i)].name for i in order[:nval]}
        for r in label_rows:
            r['split'] = 'validation' if r['filename'] in val else 'train'
        rows.extend(label_rows)
    def manifest(variant):
        eps = [r for r in rows if variant == 'SF' or r['label'] == 'success']
        return dict(schema='dsreset-episodes/v1', task=task, variant=variant, seed=seed,
                    validation_fraction=validation_fraction, episodes=eps,
                    split={s: [r['id'] for r in eps if r['split'] == s] for s in ('train', 'validation')},
                    outcomes={r['id']: r['label'] == 'success' for r in eps})
    return {v: manifest(v) for v in ('S', 'SF')}


def prep_bundle(manifests, template, output_root, *, rig_by_episode=None, reuse=None):
    """Generate standard per-episode pipeline configs, preserving reused success outputs.

    ``reuse`` entries: {id: {raw_sha256, train_data}}. Reuse is explicit and bound to
    the raw identity; its existing QA/calibration gate must still pass downstream.
    ``rig_by_episode`` permits PNP's accepted per-session documents.
    """
    task = manifests['SF']['task']
    reuse, rig_by_episode = reuse or {}, rig_by_episode or {}
    rows = []
    for ep in manifests['SF']['episodes']:
        identity = ep['id']
        if identity in reuse:
            old = reuse[identity]
            if ep['label'] != 'success' or old['raw_sha256'] != ep['sha256']:
                raise ValueError(f'unverified reuse identity: {identity}')
            rows.append(dict(episode=identity, reuse=old['train_data'], raw_sha256=ep['sha256']))
            continue
        cfg = copy.deepcopy(template)
        cfg['output_root'] = str(Path(output_root) / task / ep['label'] / ep['filename'][:-3])
        cfg['adapter']['config']['episodes'] = {identity: ep['path']}
        cfg['stages']['train_data']['task'] = task
        rig = rig_by_episode.get(identity)
        if rig is not None:
            cfg['stages']['calibrate']['rig_calibration'] = str(rig)
        if not cfg['stages']['calibrate'].get('rig_calibration'):
            raise ValueError('explicit per-task/session rig document required (may be pending)')
        # Never inherit another task's rig silently.
        if task not in str(cfg['stages']['calibrate']['rig_calibration']):
            raise ValueError(f'{task}: rig path must identify this task/session')
        rows.append(dict(episode=identity, raw_sha256=ep['sha256'], config=cfg))
    return rows


def write_bundle(out, manifests, prep):
    """Exclusive output creation makes a manifest immutable; reruns use a new directory."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for variant, m in manifests.items():
        dest = out / variant
        dest.mkdir()
        for name, obj in [('manifest.json', m), ('split_seed0.json', m['split']), ('outcomes.json', m['outcomes'])]:
            (dest / name).write_text(json.dumps(obj, indent=2) + '\n')
    configs = out / 'prep'
    configs.mkdir()
    plan = []
    for row in prep:
        if 'config' in row:
            path = configs / f"{row['episode']}.yaml"
            path.write_text(yaml.safe_dump(row['config'], sort_keys=False))
            plan.append({k: v for k, v in row.items() if k != 'config'} | {'config': str(path.resolve()),
                         'argv': ['-m', 's4d.realworld.cli', 'run', '--config', str(path.resolve()),
                                  '--episode', row['episode']]})
        else:
            plan.append(row)
    (out / 'prep-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
