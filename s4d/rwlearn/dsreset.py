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
    try:
        raw = read_raw_episode(path)
    except (KeyError, TypeError) as exc:
        raise ValueError(f'{path}: missing/invalid required action/state schema: {exc}') from exc
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
            for clock in ('frame_timestamp_us','sensor_timestamp_us'):
                if clock in g:
                    monotonic(np.asarray(g[clock],dtype=np.float64),f'{serial}/{clock}')
                if 'stereo' in g and clock in g['stereo']:
                    monotonic(np.asarray(g['stereo'][clock],dtype=np.float64),f'{serial}/stereo/{clock}')
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
                     require_verified=True, inspect=inspect_raw, success_split=None, amendment=None, verified_success=None):
    """Split each label independently; adding failures cannot change success membership.

    ``success_split`` can preserve a validated legacy split, keyed by filenames.
    Missing files/markers or wrong counts fail closed, including while transfers arrive.
    """
    if task not in TASKS or set(roots) != set(LABELS):
        raise ValueError('task/label roots must be explicit')
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must lie in (0,1)')
    rows, excluded = [], []
    digests = set()
    if amendment is not None and (amendment.get('format') != 'rw-gate-amendment/1'
            or amendment.get('task') != task or not amendment.get('decided_by') or not amendment.get('decision')
            or amendment.get('base_target') != 100
            or amendment.get('target') != len(amendment.get('episodes',{}))):
        raise ValueError('need a documented task-specific success gate amendment')
    if verified_success and amendment:
        # Excluded recordings were intentionally never transferred for learning.
        # Their accepted amendment hashes prove identity, without admitting them.
        verified_success = {**{Path(k).name:v['raw_sha256'] for k,v in amendment['excluded'].items()},
                            **verified_success}
    for label in LABELS:
        root = Path(roots[label])
        if root.is_symlink() or 'discarded' in root.parts or not root.is_dir():
            raise ValueError(f'missing/unsafe {label} root: {root}')
        has_marker = Path(str(root)+'.VERIFIED').is_file()
        if require_verified and not has_marker and not (label=='success' and verified_success):
            raise ValueError(f'missing transfer verification marker: {root}.VERIFIED')
        files = sorted(p for p in root.iterdir() if EPISODE.fullmatch(p.name))
        if not files or (expected and len(files) != expected[label]):
            raise ValueError(f'{label}: expected {expected and expected[label]} complete episodes, found {len(files)}')
        if require_verified and not has_marker and set(verified_success) != {p.name for p in files}:
            raise ValueError('verified ingest success set differs from raw directory')
        label_rows = []
        omitted = {}
        if label == 'success' and amendment:
            included = {Path(k).name:v for k,v in amendment['episodes'].items()}
            omitted = {Path(k).name:v for k,v in amendment['excluded'].items()}
            if (len(included) != len(amendment['episodes']) or len(omitted) != len(amendment['excluded'])
                    or set(included)&set(omitted) or set(included)|set(omitted) != {p.name for p in files}
                    or len(files) != amendment['base_target']):
                raise ValueError('amendment must bind the exact success recording set')
        for p in files:
            if p.is_symlink() or not p.is_file():
                raise ValueError(f'unsafe recording: {p}')
            stat = p.stat()
            digest = sha256(p)
            if require_verified and not has_marker and verified_success.get(p.name) != digest:
                raise ValueError(f'verified ingest success sha256 differs: {p}')
            if label == 'success' and amendment:
                wanted = omitted[p.name]['raw_sha256'] if p.name in omitted else included[p.name]
                if wanted != digest:
                    raise ValueError(f'amendment raw sha256 differs: {p}')
            qa = {'status':'excluded','reason':omitted[p.name]['reason']} if p.name in omitted else inspect(p)
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
                    p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns):
                raise ValueError(f'recording changed during hashing: {p}')
            if digest in digests:
                raise ValueError(f'duplicate recording content across episodes: {p}')
            digests.add(digest)
            if p.name in omitted:
                excluded.append(dict(id=f'{task}-{label}-{p.stem}',task=task,label=label,filename=p.name,
                                     path=str(p.resolve()),sha256=digest,split='excluded',qa=qa))
                continue
            # Label stays in identity even when success/failure counters overlap.
            label_rows.append(dict(id=f'{task}-{label}-{p.stem}', task=task, label=label,
                                   filename=p.name, path=str(p.resolve()), sha256=digest, qa=qa))
        files = [p for p in files if p.name not in omitted]
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
                    validation_fraction=validation_fraction, episodes=eps, excluded=excluded,
                    amendment=copy.deepcopy(amendment),
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


def learning_inputs(manifest, ingest, store):
    """Bind an immutable variant to ordinary per-episode ingest/QA evidence.

    This is an explicitly scoped gate view, not an edit of the live ingest manifest.
    The exact raw hashes and selected label set replace the old 100-only count;
    every validation, QA family and calibration check is still gate.evaluate's.
    """
    from s4d.rwlearn import gate
    from s4d.rwlearn.identity import store_identities
    validate_manifest(manifest)
    task = manifest['task']
    rows = manifest['episodes']
    if task not in TASKS or manifest['variant'] not in ('S', 'SF') or not rows:
        raise ValueError('not a paired dataset manifest')
    if len({r['id'] for r in rows}) != len(rows) or len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('duplicate manifest episode/content')
    report = manifest_gate(manifest, ingest)
    mapping = {}
    identities = store_identities(store)
    for r in rows:
        matches = [ep for ep, rec in identities.items() if rec['task'] == task
                   and rec.get('raw_sha256') == r['sha256']]
        if len(matches) != 1:
            raise ValueError(f"{r['id']}: store needs exactly one matching raw sha256")
        mapping[r['id']] = matches[0]
    if set(mapping.values()) != set(store.episodes()):
        raise ValueError('store must contain exactly the variant manifest episodes')
    report = gate.bind_store_episodes(report, {task: store.episodes()}, identities)
    split = {s: [mapping[r['id']] for r in rows if r['split'] == s] for s in ('train', 'validation')}
    if not split['train'] or not split['validation']:
        raise ValueError('both train and validation episodes required')
    outcomes = {mapping[r['id']]: r['label'] == 'success' for r in rows}
    report['dataset_manifest_sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return {'gate_report.json': report, 'split_seed0.json': split, 'outcomes.json': outcomes,
            'manifest.json': manifest}


def validate_manifest(m):
    """Validate a saved manifest without trusting its cached split/outcome maps."""
    from s4d.rwlearn.identity import recording_label
    if m.get('schema') != 'dsreset-episodes/v1' or m.get('task') not in TASKS or m.get('variant') not in ('S','SF'):
        raise ValueError('not a dsreset episode manifest')
    rows = m.get('episodes', [])
    if not rows or len({r['id'] for r in rows}) != len(rows) or len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('missing/duplicate episode identities')
    for r in rows:
        path = Path(r['path'])
        if (r['task'] != m['task'] or r['label'] not in LABELS or recording_label(r['path']) != r['label']
                or 'discarded' in path.parts or not EPISODE.fullmatch(path.name)
                or not path.is_absolute() or path.name != r['filename']
                or r['id'] != f"{m['task']}-{r['label']}-{path.stem}" or not re.fullmatch('[0-9a-f]{64}',r['sha256'])
                or r['split'] not in ('train','validation') or (m['variant']=='S' and r['label']!='success')):
            raise ValueError(f"invalid directory label/identity/split: {r['id']}")
    excluded = m.get('excluded',[])
    amendment = m.get('amendment')
    if excluded:
        if not amendment or amendment.get('format') != 'rw-gate-amendment/1' or amendment.get('task') != m['task']:
            raise ValueError('exclusions require the documented task gate amendment')
        wanted = {Path(k).name:v['raw_sha256'] for k,v in amendment['excluded'].items()}
        if ({r['filename']:r['sha256'] for r in excluded} != wanted
                or len({r['id'] for r in excluded}) != len(excluded)
                or any(r['split']!='excluded' or r['label']!='success' for r in excluded)
                or set(wanted)&{r['filename'] for r in rows if r['label']=='success'}
                or {r['filename']:r['sha256'] for r in rows if r['label']=='success'} !=
                   {Path(k).name:v for k,v in amendment['episodes'].items()}):
            raise ValueError('active/excluded raw hashes differ from the task amendment')
    want = {s:[r['id'] for r in rows if r['split']==s] for s in ('train','validation')}
    outcomes = {r['id']:r['label']=='success' for r in rows}
    if m.get('split') != want or m.get('outcomes') != outcomes:
        raise ValueError('cached split/outcome map differs from episode records')
    if not all(want.values()):
        raise ValueError('both train and validation episodes required')
    return m


def manifest_gate(manifest, ingest):
    """Evaluate the established QA rules on the exact immutable variant membership."""
    from s4d.rwlearn import gate
    validate_manifest(manifest)
    task, rows = manifest['task'], manifest['episodes']
    selected = {}
    for r in manifest.get('excluded',[]):
        matches = [e for e in ingest['episodes'].values()
                   if e.get('task')==task and e.get('sha256')==r['sha256']]
        if len(matches)!=1 or matches[0].get('validation',{}).get('status')!='invalid':
            raise ValueError('excluded task-amendment episode must remain invalid in live ingest')
    for r in rows:
        candidates = [e for e in ingest['episodes'].values()
                      if e.get('task') == task and e.get('sha256') == r['sha256']]
        if len(candidates) != 1:
            raise ValueError(f"{r['id']}: require exactly one live ingest record for raw hash")
        e = copy.deepcopy(candidates[0])
        e['filename'] = r['path']
        e['source'] = {**e.get('source', {}), 'path': r['path']}
        e['episode_label'] = r['label']
        selected[r['id']] = e
    view = {'schema': gate.SCHEMA, 'schema_version': gate.SCHEMA_VERSION,
            'episodes': selected, 'counts': {task: {'target': len(rows), 'complete': True}}}
    report = gate.evaluate(view, [task], target=len(rows), amendments={})
    if not report['passed']:
        raise ValueError(f"{task}/{manifest['variant']}: QA gate failed: {report['tasks'][task]['problems']}")
    return report


def available_preprocessing(task, roots, *, inspect=inspect_raw):
    """Provisional prep inventory while verified transfers arrive; never a training split.

    Only complete filenames are opened. Invalid raw QA is reported per episode so
    one rejected recording does not prevent phase-A prep of other complete files.
    Final manifests still require exact counts, transfer markers and all raw QA.
    """
    if task not in TASKS or set(roots) != set(LABELS):
        raise ValueError('task/label roots must be explicit')
    rows, rejected = [], []
    for label in LABELS:
        root = Path(roots[label])
        if root.is_symlink() or 'discarded' in root.parts:
            raise ValueError('unsafe raw root')
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            if not EPISODE.fullmatch(path.name):
                continue
            try:
                before = path.stat()
                qa = inspect(path)
                digest = sha256(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                        after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError('changed while inspecting/hashing')
                rows.append(dict(id=f'{task}-{label}-{path.stem}',task=task,label=label,filename=path.name,
                                 path=str(path.resolve()),sha256=digest,qa=qa))
            except (ValueError, OSError, KeyError, TypeError) as exc:
                rejected.append(dict(path=str(path),label=label,reason=str(exc)))
    return {'SF':{'task':task,'episodes':rows}}, rejected


def reuse_from_ingest(manifest, ingest):
    """Discover existing success outputs by exact raw hash; downstream full QA remains required."""
    reuse = {}
    for row in manifest['episodes']:
        if row['label'] != 'success':
            continue
        matches = [e for e in ingest['episodes'].values()
                   if e.get('task') == manifest['task'] and e.get('sha256') == row['sha256']]
        if len(matches) != 1:
            continue
        stage = matches[0].get('preprocessing',{}).get('stages',{}).get('train_data',{})
        if stage.get('status') == 'done' and stage.get('input_sha256') == row['sha256'] and stage.get('outputs'):
            reuse[row['id']] = {'raw_sha256':row['sha256'],'train_data':stage['outputs']}
    return reuse


def verified_success_from_ingest(ingest, task):
    """Existing success roots need not be pulled again or receive new marker files.

    Accept only unique identities whose current ingest transfer SHA is verified;
    paired_manifests rehashes every local file and requires exact set equality.
    Invalid amended recordings still need transfer integrity, not training QA.
    """
    from s4d.rwlearn.identity import recording_label
    verified = {}
    for key,e in ingest['episodes'].items():
        path = e.get('filename') or e.get('source',{}).get('path') or key
        if e.get('task') != task or recording_label(path) != 'success':
            continue
        transfer = e.get('transfer',{})
        digest = e.get('sha256')
        if transfer.get('status') != 'verified' or transfer.get('remote_sha256') != digest:
            continue
        name = Path(path).name
        if name in verified:
            raise ValueError('duplicate existing success transfer identity')
        verified[name] = digest
    return verified
