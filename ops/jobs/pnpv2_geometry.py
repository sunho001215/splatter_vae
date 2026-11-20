"""PNP-v2 geometry publication worker and verified compact mirror closure.

Opt-in only. Existing dsreset/drawer producers, queues and live ingest are untouched.
The worker runs in the admitted remote prep image at the NEW publication source pin.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import yaml
from s4d.realworld.calib.pnpv2_acceptance_override import installed

from ops.jobs.pnpv2_rgb import REMOTE_RW, remote_raw, require
from ops.jobs.rwprep_mig import digest
from s4d.rwlearn.pnpv2_publication import manifest_check
from s4d.rwlearn.pnpv2_offline import no_override, readiness, rig_gate, build_geometry
from s4d.rwlearn.pnpv2_spec import BUNDLE_FILES, PREP_DISK_GIB, SPEC, STEM, TASK, load_spec, values

BUNDLE = f'{STEM}-geometry'
JOB = f'{STEM}-geopub-{TASK}-S'
DOCS = tuple(n for _, n in BUNDLE_FILES)
EXTRAS = ('publication-binding.json', 'portable-manifest.json', 'remote-raw-hashes.json', 'snapshot/manifest.json',
          'snapshot/provenance.json', 'calibration-plan.json', 'measurement/scene.json')


def locations(spec):
    processed = Path(values(spec)['remote_processed'])
    require(processed == Path(REMOTE_RW) / 'processed' / STEM, 'geometry requires the pnpv2 processed release')
    return processed / 'geometry-publication' / TASK / 'S', processed / 'learning-inputs' / TASK / 'S' / 'geometry'


def load(path):
    return json.loads(Path(path).read_bytes())


def once(path, data):
    """Create atomically or require byte equality; never replace an immutable artifact."""
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), f'symlink artifact: {path}')
    if path.exists():
        require(path.read_bytes() == data, f'immutable artifact conflict: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            require(path.read_bytes() == data, f'immutable artifact conflict: {path}')
    finally:
        tmp.unlink()


def json_once(path, doc):
    once(path, (json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def portable_manifest(manifest):
    portable = copy.deepcopy(manifest)
    for row in portable['episodes'] + portable.get('excluded', []):
        row['path'] = remote_raw(row)
    return portable


def completed_phase(bundle, manifest, phase, processed):
    """Read original queues and real mirrored states, including attach's exact JSON representation."""
    bundle = Path(bundle)
    cfg, jobs = load(bundle / 'dispatcher.json'), load(bundle / 'jobs.json')
    state = load(bundle / 'state/state.json')
    episodes = {r['id']: r for r in manifest['episodes']}
    require(not (bundle / 'state/STOP').exists(), f'Phase {phase} STOP retained')
    require(cfg['processed'] == str(processed) and cfg.get('release', STEM) == STEM,
            'phase processed release differs')
    original_jobs = jobs
    if TASK == 'ball-hiding':
        from s4d.rwlearn.ball_hiding_membership import eligible_jobs
        jobs = eligible_jobs(jobs)
    require(len(jobs) == len(episodes) and {j['episode'] for j in jobs} == set(episodes),
            'phase episodes differ from the manifest')
    require(all(j['phase'] == phase and j['id'] == f"{STEM}-prep{phase}-{j['episode']}" for j in jobs),
            'phase jobs must use the v2 stem')
    # attach() normalizes priority and writes indent=2/no newline. A remote compiler may have another byte layout.
    attached = copy.deepcopy(original_jobs)
    for job in attached:
        job['priority'] = -4
    attached_sha = hashlib.sha256(json.dumps(attached, indent=2).encode()).hexdigest()
    require(state['queue_sha256'] in (digest(bundle / 'jobs.json'), attached_sha), 'phase queue/state hash mismatch')
    require(set(state['jobs']) == {j['id'] for j in original_jobs}
            and all(state['jobs'][j['id']]['status'] == 'done' for j in jobs), f'Phase {phase} not complete')
    for job in jobs:
        paths = [p for p in job['inputs'] if p.endswith('.yaml')]
        require(len(paths) == 1 and digest(paths[0]) == job['inputs'][paths[0]], 'phase config input drift')
        pipeline = yaml.safe_load(Path(paths[0]).read_text())
        row = episodes[job['episode']]
        raw = remote_raw(row)
        root = processed / TASK / 'success' / Path(row['filename']).stem
        require(pipeline['adapter']['config']['episodes'] == {job['episode']: raw}
                and job['inputs'].get(raw) == row['sha256'], 'phase raw identity differs')
        require(pipeline['output_root'] == str(root), 'phase output escapes the v2 episode root')
    return cfg, jobs


def remote_job(spec, bundle, phase_a, phase_b, phase_ids, remote_inputs):
    """Stage with the CURRENT source commit, rather than Phase A's older worker code."""
    remote = f'{REMOTE_RW}/configs/{BUNDLE}'
    files = ('manifest.json', 'source-ingest.json', 'path-provenance.json')
    return dict(id=JOB, priority=-4, gpu_profile='1g.24gb', gpu_gib=1, ram_gib=32, cpus=4, disk_gib=PREP_DISK_GIB,
                inputs={**{str(Path(bundle) / n): digest(Path(bundle) / n) for n in files}, **remote_inputs},
                deps=[], external_states={f'{p}/state/state.json': list(ids) for p, ids in phase_ids.items()},
                steps=[dict(kind='python', argv=(["-m", "ops.jobs.ball_hiding_worker", "geometry"] if TASK == "ball-hiding" else
                      ['-m', 'ops.jobs.pnpv2_geometry', 'remote']) + [
                      '--source', remote, '--phase-a', phase_a, '--phase-b', phase_b])])


def _directory(target, produce, check):
    target = Path(target)
    require(not any(p.is_symlink() for p in (target, *target.parents)), f'symlink artifact: {target}')
    if not target.exists():
        private = target.with_name('.' + target.name + '.building')
        require(not private.is_symlink(), f'symlink artifact: {private}')
        if not private.exists():
            produce(private)
        check(private)  # interrupted partial output is held for inspection
        require(not target.exists(), f'immutable publication appeared: {target}')
        private.rename(target)
    check(target)


def check_inputs(spec, path):
    readiness('calib', path / 'manifest.json', path / 'split_seed0.json', path / 'outcomes.json',
              path / 'gate_report.json', path / 'store.yaml', spec)


def check_scene(path, manifest, rig):
    from s4d.rwlearn.dsreset_measure import validate_report
    report = load(path / 'scene.json')
    no_override(report)
    validate_report(report, manifest)
    require(report['native_provenance']['rig_document_sha256'] == digest(rig), 'scene rig differs')


def closure_paths(spec):
    root, inputs = locations(spec)
    rig = Path(values(spec)['rig'])
    decisions = [rig.parent / n for n in ('automatic-decision.json', 'override-decision.json')]
    # Include decisions whenever this publication used the committed override.
    mirror_rig = rig if rig.is_file() else Path(values(spec)['rig_local'])
    if mirror_rig.is_file():
        decisions = decisions if load(mirror_rig).get('acceptance_override') else []
    else:
        decisions = []
    return [*decisions, *[root / n for n in EXTRAS], *[inputs / n for n in DOCS], rig, rig.with_suffix('.reference.npz')]


def verify_complete(spec, marker, *, source=lambda p: p):
    expected = {str(p) for p in closure_paths(spec)}
    require(set(marker) == expected, 'publication closure differs')
    for path, sha in marker.items():
        require(digest(source(Path(path))) == sha, f'publication closure hash mismatch: {path}')


@installed()
def remote_build(source, phase_a, phase_b, *, spec=None):
    """Snapshot -> calibration release -> immutable geometry bundle -> train-only native measurement."""
    from s4d.rwlearn.pnpv2_runtime import install
    install()
    from s4d.realworld.calib import release
    from s4d.realworld.calib.rig import load_rig
    from s4d.rwlearn.dsreset_measure import SampledNativeStore, measure
    from s4d.rwlearn.store import open_store
    import scripts.pick_and_place_v2 as producer

    spec = spec or load_spec(Path(__file__).resolve().parents[2] / SPEC)
    p = values(spec)
    root, inputs = locations(spec)
    source, phase_a, phase_b = map(Path, (source, phase_a, phase_b))
    m = manifest_check(load(source / 'manifest.json'), spec)
    rig = Path(p['rig'])
    doc, _, _ = load_rig(rig)
    gate = rig_gate(rig)
    require(doc.get('task') == TASK and set(doc['episodes']) == {r['id'] for r in m['episodes']},
            'rig must be accepted offline for the complete task')
    for phase, bundle in (('A', phase_a), ('B', phase_b)):
        completed_phase(bundle, m, phase, Path(p['remote_processed']))
    stable = [source / n for n in ('manifest.json', 'source-ingest.json', 'path-provenance.json')]
    stable += [bundle / n for bundle in (phase_a, phase_b) for n in ('dispatcher.json', 'jobs.json')]
    stable += [rig, rig.with_suffix('.reference.npz')]
    if gate['acceptance_override'] is not None:
        stable += [rig.parent / n for n in ('automatic-decision.json', 'override-decision.json')]
    binding = {str(f): digest(f) for f in stable}
    require(not any(p.is_symlink() for p in (root, *root.parents)), f'symlink artifact: {root}')
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        json_once(root / 'publication-binding.json', binding)
        marker = root / 'publication-complete.json'
        if marker.exists():
            verify_complete(spec, load(marker))
            check_inputs(spec, inputs)
            check_scene(root / 'measurement', m, rig)
            return dict(out=str(inputs), reused=True)
        portable = portable_manifest(m)
        json_once(root / 'portable-manifest.json', portable)
        proofs = {}
        for row in portable['episodes']:
            raw = Path(row['path'])
            before = raw.stat()
            actual = digest(raw)
            after = raw.stat()
            require((before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    == (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                    and actual == row['sha256'], f"remote raw changed: {row['id']}")
            proofs[row['sha256']] = actual
        json_once(root / 'remote-raw-hashes.json', proofs)
        snapshot = root / 'snapshot'

        def check_snapshot(path):
            ingest, prov = load(path / 'manifest.json'), load(path / 'provenance.json')
            no_override(ingest)
            require(prov == dict(source={str(source / 'source-ingest.json'): digest(source / 'source-ingest.json')},
                    manifests={str(root / 'portable-manifest.json'): digest(root / 'portable-manifest.json')},
                    remote_hashes_sha256=digest(root / 'remote-raw-hashes.json'), selection='S'), 'snapshot provenance differs')
            require(len(ingest['episodes']) == len(m['episodes']) + len(m.get('excluded', [])) and
                    {e['episode_id']: e['sha256'] for e in ingest['episodes'].values()}
                    == {r['id']: r['sha256'] for r in m['episodes'] + m.get('excluded', [])}, 'snapshot membership differs')

        _directory(snapshot, lambda target: producer.snapshot(spec, None, [
            '--success-only', '--source', str(source / 'source-ingest.json'), '--manifest', str(root / 'portable-manifest.json'),
            '--prep-config', str(phase_a / 'dispatcher.json'), '--prep-config', str(phase_b / 'dispatcher.json'),
            '--remote-hashes', str(root / 'remote-raw-hashes.json'), '--out-dir', str(target)]), check_snapshot)
        plan_path = root / 'calibration-plan.json'
        if not plan_path.exists():
            json_once(plan_path, release.plan(load(snapshot / 'manifest.json'), {TASK: rig}))
        plan = load(plan_path)
        no_override(plan)
        task = plan['tasks'][TASK]
        require(set(plan['tasks']) == {TASK} and not task['refused'] and task['problem_count'] == 0
                and len(task['records']) == len(m['episodes']) and task['rig']['sha256'] == digest(rig),
                'calibration record plan refused or incomplete')
        preview = release.apply(plan, snapshot)
        require(all('refused' not in t for t in preview['tasks'].values()), 'calibration release refused')
        current = load(snapshot / 'manifest.json')['episodes']
        if any(any(current[k]['preprocessing']['stages'].get(release.STAGE, {}).get(f) != v for f, v in rec.items())
               for k, rec in task['records'].items()):
            result = release.apply(plan, snapshot, write=True)
            require(result['tasks'][TASK].get('written') == len(m['episodes']), 'calibration release incomplete')
        _directory(inputs, lambda target: build_geometry(spec, source / 'manifest.json', snapshot / 'manifest.json', target),
                   lambda path: check_inputs(spec, path))

        def measurement(target):
            base = open_store(inputs / 'store.yaml')
            try:
                by_sha = {r['sha256']: remote_raw(r) for r in m['episodes']}
                native = SampledNativeStore(base, rig, raw_paths={ep: by_sha[base.raw_sha256(ep)] for ep in base.episodes()})
                report = measure(m, native, load(inputs / 'split_seed0.json'))
                json_once(target / 'scene.json', report)
            finally:
                base.close()

        _directory(root / 'measurement', measurement, lambda path: check_scene(path, m, rig))
        require(binding == {str(f): digest(f) for f in stable}, 'publication inputs changed during build')
        json_once(marker, {str(f): digest(f) for f in closure_paths(spec)})
        return dict(out=str(inputs), episodes=len(m['episodes']), reused=False)


def fetch(spec, client=None):
    """Read-only remote fetch; full closure hashes before publishing any local planning evidence."""
    from ops.jobs import realworld_local as local

    p = values(spec)
    root, inputs = locations(spec)
    state = load(Path(p['work']) / 'geometry-remote/state/state.json')['jobs']
    require(state.get(JOB, {}).get('status') == 'done', 'remote geometry publication job is not done')
    paths = [*closure_paths(spec), root / 'publication-complete.json']
    rels = [str(path.relative_to(REMOTE_RW)) for path in paths]
    result = local.fetch(rels, client=client, priority='dependency', mirror=local.MIRROR)
    require(not result['missing'], 'incomplete verified geometry mirror fetch')
    files = local._manifest(local.MIRROR / 'MANIFEST.json')['files']
    seen = set()
    for rel in rels:
        local._check(rel, files, local.MIRROR, seen, full=True)

    def mirrored(path):
        rel = path.relative_to(REMOTE_RW)
        return local.MIRROR / ('_remote' if path.name == 'store.yaml' else '') / rel

    marker = load(mirrored(root / 'publication-complete.json'))
    verify_complete(spec, marker, source=mirrored)
    m = manifest_check(load(mirrored(inputs / 'manifest.json')), spec)
    require(m == load(p['manifest']), 'remote geometry manifest differs from ingest')
    # Check the localized descriptor against the real H5 closure, not merely the report-level gate.
    localized = local.MIRROR / inputs.relative_to(REMOTE_RW)
    check_inputs(spec, localized)
    check_scene(mirrored(root / 'measurement'), m, mirrored(Path(p['rig'])))
    copies = {Path(p['geometry_dir_local']) / n: mirrored(inputs / n) for n in DOCS}
    copies.update({Path(p['snapshot']) / n: mirrored(root / 'snapshot' / n) for n in ('manifest.json', 'provenance.json')})
    copies[Path(p['scene_report'])] = mirrored(root / 'measurement/scene.json')
    copies[Path(p['rig_local'])] = mirrored(Path(p['rig']))
    copies[Path(p['rig_local']).with_suffix('.reference.npz')] = mirrored(Path(p['rig']).with_suffix('.reference.npz'))
    if load(copies[Path(p['rig_local'])]).get('acceptance_override'):
        for name in ('automatic-decision.json', 'override-decision.json'):
            copies[Path(p['rig_local']).parent / name] = mirrored(Path(p['rig']).parent / name)
    for target, source in copies.items():
        once(target, source.read_bytes())
    receipt = Path(p['geometry_dir_local']).parent / 'geometry.FETCHED.json'
    json_once(receipt, dict(schema='pick-and-place-v2-geometry-fetch/1', files={r: files[r]['sha256'] for r in seen},
                           publication=marker))
    return dict(out=p['geometry_dir_local'], verified_files=len(seen), **result)


def mirror_published(spec, client):
    """After upload, mirror the canonical documents used by row argv (the H5 closure is already fetched)."""
    from ops.jobs import realworld_local as local
    p = values(spec)
    rels = [str((Path(p['geometry_dir']) / n).relative_to(REMOTE_RW)) for n in DOCS]
    result = local.fetch(rels, client=client, priority='dependency', mirror=local.MIRROR)
    require(not result['missing'], 'incomplete canonical geometry mirror')
    files = local._manifest(local.MIRROR / 'MANIFEST.json')['files']
    seen = set()
    for name, rel in zip(DOCS, rels):
        local._check(rel, files, local.MIRROR, seen, full=True)
        require(files[rel]['remote_sha256'] == digest(Path(p['geometry_dir_local']) / name),
                'canonical geometry mirror differs from staging')
    json_once(Path(p['geometry_dir_local']).parent / 'geometry.PUBLISHED-MIRROR.json',
              dict(schema='pick-and-place-v2-geometry-published-mirror/1',
                   files={r: files[r]['sha256'] for r in seen}))


def mirror_evidence(spec, *, canonical=True):
    """Fast, local-only status check of this route's previously fully verified closure."""
    from ops.jobs import realworld_local as local
    p = values(spec)
    receipt = Path(p['geometry_dir_local']).parent / 'geometry.FETCHED.json'
    require(receipt.is_file(), 'geometry compact closure has not been fetched and verified')
    saved = load(receipt)
    require(saved.get('schema') == 'pick-and-place-v2-geometry-fetch/1', 'invalid geometry fetch receipt')
    files = local._manifest(local.MIRROR / 'MANIFEST.json')['files']
    seen = set()
    root, _ = locations(spec)
    expected = [*closure_paths(spec), root / 'publication-complete.json']
    for path in expected:
        rel = str(path.relative_to(REMOTE_RW))
        local._check(rel, files, local.MIRROR, seen, full=False)
    require(saved['files'] == {r: files[r]['sha256'] for r in seen}, 'geometry fetch receipt differs from mirror')
    complete_rel = str((root / 'publication-complete.json').relative_to(REMOTE_RW))
    require(saved['publication'] == load(local.MIRROR / complete_rel), 'publication receipt differs from mirror')
    for name in DOCS:
        _, inputs = locations(spec)
        require(digest(Path(p['geometry_dir_local']) / name) == saved['publication'][str(inputs / name)],
                'geometry staging differs from verified publication')
    if not canonical:
        return {str(receipt): digest(receipt)}
    published = load(Path(p['geometry_dir_local']).parent / 'geometry.PUBLISHED-MIRROR.json')
    require(published.get('schema') == 'pick-and-place-v2-geometry-published-mirror/1', 'invalid canonical mirror receipt')
    canonical_seen = set()
    for name in DOCS:
        rel = str((Path(p['geometry_dir']) / name).relative_to(REMOTE_RW))
        local._check(rel, files, local.MIRROR, canonical_seen, full=False)
        require(files[rel]['remote_sha256'] == digest(Path(p['geometry_dir_local']) / name),
                'canonical mirror differs from geometry staging')
    require(published['files'] == {r: files[r]['sha256'] for r in canonical_seen}, 'canonical mirror receipt differs')
    canonical_receipt = receipt.with_name('geometry.PUBLISHED-MIRROR.json')
    return {str(receipt): digest(receipt), str(canonical_receipt): digest(canonical_receipt)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['remote'])
    for name in ('source', 'phase-a', 'phase-b'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(remote_build(args.source, args.phase_a, args.phase_b)))


if __name__ == '__main__':
    main()
