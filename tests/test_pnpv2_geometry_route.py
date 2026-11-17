"""Synthetic PNP-v2 geometry route; no remote state, queues or controllers are modified."""
import copy
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ops.jobs import pnpv2_geometry as G, pnpv2_registration as reg, rwprep_remote as R
from ops.jobs import realworld_local as L
from s4d.rwlearn.pnpv2_arrival import task_known
from s4d.rwlearn.pnpv2_spec import SPEC, STEM, TASK, load_spec, values

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:' + 'b' * 64
COMMIT = 'a' * 40


_PROJECT_PACKAGES = ("s4d", "ops", "scripts", "tests")


@pytest.fixture(autouse=True, scope="module")
def _hermetic_imports():
    """Leave the interpreter exactly as this file found it.

    Tests here monkeypatch shared module globals (e.g. ``s4d.rwlearn.store.open_store``) and run entries that install
    the process-local pnp-v2/dsreset runtime (``sys.modules`` aliases, rebound task globals). A project module first
    imported inside such a window binds the patched objects for the rest of the process (2026-11-18: baseline_data
    bound the FakeStore opener; realworld_plan bound the patched TARGET_TASKS). On teardown every module first imported
    by this file is dropped (and its package attribute removed), replaced ``sys.modules`` entries are restored, and
    every pre-existing project module namespace is restored to its snapshot, so later files re-import cleanly.
    """
    before = dict(sys.modules)
    saved = {name: dict(vars(module)) for name, module in before.items()
             if module is not None and name.split(".")[0] in _PROJECT_PACKAGES and hasattr(module, "__dict__")}
    yield
    for name in [n for n in sys.modules if n not in before]:
        del sys.modules[name]
    for name, module in before.items():
        if sys.modules.get(name) is not module:
            sys.modules[name] = module
    for name, snapshot in saved.items():
        namespace = vars(before[name])
        for key in [k for k in namespace if k not in snapshot]:
            del namespace[key]
        namespace.update(snapshot)


@pytest.fixture(autouse=True)
def _restore_runtime_globals(monkeypatch):
    """Runtime task installs are process-local in production; never leak them into other test files.

    Binds the re-exporting modules (ops.jobs.realworld_plan, alias s4d.rwlearn.launch) to the original objects first,
    then registers restores for every global the pnp-v2/dsreset runtime installs rebind (including the sys.modules
    aliases of dsreset_runtime.install), so teardown returns them even if a test installs or patches them.
    """
    import s4d.rwlearn as rwlearn
    from ops.jobs import realworld_plan  # noqa: F401
    from s4d.rwlearn import configs, dsreset, dsreset_configs, dsreset_gate, dsreset_tasks

    for name in ("tasks", "identity", "gate", "configs", "outcomes"):
        key = "s4d.rwlearn." + name
        if key in sys.modules:
            monkeypatch.setitem(sys.modules, key, sys.modules[key])
        if hasattr(rwlearn, name):
            monkeypatch.setattr(rwlearn, name, getattr(rwlearn, name))
    for module, names in ((dsreset, ("TASKS",)), (dsreset_tasks, ("TARGETS", "TARGET_TASKS")),
                          (dsreset_configs._impl, ("TARGETS", "TARGET_TASKS", "SCENE_PARAMETERS")),
                          (dsreset_gate._impl, ("TARGETS",)), (configs, ("TARGETS", "TARGET_TASKS"))):
        for attribute in names:
            if hasattr(module, attribute):
                monkeypatch.setattr(module, attribute, getattr(module, attribute))


def write(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    spec = load_spec(ROOT / SPEC)
    spec['roots'].update(work=str(tmp_path / 'work'), processed=str(tmp_path / 'processed'),
                         raw=str(tmp_path / 'data/real_world'))
    raw = Path(values(spec)['raw_success'])
    raw.mkdir(parents=True)
    for i in range(100):
        (raw / f'episode_{i:06d}.h5').write_text(f'synthetic raw {i}')
    with task_known() as D:
        m = D.paired_manifests(TASK, {'success': raw}, variants=('S',), require_verified=False,
                               inspect=lambda p: {}, expected={'success': 100})['S']
    p = values(spec)
    write(p['manifest'], m)
    write(Path(p['work']) / 'remote-raw-hashes.json', {r['sha256']: r['sha256'] for r in m['episodes']})
    return SimpleNamespace(spec=spec, m=m, p=p, tmp=tmp_path)


def phase(bundle, m, phase='A', processed=None, raw_mapper=G.remote_raw):
    bundle = Path(bundle)
    processed = Path(processed or '/home/compu/kaist/sunho/realworld/processed/pnpv2-20261117')
    jobs = []
    for row in m['episodes']:
        raw = raw_mapper(row)
        cfg = bundle / 'configs' / (row['id'] + '.yaml')
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(yaml.safe_dump(dict(output_root=str(processed / TASK / 'success' / Path(row['filename']).stem),
              adapter={'config': {'episodes': {row['id']: raw}}})))
        jobs.append(dict(id=f"{STEM}-prep{phase}-{row['id']}", episode=row['id'], phase=phase, priority=-4,
                         inputs={str(cfg): G.digest(cfg), raw: row['sha256']}))
    write(bundle / 'jobs.json', jobs)
    config = dict(commit=COMMIT, processed=str(processed), release=STEM, remote=f'{R.RW}/configs/{STEM}-phase{phase}',
                  image=IMAGE, jobs=str(bundle / 'jobs.json'), host='remote')
    write(bundle / 'dispatcher.json', config)
    write(bundle / 'state/state.json', dict(queue_sha256=G.digest(bundle / 'jobs.json'),
                                          jobs={j['id']: {'status': 'done'} for j in jobs}))
    return config, jobs


def test_geometry_job_is_gated_hash_bound_and_uses_its_own_source(campaign, monkeypatch):
    import scripts.pick_and_place_v2 as P
    c = campaign
    ca, ja = phase(c.p['phase_a'], c.m)
    cb, jb = phase(c.p['phase_b'], c.m, 'B')
    rig = write(c.p['rig_local'], {'rig': 'accepted'})
    ref = rig.with_suffix('.reference.npz')
    ref.write_bytes(b'reference')
    src = write(c.tmp / 'source.json', {'episodes': {'old-v1': {'task': 'pick-and-place'},
                                                  'drawer': {'task': 'drawer-open-v2'}}})
    monkeypatch.setattr(P, '_check_data', lambda *a: {'passed': True})
    monkeypatch.setattr(reg, 'rig_evidence', lambda s: {'status': 'done'})
    monkeypatch.setattr(P, 'ATTACH_ROOT', c.tmp / 'repo/runs')

    class Client:
        def _remote_checksum(self, path):
            if path == c.p['rig']:
                return G.digest(rig)
            if path.endswith('.reference.npz'):
                return G.digest(ref)
            return 'e' * 64

    monkeypatch.setattr(R, 'RemoteClient', Client)
    calls = []

    def stage(jobs, out, client, **kw):
        # The compiler never asks stage to use the older Phase-A commit.
        assert 'commit' not in kw
        R.validate({'host': 'remote', 'commit': COMMIT, 'image': IMAGE}, jobs)
        calls.append((jobs[0], out, kw))
        assert G.load(out / 'source-ingest.json')['episodes'] == {}
        return {'launched': False}

    monkeypatch.setattr(R, 'stage', stage)
    P.geometry_remote(c.spec, None, ['--source', str(src)])
    job, out, kw = calls[0]
    assert out == c.tmp / f'repo/runs/{G.BUNDLE}'
    assert (Path(c.p['work']) / 'geometry-remote').readlink() == out
    assert kw['release'] == STEM and kw['name'] == G.BUNDLE and kw['image'] == IMAGE
    assert job['id'] == G.JOB and job['id'].startswith(STEM + '-')
    assert job['external_states'] == {ca['remote'] + '/state/state.json': [j['id'] for j in ja],
                                      cb['remote'] + '/state/state.json': [j['id'] for j in jb]}
    assert len(job['inputs']) == 9  # manifest/source/provenance, 2 original queues/configs, rig/reference
    assert job['steps'][0]['argv'][:3] == ['-m', 'ops.jobs.pnpv2_geometry', 'remote']
    assert all('dsreset-' not in p for p in job['inputs'])
    with pytest.raises(ValueError, match='immutable'):
        P.geometry_remote(c.spec, None, ['--source', str(src)])


@pytest.mark.parametrize('held', ['phaseA', 'phaseB', 'rig', 'raw-proof'])
def test_geometry_staging_refuses_before_remote_access(campaign, monkeypatch, held):
    import scripts.pick_and_place_v2 as P
    c = campaign
    phase(c.p['phase_a'], c.m)
    phase(c.p['phase_b'], c.m, 'B')
    monkeypatch.setattr(P, '_check_data', lambda *a: {'passed': True})
    monkeypatch.setattr(reg, 'rig_evidence', lambda s: {'status': 'pending' if held == 'rig' else 'done', 'detail': 'held'})
    if held in ('phaseA', 'phaseB'):
        bundle = c.p['phase_a' if held == 'phaseA' else 'phase_b']
        state = G.load(Path(bundle) / 'state/state.json')
        next(iter(state['jobs'].values()))['status'] = 'unknown'
        write(Path(bundle) / 'state/state.json', state)
    if held == 'raw-proof':
        (Path(c.p['work']) / 'remote-raw-hashes.json').unlink()
    monkeypatch.setattr(R, 'RemoteClient', lambda: pytest.fail('must refuse before remote access'))
    with pytest.raises(ValueError):
        P.geometry_remote(c.spec, None, [])


@pytest.mark.parametrize('change,why', [('state', 'not complete'), ('hash', 'hash mismatch'),
    ('stem', 'v2 stem'), ('config', 'input drift'), ('STOP', 'STOP'), ('release', 'release differs')])
def test_worker_binds_original_queue_configs_and_done_state(campaign, change, why):
    c = campaign
    bundle = c.tmp / 'phase'
    processed = c.tmp / 'remote-processed'
    cfg, jobs = phase(bundle, c.m, processed=processed)
    assert G.completed_phase(bundle, c.m, 'A', processed)[1] == jobs
    if change in ('state', 'hash'):
        state = G.load(bundle / 'state/state.json')
        if change == 'state':
            next(iter(state['jobs'].values()))['status'] = 'running'
        else:
            state['queue_sha256'] = 'f' * 64
        write(bundle / 'state/state.json', state)
    elif change == 'stem':
        jobs[0]['id'] = 'rwv2-20261113-v1'
        write(bundle / 'jobs.json', jobs)
    elif change == 'config':
        Path(next(p for p in jobs[0]['inputs'] if p.endswith('.yaml'))).write_text('drift')
    elif change == 'STOP':
        (bundle / 'state/STOP').touch()
    else:
        cfg['processed'] = '/processed/dsreset-20261113'
        write(bundle / 'dispatcher.json', cfg)
    with pytest.raises(ValueError, match=why):
        G.completed_phase(bundle, c.m, 'A', processed)


def test_worker_accepts_only_the_exact_attach_representation(campaign):
    c = campaign
    bundle = c.tmp / 'phase'
    processed = c.tmp / 'processed'
    _, jobs = phase(bundle, c.m, processed=processed)
    # Simulate remote compiler's differing priority, with the genuine attached normalized hash.
    attached_sha = hashlib.sha256(json.dumps(jobs, indent=2).encode()).hexdigest()
    for job in jobs:
        job['priority'] = -3
    write(bundle / 'jobs.json', jobs)
    state = G.load(bundle / 'state/state.json')
    state['queue_sha256'] = attached_sha
    write(bundle / 'state/state.json', state)
    G.completed_phase(bundle, c.m, 'A', processed)
    jobs[0]['inputs']['/different'] = 'a' * 64
    write(bundle / 'jobs.json', jobs)
    with pytest.raises(ValueError, match='hash mismatch'):
        G.completed_phase(bundle, c.m, 'A', processed)


class FakeStore:
    def __init__(self, manifest):
        self.rows = {r['id']: r for r in manifest['episodes']}
    def episodes(self): return list(self.rows)
    def task(self, ep): return TASK
    def raw_recording(self, ep): return self.rows[ep]['path']
    def raw_sha256(self, ep): return self.rows[ep]['sha256']
    def cameras(self, ep): return ['135122070988', '135122073201', '135122076054']
    def upstream(self, ep): return {'calibrate': 'cal-build-1'}
    def close(self): pass


def gate_environment(monkeypatch, m):
    # Opt-in runtime aliases are process-local in production; keep tests isolated from other files in this worker.
    from s4d.rwlearn import dsreset_gate, configs
    from s4d.rwlearn import store
    monkeypatch.setattr(dsreset_gate._impl, 'TARGETS', (*dsreset_gate._impl.TARGETS, TASK))
    monkeypatch.setattr(configs, 'TARGET_TASKS', {**configs.TARGET_TASKS, TASK: (TASK,)})
    monkeypatch.setattr(store, 'open_store', lambda *a: FakeStore(m))


def ingest_for(m, *, validated):
    from tests.test_rwlearn_gate import entry
    entries = {}
    for i, row in enumerate(m['episodes']):
        rec = entry(TASK, i, sha=row['sha256'], validated=validated)
        rec.update(episode_id=row['id'], filename=row['filename'])
        rec['source']['path'] = row['path']
        rec['preprocessing']['stages']['train_data']['outputs'] = '/processed/train_data/' + row['id']
        entries[row['id']] = rec
    return {'schema': 'rw-ingest-manifest', 'schema_version': 2, 'episodes': entries}


def test_shared_publisher_preserves_membership_split_and_calibration_hold(campaign, monkeypatch):
    from s4d.rwlearn.pnpv2_publication import build_geometry
    c = campaign
    gate_environment(monkeypatch, c.m)
    ingest = write(c.tmp / 'ingest.json', ingest_for(c.m, validated=True))
    with task_known():
        build_geometry(c.spec, c.p['manifest'], ingest, c.tmp / 'good')
    G.check_inputs(c.spec, c.tmp / 'good')
    assert G.load(c.tmp / 'good/manifest.json') == c.m
    assert G.load(c.tmp / 'good/split_seed0.json') == c.m['split']
    bad = ingest_for(c.m, validated=False)
    write(ingest, bad)
    with task_known():
        build_geometry(c.spec, c.p['manifest'], ingest, c.tmp / 'held')
    with pytest.raises(RuntimeError, match='geometry-dependent'):
        G.check_inputs(c.spec, c.tmp / 'held')
    bad['episodes'][c.m['episodes'][0]['id']]['acceptance_override'] = {'decision': 'accept'}
    write(ingest, bad)
    with pytest.raises(ValueError, match='no calibration acceptance override'):
        build_geometry(c.spec, c.p['manifest'], ingest, c.tmp / 'override')
    assert not (c.tmp / 'override').exists()


def test_private_publication_never_overwrites_partial_or_different_output(tmp_path):
    target = tmp_path / 'release'
    calls = []
    def produce(path):
        calls.append(path)
        G.once(path / 'done', b'good')
    def check(path):
        assert (path / 'done').read_bytes() == b'good'
    G._directory(target, produce, check)
    G._directory(target, produce, check)
    assert len(calls) == 1
    with pytest.raises(ValueError, match='immutable artifact conflict'):
        G.once(target / 'done', b'other')
    (target / 'done').write_bytes(b'partial')
    with pytest.raises(AssertionError):
        G._directory(target, produce, check)
    assert (target / 'done').read_bytes() == b'partial'


class Client:
    """Run the real read-only remote inventory against a synthetic remote filesystem."""
    def __init__(self, remote):
        self.remote = remote
        self.downloads = []
    def checked_ssh(self, command, **kwargs):
        assert kwargs == {'idempotent': True}
        assert command.startswith('cd ' + G.REMOTE_RW + ' && ')
        command = command.replace('cd ' + G.REMOTE_RW, 'cd ' + shlex.quote(str(self.remote)), 1)
        return subprocess.run(['bash', '-c', command], capture_output=True, text=True, check=True)
    def _remote_checksum(self, path, allow_missing=False):
        local = self.remote / Path(path).relative_to(G.REMOTE_RW)
        if allow_missing and not local.is_file():
            return None
        return G.digest(local)
    def rsync_upload(self, source, target):
        assert 'runs' in Path(source).parts
        G.once(self.remote / Path(target).relative_to(G.REMOTE_RW), Path(source).read_bytes())
    def download_batch(self, remote, local, wanted, **kwargs):
        assert kwargs['priority'] == 'dependency'
        source = self.remote / Path(remote).relative_to(G.REMOTE_RW)
        for name, sha in wanted.items():
            data = (source / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == sha
            G.once(local / name, data)
            self.downloads.append(str(local / name))
        return sorted(wanted)


@pytest.fixture
def publication(campaign, monkeypatch):
    c = campaign
    gate_environment(monkeypatch, c.m)
    from s4d.rwlearn.pnpv2_publication import build_geometry
    ingest = write(c.tmp / 'ingest.json', ingest_for(c.m, validated=True))
    docs = c.tmp / 'docs'
    with task_known():
        build_geometry(c.spec, c.p['manifest'], ingest, docs)
    root, inputs = G.locations(c.spec)
    remote = c.tmp / 'remote'
    def target(path): return remote / Path(path).relative_to(G.REMOTE_RW)
    for name in G.DOCS:
        G.once(target(inputs / name), (docs / name).read_bytes())
    paths = []
    for row in c.m['episodes']:
        ep = Path(c.p['remote_processed']) / TASK / 'success' / Path(row['filename']).stem / 'train_data' / row['id']
        G.once(target(ep / 'episode.h5'), b'synthetic compact pixels')
        G.json_once(target(ep / '_COMPLETE.json'), {'stage': 'train_data'})
        paths.append(str(ep))
    target(inputs / 'store.yaml').write_text(yaml.safe_dump({'kind': 'compact', 'paths': paths}))
    rig = Path(c.p['rig'])
    G.json_once(target(rig), {'rig': 'accepted'})
    G.once(target(rig.with_suffix('.reference.npz')), b'ref')
    report = dict(schema='dsreset-scene-measurement/1', accepted=True, task=TASK, variant='S',
                  depth_source='native_realsense', manifest_sha256=hashlib.sha256(json.dumps(c.m, sort_keys=True).encode()).hexdigest(),
                  episodes=c.m['split']['train'], bounds=[0, 0, 0, 1, 1, 1], lookat=[.5, .5, .5], pose_scale=1.,
                  containment={k: 1. for k in ('robot', 'object', 'fixture')}, scene_statistics={'verified': True},
                  native_provenance={'rig_document_sha256': G.digest(target(rig))})
    for name in G.EXTRAS:
        G.json_once(target(root / name), report if name == 'measurement/scene.json' else {'synthetic': name})
    marker = {str(path): G.digest(target(path)) for path in G.closure_paths(c.spec)}
    G.json_once(target(root / 'publication-complete.json'), marker)
    write(Path(c.p['work']) / 'geometry-remote/state/state.json', {'jobs': {G.JOB: {'status': 'done'}}})
    monkeypatch.setattr(L, 'MIRROR', c.tmp / 'repo/runs/realworld-inputs')
    client = Client(remote)
    c.remote, c.client, c.target, c.inputs, c.pubroot, c.marker = remote, client, target, inputs, root, marker
    return c


def test_fetch_verifies_full_compact_closure_and_mirrors_canonical_row_paths(publication, monkeypatch):
    import scripts.pick_and_place_v2 as P
    from ops.remote import access
    c = publication
    result = G.fetch(c.spec, c.client)
    assert result['missing'] == [] and result['verified_files'] >= 200
    assert G.load(c.p['geometry_manifest_local']) == c.m
    assert G.load(c.p['scene_report'])['episodes'] == c.m['split']['train']
    # Staging keeps the remote descriptor; rows get the localized descriptor under runs/realworld-inputs.
    localized = L.MIRROR / c.inputs.relative_to(G.REMOTE_RW) / 'store.yaml'
    assert str(L.MIRROR) in localized.read_text()
    assert G.REMOTE_RW in Path(c.p['geometry_store_local']).read_text()
    G.fetch(c.spec, c.client)  # equality-only resume, no replacement
    canonical = Path(c.p['geometry_dir'])
    monkeypatch.setattr(P, 'ATTACH_ROOT', c.tmp / 'repo/runs')
    monkeypatch.setattr(access, 'RemoteClient', lambda: c.client)
    P.upload(c.spec, None, ['--view', 'geometry'])
    assert reg.bundle_evidence(c.spec, 'geometry', c.m)['status'] == 'done'
    assert len(G.mirror_evidence(c.spec)) == 2
    assert (L.MIRROR / canonical.relative_to(G.REMOTE_RW) / 'store.yaml').is_file()
    Path(c.p['geometry_store_local']).write_text('changed')
    with pytest.raises(ValueError, match='differs'):
        G.mirror_evidence(c.spec)


@pytest.mark.parametrize('corruption', ['marker', 'scene', 'missing', 'not-done', 'local-conflict'])
def test_fetch_refuses_incomplete_corrupt_or_conflicting_publications(publication, corruption):
    c = publication
    if corruption == 'marker':
        marker = copy.deepcopy(c.marker)
        marker[str(c.inputs / 'manifest.json')] = 'f' * 64
        write(c.target(c.pubroot / 'publication-complete.json'), marker)
    elif corruption == 'scene':
        path = c.target(c.pubroot / 'measurement/scene.json')
        scene = G.load(path)
        scene['episodes'].append(c.m['split']['validation'][0])
        write(path, scene)
        c.marker[str(c.pubroot / 'measurement/scene.json')] = G.digest(path)
        write(c.target(c.pubroot / 'publication-complete.json'), c.marker)
    elif corruption == 'missing':
        ep = yaml.safe_load(c.target(c.inputs / 'store.yaml').read_text())['paths'][0]
        c.target(Path(ep) / 'episode.h5').unlink()
    elif corruption == 'not-done':
        write(Path(c.p['work']) / 'geometry-remote/state/state.json', {'jobs': {G.JOB: {'status': 'running'}}})
    else:
        write(c.p['scene_report'], {'different': True})
    with pytest.raises((ValueError, L.LocalInputError, subprocess.CalledProcessError)):
        G.fetch(c.spec, c.client)
    assert not (Path(c.p['geometry_dir_local']).parent / 'geometry.FETCHED.json').exists()
    if corruption != 'local-conflict':
        assert not Path(c.p['scene_report']).exists()


def test_status_gives_complete_remote_geometry_commands(campaign):
    ev = {stage: {'status': 'done'} for stage in ('arrival', 'check-data', 'ingest', 'phaseA', 'rgb-publication', 'rig', 'phaseB')}
    ev.update({stage: {'status': 'pending'} for stage in ('geometry-publication', 'scene')})
    commands = reg.next_steps(campaign.spec, ev)['geometry-publication']
    assert 'geometry-remote' in commands[0] and 'rwprep_remote tick' in commands[1]
    assert 'geometry-fetch' in commands[2] and '--view geometry' in commands[3]
    assert not any('realworld_prep_followup.py' in cmd for cmd in commands)


@pytest.mark.local_host_data  # Snapshot and release.apply load the repository's ingest validator/writer under runs/setup/tools.
def test_remote_worker_runs_snapshot_release_publish_measure_and_resumes(campaign, monkeypatch):
    import scripts.realworld_dsreset_snapshot as snapshot_module
    from s4d.realworld.calib import release, rig as rig_module
    from s4d.rwlearn import dsreset_measure as measurement, pnpv2_runtime
    from s4d.rwlearn.pnpv2_publication import build_geometry
    from tests.test_rwlearn_gate import calibration_record
    c = campaign
    gate_environment(monkeypatch, c.m)
    # Keep process-wide runtime aliases isolated; exercise the real snapshot, gate, release writer and report check.
    monkeypatch.setattr(pnpv2_runtime, 'install', lambda: None)
    processed = c.tmp / 'remote-processed'
    c.spec['roots']['remote_processed'] = str(processed)
    c.p = values(c.spec)
    output, inputs = processed / 'geometry-publication' / TASK / 'S', processed / 'learning-inputs' / TASK / 'S/geometry'
    monkeypatch.setattr(G, 'locations', lambda spec: (output, inputs))
    rig = write(c.p['rig'], {'rig': 'accepted'})
    rig.with_suffix('.reference.npz').write_bytes(b'reference')
    document = {'task': TASK, 'episodes': [r['id'] for r in c.m['episodes']], 'cameras': {'cam/color': {}}}
    monkeypatch.setattr(rig_module, 'load_rig', lambda p: (document, {}, {}))
    monkeypatch.setattr(rig_module, 'rig_camera_status', lambda *a: ('validated', ''))
    remote_raw = lambda row: row['path']
    monkeypatch.setattr(G, 'remote_raw', remote_raw)
    a, b = c.tmp / 'A', c.tmp / 'B'
    phase(a, c.m, processed=processed, raw_mapper=remote_raw)
    phase(b, c.m, 'B', processed=processed, raw_mapper=remote_raw)
    source = c.tmp / 'source'
    write(source / 'manifest.json', c.m)
    write(source / 'source-ingest.json', {'schema': 'rw-ingest-manifest', 'schema_version': 2, 'episodes': {}})
    write(source / 'path-provenance.json', {'synthetic': True})
    # Synthetic completion records take the same snapshot route as real Phase A/B outputs.
    from s4d.rwlearn import gate
    for row in c.m['episodes']:
        root = processed / TASK / 'success' / Path(row['filename']).stem
        for stage in ('normalized', 'foundation_stereo_depth', 'sam3_masks', 'calibrate', 'train_data'):
            path = root / 'episodes' / row['id'] / ('normalized' if stage == 'normalized' else 'derived/' + stage)
            if stage == 'train_data':
                path = root / 'train_data' / row['id']
            summary = {'format': 's4d.realworld.train', 'schema_version': '1.1.0'} if stage == 'train_data' else {}
            write(path / '_COMPLETE.json', dict(stage=stage, version='1', build_id='cal-build-1', fingerprint='fp', summary=summary))
        write(root / 'qa.json', dict(schema='rwprep-qa/3', passed=True, families={k: {'passed': True} for k in gate.QA_FAMILIES}))
    monkeypatch.syspath_prepend(str(ROOT / 'runs/setup/tools/realworld_ingest_20261106'))
    import episode_validator
    monkeypatch.setattr(episode_validator, 'validate', lambda *a, **kw: {'status': 'valid'})
    calls = []
    def plan(ingest, rigs):
        calls.append('records')
        assert rigs == {TASK: rig}
        assert len(ingest['episodes']) == 100
        records = {k: calibration_record(v['sha256']) for k, v in ingest['episodes'].items()}
        return {'tasks': {TASK: dict(refused=False, problem_count=0, problems=[], records=records,
                                    rig={'sha256': G.digest(rig)})}}
    monkeypatch.setattr(release, 'plan', plan)
    real_apply = release.apply
    def apply(*args, **kw):
        calls.append('apply' if kw.get('write') else 'preview')
        return real_apply(*args, **kw)
    monkeypatch.setattr(release, 'apply', apply)
    def publish(*args):
        calls.append('publish')
        return build_geometry(*args)
    import s4d.rwlearn.pnpv2_publication as publication_module
    monkeypatch.setattr(publication_module, 'build_geometry', publish)
    def native(store, path, *, raw_paths):
        assert path == rig and set(raw_paths) == set(store.episodes())
        return store
    monkeypatch.setattr(measurement, 'SampledNativeStore', native)
    def measure(m, store, split):
        calls.append('measure')
        assert split == m['split'] and len(split['train']) == 90 and len(split['validation']) == 10
        return dict(schema='dsreset-scene-measurement/1', accepted=True, task=TASK, variant='S',
                    depth_source='native_realsense', manifest_sha256=hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest(),
                    episodes=split['train'], bounds=[0, 0, 0, 1, 1, 1], lookat=[.5, .5, .5], pose_scale=1.,
                    containment={k: 1. for k in ('robot', 'object', 'fixture')}, scene_statistics={'synthetic': True},
                    native_provenance={'rig_document_sha256': G.digest(rig)})
    monkeypatch.setattr(measurement, 'measure', measure)
    original_reader = snapshot_module.read_manifest
    with task_known():
        result = G.remote_build(source, a, b, spec=c.spec)
    assert result['episodes'] == 100 and result['reused'] is False
    assert snapshot_module.read_manifest is original_reader
    assert calls == ['records', 'preview', 'apply', 'publish', 'measure']
    assert G.load(source / 'source-ingest.json')['episodes'] == {}  # live/source catalog never rewritten
    saved = G.load(output / 'snapshot/manifest.json')
    assert all(v['preprocessing']['stages']['calibration_validated']['status'] == 'done' for v in saved['episodes'].values())
    before = {str(p): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert G.remote_build(source, a, b, spec=c.spec)['reused'] is True
    assert before == {str(p): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert calls == ['records', 'preview', 'apply', 'publish', 'measure']
    # A published document changing is a held conflict, never an overwrite on retry.
    (inputs / 'outcomes.json').write_text('{}')
    with pytest.raises(ValueError, match='closure hash mismatch'):
        G.remote_build(source, a, b, spec=c.spec)


@pytest.mark.parametrize('decision', ['held', 'override'])
def test_worker_refuses_rig_holds_and_acceptance_overrides_before_publication(campaign, monkeypatch, decision):
    from s4d.realworld.calib import rig
    from s4d.rwlearn import pnpv2_runtime
    c = campaign
    monkeypatch.setattr(pnpv2_runtime, 'install', lambda: None)
    source = c.tmp / 'source'
    write(source / 'manifest.json', c.m)
    doc = {'task': TASK, 'episodes': [r['id'] for r in c.m['episodes']], 'cameras': {'cam/color': {}}}
    if decision == 'override':
        doc['acceptance_override'] = {'id': 'not-authorized'}
    monkeypatch.setattr(rig, 'load_rig', lambda p: (doc, {}, {}))
    monkeypatch.setattr(rig, 'rig_camera_status', lambda *a: ('held', 'failed automatic acceptance'))
    with pytest.raises(ValueError, match='automatically accepted|acceptance override'):
        G.remote_build(source, c.tmp / 'missing-A', c.tmp / 'missing-B', spec=c.spec)


def test_geometry_upload_refuses_changed_staging_before_remote_writes(publication, monkeypatch):
    import scripts.pick_and_place_v2 as P
    from ops.remote import access
    c = publication
    G.fetch(c.spec, c.client)
    write(c.p['geometry_outcomes_local'], {'changed': True})
    monkeypatch.setattr(access, 'RemoteClient', lambda: pytest.fail('must refuse before connecting for upload'))
    with pytest.raises(ValueError, match='staging differs'):
        P.upload(c.spec, None, ['--view', 'geometry'])
