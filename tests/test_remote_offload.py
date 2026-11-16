"""CPU-only offload policy, immutable publication, races and driver pin gates.

Every mutation uses tmp_path. No campaign module is imported or executed, and
the fake client translates the publisher's commands into a temporary directory.
"""
from __future__ import annotations

import ast
import fcntl
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ops.jobs import host_roles, realworld_local, scheduler
from ops.jobs import remote_offload as off

REPO = Path(__file__).resolve().parents[1]
COMMIT = 'a' * 40
IMAGE = 'sha256:' + 'b' * 64
IDENTITY = 'rwv2-20261113-export-dynamo-covering-S'
DEP = 'rwv2-20261113-pre-dynamo-covering-S'


@pytest.fixture
def placement(tmp_path, monkeypatch):
    root = tmp_path
    for folder in ('experiments', 'runs/setup', 'configs/hosts', 'runs/' + DEP):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / 'configs/hosts/roles.yaml').write_bytes((REPO / 'configs/hosts/roles.yaml').read_bytes())
    source = root / 'runs' / DEP / 'encoder.pt'
    source.write_bytes(b'encoder checkpoint bytes')
    digest = realworld_local._hash(source)
    close = root / 'runs/setup/realworld-local-finished'
    close.mkdir()
    (close / (DEP + '.json')).write_text(json.dumps(dict(job_id=DEP, files={'encoder.pt': digest})))
    job = dict(id=IDENTITY, host='local', pin_host='local', gpu='any', role='realworld_pretrain',
        script=realworld_local.DSRESET_ARTIFACT, args=['--source', f'runs/{DEP}/encoder.pt', '--kind', 'encoder',
        '--out', f'runs/{IDENTITY}/encoder.pt'], deps=[DEP], required_results=['encoder.pt'], priority=1,
        mem_gb=2, ram_gb=8, disk_gb=2, planned_commit=COMMIT, planned_image=IMAGE, release_source='local',
        registration_tool='runs/setup/tools/register_dsreset_20261113.py')
    dep = {**job, 'id': DEP, 'deps': [], 'args': []}
    queue = root / 'experiments/queue.yaml'
    queue.write_text(yaml.safe_dump(dict(hosts={'local': {}, 'remote': dict(enabled=False,
        limits={scheduler.APPROVED_HOST_GPUS['remote'][0]: dict(max_jobs=4, max_mem_gb=22)})}, jobs=[job, dep])))
    registry = root / 'experiments/registry.jsonl'
    scheduler.record(dict(event='launched', id=DEP, attempt=1, host='local'), registry)
    scheduler.record(dict(event='exited', id=DEP, attempt=1, code=0), registry)
    owner = dict(job_id=IDENTITY, kind='role_placement', state='placed', owner_host='local', attempts=0,
                 execution_scope={'source_commit': COMMIT, 'files': {'scripts/example.py': 'c' * 64}})
    host_roles.write_ownership(owner, root / 'runs/setup/ownership')
    evidence_path = root / f'runs/remote/acceptance/{COMMIT[:7]}-cc/manifest.json'
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text('{}')
    artifact = root / 'runs/setup/artifact.json'
    artifact.write_text('{}')
    evidence = {k: dict(path='runs/setup/artifact.json') for k in ('full_suite', 'cross_host', 'data')}
    evidence.update(gpu_isolation={}, manifest_sha256=realworld_local._hash(evidence_path))
    from ops.jobs import remote_backend
    calls = []
    def acceptance(host, repo):
        calls.append(host)
        assert host['commit'] == COMMIT and host['image_digest'] == IMAGE
        return evidence
    monkeypatch.setattr(remote_backend, 'acceptance_manifest', acceptance)
    return SimpleNamespace(root=root, job=job, source=source, digest=digest, owner=owner, calls=calls,
        state=scheduler.read_registry(registry), owned={IDENTITY: owner}, roles=host_roles.load_roles(), queue=queue,
        evidence=evidence)


def prepare(f, **kw):
    return off.prepare(f.job, f.root, f.state, f.owned, f.roles, [], {}, 3600, 60, **kw)


def test_prepare_exact_pin_input_identity_and_no_mutations(placement):
    f = placement
    before = f.queue.read_bytes()
    prepared = prepare(f)
    assert prepared['items'][0]['sha256'] == f.digest
    assert prepared['estimate']['missing_bytes'] == f.source.stat().st_size
    assert f.calls[-1]['commit'] == f.job['planned_commit']
    assert f.queue.read_bytes() == before
    assert not list((f.root / 'runs/setup').glob('*backup*'))


@pytest.mark.parametrize('key', ['evidence', 'provenance', 'closeout', 'launch_spec'])
def test_offload_rechecks_reused_isolation_files_before_placement(placement, key):
    f = placement
    entries = {}
    for name in ('evidence', 'provenance', 'closeout', 'launch_spec'):
        path = f.root / 'runs/setup' / ('reused-' + name + '.json')
        path.write_text('{}')
        entries[name] = {'path': str(path.relative_to(f.root)), 'sha256': realworld_local._hash(path)}
    f.evidence['reused_isolation'] = {scheduler.APPROVED_HOST_GPUS['remote'][0]: entries}
    prepared = prepare(f)
    prepared['remote_verified'] = True
    assert all(str(f.root / ref['path']) in prepared['stats'] for ref in entries.values())
    (f.root / entries[key]['path']).write_text('{"changed": true}')
    with pytest.raises(off.Refused, match='input/acceptance changed'):
        off.commit_placement(prepared, f.root)
    assert scheduler.load_queue(f.queue)['jobs'][0]['host'] == 'local'


@pytest.mark.parametrize('change,reason', [
    ({'id': IDENTITY.replace('-S', '-SF')}, 'approved S'),
    ({'id': IDENTITY.replace('covering', 'mug-hanging')}, 'approved S'),
    ({'registration_tool': 'unknown'}, 'approved S'),
    ({'planned_commit': 'a' * 7}, 'full planned_commit'),
    ({'planned_image': 'latest'}, 'immutable planned_image'),
    ({'continues': DEP}, 'continuation'),
])
def test_reject_unsupported_rows(placement, change, reason):
    placement.job.update(change)
    with pytest.raises(off.Refused, match=reason):
        prepare(placement)


def test_history_output_pause_dependency_and_ownership_gates(placement):
    f = placement
    f.state[IDENTITY] = dict(status='pending', attempts=0)
    with pytest.raises(off.Refused, match='registry history'):
        prepare(f)
    f.state.pop(IDENTITY)
    with pytest.raises(off.Refused, match='pause'):
        off.prepare(f.job, f.root, f.state, f.owned, f.roles,
            [dict(name='cancelled', active=True, ids=[IDENTITY])], {}, 3600, 60)
    (f.root / 'runs' / IDENTITY).mkdir()
    with pytest.raises(off.Refused, match='output exists'):
        prepare(f)
    (f.root / 'runs' / IDENTITY).rmdir()
    f.state[DEP]['status'] = 'running'
    with pytest.raises(off.Refused, match='dependencies'):
        prepare(f)
    f.state[DEP]['status'] = 'done'
    f.owner['owner_host'] = 'remote'
    with pytest.raises(off.Refused, match='ownership'):
        prepare(f)


def test_dependency_checksum_and_rgb_fail_closed(placement):
    f = placement
    f.source.write_bytes(b'changed checkpoint')
    with pytest.raises(off.Refused, match='closeout mismatch'):
        prepare(f)
    f.job['rgb_bundle'] = '/local/only'
    f.job['script'] = 'scripts/train_realworld_fql.py'
    with pytest.raises(off.Refused, match='remote mount contract'):
        prepare(f)


def test_cost_strict_inequality_conflict_cumulative_transfer_and_priority():
    items = [dict(target='/r/a', sha256='x', bytes=60_000_000)]
    assert off.cost(items, {}, 12, 2)['beneficial'] is False
    assert off.cost(items, {}, 13, 2)['beneficial'] is True
    assert off.cost(items, {'/r/a': dict(sha256='x')}, 3, 2)['missing_bytes'] == 0
    assert off.cost(items, {}, 13, 2, queued_bytes=60_000_000)['beneficial'] is False
    with pytest.raises(off.Refused, match='conflict'):
        off.cost(items, {'/r/a': dict(sha256='different')}, 100, 0)
    with pytest.raises(off.Refused, match='invalid'):
        off.cost(items, {}, float('inf'), 0)
    rows = [dict(id='rwv2-20261113-' + name, role=role, script=script, args=[])
            for name, role, script in (
                ('fql-ours-covering-S', 'realworld_fql', 'scripts/train_realworld_fql.py'),
                ('pre-dynamo-covering-S', 'realworld_pretrain', 'scripts/train_dynamo.py'),
                ('pre-ours-covering-S', 'realworld_pretrain', 'scripts/train.py'),
                ('prep-ours-covering-S', 'realworld_preprocess', realworld_local.DSRESET_CHECK))]
    ordered = sorted(rows, key=lambda r: off.priority(r, {}))
    assert ordered == [rows[3], rows[2], rows[1], rows[0]]


class Client:
    """Only local temporary file operations. No network or remote executables."""
    def __init__(self, root):
        self.directory = root
        self.root = off.REMOTE_ROOT
        self.uploads = 0
        self.interrupt = False
        self.race = False

    def path(self, target):
        return self.directory / target.removeprefix(off.REMOTE_ROOT + '/')

    def ensure_remote_path(self, target, create_parent=False):
        if create_parent:
            self.path(target).parent.mkdir(parents=True, exist_ok=True)

    def _remote_checksum(self, target, allow_missing=False):
        path = self.path(target)
        return realworld_local._hash(path) if path.is_file() else None

    def rsync_upload(self, source, target):
        self.uploads += 1
        path = self.path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.interrupt:
            path.write_bytes(b'partial')
            self.interrupt = False
            raise RuntimeError('interrupted')
        shutil.copyfile(source, path)

    def checked_ssh(self, command, **kw):
        if command.startswith('test ! -e '):
            command = command.replace(off.REMOTE_ROOT, str(self.directory))
            command = command.replace("docker ps -a --format '{{.Names}}'", 'true')
            return subprocess.run(['bash', '-c', command], check=True, capture_output=True, text=True)
        assert command.startswith('ln -T -- ') and kw == dict(idempotent=True)
        command = command.replace(off.REMOTE_ROOT, str(self.directory))
        if self.race:
            target = self.directory / 'runs' / DEP / 'encoder.pt'
            target.write_bytes(b'another producer')
        return subprocess.run(['bash', '-c', command], check=True, capture_output=True, text=True)


def test_upload_resume_no_clobber_and_concurrent_publisher(placement, tmp_path):
    f = placement
    item = prepare(f)['items'][0]
    client = Client(tmp_path / 'remote')
    client.interrupt = True
    with pytest.raises(RuntimeError, match='interrupted'):
        off.publish(client, item)
    assert not client.path(item['target']).exists()
    off.publish(client, item)
    off.publish(client, item)
    assert client.uploads == 2 and client.path(item['target']).read_bytes() == f.source.read_bytes()
    client.path(item['target']).write_bytes(b'other user')
    with pytest.raises(off.Refused, match='conflict'):
        off.publish(client, item)
    assert client.path(item['target']).read_bytes() == b'other user'
    client.path(item['target']).unlink()
    client.race = True
    with pytest.raises(off.Refused, match='conflict'):
        off.publish(client, item)
    assert client.path(item['target']).read_bytes() == b'another producer'


def test_commit_short_locks_backup_same_pin_and_driver_gate(placement, monkeypatch):
    f = placement
    prepared = prepare(f)
    prepared['remote_verified'] = True
    monkeypatch.setattr(realworld_local, '_hash', lambda *a: pytest.fail('hash under admission locks'))
    assert off.commit_placement(prepared, f.root) == IDENTITY
    row = scheduler.load_queue(f.queue)['jobs'][0]
    owners = host_roles.read_ownership(f.root / 'runs/setup/ownership')
    assert row['host'] == row['pin_host'] == 'remote'
    assert {k: v for k, v in row.items() if k not in host_roles.HOST_FIELDS} == {
        k: v for k, v in f.job.items() if k not in host_roles.HOST_FIELDS}
    assert off.admitted_pin(row, owners) == ((COMMIT, IMAGE), prepared['evidence_manifest'])
    row['planned_commit'] = 'f' * 40
    assert off.admitted_pin(row, owners) is None
    backups = list((f.root / 'runs/setup').glob('offload-backup-*'))
    assert len(backups) == 1 and (backups[0] / 'queue.yaml').is_file()
    with pytest.raises(off.Refused, match='row changed|row/ownership changed'):
        off.commit_placement(prepared, f.root)


@pytest.mark.parametrize('race', ['launch', 'source', 'pause', 'pin', 'dependency'])
def test_commit_revalidates_after_transfer(placement, race):
    f = placement
    prepared = prepare(f)
    prepared['remote_verified'] = True
    if race == 'launch':
        scheduler.record(dict(event='launched', id=IDENTITY, attempt=1, host='local'),
                         f.root / 'experiments/registry.jsonl')
    elif race == 'source':
        f.source.write_bytes(b'changed during transfer')
    elif race == 'pause':
        (f.root / 'runs/setup/admission-pauses.json').write_text(json.dumps(dict(pauses=[
            dict(name='cancelled', active=True, ids=[IDENTITY])])))
    elif race == 'pin':
        q = yaml.safe_load(f.queue.read_bytes())
        q['jobs'][0]['planned_commit'] = 'd' * 40
        f.queue.write_text(yaml.safe_dump(q))
    else:
        scheduler.record(dict(event='unknown', id=DEP, reason='reconcile'), f.root / 'experiments/registry.jsonl')
    with pytest.raises(off.Refused):
        off.commit_placement(prepared, f.root)
    assert scheduler.load_queue(f.queue)['jobs'][0]['host'] == 'local'


@pytest.mark.parametrize('name', ['stage2-driver.lock', 'remote-driver.lock', '../../experiments/registry.lock'])
def test_busy_admission_locks_leave_placement_untouched(placement, name):
    f = placement
    prepared = prepare(f)
    prepared['remote_verified'] = True
    with (f.root / 'runs/setup' / name).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            off.commit_placement(prepared, f.root)
    assert scheduler.load_queue(f.queue)['jobs'][0]['host'] == 'local'


def test_interrupted_queue_write_fails_closed(placement, monkeypatch):
    f = placement
    prepared = prepare(f)
    prepared['remote_verified'] = True
    def killed(*a):
        raise OSError('simulated crash')
    monkeypatch.setattr(off, 'replace_queue', killed)
    with pytest.raises(OSError, match='crash'):
        off.commit_placement(prepared, f.root)
    owners = host_roles.read_ownership(f.root / 'runs/setup/ownership')
    assert scheduler.load_queue(f.queue)['jobs'][0]['host'] == 'local'
    assert host_roles.ownership_refusal(IDENTITY, 'local', owners)
    assert off.admitted_pin(f.job, owners) is None


def test_driver_release_selection_without_importing_campaign(placement):
    tree = ast.parse((REPO / 'ops/campaign/remote_campaign_driver.py').read_text())
    names = dict(state={}, SNERL_NT_IDS=set(), OFFLOAD_PINS={IDENTITY: ((COMMIT, IMAGE), 'manifest')},
                 kind=lambda _: 'role', ROLE_COMMIT='f' * 40, ROLE_IMAGE=IMAGE)
    for name in ('job_release',):
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<driver function>', 'exec'), names)
    assert names['job_release'](placement.job) == (COMMIT, IMAGE)


@pytest.mark.local_host_data  # runs runs/setup/tools, which the remote sync does not carry
def test_offline_cli_never_constructs_remote_client(placement, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location('offload_cli', REPO / 'runs/setup/tools/offload_remote_20261116.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    from ops.remote.access import RemoteClient
    monkeypatch.setattr(RemoteClient, '__init__', lambda *a, **kw: pytest.fail('network client in preview'))
    assert cli.main(['--repo', str(placement.root), '--local-wait-seconds', '3600',
                     '--remote-start-seconds', '60', IDENTITY]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['decisions'][0]['decision'] == 'would_offload'
    assert scheduler.load_queue(placement.queue)['jobs'][0]['host'] == 'local'


def test_local_scope_revalidation_catches_source_drift(tmp_path):
    script = tmp_path / 'script.py'
    script.write_text('before')
    scope = dict(files={'script.py': realworld_local._hash(script)})
    assert realworld_local.revalidate_placement_scope(scope, root=tmp_path) == scope
    script.write_text('after')
    with pytest.raises(realworld_local.LocalInputError, match='changed'):
        realworld_local.revalidate_placement_scope(scope, root=tmp_path)


def test_fresh_remote_resources_respect_foreign_processes_ram_and_holds(placement):
    from ops.jobs.remote_backend import RemoteSnapshot
    f = placement
    gpu = scheduler.APPROVED_HOST_GPUS['remote'][0]
    snap = RemoteSnapshot(usage={gpu: []}, available_gb=1000,
        disk_total_bytes=10_000 * 2**30, disk_free_bytes=9000 * 2**30,
        containers={}, gpu_memory={gpu: (24 * 1024, 0)}, admission_records=[], total_ram_gb=1500)
    assert off.remote_room(f.job, f.root, snap) == gpu
    snap.usage[gpu] = [(123456, 100)]
    with pytest.raises(off.Refused, match='GPU room'):
        off.remote_room(f.job, f.root, snap)
    snap.usage[gpu] = []
    snap.available_gb = 1
    with pytest.raises(off.Refused, match='RAM/CPU/disk'):
        off.remote_room(f.job, f.root, snap)
    snap.launch_hold = dict(held=True)
    with pytest.raises(off.Refused, match='launch hold'):
        off.remote_room(f.job, f.root, snap)


def test_unrelated_queue_edit_during_staging_is_not_overwritten(placement, monkeypatch):
    f = placement
    prepared = prepare(f)
    prepared['remote_verified'] = True
    original = off.locks
    import contextlib
    @contextlib.contextmanager
    def changed(root):
        with original(root):
            f.queue.write_bytes(f.queue.read_bytes() + b'\n# coordinator edit\n')
            yield
    monkeypatch.setattr(off, 'locks', changed)
    with pytest.raises(off.Refused, match='queue changed'):
        off.commit_placement(prepared, f.root)
    assert f.queue.read_bytes().endswith(b'# coordinator edit\n')
    assert not list((f.root / 'experiments').glob('.offload-queue-*'))


@pytest.mark.local_host_data  # runs runs/setup/tools, which the remote sync does not carry
def test_loop_stop_file_prevents_pass_and_stops_between_passes(tmp_path, monkeypatch):
    stop = tmp_path / 'STOP'
    log = tmp_path / 'passes'
    python = tmp_path / 'fake-python'
    python.write_text('#!/bin/bash\nprintf "pass\\n" >> "$OFFLOAD_TEST_LOG"\ntouch "$OFFLOAD_STOP_FILE"\n')
    python.chmod(0o755)
    monkeypatch.setenv('OFFLOAD_STOP_FILE', str(stop))
    monkeypatch.setenv('OFFLOAD_TEST_LOG', str(log))
    monkeypatch.setenv('OFFLOAD_PYTHON', str(python))
    loop = REPO / 'runs/setup/tools/offload_remote_loop_20261116.sh'
    stop.touch()
    subprocess.run(['bash', str(loop)], check=True, timeout=5)
    assert not log.exists()
    stop.unlink()
    subprocess.run(['bash', str(loop)], check=True, timeout=5)
    assert log.read_text() == 'pass\n'


def test_localized_store_uploads_original_and_complete_episode_closure(placement):
    f = placement
    mirror = f.root / 'runs/realworld-inputs'
    name = 'learning_inputs/test/store.yaml'
    episode = 'processed/test/episode'
    original = yaml.safe_dump(dict(kind='compact', paths=[off.local.REMOTE_PREFIX + '/' + episode])).encode()
    local_store = yaml.safe_dump(dict(kind='compact', paths=[str(mirror / episode)])).encode()
    files = {}
    for relative, data in ((name, local_store), ('_remote/' + name, original),
                           (episode + '/episode.h5', b'episode bytes'), (episode + '/_COMPLETE.json', b'{}')):
        path = mirror / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files[relative] = dict(kind='copy', sha256=realworld_local._hash(path), bytes=len(data))
    files[name].update(kind='localized', remote_sha256=realworld_local._hash(mirror / '_remote' / name),
                       requires=[episode + '/episode.h5', episode + '/_COMPLETE.json'])
    (mirror / 'MANIFEST.json').write_text(json.dumps(dict(version=1, remote_root=off.local.REMOTE_PREFIX,
        local_root='runs/realworld-inputs', files=files)))
    row = {**f.job, 'script': 'scripts/train.py', 'args': ['--config', 'configs/test.yaml', '--name', IDENTITY,
        '--output-root', 'runs', '--set', 'data.regime=realworld', 'data.store=' + off.local.REMOTE_PREFIX + '/' + name],
        'deps': []}
    (f.root / 'configs/test.yaml').write_text('data:\n  regime: realworld\n')
    items = off.input_files(row, f.root, {})
    descriptor = next(i for i in items if i['target'].endswith('/store.yaml'))
    assert Path(descriptor['source']).read_bytes() == original
    assert len(items) == 3
    (f.root / 'configs/test.yaml').write_text('data:\n  regime: realworld\n  hidden: /home/ws/unchecked\n')
    with pytest.raises(off.Refused, match='unchecked/local'):
        off.input_files(row, f.root, {})


def test_historical_cancellation_is_never_offloaded(placement):
    f = placement
    (f.root / 'runs/setup/admission-pauses.json').write_text(json.dumps(dict(pauses=[
        dict(name='paused-by-cancellation', active=False, ids=[IDENTITY])])))
    with pytest.raises(off.Refused, match='pause'):
        off.prepare(f.job, f.root, f.state, f.owned, f.roles, off.pauses(f.root), {}, 3600, 60)


def test_rgb_encoder_artifact_reads_only_its_declared_checkpoint(placement):
    f = placement
    f.job.update(rgb_bundle='/local/only/S', registration_tool='runs/setup/tools/register_drawer_rgb_20261115.py',
                 realworld_target='drawer-open-v2', realworld_variant='S', rgb_method='dynamo')
    items = off.input_files(f.job, f.root, f.state)
    assert len(items) == 1 and items[0]['source'] == str(f.source)


def test_exact_acceptance_required_even_when_role_release_differs(placement):
    f = placement
    path = f.root / f'runs/remote/acceptance/{COMMIT[:7]}-cc/manifest.json'
    path.unlink()
    (f.root / 'runs/setup/role-release.json').write_text(json.dumps(dict(commit='f' * 40, image=IMAGE)))
    with pytest.raises(off.Refused, match='exact remote acceptance missing'):
        prepare(f)


def test_release_reservation_keeps_remote_device_free(placement, monkeypatch):
    from ops.jobs import release_reservation
    from ops.jobs.remote_backend import RemoteSnapshot
    f = placement
    gpu = scheduler.APPROVED_HOST_GPUS['remote'][0]
    snap = RemoteSnapshot(usage={gpu: []}, available_gb=1000,
        disk_total_bytes=10_000 * 2**30, disk_free_bytes=9000 * 2**30,
        containers={}, gpu_memory={gpu: (24 * 1024, 0)}, admission_records=[])
    monkeypatch.setattr(release_reservation, 'load', lambda *a: {})
    monkeypatch.setattr(release_reservation, 'charges', lambda *a: dict(gpus={gpu: dict(mem_gb=2, load=1)},
        ram_gb=10, cpu_threads=4, disk_gb=1))
    with pytest.raises(off.Refused, match='GPU room'):
        off.remote_room(f.job, f.root, snap)


@pytest.mark.local_host_data  # runs runs/setup/tools, which the remote sync does not carry
def test_execute_end_to_end_uses_only_fake_remote_and_never_writes_registry(placement, monkeypatch, capsys):
    from ops.jobs.remote_backend import RemoteBackend, RemoteSnapshot
    from ops.remote import access
    f = placement
    spec = importlib.util.spec_from_file_location('execute_cli', REPO / 'runs/setup/tools/offload_remote_20261116.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, 'MAIN', f.root)
    monkeypatch.setattr(cli, 'SOURCE', f.root)
    monkeypatch.chdir(f.root)
    client = Client(f.root / 'fake-remote')
    monkeypatch.setattr(access, 'RemoteClient', lambda **kw: client)
    gpu = scheduler.APPROVED_HOST_GPUS['remote'][0]
    snap = RemoteSnapshot(usage={gpu: []}, available_gb=1000,
        disk_total_bytes=10_000 * 2**30, disk_free_bytes=9000 * 2**30,
        containers={}, gpu_memory={gpu: (24 * 1024, 0)}, admission_records=[])
    monkeypatch.setattr(RemoteBackend, 'snapshot', lambda *a: snap)
    registry = f.root / 'experiments/registry.jsonl'
    before = registry.read_bytes()
    assert cli.main(['--repo', str(f.root), '--execute', '--local-wait-seconds', '3600',
                     '--remote-start-seconds', '60', IDENTITY]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['decisions'][0]['decision'] == 'offloaded'
    row = scheduler.load_queue(f.queue)['jobs'][0]
    assert row['host'] == 'remote' and row['planned_commit'] == COMMIT
    assert registry.read_bytes() == before
    assert client.uploads == 1 and f.source.read_bytes() == b'encoder checkpoint bytes'
