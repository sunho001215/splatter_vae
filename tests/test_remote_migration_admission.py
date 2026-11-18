"""Remote admission of started local-release rows migrated from a checkpoint pause (ops/jobs/remote_migration.py) and the
remote driver's candidate filter. CPU only; every file lives under tmp_path; no campaign module is imported or run."""
from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest

from ops.jobs import remote_migration as rm
from ops.jobs import scheduler

REPO = Path(__file__).resolve().parents[1]
JOB = 'rwv2-20261113-pre-snerl-drawer-open-v2-S'
SOURCE, RELEASE = '1' * 40, '2' * 40
IMAGE = 'sha256:' + '3' * 64
MANIFEST = f'runs/remote/acceptance/{RELEASE[:7]}-cc/manifest.json'
SCRIPT = 'scripts/dsreset_train_snerl.py'
SCOPE = {SCRIPT: 'a' * 64, 's4d/baselines/snerl/model.py': 'b' * 64}
THIRD = {'third_party/snerl@commit': 'c' * 40, '.gitmodules': 'd' * 64}


def _row(**changes):
    row = dict(id=JOB, host='remote', pin_host='remote', gpu='any', script=SCRIPT, role='realworld_pretrain',
               args=['--config', 'configs/realworld/baselines/snerl/drawer-open-v2.yaml', '--name', JOB,
                     '--output-root', 'runs', '--resume', 'auto'], required_results=['encoder.pt', 'metrics.jsonl'],
               release_source='local', planned_commit='4' * 40, planned_image=IMAGE, mem_gb=16, ram_gb=20)
    return {**row, **changes}


@pytest.fixture
def migrated(tmp_path):
    """A row paused locally at step 250000 and moved to the remote by a recorded migration."""
    run = tmp_path / 'runs' / JOB
    (run / 'checkpoints').mkdir(parents=True)
    (run / 'checkpoints/latest.pt').write_bytes(b'checkpoint at 250000')
    (run / 'config.yaml').write_text('train: {}\n')
    job = _row()
    fingerprint = scheduler.job_fingerprint(job)
    registry = tmp_path / 'registry.jsonl'
    latest = run / 'checkpoints/latest.pt'
    digest = rm.hashlib.sha256(latest.read_bytes()).hexdigest()
    scheduler.record(dict(event='launched', id=JOB, host='local', attempt=1, pid=1, job_sha256=fingerprint,
                          commit=SOURCE), registry)
    scheduler.record(dict(event='checkpoint_paused', id=JOB, host='local', attempt=1, intentional_stop=True,
                          checkpoint=f'runs/{JOB}/checkpoints/latest.pt', checkpoint_step=250000,
                          checkpoint_sha256=digest, reason='GPU 6/7 rollback'), registry)
    scheduler.record(dict(event='exited', id=JOB, host='local', attempt=1, code=-1), registry)
    scheduler.record(dict(event='migrated', id=JOB, from_host='local', host='remote', checkpoint_step=250000,
                          checkpoint_sha256=digest, from_commit=SOURCE, commit=RELEASE, image_digest=IMAGE,
                          sources_identical=True), registry)
    state = scheduler.read_registry(registry)
    member = dict(launched=True, attempt=1, commit=SOURCE, job_sha256=fingerprint, run_dir=f'runs/{JOB}',
                  stop=state[JOB]['intentional_stops'][-1], container_id=None)
    manifest = {'files': {f'runs/{JOB}/checkpoints/latest.pt': dict(sha256=digest, bytes=latest.stat().st_size),
                          f'runs/{JOB}/config.yaml': dict(sha256='e' * 64, bytes=(run / 'config.yaml').stat().st_size)},
                'sha256': 'f' * 64}
    record = dict(job_id=JOB, kind='migration', unit=[JOB], state='switched', owner_host='remote', from_host='local',
                  to_host='remote', member=member, checkpoint_step=250000, manifest=manifest,
                  destination=dict(commit=RELEASE, image_digest=IMAGE, scope_files=dict(SCOPE),
                                   scope_third_party=dict(THIRD), scripts=[SCRIPT], source_commit=SOURCE))
    calls = []

    def accept(commit, image, manifest_path, root):
        calls.append((commit, image, manifest_path))

    fakes = dict(releases={(RELEASE, IMAGE): MANIFEST}, accept=accept,
                 digests=lambda root, commit, paths: {p: SCOPE.get(p, 'absent') for p in paths},
                 pins=lambda root, commit: dict(THIRD))
    return dict(root=tmp_path, job=job, state=state, ownership={JOB: record}, record=record, fakes=fakes,
                calls=calls, latest=latest)


def _check(f, job=None, ownership=None, current=None, **fakes):
    return rm.check(job or f['job'], f['ownership'] if ownership is None else ownership,
                    f['state'][JOB] if current is None else current, root=f['root'], **{**f['fakes'], **fakes})


def test_switched_migrated_row_is_admitted_on_its_accepted_release(migrated):
    assert _check(migrated) == ((RELEASE, IMAGE), MANIFEST)
    assert migrated['calls'] == [(RELEASE, IMAGE, MANIFEST)]
    pins, refused = rm.admitted_pins([migrated['job']], migrated['ownership'], migrated['state'],
                                     migrated['root'], **migrated['fakes'])
    assert pins == {JOB: ((RELEASE, IMAGE), MANIFEST)} and refused == {}
    record = {**migrated['record'], 'state': 'verified'}
    assert _check(migrated, ownership={JOB: record})[0] == (RELEASE, IMAGE)


def _without(mapping, key):
    return {k: v for k, v in mapping.items() if k != key}


@pytest.mark.parametrize('change, reason', [
    (lambda f: dict(ownership={}), 'no migration ownership record'),
    (lambda f: dict(ownership={JOB: {**f['record'], 'kind': 'role_placement'}}), 'no migration ownership record'),
    (lambda f: dict(ownership={JOB: {**f['record'], 'state': 'migrating'}}), 'switched/verified required'),
    (lambda f: dict(ownership={JOB: {**f['record'], 'state': 'aborted', 'owner_host': 'local'}}), 'switched/verified'),
    (lambda f: dict(ownership={JOB: {**f['record'], 'owner_host': 'local'}}), 'remote ownership'),
    (lambda f: dict(job=_row(host='local')), 'not on the remote'),
    (lambda f: dict(current={**f['state'][JOB], 'host': 'local'}), 'migrated event'),
    (lambda f: dict(current=_without(f['state'][JOB], 'migrated_from')), 'migrated event'),
    (lambda f: dict(current={**f['state'][JOB], 'migrated_from': {'host': 'local', 'attempt': 1,
                                                                   'checkpoint_step': 240000}}), 'migrated event'),
    (lambda f: dict(current={**f['state'][JOB], 'commit': SOURCE}), 'destination release'),
    (lambda f: dict(current={**f['state'][JOB], 'repinned_from': {'commit': SOURCE}}), 're-pinned'),
    (lambda f: dict(job=_row(args=['--changed'])), 'job fingerprint'),
    (lambda f: dict(releases={}), 'not a role release'),
    (lambda f: dict(releases={(RELEASE, 'sha256:' + '9' * 64): MANIFEST}), 'not a role release'),
    (lambda f: dict(accept=lambda *a: (_ for _ in ()).throw(ValueError('suite not accepted'))), 'acceptance invalid'),
    (lambda f: dict(digests=lambda root, commit, paths: {**{p: SCOPE[p] for p in paths},
                                                         SCRIPT: '0' * 64} if commit == RELEASE else dict(SCOPE)),
     'destination release 2222222'),
    (lambda f: dict(digests=lambda root, commit, paths: {**{p: SCOPE[p] for p in paths},
                                                         SCRIPT: '0' * 64} if commit == SOURCE else dict(SCOPE)),
     'launch commit 1111111'),
    (lambda f: dict(digests=lambda root, commit, paths: {p: 'absent' for p in paths}), 'differs from the execution'),
    (lambda f: dict(pins=lambda root, commit: {**THIRD, 'third_party/snerl@commit': '5' * 40}),
     'differs from the execution'),
    (lambda f: dict(ownership={JOB: {**f['record'], 'destination': {**f['record']['destination'],
                                                                     'scripts': ['scripts/train_snerl.py']}}}),
     'another script'),
    (lambda f: dict(ownership={JOB: {**f['record'], 'destination': {**f['record']['destination'],
                                                                     'scope_files': {}}}}), 'no execution scope'),
])
def test_refused_migrations(migrated, change, reason):
    with pytest.raises(rm.Refused, match=reason):
        _check(migrated, **change(migrated))


def test_transferred_checkpoint_change_refuses_before_the_first_remote_attempt(migrated):
    migrated['latest'].write_bytes(b'checkpoint at 250000 but different')
    with pytest.raises(rm.Refused, match='transferred state'):
        _check(migrated)


def test_admitted_pins_skips_rows_without_a_remote_migration_and_fails_closed(migrated):
    other = _row(id='rwv2-20261113-pre-sincro-drawer-open-v2-S')
    placed = {'job_id': other['id'], 'kind': 'role_placement', 'state': 'placed', 'owner_host': 'local'}
    ownership = {**migrated['ownership'], other['id']: placed}

    def broken(root, commit, paths):
        raise subprocess.CalledProcessError(128, 'git')

    pins, refused = rm.admitted_pins([migrated['job'], other], ownership, migrated['state'], migrated['root'],
                                     **{**migrated['fakes'], 'digests': broken})
    assert pins == {} and list(refused) == [JOB] and 'CalledProcessError' in refused[JOB]


def test_role_releases_map_current_and_previous_entries(tmp_path):
    (tmp_path / 'runs/setup').mkdir(parents=True)
    (tmp_path / 'runs/setup/role-release.json').write_text(json.dumps(
        {'commit': '6' * 40, 'image': IMAGE, 'previous': [{'commit': RELEASE, 'image': IMAGE}]}))
    assert rm.role_releases(tmp_path) == {('6' * 40, IMAGE): 'runs/remote/acceptance/6666666-cc/manifest.json',
                                          (RELEASE, IMAGE): MANIFEST}


def test_commit_digests_read_git_objects(tmp_path):
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], check=True, capture_output=True, text=True).stdout

    git('init', '-q')
    git('config', 'user.email', 't@t')
    git('config', 'user.name', 't')
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/a.py').write_text('print(1)\n')
    (tmp_path / 'b b.txt').write_bytes(b'\x00binary\nlines\n')
    git('add', '-A')
    git('commit', '-q', '-m', 'one')
    first = git('rev-parse', 'HEAD').strip()
    (tmp_path / 'scripts/a.py').write_text('print(2)\n')
    git('commit', '-q', '-am', 'two')
    second = git('rev-parse', 'HEAD').strip()
    sha = lambda data: rm.hashlib.sha256(data).hexdigest()  # noqa: E731
    found = rm.commit_digests(tmp_path, first, ['scripts/a.py', 'b b.txt', 'missing.py', 'scripts'])
    assert found == {'scripts/a.py': sha(b'print(1)\n'), 'b b.txt': sha(b'\x00binary\nlines\n'),
                     'missing.py': 'absent', 'scripts': 'absent'}
    assert rm.commit_digests(tmp_path, second, ['scripts/a.py'])['scripts/a.py'] == sha(b'print(2)\n')


def _driver():
    return ast.parse((REPO / 'ops/campaign/remote_campaign_driver.py').read_text())


def test_driver_offers_local_release_rows_only_with_an_offload_or_migration_pin():
    tree = _driver()
    filters = [n for n in tree.body if isinstance(n, ast.Assign) and [ast.unparse(t) for t in n.targets] == ['candidates']
               and 'release_source' in ast.unparse(n.value)]
    assert len(filters) == 1
    expression = compile(ast.Expression(body=filters[0].value), '<driver candidate filter>', 'eval')
    rows = [dict(id='never-launched-local', release_source='local'), dict(id='offloaded', release_source='local'),
            dict(id='migrated', release_source='local'), dict(id='role-row', release_source='role'),
            dict(id='plain-row')]
    kept = eval(expression, {'candidates': rows, 'OFFLOAD_PINS': {'offloaded': 'pin'},  # noqa: S307 (driver source)
                             'MIGRATED_PINS': {'migrated': 'pin'}})
    assert [r['id'] for r in kept] == ['offloaded', 'migrated', 'role-row', 'plain-row']
    kept = eval(expression, {'candidates': rows, 'OFFLOAD_PINS': {}, 'MIGRATED_PINS': {}})  # noqa: S307
    assert [r['id'] for r in kept] == ['role-row', 'plain-row']


def test_driver_takes_migration_pins_from_remote_migration_only():
    assigned = {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in _driver().body
                if isinstance(n, ast.Assign) and 'MIGRATED_PINS' in ast.unparse(n.targets[0])}
    assert assigned == {'(MIGRATED_PINS, MIGRATION_REFUSED)':
                        "remote_migration.admitted_pins(queue['jobs'], OWNERSHIP, state, REPO)"}


def test_scheduler_and_role_modules_do_not_import_remote_migration():
    """Placed rows' execution scopes contain the scheduler / role modules' imports; this module stays outside them."""
    for name in ('ops/jobs/scheduler.py', 'ops/jobs/host_roles.py', 'ops/jobs/remote_offload.py',
                 'ops/jobs/migration.py', 'ops/jobs/realworld_local.py'):
        assert 'remote_migration' not in (REPO / name).read_text(), name
