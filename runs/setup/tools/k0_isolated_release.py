"""Exact-nine local source/targeted-acceptance adapter, outside the suite source map.

Imported only by fresh local driver wrappers. No scheduler daemon adoption, queue
mutation, ownership rewrite, reservation or admission-policy change. Scientific
entrypoints run byte-identical ffedec5; this operational adapter is separately
hash-pinned in the authorization and recorded in each attempt's provenance.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys

MAIN = Path('/home/ws/ws/splatter4d')
AUTHORIZATION = MAIN / 'runs/setup/k0-isolated-release.json'
COMMIT = 'ffedec5464ba10b013f5071603b7bd6bc517b937'
IDS = frozenset(f'rwk0-20261120-{stage}-ours-{task}-S-orig-k0-400k'
                for task in ('drawer-open-v2', 'covering', 'pick-and-place-v2')
                for stage in ('pre', 'export', 'eval'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def regular(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or not path.is_file():
        raise ValueError(f'not an absolute regular symlink-free file: {path}')
    return path.read_bytes()


def load_authorization(path=AUTHORIZATION):
    """All expensive hashing happens before a driver's existing admission lock."""
    raw = regular(path)
    grant = json.loads(raw)
    if (grant.get('kind') != 'k0-isolated-targeted-v1' or grant.get('active') is not True
            or grant.get('commit') != COMMIT or set(grant.get('rows', {})) != IDS
            or grant.get('adapter_sha256') != digest(regular(Path(__file__).resolve()))
            or grant.get('worker_sha256') != digest(regular(Path(__file__).resolve().with_name('k0_isolated_worker.py')))):
        raise ValueError('invalid exact-nine isolated authorization')
    snapshot_path = Path(grant['snapshot'])
    snapshot_raw = regular(snapshot_path)
    if digest(snapshot_raw) != grant['snapshot_sha256']:
        raise ValueError('snapshot manifest changed')
    snapshot = json.loads(snapshot_raw)
    root = Path(snapshot['code_root'])
    if root.resolve() != root or root.stat().st_mode & 0o222 or snapshot['commit'] != COMMIT:
        raise ValueError('isolated source root is not sealed ffedec5')
    for name, spec in snapshot['source_files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('source path escapes snapshot')
        source = root / relative
        if source.stat().st_mode & 0o222 or digest(regular(source)) != spec['sha256']:
            raise ValueError(f'isolated source changed: {name}')
    inventory_raw = regular(Path(grant['targeted_evidence']))
    if digest(inventory_raw) != grant['targeted_evidence_sha256']:
        raise ValueError('targeted acceptance inventory changed')
    inventory = json.loads(inventory_raw)
    if (set(inventory['ids']) != IDS or inventory['commit'] != COMMIT
            or inventory['targeted_scientific_evidence_verified'] is not True
            or inventory['native']['optimizer_updates'] != 120
            or inventory['native']['finite_losses_and_gradients'] is not True):
        raise ValueError('targeted native evidence is incomplete')
    for name, expected in inventory['native']['artifacts_sha256'].items():
        if digest(regular(Path(name))) != expected:
            raise ValueError('native evidence changed')
    for name, spec in inventory['inherited_targeted_logs'].items():
        if digest(regular(Path(name))) != spec['sha256']:
            raise ValueError('K targeted evidence changed')
    for name, expected in inventory.get('k2_cpu_artifacts_sha256', {}).items():
        if digest(regular(Path(name))) != expected:
            raise ValueError('K2 CPU evidence changed')
    for identity, spec in grant['rows'].items():
        row = spec['row']
        if (row['id'] != identity or row['priority'] != 5 or row['host'] != 'local'
                or row['planned_commit'] != COMMIT or row.get('release_source') != 'local'
                or row.get('local_placement_pause') != 'k0-20261120-held'
                or spec != inventory['rows'][identity]
                or digest(canonical(row)) != spec['row_sha256']):
            raise ValueError('row authorization differs from registered identity')
    return grant, snapshot, digest(raw)


def check_row(job, grant):
    """A dispatch may bind gpu=any to one instance; nothing else may change."""
    original = grant['rows'][job['id']]['row']
    normalized = dict(job)
    if original.get('gpu') == 'any':
        normalized['gpu'] = 'any'
    if normalized != original:
        raise ValueError(f"{job['id']}: registered row changed")


def fence(path, expected):
    """Cheap revocation/identity fence before localization and launch, no source walk."""
    if digest(regular(path)) != expected:
        raise ValueError('isolated authorization changed or revoked')


def configure_worker_temp(identity):
    """After _bootstrap: short AF_UNIX paths, private to this complete row ID.

    The bootstrap deliberately replaces TMPDIR with its long cache path. Override
    only socket-bearing temp locations after that import; keep all caches intact.
    The full-ID marker rejects truncated-hash collisions and pre-created paths.
    """
    import multiprocessing
    import tempfile
    if identity not in IDS:
        raise ValueError('temp directory requires an exact authorized ID')
    directory = Path('/tmp') / ('k0i-' + digest(identity.encode())[:10])
    try:
        directory.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        created = False
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError('unsafe isolated temp directory owner or mode')
        flags = os.O_NOFOLLOW | (os.O_WRONLY | os.O_CREAT | os.O_EXCL if created else os.O_RDONLY)
        marker = os.open('row-id', flags, 0o600, dir_fd=fd)
        with os.fdopen(marker, 'wb' if created else 'rb') as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise ValueError('unsafe isolated temp identity marker')
            expected = (identity + '\n').encode()
            if created:
                stream.write(expected)
            elif stream.read(len(expected) + 1) != expected:
                raise ValueError('isolated temp directory belongs to another row')
    finally:
        os.close(fd)
    # tempfile can have cached the bootstrap's path during earlier imports.
    for name in ('TMPDIR', 'TEMP', 'TMP', 'XDG_RUNTIME_DIR'):
        os.environ[name] = str(directory)
    tempfile.tempdir = str(directory)
    multiprocessing.current_process()._config.pop('tempdir', None)
    return directory


def install(path=AUTHORIZATION):
    # Default is a literal no-op, including no imports of production modules.
    if not Path(path).exists():
        return False
    try:
        grant, snapshot, authorization_sha = load_authorization(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # A bad optional grant must not hold other drivers' rows.
        print(f'K0_ISOLATED unavailable: {type(exc).__name__}', file=sys.stderr)
        return False
    from ops.jobs import scheduler, realworld_local
    if getattr(scheduler, '_k0_isolated_installed', False):
        return True
    root = Path(snapshot['code_root'])
    original_localize, original_launch, original_record = (
        realworld_local.localize, scheduler.launch, scheduler.record)
    original_arguments = realworld_local._arguments
    source = {name: spec['sha256'] for name, spec in snapshot['source_files'].items()}
    execution = {'source_commit': COMMIT, 'files': source, 'source_root': str(root)}

    def arguments(job):
        # The legacy evaluator parser recognizes only baseline/matrix config
        # names. Authorize this exact sealed k0 config without widening its grammar
        # for anyone else. All dependency/output/override validation still runs.
        if job['id'] not in IDS or '-eval-ours-' not in job['id']:
            return original_arguments(job)
        expected = grant['rows'][job['id']]['row']
        if job.get('args') != expected['args']:
            raise realworld_local.LocalInputError('exact k0 evaluation arguments changed')
        argv = list(job['args'])
        index = argv.index('--config') + 1
        config = argv[index]
        task = expected['realworld_target']
        if config != f'configs/realworld/k0/{task}-S-orig-k0-400k.yaml':
            raise realworld_local.LocalInputError('unapproved isolated evaluator config')
        argv[index] = f'configs/realworld/ours_matrix/{task}-S-orig-scratch400k.yaml'
        parsed = original_arguments({**job, 'args': argv})
        parsed.config = [config]
        return parsed

    def localize(job, state, **kwargs):
        if job['id'] not in IDS:
            return original_localize(job, state, **kwargs)
        try:
            check_row(job, grant)
            fence(path, authorization_sha)
            if not realworld_local.applies(job, kwargs['roles'], kwargs.get('by_id')):
                raise realworld_local.LocalInputError('existing local resolver/role policy refuses row')
            # Preserve the entire existing input/dependency/closeout resolver. Only
            # repository configuration lookup uses the authorized isolated root.
            effective, record = realworld_local._inputs(
                job, state, Path(kwargs.get('mirror') or realworld_local.MIRROR),
                Path(kwargs.get('manifest') or realworld_local.MANIFEST),
                Path(kwargs.get('runs') or MAIN / 'runs'),
                kwargs.get('remote_prefix', realworld_local.REMOTE_PREFIX), root,
                Path(kwargs.get('finished_dir') or realworld_local.FINISHED_DIR))
            return effective, {**record, 'execution_scope': execution,
                               'isolated_authorization_sha256': authorization_sha}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise realworld_local.LocalInputError(str(exc)) from exc

    def launch(job, gpu, attempt, *args, **kwargs):
        if job['id'] not in IDS:
            return original_launch(job, gpu, attempt, *args, **kwargs)
        check_row(job, grant)
        fence(path, authorization_sha)
        effective = kwargs.pop('args', job.get('args', []))
        wrapped = {**job, 'script': 'runs/setup/tools/k0_isolated_worker.py'}
        return original_launch(wrapped, gpu, attempt, *args, args=[
            str(path), job['id'], *map(str, effective)], **kwargs)

    def record(event, *args, **kwargs):
        if event.get('id') in IDS and event.get('event') == 'launched' and event.get('host') == 'local':
            event = {**event, 'commit': COMMIT, 'source_sha256': source,
                     'isolated_source_root': str(root), 'acceptance_kind': grant['kind'],
                     'isolated_authorization_sha256': authorization_sha,
                     'execution_adapter_sha256': grant['adapter_sha256']}
        return original_record(event, *args, **kwargs)

    realworld_local.localize = localize
    realworld_local._arguments = arguments
    scheduler.launch, scheduler.record = launch, record
    scheduler._k0_isolated_installed = True
    return True


def prepare_worker(path, identity, arguments):
    """Fresh isolated process: immutable science + separately pinned operational gate."""
    grant, snapshot, authorization_sha = load_authorization(path)
    if identity not in IDS:
        raise ValueError('worker requires one exact authorized ID')
    row = grant['rows'][identity]['row']
    root = Path(snapshot['code_root'])
    mirror = str(MAIN / 'runs/realworld-inputs')
    prefix = '/home/compu/kaist/sunho/realworld'
    # Match precisely the existing local resolver's path-only substitution.
    # Source paths are made absolute below; numeric/scientific arguments remain unchanged.
    expected = []
    for arg in row['args']:
        value = str(arg).partition('=')[2] if '=' in str(arg) else str(arg)
        # _inputs maps REMOTE_REALWORLD_ROOT, validated again against the isolated constant below.
        expected.append(str(arg)[:len(str(arg)) - len(value)] + mirror + value[len(prefix):]
                        if value == prefix or value.startswith(prefix + '/') else str(arg))
    sys.path[:0] = [str(root / 'scripts'), str(root)]
    os.chdir(MAIN)
    os.environ['S4D_HOST_CONFIG'] = str(root / 'configs/hosts/local.yaml')
    os.environ['S4D_CACHE_ROOT'] = str(MAIN / '.cache/k0-isolated' / identity)
    from ops.resources import gpu_guard
    from s4d.rwlearn.configs import REMOTE_REALWORLD_ROOT
    if REMOTE_REALWORLD_ROOT != prefix or list(arguments) != expected:
        raise ValueError('worker arguments differ from exact registered/localized argv')
    original_validate = gpu_guard.validate_runtime_path
    own_output = MAIN / 'runs' / identity
    own_cache = MAIN / '.cache/k0-isolated' / identity

    def validate_runtime_path(candidate, *, repository=None, cache=False):
        resolved = Path(candidate).expanduser().resolve()
        if resolved == own_output or own_output in resolved.parents or resolved == own_cache or own_cache in resolved.parents:
            return original_validate(candidate, repository=MAIN, cache=cache)
        return original_validate(candidate, repository=repository, cache=cache)

    # Source remains read-only. Grant only this row's ordinary main run directory
    # and private cache through the same unchanged host/path validator.
    gpu_guard.validate_runtime_path = validate_runtime_path
    import _bootstrap

    def require_targeted_tests():
        fence(path, authorization_sha)

    _bootstrap.require_passed_tests = require_targeted_tests
    own_output.mkdir(parents=True, exist_ok=True)
    policy = {'science_commit': COMMIT, 'source_root': str(root), 'acceptance_kind': grant['kind'],
              'authorization_sha256': authorization_sha, 'execution_adapter_sha256': grant['adapter_sha256'],
              'execution_worker_sha256': grant['worker_sha256'], 'full_suite_claimed': False}
    from ops.jobs.run_identity import _atomic_json
    _atomic_json(own_output / 'isolated-execution.json', policy)
    argv = [str(root / arg) if arg.startswith('configs/') else arg for arg in arguments]
    script = root / row['script']
    return script, argv
