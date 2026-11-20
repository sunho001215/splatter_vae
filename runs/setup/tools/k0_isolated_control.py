"""Claude-only exact-nine authorization/pause publication. Preview unless --execute.

Uses K's existing atomic pause publication helper/lock order and the deployed
admission_pause schema/readers. Never edits queue, ownership, registry or tests.
"""
import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k0_isolated_release as release


def revision(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def pause_document(raw, active):
    doc = json.loads(raw)
    matches = [p for p in doc['pauses'] if p.get('name') == 'k0-20261120-held']
    if (len(matches) != 1 or set(matches[0].get('ids', [])) != release.IDS
            or matches[0].get('prefixes')):
        raise ValueError('expected exact existing nine-ID k0 pause')
    matches[0]['active'] = active
    return (json.dumps(doc, indent=2) + '\n').encode()


def plan(mode, candidate, root=release.MAIN):
    import yaml
    target = root / 'runs/setup/k0-isolated-release.json'
    if mode == 'activate':
        grant, snapshot, _ = release.load_authorization(candidate)
    else:
        # Revocation must remain available even when source/evidence validation fails.
        grant = json.loads(Path(candidate).read_bytes())
        snapshot = json.loads(Path(grant['snapshot']).read_bytes())
        if set(grant.get('rows', {})) != release.IDS or grant.get('commit') != release.COMMIT:
            raise ValueError('rollback requires the same exact-nine grant')
    queue_path = root / 'experiments/queue.yaml'
    queue_raw = queue_path.read_bytes()
    rows = {r['id']: r for r in yaml.load(queue_raw, Loader=yaml.CSafeLoader)['jobs']}
    for identity in release.IDS:
        release.check_row(rows[identity], grant)
    pauses = root / 'runs/setup/admission-pauses.json'
    raw = pauses.read_bytes()
    after = pause_document(raw, mode == 'rollback')
    registry = root / 'experiments/registry.jsonl'
    token = revision(registry)
    if mode == 'activate':
        # All parsing precedes all shared locks. First publication only: retries
        # thereafter use this same grant; an activation never re-pins started rows.
        for line in registry.open('rb'):
            if b'rwk0-20261120-' in line:
                event = json.loads(line)
                if event.get('id') in release.IDS:
                    raise ValueError('activation requires never-launched exact-nine rows')
    if revision(registry) != token:
        raise ValueError('registry changed during preparation; retry later')
    if target.exists() and target.read_bytes() != Path(candidate).read_bytes():
        raise ValueError('refuse replacement of an existing different isolated grant')
    return dict(target=target, candidate=Path(candidate).read_bytes(), queue=queue_path,
                queue_raw=queue_raw, pauses=pauses, pause_raw=raw, after=after,
                registry=registry, token=token, snapshot=snapshot)


def apply(mode, prepared, root=release.MAIN):
    # Reuse the already-reviewed K registrar's exact atomic writer and NB lock order.
    source = Path(prepared['snapshot']['code_root']) / 'scripts/k0_register.py'
    spec = importlib.util.spec_from_file_location('_k0_registration_atomic', source)
    existing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(existing)
    with contextlib.ExitStack() as stack:
        for name in existing.LOCKS:
            lock = stack.enter_context((root / name).open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (prepared['queue'].read_bytes() != prepared['queue_raw']
                or prepared['pauses'].read_bytes() != prepared['pause_raw']
                or revision(prepared['registry']) != prepared['token']):
            raise ValueError('shared evidence changed; retry preview later')
        if prepared['target'].exists() and prepared['target'].read_bytes() != prepared['candidate']:
            raise ValueError('isolated grant changed; refusing pause publication')
        if mode == 'activate':
            if not prepared['target'].exists():
                existing.atomic(prepared['target'], prepared['candidate'])
            existing.atomic(prepared['pauses'], prepared['after'])  # release original named pause LAST
        else:
            existing.atomic(prepared['pauses'], prepared['after'])  # hold exact nine FIRST
            prepared['target'].unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('activate', 'rollback'))
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    prepared = plan(args.mode, args.candidate)
    print(json.dumps({'mode': args.mode, 'ids': sorted(release.IDS),
                      'pause_active_after': args.mode == 'rollback', 'execute': args.execute}))
    if args.execute:
        apply(args.mode, prepared)


if __name__ == '__main__':
    main()
