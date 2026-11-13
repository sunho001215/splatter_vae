"""Read-only adoption audit for every live queue row with a local placement scope.

Uses migration's import-closure algorithm, config/environment scope and operational
exceptions (the same scope realworld_local.placement_scope uses). Checks both the
recorded hashes and newly imported files within the adoption delta, so a changed
import cannot evade the audit. Historical mismatches outside that delta are not
introduced by adoption and are outside this report.
No campaign/scheduler entry module is imported; queue, registry and ownership files
are only read. Run after committing with ``python -m ops.jobs.closure_impact``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import yaml

from ops.jobs import migration
from ops.provenance.upstreams import commit_identity


def scopes_in(value):
    if isinstance(value, dict):
        if isinstance(value.get('files'), dict) and value.get('source_commit'):
            yield value
        elif isinstance(value.get('scope_files'), dict) and value.get('source_commit'):
            yield {'source_commit': value['source_commit'],
                   'files': {**value['scope_files'], **value.get('scope_third_party', {})}}
        for child in value.values():
            yield from scopes_in(child)
    elif isinstance(value, list):
        for child in value:
            yield from scopes_in(child)


def audit(root, live_root, revision='HEAD', *, base=None, preserved_commit=None):
    root, live_root = Path(root), Path(live_root)
    queue = yaml.safe_load((live_root / 'experiments/queue.yaml').read_text())
    # Include recorded launch inputs as well as ownership placement, including retries.
    events = {}
    for line in (live_root / 'experiments/registry.jsonl').read_text().splitlines():
        if line.strip():
            event = json.loads(line)
            events.setdefault(event.get('id'), []).append(event)
    target_read, target_listed = migration.commit_reader(root, revision)
    target_read = lru_cache(None)(target_read)
    target_tp = commit_identity(root, revision)
    # Audit the adoption delta, not unrelated historical drift between the live
    # placement's old source release and this checkout. The owning coordinator may
    # have changed other files independently. A separately owned, preserved commit
    # is excluded only after proving every one of its files is byte-identical here.
    base = base or migration._git(root, 'merge-base', revision, '009bf0c').decode().strip()
    changed = set(migration._git(root, 'diff', '--name-only', base, revision).decode().splitlines())
    base_tp = commit_identity(root, base)
    changed |= {key for key in set(base_tp) | set(target_tp) if base_tp.get(key) != target_tp.get(key)}
    preserved = set()
    if preserved_commit:
        preserved = set(migration._git(root, 'diff-tree', '--root', '--no-commit-id', '--name-only', '-r',
                                        preserved_commit).decode().splitlines())
        preserved_read, _ = migration.commit_reader(root, preserved_commit)
        if any(target_read(p) != preserved_read(p) for p in preserved):
            raise ValueError('HEAD modified a file of the separately preserved commit')
        changed -= preserved

    @lru_cache(None)
    def source(commit):
        read, listed = migration.commit_reader(root, commit)
        return lru_cache(None)(read), listed, commit_identity(root, commit)

    @lru_cache(None)
    def closure(script, commit):
        read, listed, pins = source(commit)
        paths = migration.import_closure([script], read) | migration.import_closure([script], target_read)
        paths |= {p for p in listed + target_listed if p.startswith(migration.CONFIG_SCOPE)}
        paths |= set(migration.ENVIRONMENT_SCOPE) | set(pins) | set(target_tp)
        return paths - migration.OPERATIONAL_ALLOWED

    rows = []
    for job in queue['jobs']:
        owned = live_root / 'runs/setup/ownership' / (job['id'] + '.json')
        placement = json.loads(owned.read_text()) if owned.is_file() else {}
        records = events.get(job['id'], [])
        # Ownership is current placement evidence; registry input scopes cover
        # placements predating ownership. Do not mistake superseded history for
        # the current placement contract.
        scopes = list(scopes_in(placement.get('execution_scope') or placement.get('destination') or {}))
        if not scopes:
            scopes = next((found for record in reversed(records) if (found := list(scopes_in(record)))), [])
        if job.get('host', 'local') != 'local' or not scopes:
            continue
        differences, commits, closure_files_changed = set(), set(), set()
        for scope in scopes:
            commit = scope['source_commit']
            commits.add(commit)
            paths = closure(job['script'], commit) | set(scope['files'])
            closure_files_changed |= (paths & changed) - migration.OPERATIONAL_ALLOWED
            for name in (paths & changed) - migration.OPERATIONAL_ALLOWED:
                digest = scope['files'].get(name)
                data = target_read(name)
                actual = target_tp.get(name) if name in target_tp else (
                    hashlib.sha256(data).hexdigest() if data is not None else None)
                if name not in migration.OPERATIONAL_ALLOWED and actual != digest:
                    differences.add(name)
        rows.append({'id': job['id'], 'source_commits': sorted(commits),
                     'affected': bool(differences), 'changed_closure_files': sorted(closure_files_changed),
                     'differences': sorted(differences)})
    affected = sum(row['affected'] for row in rows)
    return {'revision': revision, 'commit': migration._git(root, 'rev-parse', revision).decode().strip(),
            'queue': str(live_root / 'experiments/queue.yaml'),
            'registry': str(live_root / 'experiments/registry.jsonl'), 'base': base,
            'preserved_commit': preserved_commit, 'preserved_files': sorted(preserved),
            'adoption_files': sorted(changed), 'rows_checked': len(rows),
            'affected_rows': affected, 'result': 'no existing row affected' if not affected else 'existing rows affected',
            'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--live-root', type=Path, default=Path('/home/ws/ws/splatter4d'))
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--base', default='009bf0c', help='common checkout before this implementation')
    parser.add_argument('--preserved-commit', help='separately owned commit whose files must remain byte-identical')
    args = parser.parse_args()
    result = audit(args.root, args.live_root, args.revision, base=args.base, preserved_commit=args.preserved_commit)
    print(json.dumps(result, indent=2))
    return int(bool(result['affected_rows']))


if __name__ == '__main__':
    raise SystemExit(main())
