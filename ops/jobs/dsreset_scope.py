"""Read-only comparison of persisted placement scopes with a candidate checkout.

The baseline comparison distinguishes release regressions from drift already in
adopted main. Absolute compatibility is reported separately and never inferred
from a clean baseline comparison. No queues, ownership records or pins are edited.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

from ops.provenance.upstreams import working_tree_identity

BASELINE = '9c901bf7b0acabe399420c9b5e9a1c2b6bc42479'


def audit(ownership_dir, candidate, baseline=BASELINE):
    candidate = Path(candidate)
    upstreams = working_tree_identity(candidate)

    @lru_cache(None)
    def current(name):
        if name in upstreams:
            return upstreams[name]
        path = candidate/name
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    @lru_cache(None)
    def adopted(name):
        if name.endswith('@commit'):
            args = ['rev-parse', f'{baseline}:{name.removesuffix("@commit")}']
        else:
            args = ['show', f'{baseline}:{name}']
        result = subprocess.run(['git','-C',str(candidate),*args],capture_output=True)
        if result.returncode:
            return None
        return result.stdout.decode().strip() if name.endswith('@commit') else hashlib.sha256(result.stdout).hexdigest()

    ownership = Path(ownership_dir)
    if ownership.is_file():
        snapshot = json.loads(ownership.read_bytes())
        if snapshot['baseline'] != baseline:
            raise ValueError('ownership snapshot baseline differs from audit baseline')
        records = [(dict(row, execution_scope=snapshot['scopes'][row['scope_id']]), row['record_sha256'])
                   for row in snapshot['rows']]
    else:
        records = [(json.loads(path.read_bytes()), hashlib.sha256(path.read_bytes()).hexdigest())
                   for path in sorted(ownership.glob('*.json'))]
    rows = []
    for record, record_sha in records:
        scope = record.get('execution_scope') or {}
        if not scope and record.get('destination'):
            destination = record['destination']
            scope = {'source_commit':destination.get('source_commit'),
                     'files':{**destination.get('scope_files',{}),**destination.get('scope_third_party',{})}}
        files = scope.get('files') or {}
        differences = sorted(name for name, sha in files.items() if current(name) != sha)
        baseline_differences = sorted(name for name, sha in files.items() if adopted(name) != sha)
        release_changes = sorted(name for name in files if current(name) != adopted(name))
        rows.append({'job_id':record['job_id'], 'state':record['state'],
                     'source_commit':scope.get('source_commit'), 'files':len(files),
                     'scope_persisted':bool(files), 'differences':differences,
                     'baseline_differences':baseline_differences, 'running':record.get('running',False),
                     'release_changes':release_changes,
                     'record_sha256':record_sha})
    return {'schema':'dsreset-execution-scope-audit/2', 'baseline':baseline,
            'ownership_dir':str(Path(ownership_dir).resolve()), 'rows':rows,
            'absolute_scope_identical':bool(rows) and all(r['scope_persisted'] and not r['differences'] for r in rows),
            'same_scope_differences_as_main':bool(rows) and all(r['differences'] == r['baseline_differences'] for r in rows),
            'zero_release_changes':bool(rows) and all(not r['release_changes'] for r in rows)}
