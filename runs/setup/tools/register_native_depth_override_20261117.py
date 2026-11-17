"""Amend ONLY the never-placed drawer S native-depth worker; default is read-only.

Normal stage-5 registration remains authoritative. Its original proof, queue and
registry are durably backed up before the one-field amendment. The amended proof
explicitly names that original, so normal stage-5 resume validation still works.
No registry event, pin, argv, dependency, hold or placed row is rewritten.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE))
import yaml
from ops.jobs.native_depth_override_registration import BASE_COMMIT, OVERRIDE_SCRIPT, TARGET

MAIN = Path('/home/ws/ws/splatter4d')
DEFAULT_EVIDENCE = Path('/home/ws/ws/realworld-ops/drawer-geo-20261116/mvp3dnd-20261117')
OLD_SCRIPT = 'scripts/dsreset_realworld_native_depth.py'
PAUSE = 'dsreset-await-placement-20261113'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(doc):
    return (json.dumps(doc, indent=2, sort_keys=True) + '\n').encode()


def atomic(path, raw):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def locks(root):
    # Same order as the normal registrar / placement publication locks.
    handles = []
    try:
        for name in ('runs/setup/stage2-driver.lock', 'runs/setup/remote-driver.lock', 'experiments/registry.lock'):
            f = (root/name).open('a')
            handles.append(f)
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for f in reversed(handles):
            f.close()


def verify_pin(root, row):
    commit = row.get('planned_commit', '')
    require(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'full planned_commit required')
    subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', BASE_COMMIT, commit], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    expected = subprocess.run(['git', '-C', str(root), 'show', f'{commit}:{OVERRIDE_SCRIPT}'],
                              check=True, capture_output=True).stdout
    require((root/OVERRIDE_SCRIPT).read_bytes() == expected, 'override worker differs from pinned release')


def scalar_replacement(raw):
    text = raw.decode()
    node = yaml.compose(text)
    jobs = next(v for k, v in node.value if k.value == 'jobs')
    matches = []
    for row in jobs.value:
        fields = {k.value: v for k, v in row.value}
        if fields['id'].value == TARGET:
            matches.append(fields['script'])
    require(len(matches) == 1, 'unique target script required')
    script = matches[0]
    return (text[:script.start_mark.index] + OVERRIDE_SCRIPT + text[script.end_mark.index:]).encode()


def inspect(root):
    queue = root/'experiments/queue.yaml'
    raw = queue.read_bytes()
    doc = yaml.safe_load(raw)
    rows = {r['id']: r for r in doc['jobs']}
    require(len(rows) == len(doc['jobs']), 'duplicate queue IDs')
    if TARGET not in rows:
        return {'status': 'awaiting-registration', 'target': TARGET}, None
    row = rows[TARGET]
    require(row.get('script') in (OLD_SCRIPT, OVERRIDE_SCRIPT), 'unexpected target entrypoint')
    require(row.get('realworld_target') == 'drawer-open-v2'
            and row.get('registration_tool') == 'runs/setup/tools/register_dsreset_20261113.py',
            'target is not the normal drawer S publication')
    verify_pin(root, row)
    proofs = []
    for path in (root/'runs/setup').glob('rwv2-20261113-S-*.registration.json'):
        d = json.loads(path.read_bytes())
        if any(r['id'] == TARGET for r in d.get('rows', [])):
            proofs.append((path, d))
    require(len(proofs) == 1, 'one original published registration proof required')
    proof_path, proof = proofs[0]
    require(proof.get('registered') is True and proof.get('status') == 'published'
            and proof.get('variants') == ['S'] and len(proof['rows']) == 39, 'complete 39-row S publication required')
    require(not proof_path.with_name(proof_path.name.replace('.registration.json', '.pending.json')).exists(),
            'registration journal still pending')
    proof_row = next(r for r in proof['rows'] if r['id'] == TARGET)
    if row['script'] == OVERRIDE_SCRIPT and proof_row['script'] == OVERRIDE_SCRIPT:
        amendment = proof.get('native_depth_entrypoint_amendment', {})
        require(amendment.get('target') == TARGET and amendment.get('changed_fields') == ['script'],
                'override row has no explicit amendment provenance')
        return {'status': 'already-applied', 'target': TARGET, 'planned_commit': row['planned_commit']}, None
    require(row.get('host') == row.get('pin_host') == 'remote', 'target must still be an unplaced remote row')
    require(not (root/'runs'/TARGET).exists() and not (root/'runs'/TARGET).is_symlink(), 'target run output exists')
    owner = root/'runs/setup/ownership'/f'{TARGET}.json'
    require(not owner.exists() and not owner.is_symlink(), 'target has ownership; never amend a placed row')
    pool_path = root/'runs/setup/pool-placement.json'
    pool = json.loads(pool_path.read_bytes())
    require(TARGET not in pool.get('effective_jobs', {})
            and TARGET not in json.dumps(pool.get('moves', [])), 'target has pool placement history')
    registry = root/'experiments/registry.jsonl'
    registry_raw = registry.read_bytes()
    events = [json.loads(line) for line in registry_raw.splitlines() if line.strip()]
    require(not any(e.get('id') == TARGET for e in events), 'target has registry history; never amend a started row')
    pauses = json.loads((root/'runs/setup/admission-pauses.json').read_bytes())
    require(any(p.get('name') == PAUSE and p.get('active') and TARGET in p.get('ids', [])
                and not p.get('prefixes') for p in pauses['pauses']), 'active exact-ID publication hold required')
    require(proof_row['script'] == OLD_SCRIPT, 'unexpected original proof script')
    for key in ('planned_commit', 'planned_image', 'args', 'deps', 'required_results'):
        require(row.get(key) == proof_row.get(key), f'target differs from registration: {key}')
    candidate = scalar_replacement(raw) if row['script'] == OLD_SCRIPT else raw
    wanted = copy.deepcopy(doc)
    next(r for r in wanted['jobs'] if r['id'] == TARGET)['script'] = OVERRIDE_SCRIPT
    require(yaml.safe_load(candidate) == wanted, 'queue edit changed more than target script')
    return {'status': 'dry-run', 'target': TARGET, 'changed_fields': ['script'],
            'planned_commit': row['planned_commit'], 'old_script': OLD_SCRIPT, 'new_script': OVERRIDE_SCRIPT,
            'queue_sha256_before': sha(raw), 'queue_sha256_after': sha(candidate),
            'registry_sha256': sha(registry_raw), 'proof': str(proof_path)}, (raw, candidate, registry_raw, proof)


def repair(root, evidence, execute=False):
    # Dry-run deliberately creates no lock or evidence files.
    if not execute:
        return inspect(root)[0]
    with locks(root):
        record, blobs = inspect(root)
        if blobs is None:
            return record
        evidence.mkdir(parents=True, exist_ok=True)
        queue_raw, candidate, registry_raw, proof = blobs
        proof_path = Path(record['proof'])
        pending = evidence/'pending.json'
        if pending.exists():
            saved = json.loads(pending.read_bytes())
            require(saved['queue_sha256_after'] == record['queue_sha256_after']
                    and saved['registry_sha256'] == record['registry_sha256'], 'pending amendment inputs drifted')
            record = saved
        else:
            require(next(r for r in yaml.safe_load(queue_raw)['jobs'] if r['id'] == TARGET)['script'] == OLD_SCRIPT,
                    'partial amendment has no recovery journal')
            backups = {}
            for name, raw in [('queue.yaml', queue_raw), ('registry.jsonl', registry_raw),
                              (proof_path.name, proof_path.read_bytes())]:
                path = evidence/(name+'.before-mvp3dnd-20261117')
                if path.exists():
                    require(path.read_bytes() == raw, 'backup conflict: '+str(path))
                else:
                    with path.open('xb') as f:
                        f.write(raw)
                        f.flush()
                        os.fsync(f.fileno())
                backups[str(path)] = sha(raw)
            record = {**record, 'backups': backups, 'created_utc': datetime.now(timezone.utc).isoformat()}
            atomic(pending, encoded(record))
        amended = copy.deepcopy(proof)
        next(r for r in amended['rows'] if r['id'] == TARGET)['script'] = OVERRIDE_SCRIPT
        # Registrar uses default-separator, sorted JSON for this digest.
        amended['rows_sha256'] = sha(json.dumps(amended['rows'], sort_keys=True).encode())
        amended['native_depth_entrypoint_amendment'] = {**record, 'status': 'applied'}
        require((root/'experiments/queue.yaml').read_bytes() == queue_raw, 'queue drift under locks')
        require((root/'experiments/registry.jsonl').read_bytes() == registry_raw, 'registry drift under locks')
        atomic(root/'experiments/queue.yaml', candidate)
        atomic(proof_path, encoded(amended))
        applied = {**record, 'status': 'applied', 'proof_sha256_after': sha(encoded(amended))}
        atomic(evidence/'applied.json', encoded(applied))
        pending.unlink()
        return applied


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=MAIN)
    p.add_argument('--evidence', type=Path, default=DEFAULT_EVIDENCE)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    require(not a.execute or (a.repo.resolve() == MAIN and SOURCE == MAIN), 'execute only the adopted coordinator copy')
    try:
        result = repair(a.repo, a.evidence, a.execute)
    except BlockingIOError:
        raise SystemExit('publication/driver lock busy; safe to retry')
    print(json.dumps(result, indent=2))
    if a.execute and result['status'] == 'awaiting-registration':
        raise SystemExit('normal 39-row registration must finish first')


if __name__ == '__main__':
    main()
