"""Replacement rows for rwnq FQL rows whose LOCAL attempt was unusable (coordinator, 2026-11-12).

rwnq-20261111-fql-reviwo-drawer-open-s{0,1} launched locally at 00:03 on 2026-11-12 WITHOUT a thread limit. The local
launch path set no OMP/MKL/OPENBLAS_NUM_THREADS (fixed in 37bfd30). Each process spawned 325 threads, and the main
thread got about 9 % of a core. After 50 min neither had logged step 5000, so no result exists. Retries of those ids
would be refused by the local placement check after the fd3405f adoption: the pinned scope differs, and a retry keeps
its first pin.

Each replacement is the original queue row with only these changes:
- id `rwnq-20261111b-fql-...` and the matching --name argument;
- the local pin fields dropped (release_source, local_placement_pause, repinned_from);
- pinned to the current role release;
- `replaces: <old id>` and a purpose note.

Arguments, datasets, setting (alpha=10 + normalize_q_loss) and seed are identical. The rows are held by
fqlnq-await-placement-20261111; the coordinator adds their ids to that pause before executing.
Usage: register_fqlnq_replacements_20261112.py --commit C --image I OLD_ID [OLD_ID ...] [--execute]
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

import yaml

REPO = Path('/home/ws/ws/splatter4d')
QUEUE = REPO / 'experiments/queue.yaml'
spec = importlib.util.spec_from_file_location('regnq', REPO / 'runs/setup/tools/register_fql_nq_20261111.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
DROP = ('release_source', 'local_placement_pause', 'repinned_from', 'placement')  # host/pin_host reset to remote below
PAUSE = 'fqlnq-await-placement-20261111'
TOOL = 'runs/setup/tools/' + Path(__file__).name


def new_id(old):
    assert old.startswith('rwnq-20261111-fql-'), old
    return old.replace('rwnq-20261111-fql-', 'rwnq-20261111b-fql-', 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--commit', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('ids', nargs='+')
    a = ap.parse_args()
    role = json.loads((REPO / 'runs/setup/role-release.json').read_text())
    assert role['commit'] == a.commit and role['image'] == a.image, 'role-release.json differs from --commit/--image'
    manifest = f"runs/remote/acceptance/{a.commit[:7]}-cc/manifest.json"
    assert (REPO / manifest).is_file(), f'no remote acceptance manifest {manifest}'
    with (REPO / 'runs/setup/stage2-driver.lock').open('a') as l1:
        fcntl.flock(l1, fcntl.LOCK_EX)
        with (REPO / 'experiments/registry.lock').open('a') as l2:
            fcntl.flock(l2, fcntl.LOCK_EX)
            before = QUEUE.read_bytes()
            jobs = {j['id']: j for j in yaml.safe_load(before)['jobs']}
            seen = set()
            for line in (REPO / 'experiments/registry.jsonl').read_text().splitlines():
                try:
                    seen.add(json.loads(line).get('id'))
                except ValueError:
                    pass
            pauses = json.loads((REPO / 'runs/setup/admission-pauses.json').read_text())['pauses']
            held = next(p for p in pauses if p.get('name') == PAUSE)
            rows = []
            for old in a.ids:
                nid = new_id(old)
                assert old in jobs and nid not in jobs and nid not in seen, (old, nid)
                row = copy.deepcopy(jobs[old])
                for k in DROP:
                    row.pop(k, None)
                args = row['args']
                args[args.index('--name') + 1] = nid
                row.update(host='remote', pin_host='remote')  # placement belongs to the old row's placement record
                row.update(id=nid, planned_commit=a.commit, planned_image=a.image, evidence_manifest=manifest,
                           registration_tool=TOOL, replaces=old,
                           purpose=row.get('purpose', '') + ' | REPLACEMENT 2026-11-12 of ' + old + ': its local '
                           'attempt ran without a thread limit (no result); identical arguments and setting')
                diff = {k for k in set(row) | set(jobs[old]) if row.get(k) != jobs[old].get(k)}
                assert diff <= {'host', 'pin_host', 'id', 'args', 'planned_commit', 'planned_image', 'evidence_manifest',
                                'registration_tool', 'replaces', 'purpose', *DROP}, diff
                assert [x for x in args if x != nid] == [x for x in jobs[old]['args'] if x != old]
                if a.execute:
                    assert held.get('active') and nid in held.get('ids', []), f'{nid} not held by {PAUSE}'
                rows.append(row)
            candidate = R.queue_candidate(before, rows)
            report = {'new_ids': [r['id'] for r in rows], 'replaces': a.ids, 'commit': a.commit,
                      'queue_sha256_before': hashlib.sha256(before).hexdigest(), 'execute': a.execute}
            print(json.dumps(report, indent=1))
            if not a.execute:
                return
            backup = QUEUE.with_name('queue.yaml.before-fqlnq-replacements-' + time.strftime('%Y%m%d-%H%M%S'))
            shutil.copy2(QUEUE, backup)
            tmp = QUEUE.with_name(QUEUE.name + '.tmp-fqlnq-repl')
            tmp.write_bytes(candidate)
            with tmp.open('rb') as handle:
                os.fsync(handle.fileno())
            shutil.copymode(QUEUE, tmp)
            os.replace(tmp, QUEUE)
            (REPO / 'runs/setup/rw-registration-fqlnq-replacements-20261112.json').write_text(
                json.dumps({**report, 'rows': rows, 'registered_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                            'backup': str(backup)}, indent=1) + '\n')
            print('written; backup', backup)


if __name__ == '__main__':
    main()
