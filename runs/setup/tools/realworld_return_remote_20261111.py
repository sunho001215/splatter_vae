"""Return NEVER-LAUNCHED real-world rows that `ops/jobs/realworld_local.py place` moved to local back to the remote
(unified pool; user 2026-11-11 ~21:50: MIG reboot postponed, remote GPUs unheld; e2e_cnn runs ~3x faster there).

Each row is restored to EXACTLY its pre-placement row as recorded in runs/setup/pool-placement.json (the move entry's
`original`), which differs from the current row only in host fields (host, placement, pin_host, gpu); the job
fingerprint is unchanged and asserted. Refused per row unless: it is in the queue on local; it has no registry state
and no runs/<id> (never launched); its ownership record is the role_placement written by `place` (owner local, state
placed); the latest pool move of the row went remote -> local with an `original` row on the remote; no active admission
pause holds it (unless named with --allow-pause); the remote roles admit the restored row (ownership aside). Under
runs/setup/remote-driver.lock -> runs/setup/stage2-driver.lock -> experiments/registry.lock (each polled up to 120 s;
busy -> nothing changes, retry), every rule is re-checked, then: ownership record -> owner remote (state placed, history kept;
blocks local admission), queue rewrite (host_roles.write_queue: the rest of the file byte for byte, validated before
the atomic rename), pool-placement.json move entry. Remote inputs are the originals on the remote; the remote driver
then admits the rows on the role release like every other remote real-world row. Local mirror files are untouched.

    .venv/bin/python -I runs/setup/tools/realworld_return_remote_20261111.py [--dry-run] [--allow-pause NAME] <ids...>
"""
import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from ops.jobs import host_roles, scheduler  # noqa: E402
from ops.jobs.admission_pause import active_pauses  # noqa: E402

LOCKS = (ROOT / 'runs/setup/remote-driver.lock', host_roles.ADMISSION_LOCK, scheduler.REGISTRY.with_suffix('.lock'))


def original_row(job_id, pool):
    """The row as it was before its latest move to local, or a refusal reason."""
    moves = [m for m in pool.get('moves', []) if job_id in m.get('ids', [])]
    if not moves or moves[-1].get('to') != 'local':
        return None, 'its latest pool move is not a move to local'
    original = (moves[-1].get('original') or {}).get(job_id)
    if not isinstance(original, dict) or original.get('host') != 'remote':
        return None, 'no recorded original remote row'
    return original, None


def release_scope_differences(job, planned, current):
    """Paths whose bytes differ between the row's pinned release and the current role release within the row's
    execution scope (ops/jobs/migration.py: the script's import closure, configs/metaworld|rl|baselines, uv.lock /
    pyproject, third-party pins) plus every repository file the arguments name. The remote driver launches a
    never-launched role row on the CURRENT role release (job_release), so a return is allowed only when that release
    runs exactly the code the row was pinned to (Codex review 6)."""
    from ops.jobs import migration
    from ops.provenance.upstreams import commit_identity

    read_a, listed_a = migration.commit_reader(ROOT, planned)
    read_b, listed_b = migration.commit_reader(ROOT, current)
    script = [job['script']]
    paths = migration.import_closure(script, read_a) | migration.import_closure(script, read_b)
    paths |= {p for p in listed_a + listed_b if p.startswith(migration.CONFIG_SCOPE)} | set(migration.ENVIRONMENT_SCOPE)
    paths |= {str(a).split('=', 1)[-1] for a in job.get('args', [])
              if str(a).split('=', 1)[-1].startswith(('configs/', 'scripts/', 's4d/', 'ops/'))}
    differences = sorted(p for p in paths if p not in migration.OPERATIONAL_ALLOWED and read_a(p) != read_b(p))
    pins_a, pins_b = commit_identity(ROOT, planned), commit_identity(ROOT, current)
    return differences + sorted(k for k in set(pins_a) | set(pins_b) if pins_a.get(k) != pins_b.get(k))


def check(job_id, by_id, state, ownership, pool, roles, allow, role_commit=None):
    job = by_id.get(job_id)
    if job is None or job.get('host', 'local') != 'local':
        return None, 'not a local queue row'
    if job_id in state or (ROOT / 'runs' / job_id).exists() or (ROOT / 'runs' / job_id).is_symlink():
        return None, 'launched or has a run directory (started rows move only by explicit migration)'
    record = ownership.get(job_id) or {}
    if record.get('kind') != 'role_placement' or record.get('owner_host') != 'local' or record.get('state') != 'placed':
        return None, 'no local role_placement ownership record'
    pauses = set(active_pauses(job_id, fresh=True)) - set(allow)
    if pauses:
        return None, f'held by active pause(s) {sorted(pauses)}'
    original, why = original_row(job_id, pool)
    if why:
        return None, why
    same = {k: v for k, v in job.items() if k not in host_roles.HOST_FIELDS}
    if same != {k: v for k, v in original.items() if k not in host_roles.HOST_FIELDS}:
        return None, 'the row changed beyond host fields since it was placed'
    if scheduler.job_fingerprint(original) != scheduler.job_fingerprint(job):
        return None, 'job fingerprint differs from the recorded original'
    refused = host_roles.refusal(original, 'remote', roles, {}, by_id)
    if refused is not None:
        return None, f'remote roles refuse it: {refused}'
    planned = original.get('planned_commit')
    if not planned or not role_commit:
        return None, 'no pinned release (planned_commit) or no current role release'
    if planned != role_commit:
        differences = release_scope_differences(original, planned, role_commit)
        if differences:
            return None, (f'the current role release {role_commit[:7]} differs from its pinned release {planned[:7]} '
                          f'within its execution scope: {differences[:5]}')
    return original, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('ids', nargs='+')
    ap.add_argument('--allow-pause', action='append', default=[])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    with contextlib.ExitStack() as stack:
        for path in LOCKS:  # in this order; each polled for up to 120 s (a driver pass / scheduler tick is short)
            handle = stack.enter_context(open(path, 'a'))
            deadline = time.time() + 120
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() > deadline:
                        print(json.dumps({'moved': [], 'busy': str(path)}))
                        return 3
                    time.sleep(0.5)
        roles = host_roles.load_roles()
        queue = scheduler.load_queue()
        by_id = {j['id']: j for j in queue['jobs']}
        state = scheduler.read_registry()
        ownership = host_roles.read_ownership()
        pool = json.loads(host_roles.POOL_MANIFEST.read_text())
        role_commit = json.loads((ROOT / 'runs/setup/role-release.json').read_text()).get('commit')
        chosen, kept = {}, {}
        for job_id in dict.fromkeys(args.ids):
            original, why = check(job_id, by_id, state, ownership, pool, roles, args.allow_pause, role_commit)
            if why:
                kept[job_id] = why
            else:
                chosen[job_id] = original
        if args.dry_run or not chosen:
            print(json.dumps({'would_move' if args.dry_run else 'moved': sorted(chosen), 'kept': kept}, indent=1))
            return 0 if chosen or args.dry_run else 2
        stamp = host_roles.now()
        reason = ('unified pool, user 2026-11-11 ~21:50 (remote unheld): never-launched row returned from local '
                  'placement to the remote (restored to its recorded original row; current role release ' + str(role_commit) + ' has the identical execution scope)')
        for job_id in chosen:  # ownership first: an interrupted run leaves both drivers refusing locally, never doubled
            previous = ownership[job_id]
            host_roles.write_ownership({
                'job_id': job_id, 'kind': 'role_placement', 'state': 'placed', 'owner_host': 'remote',
                'from_host': 'local', 'to_host': 'remote', 'attempts': 0, 'time': stamp, 'reason': reason,
                'roles_sha256': roles['sha256'],
                'history': previous.get('history', []) + [{k: v for k, v in previous.items() if k != 'history'}]})
        before = scheduler.QUEUE.read_bytes()
        effective = {i: copy.deepcopy(o) for i, o in chosen.items()}
        host_roles.append_pool_record(host_roles.POOL_MANIFEST, {
            'time': stamp, 'to': 'remote', 'reason': f"{reason} (configs/hosts/roles.yaml sha256 {roles['sha256']})",
            'ids': list(chosen), 'original': {i: by_id[i] for i in chosen},
            'queue_sha256_before': hashlib.sha256(before).hexdigest(), 'queue_sha256_after': None,
            'tool': 'runs/setup/tools/realworld_return_remote_20261111.py'}, effective)
        _, after = host_roles.write_queue(scheduler.QUEUE, [effective.get(j['id'], j) for j in queue['jobs']],
                                          scheduler.load_queue)
        pool = json.loads(host_roles.POOL_MANIFEST.read_text())
        pool['moves'][-1]['queue_sha256_after'] = hashlib.sha256(after).hexdigest()
        tmp = host_roles.POOL_MANIFEST.with_suffix('.tmp')
        tmp.write_text(json.dumps(pool, indent=1) + '\n')
        tmp.replace(host_roles.POOL_MANIFEST)
        check_queue = {j['id']: j for j in scheduler.load_queue()['jobs']}
        assert all(check_queue[i] == chosen[i] for i in chosen), 'queue rewrite differs from the recorded originals'
        print(json.dumps({'moved': sorted(chosen), 'kept': kept}, indent=1))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
