"""Remove our remaining stopped containers on the shared remote Docker host. Dated copy (user directive 2026-11-11
~17:30: stopped containers that are EXCLUSIVELY ours, verified by labels + identity + scheduler records, logs/metadata
archived first, `docker rm <id>` per container, never prune) of remote_container_cleanup_20261025.py (user request
2026-10-25). Changes: the scheduler is not imported (registry replayed by disk_cleanup_common_20261111), remote access
through ops.remote.access.RemoteClient, and stricter retention: only containers of runs whose registry state is `done`
(job containers) or `done`/`failed` (verification containers) are removed; crashed (resumable), paused, pending, active
or unknown runs keep theirs; protected families (real-world, Arm4, paused; disk_cleanup_common_20261111) keep theirs;
verification containers of pinned / recently accepted releases are kept; containers without s4d.managed=true (incl.
the real-world preprocessing s4d-rwprep-* containers and other users' containers) are only counted, never inspected.

Ownership needs all of: label s4d.managed=true; for job and verification containers the labels s4d.host=remote,
s4d.run_id, s4d.attempt, the deterministic name s4d-<run_id>-a<attempt>, and a registry record (launched or
remote_exited) of that run carrying exactly this container id; for the one runtime-integrity container (no run labels)
the s4d.purpose=runtime-integrity label, our runtime manifest digest in s4d.runtime_manifest and in the name, and the
id recorded by the 2026-10-21 inventory. Containers without s4d.managed=true are only listed, never inspected.
Retained: running, paused, restarting or otherwise not exited containers; containers of runs whose registry state is
active or unknown (the scheduler may still reconcile them); anything ambiguous.

Writable layers: these containers ran with a read-only root filesystem; their diffs (reviewed 2026-10-25) contain only
NVIDIA container-toolkit injections (driver libraries, ld.so/vulkan config, firmware), ldconfig caches and mount-point
stubs. Each container's `docker diff`, Docker log and inspect record are still archived with checksums and a receipt
before removal: job containers under runs/<run_id>/containers/, others under runtime/container-archive/<name>/.
Removal re-checks the full identity by explicit id under a per-container lock, then `docker rm <id>` (no force, no
volumes, no prune). Calls are paced (one SSH call every 2 s).

    nice -n 19 .venv/bin/python -I runs/setup/tools/remote_container_cleanup_20261111.py            # inventory + plan
    nice -n 19 .venv/bin/python -I runs/setup/tools/remote_container_cleanup_20261111.py --apply
"""
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import disk_cleanup_common_20261111 as C  # noqa: E402
from ops.jobs.remote_backend import _INSPECT_IDENTITY, _REMOVAL_SCRIPT  # noqa: E402
from ops.remote.access import RemoteClient  # noqa: E402
import disk_cleanup_remote_20261111 as R  # noqa: E402  (release_pins; its import opens no connection by itself)

REMOTE = '/home/compu/kaist/sunho'
OUT = C.OUT / 'remote-containers.json'
APPLY = '--apply' in sys.argv
RUNTIME_MANIFEST = '03cca5218096dc8837e63e10ef2ddc847af8a12787dd57e3eceb0be6381b2981'
INTEGRITY_ID = '9401ba7abb6d7e5a0dfd40bf96094faa054d40be6c68e0f0547e94bc2d2f0397'  # 2026-10-21 inventory
SHA = re.compile(r'[0-9a-f]{64}')
# kept on 2026-10-25 and 2026-11-06 (no labels / registry mismatch): never removed by this tool
NEVER = {'s4d-verify-native-e267ba8-20261011-a1'}
RUN_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,199}')


CLIENT = RemoteClient()


def ssh(command, timeout=600, idempotent=True):
    time.sleep(2)
    return CLIENT.ssh(command, wait=True, idempotent=idempotent, timeout=timeout)


def inventory():
    listing = ssh("docker ps -a --no-trunc --format '{{.ID}}|{{.Names}}|{{.State}}|{{.Label \"s4d.managed\"}}'; echo __END__")
    assert listing.returncode == 0 and '__END__' in listing.stdout, listing.stderr[-300:]
    rows = [line.split('|') for line in listing.stdout.split('__END__')[0].strip().splitlines()]
    ours = [r[0] for r in rows if r[3] == 'true' and r[2] != 'running']
    details = {}
    for i in range(0, len(ours), 10):
        ids = ' '.join(ours[i:i + 10])
        fmt = shlex.quote(_INSPECT_IDENTITY + '|{{.State.FinishedAt}}|{{json .Config.Labels}}')
        out = ssh(f"for id in {ids}; do printf '__C__ %s\\n' \"$id\"; docker inspect --format {fmt} \"$id\"; done; echo __END__")
        assert out.returncode == 0 and '__END__' in out.stdout, out.stderr[-300:]
        for block in out.stdout.split('__END__')[0].split('__C__ ')[1:]:
            cid, line = block.strip().split('\n', 1)
            # the identity has 11 '|'-separated fields without '|' inside; FinishedAt and the labels JSON follow
            parts = line.split('|', 12)
            identity, finished, labels = '|'.join(parts[:11]), parts[11], parts[12]
            details[cid] = {'identity': identity, 'finished': finished, 'labels': json.loads(labels)}
    return rows, details


def registry_ids():
    ids = {}
    for line in C.REGISTRY.read_text().splitlines():
        e = json.loads(line)
        if e.get('container_id') and e['event'] in ('launched', 'remote_exited'):
            ids.setdefault(e['container_id'], set()).add((e['id'], e.get('attempt')))
    return ids


def classify(cid, d, state, recorded, pins):
    f = d['identity'].split('|')
    labels = d['labels']
    full_id, status, code, managed, host, run_id, attempt, name, ro, verification, eligible = f
    row = {'id': full_id, 'name': name.lstrip('/'), 'status': status, 'code': code, 'readonly_rootfs': ro,
           'run_id': run_id, 'attempt': attempt, 'verification_purpose': verification, 'identity': d['identity']}
    if full_id != cid or managed != 'true':
        return row, 'not ours (identity or managed label differs)'
    if status != 'exited':
        return row, f'not stopped ({status})'
    if name.lstrip('/') in NEVER:
        return row, 'kept since 2026-10-25 (no labels / registry mismatch)'
    if labels.get('s4d.purpose') == 'runtime-integrity' and 's4d.run_id' not in labels:
        if (labels.get('s4d.runtime_manifest') == RUNTIME_MANIFEST and name == f'/s4d-runtime-integrity-{RUNTIME_MANIFEST[:16]}'
                and full_id == INTEGRITY_ID):
            row.update(kind='runtime_integrity', keep_dir=f'{REMOTE}/runtime/container-archive/{name.lstrip("/")}')
            return row, None
        return row, 'ambiguous runtime-integrity container'
    if host != 'remote' or not RUN_ID.fullmatch(run_id) or not attempt.isdigit() or name != f'/s4d-{run_id}-a{attempt}':
        return row, 'labels or name not ours for a run'
    if (run_id, int(attempt)) not in {(r, int(a)) for r, a in recorded.get(full_id, set()) if a is not None}:
        return row, 'no registry record of this run and attempt with this container id'
    if ro != 'true':
        return row, 'root filesystem not read-only (writable layer may hold outputs)'
    s = state.get(run_id, {})
    if s.get('unknown') or s.get('status') in C.ACTIVE or s.get('status') in (None, C.PENDING):
        return row, f"registry state {s.get('status')}{' unknown' if s.get('unknown') else ''}"
    protected = C.protected_prefix(run_id)
    if protected:
        return row, f'protected family: {protected}'
    if C.paused(run_id, PAUSES):
        return row, 'admission pause active'
    row['registry_status'] = s.get('status')
    if verification:
        if s.get('status') != 'done':  # failed suites keep their containers (Codex review 2026-11-11 finding 1/9)
            return row, f"verification job {s.get('status')} (kept for diagnosis / retry)"
        commit7 = re.match(r'verify-(?:native|isolation-\d+)-([0-9a-f]{7})', run_id)
        if not commit7 or commit7.group(1) in pins:  # pins: disk_cleanup_common_20261111.release_pins
            return row, 'release pinned / recently accepted (or unparsable release)'
        try:
            finished = datetime.fromisoformat(d['finished'][:26].rstrip('Z') + '+00:00').timestamp()
        except ValueError:
            return row, 'finish time unparsable'
        row['finished'] = d['finished']
        if time.time() - finished < 24 * 3600:
            return row, 'finished within 24 h'
        report = 'gpu_isolation.json' if verification == 'gpu_isolation' else 'tests.json'
        row.update(kind='verification', keep_dir=f'{REMOTE}/runtime/container-archive/{name.lstrip("/")}',
                   required=f'{REMOTE}/runtime/verification/{run_id}/a{attempt}/evidence/{report}')
    else:
        if s.get('status') != 'done':
            return row, f"job {s.get('status')} (failed / crashed containers kept for diagnosis)"
        row.update(kind='job', keep_dir=f'{REMOTE}/runs/{run_id}/containers')
    return row, None


def removal_script(row):
    q = shlex.quote
    name, keep = row['name'], row['keep_dir']
    log, record, diff = f'{keep}/{name}.docker.log', f'{keep}/{name}.inspect.json', f'{keep}/{name}.diff.txt'
    receipt = f'{keep}/{name}.removal-receipt'
    script = _REMOVAL_SCRIPT
    if row['kind'] == 'job':
        run_dir_check = f"test -d {q(REMOTE + '/runs/' + row['run_id'])} || exit 4\n"
    elif row['kind'] == 'verification':  # the retained verification report must exist (Codex review finding 6)
        run_dir_check = f"test -s {q(row['required'])} || {{ printf 'missing retained report\\n' >&2; exit 4; }}\n"
    else:
        run_dir_check = ''
    assert row['kind'] != 'verification' or 'test -s' in run_dir_check
    script = script.replace('test -d @RUN_DIR@ || exit 4\n@REQUIRED@\n', run_dir_check)
    script = script.replace("id=@ID@; tmp_log=; tmp_record=", "id=@ID@; tmp_log=; tmp_record=; tmp_diff=")
    script = script.replace("""trap 'rm -f -- "$tmp_log" "$tmp_record"' EXIT""",
                            """trap 'rm -f -- "$tmp_log" "$tmp_record" "$tmp_diff"' EXIT""")
    script = script.replace('docker inspect "$id" > "$tmp_record" || exit 5\n',
                            'docker inspect "$id" > "$tmp_record" || exit 5\n'
                            'tmp_diff="$(mktemp @DIFF_TMP@)" || exit 5\ndocker diff "$id" > "$tmp_diff" || exit 5\n'
                            'digest "$tmp_diff"; diffsum="$d"\nmv -f -- "$tmp_diff" @DIFF@ || exit 5\ntmp_diff=\n')
    script = script.replace("""printf '%s %s %s\\n' "$id" "$logsum" "$inspectsum" > @RECEIPT_TMP@""",
                            """printf '%s %s %s %s\\n' "$id" "$logsum" "$inspectsum" "$diffsum" > @RECEIPT_TMP@""")
    script = script.replace('read -r rid rlog rinspect < @RECEIPT@', 'read -r rid rlog rinspect rdiff < @RECEIPT@')
    values = {'@ID@': q(row['id']), '@FORMAT@': q(_INSPECT_IDENTITY), '@EXPECTED@': q(row['identity']),
              '@KEEP@': q(keep), '@LOCK@': q(f'{keep}/.{name}.lock'), '@LOG@': q(log), '@RECORD@': q(record),
              '@DIFF@': q(diff), '@LOG_TMP@': q(log + '.tmp.') + 'XXXXXX', '@RECORD_TMP@': q(record + '.tmp.') + 'XXXXXX',
              '@DIFF_TMP@': q(diff + '.tmp.') + 'XXXXXX', '@RECEIPT_TMP@': q(receipt + '.tmp'), '@RECEIPT@': q(receipt)}
    for k, v in values.items():
        script = script.replace(k, v)
    assert '@' not in re.sub(r"'[^']*'", '', script), 'unfilled placeholder'
    for needle in ('docker rm "$id"', 'check\ndocker rm', '"$diffsum" > ', 'rinspect rdiff', 'docker diff "$id"'):
        assert needle in script, needle
    assert ' -f "$id"' not in script and '--force' not in script and ' -v ' not in script and 'prune' not in script
    return script


rows, details = inventory()
state = C.read_registry()
PAUSES = C.active_pauses()
pins = R.release_pins(state, C.load_queue_rows())
recorded = registry_ids()
plan, kept = [], []
for cid, d in sorted(details.items()):
    row, why = classify(cid, d, state, recorded, pins)
    (kept if why else plan).append({**row, **({'kept': why} if why else {}), 'labels': d['labels']})
# not ours (no s4d.managed=true): counted by name prefix only, never inspected or touched
others = {}
for r in rows:
    if r[3] != 'true':
        key = 's4d-rwprep-* (real-world preprocessing dispatcher; protected)' if r[1].startswith('s4d-rwprep-') else r[1]
        others.setdefault(key, {'count': 0, 'states': {}})
        others[key]['count'] += 1
        others[key]['states'][r[2]] = others[key]['states'].get(r[2], 0) + 1
record = {'recorded_local': datetime.now().astimezone().isoformat(), 'host': 'remote',
          'containers_total': len(rows), 'ours_stopped_inspected': len(details), 'running_ours': sum(1 for r in rows if r[3] == 'true' and r[2] == 'running'),
          'not_ours_listed_only': others, 'release_pins': sorted(pins), 'plan': plan, 'kept': kept}
print(f"containers {len(rows)}; ours stopped {len(details)}; plan {len(plan)}; kept {len(kept)}; not ours {len(others)}")
for k in kept:
    print('KEEP', k['name'], k['kept'])
from collections import Counter  # noqa: E402
print(Counter((p['kind'], p.get('registry_status')) for p in plan))
scripts = {row['id']: removal_script(row) for row in plan}  # every script is built (and checked) before any removal
if not APPLY:
    OUT.write_text(json.dumps(record, indent=1) + '\n')
    print(scripts[plan[0]['id']] if plan else '')
    raise SystemExit(0)
# --apply removes only containers that were in the reviewed plan (Codex review 2026-11-11 finding 6) AND are eligible now
REVIEWED = C.OUT / 'remote-containers.reviewed.json'
allow = {p['id'] for p in json.loads(REVIEWED.read_text())['plan']}
record['reviewed_allowlist'] = str(REVIEWED)
record['not_in_reviewed_plan'] = [p['name'] for p in plan if p['id'] not in allow]
plan = [p for p in plan if p['id'] in allow]
results = []
for row in plan:
    st_now = C.read_registry()  # controller-side recheck right before each removal: state, pins, pauses
    s_now = st_now.get(row['run_id'], {})
    pins_now = R.release_pins(st_now, C.load_queue_rows())
    commit7 = re.search(r'-([0-9a-f]{7})(?:-|$)', row['run_id'])
    if (s_now.get('status') != row['registry_status'] or s_now.get('unknown') or not commit7
            or commit7.group(1) in pins_now or C.paused(row['run_id'], C.active_pauses())):
        results.append({'name': row['name'], 'id': row['id'], 'returncode': None, 'result': None,
                        'stderr': 'controller recheck failed (state, pin or pause changed); skipped'})
        continue
    r = ssh(scripts[row['id']], timeout=600)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''
    try:
        payload = json.loads(last)
    except ValueError:
        payload = None
    results.append({'name': row['name'], 'id': row['id'], 'returncode': r.returncode, 'result': payload,
                    'stderr': r.stderr[-300:] if r.returncode else ''})
    print(row['name'], r.returncode, last[:120])
after = ssh("docker ps -a --no-trunc --format '{{.ID}}|{{.Names}}|{{.State}}|{{.Label \"s4d.managed\"}}' && echo __END__")
if after.returncode or not after.stdout.rstrip().endswith('__END__'):  # fail closed (Codex review finding 7)
    raise SystemExit(f'final container listing failed ({after.returncode}); results unverified: {results}')
remaining = {line.split('|')[0] for line in after.stdout.split('__END__')[0].strip().splitlines()}
for res in results:
    res['verified_absent'] = res['id'] not in remaining
record.update(applied_local=datetime.now().astimezone().isoformat(), results=results,
              removed_verified=sum(r['verified_absent'] for r in results),
              containers_after=len(remaining))
OUT.write_text(json.dumps(record, indent=1) + '\n')
print(f"removed (verified absent) {record['removed_verified']} of {len(plan)}; containers now {len(remaining)}")
