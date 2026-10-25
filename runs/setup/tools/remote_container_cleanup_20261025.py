"""Remove our remaining stopped containers on the shared remote Docker host (user request 2026-10-25: stopped Docker
containers created by this campaign, after verifying ownership and preserving any needed files from their writable
layers; never another person's containers). Complements remote_container_cleanup.py (2026-10-21, completed jobs) and the
scheduler's automatic cleanup, which both keep failed-job and verification containers.

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

    python -I remote_container_cleanup_20261025.py            # inventory + plan (read-only)
    python -I remote_container_cleanup_20261025.py --apply
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
from scripts import jobs  # noqa: E402
from s4d.remote_jobs import _INSPECT_IDENTITY, _REMOVAL_SCRIPT  # noqa: E402

REMOTE = '/home/compu/kaist/sunho'
OUT = ROOT / 'runs/setup/disk-cleanup-20261025' / ('remote-containers-rundir-missing.json' if '--archive-missing-rundir'
                                                   in sys.argv else 'remote-containers.json')
APPLY = '--apply' in sys.argv
RUNTIME_MANIFEST = '03cca5218096dc8837e63e10ef2ddc847af8a12787dd57e3eceb0be6381b2981'
INTEGRITY_ID = '9401ba7abb6d7e5a0dfd40bf96094faa054d40be6c68e0f0547e94bc2d2f0397'  # 2026-10-21 inventory
SHA = re.compile(r'[0-9a-f]{64}')
# verified missing on the remote at the first --apply (exit 4, run directory absent)
MISSING_RUN_DIRS = {'s4-snerl-hammer-rem', 's4-dynamo-hammer-rem', 'bxs-dynamo-hammer-rand-s1001-rem-eval',
                    'bxs-snerl-hammer-rand-s1000-rem', 'bxs-dynamo-hammer-rand-s1001-rem',
                    'bxs-snerl-hammer-rand-s1000-rem-eval'}
RUN_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,199}')


def ssh(command, timeout=600):
    time.sleep(2)
    result = subprocess.run([sys.executable, '-I', str(ROOT / 's4d/remote_access.py'), 'ssh', '--idempotent', '--',
                             command], capture_output=True, text=True, timeout=timeout)
    return result


def inventory():
    listing = ssh("docker ps -a --no-trunc --format '{{.ID}}|{{.Names}}|{{.State}}|{{.Label \"s4d.managed\"}}'; echo __END__")
    assert listing.returncode == 0 and '__END__' in listing.stdout, listing.stderr[-300:]
    rows = [line.split('|') for line in listing.stdout.split('__END__')[0].strip().splitlines()]
    ours = [r[0] for r in rows if r[3] == 'true' and r[2] != 'running']
    details = {}
    for i in range(0, len(ours), 10):
        ids = ' '.join(ours[i:i + 10])
        fmt = shlex.quote(_INSPECT_IDENTITY + '|{{json .Config.Labels}}')
        out = ssh(f"for id in {ids}; do printf '__C__ %s\\n' \"$id\"; docker inspect --format {fmt} \"$id\"; done; echo __END__")
        assert out.returncode == 0 and '__END__' in out.stdout, out.stderr[-300:]
        for block in out.stdout.split('__END__')[0].split('__C__ ')[1:]:
            cid, line = block.strip().split('\n', 1)
            # the identity has 11 '|'-separated fields without '|' inside; the labels JSON follows
            parts = line.split('|', 11)
            identity, labels = '|'.join(parts[:11]), parts[11]
            details[cid] = {'identity': identity, 'labels': json.loads(labels)}
    return rows, details


def registry_ids():
    ids = {}
    for line in jobs.REGISTRY.read_text().splitlines():
        e = json.loads(line)
        if e.get('container_id') and e['event'] in ('launched', 'remote_exited'):
            ids.setdefault(e['container_id'], set()).add((e['id'], e.get('attempt')))
    return ids


def classify(cid, d, state, recorded):
    f = d['identity'].split('|')
    labels = d['labels']
    full_id, status, code, managed, host, run_id, attempt, name, ro, verification, eligible = f
    row = {'id': full_id, 'name': name.lstrip('/'), 'status': status, 'code': code, 'readonly_rootfs': ro,
           'run_id': run_id, 'attempt': attempt, 'verification_purpose': verification, 'identity': d['identity']}
    if full_id != cid or managed != 'true':
        return row, 'not ours (identity or managed label differs)'
    if status != 'exited':
        return row, f'not stopped ({status})'
    if labels.get('s4d.purpose') == 'runtime-integrity' and 's4d.run_id' not in labels:
        if (labels.get('s4d.runtime_manifest') == RUNTIME_MANIFEST and name == f'/s4d-runtime-integrity-{RUNTIME_MANIFEST[:16]}'
                and full_id == INTEGRITY_ID):
            row.update(kind='runtime_integrity', keep_dir=f'{REMOTE}/runtime/container-archive/{name.lstrip("/")}')
            return row, None
        return row, 'ambiguous runtime-integrity container'
    if host != 'remote' or not RUN_ID.fullmatch(run_id) or not attempt.isdigit() or name != f'/s4d-{run_id}-a{attempt}':
        return row, 'labels or name not ours for a run'
    if run_id not in {r for r, _ in recorded.get(full_id, set())}:  # container ids are unique
        return row, 'no registry record of this run with this container id'
    s = state.get(run_id, {})
    if s.get('unknown') or s.get('status') in jobs.ACTIVE or s.get('status') in (None, jobs.PENDING):
        return row, f"registry state {s.get('status')}{' unknown' if s.get('unknown') else ''}"
    row['registry_status'] = s.get('status')
    if verification:
        row.update(kind='verification', keep_dir=f'{REMOTE}/runtime/container-archive/{name.lstrip("/")}')
    elif '--archive-missing-rundir' in sys.argv and run_id in MISSING_RUN_DIRS:
        # failed attempts whose runs/<run_id> no longer exists on the remote (refused with exit 4 by the first
        # --apply): their archive goes to runtime/container-archive/<name> instead
        row.update(kind='job_without_run_dir', keep_dir=f'{REMOTE}/runtime/container-archive/{name.lstrip("/")}')
    else:
        row.update(kind='job', keep_dir=f'{REMOTE}/runs/{run_id}/containers')
    return row, None


def removal_script(row):
    q = shlex.quote
    name, keep = row['name'], row['keep_dir']
    log, record, diff = f'{keep}/{name}.docker.log', f'{keep}/{name}.inspect.json', f'{keep}/{name}.diff.txt'
    receipt = f'{keep}/{name}.removal-receipt'
    script = _REMOVAL_SCRIPT
    run_dir_check = (f"test -d {q(REMOTE + '/runs/' + row['run_id'])} || exit 4\n" if row['kind'] == 'job' else '')
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
state = jobs.read_registry()
recorded = registry_ids()
plan, kept = [], []
for cid, d in sorted(details.items()):
    row, why = classify(cid, d, state, recorded)
    (kept if why else plan).append({**row, **({'kept': why} if why else {})})
if '--archive-missing-rundir' in sys.argv:  # this mode handles only those six containers
    kept += [{**p, 'kept': 'not part of the run-dir-missing follow-up'} for p in plan if p['kind'] != 'job_without_run_dir']
    plan = [p for p in plan if p['kind'] == 'job_without_run_dir']
others = [{'name': r[1], 'state': r[2]} for r in rows if r[3] != 'true']
record = {'recorded_local': datetime.now().astimezone().isoformat(), 'host': 'remote',
          'containers_total': len(rows), 'ours_stopped_inspected': len(details), 'running_ours': sum(1 for r in rows if r[3] == 'true' and r[2] == 'running'),
          'not_ours_listed_only': others, 'plan': plan, 'kept': kept}
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
results = []
for row in plan:
    r = ssh(scripts[row['id']], timeout=600)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''
    try:
        payload = json.loads(last)
    except ValueError:
        payload = None
    results.append({'name': row['name'], 'id': row['id'], 'returncode': r.returncode, 'result': payload,
                    'stderr': r.stderr[-300:] if r.returncode else ''})
    print(row['name'], r.returncode, last[:120])
after = ssh("docker ps -a --no-trunc --format '{{.ID}}|{{.Names}}|{{.State}}|{{.Label \"s4d.managed\"}}'; echo __END__")
remaining = {line.split('|')[0] for line in after.stdout.split('__END__')[0].strip().splitlines()}
for res in results:
    res['verified_absent'] = res['id'] not in remaining
record.update(applied_local=datetime.now().astimezone().isoformat(), results=results,
              removed_verified=sum(r['verified_absent'] for r in results),
              containers_after=len(remaining))
OUT.write_text(json.dumps(record, indent=1) + '\n')
print(f"removed (verified absent) {record['removed_verified']} of {len(plan)}; containers now {len(remaining)}")
