"""One-time cleanup of our completed containers on the shared remote Docker host (user instruction 2026-10-21).

Inventory (read-only, default): every container on the host is listed and classified. A container is ours only with
positive evidence from all three sources: the project labels (s4d.managed=true, s4d.host=remote, s4d.run_id,
s4d.attempt; the s4d.project/s4d.owner labels too when present), the deterministic name s4d-<run_id>-a<attempt>, and
a registry launch record (experiments/registry.jsonl) of that run and attempt carrying exactly this container id.
Names or timestamps alone never count. Anything else is reported and left alone: containers of other people
(no s4d.managed label) are not even inspected beyond the listing.

Eligible for removal (ours, stopped, nothing needs it):
  - the final attempt of a job that is done, whose final results were synced and checksum-verified, exit code 0;
  - an earlier, failed or interrupted attempt of such a job (its recovery is complete), when the registry recorded
    that attempt's exit code and it equals the container's.
Retained: running, paused or otherwise not exited containers; jobs that are running, pending, crashed (retryable),
failed, unknown or unsynced; verification containers; anything ambiguous.

    .venv/bin/python -I runs/setup/tools/remote_container_cleanup.py            # inventory and plan (no change)
    .venv/bin/python -I runs/setup/tools/remote_container_cleanup.py --apply    # remove the eligible ones

--apply removes each eligible container separately through RemoteBackend.remove_container (s4d/remote_jobs.py), which
re-checks identity, labels, stopped state and exit code by explicit id immediately before `docker rm` (no force, no
volumes, no prune), after checking the run's result files and saving the container's Docker log and inspect record
under the run directory. Each outcome is recorded in the registry (container_removed / container_kept), so a rerun
after a lost connection continues where it stopped (container_absent when it is already gone).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402
from s4d.remote_access import RemoteClient, RemoteUnreachable  # noqa: E402

REPORT = ROOT / 'runs/setup/remote-container-cleanup-20261021.json'
FINAL_FILES = ('exit.json', 'provenance.json', 'console.log')
OWNERSHIP = {'s4d.project': 'splatter4d', 's4d.owner': 'kaist-sunho'}
INVENTORY = ("set -eu; export LC_ALL=C; ids=$(docker ps -aq --no-trunc); "
             "if [ -n \"$ids\" ]; then docker inspect --format "
             "'{{.Id}}|{{.Name}}|{{.State.Status}}|{{.State.ExitCode}}|{{.Created}}|{{json .Config.Labels}}' $ids; fi")


def registry_evidence():
    launched, codes, settled = {}, {}, set()
    for line in jobs.REGISTRY.read_text().splitlines():
        event = json.loads(line)
        kind = event['event']
        if kind == 'launched' and event.get('container_id'):
            launched.setdefault((event['id'], int(event['attempt'])), set()).add(event['container_id'])
        elif kind == 'exited' and 'attempt' in event:
            codes[(event['id'], int(event['attempt']))] = event['code']
        elif kind in ('container_removed', 'container_kept', 'container_absent'):
            settled.add(event.get('container_id'))
        elif kind == 'container_kept_cleared':  # a refusal caused by a defect of the check itself, since fixed
            settled.discard(event.get('container_id'))
    return launched, codes, settled


def classify(item, state, launched, codes, settled):
    labels = item['labels'] or {}
    if labels.get('s4d.managed') != 'true':
        return 'not_ours', 'no s4d.managed label (another user or unrelated)'
    run_id, attempt, host = labels.get('s4d.run_id'), labels.get('s4d.attempt'), labels.get('s4d.host')
    if host != 'remote' or not run_id or not str(attempt or '').isdigit():
        return 'ambiguous', 'project label without complete host/run/attempt labels'
    attempt = int(attempt)
    if item['name'] != f's4d-{run_id}-a{attempt}':
        return 'ambiguous', 'name does not match its labels'
    if any(key in labels and labels[key] != value for key, value in OWNERSHIP.items()):
        return 'ambiguous', 'ownership labels differ'
    if item['id'] not in launched.get((run_id, attempt), set()):
        return 'ambiguous', 'no registry launch record with this container id'
    if item['id'] in settled:
        return 'ambiguous', 'registry already records it removed or kept'
    if item['status'] != 'exited':
        return 'retained', f'container {item["status"]}'
    if labels.get('s4d.verification_purpose'):
        return 'retained', 'verification container'
    job = state.get(run_id, {})
    if job.get('verification_purpose') is not None:
        return 'retained', 'verification job'
    if job.get('status') != jobs.DONE or job.get('host') != 'remote':
        return 'retained', f'job {job.get("status", "unregistered")} (may be retried or diagnosed)'
    if (job.get('sync') or {}).get('final') is not True:
        return 'retained', 'final results not synced and verified'
    final = int(job.get('attempt', job.get('attempts', 0)))
    if attempt == final:
        if item['id'] != job.get('container_id'):
            return 'ambiguous', 'final attempt id differs from the job record'
        if item['code'] != 0:
            return 'ambiguous', f'final attempt exit code {item["code"]}'
        return 'eligible', 'final attempt of a completed, verified job'
    if attempt > final:
        return 'ambiguous', 'attempt newer than the job record'
    if (run_id, attempt) not in codes:
        return 'retained', 'earlier attempt without a recorded exit code'
    if codes[(run_id, attempt)] != item['code']:
        return 'ambiguous', 'earlier attempt exit code differs from the record'
    return 'eligible', 'earlier attempt of a completed, verified job'


def inventory(client):
    result = client.ssh(INVENTORY, wait=False, idempotent=True, timeout=180)
    if result.returncode:
        raise SystemExit(f'inventory failed (exit {result.returncode})')
    items = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        cid, name, status, code, created, labels = line.split('|', 5)
        items.append({'id': cid, 'name': name.lstrip('/'), 'status': status, 'code': int(code), 'created': created,
                      'labels': json.loads(labels) if labels.strip() not in ('', 'null') else {}})
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    client = RemoteClient()
    items = inventory(client)
    state = jobs.read_registry()
    launched, codes, settled = registry_evidence()
    rows = []
    for item in items:
        category, reason = classify(item, state, launched, codes, settled)
        row = {'name': item['name'], 'id': item['id'], 'status': item['status'], 'category': category, 'reason': reason}
        if category != 'not_ours':
            row.update(code=item['code'], created=item['created'],
                       run_id=item['labels'].get('s4d.run_id'), attempt=item['labels'].get('s4d.attempt'))
        rows.append(row)
    summary = {}
    for row in rows:
        key = f"{row['category']}: {row['reason']}"
        summary[key] = summary.get(key, 0) + 1
    report = {'time': time.strftime('%F %T'), 'total': len(rows), 'summary': dict(sorted(summary.items())),
              'containers': rows, 'applied': args.apply}
    print(json.dumps({k: report[k] for k in ('time', 'total', 'summary')}, indent=1))
    if args.apply:
        from s4d.remote_jobs import ContainerRemovalRefused, RemoteBackend, RemoteJobError
        backend = RemoteBackend(ROOT, ROOT / 'runs', client)
        outcomes = {}
        for row in rows:
            if row['category'] != 'eligible':
                continue
            job = state[row['run_id']]
            attempt = int(row['attempt'])
            required = sorted(set(FINAL_FILES) | {str(v) for v in job.get('required_results', [])})
            base = {'event': 'container_removed', 'id': row['run_id'], 'host': 'remote', 'attempt': attempt,
                    'container': row['name'], 'container_id': row['id'], 'source': 'initial-cleanup-20261021'}
            try:
                result = backend.remove_container(row['run_id'], attempt, row['name'], row['id'], row['code'], required)
            except RemoteUnreachable:
                row['outcome'] = 'not attempted (host unreachable); rerun to continue'
                break
            except ContainerRemovalRefused as exc:
                jobs.record({**base, 'event': 'container_kept', 'reason': exc.reason})
                row['outcome'] = f'kept: {exc.reason}'
            except (RemoteJobError, OSError, ValueError, RuntimeError) as exc:
                row['outcome'] = f'kept for now: {type(exc).__name__}: {str(exc)[:200]}'
            else:
                if result.get('absent'):  # gone already; hashes come from our receipt if an earlier run removed it
                    jobs.record({**base, **result, 'event': 'container_absent'})
                    row['outcome'] = 'already absent' + (' (receipt)' if 'log_sha256' in result else '')
                else:
                    jobs.record({**base, **result})
                    row['outcome'] = 'removed'
            outcomes[row['outcome'].split(':')[0]] = outcomes.get(row['outcome'].split(':')[0], 0) + 1
        report['outcomes'] = outcomes
        print(json.dumps({'outcomes': outcomes}))
    path = REPORT if not args.apply else REPORT.with_name(REPORT.stem + '-applied.json')
    path.write_text(json.dumps(report, indent=1) + '\n')
    print(f'report: {path}')


if __name__ == '__main__':
    main()
