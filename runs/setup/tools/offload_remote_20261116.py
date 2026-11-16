#!/usr/bin/env python3
"""Offline preview by default; --execute publishes inputs and placement, never launches.

Wait forecasts are explicit estimates, in seconds (globally or per exact ID).
Without both estimates a row stays local. Inventory files may contain remote_plan
relative names or absolute targets; cached evidence is only a preview optimization.
Execute checks the remote files again and serializes the 6 MB/s upload budget.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE))
from ops.jobs import host_roles, scheduler  # noqa: E402
from ops.jobs import remote_offload as offload  # noqa: E402

MAIN = Path('/home/ws/ws/splatter4d')


def cached_inventory(root):
    result = {}
    path = root / 'runs/realworld-inputs/MANIFEST.json'
    if path.exists():
        for name, item in json.loads(path.read_bytes())['files'].items():
            # Local publications deliberately use remote-shaped locators. They
            # are NOT evidence that the corresponding remote file exists.
            if item.get('origin') or name.startswith('_remote/'):
                continue
            if item.get('remote_sha256'):
                result[offload.local.REMOTE_PREFIX + '/' + name] = dict(sha256=item['remote_sha256'])
    for identity, state in scheduler.read_registry(root / 'experiments/registry.jsonl').items():
        if state.get('host') == 'remote':
            for name, digest in state.get('remote_verified', {}).get('files', {}).items():
                result[f'{offload.REMOTE_ROOT}/runs/{identity}/{name}'] = dict(sha256=digest)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ids', nargs='*', help='default: all approved local S rows')
    p.add_argument('--repo', type=Path, default=MAIN)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--stop-file', type=Path, default=MAIN / 'runs/setup/STOP-OFFLOAD-REMOTE')
    p.add_argument('--inventory', type=Path, help='cached checksum inventory; no SSH in preview')
    p.add_argument('--forecast', type=Path, help='JSON: checked_at, rows: {id: {local_wait_seconds, remote_start_seconds}}')
    p.add_argument('--max-forecast-age', type=float, default=300)
    p.add_argument('--local-wait-seconds', type=float, help='explicit expected local wait for this pass')
    p.add_argument('--remote-start-seconds', type=float, help='expected remote admission delay after transfer')
    p.add_argument('--upload-mbps', type=float, default=6, help='aggregate measured MB/s, decimal bytes')
    a = p.parse_args(argv)
    started = time.monotonic()
    root = a.repo.resolve()
    if a.execute:
        offload.require(SOURCE == root == MAIN and Path.cwd().resolve() == MAIN,
                        'execute only from the adopted coordinator copy')
        # This lock covers planning + transfer, never blocks either admission driver.
        upload_lock = (root / 'runs/setup/offload-upload.lock').open('a')
        try:
            fcntl.flock(upload_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(moved=[], busy=True)))
            return 0
    queue = scheduler.load_queue(root / 'experiments/queue.yaml')
    state = scheduler.read_registry(root / 'experiments/registry.jsonl')
    ownership = host_roles.read_ownership(root / 'runs/setup/ownership')
    roles = host_roles.load_roles(root / 'configs/hosts/roles.yaml')
    holds = offload.pauses(root)
    by_id = {j['id']: j for j in queue['jobs']}
    inventory = cached_inventory(root)
    if a.inventory:
        doc = json.loads(a.inventory.read_bytes())
        for name, item in doc.get('files', doc).items():
            target = name if name.startswith('/') else offload.local.REMOTE_PREFIX + '/' + offload.local._rel(name)
            inventory[target] = item
    forecasts, forecast_age = {}, 0
    if a.forecast:
        doc = json.loads(a.forecast.read_bytes())
        forecast_age = time.time() - doc['checked_at']
        offload.require(0 <= forecast_age <= a.max_forecast_age, 'stale wait forecast')
        forecasts = doc['rows']
    ids = a.ids or [j['id'] for j in queue['jobs'] if offload.valid_s(j) and j.get('host') == 'local']
    offload.require(len(ids) == len(set(ids)) and all(i in by_id for i in ids), 'unique registered IDs required')
    report, queued_bytes, client = [], 0, None
    for identity in sorted(ids, key=lambda i: offload.priority(by_id[i], by_id)):
        if a.stop_file.exists():
            break
        job = by_id[identity]
        decision = dict(id=identity, decision='stay_local', pin=[job.get('planned_commit'), job.get('planned_image')])
        try:
            offload.untouched(job, state, ownership, root, holds, roles, by_id, remote_host=queue['hosts']['remote'])
            # Reject an unavailable exact release before hashing potentially very
            # large stores. No transfer can make an unaccepted pin admissible.
            offload.release_manifest(job, root)
            items = offload.input_files(job, root, state)
            forecast = forecasts.get(identity, {})
            local_wait = forecast.get('local_wait_seconds', a.local_wait_seconds)
            remote_start = forecast.get('remote_start_seconds', a.remote_start_seconds)
            if local_wait is not None:
                local_wait = max(0, local_wait - (forecast_age if forecast else 0)
                                 - (time.monotonic() - started if a.execute else 0))
            decision['input_bytes'] = sum(i['bytes'] for i in items)
            # Show transfer costs even if the wait is not yet known.
            if not a.execute:
                decision.update(offload.cost(items, inventory, local_wait or 0, remote_start or 0,
                                            bytes_per_second=a.upload_mbps * 1_000_000, queued_bytes=queued_bytes))
            offload.require(local_wait is not None and remote_start is not None, 'local/remote wait forecast required')
            if a.execute:
                if client is None:
                    from ops.remote.access import RemoteClient
                    client = RemoteClient(repo=root)
                from ops.jobs.remote_backend import RemoteBackend
                backend = RemoteBackend(root, root / 'runs', client)
                decision['remote_gpu_room'] = offload.remote_room(job, root, backend.snapshot())
                for item in items:
                    client.ensure_remote_path(item['target'])
                    digest = client._remote_checksum(item['target'], allow_missing=True)
                    if digest is None:
                        inventory.pop(item['target'], None)
                    else:
                        inventory[item['target']] = dict(sha256=digest)
            prepared = offload.prepare(job, root, state, ownership, roles, holds, inventory,
                                       local_wait, remote_start, bytes_per_second=a.upload_mbps * 1_000_000,
                                       queued_bytes=queued_bytes, queue=queue)
            decision.update(prepared['estimate'])
            if a.execute:
                transfer_started = time.monotonic()
                offload.probe(client, prepared)
                for item in prepared['items']:
                    if a.stop_file.exists():
                        raise offload.Refused('STOP requested; row stays local, staged inputs retained')
                    offload.publish(client, item)
                offload.probe(client, prepared)
                offload.remote_room(job, root, backend.snapshot())
                for item in prepared['items']:
                    offload.require(client._remote_checksum(item['target']) == item['sha256'], 'input changed remotely')
                offload.require(remote_start < local_wait - (time.monotonic() - transfer_started),
                                'actual transfer time exhausted the local wait advantage')
                prepared['remote_verified'] = True
                offload.commit_placement(prepared, root)
                decision.update(decision='offloaded', reason='verified inputs; normal remote admission next')
            else:
                decision.update(decision='would_offload', reason='estimated earlier start; cached input evidence')
                queued_bytes += prepared['estimate']['missing_bytes']
            # Deduplicate shared inputs across this serialized pass.
            inventory.update({i['target']: dict(sha256=i['sha256']) for i in prepared['items']})
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            decision['reason'] = str(exc)
        report.append(decision)
    print(json.dumps(dict(execute=a.execute, launches=False, decisions=report), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
