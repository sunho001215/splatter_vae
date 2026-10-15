"""Pull remote offline W&B run folders (checksum-verified) and upload finished runs with `wandb sync`.

W&B is visualization only: failures are recorded in runs/remote/wandb/status.json and never block the campaign.
Usage: .venv/bin/python -I runs/setup/tools/sync_wandb_remote.py [--run ID ...] [--no-upload]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(REPO))
from scripts import jobs  # noqa: E402
from s4d.remote_access import RemoteClient  # noqa: E402

ROOT = '/home/compu/kaist/sunho'
STATUS = REPO / 'runs/remote/wandb/status.json'
MEDIA_KINDS = ('images', 'object3D', 'videos', 'table')
FINISHED = {jobs.DONE, jobs.FAILED, 'crashed'}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def remote_listing(client: RemoteClient, run_id: str) -> dict[str, tuple[int, str]]:
    base = f'{ROOT}/runs/{run_id}'
    out = client.checked_ssh(
        f'set -eu; cd {shlex.quote(base)} 2>/dev/null || exit 0; [ -d wandb ] || exit 0; '
        "find wandb -type f -printf '%s\\t%p\\n' | while IFS=$'\\t' read -r size path; do "
        'printf "%s\\t%s\\t%s\\n" "$size" "$(sha256sum -- "$path" | cut -d" " -f1)" "$path"; done',
        wait=False, timeout=600, idempotent=True).stdout
    listing = {}
    for line in out.splitlines():
        size, digest, path = line.split('\t', 2)
        listing[path] = (int(size), digest)
    return listing


def media_inventory(local_wandb: Path) -> dict[str, int]:
    counts = dict.fromkeys(MEDIA_KINDS, 0)
    for path in local_wandb.glob('*/files/media/*/**/*'):
        if path.is_file():
            kind = path.relative_to(local_wandb).parts[3]
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append')
    parser.add_argument('--no-upload', action='store_true')
    args = parser.parse_args()
    state = jobs.read_registry()
    runs = args.run or sorted(i for i, s in state.items()
                              if s.get('host') == 'remote' and not i.startswith('verify-') and s.get('attempts'))
    status = json.loads(STATUS.read_text()) if STATUS.is_file() else {}
    client = RemoteClient()
    for run_id in runs:
        entry = status.setdefault(run_id, {'uploaded': {}, 'errors': []})
        job_status = state.get(run_id, {}).get('status')
        try:
            listing = remote_listing(client, run_id)
        except Exception as exc:  # noqa: BLE001  (visualization only)
            entry['errors'].append(f'{datetime.now().astimezone().isoformat()} listing: {type(exc).__name__}: {exc}')
            continue
        entry.update(job_status=job_status, remote_files=len(listing),
                     remote_bytes=sum(size for size, _ in listing.values()))
        if not listing:
            entry['note'] = 'no offline W&B folder yet'
            continue
        local = REPO / 'runs' / run_id
        local.mkdir(parents=True, exist_ok=True)
        try:
            client._rsync(f'{client.target}:{ROOT}/runs/{run_id}/wandb/', str(local / 'wandb') + '/', wait=False)
        except Exception as exc:  # noqa: BLE001
            entry['errors'].append(f'{datetime.now().astimezone().isoformat()} rsync: {type(exc).__name__}: {exc}')
            continue
        mismatched = [p for p, (size, digest) in listing.items()
                      if not (local / p).is_file() or sha(local / p) != digest]
        entry.update(pulled_at=datetime.now().astimezone().isoformat(), checksum_mismatches=mismatched,
                     media=media_inventory(local / 'wandb'))
        if args.no_upload or job_status not in FINISHED or mismatched:
            entry['upload'] = 'pending: ' + ('job running' if job_status not in FINISHED else
                                             'checksum mismatch' if mismatched else 'upload disabled')
            continue
        for offline in sorted((local / 'wandb').glob('offline-run-*')):
            if entry['uploaded'].get(offline.name):
                continue
            try:
                result = subprocess.run([str(REPO / '.venv/bin/wandb'), 'sync', '--mark-synced', str(offline)],
                                        capture_output=True, text=True, timeout=1800, cwd=str(local))
                ok = result.returncode == 0
                detail = (result.stdout + result.stderr)[-600:]
            except subprocess.TimeoutExpired:
                ok, detail = False, 'timeout after 1800 s'
            if ok:
                entry['uploaded'][offline.name] = datetime.now().astimezone().isoformat()
            else:
                entry['errors'].append(f'{datetime.now().astimezone().isoformat()} wandb sync {offline.name}: {detail}')
        pending = [o.name for o in (local / 'wandb').glob('offline-run-*') if not entry['uploaded'].get(o.name)]
        entry['upload'] = 'complete' if not pending else f'not yet synced: {pending}'
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True))
    os.replace(tmp, STATUS)
    not_synced = sorted(r for r, e in status.items() if e.get('remote_files') and e.get('upload') != 'complete')
    print(json.dumps({'checked': len(runs), 'not_yet_synced': not_synced,
                      'remote_wandb_bytes': sum(e.get('remote_bytes', 0) for e in status.values())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
