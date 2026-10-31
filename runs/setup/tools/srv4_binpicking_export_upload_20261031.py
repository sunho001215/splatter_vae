"""Local shutdown 2026-10-31: once srv4-csw4-seed0-bin-picking-200k is done, admit its export locally (one row, through
the scheduler), wait for it, then upload runs/srv4-csw4-export-seed0-bin-picking-200k/encoder.pt to the remote with a
sha256 check and write the remote driver's export-upload marker (runs/remote/exports/<path>.json), so the remote runner
can start the three srv4 bin-picking FlashSAC seeds during the outage."""
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402
from s4d.remote_access import RemoteClient  # noqa: E402

PRE, EXP = 'srv4-csw4-seed0-bin-picking-200k', 'srv4-csw4-export-seed0-bin-picking-200k'
GPU = 'MIG-c3511b41-5f78-52f1-9b93-9c6794cdbbdd'
EXPORT = f'runs/{EXP}/encoder.pt'


def status(jid):
    return jobs.read_registry().get(jid, {}).get('status', jobs.PENDING)


deadline = time.time() + 3 * 3600
while status(PRE) != jobs.DONE:
    assert time.time() < deadline and status(PRE) in jobs.ACTIVE | {jobs.PENDING}, status(PRE)
    time.sleep(30)
print('pretraining done', datetime.now().strftime('%T'), flush=True)
if status(EXP) == jobs.PENDING:
    out = subprocess.run([str(ROOT / '.venv/bin/python'), '-I', str(ROOT / 'runs/setup/tools/admit_one_local_20261031.py'),
                          EXP, GPU], capture_output=True, text=True, timeout=600)
    print(out.stdout[-2000:], out.stderr[-2000:], flush=True)
while status(EXP) != jobs.DONE:
    assert time.time() < deadline and status(EXP) in jobs.ACTIVE | {jobs.PENDING}, status(EXP)
    time.sleep(15)
digest = hashlib.sha256((ROOT / EXPORT).read_bytes()).hexdigest()
client = RemoteClient()
client.rsync_upload(ROOT / EXPORT, f'~/kaist/sunho/{EXPORT}')
remote = client.checked_ssh(f'sha256sum /home/compu/kaist/sunho/{EXPORT}', timeout=300).stdout.split()[0]
assert remote == digest, (remote, digest)
marker = ROOT / 'runs/remote/exports' / (EXPORT.replace('/', '__') + '.json')
marker.parent.mkdir(parents=True, exist_ok=True)
marker.write_text(json.dumps({'export': EXPORT, 'sha256': digest, 'uploaded': datetime.now().astimezone().isoformat()}) + '\n')
print(json.dumps({'uploaded': EXPORT, 'sha256': digest, 'time': datetime.now().strftime('%T')}), flush=True)
