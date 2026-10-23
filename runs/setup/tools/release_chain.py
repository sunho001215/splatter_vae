"""Deploy and verify a remote release (the established typed-verification route; image unchanged):
sync the pushed commit's bundle to ~/kaist/sunho/releases/<commit>/code (identity: clean pushed ancestry, checksum),
run the remote native suite (functional check of the new code in the accepted image) on one authorized GPU, then the four
one-UUID CUDA/EGL isolation checks, and assemble the campaign acceptance manifest. Logs to stdout; one tick per minute.
    python -I release_chain.py <commit> [--native-gpu <uuid>]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402
from s4d.remote_jobs import RemoteBackend  # noqa: E402

PY = str(ROOT / '.venv/bin/python')
commit = sys.argv[1]
short = commit[:7]
native_gpu = sys.argv[sys.argv.index('--native-gpu') + 1] if '--native-gpu' in sys.argv else None


def log(msg):
    print(time.strftime('%T'), msg, flush=True)


def run(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True)
    log(f'{" ".join(cmd[-3:])} -> {out.returncode} {out.stdout.strip()[-300:]} {out.stderr.strip()[-300:]}')
    if out.returncode:
        raise SystemExit(out.returncode)
    return out.stdout.strip().splitlines()[-1]


def drive(qpath, ids):
    backend = RemoteBackend(ROOT, ROOT / 'runs')
    while True:
        for m in jobs.tick(queue_path=Path(qpath), remote_backend=backend):
            log(m)
        state = jobs.read_registry()
        status = {i: state.get(i, {}).get('status') for i in ids}
        if all(s == jobs.DONE for s in status.values()):
            log(f'done {status}')
            return
        if any(s == jobs.FAILED for s in status.values()):
            log(f'FAILED {status}')
            raise SystemExit(2)
        log(f'wait {status}')
        time.sleep(60)


log(run([PY, '-I', str(ROOT / 's4d/remote_access.py'), 'sync-code', commit]))
qpath = run([PY, '-I', str(ROOT / 'runs/setup/tools/prepare_remote_verification.py'), commit])
if native_gpu:  # the native suite may run on any authorized GPU; use an idle one
    q = yaml.safe_load(Path(qpath).read_text())
    q['jobs'][0]['gpu'] = native_gpu
    Path(qpath).write_text(yaml.safe_dump(q, sort_keys=False))
drive(qpath, [f'verify-native-{short}'])
suite = ROOT / f'runs/verify-native-{short}/evidence/tests.json'
log(json.dumps({k: v for k, v in json.loads(suite.read_text()).items() if k in ('counts', 'exit_code', 'full_suite_passed')}))
qpath = run([PY, '-I', str(ROOT / 'runs/setup/tools/prepare_remote_verification.py'), commit, '--suite', str(suite)])
drive(qpath, [f'verify-isolation-{i}-{short}' for i in range(4)])
log(run([PY, '-I', str(ROOT / 'runs/setup/tools/assemble_remote_acceptance.py'), commit]))
