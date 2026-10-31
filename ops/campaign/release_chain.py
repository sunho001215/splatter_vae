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
from ops.jobs import scheduler as jobs  # noqa: E402
from ops.jobs.remote_backend import RemoteBackend  # noqa: E402

PY = str(ROOT / '.venv/bin/python')
commit = sys.argv[1]
IMAGE_ARGS = ['--image', sys.argv[sys.argv.index('--image') + 1], '--tag', sys.argv[sys.argv.index('--tag') + 1]] \
    if '--image' in sys.argv else []
short = commit[:7] + (f'-{IMAGE_ARGS[3]}' if IMAGE_ARGS else '')
native_gpu = sys.argv[sys.argv.index('--native-gpu') + 1] if '--native-gpu' in sys.argv else None
# --sync-repo: a worktree checked out at <commit> (clean), when the main checkout is behind the pushed release commit
sync_repo = Path(sys.argv[sys.argv.index('--sync-repo') + 1]) if '--sync-repo' in sys.argv else ROOT


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


if '--skip-sync' not in sys.argv:  # the release bundle is already deployed (same commit, another image)
    log(run([PY, '-I', str(sync_repo / 's4d/remote_access.py'), 'sync-code', commit]))
qpath = run([PY, '-I', str(ROOT / 'runs/setup/tools/prepare_remote_verification.py'), commit, *IMAGE_ARGS])
if native_gpu:  # the native suite may run on any authorized GPU; use an idle one
    q = yaml.safe_load(Path(qpath).read_text())
    q['jobs'][0]['gpu'] = native_gpu
    Path(qpath).write_text(yaml.safe_dump(q, sort_keys=False))
drive(qpath, [f'verify-native-{short}'])
suite = ROOT / f'runs/verify-native-{short}/evidence/tests.json'
log(json.dumps({k: v for k, v in json.loads(suite.read_text()).items() if k in ('counts', 'exit_code', 'full_suite_passed')}))
qpath = run([PY, '-I', str(ROOT / 'runs/setup/tools/prepare_remote_verification.py'), commit, '--suite', str(suite), *IMAGE_ARGS])
drive(qpath, [f'verify-isolation-{i}-{short}' for i in range(4)])
log(run([PY, '-I', str(ROOT / 'runs/setup/tools/assemble_remote_acceptance.py'), commit, *IMAGE_ARGS]))
