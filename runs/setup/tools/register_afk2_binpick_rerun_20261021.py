"""Fresh retraining of future_k2 bin-picking from scratch under a new run ID (user request 2026-10-21, spike
investigation runs/setup/afk2-binpick-spike-20261021/). The original afk2-seed0-bin-picking-200k and its artifacts stay
untouched and running. The new row equals the original's apart from the name/paths and, only with --with-fix, the
verified fix (loss.visibility_front_only=true): same base/task/variant config, seed 0, 200k steps, resources,
restarts and required results. Logged to W&B (the training default). Registered before any result of the new run.

    python -I register_afk2_binpick_rerun_20261021.py [--with-fix] [--publish]
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--with-fix', action='store_true')
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
ORIGINAL = 'afk2-seed0-bin-picking-200k'
NEW = 'afk2fix-seed0-bin-picking-200k' if args.with_fix else 'afk2rerun-seed0-bin-picking-200k'
queue = jobs.load_queue()
by_queue = {j['id']: j for j in queue['jobs']}
orig = by_queue[ORIGINAL]
new = json.loads(json.dumps(orig))
new['id'] = NEW
args_ = new['args']
args_[args_.index('--name') + 1] = NEW
if args.with_fix:
    args_.append('loss.visibility_front_only=true')
new['investigation'] = 'runs/setup/afk2-binpick-spike-20261021'
new['rerun_of'] = ORIGINAL
# the protocol equals the original row apart from the name and (optionally) the fix
diff = {k for k in set(orig) | set(new) if orig.get(k) != new.get(k)}
assert diff == {'id', 'args', 'investigation', 'rerun_of'}, diff
assert [a for a in args_ if a not in orig['args']] == ([NEW, 'loss.visibility_front_only=true'] if args.with_fix else [NEW])
assert NEW not in by_queue, 'already registered'
commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
proof = ROOT / f'runs/setup/{NEW}-registration-20261021.json'
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0], 'commit': commit,
          'original': orig, 'jobs': [new], 'differences_from_original': sorted(diff),
          'fix': 'loss.visibility_front_only=true' if args.with_fix else None}
if not args.publish:
    print(json.dumps({'id': NEW, 'args': args_, 'commit': commit}, indent=1))
    raise SystemExit(0)
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not proof.exists()
    before = jobs.QUEUE.read_bytes()
    text = before.decode()
    node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    staged = ROOT / f'runs/setup/{NEW}-queue-candidate-20261021.yaml'
    staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                      + yaml.safe_dump(json.loads(json.dumps([new])), sort_keys=False) + '\n' + text[node.end_mark.index:])
    assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + [new]
    tmp = jobs.QUEUE.with_suffix('.rerun.tmp')
    tmp.write_bytes(staged.read_bytes())
    os.replace(tmp, jobs.QUEUE)
    record.update(registered_at=datetime.now().astimezone().isoformat(),
                  queue_sha256_before=hashlib.sha256(before).hexdigest(),
                  queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
    proof.write_text(json.dumps(record, indent=2) + '\n')
    print('REGISTERED', NEW, proof)
