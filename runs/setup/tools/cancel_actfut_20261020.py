"""Cancel the superseded future-prediction campaign `action_future` (history stride = future stride = k in {2,4,6};
registration runs/setup/actfut-registration-hammer-pick-place-20261020.json), user instruction 2026-10-20: it is
replaced by the fixed-history / sampled-future-stride variant. No observation-time-embedding variant was registered.

--pending : record a terminal `failed` event (cancelled: true, reason, superseded_by) for every job of the registration
            that never started, so neither a driver nor a retry can launch it (the local driver no longer lists the
            registration; the scheduler daemon only launches under drivers).
--finalize: for the running pretrains, after their lossless checkpoint pause (pause_pretrain_at_checkpoint.py) has
            exited, record the same terminal event, and write runs/setup/actfut-cancellation-20261020.json with
            run ids, last published checkpoint (step, path, sha256), last logged step and the reason.
Completed results (none exist) would keep their status; partial runs are preserved, never reused or relabelled.

    python -I cancel_actfut_20261020.py --pending | --finalize
"""
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

REASON = ('cancelled: superseded on 2026-10-20 by the replacement future-prediction variant (fixed history stride 2, '
          'future stride k2 in 1..6); user instruction')
REG = ROOT / 'runs/setup/actfut-registration-hammer-pick-place-20261020.json'
OUT = ROOT / 'runs/setup/actfut-cancellation-20261020.json'


def record(event):
    with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with jobs.REGISTRY.open('a') as stream:
            stream.write(json.dumps({**event, 'time': time.time()}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ids = [j['id'] for j in json.loads(REG.read_text())['jobs']]
    state = jobs.read_registry()
    record_out = json.loads(OUT.read_text()) if OUT.is_file() else {'reason': REASON, 'registration': str(REG.relative_to(ROOT)),
                                                                      'jobs': {}}
    if '--pending' in sys.argv:
        for jid in ids:
            cur = state.get(jid, {})
            if cur.get('status', jobs.PENDING) == jobs.PENDING and not cur.get('attempts'):
                record({'event': 'failed', 'id': jid, 'host': 'local', 'cancelled': True, 'reason': REASON,
                        'superseded': True})
                record_out['jobs'][jid] = {'state_before': 'pending (never started)', 'cancelled_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
                print('cancelled pending', jid)
    if '--finalize' in sys.argv:
        for jid in ids:
            cur = state.get(jid, {})
            if not cur.get('attempts') or jid in record_out['jobs'] and 'last_checkpoint' in record_out['jobs'][jid]:
                continue
            if cur['status'] in jobs.ACTIVE:
                raise SystemExit(f'{jid} is still {cur["status"]}; finalize after its pause has exited')
            run = ROOT / 'runs/pretrain' / jid
            ckpts = sorted((run / 'checkpoints').glob('step_*.pt'))
            last = ckpts[-1] if ckpts else None
            logged = None
            for line in (run / 'metrics.jsonl').read_text().splitlines()[::-1]:
                if '"step"' in line:
                    logged = json.loads(line).get('step')
                    break
            record({'event': 'failed', 'id': jid, 'host': 'local', 'cancelled': True, 'reason': REASON, 'superseded': True})
            record_out['jobs'][jid] = {
                'state_before': f"running (attempt {cur.get('attempts')}), paused at a published checkpoint",
                'device': cur.get('gpu'), 'commit': cur.get('commit'), 'run_dir': str(run.relative_to(ROOT)),
                'last_checkpoint': None if last is None else {'path': str(last.relative_to(ROOT)), 'step': int(last.stem.split('_')[1]),
                                                              'sha256': sha(last)},
                'last_logged_step': logged, 'pause_events': cur.get('intentional_stops'),
                'cancelled_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
            print('cancelled running', jid, record_out['jobs'][jid]['last_checkpoint'])
    record_out['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    OUT.write_text(json.dumps(record_out, indent=1))


if __name__ == '__main__':
    main()
