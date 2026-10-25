"""Pre-registered fallback trigger of the wide encoder pilots w4 and w8 (runs/setup/encp-definition-20261025.json).

A pretraining run diverged if, within its first 20k steps,
  (a) the total loss became non-finite (the trainer logs "non-finite loss at step N" and raises), or a logged loss is
      non-finite, at a step <= 20000; or
  (b) after warm-up (train.warmup_steps = 10000) the logged training InfoNCE stayed at chance for 2000 consecutive steps:
      every logged value (every 50 steps) >= 0.95 * ln((16 * 8 - 1) / (8 - 1)) = 0.95 * ln(127 / 7) (B = 16 windows,
      6 training cameras + 2 synthetic views per window; the multi-positive InfoNCE at uniform similarities) over logged
      steps s with 10000 < s <= 20000 spanning >= 2000 steps without a gap.
Then the arm/task is restarted once with train.encoder_lr_mult = 0.6 (rows enc-<arm>lr06-*); both attempts are reported.
The check never stops or changes any run. Output: runs/setup/encp-fallback-status.json.

    python -I enc_fallback_check.py
"""
import json
import math
import re
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
TASKS = ('hammer', 'pick-place', 'bin-picking')
ARMS = ('w4', 'w8')
WINDOW, WARMUP, SPAN = 20_000, 10_000, 2_000
CHANCE = math.log((16 * 8 - 1) / (8 - 1))
THRESHOLD = 0.95 * CHANCE
NONFINITE = re.compile(r'non-finite (?:total )?loss at step (\d+)')


def run_dir(job_id):
    for d in (ROOT / 'runs/pretrain' / job_id, ROOT / 'runs' / job_id):
        if (d / 'metrics.jsonl').is_file() or (d / 'console.log').is_file():
            return d
    return None


def check(job_id):
    d = run_dir(job_id)
    if d is None:
        return {'status': 'not started'}
    rows = []
    if (d / 'metrics.jsonl').is_file():
        for line in (d / 'metrics.jsonl').read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    train = sorted(((r['step'], r) for r in rows if 'loss/infonce' in r), key=lambda x: x[0])
    last = train[-1][0] if train else 0
    texts = [p.read_text(errors='replace') for p in (d / 'console.log', ROOT / 'runs' / job_id / 'console.log', d / 'log.txt')
             if p.is_file()]
    steps = [int(m) for t in texts for m in NONFINITE.findall(t)]
    bad = [s for s in steps if s <= WINDOW] + [s for s, r in train if s <= WINDOW and any(
        isinstance(v, float) and not math.isfinite(v) for k, v in r.items() if k.startswith('loss/'))]
    if bad:
        return {'status': 'triggered', 'reason': f'non-finite loss at step {min(bad)}', 'last_logged_step': last}
    run_start = prev = None
    for s, r in train:
        if s <= WARMUP or s > WINDOW:
            continue
        if r['loss/infonce'] >= THRESHOLD and (prev is None or s - prev <= 50):
            run_start = s if run_start is None else run_start
            if s - run_start >= SPAN:
                return {'status': 'triggered', 'reason': f'InfoNCE >= {THRESHOLD:.3f} from step {run_start} to {s}',
                        'last_logged_step': last}
        else:
            run_start = None if r['loss/infonce'] < THRESHOLD else s
        prev = s
    return {'status': 'ok' if last >= WINDOW else 'pending', 'last_logged_step': last}


def main():
    out = {'checked': datetime.now().astimezone().isoformat(), 'chance': CHANCE, 'threshold': THRESHOLD,
           'runs': {f'enc-{a}-seed0-{t}-200k': check(f'enc-{a}-seed0-{t}-200k') for a in ARMS for t in TASKS}}
    (ROOT / 'runs/setup/encp-fallback-status.json').write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out['runs'].items():
        print(k, v)


if __name__ == '__main__':
    main()
