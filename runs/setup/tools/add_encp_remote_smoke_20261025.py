"""Remote smoke rows of the encoder pilots (user directive 2026-10-25: smoke runs per arm on both hosts' images, recording
s/it and peak memory, before registration). Not experiment rows: 600 pretraining steps on hammer per arm (hn, cs, h2,
w4, w8) and chlrm under the same conditions, on the encoder-pilot release (remote driver ENC_COMMIT, derived image);
train.steps=600 (inside the 10k warm-up the learning rate equals the 200k schedule's), no start evaluation, the end
evaluation and checkpoint included. Priority 4 and the enc gates of the remote driver (after every other row, no backfill).

    python -I add_encp_remote_smoke_20261025.py [--publish]
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
from scripts import jobs  # noqa: E402
import pool_placement as pool  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
BASE = ['configs/metaworld/base.yaml', 'configs/metaworld/tasks/hammer.yaml', 'configs/metaworld/variants/chamfer_lean.yaml',
        'configs/metaworld/variants/chamfer_lean_randmask.yaml']
rows = []
for arm in ('chlrm', 'hn', 'cs', 'h2', 'w4', 'w8'):
    jid = f'encsmoke-{arm}-hammer'
    cfgs = BASE + ([f'configs/metaworld/variants/enc_{arm}.yaml'] if arm != 'chlrm' else [])
    rows.append({'id': jid, 'host': 'remote', 'placement': 'pool smoke 2026-10-25 (encoder-pilot release check; not an experiment row)',
                 'script': 'scripts/train.py',
                 'args': ['--config', *cfgs, '--name', jid, '--output-root', 'runs', '--resume', 'auto', '--set',
                          'train.steps=600', 'train.seed=0', 'train.eval_at_start=false'],
                 'gpu': 'any', 'mem_gb': 10, 'ram_gb': 13, 'disk_gb': 30, 'priority': 4, 'deps': [], 'max_restarts': 0,
                 'stage': 5, 'required_results': ['completion.json', 'metrics.jsonl'], 'variant': 'encsmoke'})
queue = jobs.load_queue()
assert not {r['id'] for r in rows} & {j['id'] for j in queue['jobs']}
proof = ROOT / 'runs/setup/encp-remote-smoke-rows-20261025.json'
if not args.publish:
    print(json.dumps(rows[1], indent=1))
    raise SystemExit(0)
with open(pool.ADMISSION_LOCK, 'w') as admission:
    fcntl.flock(admission, fcntl.LOCK_EX)
    with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = jobs.QUEUE.read_bytes()
        text = before.decode()
        node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
        staged = ROOT / 'runs/setup/encp-smoke-queue-candidate-20261025.yaml'
        staged.write_text(text[:node.end_mark.index].rstrip() + '\n' + yaml.safe_dump(rows, sort_keys=False) + '\n'
                          + text[node.end_mark.index:])
        assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + rows
        tmp = jobs.QUEUE.with_suffix('.encsmoke.tmp')
        tmp.write_bytes(staged.read_bytes())
        os.replace(tmp, jobs.QUEUE)
        proof.write_text(json.dumps({'added': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
                                     'queue_sha256_before': hashlib.sha256(before).hexdigest(),
                                     'queue_sha256_after': hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest(),
                                     'rows': rows}, indent=1) + '\n')
        print('ADDED', len(rows))
