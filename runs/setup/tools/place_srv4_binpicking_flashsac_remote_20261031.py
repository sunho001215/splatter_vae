"""Place the six never-started self-render v4 bin-picking FlashSAC rows (ds-flashsac-srv4-csw4-bin-picking-s200X[-eval])
on the remote (host fields only, pool_placement.placed_row; recorded in runs/setup/pool-placement.json) for the local
outage of 2026-10-31 23:00 KST: the remote runner launches them once the locally produced export is uploaded."""
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

IDS = [f'ds-flashsac-srv4-csw4-bin-picking-s{s}{e}' for s in (2000, 2001, 2002) for e in ('', '-eval')]
with open(pool.ADMISSION_LOCK, 'w') as admission:
    fcntl.flock(admission, fcntl.LOCK_EX)
    with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = jobs.load_queue()
        by_id = {j['id']: j for j in queue['jobs']}
        state = jobs.read_registry()
        stamp = datetime.now().astimezone().isoformat(timespec='seconds')
        effective = {}
        for rid in IDS:
            assert state.get(rid, {}).get('status', jobs.PENDING) == jobs.PENDING and not state.get(rid, {}).get('attempts'), rid
            if by_id[rid].get('host', 'local') != 'remote':
                effective[rid] = pool.placed_row(by_id[rid], 'remote', stamp)
                assert jobs.job_fingerprint(effective[rid]) == jobs.job_fingerprint(by_id[rid]), rid
        if not effective:
            print('nothing to place')
            raise SystemExit(0)
        before = jobs.QUEUE.read_bytes()
        text = before.decode()
        node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
        new_jobs = [effective.get(j['id'], j) for j in queue['jobs']]
        staged = ROOT / 'runs/setup/srv4-binpicking-placement-queue-candidate-20261031.yaml'
        staged.write_text(text[:node.start_mark.index] + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False)
                          + text[node.end_mark.index:])
        assert jobs.load_queue(staged)['jobs'] == new_jobs and jobs.load_queue(staged)['hosts'] == queue['hosts']
        tmp = jobs.QUEUE.with_suffix('.srv4bp.tmp')
        tmp.write_bytes(staged.read_bytes())
        os.replace(tmp, jobs.QUEUE)
        manifest = json.loads(pool.MANIFEST.read_text())
        manifest['moves'].append({'time': stamp, 'to': 'remote', 'reason': 'local shutdown 2026-10-31: srv4 bin-picking '
                                  'FlashSAC rows run on the remote runner during the outage (export produced locally, '
                                  'uploaded with sha256 check)', 'ids': sorted(effective), 'rewritten_exports': [],
                                  'original': {i: by_id[i] for i in effective},
                                  'queue_sha256_before': hashlib.sha256(before).hexdigest(),
                                  'queue_sha256_after': hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest()})
        manifest['effective_jobs'].update(effective)
        tmp = pool.MANIFEST.with_suffix('.tmp')
        tmp.write_text(json.dumps(manifest, indent=1) + '\n')
        os.replace(tmp, pool.MANIFEST)
        print(json.dumps({'placed': sorted(effective)}))
