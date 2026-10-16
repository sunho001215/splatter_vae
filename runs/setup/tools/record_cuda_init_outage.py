"""Record the 2026-10-16 14:46-14:51 local CUDA-initialisation outage for the jobs it crashed (registry
`infrastructure_failure`, reason cuda_init_unavailable), so the outage is not charged as a scientific retry.
Each job must have exactly one failure in the window and its last attempt's console must show the guard's
"torch sees 0 CUDA devices" refusal. Under the registry lock; append-only.

    python -I record_cuda_init_outage.py <job_id> ...
"""
import fcntl
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

START = datetime.fromisoformat('2026-10-16T14:46:00+09:00').timestamp()
END = datetime.fromisoformat('2026-10-16T14:51:00+09:00').timestamp()
MARK = 'torch sees 0 CUDA devices but CUDA_VISIBLE_DEVICES lists 1'
with open(jobs.REGISTRY.with_suffix('.lock'), 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    state = jobs.read_registry()
    for job_id in sys.argv[1:]:
        current = state[job_id]
        window = [f for f in current['failed_attempts'] if START <= f['time'] and f['launch_time'] <= END]
        assert len(window) == 1, (job_id, current['failed_attempts'])
        failure = window[0]
        log = (ROOT / 'runs' / job_id / 'console.log').read_text()
        last = log[log.rfind('=== attempt'):]
        assert MARK in last, job_id
        jobs.record({'event': 'infrastructure_failure', 'id': job_id, 'host': 'local', 'reason': 'cuda_init_unavailable',
                     'attempt': failure['attempt'], 'incident_started': START, 'incident_ended': END,
                     'failure_time': failure['time'],
                     'evidence': {'console': f'runs/{job_id}/console.log', 'message': MARK,
                                  'note': 'fresh processes on the same MIG instances initialised CUDA normally from '
                                          '~14:51; the long-running trainers on those instances were unaffected'}})
        print('RECORDED', job_id, jobs.read_registry()[job_id]['failures'])
