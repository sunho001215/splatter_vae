"""Memory watchdog helpers for the local budget (user decision 2026-10-21, option (b); runs/setup/tools/memory_budget.py:
memory.current minus verified clean disk-backed file cache).

    memory_watch.py line            one JSON line: budget usage and its components, raw charge, Docker working set,
                                    memory and I/O pressure, host MemAvailable, and the latest step time of every
                                    running local pretraining run (training speed)
    memory_watch.py fits BYTES      exit 0 when BYTES more (e.g. a test suite: memory_budget.SUITE_RESERVE +
                                    SUITE_HEADROOM) fit: budget usage + the running jobs' remaining growth + BYTES +
                                    transient overhead <= ceiling, and host MemAvailable - BYTES >= the host reserve
    memory_watch.py alert           print MEMORY_ALERT and exit 0 when budget usage >= 147 GB, memory pressure
                                    (PSI full avg60) >= 5 %, I/O pressure (PSI full avg60) >= 20 %, or host
                                    MemAvailable < 60 GiB (the scheduler's host reserve); else exit 1
No mode evicts cache.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_budget as mb  # noqa: E402

ROOT = Path('/home/ws/ws/splatter4d')
ALERT_BUDGET = 147_000_000_000
ALERT_PSI_FULL_AVG60 = 5.0
ALERT_IO_FULL_AVG60 = 20.0
MEASURED = {'scripts/train.py': 10.77e9, 'scripts/train_rl.py': 3.5e9, 'scripts/eval_rl.py': 4.5e9,
            'scripts/evaluate.py': 10.0e9}


def running_local():
    sys.path.insert(0, str(ROOT))
    from scripts import jobs  # noqa: PLC0415
    state = jobs.read_registry()
    queue = {j['id']: j for j in jobs.load_queue()['jobs']}
    return {i: (s, queue.get(i, {})) for i, s in state.items()
            if s.get('host', 'local') == 'local' and s.get('status') in jobs.ACTIVE}


def remaining_growth(active):
    pss = mb.session_nonfile_pss()
    total = 0
    for _, (s, job) in active.items():
        footprint = MEASURED.get(job.get('script'), float(s.get('ram_gb', job.get('ram_gb', 0))) * 2**30)
        total += max(0, int(footprint) - pss.get(int(s.get('pid', -1)), 0))
    return total


def step_times(active):
    out = {}
    for i, (_, job) in active.items():
        if job.get('script') != 'scripts/train.py':
            continue
        path = ROOT / 'runs/pretrain' / i / 'metrics.jsonl'
        try:
            with path.open('rb') as f:
                f.seek(max(0, path.stat().st_size - 200_000))
                rows = [json.loads(line) for line in f.read().decode(errors='ignore').splitlines()[1:] if line.strip()]
            last = [r for r in rows if 'train/step_time' in r][-1]
            out[i] = [last['step'], round(last['train/step_time'], 3)]
        except (OSError, IndexError, ValueError):
            continue
    return out


def main():
    s = mb.snapshot()
    mode = sys.argv[1] if len(sys.argv) > 1 else 'line'
    full = s['pressure'].get('full', {}).get('avg60', 0.0)
    io_full = s['io_pressure'].get('full', {}).get('avg60', 0.0)
    if mode == 'line':
        print(json.dumps({'time': time.strftime('%F %T'), 'budget_usage': s['budget_usage'], 'raw': s['current'],
                          'clean_disk_file_excluded': s['clean_disk_file'], 'working_set': s['working_set'],
                          'anon': s['anon'], 'shmem': s['shmem'], 'kernel': s['kernel'], 'file_dirty': s['file_dirty'],
                          'file_writeback': s['file_writeback'], 'mapped_disk_file': s['mapped_disk_file'],
                          'unevictable': s['unevictable'], 'inactive_file': s['inactive_file'],
                          'active_file': s['active_file'],
                          'psi_some_avg60': s['pressure'].get('some', {}).get('avg60'), 'psi_full_avg60': full,
                          'io_psi_some_avg60': s['io_pressure'].get('some', {}).get('avg60'), 'io_psi_full_avg60': io_full,
                          'host_mem_available': s['host_mem_available'], 'step_time': step_times(running_local())}))
        return 0
    if mode == 'fits':
        extra = int(float(sys.argv[2]))
        growth = remaining_growth(running_local())
        total = s['budget_usage'] + growth + extra + mb.TRANSIENT_OVERHEAD
        ok = total <= mb.CEILING and s['host_mem_available'] - extra >= mb.HOST_RESERVE
        print(json.dumps({'fits': ok, 'budget_usage': s['budget_usage'], 'remaining_growth': growth, 'extra': extra,
                          'transient': mb.TRANSIENT_OVERHEAD, 'projected': total, 'ceiling': mb.CEILING,
                          'host_mem_available': s['host_mem_available']}))
        return 0 if ok else 1
    if mode == 'alert':
        reasons = []
        if s['budget_usage'] >= ALERT_BUDGET:
            reasons.append(f"budget_usage={s['budget_usage']}")
        if full >= ALERT_PSI_FULL_AVG60:
            reasons.append(f'psi_full_avg60={full}')
        if io_full >= ALERT_IO_FULL_AVG60:
            reasons.append(f'io_psi_full_avg60={io_full}')
        if s['host_mem_available'] < mb.HOST_RESERVE:
            reasons.append(f"host_mem_available={s['host_mem_available']}")
        if reasons:
            print(f"MEMORY_ALERT {time.strftime('%T')} {' '.join(reasons)} raw={s['current']}")
            return 0
        return 1
    raise SystemExit(f'unknown mode {mode}')


if __name__ == '__main__':
    sys.exit(main())
