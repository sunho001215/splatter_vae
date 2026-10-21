"""Memory watchdog helpers for the working-set budget (user instruction 2026-10-21; runs/setup/tools/memory_budget.py).

    memory_watch.py line            one JSON line: raw charge, working set, file caches, shmem, PSI, host MemAvailable
    memory_watch.py over BYTES      exit 0 when the working set is at or above BYTES, else 1 (eviction trigger)
    memory_watch.py alert           print MEMORY_ALERT and exit 0 when the working set is >= 147 GB, memory pressure
                                    (PSI full avg60) >= 5 %, or host MemAvailable < 60 GiB (the scheduler's host
                                    reserve); else exit 1
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_budget as mb  # noqa: E402

ALERT_WS = 147_000_000_000
ALERT_PSI_FULL_AVG60 = 5.0
HOST_RESERVE = 60 * 2**30


def main():
    s = mb.snapshot()
    mode = sys.argv[1] if len(sys.argv) > 1 else 'line'
    if mode == 'line':
        print(json.dumps({'time': time.strftime('%F %T'), 'working_set': s['working_set'], 'raw': s['current'],
                          'inactive_file': s['inactive_file'], 'active_file': s['active_file'], 'anon': s['anon'],
                          'shmem': s['shmem'], 'psi_some_avg60': s['pressure'].get('some', {}).get('avg60'),
                          'psi_full_avg60': s['pressure'].get('full', {}).get('avg60'),
                          'host_mem_available': s['host_mem_available']}))
        return 0
    if mode == 'over':
        return 0 if s['working_set'] >= int(float(sys.argv[2])) else 1
    if mode == 'alert':
        full = s['pressure'].get('full', {}).get('avg60', 0.0)
        reasons = []
        if s['working_set'] >= ALERT_WS:
            reasons.append(f"working_set={s['working_set']}")
        if full >= ALERT_PSI_FULL_AVG60:
            reasons.append(f'psi_full_avg60={full}')
        if s['host_mem_available'] < HOST_RESERVE:
            reasons.append(f"host_mem_available={s['host_mem_available']}")
        if reasons:
            print(f"MEMORY_ALERT {time.strftime('%T')} {' '.join(reasons)} raw={s['current']}")
            return 0
        return 1
    raise SystemExit(f'unknown mode {mode}')


if __name__ == '__main__':
    sys.exit(main())
