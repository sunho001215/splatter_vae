"""Local memory-budget accounting (user instruction 2026-10-21): the 150 GB local budget counts the Docker-style cgroup
v2 working set, not the raw charge.

    budget_usage = max(0, memory.current - inactive_file)

Reclaimable inactive file cache is excluded; everything else stays counted: anonymous memory, shared memory (shmem
pages sit on the anonymous LRU lists, not in inactive_file), active file cache, kernel memory, and pinned memory.
The ceiling stays 150,000,000,000 bytes. Admission (baseline/stage2 local drivers) requires

    working set + remaining growth of every running job + candidate peak footprint + active-cache allowance
      + transient overhead <= ceiling,

where a running job's remaining growth is its measured peak footprint minus its current non-file PSS
(Pss_Anon + Pss_Shmem from smaps_rollup; Pss_File is not used, so mapped file cache cannot re-enter the budget),
the active-cache allowance is ACTIVE_CACHE_FRACTION of each dataset a data-reading job uses (the share of such
caches observed on the active list), and TRANSIENT_OVERHEAD covers loader-worker spikes, checkpoint writes and
evaluation passes. Raw usage, host MemAvailable (with the scheduler's existing 60 GiB host reserve) and memory
pressure (PSI) are still monitored and reported.
"""
from __future__ import annotations

from pathlib import Path

CGROUP = Path('/sys/fs/cgroup')
CEILING = 150_000_000_000
TRANSIENT_OVERHEAD = 10_000_000_000  # the previous 140 GB admission ceiling kept the same 10 GB margin
ACTIVE_CACHE_FRACTION = 0.05  # 2026-10-21: 2.3 GB active_file with ~60 GB of three training datasets resident


def cgroup_stat() -> dict[str, int]:
    out = {}
    for line in (CGROUP / 'memory.stat').read_text().splitlines():
        key, value = line.split()
        out[key] = int(value)
    return out


def snapshot() -> dict:
    """Raw charge, the working set and its components, pressure and host MemAvailable (bytes)."""
    stat = cgroup_stat()
    current = int((CGROUP / 'memory.current').read_text())
    ws = max(0, current - stat.get('inactive_file', 0))
    pressure = {}
    try:
        for line in (CGROUP / 'memory.pressure').read_text().splitlines():
            kind, *fields = line.split()
            pressure[kind] = {k: float(v) for k, v in (f.split('=') for f in fields)}
    except OSError:
        pass
    host_available = 0
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            host_available = int(line.split()[1]) * 1024
    return {'current': current, 'working_set': ws, 'inactive_file': stat.get('inactive_file', 0),
            'active_file': stat.get('active_file', 0), 'anon': stat.get('anon', 0), 'shmem': stat.get('shmem', 0),
            'kernel': stat.get('kernel', 0), 'file': stat.get('file', 0), 'pressure': pressure,
            'host_mem_available': host_available, 'ceiling': CEILING}


def working_set() -> int:
    return snapshot()['working_set']


def nonfile_pss(pid_dir: Path) -> int:
    """Pss_Anon + Pss_Shmem (bytes) of one process; raises OSError/StopIteration if unreadable."""
    total = 0
    found = False
    for line in (pid_dir / 'smaps_rollup').read_text().splitlines():
        if line.startswith(('Pss_Anon:', 'Pss_Shmem:')):
            total += int(line.split()[1]) * 1024
            found = True
    if not found:
        raise StopIteration
    return total


def session_nonfile_pss() -> dict[int, int]:
    """Non-file PSS (bytes) per session id over all live processes (scheduler launches are session leaders)."""
    out = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            session = int((entry / 'stat').read_text().rsplit(')', 1)[1].split()[3])
            pss = nonfile_pss(entry)
        except (OSError, IndexError, StopIteration, ValueError):
            continue
        out[session] = out.get(session, 0) + pss
    return out


if __name__ == '__main__':
    import json
    s = snapshot()
    print(json.dumps({**{k: (round(v / 1e9, 2) if isinstance(v, int) else v) for k, v in s.items()}}, indent=1))
