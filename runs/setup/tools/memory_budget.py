"""Local memory-budget accounting (user decision 2026-10-21, option (b)): the 150 GB local budget counts every byte of the
container's cgroup except verified reclaimable, clean, disk-backed file cache, whether on the active or the inactive
LRU generations.

    budget_usage = memory.current - clean_disk_file
    clean_disk_file = max(0, file - shmem - file_dirty - file_writeback - mapped_disk_file - unevictable)

Counted: anonymous memory (incl. pinned host buffers), shared memory / tmpfs (shmem), kernel memory (slab, stacks,
page tables, ...), dirty and writeback file pages, file pages mapped by a process (mapped_disk_file = the sum of
Pss_File over all processes, exact for the pages' total and excluding shmem; counted conservatively although clean),
and unevictable (mlocked) pages. Excluded: only the remainder, clean unmapped page cache of disk files, which the kernel
can drop at any time without writeback. (Earlier definitions, kept in memory_budget.before-*.py: the Docker working
set max(0, memory.current - inactive_file), which counted clean active dataset cache; that share swung 2.6-62 GB with
the multi-generational LRU's aging.)

The ceiling stays 150,000,000,000 bytes. Admission (baseline/stage2 local drivers) requires

    budget_usage + remaining growth of every running job + candidate peak footprint + transient overhead <= ceiling,

where a running job's remaining growth is its measured peak footprint minus its current non-file PSS
(Pss_Anon + Pss_Shmem). No dataset-cache allowance is needed: clean dataset cache is excluded. A full native test
suite reserves SUITE_RESERVE (measured ~40 GB) plus SUITE_HEADROOM before it is launched (memory_watch.py fits).
Raw memory.current, the Docker working set, host MemAvailable (60 GiB host reserve), memory and I/O pressure (PSI)
and training speed are monitored (memory_watch.py line / alert). No cache is force-evicted for the budget.
"""
from __future__ import annotations

from pathlib import Path

CGROUP = Path('/sys/fs/cgroup')
CEILING = 150_000_000_000
TRANSIENT_OVERHEAD = 10_000_000_000  # the previous 140 GB admission ceiling kept the same 10 GB margin
SUITE_RESERVE = 40_000_000_000  # full native suite (scripts/run_tests.py): +40 GB measured 2026-10-21
SUITE_HEADROOM = 10_000_000_000
HOST_RESERVE = 60 * 2**30  # the scheduler's host MemAvailable reserve


def cgroup_stat() -> dict[str, int]:
    out = {}
    for line in (CGROUP / 'memory.stat').read_text().splitlines():
        key, value = line.split()
        out[key] = int(value)
    return out


def _pressure(path: Path) -> dict:
    out = {}
    try:
        for line in path.read_text().splitlines():
            kind, *fields = line.split()
            out[kind] = {k: float(v) for k, v in (f.split('=') for f in fields)}
    except OSError:
        pass
    return out


MAPPED_CACHE = Path('/home/ws/ws/splatter4d/runs/setup/memory-mapped-disk-file.json')
MAPPED_MAX_AGE_S = 300  # the full scan walks every process's page tables (~10 s of system time with eight trainers)


def mapped_disk_file_cached(stat: dict[str, int]) -> int:
    """mapped_disk_file() refreshed at most every MAPPED_MAX_AGE_S; between scans the larger of the last scan and the
    memory.stat lower bound file_mapped - shmem (mapped shared memory is counted in shmem already)."""
    import json
    import time
    floor = max(0, stat.get('file_mapped', 0) - stat.get('shmem', 0))
    try:
        cached = json.loads(MAPPED_CACHE.read_text())
        if time.time() - cached['time'] < MAPPED_MAX_AGE_S:
            return max(int(cached['bytes']), floor)
    except (OSError, ValueError, KeyError):
        pass
    value = mapped_disk_file()
    if value is None:
        return stat.get('file_mapped', 0)  # unreadable: count every mapped file page (incl. mapped shmem)
    try:
        tmp = MAPPED_CACHE.with_suffix('.tmp')
        tmp.write_text(json.dumps({'time': time.time(), 'bytes': value}))
        tmp.replace(MAPPED_CACHE)
    except OSError:
        pass
    return max(value, floor)


def mapped_disk_file() -> int | None:
    """Bytes of disk-backed file pages mapped by any process (sum of Pss_File: shared pages split, total exact)."""
    total, found = 0, False
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for line in (entry / 'smaps_rollup').read_text().splitlines():
                if line.startswith('Pss_File:'):
                    total += int(line.split()[1]) * 1024
                    found = True
        except OSError:
            continue
    return total if found else None


def snapshot() -> dict:
    """Budget usage and its components, the raw charge, the Docker working set, pressure and host MemAvailable."""
    stat = cgroup_stat()
    current = int((CGROUP / 'memory.current').read_text())
    mapped = mapped_disk_file_cached(stat)
    clean = max(0, stat.get('file', 0) - stat.get('shmem', 0) - stat.get('file_dirty', 0)
                - stat.get('file_writeback', 0) - mapped - stat.get('unevictable', 0))
    host_available = 0
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            host_available = int(line.split()[1]) * 1024
    return {'current': current, 'budget_usage': max(0, current - clean), 'clean_disk_file': clean,
            'working_set': max(0, current - stat.get('inactive_file', 0)),
            'anon': stat.get('anon', 0), 'shmem': stat.get('shmem', 0), 'kernel': stat.get('kernel', 0),
            'file': stat.get('file', 0), 'file_dirty': stat.get('file_dirty', 0),
            'file_writeback': stat.get('file_writeback', 0), 'mapped_disk_file': mapped,
            'unevictable': stat.get('unevictable', 0), 'inactive_file': stat.get('inactive_file', 0),
            'active_file': stat.get('active_file', 0), 'pressure': _pressure(CGROUP / 'memory.pressure'),
            'io_pressure': _pressure(CGROUP / 'io.pressure'), 'host_mem_available': host_available,
            'ceiling': CEILING}


def budget_usage() -> int:
    return snapshot()['budget_usage']


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
    print(json.dumps({k: (round(v / 1e9, 2) if isinstance(v, int) else v) for k, v in s.items()}, indent=1))
