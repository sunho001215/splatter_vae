"""Evict page cache of leftover live-sync partial downloads (runs/**/*.s4d-partial-*) that have not been modified for
10 minutes. Files are neither modified nor deleted; only their clean cached pages are released (POSIX_FADV_DONTNEED),
so the local full cgroup charge (150 GB rule) does not grow with files nothing reads again."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
before = int(Path('/sys/fs/cgroup/memory.current').read_text())
count = size = 0
for path in (ROOT / 'runs').rglob('*.s4d-partial-*'):
    try:
        stat = path.stat()
        if not path.is_file() or path.is_symlink() or time.time() - stat.st_mtime < 600:
            continue
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError as exc:
        print(f'skip {path}: {exc}', file=sys.stderr)
        continue
    count += 1
    size += stat.st_size
after = int(Path('/sys/fs/cgroup/memory.current').read_text())
print(json.dumps({'time': time.strftime('%F %T'), 'files': count, 'bytes': size, 'charge_before': before,
                  'charge_after': after}))
