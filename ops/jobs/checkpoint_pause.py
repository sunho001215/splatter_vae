"""Import-safe checkpoint-pause primitives; callers own registry paths and process callbacks."""

import fcntl
import itertools
import json
import os
import pickletools
import time
import zipfile
from pathlib import Path


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'group': int(fields[2]), 'session': int(fields[3]), 'start_ticks': int(fields[19]),
            'state': fields[0]}


def record(event, registry):
    with registry.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with registry.open('a') as stream:
            stream.write(json.dumps({**event, 'time': time.time()}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def checkpoint_step(path):
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        ops = list(itertools.islice(pickletools.genops(archive.read(name)), 7))
    if ops[-3][1] != 'step' or ops[-1][0].name not in {'BININT', 'BININT1', 'BININT2'}:
        raise RuntimeError('unexpected checkpoint step metadata')
    return ops[-1][1]


def session_alive(pid, identity):
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                status = identity(int(entry.name))
            except (FileNotFoundError, ProcessLookupError, IndexError):
                continue
            if status['session'] == pid and status['state'] not in {'Z', 'X'}:
                return True
    return False

