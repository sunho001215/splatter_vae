"""Pause one owned local training session after a fresh latest.pt save.

python -I ops/jobs/pause_at_checkpoint.py JOB_ID --reason TEXT [--execute]

Dry-run is read-only: check ownership and checkpoint location, then exit. Execute
waits for a new checkpoint and a 20-second publication grace period, verifies it,
rechecks the registry attempt and every process identity under registry.lock,
records checkpoint_paused and sends SIGTERM only to that process group. Existing
scheduler retry/resume policy applies; this command does not install a hold.
Remote/container jobs and ambiguous or foreign process groups are refused.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from ops.jobs import checkpoint_pause  # noqa: E402


class PauseError(RuntimeError):
    pass


def members(pgid: int) -> dict:
    found = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                status = checkpoint_pause.identity(int(entry.name))
                if status['group'] == pgid and status['state'] not in {'Z', 'X'}:
                    found[int(entry.name)] = {**status, 'command': (entry / 'cmdline').read_bytes().split(b'\0')}
            except (FileNotFoundError, ProcessLookupError):
                continue
    return found


def owned(root: Path, job_id: str, expected=None):
    from ops.jobs import scheduler

    current = scheduler.read_registry(root / 'experiments/registry.jsonl').get(job_id, {})
    if current.get('status') != 'running' or current.get('host') != 'local':
        raise PauseError('job must be running locally')
    script = current.get('script', '')
    if not script.startswith('scripts/train') or not script.endswith('.py'):
        raise PauseError('job is not a registered training entry point')
    pid, attempt = int(current['pid']), int(current['attempt'])
    processes = members(pid)
    leader = processes.get(pid, {})
    if (pid <= 1 or attempt < 1 or leader.get('session') != pid or leader.get('group') != pid
            or len(processes) < 2):
        raise PauseError('owned session leader and training child not verified')
    for member in processes.values():
        command = member['command']
        if (member.get('session') != pid or command.count(b'--name') != 1
                or command.index(b'--name') + 1 >= len(command)
                or command[command.index(b'--name') + 1] != job_id.encode()):
            raise PauseError('foreign or ambiguous process in job group')
    identity = dict(pid=pid, attempt=attempt, gpu=current.get('gpu'),
                    processes={p: m['start_ticks'] for p, m in processes.items()})
    if expected is not None and identity != expected:
        raise PauseError('job attempt or process identity changed')
    return identity


def checkpoint_path(root: Path, job_id: str) -> Path:
    # Cover established local output layouts without accepting arbitrary paths or new per-job copies.
    candidates = [root / 'runs' / prefix / job_id / 'checkpoints/latest.pt'
                  for prefix in ('', 'pretrain', 'rl', 'snerl', 'realworld_fql')]
    existing = [p for p in candidates if p.is_file()]
    if len(existing) != 1 or not existing[0].resolve().is_relative_to((root / 'runs').resolve()):
        raise PauseError('expected exactly one checkpoint inside this checkout runs/')
    return existing[0]


def token(path: Path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def checkpoint_info(path: Path):
    import torch

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    step = next((checkpoint[k] for k in ('step', 'global_step', 'iteration') if k in checkpoint), None)
    if type(step) is not int or step < 0:
        raise PauseError('checkpoint has no valid step')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return step, digest.hexdigest()


def signal_owned(root, job_id, identity, checkpoint, published, event):
    registry = root / 'experiments/registry.jsonl'
    with registry.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        owned(root, job_id, identity)
        if token(checkpoint) != published:
            raise PauseError('checkpoint changed before signalling')
        with registry.open('a') as stream:
            stream.write(json.dumps({**event, 'time': time.time()}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.killpg(identity['pid'], signal.SIGTERM)


def pause(job_id: str, reason: str, *, execute=False, root=REPO, timeout=6 * 3600):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', job_id) or not reason.strip():
        raise PauseError('a job id and nonempty reason are required')
    root = Path(root).resolve()
    identity = owned(root, job_id)
    checkpoint = checkpoint_path(root, job_id)
    initial = token(checkpoint)
    plan = dict(id=job_id, pid=identity['pid'], attempt=identity['attempt'], checkpoint=str(checkpoint),
                reason=reason, execute=execute)
    if not execute:
        return plan
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Registry replay is ~180 MB: do it at arming and under the final lock, not every five seconds.
        live = members(identity['pid']).get(identity['pid'], {})
        if live.get('start_ticks') != identity['processes'][identity['pid']]:
            raise PauseError('process identity changed while waiting')
        published = token(checkpoint)
        if published == initial:
            time.sleep(5)
            continue
        time.sleep(20)  # the dated pause tools' grace period for policy/encoder publication
        if token(checkpoint) != published:
            continue
        step, digest = checkpoint_info(checkpoint)
        if time.monotonic() >= deadline:
            break
        event = dict(event='checkpoint_paused', id=job_id, host='local', attempt=identity['attempt'],
                     intentional_stop=True, checkpoint=str(checkpoint.relative_to(root)), checkpoint_step=step,
                     checkpoint_sha256=digest, reason=reason, pid=identity['pid'],
                     start_ticks=identity['processes'][identity['pid']], requested_signal='SIGTERM')
        signal_owned(root, job_id, identity, checkpoint, published, event)
        return event
    raise TimeoutError('no stable fresh checkpoint before deadline; no signal sent')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id')
    parser.add_argument('--reason', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(pause(args.job_id, args.reason, execute=args.execute), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
