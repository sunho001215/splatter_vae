"""Temporary local MIG expansion (user 2026-11-15). Dry-run unless --apply, --apply-ram400 or --rollback.

Only the RAM policy and effective local limits mapping change (legacy limits: or hosts.local.limits).
Lock order matches placement: local admission, then registry/queue lock. No job row, process or GPU topology is modified.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from ops.resources import memory_budget as mb  # noqa: E402
from ops.resources.gpu_guard import APPROVED_HOST_GPUS  # noqa: E402

NEW_MIGS = APPROVED_HOST_GPUS['local'][8:]
LIMIT = {'max_jobs': 3, 'max_mem_gb': 22}


def live_jobs(root: Path) -> list[dict]:
    from ops.jobs.scheduler import RESERVING, read_registry

    found = [{'id': i, 'gpu': s['gpu'], 'status': s['status'], 'pid': s.get('pid')}
             for i, s in read_registry(root / 'experiments/registry.jsonl').items()
             if s.get('host', 'local') == 'local' and s.get('gpu') in NEW_MIGS
             and s.get('status') in RESERVING]
    # Also catch processes outside the registry; read-only, never signal them.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            entries = (proc / 'environ').read_bytes().split(b'\0')
        except OSError:
            continue
        for item in entries:
            if item.startswith(b'CUDA_VISIBLE_DEVICES='):
                for gpu in item.split(b'=', 1)[1].decode(errors='replace').split(','):
                    if gpu.strip() in NEW_MIGS:
                        found.append({'pid': int(proc.name), 'gpu': gpu.strip(), 'status': 'live process'})
    return found


def rewrite(before: bytes, rollback: bool) -> bytes:
    text = before.decode()
    data = yaml.safe_load(text)
    location = ('hosts', 'local', 'limits') if 'limits' in data.get('hosts', {}).get('local', {}) else ('limits',)
    limits = data
    for key in location:
        limits = limits[key]
    expected = copy.deepcopy(data)
    target = expected
    for key in location:
        target = target[key]
    for gpu in NEW_MIGS:
        if gpu in target and target[gpu] != LIMIT:
            raise ValueError(f'conflicting limit for {gpu}: {target[gpu]}')
        if rollback:
            target.pop(gpu, None)
        else:
            target[gpu] = dict(LIMIT)
    if target == limits:
        return before
    def child(node, key):
        return next(v for k, v in node.value if k.value == key)
    node = yaml.compose(text)
    for key in location:
        node = child(node, key)
    lines = text.splitlines(keepends=True)
    indent = node.start_mark.column
    block = yaml.safe_dump(target, sort_keys=False)
    replacement = ''.join(' ' * indent + line if line.strip() else line for line in block.splitlines(keepends=True))
    after = ''.join(lines[:node.start_mark.line]) + replacement + ''.join(lines[node.end_mark.line:])
    if yaml.safe_load(after) != expected:
        raise ValueError('queue rewrite changed fields outside local limits')
    return after.encode()


def atomic(path: Path, data: bytes):
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            os.fchmod(stream.fileno(), path.stat().st_mode & 0o777 if path.exists() else 0o644)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(tmp).unlink(missing_ok=True)


def update(root: Path, *, apply=False, apply_ram400=False, rollback=False, live=live_jobs) -> dict:
    if sum(map(bool, (apply, apply_ram400, rollback))) > 1:
        raise ValueError('choose one mutation mode')
    root = Path(root).resolve()
    queue = root / 'experiments/queue.yaml'
    control = root / 'runs/setup/local-memory-budget.json'
    target = mb.SAFE_CEILING if rollback else mb.MAX_CEILING if apply_ram400 else None
    policy = (json.dumps({'ceiling_bytes': target}, indent=2) + '\n').encode() if target else None
    with ExitStack() as stack:
        for path in (root / 'runs/setup/stage2-driver.lock', root / 'experiments/registry.lock'):
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX)
        if rollback:
            active = live(root)
            if active:
                raise ValueError('rollback refused: GPU 6/7 jobs still live:\n' + json.dumps(active, indent=2))
        before = queue.read_bytes()
        after = rewrite(before, rollback)
        policy_changed = policy is not None and (not control.exists() or control.read_bytes() != policy)
        result = {'mode': 'rollback' if rollback else 'apply-ram400' if apply_ram400 else 'apply' if apply else 'dry-run',
                  'changed': after != before or policy_changed, 'gpus': list(NEW_MIGS),
                  'ceiling_bytes': target if target else mb.read_ceiling(control)}
        if apply or apply_ram400 or rollback:
            if after != before:
                stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
                backup = queue.with_name(queue.name + f'.before-local-mig16-{stamp}')
                with backup.open('xb') as stream:
                    stream.write(before)
                    stream.flush()
                    os.fsync(stream.fileno())
                result['backup'] = str(backup)
            # Lower RAM before removing devices; raise it only after adding devices.
            # Both locks cover both files. If interrupted, rerun the same command.
            if rollback and policy_changed:
                atomic(control, policy)
            if after != before:
                atomic(queue, after)
            if apply_ram400 and policy_changed:
                atomic(control, policy)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--apply-ram400', action='store_true', help='400 GB RAM and 16 MIGs on GPUs 4-7')
    mode.add_argument('--rollback', action='store_true', help='150 GB RAM and 8 MIGs on GPUs 4/5')
    parser.add_argument('--root', type=Path, default=REPO)
    args = parser.parse_args()
    try:
        result = update(args.root, apply=args.apply, apply_ram400=args.apply_ram400, rollback=args.rollback)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
