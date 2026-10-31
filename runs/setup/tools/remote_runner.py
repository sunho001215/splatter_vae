"""Remote-resident campaign runner for the local outage (user authorization 2026-10-31; standard library only, runs on
the remote host under ~/kaist/sunho/remote_runner with the host's python3, detached from any SSH session).

Executes the pre-generated launch commands of plan.json (gen_remote_autonomy_plan_20261031.py) in plan order while the
local campaign driver is offline. Each command is the exact command the local driver would run over SSH for that job
attempt (its own checks refuse foreign GPU processes, low RAM, a low disk floor and exceeded per-GPU job/memory caps);
one command per authorized GPU, chosen at launch time (no text substitution). Rules per pass (every POLL seconds):
  - a plan item is launched at most once per attempt; an existing container of that name counts as launched;
  - dependencies: every dep's container (any attempt) exited 0 and every listed result file exists under runs/<dep>;
  - GPU: least-loaded authorized GPU with load + item load <= GPU_LOAD_LIMIT (load units of the local driver:
    heavy trainer/pretraining 1.0, RL trainer 0.5, evaluator 0.3, export 0.1) and nvidia-smi memory.used + mem_gb within
    MEM_CAP_GB; host MemAvailable must keep RAM_RESERVE_GB; at most MAX_LAUNCHES_PER_PASS launches per pass;
  - a failed launch command (non-zero exit, nothing started) is retried on a later pass, at most 6 times per attempt;
  - a launched attempt that exits non-zero is followed by the item's next pre-generated attempt (if any) after its
    dependencies still hold; no other retries.
Everything is appended to events.jsonl; heartbeat.json is rewritten every pass. Create the file STOP to make the runner
exit at the next pass (the local reconciliation does this before the local scheduler resumes).
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = Path('/home/compu/kaist/sunho')
POLL = 60
GPU_LOAD_LIMIT = float(os.environ.get('S4D_RUNNER_GPU_LOAD_LIMIT', '5.0'))
MEM_CAP_GB = 90.0
RAM_RESERVE_GB = 120.0
MAX_LAUNCHES_PER_PASS = 4
MAX_COMMAND_FAILURES = 6
MAX_TRIES_PER_PASS = 6  # launch commands executed per pass (successful or not)
DRY = os.environ.get('S4D_RUNNER_DRY') == '1'  # one pass, log decisions only


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def event(**fields):
    with (BASE / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps({'time': now(), **fields}) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def run(cmd, timeout=120):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def containers():
    """name -> (state, exit_code, gpu label) for every s4d-* container (running or exited)."""
    out = run(['docker', 'ps', '-a', '--format', '{{.Names}}\t{{.State}}\t{{.Label "s4d.gpu"}}'], 60)
    if out.returncode:
        raise RuntimeError('docker ps failed')
    result = {}
    exited = []
    for line in out.stdout.splitlines():
        parts = line.split('\t')
        if len(parts) != 3 or not parts[0].startswith('s4d-'):
            continue
        result[parts[0]] = [parts[1], None, parts[2]]
        if parts[1] == 'exited':
            exited.append(parts[0])
    for i in range(0, len(exited), 50):
        chunk = exited[i:i + 50]
        out = run(['docker', 'inspect', '--format', '{{.Name}}\t{{.State.ExitCode}}', *chunk], 60)
        for line in out.stdout.splitlines():
            name, code = line.split('\t')
            result[name.lstrip('/')][1] = int(code)
    return result


def job_of(container):
    """job id of a container name s4d-<id>-a<attempt>."""
    base = container[4:]
    return base.rsplit('-a', 1)[0]


def gpu_memory_used():
    out = run(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], 60)
    used = {}
    for line in out.stdout.splitlines():
        uuid, mib = [x.strip() for x in line.split(',')]
        used[uuid] = float(mib) / 1024
    return used


def mem_available_gb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20
    return 0.0


def main():
    lock = open(BASE / ('runner-dry.lock' if DRY else 'runner.lock'), 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('another runner holds the lock', file=sys.stderr)
        return 1
    plan_mtime = None
    failures = {}
    event(event='runner_started', pid=os.getpid(), gpu_load_limit=GPU_LOAD_LIMIT)
    while True:
      try:
        mtime = (BASE / 'plan.json').stat().st_mtime  # the plan may be replaced (atomically) while running
        if mtime != plan_mtime:
            plan = json.loads((BASE / 'plan.json').read_text())
            items, loads = plan['items'], plan['loads']
            placeholder, gpus = plan['placeholder_gpu'], plan['gpus']
            plan_ids = {i['id'] for i in items}
            plan_mtime = mtime
            event(event='plan_loaded', items=len(items), generated=plan.get('generated'))
        if (BASE / 'STOP').exists():
            event(event='runner_stopped', reason='STOP file')
            return 0
        try:
            cons = containers()
            mem_used = gpu_memory_used()
            avail = mem_available_gb()
        except Exception as exc:  # noqa: BLE001
            event(event='probe_failed', error=repr(exc))
            time.sleep(POLL)
            continue
        limit_file = BASE / 'gpu_load_limit.txt'  # optional override, re-read every pass
        try:
            limit = float(limit_file.read_text().strip()) if limit_file.is_file() else GPU_LOAD_LIMIT
        except ValueError:
            limit = GPU_LOAD_LIMIT
        load = {g: 0.0 for g in gpus}
        for name, (st, _, gpu) in cons.items():
            if st in ('running', 'created', 'restarting') and gpu in load:
                load[gpu] += float(loads.get(job_of(name), 1.0))
        by_job = {}
        for name, (st, code, _) in cons.items():
            by_job.setdefault(job_of(name), []).append((name, st, code))
        launched_now = 0
        tries = 0
        reserved_gpu = None  # head-of-line reservation for the first ready item that does not fit (as the local driver)
        pending = 0
        for item in items:
            attempts = item['attempts']
            names = {a['container'] for a in attempts}
            present = [c for c in cons if c in names]
            # which attempt to use: the first whose container does not exist, if every earlier one exited non-zero
            target = None
            for a in attempts:
                if a['container'] not in cons:
                    target = a
                    break
                st, code, _ = cons[a['container']]
                if st != 'exited' or code == 0:
                    target = None
                    break
            if target is None:
                continue
            if present and not all(cons[c][0] == 'exited' and cons[c][1] != 0 for c in present):
                continue
            pending += 1
            if launched_now >= MAX_LAUNCHES_PER_PASS or tries >= MAX_TRIES_PER_PASS:
                continue
            ok = True
            if item.get('companion_of') in plan_ids and not by_job.get(item['companion_of']):
                continue  # a companion evaluator starts only once its trainer has a container
            if any(not (ROOT / rel).exists() for rel in item.get('require_files', [])):
                continue
            for dep in item['deps']:
                runs = by_job.get(dep, [])
                if not any(st == 'exited' and code == 0 for _, st, code in runs):
                    ok = False
                    break
                for rel in item['dep_results'].get(dep, []):
                    if not (ROOT / 'runs' / dep / rel).exists():
                        ok = False
                        break
            if not ok:
                continue
            key = target['container']
            if failures.get(key, 0) >= MAX_COMMAND_FAILURES:
                continue
            need = float(item['load'])
            candidates = sorted((g for g in gpus if g != reserved_gpu), key=lambda g: load[g])
            gpu = next((g for g in candidates if load[g] + need <= limit + 1e-9
                        and mem_used.get(g, 0.0) + float(item['mem_gb']) <= MEM_CAP_GB), None)
            if gpu is None or avail - float(item['ram_gb']) < RAM_RESERVE_GB:
                if reserved_gpu is None:  # later items may not use the GPU this item is waiting for
                    reserved_gpu = min(gpus, key=lambda g: load[g])
                    event(event='reserved', id=item['id'], gpu=reserved_gpu, load=round(load[reserved_gpu], 2), need=need) \
                        if (BASE / 'log_reservations').exists() else None
                continue
            if DRY:
                event(event='dry_would_launch', id=item['id'], container=key, gpu=gpu, load_before=round(load[gpu], 2))
                load[gpu] += need
                launched_now += 1
                continue
            command = (BASE / target['commands'][gpu]).read_text()  # per-GPU command, no substitution
            script = BASE / 'launch' / f"{key}.sh"
            script.parent.mkdir(exist_ok=True)
            script.write_text(command)
            tries += 1
            result = run(['bash', str(script)], 900)
            if result.returncode:
                failures[key] = failures.get(key, 0) + 1
                event(event='launch_failed', id=item['id'], container=key, gpu=gpu, code=result.returncode,
                      stderr=result.stderr[-2000:], failures=failures[key])
                continue
            try:
                payload = json.loads(result.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                payload = {'stdout': result.stdout[-500:]}
            event(event='launched', id=item['id'], container=key, attempt=target['attempt'], gpu=gpu,
                  load_before=round(load[gpu], 2), item_load=need,
                  **{k: v for k, v in payload.items() if k not in ('container', 'id', 'event', 'gpu', 'attempt')})
            load[gpu] += need
            mem_used[gpu] = mem_used.get(gpu, 0.0) + float(item['mem_gb'])
            avail -= float(item['ram_gb'])
            launched_now += 1
        (BASE / 'heartbeat.json').write_text(json.dumps({'time': now(), 'pid': os.getpid(), 'gpu_load': load,
                                                         'pending_items': pending, 'launched_this_pass': launched_now,
                                                         'gpu_load_limit': limit, 'reserved_gpu': reserved_gpu}) + '\n')
        if DRY:
            event(event='dry_pass_done')
            return 0
      except Exception as exc:  # noqa: BLE001  never die: log, keep the previous plan, retry next pass
        event(event='pass_error', error=repr(exc)[-1000:])
      time.sleep(POLL)


if __name__ == '__main__':
    sys.exit(main())
