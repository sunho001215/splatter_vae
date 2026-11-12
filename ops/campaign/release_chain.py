"""Deploy and verify a remote release (the established typed-verification route; image unchanged):
sync the pushed commit's bundle to ~/kaist/sunho/releases/<commit>/code (identity: clean pushed ancestry, checksum),
run the remote native suite (functional check of the new code in the accepted image) on one authorized GPU, then the four
one-UUID CUDA/EGL isolation checks, and assemble the campaign acceptance manifest. Logs to stdout.
    python -I release_chain.py <commit> [--native-gpu <uuid>] [--image <digest> --tag <suffix>] [--sync-repo <dir>]
        [--skip-sync]

Release turnaround (user 2026-11-12; every gate step above is unchanged):
- priority: from the start until the end (success or failure) runs/setup/release-reservation.json reserves the native
  suite's resources on its GPU and the isolation checks' share on every GPU (ops/jobs/release_reservation.py), renewed
  at every poll; every remote admission path charges it, so new campaign work never takes that capacity. Running jobs
  are never touched. The isolation stage keeps only the isolation share.
- polling: the chain ticks the scheduler only for its own verification jobs (``scope``: no campaign housekeeping
  inside the chain's ticks), and between ticks watches its running containers with one cheap ``docker inspect`` every
  POLL_SECONDS (no snapshot, no registry lock), ticking as soon as one stops (at least every MAX_QUIET_SECONDS for the
  live evidence sync); a job still waiting for admission is retried every PENDING_RETRY_SECONDS.
- timing: the wall time of every stage is logged and written to runs/setup/release-timing/<short>.json.
"""
import fcntl
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from ops.containers.verification import POLICIES  # noqa: E402
from ops.jobs import release_poll, release_reservation  # noqa: E402
from ops.jobs import scheduler as jobs  # noqa: E402
from ops.jobs.remote_backend import COTENANT_GPU_MARGIN_GB, RemoteBackend, cotenant_charges  # noqa: E402
from ops.resources.gpu_guard import APPROVED_HOST_GPUS  # noqa: E402

PY = str(ROOT / '.venv/bin/python')
POLL_SECONDS = 10
HEARTBEAT_SECONDS = 60  # reservation renewal by a background thread (independent of syncs, ticks and lock waits)
# the chain's supported duration: past it the chain stops renewing and fails (the reservation then lapses within
# EXPIRY_SECONDS, well before release_reservation.MAX_LIFETIME_SECONDS); the verification can simply be re-run
DEADLINE_SECONDS = release_reservation.MAX_LIFETIME_SECONDS - release_reservation.EXPIRY_SECONDS - 600
MAX_QUIET_SECONDS = 240
PENDING_RETRY_SECONDS = 15
NATIVE_LOAD = 1.0  # remote driver load units of the native suite on its GPU
ISOLATION_LOAD = 0.0  # the isolation checks take about a minute; memory and RAM are reserved, no load unit


def log(msg):
    print(time.strftime('%T'), msg, flush=True)


def run(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True)
    log(f'{" ".join(cmd[-3:])} -> {out.returncode} {out.stdout.strip()[-300:]} {out.stderr.strip()[-300:]}')
    if out.returncode:
        raise SystemExit(out.returncode)
    return out.stdout.strip().splitlines()[-1]


class Release:
    def __init__(self, argv):
        self.commit = argv[1]
        self.image_args = (['--image', argv[argv.index('--image') + 1], '--tag', argv[argv.index('--tag') + 1]]
                           if '--image' in argv else [])
        self.short = self.commit[:7] + (f'-{self.image_args[3]}' if self.image_args else '')
        self.native_gpu = argv[argv.index('--native-gpu') + 1] if '--native-gpu' in argv else None
        # --sync-repo: a worktree checked out at <commit> (clean), when the main checkout is behind the release commit
        self.sync_repo = Path(argv[argv.index('--sync-repo') + 1]) if '--sync-repo' in argv else ROOT
        self.skip_sync = '--skip-sync' in argv
        self.gpus = list(APPROVED_HOST_GPUS['remote'])
        self.stage = 'native'
        self.native = None
        self.timing = {'commit': self.commit, 'short': self.short, 'stages': {}}
        self.started = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._heartbeat = None

    # ---- reservation (release_reservation) ----
    def isolation_share(self):
        _, _, ram, gpu, _ = POLICIES['gpu_isolation']
        return {'mem_gb': gpu, 'load': ISOLATION_LOAD, 'ram_gb': ram, 'cpu_threads': 8, 'disk_gb': 1}

    def renew(self):
        """Write the reservation for the current stage (ownership-checked; created stays the chain's start). Never
        after clear() (checked under the same mutex) or past the chain's deadline."""
        with self._lock:
            if self._stop.is_set():
                return
            if time.time() - self.started > DEADLINE_SECONDS:
                self._stop.set()
                raise SystemExit(f'release chain deadline ({DEADLINE_SECONDS / 3600:.1f} h) passed; reservation not renewed')
            native = self.native or {'mem_gb': POLICIES['native_suite'][3], 'load': NATIVE_LOAD,
                                     'ram_gb': POLICIES['native_suite'][2], 'cpu_threads': 8, 'disk_gb': 1}
            release_reservation.write(release_reservation.plan(
                self.commit, self.short, self.stage, self.native_gpu or self.gpus[0], self.gpus, native=native,
                isolation=self.isolation_share(), created=self.started))

    def publish(self, stage):
        """Publish a new reservation content under the scheduler's registry lock, so that a tick either ran entirely
        before it or sees it (every remote launch goes through the tick under that lock)."""
        with open(ROOT / 'experiments/registry.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.stage = stage
            self.renew()

    def heartbeat(self):
        while not self._stop.wait(HEARTBEAT_SECONDS):
            try:
                self.renew()
            except BaseException as exc:  # noqa: BLE001  (logged; the reservation lapses by itself if it keeps failing)
                log(f'reservation heartbeat failed ({type(exc).__name__}: {str(exc)[:120]})')

    def clear(self):
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=HEARTBEAT_SECONDS)
        with self._lock:
            if release_reservation.clear(self.commit):
                log('release reservation cleared')

    # ---- timing ----
    def mark(self, name, start):
        self.timing['stages'][name] = round(time.time() - start, 1)
        log(f'stage {name}: {self.timing["stages"][name]:.1f} s')

    def write_timing(self, outcome):
        self.timing.update(outcome=outcome, total_seconds=round(time.time() - self.started, 1),
                           finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        out = ROOT / 'runs/setup/release-timing' / f'{self.short}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.timing, indent=1) + '\n')
        log('timing ' + json.dumps(self.timing['stages']) + f' total {self.timing["total_seconds"]:.1f} s')

    # ---- driving the verification jobs ----
    def drive(self, qpath, ids):
        backend = RemoteBackend(ROOT, ROOT / 'runs')
        scope = set(ids)
        launched_logged = set()
        stage_start = time.time()
        while True:
            for m in jobs.tick(queue_path=Path(qpath), remote_backend=backend, scope=scope):
                log(m)
            state = jobs.read_registry()
            status = {i: state.get(i, {}).get('status') for i in ids}
            for i in ids:
                if i not in launched_logged and status[i] in (jobs.RUNNING, 'sync_pending', jobs.DONE):
                    launched_logged.add(i)  # observed after the tick that launched it (an upper bound)
                    self.timing['stages'].setdefault(f'observed_admit:{i}', round(time.time() - stage_start, 1))
            action, running = release_poll.next_action(state, ids)
            if action == 'done':
                log(f'done {status}')
                return
            if action == 'failed':
                log(f'FAILED {status}')
                raise SystemExit(2)
            log(f'wait {status}')
            if action == 'watch':
                release_poll.wait_for_exit(backend.container_states, list(running.values()),
                                           poll_seconds=POLL_SECONDS, max_quiet=MAX_QUIET_SECONDS, log=log)
            else:  # admission pending, launching, or an exited job awaiting its result sync: tick again soon
                time.sleep(PENDING_RETRY_SECONDS)

    def pick_native_gpu(self):
        """Without --native-gpu: the scheduler's own criterion (least declared memory of executing remote jobs and
        fewest jobs, as choose_gpu), measured free memory as the tie-break; GPUs with processes of other users are
        avoided. A choice only: admission and the launch guard still decide; first GPU on any error."""
        try:
            snapshot = RemoteBackend(ROOT, ROOT / 'runs').snapshot()
            state = jobs.read_registry()
            executing = [c for c in state.values() if c.get('host') == 'remote' and c.get('status') in jobs.RESERVING]
            declared = {g: sum(float(c.get('mem_gb', 0)) for c in executing if c.get('gpu') == g) for g in self.gpus}
            count = {g: sum(1 for c in executing if c.get('gpu') == g) for g in self.gpus}
            cotenants = cotenant_charges(snapshot)['gpus']
            need = float(POLICIES['native_suite'][3])
            short = set()  # GPUs whose co-tenant-measured free memory cannot hold the suite (choose_gpu's margin test)
            for g, charge in cotenants.items():
                if g in declared:
                    declared[g] += charge['mem_gb']
                    count[g] += charge['jobs']
                    if charge['free_gb'] - COTENANT_GPU_MARGIN_GB < need:
                        short.add(g)
            free = {g: (snapshot.gpu_memory.get(g) or (0, 0)) for g in self.gpus}
            free = {g: total - used for g, (total, used) in free.items()}
            owned = set().union(*[set(item.get('pids') or ()) for item in snapshot.containers.values()])
            known = {g: set(c['pids']) for g, c in cotenants.items()}
            busy = {g for g, procs in snapshot.usage.items()  # processes of other users (the tick's foreign test)
                    if g in self.gpus and any(pid not in owned and pid not in known.get(g, ()) for pid, _ in procs)}
            order = sorted(self.gpus, key=lambda g: (g in busy or g in short, declared[g], count[g], -free[g]))
            self.native_gpu = order[0]
            log(f'native GPU {self.native_gpu} (declared {declared[self.native_gpu]:.0f} GB, '
                f'{count[self.native_gpu]} jobs, {free[self.native_gpu] / 1024:.0f} GiB measured free)')
            return
        except Exception as exc:  # noqa: BLE001  (a choice, never a gate)
            log(f'native GPU choice fell back to the first GPU ({type(exc).__name__})')
        self.native_gpu = self.gpus[0]

    def main(self):
        try:
            if self.native_gpu is None:
                self.pick_native_gpu()
            self.publish('native')  # reserve from the start: capacity freed during the code sync is kept for the suite
            self._heartbeat = threading.Thread(target=self.heartbeat, name='reservation-heartbeat', daemon=True)
            self._heartbeat.start()
            start = time.time()
            if not self.skip_sync:  # the release bundle is already deployed (same commit, another image)
                log(run([PY, '-I', str(self.sync_repo / 's4d/remote_access.py'), 'sync-code', self.commit]))
            self.mark('sync', start)
            start = time.time()
            qpath = run([PY, '-I', str(ROOT / 'runs/setup/tools/prepare_remote_verification.py'), self.commit,
                         *self.image_args])
            q = yaml.safe_load(Path(qpath).read_text())
            q['jobs'][0]['gpu'] = self.native_gpu  # the native suite may run on any authorized GPU
            Path(qpath).write_text(yaml.safe_dump(q, sort_keys=False))
            job = q['jobs'][0]
            self.native_gpu = job['gpu']
            self.native = {'mem_gb': float(job['mem_gb']), 'load': NATIVE_LOAD, 'ram_gb': float(job['ram_gb']),
                           'cpu_threads': int(job.get('cpu_threads', 8)), 'disk_gb': float(job.get('disk_gb', 1))}
            self.publish('native')  # the prepared job's exact resources
            self.mark('prepare_native', start)
            start = time.time()
            self.drive(qpath, [f'verify-native-{self.short}'])
            self.mark('native', start)
            suite = ROOT / f'runs/verify-native-{self.short}/evidence/tests.json'
            report = json.loads(suite.read_text())
            self.timing['suite_seconds'] = round(float(report.get('duration_seconds', 0)), 1)
            log(json.dumps({k: v for k, v in report.items() if k in ('counts', 'exit_code', 'full_suite_passed')}))
            start = time.time()
            self.publish('isolation')
            qpath = run([PY, '-I', str(ROOT / 'runs/setup/tools/prepare_remote_verification.py'), self.commit,
                         '--suite', str(suite), *self.image_args])
            self.mark('prepare_isolation', start)
            start = time.time()
            self.drive(qpath, [f'verify-isolation-{i}-{self.short}' for i in range(len(self.gpus))])
            self.mark('isolation', start)
            start = time.time()
            log(run([PY, '-I', str(ROOT / 'runs/setup/tools/assemble_remote_acceptance.py'), self.commit,
                     *self.image_args]))
            self.mark('acceptance', start)
            self.write_timing('accepted')
        except BaseException:
            self.write_timing('failed')
            raise
        finally:
            self.clear()


if __name__ == '__main__':
    Release(sys.argv).main()
