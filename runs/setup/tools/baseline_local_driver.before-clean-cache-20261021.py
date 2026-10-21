"""One pass of local admission for the new-baseline jobs (registrations listed below), job by job under the local
full-charge cap through scripts/jobs.py. Same accounting as stage2_local_driver.py (full cgroup charge + remaining
growth of every active local job + non-resident dataset cache, 140 GB ceiling) and the same lock file, so the two
drivers never admit at the same time. Never deletes anything.

    python -I baseline_local_driver.py [--dry-run]
"""
import copy
import fcntl
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
import memory_budget as mb  # noqa: E402

# Local budget (user instruction 2026-10-21): the cgroup working set max(0, memory.current - inactive_file) against the
# 150 GB ceiling, minus a transient-overhead margin; see runs/setup/tools/memory_budget.py.
CAP = mb.CEILING - mb.TRANSIENT_OVERHEAD
# The local sanity set moved to the remote (baselines-sanity-remote-registration-20261016.json). Only the evaluator of the
# kept local pair `bxs-dynamo-hammer-rand-s1000` (trainer done locally) is still admitted here (2026-10-17).
# Eight-task runs (2026-10-17): the local rows of baselines-eight-task-registration-20261017.json (DynaMo and 3D-MVP
# condition B DrM; their pretraining exports come from the remote).
# SNeRL eight-task runs (user go-ahead 2026-10-20): the local DrM rows of baselines-snerl-eight-task-registration-20261020.json.
# Variant action_future (k = history = future stride) was CANCELLED and superseded on 2026-10-20 (user instruction;
# runs/setup/actfut-cancellation-20261020.json): its registration is no longer admitted.
# Its replacement future_k2 (fixed history stride 2, future stride k2 in 1..6; runs/setup/afk2-definition-20261020.json).
REGISTRATIONS = ['baselines-sanity-registration-20261016.json', 'baselines-eight-task-registration-20261017.json',
                 'baselines-snerl-eight-task-registration-20261020.json', 'afk2-registration-hammer-pick-place-20261020.json',
                 'afk2-registration-bin-picking-coffee-push-door-open-peg-insert-side-peg-unplug-side-stick-push-20261020.json']
ONLY = {'baselines-sanity-registration-20261016.json': {'bxs-dynamo-hammer-rand-s1000-eval'}}
PLACEMENT = Path('/home/ws/ws/splatter4d/runs/setup/baselines-placement-20261018.json')
DATA = Path('/home/ws/data/metaworld/splatter4d_v1')
DRY = '--dry-run' in sys.argv
SETUP = ROOT / 'runs/setup'
stamp = datetime.now().astimezone()
tag = stamp.strftime('%Y%m%d-%H%M%S')
_lock = open(SETUP / 'stage2-driver.lock', 'w')  # shared with the Stage 2 local driver
try:
    fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print('BASELINE_DRIVER the Stage 2 driver (or another pass) holds the admission lock; skipped')
    raise SystemExit(0)

queue = jobs.load_queue()
by_queue = {j['id']: j for j in queue['jobs']}
registered = []
for name in REGISTRATIONS:
    if (SETUP / name).is_file():
        record = json.loads((SETUP / name).read_text())
        # host-neutral execution overlay (scientific fields unchanged): baselines-placement-20261018.json
        overlay = json.loads(PLACEMENT.read_text())['effective_jobs'] if PLACEMENT.is_file() else {}
        record_jobs = [overlay.get(j['id'], j) for j in record['jobs']]
        assert [by_queue[j['id']] for j in record_jobs] == record_jobs, f'{name}: jobs changed in the queue'
        registered += [j for j in record_jobs if name not in ONLY or j['id'] in ONLY[name]]
by_id = {j['id']: j for j in registered}
if not registered:
    print('BASELINES nothing registered for local admission')
    raise SystemExit(0)
state = jobs.read_registry()
messages = []


def status(job_id):
    current = state.get(job_id, {})
    if current.get('status') == 'crashed' and current['failures'] <= int(by_id[job_id].get('max_restarts', 1)):
        return jobs.PENDING
    return current.get('status', jobs.PENDING)


def task_files(job):
    if job['script'] in ('scripts/train_rl.py', 'scripts/eval_rl.py'):
        return []  # DrM simulates Meta-World live; it reads no dataset file
    trainer = by_id.get(job['id'].removesuffix('-eval'), job)
    args = trainer['args']
    task_cfg = [a for a in args if a.startswith('configs/metaworld/tasks/')]  # our pretraining (scripts/train.py)
    if '--task' in args:
        task = args[args.index('--task') + 1]
    elif task_cfg:
        task = Path(task_cfg[0]).stem
    else:
        task = 'hammer' if 'hammer' in job['id'] else job['id'].split('-')[2]
    return [DATA / f'{task}.hdf5', DATA / f'heldout_sets/{task}.hdf5']


def session_pss():
    # non-file PSS (Pss_Anon + Pss_Shmem): mapped file cache must not re-enter the working-set budget (2026-10-21)
    out = mb.session_nonfile_pss()
    return out


def footprint(job):
    # DrM rows use the measured campaign footprints (scheduling rule: measured, not declared; observed baseline DrM
    # trainers 2.7-3.4 GB and evaluators ~1.9 GB RSS on 2026-10-17); other new-baseline jobs: declared ram_gb.
    if job['script'] in ('scripts/train_rl.py', 'scripts/eval_rl.py'):
        # + the trainer's preallocated replay file, which DrM sampling keeps resident in the page cache (measured
        # 2026-10-17: DynaMo 1.1 GB, 3D-MVP 2.5 GB; evicting it refills within minutes)
        return int(MEASURED[job['script']] + (2.6e9 if job['script'] == 'scripts/train_rl.py' else 0))
    if job['script'] == 'scripts/train.py':  # our pretraining trainer: measured 10.77 GB max sampled PSS (rule 2026-10-11)
        return int(MEASURED[job['script']])
    return int(float(job['ram_gb']) * 2**30)


# Baselines rank below the registered Stage 2 work (user 2026-10-13): yield while any local Stage 2 job is waiting
# (pending with its dependencies done, or crashed but restartable).
stage2_waiting = []
for job in queue['jobs']:
    if job.get('stage') != 2 or job.get('host', 'local') != 'local':
        continue
    current = state.get(job['id'], {})
    restartable = current.get('status') == 'crashed' and current['failures'] <= int(job.get('max_restarts', 1))
    if restartable or (current.get('status', jobs.PENDING) == jobs.PENDING
                       and all(state.get(d, {}).get('status') == jobs.DONE for d in job.get('deps', []))):
        stage2_waiting.append(job['id'])
deps = {d for j in registered for d in j.get('deps', [])}
view = {i: ({**s, 'status': status(i)} if i in by_id else s) for i, s in state.items() if i in by_id or i in deps}
candidates = [j for j in jobs.eligible(registered, view) if j.get('host', 'local') == 'local']
if stage2_waiting:
    messages.append(f'YIELD to {len(stage2_waiting)} waiting Stage 2 local jobs, e.g. {stage2_waiting[:3]}')
    candidates = []
# evaluators of already finished trainers first (short, and they release the run's cached snapshots), then by
# priority with trainers before their companions
candidates.sort(key=lambda j: (not (j['script'] == 'scripts/eval_rl.py'
                                    and status(j['id'].removesuffix('-eval')) == jobs.DONE),
                               j['priority'], j['script'] == 'scripts/eval_rl.py', registered.index(j)))
memory = mb.snapshot()
# Dataset cache (revised 2026-10-21): pages of a dataset read twice join the active generations and count in the
# working set until host reclaim ages them (memory_budget.ACTIVE_CACHE_FRACTION, observed peak 0.4). Projection: the
# working set without the counted datasets' pages already in it (at most their resident bytes, beyond a 2.5 GB floor of
# other active file cache) plus ACTIVE_CACHE_FRACTION of their sizes; a candidate adds that share of its new datasets.
files = sorted({f for j in registered for f in task_files(j) if f.is_file()})
allowance = {f: int(mb.ACTIVE_CACHE_FRACTION * f.stat().st_size) for f in files}
nonresident = allowance  # name kept for the admission loop below
counted = {f for j in registered if status(j['id']) in jobs.ACTIVE for f in task_files(j) if f in allowance}
cache_reserve = sum(allowance[f] for f in counted)
counted_resident = mb.resident_bytes(sorted(counted))
dataset_pages_in_ws = min(max(0, memory['active_file'] - mb.OTHER_ACTIVE_FILE), counted_resident)
charge = memory['working_set'] - dataset_pages_in_ws  # working set max(0, memory.current - inactive_file) minus those
active_share = memory['active_file'] / max(1, memory['active_file'] + memory['inactive_file'])
pss = session_pss()
# Measured footprints of the existing campaign scripts (the Stage 2 driver's table); new-baseline jobs: declared ram_gb.
MEASURED = {'scripts/train.py': 10.77e9, 'scripts/train_rl.py': 3.5e9, 'scripts/eval_rl.py': 4.5e9,
            'scripts/evaluate.py': 10.0e9}


def active_footprint(job_id, current):
    if job_id in by_id:
        return footprint(by_id[job_id])
    script = by_queue.get(job_id, {}).get('script')
    return int(MEASURED.get(script, float(current.get('ram_gb', 0)) * 2**30))


def resident_replay(job_id):
    """Bytes of a running trainer's replay already resident in the page cache (part of the measured charge)."""
    files = [f for f in (ROOT / 'runs' / job_id / 'replay').glob('*') if f.is_file()] if \
        (ROOT / 'runs' / job_id / 'replay').is_dir() else []
    if not files:
        return 0
    out = subprocess.run(['fincore', '--bytes', '--noheadings', '--output', 'RES', *map(str, files)],
                         capture_output=True, text=True).stdout.split()
    return sum(int(x) for x in out if x.isdigit())


# remaining growth: footprint minus what the job already uses (its non-file PSS, and for trainers the part of the
# resident replay cache that is in the working set: the resident bytes times the cgroup's active-file share)
growth = {i: max(0, active_footprint(i, s) - pss.get(int(s.get('pid', -1)), 0) - int(active_share * resident_replay(i)))
          for i, s in state.items() if s.get('host', 'local') == 'local' and s['status'] in jobs.ACTIVE}
total = charge + sum(growth.values()) + cache_reserve
admitted, taken = [], set()
for job in candidates:
    if job['script'] == 'scripts/eval_rl.py':
        trainer = job['id'].removesuffix('-eval')
        if status(trainer) not in jobs.ACTIVE | {jobs.DONE} and trainer not in taken:
            continue
    extra = sum(nonresident[f] for f in set(task_files(job)) - counted if f in nonresident)
    if total + footprint(job) + extra >= CAP:
        continue
    admitted.append(job)
    taken.add(job['id'])
    total += footprint(job) + extra
    counted |= {f for f in task_files(job) if f in nonresident}
if admitted and not DRY:
    source = jobs.source_identity()
    report = json.loads((ROOT / 'docs/tests.json').read_text())
    assert source['source_sha256'] == report['source_sha256'] and report['full_suite_passed'] is True
    qpath = SETUP / f'bx-local-{tag}.yaml'
    with qpath.open('x') as stream:
        yaml.safe_dump({'hosts': {'local': copy.deepcopy(queue['hosts']['local'])},
                        'jobs': json.loads(json.dumps(admitted))}, stream, sort_keys=False)
    with (SETUP / f'bx-local-admission-{tag}.json').open('x') as stream:
        json.dump({'recorded_local': stamp.isoformat(), 'charge_bytes': charge, 'budget_metric': 'working_set',
                   'dataset_pages_in_working_set_bytes': dataset_pages_in_ws,
                   'memory_snapshot': memory, 'remaining_growth_by_job': growth,
                   'active_cache_allowance_bytes': cache_reserve, 'accounted_plus_admitted_bytes': total,
                   'admission_ceiling_bytes': CAP, 'ceiling_bytes': mb.CEILING,
                   'transient_overhead_bytes': mb.TRANSIENT_OVERHEAD, 'commit': source['commit'],
                   'admitted': {j['id']: footprint(j) for j in admitted}}, stream, indent=2)
    messages.append(f"ADMITTING {[j['id'] for j in admitted]} charge {charge} reserve {cache_reserve}")
    messages.extend(jobs.tick(queue_path=qpath))
elif admitted:
    messages.append(f"DRY_RUN would admit {[j['id'] for j in admitted]} total {total} charge {charge}")
counts = {}
for job in registered:
    counts.setdefault(job['script'].split('/')[-1], {}).setdefault(status(job['id']), 0)
    counts[job['script'].split('/')[-1]][status(job['id'])] += 1
print('\n'.join(messages))
print("ACCOUNTING working_set", memory["working_set"], "dataset_pages_in_ws", dataset_pages_in_ws, "raw", memory["current"], "growth", sum(growth.values()), "active_cache_allowance", cache_reserve, "total", total, "cap", CAP, "candidates", len(candidates), "host_available", memory["host_mem_available"], "psi_some_avg60", memory["pressure"].get("some", {}).get("avg60"))
print("BASELINES", json.dumps(counts, sort_keys=True))
