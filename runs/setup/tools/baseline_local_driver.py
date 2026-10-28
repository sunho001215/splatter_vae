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
                 'afk2-registration-bin-picking-coffee-push-door-open-peg-insert-side-peg-unplug-side-stick-push-20261020.json',
                 # fresh unchanged bin-picking retraining (spike investigation 2026-10-21; cause unresolved)
                 'afk2rerun-seed0-bin-picking-200k-registration-20261021.json',
                 # chamfer_lean pilot pretrains (hammer, pick-place; 200k schedule stopped at 50k; 2026-10-21)
                 'chl-registration-hammer-pick-place-20261021.json',
                 # chamfer_lean 200k, RL and eight-task expansion (user instruction 2026-10-22)
                 'chl-registration-eight-task-20261022.json',
                 # masking ablation, per-frame pilots and T=1 stacking, hammer + pick-place (user instructions 2026-10-23)
                 'mpf-registration-20261023.json',
                 # vision-only DrM, original frozen encoder vs CNN, eight tasks (user instruction 2026-10-23)
                 'vision-only-registration-20261023.json',
                 # chamfer_lean_randmask eight-task expansion, the six non-pilot tasks (user instruction 2026-10-24)
                 'chlrm-registration-eight-task-20261024.json',
                 # encoder pilots on chlrm, hammer / pick-place / bin-picking (user directive 2026-10-25): lowest
                 # priority (4); capacity-based admission after every other ready row (user 2026-10-25 16:xx)
                 'encp-registration-20261025.json',
                 # encoder 2x2 on all eight tasks (user directive 2026-10-26): cs / w4 on five tasks, cs+w4 on eight;
                 # priority 4, the same enc- handling (after every other ready candidate)
                 'encx-registration-20261026.json',
                 # SNeRL-code SAC comparison groups (user directive 2026-10-28): priority 2, ds- rows (local pool rows of the
                 # frozen-encoder families; the CNN and SNeRL rows are remote)
                 'downstream-sac-registration-s2000-20261028.json']
ONLY = {'baselines-sanity-registration-20261016.json': {'bxs-dynamo-hammer-rand-s1000-eval'}}
PLACEMENT = Path('/home/ws/ws/splatter4d/runs/setup/baselines-placement-20261018.json')
POOL = Path('/home/ws/ws/splatter4d/runs/setup/pool-placement.json')
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
        # host-neutral pool placement (2026-10-23; host-specific fields only): runs/setup/pool-placement.json
        if POOL.is_file():
            overlay = {**overlay, **json.loads(POOL.read_text())['effective_jobs']}
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
    if job['script'] in ('scripts/train_rl.py', 'scripts/eval_rl.py', 'scripts/train_sac.py'):
        return []  # DrM / SAC simulate Meta-World live; they read no dataset file
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
    args = job['args']
    if job['script'] == 'scripts/train_rl.py' and '--encoder' in args and args[args.index('--encoder') + 1] == 'cnn':
        # end-to-end CNN DrM (2026-10-24): its pixel replay (49 GB at 1M steps) is a mapped memmap whose resident
        # pages count in the budget (Pss_File) and grow with the steps; charge the declared ram_gb (48)
        return int(float(job['ram_gb']) * 2**30)
    if job['script'] in ('scripts/train_rl.py', 'scripts/eval_rl.py'):
        # + the trainer's preallocated replay file, which DrM sampling keeps resident in the page cache (measured
        # 2026-10-17: DynaMo 1.1 GB, 3D-MVP 2.5 GB; evicting it refills within minutes)
        # (the replay file's clean page cache is excluded from the budget since 2026-10-21, option (b))
        return int(MEASURED[job['script']])
    if job['script'] == 'scripts/train.py':  # our pretraining trainer: measured 10.77 GB max sampled PSS (rule 2026-10-11)
        return int(MEASURED[job['script']])
    if job['script'] == 'scripts/train_sac.py' and job.get('host', 'local') == 'local':  # frozen-encoder SAC (2026-10-28)
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
# CNN trainers frozen by SIGSTOP for the budget (2026-10-24; runs/setup/cnn-freeze-20261024.json) do not grow while
# frozen (their current pages are in the measured charge); they and their evaluators migrate to the remote (user
# 2026-10-24) and are never admitted locally again
FROZEN = set(json.loads((SETUP / 'cnn-freeze-20261024.json').read_text())['frozen']) \
    if (SETUP / 'cnn-freeze-20261024.json').is_file() else set()
candidates = [j for j in jobs.eligible(registered, view) if j.get('host', 'local') == 'local'
              and j['id'].removesuffix('-eval') not in FROZEN]
from admission_pause import paused  # noqa: E402  (intentional pauses, user 2026-10-28)
candidates = [j for j in candidates if not paused(j['id'])]
if stage2_waiting:
    messages.append(f'YIELD to {len(stage2_waiting)} waiting Stage 2 local jobs, e.g. {stage2_waiting[:3]}')
    candidates = []
# Encoder pilots (user directive 2026-10-25): lowest priority (4). Capacity-based scheduling (user 2026-10-25 16:xx,
# superseding the blanket ENC_HOLD): ready rows are offered in priority order and every ready row that fits the
# remaining capacity is admitted; rows waiting for unfinished dependencies never block others. Encoder-pilot rows are
# offered after every other ready candidate.
ENC = 'enc-'
# evaluators of already finished trainers first (short, and they release the run's cached snapshots), then by
# priority with trainers before their companions
candidates.sort(key=lambda j: (not (j['script'] == 'scripts/eval_rl.py'
                                    and status(j['id'].removesuffix('-eval')) == jobs.DONE),
                               j['priority'], j['script'] == 'scripts/eval_rl.py', registered.index(j)))
candidates.sort(key=lambda j: j['id'].startswith(ENC))  # stable: encoder-pilot rows after every other candidate
memory = mb.snapshot()
# Budget (user decision 2026-10-21, option (b); memory_budget.py): every byte of the cgroup except verified clean,
# disk-backed file cache, active or inactive. Clean dataset and replay cache are therefore excluded, and no dataset-cache
# allowance is projected.
files = sorted({f for j in registered for f in task_files(j) if f.is_file()})
allowance = {f: 0 for f in files}
nonresident = allowance  # name kept for the admission loop below
counted = {f for j in registered if status(j['id']) in jobs.ACTIVE for f in task_files(j) if f in allowance}
cache_reserve = 0
charge = memory['budget_usage']
active_share = memory['active_file'] / max(1, memory['active_file'] + memory['inactive_file'])
pss = session_pss()
# Measured footprints of the existing campaign scripts (the Stage 2 driver's table); new-baseline jobs: declared ram_gb.
# scripts/train_sac.py (2026-10-28): measured 2.0-2.2 GB RSS for frozen-encoder SAC (csw4, ReViWo, 3D-MVP A, 100k-latent replay);
# charged as the DrM trainer's 3.5 GB
MEASURED = {'scripts/train.py': 10.77e9, 'scripts/train_rl.py': 3.5e9, 'scripts/train_sac.py': 3.5e9,
            'scripts/eval_rl.py': 4.5e9,
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
growth = {i: max(0, active_footprint(i, s) - pss.get(int(s.get('pid', -1)), 0))
          for i, s in state.items() if s.get('host', 'local') == 'local' and s['status'] in jobs.ACTIVE
          and i not in FROZEN}
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
        json.dump({'recorded_local': stamp.isoformat(), 'charge_bytes': charge, 'budget_metric': 'budget_usage',
                   'budget_definition': 'memory.current - clean disk-backed file cache (option b)',
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
print("ACCOUNTING budget_usage", charge, "clean_disk_file_excluded", memory["clean_disk_file"], "raw", memory["current"], "growth", sum(growth.values()), "active_cache_allowance", cache_reserve, "total", total, "cap", CAP, "candidates", len(candidates), "host_available", memory["host_mem_available"], "psi_some_avg60", memory["pressure"].get("some", {}).get("avg60"),
      "io_psi_full_avg60", memory["io_pressure"].get("full", {}).get("avg60"))
print("BASELINES", json.dumps(counts, sort_keys=True))
