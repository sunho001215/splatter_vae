"""One pass of local admission for the new-baseline jobs (registrations listed below), job by job under the local
full-charge cap through the scheduler. Same accounting as stage2_local_driver.py (cgroup budget usage + remaining
growth of every active local job, 150 GB budget less transient, fixed 2 GB staging, and validation reservations) and the same lock file, so the two
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
from ops.jobs import scheduler as jobs  # noqa: E402
from ops.jobs.retired import retired_reason  # noqa: E402

sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
from ops.resources import memory_budget as mb  # noqa: E402
from ops.resources.reservations import validation_reservation as _validation_reservation  # noqa: E402

# Local budget (user instruction 2026-10-21): the cgroup working set max(0, memory.current - inactive_file) against the
# 150 GB ceiling, minus a transient-overhead margin; see runs/setup/tools/memory_budget.py.
# Temporary local validation reservation (user 2026-11-02): RAM withheld from local admission for the real-world/DROID
# validation and the integration suite (scripts/run_tests.py, +40 GB). runs/setup/local-validation-reservation.json
# {"ram_bytes": N, "expires_unix": T, "reason": ...}; absent or expired = 0. GPUs are reserved by leaving them out of
# experiments/queue.yaml limits.


VALIDATION_RESERVATION = _validation_reservation()
CAP = mb.TRAINING_CEILING - VALIDATION_RESERVATION
# The local sanity set moved to the remote (baselines-sanity-remote-registration-20261016.json). Only the evaluator of the
# kept local pair `bxs-dynamo-hammer-rand-s1000` (trainer done locally) is still admitted here (2026-10-17).
# Eight-task runs (2026-10-17): the local rows of baselines-eight-task-registration-20261017.json (DynaMo and 3D-MVP
# condition B DrM; their pretraining exports come from the remote).
# SNeRL eight-task runs (2026-10-20): local DrM rows of baselines-snerl-eight-task-registration-20261020.json.
# Variant action_future (k = history = future stride) was CANCELLED and superseded on 2026-10-20 (user instruction;
# runs/setup/actfut-cancellation-20261020.json): its registration is no longer admitted.
# Its replacement future_k2 (fixed history stride 2, future stride k2 in 1..6; runs/setup/afk2-definition-20261020.json).
REGISTRATIONS = ['baselines-sanity-registration-20261016.json', 'baselines-eight-task-registration-20261017.json',
                 'baselines-snerl-eight-task-registration-20261020.json',
                 'afk2-registration-hammer-pick-place-20261020.json',
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
                 'downstream-sac-registration-s2000-20261028.json',
                 # camera suite v2 re-evaluation of completed DrM policies (user directive 2026-10-29): rows on the
                 # host holding the final snapshot; csw4 and CNN first (priority 1)
                 'cameras-v2-reeval-registration-drm-20261029.json',
                 # self-render pilot pretrainings and exports (user 2026-10-30: raise the pilot's priority, next suitable
                 # capacity on either host): pool rows, admitted here only while placed local
                 'selfrender-registration-20261029.json',
                 # self-render v4 = v2 + +-5 % radial camera jitter + 5 %-relaxed viewpoint domain (user 2026-10-31;
                 # replaces the stopped v2 pilot): pretrainings, exports (pool rows, admitted here only while placed
                 # local); retired trainer/evaluator rows are filtered below
                 'selfrender-v4-registration-20261031.json',
                 # SNeRL drawer-open rerun on release dd86d78 (user 2026-10-31); filter retired rows
                 'snerl-drawer-open-r2-registration-20261031.json',
                 # 75 % hybrid masking on chlrm, eight tasks (user directive 2026-11-03, HIGH priority: pretraining,
                 # DrM trainers and evaluators 1, exports 0; non-preemptive)
                 'chlhm-registration-eight-task-20261103.json',
                 'cst-registration-eight-task-20261104.json',
                 # hybrid analyses run locally (rendering; 2026-11-04)
                 'ana-chlhm-registration-20261104.json',
                 # four-arm local campaign (user directive 2026-11-06; runs/setup/arm4-definition-20261106.json):
                 # Original + CSW4 encoder / + Chamfer, cst50 + depth gradient / + visibility, eight tasks, and their
                 # render-analysis rows; every row pinned local (Meta-World stays local). Priorities as the CSW4
                 # tube-ratio campaign (1; exports 0; render rows 1, registered after the training rows). Listed last:
                 # within the order below (evaluators of finished trainers first, then priority, trainers before
                 # evaluators, registration order; enc- rows last) equal-priority ties go to the earlier registrations
                 'arm4-registration-eight-task-20261106.json']
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
    raise SystemExit(0) from None

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
        registered += [j for j in record_jobs if retired_reason(j, by_queue) is None
                       and (name not in ONLY or j['id'] in ONLY[name])
                       and not (j.get('host', 'local') == 'local' and j.get('pin_host') == 'remote')]  # waits for its move
# Unified pool (2026-11-11 ~20:15): explicitly placed remote-resident training shares the local budget.
from ops.jobs import fair_share, host_roles, realworld_local  # noqa: E402

_registered_ids = {j['id'] for j in registered}
registered.extend(j for j in realworld_local.local_rows(queue['jobs'], host_roles.load_roles())
                  if j['id'] not in _registered_ids)

by_id = {j['id']: j for j in registered}
if not registered:
    print('BASELINES nothing registered for local admission')
state = jobs.read_registry()
messages = []


def status(job_id):
    current = state.get(job_id, {})
    if current.get('status') == 'crashed' and current['failures'] <= int(by_id[job_id].get('max_restarts', 1)):
        return jobs.PENDING
    return current.get('status', jobs.PENDING)


def task_files(job):
    if host_roles.remote_resident(job, _roles, by_queue):
        return []  # resolved remote-resident inputs read no Meta-World dataset
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


def trained_by(job):
    """The training run an evaluator reads: its companion trainer, or the completed run a camera suite v2
    re-evaluation row names (2026-10-29: 'reevaluates')."""
    return job.get('reevaluates') or job['id'].removesuffix('-eval')


def session_pss():
    # non-file PSS (Pss_Anon + Pss_Shmem): mapped file cache must not re-enter the working-set budget (2026-10-21)
    out = mb.session_nonfile_pss()
    return out


def footprint(job):
    from ops.jobs.footprints import local_footprint

    return local_footprint(job, globals().get('by_queue', {}))


# Baselines rank below the registered Stage 2 work (user 2026-10-13): yield while any local Stage 2 job is waiting
# (pending with its dependencies done, or crashed but restartable).
from ops.jobs.admission_pause import paused as _paused_now  # noqa: E402

stage2_waiting = []
for job in queue['jobs']:
    if job.get('stage') != 2 or job.get('host', 'local') != 'local':
        continue
    current = state.get(job['id'], {})
    restartable = current.get('status') == 'crashed' and current['failures'] <= int(job.get('max_restarts', 1))
    if _paused_now(job['id']):  # an intentionally paused Stage 2 row cannot start, so it never forces a yield
        continue
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
from ops.jobs.admission_pause import paused  # noqa: E402  (intentional pauses, user 2026-10-28)

candidates = [j for j in candidates if not paused(j['id'])]
# Unified roles (configs/hosts/roles.yaml): locally admitted classes, including remote-resident training, and
# no row held by migration ownership. The scheduler checks roles and verified local inputs again at launch.
_roles, _ownership, _context = host_roles.load_roles(), host_roles.read_ownership(), by_queue
candidates = [j for j in candidates if host_roles.refusal(j, 'local', _roles, _ownership, _context) is None]
if stage2_waiting:
    messages.append(f'YIELD equal/lower priority to {len(stage2_waiting)} waiting Stage 2 local jobs, '
                    f'e.g. {stage2_waiting[:3]}')
    ceiling = min(by_queue[i].get('priority', 5) for i in stage2_waiting)
    candidates = [j for j in candidates if j.get('priority', 5) < ceiling]
# Encoder pilots (user directive 2026-10-25): lowest priority (4). Capacity-based scheduling (user 2026-10-25 16:xx,
# superseding the blanket ENC_HOLD): ready rows are offered in priority order and every ready row that fits the
# remaining capacity is admitted; rows waiting for unfinished dependencies never block others. Encoder-pilot rows are
# offered after every other ready candidate.
ENC = 'enc-'
# evaluators of already finished trainers first (short, and they release the run's cached snapshots), then by
# priority with trainers before their companions
candidates.sort(key=lambda j: (not (j['script'] == 'scripts/eval_rl.py'
                                    and status(trained_by(j)) == jobs.DONE),
                               j.get('priority', 5), j['script'] == 'scripts/eval_rl.py', registered.index(j)))
# Stable: encoder pilots follow the old order within each queue-priority tier.
candidates.sort(key=lambda j: (j.get('priority', 5), j['id'].startswith(ENC)))
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
# Real-world FQL (2026-11-12): VmHWM 1.5 GiB frozen / 4.3 GiB e2e_cnn; charge 3 / 8 GiB
# via footprints.fql_ram_gb for admission AND active_footprint growth reservation.
# scripts/train_sac.py (2026-10-28): measured 2.0-2.2 GB RSS (csw4, ReViWo, 3D-MVP A; 100k-latent replay),
# charged as the DrM trainer's 3.5 GB
MEASURED = {'scripts/train.py': 10.77e9, 'scripts/train_rl.py': 3.5e9, 'scripts/train_sac.py': 3.5e9,
            'scripts/eval_rl.py': 4.5e9,
            'scripts/evaluate.py': 10.0e9}


def active_footprint(job_id, current):
    if job_id in by_queue:
        return footprint(by_queue[job_id])
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
# Equal dominant-resource guarantees across active/ready domains. Retain the existing candidate order within each
# family and reserve the first blocked trainer per family; evaluators may still fill its remaining capacity.
def family(job):
    return host_roles.job_class(job, by_queue).domain or 'metaworld'


fair_active = [{'id': i, 'family': family(by_queue.get(i, {})),
                'ram_bytes': max(active_footprint(i, current), pss.get(int(current.get('pid') or -1), 0))
                if i not in FROZEN else pss.get(int(current.get('pid') or -1), 0),
                'exclusive_gpu': current.get('exclusive_gpu', False) or by_queue.get(i, {}).get('exclusive_gpu', False),
                'gpu': current.get('gpu'), 'mem_gb': jobs.reservation(current, by_queue.get(i, {}), 'mem_gb')}
               for i, current in state.items() if current.get('host', 'local') == 'local'
               and current['status'] in jobs.RESERVING]
fair_candidates = [{'id': j['id'], 'priority': j.get('priority', 5), 'family': family(j), 'ram_bytes': footprint(j),
                    'exclusive_gpu': j.get('exclusive_gpu', False),
                    'gpu': j.get('gpu', 'any'), 'mem_gb': float(j.get('mem_gb', 0)),
                    'evaluator': j['script'] == 'scripts/eval_rl.py', 'trainer': trained_by(j),
                    'trainer_ready': status(trained_by(j)) in jobs.ACTIVE | {jobs.DONE}}
                   for j in candidates]
fair = fair_share.select(fair_candidates, fair_active, queue['hosts']['local']['limits'], CAP, total)
admitted = [by_id[i] for i in fair['admitted']]
total = fair['total_ram_bytes']
trainer_blocked = fair['blocked']
print('FAIR_SHARE ' + json.dumps(fair['report'], sort_keys=True))
if admitted and not DRY:
    source = jobs.source_identity()
    report = json.loads((ROOT / 'docs/tests.json').read_text())
    assert source['source_sha256'] == report['source_sha256'] and report['full_suite_passed'] is True
    qpath = SETUP / f'bx-local-{tag}.yaml'
    # Dispatch copies carry their class as an explicit role (host_roles.with_role, as the remote driver does): the
    # tick sees only the admitted rows, so an export classified through its real-world producer stays remote-resident
    # (localized at launch) instead of becoming a domain-free row.
    # The MIG instance fair_share chose (the scheduler's own rule, honouring reservations for blocked heads) travels
    # with the row, so the tick cannot fragment the instances differently (Codex review 2).
    dispatched = [{**host_roles.with_role(j, by_queue), 'gpu': fair['assigned'][j['id']]} for j in admitted]
    with qpath.open('x') as stream:
        yaml.safe_dump({'hosts': {'local': copy.deepcopy(queue['hosts']['local'])},
                        'jobs': json.loads(json.dumps(dispatched))}, stream, sort_keys=False)
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
for family_name, trainer_id in trainer_blocked.items():
    messages.append(f'RESERVED for waiting {family_name} trainer {trainer_id} '
                    '(later family trainers wait; evaluators may fill)')
counts = {}
for job in registered:
    counts.setdefault(job['script'].split('/')[-1], {}).setdefault(status(job['id']), 0)
    counts[job['script'].split('/')[-1]][status(job['id'])] += 1
print('\n'.join(messages))
print("ACCOUNTING budget_usage", charge, "clean_disk_file_excluded", memory["clean_disk_file"],
      "raw", memory["current"], "growth", sum(growth.values()), "active_cache_allowance", cache_reserve,
      "total", total, "cap", CAP, "candidates", len(candidates), "host_available", memory["host_mem_available"],
      "psi_some_avg60", memory["pressure"].get("some", {}).get("avg60"),
      "io_psi_full_avg60", memory["io_pressure"].get("full", {}).get("avg60"))
print("BASELINES", json.dumps(counts, sort_keys=True))
