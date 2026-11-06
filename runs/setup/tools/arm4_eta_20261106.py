"""Four-arm campaign (arm4): throughput / ETA estimate for the 304 local rows (read-only model, no scheduler access).

Discrete-event replica of ops/campaign/baseline_local_driver.py's admission (one pass every 5 min): candidates in the
driver's order (evaluators of finished trainers first, then priority, trainers before evaluators, registration order);
head-of-line reservation (the first non-evaluator that does not fit blocks later non-evaluators; evaluators still
fill); an evaluator only with its trainer active or done; RAM charged at the measured footprints against the
admission ceiling (150 GB - 10 GB transient) minus the non-campaign baseline; GPU per MIG instance: max_jobs 3 and
declared mem_gb <= 22 (scheduler.choose_gpu: least declared memory first). Durations are measured local MIG durations
under campaign load (inputs below; ranges give the optimistic / pessimistic ETA). Not a guarantee: other admitted work,
the test-suite reservation and remote-role changes are not modelled except as the baseline argument.

    python -I arm4_eta_20261106.py [--baseline-gb 17] [--pre-a 26 --pre-b 28 --pre-c 22 --pre-d 22] [--drm 4.6]
"""
import argparse
import heapq
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm4_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--baseline-gb', type=float, default=17.0)
ap.add_argument('--ceiling-gb', type=float, default=140.0)
for arm, h in (('a', 26.0), ('b', 28.0), ('c', 22.0), ('d', 22.0)):
    ap.add_argument(f'--pre-{arm}', type=float, default=h)
ap.add_argument('--drm', type=float, default=4.6)
ap.add_argument('--eval-tail', type=float, default=0.15)
ap.add_argument('--export', type=float, default=0.05)
ap.add_argument('--render', type=float, default=0.05)
ap.add_argument('--pre-gb', type=float, default=10.77)
ap.add_argument('--drm-gb', type=float, default=3.5)
ap.add_argument('--eval-gb', type=float, default=4.5)
ap.add_argument('--render-gb', type=float, default=12.0)
ap.add_argument('--export-gb', type=float, default=4.0)
ap.add_argument('--tick', type=float, default=5 / 60)
ap.add_argument('--json', action='store_true')
args = ap.parse_args()
MIGS = 8
PRE = {'A': args.pre_a, 'B': args.pre_b, 'C': args.pre_c, 'D': args.pre_d}
rows = []  # (id, kind, priority, deps, ram, gpu_mem, duration, trainer)
for task in C.TASKS:
    for arm in C.ARMS:
        i = C.ids(arm, task)
        rows.append((i['pretrain'], 'pre', 1, [], args.pre_gb, 10, PRE[arm], None))
        rows.append((i['export'], 'export', 0, [i['pretrain']], args.export_gb, 2, args.export, None))
        for d, e in zip(i['drm'], i['eval']):
            rows.append((d, 'drm', 1, [i['export']], args.drm_gb, 4, args.drm, None))
            rows.append((e, 'eval', 1, [i['export']], args.eval_gb, 3, None, d))
for task in C.TASKS:
    for model, arm in C.RENDER_MODELS.items():
        rows.append((C.render_row_id(model, task), 'render', 1, [C.ids(arm, task)['pretrain']] if arm in C.ARMS else [],
                     args.render_gb, 6, args.render, None))
index = {r[0]: n for n, r in enumerate(rows)}
status = {r[0]: 'pending' for r in rows}
finish_at, started, done_at = {}, {}, {}
gpu_of, mig = {}, [[] for _ in range(MIGS)]
t, events = 0.0, []
cap = args.ceiling_gb - args.baseline_gb
used = 0.0
while any(s != 'done' for s in status.values()):
    # completions up to t
    while events and events[0][0] <= t + 1e-9:
        _, rid = heapq.heappop(events)
        if status[rid] != 'running':
            continue
        status[rid] = 'done'
        done_at[rid] = finish_at[rid]
        r = rows[index[rid]]
        used -= r[4]
        mig[gpu_of[rid]].remove(rid)
        # an evaluator ends shortly after its trainer
        for e in rows:
            if e[1] == 'eval' and e[7] == rid and status[e[0]] == 'running':
                finish_at[e[0]] = t + args.eval_tail
                heapq.heappush(events, (finish_at[e[0]], e[0]))
    ready = [r for r in rows if status[r[0]] == 'pending' and all(status[d] == 'done' for d in r[3])]
    ready.sort(key=lambda r: (not (r[1] == 'eval' and status[r[7]] == 'done'), r[2], r[1] == 'eval', index[r[0]]))
    blocked = False
    for r in ready:
        evaluator = r[1] == 'eval'
        if evaluator and status[r[7]] not in ('running', 'done'):
            continue
        if not evaluator and blocked:
            continue
        g = min((k for k in range(MIGS) if len(mig[k]) < 3 and sum(rows[index[x]][5] for x in mig[k]) + r[5] <= 22),
                key=lambda k: sum(rows[index[x]][5] for x in mig[k]), default=None)
        if used + r[4] >= cap or g is None:
            if not evaluator:
                blocked = True
            continue
        status[r[0]] = 'running'
        started[r[0]] = t
        used += r[4]
        mig[g].append(r[0])
        gpu_of[r[0]] = g
        if r[6] is not None:
            finish_at[r[0]] = t + r[6]
            heapq.heappush(events, (finish_at[r[0]], r[0]))
        elif status[r[7]] == 'done':
            finish_at[r[0]] = t + args.eval_tail
            heapq.heappush(events, (finish_at[r[0]], r[0]))
    t += args.tick
    if t > 2000:
        raise SystemExit('no progress')
per_arm = {arm: max(done_at[x] for task in C.TASKS for x in [C.ids(arm, task)['pretrain']] + C.ids(arm, task)['eval'])
           for arm in C.ARMS}
first_task = {arm: min(max(done_at[x] for x in C.ids(arm, task)['eval']) for task in C.TASKS) for arm in C.ARMS}
pre_done = max(done_at[C.ids(a, t_)['pretrain']] for a in C.ARMS for t_ in C.TASKS)
out = {'inputs': vars(args), 'capacity_gb': cap, 'makespan_h': round(max(done_at.values()), 1),
       'all_pretraining_done_h': round(pre_done, 1),
       'arm_complete_h': {a: round(v, 1) for a, v in per_arm.items()},
       'first_complete_task_per_arm_h': {a: round(v, 1) for a, v in first_task.items()}}
if args.json:
    print(json.dumps(out))
else:
    print(json.dumps(out, indent=1))
