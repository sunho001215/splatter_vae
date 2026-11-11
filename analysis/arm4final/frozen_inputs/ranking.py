"""Candidate ranking on the original eight Meta-World tasks under DrM (reads data/per_run_summaries.csv only).

Criteria (each as last-5 = mean of evaluations 960k..1M, and AUC = mean over the 10k..1M grid):
  training-camera nR, training-camera success, held-out static cameras success (eval0-3),
  moving cameras traj 1x (traj_lateral + traj_circular) success and nR.
Point values = equal-task means of per-task seed means (as rl_analysis.py). Uncertainty = the same task-stratified
percentile bootstrap (B = 10000, seeds resampled within each task, arms independent); here one joint draw per arm
resamples the seed indices ONCE and applies them to every criterion, so the per-draw ranks and the mean rank across
criteria keep the between-criterion correlation of an arm. Reported: rank of the point estimate, P(rank 1) and P(top 3)
over draws, the 95 % interval of the rank, the leader's margin over the runner-up (95 % CI), and the mean rank across
criteria (point, 95 % interval, P(best mean rank)).
Sets: 'ours9' = full, chl, chlrm, chlhm, cs, w4, csw4, cst50, cst75 (all 8 tasks; cst50 stick-push 2 complete seeds,
no imputation); 'chamfer8' = ours9 without the original (full is the reference, not a candidate, under the
Chamfer-stays directive); 'all17' = ours9 + the 8 baselines. Sensitivity: the same on the 7 tasks without stick-push (3/3 seeds
everywhere). Pilots w8 / h2 / hn: ranked only on their 3 tasks (bin-picking, hammer, pick-place) against every 8-task arm
restricted to those tasks - flagged, never mixed into the 8-task ranking.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rl_common as C  # noqa: E402

D = Path(__file__).resolve().parent.parent / 'data'
B = 10000
OURS9 = ('full', 'chl', 'chlrm', 'chlhm', 'cs', 'w4', 'csw4', 'cst50', 'cst75')
BASE = ('cnn', 'cnnnoshift', 'snerl', 'sincro', 'mvp3d-single', 'mvp3d-multiview', 'reviwo', 'dynamo')
PILOTS = ('w8', 'h2', 'hn')
CRIT = [('train_nR_last5', 'nR last-5'), ('train_nR_auc', 'nR AUC'),
        ('train_success_last5', 'success last-5'), ('train_success_auc', 'success AUC'),
        ('heldout_success_last5', 'held-out success last-5'), ('heldout_success_auc', 'held-out success AUC'),
        ('traj_success_last5', 'moving success last-5'), ('traj_success_auc', 'moving success AUC'),
        ('traj_nR_last5', 'moving nR last-5'), ('traj_nR_auc', 'moving nR AUC')]
# the four headline criteria of the request (return nR, success, held-out, moving), each last-5 and AUC; moving uses nR
HEADLINE8 = ['train_nR_last5', 'train_nR_auc', 'train_success_last5', 'train_success_auc', 'heldout_success_last5',
             'heldout_success_auc', 'traj_nR_last5', 'traj_nR_auc']


def load():
    v = defaultdict(lambda: defaultdict(list))  # v[arm][task] = list of per-run dicts (complete runs only)
    for r in csv.DictReader(open(D / 'per_run_summaries.csv')):
        if r['status'] == 'complete':
            v[r['arm']][r['task']].append({m: float(r[m]) for m, _ in CRIT})
    return v


def draws(v, arm, tasks, rng):
    """(B, n_crit) bootstrap distribution of the equal-task means; one seed resample per task and draw, shared by all
    criteria. Also returns the point estimates."""
    acc = np.zeros((B, len(CRIT)))
    point = np.zeros(len(CRIT))
    for t in tasks:
        X = np.array([[run[m] for m, _ in CRIT] for run in v[arm][t]])  # (n_seeds, n_crit)
        n = len(X)
        idx = rng.integers(0, n, size=(B, n))
        acc += X[idx].mean(1)
        point += X.mean(0)
    return acc / len(tasks), point / len(tasks)


def rank_desc(M):
    """M (..., n_arms) -> ranks 1 = highest, along the last axis."""
    return np.argsort(np.argsort(-M, axis=-1, kind='stable'), axis=-1, kind='stable') + 1


def analyse(v, arms, tasks, rng, set_name, scope):
    Dr, P = {}, {}
    for a in arms:
        Dr[a], P[a] = draws(v, a, tasks, rng)
    rows, summ = [], []
    hidx = [i for i, (m, _) in enumerate(CRIT) if m in HEADLINE8]
    allR = np.zeros((B, len(arms), len(CRIT)))
    for ci, (m, lab) in enumerate(CRIT):
        M = np.stack([Dr[a][:, ci] for a in arms], axis=1)  # (B, n_arms)
        R = rank_desc(M)
        allR[:, :, ci] = R
        pt = np.array([P[a][ci] for a in arms])
        prank = rank_desc(pt[None, :])[0]
        lead, second = np.argsort(-pt)[:2]
        margin = M[:, lead] - M[:, second]
        for k, a in enumerate(arms):
            rows.append({'scope': scope, 'set': set_name, 'criterion': m, 'label': lab, 'arm': a, 'value': pt[k],
                         'ci_lo': float(np.percentile(M[:, k], 2.5)), 'ci_hi': float(np.percentile(M[:, k], 97.5)),
                         'rank': int(prank[k]), 'rank_ci_lo': int(np.percentile(R[:, k], 2.5)),
                         'rank_ci_hi': int(np.percentile(R[:, k], 97.5)), 'p_rank1': float((R[:, k] == 1).mean()),
                         'p_top3': float((R[:, k] <= 3).mean()),
                         'leader': arms[lead], 'runner_up': arms[second], 'leader_margin': float(pt[lead] - pt[second]),
                         'leader_margin_ci_lo': float(np.percentile(margin, 2.5)),
                         'leader_margin_ci_hi': float(np.percentile(margin, 97.5))})
    for name, cols in (('headline8', hidx), ('all10', list(range(len(CRIT))))):
        mr = allR[:, :, cols].mean(2)  # (B, n_arms)
        ptR = np.stack([rank_desc(np.array([P[a][c] for a in arms])[None, :])[0] for c in cols], 1).mean(1)
        best = (mr == mr.min(1, keepdims=True))
        for k, a in enumerate(arms):
            summ.append({'scope': scope, 'set': set_name, 'summary': name, 'arm': a, 'mean_rank': float(ptR[k]),
                         'mean_rank_ci_lo': float(np.percentile(mr[:, k], 2.5)),
                         'mean_rank_ci_hi': float(np.percentile(mr[:, k], 97.5)),
                         'p_best_mean_rank': float((best[:, k] / best.sum(1)).mean()),
                         'n_criteria_rank1_point': int(sum(1 for c in cols if
                                                           rank_desc(np.array([P[x][c] for x in arms])[None, :])[0][k] == 1))})
    return rows, summ


def main():
    v = load()
    rng = np.random.default_rng(20261106)
    nseeds = {a: {t: len(v[a][t]) for t in C.TASKS_OF[a]} for a in OURS9 + BASE + PILOTS}
    assert all(nseeds[a][t] == 3 for a in OURS9 + BASE + PILOTS for t in C.TASKS_OF[a] if (a, t) != ('cst50', 'stick-push'))
    rows, summ = [], []
    for scope, tasks in (('8 tasks (cst50 stick-push n=2)', C.TASKS), ('7 tasks without stick-push', C.NO_STICK)):
        for set_name, arms in (('ours9', OURS9), ('chamfer8', OURS9[1:]), ('all17', OURS9 + BASE)):
            r, s = analyse(v, arms, tasks, rng, set_name, scope)
            rows += r
            summ += s
    r, s = analyse(v, OURS9 + PILOTS, C.PILOT, rng, 'ours9+pilots (PILOT, 3 tasks)', '3 pilot tasks (bin-picking, hammer, pick-place)')
    rows += r
    summ += s
    with open(D / 'ranking_by_criterion.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(D / 'ranking_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    (D / 'ranking_meta.json').write_text(json.dumps({'doc': __doc__, 'B': B, 'rng_seed': 20261106,
                                                     'n_complete_seeds': nseeds, 'headline8': HEADLINE8}, indent=1))
    for s in summ:
        if s['summary'] == 'headline8':
            print(f"{s['scope'][:8]} {s['set'][:12]:12s} {s['arm']:16s} mean rank {s['mean_rank']:5.2f} "
                  f"[{s['mean_rank_ci_lo']:4.2f},{s['mean_rank_ci_hi']:4.2f}] P(best) {s['p_best_mean_rank']:.2f} "
                  f"#1 on {s['n_criteria_rank1_point']}")


if __name__ == '__main__':
    main()
