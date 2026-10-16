"""Stage 2 attribution report (frozen method; attribution only, no method decision). Reports only comparisons whose
runs are all verified by the Stage 2 local driver.

Representation: the variant's 200k full-split evaluation (seed 0) against the full method's own 200k evaluation
(fulleval-ls-a200-m3d-seed0-<task>-200k), every metric of the prospective reporting margins
(runs/setup/stage2-reporting-margins-20261013.json, recorded before registration), with that file's readers.

RL (variants with RL arms): DrM 1M, seeds 1000-1002, against the full-method reference trainers on the same seeds.
Last-5 success on training cameras (the decision quantity of the existing RL rule), held-out cameras and moving-camera
trajectories. Margin = max(0.10, 2 x standard error of the difference of means), the RL rule adopted 2026-10-09
(docs/EXPERIMENT_LOG.md "RL proxies (S2 adopted)"); no separate Stage 2 RL margin was registered, so that existing
rule is applied as is, and to held-out/trajectory success as context.

    python -I stage2_report.py            # all complete comparisons
"""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
SETUP = ROOT / 'runs/setup'
MARGINS = SETUP / 'stage2-reporting-margins-20261013.json'
MARGINS_SHA = '5fadd04fd5050a440f13d5adad540e424a0f78257ed3f82eacec36df31db5677'
SEEDS = (1000, 1001, 1002)
TASKS = ('hammer', 'pick-place')
SUCCESS = {'train': 'train_cameras_success', 'heldout': 'heldout_cameras_success', 'traj': 'trajectories_success'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proof(job):
    path = SETUP / f's2-{job}-completion.json'
    return json.loads(path.read_text()) if path.is_file() else None


def value(s, column, stride):
    return float(s[f'metric/{column}@s{stride}'])


def read(s, reader, column):
    if reader == 'mean':
        return 0.5 * (value(s, column, 2) + value(s, column, 6))
    if reader == 's6':
        return value(s, column, 6)
    p90 = {k: 0.5 * (value(s, f'cd_{column}_p2g_p90', k) + value(s, f'cd_{column}_g2p_p90', k)) for k in (2, 6)}
    return 0.5 * (p90[2] + p90[6]) if reader == 'sym_mean' else p90[6]


def summary(job):
    path = ROOT / 'runs' / job / 'eval/step_0200000/summary.json'
    return json.loads(path.read_text()), sha(path)


def last5(job):
    record = proof(job)
    run = ROOT / 'runs' / job
    path = run / 'eval.jsonl'
    if not path.is_file():
        path = ROOT / 'runs' / (job + '-eval') / 'eval.jsonl'
    assert record and sha(path) == record['eval_jsonl_sha256'], job
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [r['step'] for r in rows] == list(range(10000, 1000001, 10000)), job
    return {k: sum(r[c] for r in rows[-5:]) / 5 for k, c in SUCCESS.items()}


def rl_compare(arm, ref):
    out = {}
    for kind in SUCCESS:
        a, b = [x[kind] for x in arm], [x[kind] for x in ref]
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        var = lambda xs, m: sum((x - m) ** 2 for x in xs) / (len(xs) - 1)  # noqa: E731
        se = math.sqrt(var(a, ma) / len(a) + var(b, mb) / len(b))
        margin = max(0.10, 2 * se)
        delta = ma - mb
        out[kind] = {'arm_seeds': a, 'reference_seeds': b, 'arm_mean': ma, 'reference_mean': mb, 'delta': delta,
                     'se_difference': se, 'margin': margin,
                     'verdict': 'better' if delta > margin else 'worse' if delta < -margin else 'within margin'}
    return out


assert sha(MARGINS) == MARGINS_SHA
margins = json.loads(MARGINS.read_text())
NAMES = ('stage2-registration-20261013.json', 'stage2-crop-registration-20261015.json',
         'stage2-nosynth-rl-registration-20261016.json')
registrations = [json.loads((SETUP / n).read_text()) for n in NAMES if (SETUP / n).is_file()]
variants = {}
for registration in registrations:
    for job in registration['jobs']:
        if job['script'] == 'scripts/train.py':
            variants.setdefault(job['stage2_variant'], set()).add(job['id'].rsplit('-seed0-', 1)[1])
rl_arms = {j['stage2_variant'] for r in registrations for j in r['jobs'] if j['script'] == 'scripts/train_rl.py'}

report = {'recorded_local': datetime.now().astimezone().isoformat(), 'margins_sha256': MARGINS_SHA,
          'purpose': 'attribution only; the frozen method does not change', 'comparisons': {}}
lines = []
for variant in sorted(variants):
    for task in TASKS:
        evaluation = f'fulleval-s2ab-{variant}-seed0-{task}-200k'
        reference = f'fulleval-ls-a200-m3d-seed0-{task}-200k'
        if not proof(evaluation):
            continue
        entry = {}
        (sv, sv_sha), (sr, sr_sha) = summary(evaluation), summary(reference)
        assert sv_sha == proof(evaluation)['summary_sha256']
        rows, flagged = {}, []
        for metric, spec in margins['tasks'][task]['metrics'].items():
            v, r = read(sv, spec['reader'], spec['column']), read(sr, spec['reader'], spec['column'])
            delta = v / r - 1.0 if spec['relative'] else v - r
            signed = delta if spec['higher_is_better'] else -delta
            margin = spec['primary']['margin']
            verdict = 'better' if signed > margin else 'worse' if signed < -margin else 'within margin'
            rows[metric] = {'variant': v, 'reference': r, 'delta': delta, 'relative': spec['relative'],
                            'margin': margin, 'verdict': verdict}
            if verdict != 'within margin':
                flagged.append(metric)
        entry['representation'] = {'variant_eval': evaluation, 'reference_eval': reference,
                                   'reference_summary_sha256': sr_sha, 'metrics': rows}
        lines.append(f'\n== {variant} / {task}: representation vs full method (200k, seed 0); '
                     f'{len(flagged)} of {len(rows)} metrics beyond margin')
        for metric in flagged:
            r = rows[metric]
            shown = f"{100 * r['delta']:+.1f}% (margin {100 * r['margin']:.1f}%)" if r['relative'] else \
                f"{r['delta']:+.3f} (margin {r['margin']:.3f})"
            lines.append(f"   {metric:22s} {r['reference']:9.3f} -> {r['variant']:9.3f}  {shown}  {r['verdict']}")
        if variant in rl_arms:
            jobs = {s: (f's2rl-{variant}-{task}-s{s}', f's2rl-full-{task}-s{s}') for s in SEEDS}
            if all(proof(a) and proof(b) for a, b in jobs.values()):
                rl = rl_compare([last5(jobs[s][0]) for s in SEEDS], [last5(jobs[s][1]) for s in SEEDS])
                entry['rl'] = {'seeds': list(SEEDS), 'jobs': jobs, 'success_last5': rl}
                for kind, r in rl.items():
                    ref_seeds = [round(x, 3) for x in r['reference_seeds']]
                    arm_seeds = [round(x, 3) for x in r['arm_seeds']]
                    lines.append(f"   RL {kind:7s} full {r['reference_mean']:.3f} {ref_seeds} -> {variant} "
                                 f"{r['arm_mean']:.3f} {arm_seeds}  delta {r['delta']:+.3f} "
                                 f"(margin {r['margin']:.3f})  {r['verdict']}")
            else:
                entry['rl'] = 'incomplete: waiting for all seeds of both arms'
                lines.append('   RL: waiting for all seeds of both arms')
        report['comparisons'][f'{variant}/{task}'] = entry
out = SETUP / f"stage2-report-{datetime.now():%Y%m%d-%H%M%S}.json"
out.write_text(json.dumps(report, indent=2) + '\n')
print('\n'.join(lines))
print('\nREPORT', out, sha(out))
