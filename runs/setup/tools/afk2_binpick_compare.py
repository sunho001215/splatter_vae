"""Compare the fresh bin-picking retraining with the original future_k2 run at matched steps (2026-10-21).

    python -I afk2_binpick_compare.py [--new afk2rerun-seed0-bin-picking-200k] [--json OUT]

1. Training logs (every 50 steps): the anomaly signature of the original's 12.0-12.9k event (visibility loss > 0.1,
   encoder gradient share < 0.1 after clipping, InfoNCE > 2, positive/negative cosine both > 0.95) in both runs,
   with the rerun's status over the problematic interval 10k-15k.
2. Validation evaluations (every 5k, fixed samples): reconstruction (PSNR, PSNR moving, SSIM, depth AbsRel, held-out
   PSNR), motion (relative EPE 0-2 at k2 = 2 and 6), representation (retrieval top-1 train/held-out, effective rank,
   probe R^2 where logged), side by side at each step both runs have reached.
"""
import argparse
import json
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d/runs/pretrain')
ORIGINAL = 'afk2-seed0-bin-picking-200k'
EVAL_KEYS = [
    ('psnr', 'metric/psnr@f2'), ('psnr_moving', 'metric/psnr_moving@f2'), ('ssim', 'metric/ssim@f2'),
    ('depth_absrel', 'metric/depth_absrel@f2'), ('psnr_heldout', 'metric/psnr_heldout@f2'),
    ('rel_epe02_k2', 'metric/rel_epe_02@f2'), ('rel_epe02_k6', 'metric/rel_epe_02@f6'),
    ('retrieval_top1_train', 'metric/retrieval_top1_train@f2'),
    ('retrieval_top1_heldout', 'metric/retrieval_top1_heldout@f2'), ('effective_rank', 'metric/effective_rank@f2'),
]


def train_rows(run):
    path = ROOT / run / 'metrics.jsonl'
    if not path.is_file():
        return []
    return [r for r in map(json.loads, path.open()) if 'loss/total' in r]


def anomalies(rows):
    """Logs after warm-up (step >= 2000) where the visibility surge coincides with encoder starvation or an InfoNCE
    collapse: the original's 12.0-12.9k signature (isolated InfoNCE or gradient-share outliers alone are routine)."""
    out = []
    for r in rows:
        if r['step'] < 2000:
            continue
        flags = []
        if r.get('loss/visibility', 0) > 0.1:
            flags.append('visibility')
        if r.get('train/grad_norm_encoder', 1) < 0.1:
            flags.append('encoder_starved')
        if r.get('loss/infonce', 0) > 2.0:
            flags.append('infonce_chance')
        if r.get('metric/positive_cosine', 0) > 0.95 and r.get('metric/negative_cosine', 0) > 0.95:
            flags.append('embedding_collapse')
        if 'visibility' in flags and len(flags) > 1:
            out.append((r['step'], flags))
    return out


def evals(run):
    out = {}
    for d in (ROOT / run / 'eval').glob('step_*'):
        f = d / 'summary.json'
        if f.is_file():
            out[int(d.name.split('_')[1])] = json.loads(f.read_text())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', default='afk2rerun-seed0-bin-picking-200k')
    ap.add_argument('--json')
    args = ap.parse_args()
    a, b = train_rows(ORIGINAL), train_rows(args.new)
    last = b[-1]['step'] if b else 0
    an_a, an_b = anomalies(a), anomalies(b)
    window = [s for s, _ in an_b if 10000 <= s <= 15000]
    report = {'new_run': args.new, 'new_last_step': last,
              'original_anomalies': [s for s, _ in an_a], 'new_anomalies': an_b,
              'interval_10k_15k': ('not reached' if last < 10000 else ('in progress' if last < 15000 else 'passed'))
              + f', anomalous logs: {len(window)}'}
    ea, eb = evals(ORIGINAL), evals(args.new)
    report['evals'] = {s: {k: [ea[s].get(key), eb[s].get(key)] for k, key in EVAL_KEYS}
                       for s in sorted(set(ea) & set(eb))}
    print(f"{args.new}: last step {last}; 10k-15k: {report['interval_10k_15k']}; "
          f"anomalous logs original {len(an_a)} (steps {an_a[0][0] if an_a else '-'}-{an_a[-1][0] if an_a else '-'}), new {len(an_b)}")
    if report['evals']:
        print('step'.ljust(7) + ''.join(k[:13].rjust(28) for k, _ in EVAL_KEYS[:6]))
        for s, row in report['evals'].items():
            print(str(s).ljust(7) + ''.join(f"{row[k][0] or float('nan'):13.4f} {row[k][1] or float('nan'):13.4f} "
                                            for k, _ in EVAL_KEYS[:6]))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
