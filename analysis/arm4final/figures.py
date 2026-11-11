"""arm4 final: figures for the results artifact, drawn ONLY from the frozen analysis outputs (rl.json,
render_metrics.json), the supplementary JSON and the registered render arrays. No statistic is computed here except
plotting transforms (8-task means are taken from the JSON when present).

    report-venv/bin/python -I -B figures.py <arm4-dir with rl.json + render_metrics.json> <supplementary.json> <out-dir>

Representative renders: a fixed, non-selective rule - per task, the registered validation sample whose ORIGINAL
training-view PSNR is the median over the 40 samples; shown for its first novel training camera and its first held-out
camera, with the same sample and cameras for every model.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TASKS = ('bin-picking', 'coffee-push', 'door-open', 'hammer', 'peg-insert-side', 'peg-unplug-side', 'pick-place',
         'stick-push')
# fixed names / colours (colour-blind-safe; references grey-blue family, arms distinct hues)
MODELS = {
    'original': ('Original', '#0B6E99', '-'),
    'cst50': ('cst50 (CSW4 + Chamfer-lean, 50 % tube)', '#5B6570', '--'),
    'chl': ('chl (Chamfer-lean)', '#A0A8B0', ':'),
    'A': ('A ocsw4 · Original + CSW4 encoder', '#C0392B', '-'),
    'B': ('B ochf · Original + Chamfer', '#E59E00', '-'),
    'C': ('C c50dg · cst50 + depth gradient', '#2E8540', '-'),
    'D': ('D c50vis · cst50 + visibility', '#7B4FA6', '-'),
}
RENDER_STEM = {'original': 'orig_200k', 'cst50': 'cst50_200k', 'A': 'ocsw4_200k', 'B': 'ochf_200k',
               'C': 'c50dg_200k', 'D': 'c50vis_200k'}
INK, MUTED, GRID = '#1E2329', '#5B6570', '#E3E7EB'
COL2 = 7.0


def setup():
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID,
                         'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 200, 'font.family': 'DejaVu Sans'})


def save(fig, out, stem):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(out / f'{stem}.{ext}', bbox_inches='tight')
    plt.close(fig)
    svg = out / f'{stem}.svg'
    txt = svg.read_text()
    if txt.startswith('<?xml'):
        txt = txt[txt.index('<svg'):]  # the platform refuses SVG with an XML/DOCTYPE header
    svg.write_text(txt)


def curves_figure(rl, out):
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3), sharey=False)
    for ax, (suite, title) in zip(axes, (('train', 'Training cameras'), ('heldout', 'Held-out cameras'),
                                         ('traj', 'Moving cameras'))):
        for m, (label, col, ls) in MODELS.items():
            c = rl['curves'].get(m)
            if not c or f'{suite}/success' not in c.get('aggregate', {}):
                continue
            steps = np.asarray(c['steps']) / 1e6
            agg = c['aggregate'][f'{suite}/success']
            ax.plot(steps, 100 * np.asarray(agg['mean']), color=col, ls=ls, lw=1.2,
                    label=f"{label} ({c['scope']})")
            band = agg.get('pointwise_95_whole_curve_seed_bootstrap')
            if band:
                lo, hi = band
                ax.fill_between(steps, 100 * np.asarray(lo), 100 * np.asarray(hi), color=col, alpha=0.10, lw=0)
        ax.set_title(title)
        ax.set_xlabel('agent steps (M)')
    axes[0].set_ylabel('success (%), task mean')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.5, -0.28), frameon=False, ncol=3)
    save(fig, out, 'arm4_curves')


def forest_figure(rl, out):
    rows = [('A − Original', rl['replacement_decision_family']['A_vs_original']),
            ('B − Original', rl['replacement_decision_family']['B_vs_original']),
            ('C − Original', rl['replacement_decision_family']['C_vs_original']),
            ('D − Original', rl['replacement_decision_family']['D_vs_original']),
            ('C − cst50', rl['main']['C_vs_cst50']), ('D − cst50', rl['main']['D_vs_cst50']),
            ('cst50 − Original', rl['context']['cst50_vs_original']),
            ('chl − Original', rl['context']['chl_vs_original'])]
    fig, ax = plt.subplots(figsize=(COL2 * 0.62, 2.6))
    for i, (name, v) in enumerate(rows):
        p = v.get('primary')
        if not p:
            continue
        y = len(rows) - 1 - i
        d = 100 * p['difference']
        b95, b98 = p['bootstrap']['0.95'], p['bootstrap']['0.9875']
        w98 = p['welch']['0.9875']['interval']
        col = INK if v['status'] == 'final' else MUTED
        ax.plot([100 * b98[0], 100 * b98[1]], [y, y], color=col, lw=1.0)
        ax.plot([100 * b95[0], 100 * b95[1]], [y, y], color=col, lw=3.0, solid_capstyle='butt')
        if w98:
            ax.plot([100 * w98[0], 100 * w98[1]], [y - 0.22, y - 0.22], color=col, lw=0.8, ls=':')
        ax.plot(d, y, 'o', color='white', mec=col, ms=5)
        lab = name + ('' if v['status'] == 'final' else f" (interim, k={p['k']})")
        ax.text(1.02, y, f'{d:+.1f}', transform=ax.get_yaxis_transform(), va='center', fontsize=7, color=col)
        rows[i] = (lab, v)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(-5, color='#C0392B', lw=0.8, ls='--')
    ax.set_xlabel('Δ training success AUC (pp)\nthick 95 %, thin 98.75 % bootstrap; dotted 98.75 % Welch; red −5 pp')
    save(fig, out, 'arm4_forest')


def pertask_figure(rl, out):
    fig, axes = plt.subplots(2, 4, figsize=(COL2, 3.6), sharey=False)
    order = ['original', 'cst50', 'chl', 'A', 'B', 'C', 'D']
    for ax, t in zip(axes.flat, TASKS):
        for j, m in enumerate(order):
            c = rl['curves'].get(m)
            if not c or t not in c.get('per_task', {}):
                continue
            vals = [100 * float(np.mean(s)) for s in c['per_task'][t]['train/success']['seed_curves']]
            ax.plot([j] * len(vals), vals, 'o', color=MODELS[m][1], ms=3, alpha=0.8)
            ax.plot([j - 0.3, j + 0.3], [np.mean(vals)] * 2, color=MODELS[m][1], lw=1.5)
        ax.set_title(t)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(['Orig', 'cst50', 'chl', 'A', 'B', 'C', 'D'], rotation=90)
    axes[0, 0].set_ylabel('training success AUC (%)')
    axes[1, 0].set_ylabel('training success AUC (%)')
    save(fig, out, 'arm4_pertask')


def montage(render_dir, ref_dir, render_metrics, out):
    """Fixed-rule representative renders (see module doc)."""
    from PIL import Image
    tiles = []
    for t in TASKS:
        idx = json.loads((ref_dir / f'{t}__index.json').read_text())
        gt = np.load(ref_dir / f'{t}__gt.npz')['images']
        rows = idx['rows']
        cats = np.array([r['category'] for r in rows])
        samples = np.array([r['sample'] for r in rows])
        orig = np.load(render_dir / f"{t}__{RENDER_STEM['original']}.npz")['images']
        tv = np.isin(cats, ('input', 'inputs_multiview', 'novel_train'))
        psnr = []
        for s in sorted(set(samples.tolist())):
            sel = tv & (samples == s)
            mse = np.mean((orig[sel].astype(float) - gt[sel].astype(float)) ** 2) / 255 ** 2
            psnr.append((10 * np.log10(1 / mse), s))
        psnr.sort()
        s_med = psnr[len(psnr) // 2][1]
        pick = []
        for cat in ('novel_train', 'heldout'):
            ii = [i for i, r in enumerate(rows) if r['sample'] == s_med and r['category'] == cat]
            if ii:
                pick.append((cat, ii[0]))
        for cat, i in pick:
            row = [gt[i]]
            for m in RENDER_STEM:
                p = render_dir / f'{t}__{RENDER_STEM[m]}.npz'
                row.append(np.load(p)['images'][i] if p.is_file() else np.full_like(gt[i], 255))
            tiles.append((t, cat, rows[i]['target_name'], np.concatenate(row, axis=1)))
    sheet = np.concatenate([x[3] for x in tiles], axis=0)
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 12)
    except OSError:
        font = ImageFont.load_default()
    head, left, tile = 22, 150, gt.shape[1]
    canvas = Image.new('RGB', (left + sheet.shape[1], head + sheet.shape[0]), 'white')
    canvas.paste(Image.fromarray(sheet), (left, head))
    d = ImageDraw.Draw(canvas)
    cols = ['GT', 'Original', 'cst50', 'A ocsw4', 'B ochf', 'C c50dg', 'D c50vis']
    for j, c in enumerate(cols):
        d.text((left + j * tile + 6, 4), c, fill=(30, 35, 41), font=font)
    for i, (t, c, n, _) in enumerate(tiles):
        d.text((6, head + i * tile + tile // 2 - 14), t, fill=(30, 35, 41), font=font)
        d.text((6, head + i * tile + tile // 2 + 2), f"{'held-out' if c == 'heldout' else 'novel train'} · {n}",
               fill=(91, 101, 112), font=font)
    canvas.save(out / 'arm4_renders.png', optimize=True)
    meta = {'columns': ['ground truth'] + [MODELS[m][0] for m in RENDER_STEM],
            'rows': [{'task': t, 'group': c, 'camera': n} for t, c, n, _ in tiles],
            'rule': 'per task: validation sample with the median Original training-view PSNR; first novel training '
                    'camera and first held-out camera of that sample; identical sample/camera for every model'}
    (out / 'arm4_renders.json').write_text(json.dumps(meta, indent=1))


def main():
    arm4, supp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    setup()
    rl = json.loads((arm4 / 'rl.json').read_text())
    curves_figure(rl, out)
    forest_figure(rl, out)
    pertask_figure(rl, out)
    rm = arm4 / 'render_metrics.json'
    rmet = json.loads(rm.read_text()) if rm.is_file() else None
    montage(arm4 / 'render', Path('/home/ws/ws/splatter4d/reports/results-artifact/render'), rmet, out)
    print('figures in', out)


if __name__ == '__main__':
    main()
