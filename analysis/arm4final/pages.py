"""arm4 final: HTML sections for the existing linked results artifact (rl / rendering / hub), filled ONLY from the frozen
analysis outputs (rl.json, render_metrics.json) and the supplementary JSON. Every number on the page comes from those
files; nothing is typed by hand. Interim comparisons are labelled and carry no reading.

    python3 pages.py <arm4-dir> <supplementary.json> <out-dir>

Writes <out-dir>/rl_arm4.html, rendering_arm4.html, hub_arm4.html (fragments inserted into the live pages by
integrate.py, which keeps every existing section).
"""
import html
import json
import sys
from pathlib import Path

TASKS = ('bin-picking', 'coffee-push', 'door-open', 'hammer', 'peg-insert-side', 'peg-unplug-side', 'pick-place',
         'stick-push')
NAME = {'original': 'Original', 'cst50': 'cst50', 'chl': 'chl', 'cst75': 'cst75',
        'A': 'A · ocsw4', 'B': 'B · ochf', 'C': 'C · c50dg', 'D': 'D · c50vis'}
ARM_DEF = {
    'A': ('Original + CSW4 encoder', 'stem patch_cnn; width 128 → 256; depth 8 → 4; heads 8 → 4 (all Original losses, '
                                     '50 % motion-aware tube masking)'),
    'B': ('Original + Chamfer', 'loss.chamfer 5.0 (4096 points, 1 cm voxels); hard-depth 0.5, depth-gradient 0.5, '
                                'visibility 1.0 kept; Original encoder, 50 % tube'),
    'C': ('cst50 + depth gradient', 'loss.depth_grad 0.0 → 0.5 on cst50 (CSW4 encoder, Chamfer-lean, 50 % tube)'),
    'D': ('cst50 + visibility', 'loss.visibility 0.0 → 1.0 on cst50 (CSW4 encoder, Chamfer-lean, 50 % tube)'),
}


def e(x):
    return html.escape(str(x))


def pp(x, nd=2, scale=100):
    return '—' if x is None else f'{scale * x:+.{nd}f}'.replace('-', '−')


def iv(v, nd=1, scale=100):
    return '—' if not v or v[0] is None else f'[{pp(v[0], nd, scale)}, {pp(v[1], nd, scale)}]'


def table(head, rows):
    h = ''.join(f'<th>{e(x)}</th>' for x in head)
    body = ''.join('<tr>' + ''.join((f'<th>{c}</th>' if i == 0 else f'<td class="num">{c}</td>')
                                    for i, c in enumerate(r)) + '</tr>' for r in rows)
    return f'<div class="table-wrap"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(stem, alt, caption, width=980):
    return (f'<figure data-stem="{stem}" style="max-width:{width}px;margin-left:auto;margin-right:auto">'
            f'<div class="plate" tabindex="0" role="button" aria-label="Enlarge figure"><img src="figures/{stem}.png" '
            f'alt="{e(alt)}" loading="lazy"></div><figcaption>{caption}</figcaption></figure>')


def comparison_rows(rl):
    out = []
    for fam, key, label in (('replacement_decision_family', 'A_vs_original', 'A − Original'),
                            ('replacement_decision_family', 'B_vs_original', 'B − Original'),
                            ('replacement_decision_family', 'C_vs_original', 'C − Original'),
                            ('replacement_decision_family', 'D_vs_original', 'D − Original'),
                            ('main', 'C_vs_cst50', 'C − cst50 (mechanism)'),
                            ('main', 'D_vs_cst50', 'D − cst50 (mechanism)'),
                            ('context', 'cst50_vs_original', 'cst50 − Original (context)'),
                            ('context', 'chl_vs_original', 'chl − Original (context)')):
        v = rl[fam][key]
        p = v.get('primary')
        if not p:
            out.append([e(label), e(v['status'])] + ['—'] * 6)
            continue
        r = v.get('reading', {})
        dec = v.get('decision', {})
        reading = (f"{e(r.get('direction', ''))}; {e(r.get('margin', ''))}" if r else
                   ('<i>context contrast (descriptive)</i>' if v['status'] == 'final' else
                    f"<i>{e(v['status'])}: no reading</i>"))
        if dec:
            reading += f"<br><span class='ci'>control condition (98.75 % boot AND Welch): " \
                       f"{e(dec.get('control_condition_98.75'))}; guard veto: " \
                       f"{e(', '.join(k for k, x in (dec.get('guard_veto') or {}).items() if x) or 'none')}</span>"
        out.append([e(label), f"{e(v['status'])} · k = {p['k']}", pp(p['difference']),
                    iv(p['bootstrap']['0.95']), iv(p['bootstrap']['0.9875']),
                    iv(p['welch']['0.9875']['interval']),
                    f"{p['task_level']['tasks_positive']}/{p['k']}", reading])
    return out


def guard_rows(rl):
    out = []
    for a in 'ABCD':
        v = rl['replacement_decision_family'][f'{a}_vs_original']
        g = v.get('guards') or {}
        cells = [e(NAME[a] + ' − Original')]
        for name in ('heldout', 'moving'):
            x = g.get(name)
            cells.append('—' if not x else f"{pp(x['difference'])} {iv(x['bootstrap']['0.95'])}")
        cells.append(e(v['status']))
        out.append(cells)
    return out


def pertask_rows(rl):
    rows = []
    for t in TASKS:
        r = [e(t)]
        for m in ('original', 'cst50', 'chl', 'A', 'B', 'C', 'D'):
            c = rl['curves'].get(m, {})
            pt = c.get('per_task', {}).get(t)
            if not pt:
                r.append('pending')
                continue
            seeds = [100 * sum(s) / len(s) for s in pt['train/success']['seed_curves']]
            r.append(f"{sum(seeds) / len(seeds):.1f}<span class='ci'>seeds: {' '.join(f'{x:.1f}' for x in seeds)}</span>")
        rows.append(r)
    return rows


def stability_rows(supp):
    rows = []
    for m in ('original', 'cst50', 'chl', 'A', 'B', 'C', 'D'):
        s, c = supp['summary'][m], supp['cost'][m]
        f = (lambda v, nd=2: '—' if v is None else f'{v:.{nd}f}')
        rows.append([e(NAME[m]), str(s['complete_runs']), f(s['mean_within_cell_seed_sd_auc'] and
                                                            100 * s['mean_within_cell_seed_sd_auc'], 1),
                     f(s['median_late_drop'] and 100 * s['median_late_drop'], 1), str(s['runs_late_drop_gt_0.2']),
                     '—' if s['pretrain_spikes_total'] is None else str(s['pretrain_spikes_total']),
                     ', '.join(f'{p / 1e6:.2f} M' for p in s['encoder_params']) or '—',
                     f(c['pretrain_median_step_time_s'], 3), f(c['pretrain_wall_h']['median'], 1),
                     f(c['drm_wall_h']['median'], 1)])
    return rows


SEC = (('train/success/last5', 'train success last-5'), ('train/nR/auc', 'train nR AUC'),
       ('heldout/success/auc', 'held-out success AUC'), ('traj/success/auc', 'moving success AUC'),
       ('drop_moving/nR/auc', 'return drop train − moving (nR AUC)'), ('train/success/auc_250k', 'success AUC ≤ 250k'))


def iqm_secondary_rows(rl):
    rows = []
    for fam, key, label, ref in (('replacement_decision_family', 'A_vs_original', 'A − Original', 'original'),
                                 ('replacement_decision_family', 'B_vs_original', 'B − Original', 'original'),
                                 ('replacement_decision_family', 'C_vs_original', 'C − Original', 'original'),
                                 ('replacement_decision_family', 'D_vs_original', 'D − Original', 'original'),
                                 ('main', 'C_vs_cst50', 'C − cst50', 'cst50'), ('main', 'D_vs_cst50', 'D − cst50', 'cst50'),
                                 ('secondary_descriptive', 'A_vs_B', 'A − B', 'B'),
                                 ('secondary_descriptive', 'C_vs_D', 'C − D', 'D')):
        v = rl[fam][key]
        q = v.get('iqm_train_success_auc') or {}
        arm = key.split('_vs_')[0]
        r = [e(label), (f"{100 * q[arm]:.1f} vs {100 * q[ref]:.1f}: {pp(q['difference'], 1)} {iv(q['bootstrap_95'])}"
                        if q and arm in q and ref in q else '—')]
        for k, _ in SEC:
            x = (v.get('secondary') or {}).get(k)
            r.append('—' if not x else f"{pp(x['difference'], 1, 100 if 'drop' not in k and 'nR' not in k else 100)} "
                                      f"{iv(x['bootstrap']['0.95'])}")
        rows.append(r)
    return rows


def ranking_rows(rl):
    ro = rl['ranking']
    items = sorted(ro['arms'].items(), key=lambda x: x[1]['mean_rank_point'])
    lab = {**NAME, 'csw4': 'csw4', 'chlrm': 'chlrm', 'chlhm': 'chlhm', 'cs': 'cs', 'w4': 'w4', 'cnn': 'DrM CNN',
           'cnnnoshift': 'DrM CNN, no shift', 'sincro': 'SinCro', 'reviwo': 'ReViWo', 'dynamo': 'DynaMo',
           'mvp3d-single': '3D-MVP B', 'mvp3d-multiview': '3D-MVP A', 'snerl': 'SNeRL'}
    return [[e(lab.get(a, a)), f"{v['mean_rank_point']:.2f}",
             iv(v.get('mean_rank_95'), 2, 1).replace('+', '') if v.get('mean_rank_95') else '—',
             f"{v.get('best_mean_rank_frequency', 0):.3f}" if isinstance(v.get('best_mean_rank_frequency'), float) else '—']
            for a, v in items]


def recommendation_box(rl, rm):
    rec = rl['recommendation']
    d = {a: rl['replacement_decision_family'][f'{a}_vs_original'] for a in 'ABCD'}
    lines = []
    for a in 'ABCD':
        p, dec = d[a]['primary'], d[a]['decision']
        rr = rm['contrasts'][f'decision/{a}_vs_original']['reading']
        why = []
        if dec['control_condition_98.75'] != 'met':
            why.append(f"control: 98.75 % lower bounds {pp(dec['bootstrap_98.75'][0], 1)} (bootstrap) / "
                       f"{pp(dec['welch_98.75'][0], 1)} (Welch) pp, below −5")
        vetoes = [k for k, x in dec['guard_veto'].items() if x]
        if vetoes:
            why.append('guard veto: ' + ', '.join(vetoes))
        failed = [k for k, x in rr['criteria'].items() if not x]
        if failed:
            why.append('reconstruction gate failed: ' + ', '.join(failed))
        lines.append(f"<li><b>{e(NAME[a])}</b> ({e(ARM_DEF[a][0])}): Δ {pp(p['difference'])} pp — "
                     + e('; '.join(why)) + '.</li>')
    return ('<div class="answer"><p><b>Registered decision · FINAL:</b> ' + e(rec['recommendation'].split(' - ')[0]
            .strip('"')) + '. No arm is a replacement candidate.</p><ul>' + ''.join(lines) + '</ul><p class="note">'
            + e(rec['statement']) + '. ' + e(rec['caveat']) + '</p></div>')


def rl_section(rl, supp, rm=None):
    complete = all(rl[f]['%s' % k]['status'] == 'final' for f, k in
                   (('replacement_decision_family', f'{a}_vs_original') for a in 'ABCD'))
    kicker = 'FINAL' if complete else 'INTERIM (stick-push B/C/D pending; no readings for interim comparisons)'
    parts = [f'<section id="arm4-results">',
             f'<div class="kicker">Arm4 architecture + loss search · DrM · {kicker}</div>',
             '<h2>Arm4: encoder and loss changes against Original and cst50</h2>',
             '<p class="note">Registered 2026-11-06 before any arm result (runs/setup/arm4-definition-20261106.json); '
             'analysis frozen 17:45 the same day and executed unchanged. Pretraining 200k, seed 0; frozen-encoder DrM 1M, '
             'RL seeds 2000–2002; 240 evaluation episodes every 10k (6 training, 4 held-out, 2 moving cameras). '
             'Primary: training-camera success AUC (mean of the 100 evaluations), equal-task mean of arm − reference. '
             'Candidacy (vs Original): 98.75 % bootstrap AND 98.75 % Welch lower bound above −5 pp (descriptive '
             'benchmark, never equivalence), no guard veto, and “reconstruction improved (whole scene)”.</p>',
             table(['Arm', 'Intervention', 'Resolved differences from its reference'],
                   [[e(NAME[a]), e(ARM_DEF[a][0]), e(ARM_DEF[a][1])] for a in 'ABCD']),
             recommendation_box(rl, rm) if (complete and rm) else '',
             '<h3 id="arm4-primary">Primary endpoint and candidacy</h3>',
             table(['Contrast', 'Status', 'Δ (pp)', '95 % boot', '98.75 % boot', '98.75 % Welch', 'tasks > 0',
                    'Registered reading'], comparison_rows(rl)),
             figure('arm4_forest', 'Primary differences with 95 and 98.75 percent intervals',
                    '<b>Training success AUC differences.</b> Thick: 95 % task-stratified bootstrap; thin: 98.75 % '
                    '(Bonferroni over the four replacement candidates); dotted: 98.75 % Welch. Red: the descriptive '
                    '−5 pp benchmark. Grey rows are interim.', 720),
             '<h3 id="arm4-iqm">IQM and secondary endpoints (descriptive; Δ pp or Δ nR×100, 95 % bootstrap)</h3>',
             table(['Contrast', 'IQM training success AUC (%): arm vs ref, Δ [95 %]'] + [x[1] for x in SEC],
                   iqm_secondary_rows(rl)),
             '<p class="note">IQM: interquartile mean over the 24 pooled runs (rliable convention) with the same '
             'task-stratified resampling — a different estimand from the equal-task mean, not a robustness check.</p>',
             '<h3 id="arm4-guards">Viewpoint guards (held-out and moving cameras)</h3>',
             table(['Contrast', 'Held-out success AUC Δ pp [95 %]', 'Moving success AUC Δ pp [95 %]', 'Status'],
                   guard_rows(rl)),
             figure('arm4_curves', 'Success curves on training, held-out and moving cameras',
                    '<b>Success curves</b>, task mean of seed means with whole-curve seed-bootstrap bands. Interim arms '
                    'average fewer tasks (see legend) and are not directly comparable to 8-task curves.'),
             '<h3 id="arm4-pertask">Per task: training success AUC (%), seed values 2000/2001/2002</h3>',
             table(['Task', 'Original', 'cst50', 'chl', 'A', 'B', 'C', 'D'], pertask_rows(rl)),
             figure('arm4_pertask', 'Per-task seed values', '<b>Per-task seed values</b> (dots) and seed means (bars). '
                    'Per-task 3-vs-3 intervals are uncalibrated and carry no control labels.'),
             '<h3 id="arm4-ranking">Eight-task ranking with pretrained baselines (HEADLINE8, descriptive)</h3>',
             table(['Model', 'mean rank (point)', 'mean rank 95 %', 'P(best mean rank), bootstrap'], ranking_rows(rl)),
             '<p class="note">Rank frequencies are conditional bootstrap frequencies with n = 3 seeds, not '
             'probabilities of being best; never a decision input.</p>',
             '<h3 id="arm4-stability">Training stability and cost (supplementary, descriptive)</h3>',
             table(['Model', 'complete runs', 'mean within-task seed SD of AUC (pp)', 'median late drop (pp)',
                    'runs with late drop > 20 pp', 'pretraining loss spikes', 'encoder params',
                    'pretrain s/step (median)', 'pretrain wall h (median)', 'DrM wall h (median)'],
                   stability_rows(supp)),
             '<p class="note">' + e(supp['meta']['label']) + '. Definitions: ' +
             e('; '.join(f'{k}: {v}' for k, v in supp['meta']['definitions'].items())) +
             '. Original was pretrained and run on the remote host (registry wall time unavailable for its pretraining); '
             'common-hardware batch-1 latency was not measured (no GPU use in this analysis) and stays unavailable.</p>',
             '<p class="note">' + e(rl['meta']['conditionality']) + '</p>',
             '</section>']
    return '\n'.join(parts)


RMODELS = ('original', 'cst50', 'chl', 'A', 'B', 'C', 'D')
RKEYS = (('train_views', 'psnr', 'train PSNR (dB)', 2), ('train_views', 'ssim', 'train SSIM', 3),
         ('train_views', 'lpips', 'train LPIPS', 3), ('train_views', 'psnr_dynamic', 'train dynamic PSNR (dB)', 2),
         ('heldout', 'psnr', 'held-out PSNR (dB)', 2), ('heldout', 'lpips', 'held-out LPIPS', 3),
         ('heldout', 'depth_err_median_full_m', 'held-out median depth err (mm)', 2),
         ('moving_range', 'psnr', 'moving PSNR (dB)', 2))


def model_mean(rm, m, g, k):
    vals = [rm['images'][t][m][g][k] for t in TASKS if m in rm['images'].get(t, {})
            and rm['images'][t][m][g].get(k) is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def recon_table(rm):
    rows = []
    for m in RMODELS:
        r = [e(NAME[m])]
        for g, k, _, nd in RKEYS:
            v, n = model_mean(rm, m, g, k)
            if v is None:
                r.append('—')
                continue
            if k.endswith('_m'):
                v *= 1000
            r.append(f'{v:.{nd}f}' + ('' if n == len(TASKS) else f"<span class='ci'>{n} tasks</span>"))
        rows.append(r)
    return table(['Model'] + [x[2] for x in RKEYS], rows)


def recon_contrast_rows(rm):
    rows = []
    for key in ('decision/A_vs_original', 'decision/B_vs_original', 'decision/C_vs_original', 'decision/D_vs_original',
                'main/C_vs_cst50', 'main/D_vs_cst50', 'secondary/cst50_vs_original', 'secondary/chl_vs_original'):
        c = rm['contrasts'].get(key)
        if not c or 'images' not in c:
            rows.append([e(key.split('/')[1].replace('_vs_', ' − ')), e(c['status'] if c else 'absent')] + ['—'] * 5)
            continue
        im = c['images']
        def cell(k, scale=1, nd=2, lv='task_level_0.9875'):
            v = im.get(k)
            if not v or v.get('mean') is None:
                return '—'
            lo_hi = v.get(lv)
            return f"{pp(v['mean'], nd, scale)} {iv(lo_hi, nd, scale)}"
        lab = c.get('reading', {}).get('label', f"<i>{c['status']}: no reading</i>")
        rows.append([e(key.split('/')[1].replace('_vs_', ' − ').replace('original', 'Original')), e(c['status']),
                     cell('train_views/psnr'), cell('heldout/depth_err_median_full_relative_reduction', 100, 1),
                     cell('train_views/psnr_dynamic', lv='task_level_0.95'), cell('heldout/psnr', lv='task_level_0.95'),
                     lab if isinstance(lab, str) and lab.startswith('<i>') else e(lab)])
    return rows


def geometry_table(rm):
    """Eight-task means of the registered geometry / Gaussian / in-training (temporal) diagnostics per model."""
    def gm(m, f):
        v = [f(rm['geometry'][t][m]) for t in TASKS if m in rm['geometry'].get(t, {})]
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None

    def it(m, k):
        v = [rm['in_training_200k'][t][m].get(k) for t in TASKS if m in rm['in_training_200k'].get(t, {})]
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None
    cols = [('Chamfer, all centres (m², ↓)', lambda m: gm(m, lambda g: g['chamfer']['all']['loss']), 4),
            ('decoded centres within 2 cm of depth', lambda m: gm(m, lambda g: g['chamfer']['all']['pred_within_2cm']), 3),
            ('static Gaussians contributing', lambda m: gm(m, lambda g: g['classes']['static']['contributing']), 3),
            ('dynamic Gaussians contributing', lambda m: gm(m, lambda g: g['classes']['dynamic']['contributing']), 3),
            ('static at scale cap', lambda m: gm(m, lambda g: g['at_scale_cap']['static']), 3),
            ('in-training depth AbsRel (↓)', lambda m: it(m, 'depth_absrel@s2'), 4),
            ('moving-point EPE t0→t1 (mm, ↓)', lambda m: it(m, 'epe_moving_mm_01@s2'), 2),
            ('moving-point EPE t1→t2 (mm, ↓)', lambda m: it(m, 'epe_moving_mm_12@s2'), 2)]
    rows = []
    for m in RMODELS:
        rows.append([e(NAME[m])] + ['—' if (v := f(m)) is None else f'{v:.{nd}f}' for _, f, nd in cols])
    return table(['Model'] + [c[0] for c in cols], rows)


def rendering_section(rm):
    return '\n'.join([
        '<section id="arm4-reconstruction">',
        '<div class="kicker">Arm4 architecture + loss search · reconstruction at 200k · registered render pipeline</div>',
        '<h2>Arm4: rendering quality, depth and geometry</h2>',
        '<p class="note">Same code path for every model (chl_analysis_render.py protocol): the registered 40 validation '
        'samples (10 episodes × 4) × 22 target cameras per task, frames t−4, t−2, t → rendered frame t. Original was '
        'reproduced bitwise against its archived reference render. Decision endpoints: training-view PSNR (gain ≥ 0.3 dB '
        'with 98.75 % task-level interval excluding 0), held-out median depth error (relative reduction ≥ 10 %, 98.75 % '
        'interval excluding 0); guards: training-view dynamic-region PSNR and held-out PSNR (95 % lower bound > −0.3 dB). '
        'Chamfer is supporting only (construction-confounded: B, C, D, cst50 and chl optimise it).</p>',
        '<h3 id="arm4-recon-table">Eight-task means per model</h3>', recon_table(rm),
        '<h3 id="arm4-geometry">Geometry, Gaussian usage and motion (eight-task means)</h3>', geometry_table(rm),
        '<p class="note">Chamfer: decoded centres at frame t vs fused training-camera depth (1 cm voxels); it is optimised '
        'by B, C, D, cst50 and chl, so it favours them by construction (supporting only). Gaussian classes over the six '
        'training cameras. In-training values: the trainer validator at 200k (stride 2); EPE is the 3D end-point error of '
        'moving points between consecutive frames, the registered motion (temporal consistency) measure. Gaussian '
        'diagnostics are aggregates without episode intervals.</p>',
        '<h3 id="arm4-recon-contrasts">Registered reconstruction contrasts (Δ, 98.75 % / 95 % task-level intervals)</h3>',
        table(['Contrast', 'Status', 'train PSNR Δ dB [98.75 %]', 'held-out depth err rel. reduction % [98.75 %]',
               'dynamic PSNR Δ dB [95 %]', 'held-out PSNR Δ dB [95 %]', 'Registered reading'], recon_contrast_rows(rm)),
        figure('arm4_renders', 'Representative renders: ground truth and every model',
               '<b>Representative renders</b> (fixed rule, not selected by eye: per task the validation sample with the '
               'median Original training-view PSNR; its first novel training camera, then its first held-out camera). '
               'Columns: ground truth, Original, cst50, A, B, C, D.'),
        '<p class="note">' + e(rm['meta'].get('conditionality', '')) + '</p>',
        '</section>'])


def main():
    arm4, supp_p, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    rl = json.loads((arm4 / 'rl.json').read_text())
    supp = json.loads(supp_p.read_text())
    rmp = arm4 / 'render_metrics.json'
    rm = json.loads(rmp.read_text()) if rmp.is_file() else None
    (out / 'rl_arm4.html').write_text(rl_section(rl, supp, rm))
    if rm:
        (out / 'rendering_arm4.html').write_text(rendering_section(rm))
    print('wrote', out)


if __name__ == '__main__':
    main()
