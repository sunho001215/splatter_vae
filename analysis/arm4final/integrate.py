"""arm4 final: insert the generated Arm4 sections into the LIVE published pages of the linked results artifact, keeping
every existing section (the hub, RL and rendering pages were last published from direct pub/index.html edits, so the
live HTML read back from the artifact service is the base, not the build sources).

    report-venv/bin/python -I -B integrate.py <live-dir> <fragments-dir> <fig-dir> <rl.json> <render_metrics.json> <site_split>

<live-dir> holds rl.html, rendering.html, hub.html (the live versions). Writes <site_split>/<page>/pub/index.html and
<site_split>/<page>/pub/img/<stem>.webp for the new figures, after backing up the previous pub/index.html into
<site_split>/backups-arm4-20261111/. Prints the files each page must publish.
"""
import json
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

RL_FIGS = ('arm4_forest', 'arm4_curves', 'arm4_pertask')
RENDER_FIGS = ('arm4_renders',)
URL = {'hub': 'https://claude.ai/artifact/ER1JqgQE3dSTwTg4Ln8T73', 'rl': 'https://claude.ai/artifact/CK19qFUzCu1kxoj77a9KoL',
       'rendering': 'https://claude.ai/artifact/ALYpGCopocHirAjEJrX5ez'}


def strip_skeleton(html):
    body = html[html.find('<title>'):]
    return re.sub(r'\s*</body></html>\s*$', '', body).rstrip() + '\n'


def webp(fig_dir, stem, img_dir):
    im = Image.open(fig_dir / f'{stem}.png').convert('RGB')
    if im.width > 1800:
        im = im.resize((1800, round(im.height * 1800 / im.width)), Image.LANCZOS)
    img_dir.mkdir(parents=True, exist_ok=True)
    im.save(img_dir / f'{stem}.webp', 'WEBP', quality=86, method=6)
    return im.width, im.height


def figure_markup(frag, dims):
    def rep(m):
        stem = m.group(1)
        w, h = dims[stem]
        return f'<img src="img/{stem}.webp" width="{w}" height="{h}" alt="{m.group(2)}" loading="lazy" decoding="async">'
    return re.sub(r'<img src="figures/([A-Za-z0-9_]+)\.png" alt="([^"]*)" loading="lazy">', rep, frag)


def add_nav(page, anchor_after, new_id, title):
    for cls in ('toc', 'mobile'):
        pat = re.compile(rf'(<nav class="{cls}"[^>]*>.*?<a href="#{anchor_after}">[^<]*</a>)', re.S)
        page, n = pat.subn(lambda m: m.group(1) + f'<a href="#{new_id}">{title}</a>', page, count=1)
        assert n == 1, (cls, anchor_after)
    return page


def insert_after_section(page, sec_id, frag):
    i = page.index(f'<section id="{sec_id}">')
    j = page.index('</section>', i) + len('</section>')
    return page[:j] + '\n' + frag + '\n' + page[j:]


def hub_box(rl, rm):
    f = lambda x, nd=2: f'{100 * x:+.{nd}f}'.replace('-', '−')  # noqa: E731
    iv = lambda v: f'[{f(v[0], 1)}, {f(v[1], 1)}]'  # noqa: E731
    a0 = rl['replacement_decision_family']['A_vs_original']
    orig_auc, orig_iqm = a0['primary']['original'], a0['iqm_train_success_auc']['original']
    rows = []
    for a, name in (('A', 'A · ocsw4 · Original + CSW4 encoder'), ('B', 'B · ochf · Original + Chamfer'),
                    ('C', 'C · c50dg · cst50 + depth gradient'), ('D', 'D · c50vis · cst50 + visibility')):
        v = rl['replacement_decision_family'][f'{a}_vs_original']
        p, g = v['primary'], v['guards']
        q = v['iqm_train_success_auc']
        im = rm['contrasts'][f'decision/{a}_vs_original']['images']
        veto = ', '.join(k for k, x in v['decision']['guard_veto'].items() if x) or 'none'
        rows.append(
            f'<tr><th>{name}</th><td class="num">{f(p["difference"])} {iv(p["bootstrap"]["0.95"])}</td>'
            f'<td class="num">{f(q["difference"], 1)} {iv(q["bootstrap_95"])}</td>'
            f'<td class="num">{f(g["moving"]["difference"], 1)} {iv(g["moving"]["bootstrap"]["0.95"])}</td>'
            f'<td class="num">{f(im["train_views/psnr"]["mean"] / 100)} dB</td>'
            f'<td class="num">{f(im["heldout/depth_err_median_full_relative_reduction"]["mean"], 1)} %</td>'
            f'<td class="num">{f(im["heldout/psnr"]["mean"] / 100)} dB</td>'
            f'<td>{v["decision"]["control_condition_98.75"]}; veto: {veto}; reconstruction: not improved</td></tr>')
    return (
        '<div class="box" id="arm4-status"><h3>Four-arm local campaign (Arm4) · complete, analysed 2026-11-11</h3>\n'
        '<p>All 304 registered rows finished without failure: 4 arms × 8 tasks × (pretraining, export, render, DrM seeds '
        '2000–2002 and their evaluators), plus the Original and cst50 reference renders. The pre-registered analysis ran '
        'unchanged on the complete data. <b>Registered decision: no replacement recommended; Original retained</b> '
        '(evidence insufficient at n = 3 RL seeds and one pretraining seed). No arm meets the control condition (98.75 % '
        'bootstrap and Welch lower bounds above −5 pp), C and D have a moving-camera veto, and no arm meets '
        '“reconstruction improved (whole scene)”. Independent Codex recomputation of the RL aggregates agrees.</p>\n'
        '<div class="table-wrap"><table><thead><tr><th>Arm vs Original</th><th>Δ training success AUC, pp [95 %]</th>'
        '<th>Δ IQM, pp [95 %]</th><th>Δ moving success AUC, pp [95 %]</th><th>Δ training-view PSNR</th>'
        '<th>held-out depth error reduction</th><th>Δ held-out PSNR</th><th>Registered reading</th></tr></thead><tbody>\n'
        + '\n'.join(rows) + '\n</tbody></table></div>\n'
        f'<p class="note">Original: training success AUC {100 * orig_auc:.1f} %, IQM {100 * orig_iqm:.1f} %. Details: '
        f'<a href="{URL["rl"]}#arm4-results">control, robustness, stability and cost</a> · '
        f'<a href="{URL["rendering"]}#arm4-reconstruction">rendering, depth and geometry</a>. '
        'Statements condition on one pretraining seed per arm and task; nonsignificance is not equivalence.</p>\n'
        '<details><summary>Registered arm definitions (resolved differences)</summary>\n')


def main():
    live, frag, figs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rl = json.loads(Path(sys.argv[4]).read_text())
    rm = json.loads(Path(sys.argv[5]).read_text())
    site = Path(sys.argv[6])
    bak = site / 'backups-arm4-20261111'
    bak.mkdir(exist_ok=True)
    out = {}

    # RL page
    page = strip_skeleton((live / 'rl.html').read_text())
    dims = {s: webp(figs, s, site / 'rl/pub/img') for s in RL_FIGS}
    sec = figure_markup((frag / 'rl_arm4.html').read_text(), dims)
    old_note = ('The registered four-arm local campaign is running (status <a href="https://claude.ai/artifact/'
                'ER1JqgQE3dSTwTg4Ln8T73#arm4-status">on the hub</a>); its results appear only after the pre-registered '
                'analysis of the completed arms.')
    assert page.count(old_note) == 1
    page = page.replace(old_note, 'The registered four-arm local campaign is complete; its pre-registered analysis is '
                                  'in <a href="#arm4-results">Arm4: encoder and loss changes</a> below.')
    page = insert_after_section(page, 'architecture-final', sec)
    page = add_nav(page, 'architecture-final', 'arm4-results', 'Arm4: encoder and loss changes against Original and cst50')
    if not (bak / 'rl-index.html').exists():
        shutil.copyfile(site / 'rl/pub/index.html', bak / 'rl-index.html')
    (site / 'rl/pub/index.html').write_text(page)
    out['rl'] = ['index.html'] + [f'img/{s}.webp' for s in RL_FIGS]

    # Rendering page
    page = strip_skeleton((live / 'rendering.html').read_text())
    dims = {s: webp(figs, s, site / 'rendering/pub/img') for s in RENDER_FIGS}
    sec = figure_markup((frag / 'rendering_arm4.html').read_text(), dims)
    page = insert_after_section(page, 'chlcmp', sec)
    page = add_nav(page, 'chlcmp', 'arm4-reconstruction', 'Arm4: rendering quality, depth and geometry')
    if not (bak / 'rendering-index.html').exists():
        shutil.copyfile(site / 'rendering/pub/index.html', bak / 'rendering-index.html')
    (site / 'rendering/pub/index.html').write_text(page)
    out['rendering'] = ['index.html'] + [f'img/{s}.webp' for s in RENDER_FIGS]

    # Hub: final status box (registration table kept, folded); overview bullet; key finding; page-card links
    page = strip_skeleton((live / 'hub.html').read_text())
    i = page.index('<div class="box" id="arm4-status">')
    j = page.index('<div class="table-wrap">', i)
    k = page.index('</table></div>', j) + len('</table></div>')
    reg_table = page[j:k]
    end = page.index('<!-- HISTORICAL CAMPAIGNS -->', k)
    old_rest = page[k:end]
    reg_p = re.search(r'<p><b>Registered analysis:</b>.*?</p>', old_rest, re.S).group(0)
    new_box = hub_box(rl, rm) + reg_table + '\n' + reg_p + '\n</details></div>\n'
    page = page[:i] + new_box + page[end:]
    old_b = ('The <a href="#arm4-status">registered four-arm campaign</a> tests encoder and loss changes against Original '
             'and cst50.')
    assert page.count(old_b) == 1
    page = page.replace(old_b, 'The <a href="#arm4-status">four-arm campaign (Arm4)</a> is complete: no encoder or loss '
                               'change replaces the original method (2026-11-11).')
    anchor = '<!-- /CSW4 FINAL -->'
    v = {a: rl['replacement_decision_family'][f'{a}_vs_original']['primary']['difference'] for a in 'ABCD'}
    red = sorted(100 * rm['contrasts'][f'decision/{a}_vs_original']['images'][
        'heldout/depth_err_median_full_relative_reduction']['mean'] for a in 'CD')
    fm = lambda x: f'{100 * x:+.2f}'.replace('-', '\u2212')  # noqa: E731
    kf = (f'<!-- ARM4 FINAL -->\n<p><b>Arm4 · FINAL:</b> training success AUC vs Original A {fm(v["A"])}, '
          f'B {fm(v["B"])}, C {fm(v["C"])}, D {fm(v["D"])} pp; no arm is a replacement candidate and the original '
          f'method is retained. Chamfer-trained models reconstruct depth far better (held-out depth error '
          f'\u2212{red[0]:.0f} to \u2212{red[1]:.0f} % for C and D) without better control. <a href="{URL["rl"]}#arm4-results">Arm4 analysis</a>.</p>\n'
          f'<!-- /ARM4 FINAL -->')
    page = page.replace(anchor, anchor + '\n' + kf, 1)
    for pg, sid, title in (('CK19qFUzCu1kxoj77a9KoL', 'architecture-final', 'Arm4: encoder and loss changes against Original and cst50'),
                           ('ALYpGCopocHirAjEJrX5ez', 'chlcmp', 'Arm4: rendering quality, depth and geometry')):
        pat = re.compile(rf'(<li><a href="https://claude.ai/artifact/{pg}#{sid}">[^<]*</a></li>)')
        new_id = 'arm4-results' if sid == 'architecture-final' else 'arm4-reconstruction'
        page, n = pat.subn(lambda m: m.group(1) + f'<li><a href="https://claude.ai/artifact/{pg}#{new_id}">{title}</a></li>',
                           page, count=1)
        assert n == 1
    if not (bak / 'hub-index.html').exists():
        shutil.copyfile(site / 'hub/pub/index.html', bak / 'hub-index.html')
    (site / 'hub/pub/index.html').write_text(page)
    out['hub'] = ['index.html']
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
