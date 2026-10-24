"""Learning-curve figures shared by the report tools: dependency-free SVG and a PIL-rasterised PNG of the same layout.

panels: list of (label, {arm: (seeds x 100) success array}); arms are drawn as seed mean with a min-max band.
"""
import numpy as np

STEPS = list(range(10000, 1000001, 10000))

COLORS = {'full': '#1f77b4', 'crop': '#d62728', 'shift': '#2ca02c', 'cnn': '#7f7f7f', 'cnnnoshift': '#ff7f0e',
          'ours': '#1f77b4', 'sincro': '#9467bd', 'reviwo': '#8c564b', 'dynamo': '#e377c2',
          'mvp3d-multiview': '#bcbd22', 'mvp3d-single': '#17becf', 'snerl': '#000000',
          'vo_ours': '#1f77b4', 'p_ours': '#9ecae1', 'vo_cnn': '#d62728', 'p_cnn': '#fcae91'}


def svg_curves(path, title, panels, arms):
    """panels: list of (label, {arm: (seeds x steps) array}); one row per task, columns train/heldout/traj."""
    w, h, pad, cols = 230, 150, 34, 3
    rows = (len(panels) + cols - 1) // cols
    W, H = cols * (w + pad) + pad, rows * (h + pad + 14) + 60
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="10">',
           f'<rect width="{W}" height="{H}" fill="white"/>', f'<text x="{pad}" y="18" font-size="13">{title}</text>']
    for i, arm in enumerate(arms):  # legend rows of seven (unchanged layout up to seven arms)
        x, dy = pad + (i % 7) * 110, (i // 7) * 13
        out.append(f'<rect x="{x}" y="{28 + dy}" width="14" height="4" fill="{COLORS[arm]}"/>'
                   f'<text x="{x + 18}" y="{33 + dy}">{arm}</text>')
    for n, (label, series) in enumerate(panels):
        x0, y0 = pad + (n % cols) * (w + pad), 60 + (n // cols) * (h + pad + 14)
        out.append(f'<text x="{x0}" y="{y0 - 3}">{label}</text>')
        out.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#999"/>')
        for g in (0.25, 0.5, 0.75):
            out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y0 + h * (1 - g):.1f}" y2="{y0 + h * (1 - g):.1f}" '
                       f'stroke="#eee"/>')
        out.append(f'<text x="{x0 - 4}" y="{y0 + 4}" text-anchor="end">1</text>'
                   f'<text x="{x0 - 4}" y="{y0 + h}" text-anchor="end">0</text>'
                   f'<text x="{x0 + w}" y="{y0 + h + 11}" text-anchor="end">1M</text>')
        for arm in arms:
            if arm not in series:
                continue
            x = np.asarray(series[arm])
            xs = [x0 + w * s / 1e6 for s in STEPS]
            pt = lambda v: [f'{a:.1f},{y0 + h * (1 - b):.1f}' for a, b in zip(xs, v)]  # noqa: E731
            band = pt(x.max(0)) + pt(x.min(0))[::-1]
            out.append(f'<polygon points="{" ".join(band)}" fill="{COLORS[arm]}" fill-opacity="0.15" stroke="none"/>')
            out.append(f'<polyline points="{" ".join(pt(x.mean(0)))}" fill="none" stroke="{COLORS[arm]}" '
                       f'stroke-width="1.3"/>')
    out.append('</svg>')
    path.write_text('\n'.join(out) + '\n')


def png_curves(path, title, panels, arms, k=2):
    """The same figure rasterised with PIL (2x), for viewers without SVG support."""
    from PIL import Image, ImageDraw, ImageFont
    w, h, pad, cols = 230 * k, 150 * k, 34 * k, 3
    rows = (len(panels) + cols - 1) // cols
    W, H = cols * (w + pad) + pad, rows * (h + pad + 14 * k) + 60 * k
    img = Image.new('RGBA', (W, H), 'white')
    font, big = ImageFont.load_default(size=10 * k), ImageFont.load_default(size=13 * k)
    d = ImageDraw.Draw(img)
    d.text((pad, 6 * k), title, fill='black', font=big)
    for i, arm in enumerate(arms):  # legend rows of seven, as in svg_curves
        x, dy = pad + (i % 7) * 110 * k, (i // 7) * 13 * k
        d.rectangle([x, 28 * k + dy, x + 14 * k, 32 * k + dy], fill=COLORS[arm])
        d.text((x + 18 * k, 24 * k + dy), arm, fill='black', font=font)
    for n, (label, series) in enumerate(panels):
        x0, y0 = pad + (n % cols) * (w + pad), 60 * k + (n // cols) * (h + pad + 14 * k)
        d.text((x0, y0 - 13 * k), label, fill='black', font=font)
        for g in (0.25, 0.5, 0.75):
            d.line([x0, y0 + h * (1 - g), x0 + w, y0 + h * (1 - g)], fill='#eeeeee', width=k)
        d.rectangle([x0, y0, x0 + w, y0 + h], outline='#999999', width=k)
        d.text((x0 - 4 * k, y0), '1', fill='black', font=font, anchor='ra')
        d.text((x0 - 4 * k, y0 + h), '0', fill='black', font=font, anchor='rs')
        d.text((x0 + w, y0 + h + 2 * k), '1M', fill='black', font=font, anchor='ra')
        for arm in arms:
            if arm not in series:
                continue
            x = np.asarray(series[arm])
            xs = [x0 + w * s / 1e6 for s in STEPS]
            pt = lambda v: [(a, y0 + h * (1 - b)) for a, b in zip(xs, v)]  # noqa: E731
            layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
            rgb = tuple(int(COLORS[arm][i:i + 2], 16) for i in (1, 3, 5))
            ImageDraw.Draw(layer).polygon(pt(x.max(0)) + pt(x.min(0))[::-1], fill=(*rgb, 40))
            img = Image.alpha_composite(img, layer)
            d = ImageDraw.Draw(img)
            d.line(pt(x.mean(0)), fill=COLORS[arm], width=k + 1)
    img.convert('RGB').save(path)
