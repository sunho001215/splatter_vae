"""Sanity panels for a DynaMo pretraining run (static PNGs, drawn with PIL).

- ``loss_curves.png``: training (step means) and validation (epoch means) dynamics and covariance losses, and the
  official per-epoch learning rates (encoder/projector scheduled, forward model constant).
- ``nn_retrieval.png``: validation query frames and their nearest training frames (cosine similarity of the frozen
  512-d feature, same camera) for the trained encoder and the same encoder at its random initialisation, with the
  retrieved frames' state errors; ``retrieval.json`` has the statistics over many queries.
- ``pca_episode.png``: the first two principal components of the features along validation episodes, coloured by
  time, trained vs random initialisation.

    CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/dynamo_panels.py --run runs/pretrain/<name> \
        --out docs/figures/baselines/dynamo
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus  # noqa: E402

guard_gpus()

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from s4d.baselines.data import split_episodes  # noqa: E402
from s4d.baselines.dynamo.model import load_export  # noqa: E402
from s4d.config import load_config  # noqa: E402
from s4d.gpu_guard import validate_runtime_path  # noqa: E402

# Validated categorical slots 1-2 (blue, orange), ink and chrome of the reference light palette.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
        "#184f95", "#104281", "#0d366b"]  # sequential blue 150 -> 700
HAND, HAMMER = slice(0, 3), slice(4, 7)  # Meta-World observation: hand xyz, object (hammer) xyz


def font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def nice_ticks(low: float, high: float, log: bool) -> list[float]:
    if log:
        decades = range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
        return [m * 10.0**e for e in decades for m in (1, 2, 5)]
    base = 10 ** (math.floor(math.log10(max(high - low, 1e-12))) - 1)
    step = next(base * m for m in (1, 2, 5, 10, 20, 50, 100) if (high - low) / (base * m) <= 6)
    return [v for v in np.arange(math.floor(low / step) * step, high + step * 1.01, step) if low <= v <= high]


def line_panel(draw, box, series, title, x_max, log=False, fmt="{:.3g}"):
    """series: [(name, colour, xs, ys, markers, end_label)] on one shared y scale. Title, then the legend on its own
    line; end labels only where asked (selective labelling)."""
    x0, y0, x1, y1 = box
    values = [v for s in series for v in s[3] if (v > 0 or not log)]
    low, high = min(values), max(values)
    if log:
        low, high = low * 0.8, high * 1.25
    else:
        pad = 0.05 * (high - low or abs(high) or 1.0)
        low, high = min(0.0, low) if low >= 0 and low < 0.2 * high else low - pad, high + pad
    ty = lambda v: math.log10(v) if log else v  # noqa: E731
    sx = lambda x: x0 + (x1 - x0) * x / x_max  # noqa: E731
    sy = lambda v: y1 - (y1 - y0) * (ty(v) - ty(low)) / (ty(high) - ty(low))  # noqa: E731
    small, mid = font(13), font(14)
    draw.text((x0, y0 - 58), title, fill=INK, font=font(17))
    legend_x = x0
    for name, colour, *_ in series:
        draw.line([(legend_x, y0 - 22), (legend_x + 18, y0 - 22)], fill=colour, width=3)
        draw.text((legend_x + 24, y0 - 31), name, fill=INK2, font=mid)
        legend_x += draw.textlength(name, font=mid) + 48
    for tick in nice_ticks(low, high, log):
        if low <= tick <= high:
            draw.line([(x0, sy(tick)), (x1, sy(tick))], fill=GRID, width=1)
            label = (f"{tick:.0e}" if tick < 1e-3 else f"{tick:g}") if log else fmt.format(tick)
            draw.text((x0 - 8 - draw.textlength(label, font=small), sy(tick) - 8), label, fill=MUTED, font=small)
    for tick in nice_ticks(0, x_max, False):
        draw.line([(sx(tick), y1), (sx(tick), y1 + 4)], fill=AXIS, width=1)
        label = f"{int(tick)}"
        draw.text((sx(tick) - draw.textlength(label, font=small) / 2, y1 + 6), label, fill=MUTED, font=small)
    draw.line([(x0, y1), (x1, y1)], fill=AXIS, width=1)
    for _, colour, xs, ys, markers, end_label in series:
        points = [(sx(x), sy(v)) for x, v in zip(xs, ys) if v > 0 or not log]
        if len(points) > 1:
            draw.line(points, fill=colour, width=2, joint="curve")
        if markers:
            for px, py in points:
                draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=SURFACE)
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=colour)
        if end_label:
            draw.text((points[-1][0] + 9, points[-1][1] - 8), fmt.format(ys[-1]), fill=INK2, font=small)


def loss_figure(rows: list[dict], out: Path) -> dict:
    steps = [r for r in rows if "train/dynamics_loss_total" in r]
    epochs = [r for r in rows if "epoch" in r and "ssl_eval/dynamics_loss_total" in r]
    x_max = max(r["step"] for r in rows)
    width, height = 1100, 1120
    image = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(image)
    box = lambda i: (110, 100 + i * 345, width - 110, 340 + i * 345)  # noqa: E731
    xs, ex = [r["step"] for r in steps], [r["step"] for r in epochs]
    window = steps[1]["step"] - steps[0]["step"] if len(steps) > 1 else 1
    train, val = f"train ({window}-step mean)", "validation (epoch mean, all validation windows)"
    line_panel(draw, box(0), [
        (train, BLUE, xs, [r["train/dynamics_loss_total"] for r in steps], False, False),
        (val, ORANGE, ex, [r["ssl_eval/dynamics_loss_total"] for r in epochs], True, True),
    ], "Dynamics loss  1 - cos(f(s_t, z_t+1), sg(s_t+1))  (log scale)", x_max, log=True, fmt="{:.4f}")
    line_panel(draw, box(1), [
        (train, BLUE, xs, [r["train/covariance_loss"] for r in steps], False, False),
        (val, ORANGE, ex, [r["ssl_eval/covariance_loss"] for r in epochs], True, True),
    ], "Covariance regulariser  0.04 x mean off-diagonal cov(s)^2  (log scale)", x_max, log=True, fmt="{:.1e}")
    line_panel(draw, box(2), [
        ("forward model (constant)", ORANGE, xs, [r["train/lr_forward_dynamics"] for r in steps], False, False),
        ("encoder + projector (set per epoch)", BLUE, xs, [r["train/lr_encoder"] for r in steps], False, False),
    ], "Learning rate (official: 5 warmup epochs from 0, then cosine over 40)", x_max, fmt="{:.0e}")
    draw.text((width / 2 - 40, height - 36), "training step", fill=INK2, font=font(15))
    image.save(out, optimize=True)
    return {"val_dynamics_loss_by_epoch": [r["ssl_eval/dynamics_loss_total"] for r in epochs]}


@torch.no_grad()
def embed(encoder, frames: np.ndarray, device) -> torch.Tensor:
    out = []
    for i in range(0, len(frames), 256):
        x = torch.from_numpy(frames[i : i + 256]).to(device).permute(0, 3, 1, 2).float() / 255.0
        out.append(torch.nn.functional.normalize(encoder(x), dim=-1).cpu())
    return torch.cat(out)


def load_frames(path: str, episodes: list[str], cameras: list[int], spacing: int):
    frames, meta = [], []
    with h5py.File(path, "r") as f:
        for ep in episodes:
            rgb, obs = f["episodes"][ep]["rgb"], f["episodes"][ep]["obs"][:]
            ts = list(range(0, int(f["episodes"][ep].attrs["length"]), spacing))
            for cam in cameras:
                frames.append(rgb[ts, cam])
                meta += [(ep, cam, t, obs[t]) for t in ts]
    return np.concatenate(frames), meta


def errors(found: np.ndarray, wanted: np.ndarray) -> dict:
    """Median hand / hammer position error (cm) between retrieved and query states."""
    err = lambda part: float(np.median(np.linalg.norm(found[:, part] - wanted[:, part], axis=1)) * 100)  # noqa: E731
    return {"hand_error_cm_median": err(HAND), "hammer_error_cm_median": err(HAMMER)}


def label(draw, xy, text, size=12):
    draw.text(xy, text, fill=INK2, font=font(size))


def retrieval_figure(cfg, encoders, device, out: Path, stats_out: Path) -> dict:
    ds = cfg["dataset"]
    manifest = cfg["data_interface"]["split_manifest"]
    pool_eps = split_episodes(manifest, "train")[: ds.get("max_episodes") or None][:40]
    val_eps = split_episodes(manifest, "validation")
    cams = list(range(6))
    pool, pool_meta = load_frames(ds["hdf5_path"], pool_eps, cams, 2)
    queries, query_meta = load_frames(ds["hdf5_path"], val_eps, cams, 6)
    pool_cam = torch.tensor([m[1] for m in pool_meta])
    pool_state = np.stack([m[3] for m in pool_meta])
    query_state = np.stack([m[3] for m in query_meta])
    stats, top = {}, {}
    rng = np.random.default_rng(0)
    for name, encoder in encoders.items():
        P, Q = embed(encoder, pool, device), embed(encoder, queries, device)
        sim = Q @ P.T
        sim[torch.tensor([m[1] for m in query_meta])[:, None] != pool_cam[None, :]] = -2.0  # same camera only
        top[name] = sim.topk(3, dim=1).indices.numpy()
        nn = top[name][:, 0]
        stats[name] = errors(pool_state[nn], query_state)
    same_cam = [np.flatnonzero(pool_cam.numpy() == m[1]) for m in query_meta]
    chance = np.array([rng.choice(idx) for idx in same_cam])
    stats["random frame (chance)"] = errors(pool_state[chance], query_state)
    summary = {
        "queries": len(queries), "pool_frames": len(pool), "pool_episodes": len(pool_eps),
        "validation_episodes": len(val_eps), "metric": "median state error of the top-1 same-camera neighbour", **stats,
    }
    stats_out.write_text(json.dumps(summary, indent=2))
    picks = [int(i) for i in np.linspace(0, len(queries) - 1, 5).round()]
    tile, gap, head = 112, 6, 40
    names = list(encoders)
    width = 150 + (1 + 3 * len(names)) * (tile + gap) + gap * 6
    height = 70 + len(picks) * (tile + head) + 60
    image = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), "Nearest training frames (same camera, cosine similarity of the frozen 512-d feature)",
              fill=INK, font=font(17))
    x_cols = [150]
    for _ in names:
        x_cols.append(x_cols[-1] + (tile + gap) * (1 if len(x_cols) == 1 else 3) + gap * 3)
    label(draw, (150, 44), "validation query", 13)
    for j, name in enumerate(names):
        label(draw, (x_cols[j + 1], 44), f"{name}: top-3 (hand / hammer error, cm)", 13)
    for row, q in enumerate(picks):
        y = 70 + row * (tile + head)
        ep, cam, t, state = query_meta[q]
        label(draw, (16, y + 40), f"{ep} cam{cam}\nt={t}", 13)
        image.paste(Image.fromarray(queries[q]).resize((tile, tile)), (150, y))
        for j, name in enumerate(names):
            for k, idx in enumerate(top[name][q]):
                x = x_cols[j + 1] + k * (tile + gap)
                image.paste(Image.fromarray(pool[idx]).resize((tile, tile)), (x, y))
                hand = np.linalg.norm(pool_state[idx, HAND] - state[HAND]) * 100
                hammer = np.linalg.norm(pool_state[idx, HAMMER] - state[HAMMER]) * 100
                label(draw, (x, y + tile + 3), f"{hand:.1f} / {hammer:.1f}", 12)
    label(draw, (16, height - 48), f"Median top-1 state error over {len(queries)} validation queries:", 13)
    foot = "    ".join(f"{n}: hand {v['hand_error_cm_median']:.1f} cm, hammer {v['hammer_error_cm_median']:.1f} cm"
                       for n, v in stats.items())
    label(draw, (16, height - 28), foot, 13)
    image.quantize(256, dither=Image.Dither.NONE).save(out, optimize=True)
    return summary


def pca_figure(cfg, encoders, device, out: Path) -> None:
    ds = cfg["dataset"]
    val_eps = split_episodes(cfg["data_interface"]["split_manifest"], "validation")[:2]
    rows = [(val_eps[0], 0), (val_eps[0], 3), (val_eps[-1], 0)]  # two cameras of one episode, another episode
    cell, pad = 330, 50
    width, height = pad + len(encoders) * (cell + pad), 90 + len(rows) * (cell + 40)
    image = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), "Feature PCA along a validation episode (one camera); colour = time, light -> dark",
              fill=INK, font=font(17))
    for j, name in enumerate(encoders):
        label(draw, (pad + j * (cell + pad), 48), name, 15)
    for i, (ep, cam) in enumerate(rows):
        frames, meta = load_frames(ds["hdf5_path"], [ep], [cam], 2)
        for j, encoder in enumerate(encoders.values()):
            z = embed(encoder, frames, device).double()
            z = z - z.mean(0)
            u, s, _ = torch.linalg.svd(z, full_matrices=False)
            pcs = (u[:, :2] * s[:2]).numpy()
            var = (s[:2] ** 2 / (s**2).sum()).numpy()
            x0, y0 = pad + j * (cell + pad), 80 + i * (cell + 40)
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), outline=GRID, width=1)
            lo, hi = pcs.min(0), pcs.max(0)
            span = np.maximum(hi - lo, 1e-9)
            unit = (pcs - lo) / span
            pts = [(x0 + 12 + (cell - 24) * u, y0 + cell - 12 - (cell - 24) * v) for u, v in unit]
            for k in range(len(pts) - 1):
                colour = RAMP[min(len(RAMP) - 1, int(len(RAMP) * k / len(pts)))]
                draw.line([pts[k], pts[k + 1]], fill=colour, width=2)
            for k in (0, len(pts) - 1):
                px, py = pts[k]
                colour = RAMP[0] if k == 0 else RAMP[-1]
                draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=SURFACE)
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=colour)
                label(draw, (px + 7, py - 7), "start" if k == 0 else "end", 12)
            label(draw, (x0 + 4, y0 + cell + 4), f"{ep} cam{cam}, {len(pts)} frames; PC1 {var[0]:.0%}, PC2 {var[1]:.0%}", 12)
    image.save(out, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=str(REPO / "docs/figures/baselines/dynamo"))
    args = ap.parse_args()
    run = Path(args.run)
    out = validate_runtime_path(Path(args.out), repository=REPO)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config([run / "config.yaml"])
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"run": str(run), "losses": loss_figure(rows, out / "loss_curves.png")}
    encoders = {}
    for name, file in (("trained", "encoder.pt"), ("random init", "encoder_init.pt")):
        payload, encoder = load_export(run / file)
        encoders[name] = encoder.to(device)
        report[f"{file}_step"] = payload["step"]
    report["retrieval"] = retrieval_figure(cfg, encoders, device, out / "nn_retrieval.png", out / "retrieval.json")
    pca_figure(cfg, encoders, device, out / "pca_episode.png")
    (out / "panels.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
