"""Smoke check of the DROID 3D track arrays (review item 2c); no DROID training.

Loads a few cache windows through the loader and the shared contract, renders track overlays, and runs a short M3D
overfit on the sample episode's training windows. Run with python -I and one approved GPU UUID.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from s4d.config import get, load_config  # noqa: E402
from s4d.data import DROID_CACHE_ROOT, writable_path  # noqa: E402
from s4d.data.contract import collate, validate_batch  # noqa: E402
from s4d.data.droid.dataset import DroidCacheDataset  # noqa: E402
from s4d.data.droid.inspect import cached_coordinate_overlays  # noqa: E402
from s4d.train.loop import build_model, build_optimizer, forward_losses, move_batch, seed_everything  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WINDOW = 25  # steps averaged at each end of the curve


def contract_checks(root: Path, count: int) -> dict:
    """Load ``count`` windows per split one by one and as one collated batch; both pass validate_batch."""
    report = {}
    for split in ("train", "validation"):
        dataset = DroidCacheDataset(root, split=split)
        samples = [dataset[i] for i in range(min(count, len(dataset)))]
        batch = collate(samples)
        sizes = validate_batch(batch)
        weight, disp = batch["track_weight"], batch["track_disp"]
        report[split] = {
            "windows": len(dataset),
            "checked": len(samples),
            "sizes": sizes,
            "track_capacity": int(weight.shape[-1]),
            "track_count": batch["track_count"].tolist(),
            "supervised_fraction": float(weight.sum() / (batch["track_count"].sum() * 3)),
            "moving_5mm_supervised": int(((disp.norm(dim=-1) > 0.005) & (weight > 0)).sum()),
            "gripper_points": int(batch["gripper_points"].shape[2]),
            "gripper_visible_fraction": float(batch["gripper_weight"].mean()),
            "pixel_grid_motion_fields": [k for k in ("motion3d", "motion_weight") if k in batch],
        }
        dataset.close()
    return report


def curve_png(path: Path, steps: list[int], series: dict[str, list[float]]) -> None:
    width, height, pad = 900, 420, 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    values = [v for vs in series.values() for v in vs]
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    x = lambda s: pad + (width - 2 * pad) * s / max(steps[-1], 1)  # noqa: E731
    y = lambda v: height - pad - (height - 2 * pad) * (v - low) / span  # noqa: E731
    draw.rectangle((pad, pad, width - pad, height - pad), outline="#999999")
    for (name, vs), color in zip(series.items(), ("#2a78d6", "#eb6834")):
        draw.line([(x(s), y(v)) for s, v in zip(steps, vs)], fill=color, width=2)
        draw.text((pad + 8, pad + 8 + 16 * list(series).index(name)), name, fill=color)
    draw.text((pad, height - pad + 10), f"step 0 .. {steps[-1]}", fill="#252a32")
    draw.text((8, pad - 20), f"{high:.4g}", fill="#252a32")
    draw.text((8, height - pad), f"{low:.4g}", fill="#252a32")
    image.save(path)


def overfit(cfg: dict, root: Path, steps: int, device: torch.device) -> dict:
    seed_everything(int(get(cfg, "train.seed", 0)))
    model = build_model(cfg).to(device)
    stats_path = get(cfg, "model.anchor_stats")
    if stats_path:
        model.decoder.set_anchor_statistics(json.loads(Path(stats_path).read_text()))
    optimizer, scheduler = build_optimizer(model, cfg)
    dataset = DroidCacheDataset(root, split="train")
    loader = DataLoader(
        dataset,
        batch_size=int(get(cfg, "train.batch_size", 2)),
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(0),
    )
    # a fixed probe batch and fixed point sampling, so the probe curve shows the fit rather than sampling noise
    probe = move_batch(collate([dataset[0], dataset[len(dataset) // 2]]), device)
    rows, probes = [], []
    step, start = 0, time.time()
    clip = float(get(cfg, "train.grad_clip", 1.0))
    while step < steps:
        for raw in loader:
            if step % WINDOW == 0 or step == steps - 1:
                model.eval()
                with torch.no_grad():
                    torch.manual_seed(1234)
                    source = torch.zeros(2, dtype=torch.long, device=device)
                    out = forward_losses(model, probe, cfg, step, source=source, mask_ratio=0.0)
                losses = out["losses"]
                probes.append({"step": step, "motion": float(losses["motion"]), "total": float(losses["total"])})
                model.train()
            batch = move_batch(raw, device)
            out = forward_losses(model, batch, cfg, step)
            if not torch.isfinite(out["total"]):
                raise FloatingPointError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            metrics = out["metrics"]
            rows.append(
                {
                    "step": step,
                    "motion": float(out["losses"]["motion"]),
                    "total": float(out["losses"]["total"]),
                    **{k: float(metrics[k]) for k in ("m3d_disp_01", "m3d_attr_01", "m3d_covered_01", "psnr")},
                }
            )
            step += 1
            if step >= steps:
                break
    dataset.close()
    first = sum(r["motion"] for r in rows[:WINDOW]) / WINDOW
    last = sum(r["motion"] for r in rows[-WINDOW:]) / WINDOW
    return {
        "steps": steps,
        "seconds": time.time() - start,
        "train_windows": len(dataset),
        "m3d_loss_first_mean": first,
        "m3d_loss_last_mean": last,
        "m3d_loss_ratio_last_first": last / first,
        "probe_m3d_first": probes[0]["motion"],
        "probe_m3d_last": probes[-1]["motion"],
        "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
        "train": rows,
        "probe": probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="converted track-array cache")
    parser.add_argument("--out", type=Path, default=REPO / "docs/figures/droid_smoke")
    parser.add_argument("--config", nargs="+", default=[str(REPO / "configs/droid/pretrain.yaml")])
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--windows", type=int, default=4)
    args = parser.parse_args()
    root = writable_path(args.cache, DROID_CACHE_ROOT)
    out = writable_path(args.out, DROID_CACHE_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    report = {"gpu_mapping": GPU_MAPPING, "cache": str(root), "contract": contract_checks(root, args.windows)}
    manifest = json.loads((root / "manifest.json").read_text())
    train_index = next(i for i, row in enumerate(manifest["windows"]) if row["split"] == "train")
    overlays = cached_coordinate_overlays(root, out) + cached_coordinate_overlays(root, out, window_index=train_index)
    report["overlays"] = [str(path.relative_to(REPO)) if REPO in path.parents else str(path) for path in overlays]
    cfg = load_config(args.config, ["loss.ramp_steps=0", "train.warmup_steps=20", f"train.steps={args.steps}", *args.set])
    report["config_overrides"] = ["loss.ramp_steps=0", "train.warmup_steps=20", f"train.steps={args.steps}", *args.set]
    fit = overfit(cfg, root, args.steps, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    curve_png(
        out / "m3d_overfit.png",
        [r["step"] for r in fit["probe"]],
        {"probe M3D loss (fixed batch, fixed sampling)": [r["motion"] for r in fit["probe"]]},
    )
    report["overfit"] = {k: v for k, v in fit.items() if k not in ("train", "probe")}
    (out / "m3d_overfit_curve.json").write_text(json.dumps({"train": fit["train"], "probe": fit["probe"]}))
    (out / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("contract", "overlays", "overfit")}, indent=2))


if __name__ == "__main__":
    main()
