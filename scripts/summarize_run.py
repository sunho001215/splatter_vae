"""Concise text report of a run: latest losses, trends, §10 metric table, warnings. CPU only.

python scripts/summarize_run.py outputs/metaworld-hammer-full-0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import guard_gpus

GPU_MAPPING = guard_gpus()

# (summary key, threshold, higher_is_better, label)
CRITERIA = [
    ("metric/psnr", 28.0, True, "M3 PSNR train cams >= 28"),
    ("metric/psnr_moving", 25.0, True, "M3 PSNR moving pixels >= 25"),
    ("metric/psnr_heldout", 24.0, True, "M3 PSNR held-out cams >= 24"),
    ("metric/depth_absrel", 0.03, False, "M3 depth AbsRel <= 0.03"),
    ("metric/rel_epe_02", 0.35, False, "M3 relative EPE 0->2 <= 0.35"),
    ("metric/retrieval_top1_train", 0.95, True, "M4 retrieval top-1 train cams >= 0.95"),
    ("metric/retrieval_top1_heldout", 0.85, True, "M4 retrieval top-1 held-out >= 0.85"),
    ("metric/effective_rank_retrieval", 32.0, True, "M4 effective rank >= 32"),
    ("metric/r2_hand_pos_val_traincams", 0.95, True, "M5 R2 hand pos (train cams) >= 0.95"),
    ("metric/r2_obj_pos_val_traincams", 0.90, True, "M5 R2 object pos (train cams) >= 0.90"),
    ("metric/r2_hand_vel_val_traincams", 0.60, True, "M5 R2 hand velocity (train cams) >= 0.6"),
    ("metric/r2_hand_pos_val_heldout", 0.95, True, "M5 R2 hand pos (held-out) >= 0.95"),
    ("metric/r2_obj_pos_val_heldout", 0.90, True, "M5 R2 object pos (held-out) >= 0.90"),
    ("metric/r2_hand_vel_val_heldout", 0.60, True, "M5 R2 hand velocity (held-out) >= 0.6"),
    ("metric/dyn_alpha_share_moving", 0.70, True, "M7 dynamic alpha share on moving pixels >= 0.7"),
    ("metric/active_fraction_scene", 0.10, True, "M1 active fraction scene > 0.1"),
    ("metric/active_fraction_dynamic", 0.10, True, "M1 active fraction dynamic > 0.1"),
]


def load_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_eval(run_dir: Path) -> tuple[Path | None, dict]:
    evals = sorted((run_dir / "eval").glob("step_*/summary.json")) if (run_dir / "eval").exists() else []
    if not evals:
        return None, {}
    return evals[-1], json.loads(evals[-1].read_text())


def trend(records: list[dict], key: str, window: int = 20) -> str:
    vals = [r[key] for r in records if key in r and math.isfinite(r[key])]
    if len(vals) < 2 * window:
        return "n/a"
    a, b = sum(vals[-2 * window : -window]) / window, sum(vals[-window:]) / window
    return f"{a:.4f} -> {b:.4f} ({'down' if b < a else 'up'})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    records = load_metrics(run_dir)
    train_records = [r for r in records if "loss/total" in r]
    print(f"run: {run_dir}")
    warnings = []
    if train_records:
        last = train_records[-1]
        print(f"latest train step {last['step']}:")
        for key in sorted(k for k in last if k.startswith("loss/")):
            print(f"  {key:28s} {last[key]:.5f}   trend {trend(train_records, key)}")
        for key in (
            "metric/psnr",
            "metric/psnr_moving",
            "metric/epe_02",
            "metric/rel_epe_02",
            "metric/dyn_alpha_share_moving",
            "metric/active_fraction_scene",
            "metric/active_fraction_dynamic",
            "metric/effective_rank",
            "train/lr",
            "train/step_time",
            "train/gpu_mem_gb",
        ):
            if key in last:
                print(f"  {key:28s} {last[key]:.5f}")
        for r in train_records:
            for k, v in r.items():
                if isinstance(v, float) and not math.isfinite(v) and k.startswith("loss/"):
                    warnings.append(f"non-finite {k} at step {r['step']}")
                    break
        for name in ("metric/active_fraction_scene", "metric/active_fraction_dynamic"):
            if name in last and last[name] <= 0.10:
                warnings.append(f"{name} = {last[name]:.3f} <= 0.10 (collapse)")
        for key in ("loss/total", "loss/render", "loss/motion"):
            t = trend(train_records, key)
            if t.endswith("(up)"):
                warnings.append(f"{key} rising: {t}")
    path, summary = latest_eval(run_dir)
    if path:
        print(f"\nlatest evaluation: {path}")
        # Every validation stride (simulator steps between frames) is reported separately: key@s<stride>.
        suffixes = sorted({k.rsplit("@", 1)[1] for k in summary if "@s" in k}, key=lambda x: int(x[1:])) or [""]
        print(f"  {'criterion':48s} " + " ".join(f"{('@' + x if x else 'value'):>13s}" for x in suffixes))
        for key, thr, higher, label in CRITERIA:
            cells = []
            for suffix in suffixes:
                v = summary.get(f"{key}@{suffix}" if suffix else key)
                if v is None:
                    cells.append(f"{'missing':>13s}")
                    continue
                if not math.isfinite(v):
                    cells.append(f"{'unavailable':>13s}")
                    continue
                strict = key in ("metric/active_fraction_scene", "metric/active_fraction_dynamic")
                ok = (v > thr if strict else v >= thr) if higher else (v <= thr)
                cells.append(f"{v:8.4f} {'PASS' if ok else 'FAIL'}")
            print(f"  {label:48s} " + " ".join(cells))
    else:
        warnings.append("no evaluation summary found")
    print("\nwarnings:" if warnings else "\nwarnings: none")
    for w in warnings:
        print(f"  - {w}")


if __name__ == "__main__":
    main()
