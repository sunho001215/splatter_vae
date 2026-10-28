import json, math, statistics, sys
from pathlib import Path
root = Path("/home/ws/ws/splatter4d/.worktrees/selfrender/runs/smoke")
out = {}
for name in ("smoke-csw4-hammer", "smoke-sr-full-hammer", "smoke-sr-warmup-hammer"):
    p = root / name / "metrics.jsonl"
    if not p.is_file():
        out[name] = "missing"; continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    tr = [r for r in rows if "train/step_time" in r]
    late = [r for r in tr if r["step"] > 150]  # steady state (full self-render strength from step 150 in the sr-full run)
    finite = all(isinstance(v, (int, float)) and math.isfinite(v) for r in tr for k, v in r.items()
                 if k.startswith(("loss/", "metric/")) and not k.endswith(("_moving", "moving_t0", "moving_t1", "moving_t2")))
    nonfinite = sorted({k for r in tr for k, v in r.items() if isinstance(v, float) and not math.isfinite(v)})
    out[name] = {
        "logged_steps": [r["step"] for r in tr],
        "step_time_s_steps_151_300": statistics.mean(r["train/step_time"] for r in late),
        "gpu_time_s_steps_151_300": statistics.mean(r["train/gpu_time"] for r in late),
        "data_time_s_steps_151_300": statistics.mean(r["train/data_time"] for r in late),
        "peak_gpu_mem_gb": max(r["train/gpu_mem_gb"] for r in tr),
        "all_logged_losses_metrics_finite": finite, "nonfinite_keys": nonfinite,
        "first": {k: tr[0][k] for k in ("loss/total", "loss/infonce", "metric/psnr") if k in tr[0]},
        "last": {k: v for k, v in tr[-1].items() if k.startswith(("loss/", "metric/psnr", "metric/synth", "metric/selfrender",
                                                                    "metric/positive_cosine")) and not k.startswith(("loss/rgb_t", "loss/coverage_t", "loss/depth_l1_t", "loss/chamfer_"))},
        "selfrender_weight_by_step": {r["step"]: r.get("metric/selfrender_weight") for r in tr},
        "completion": (root / name / "completion.json").is_file(),
    }
print(json.dumps(out, indent=1))
