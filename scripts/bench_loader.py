"""Measure DataLoader prefetch/worker settings on real pretraining steps (directive item 3).

For each (prefetch_factor, num_workers) setting, in the order given and then reversed so that host-load drift cancels:
``--warmup`` steps, then ``--steps`` timed steps of the real training step (forward, backward, optimizer). Reports mean
step time, mean data wait, mean GPU time per step and the proportional set size (PSS, shared memory counted once) of the
whole process tree (main process + loader workers), sampled every ``--sample-every`` steps.

    CUDA_VISIBLE_DEVICES=$GPU python scripts/bench_loader.py --config configs/metaworld/base.yaml \
        configs/metaworld/tasks/hammer.yaml --set aug.synth_views=2 aug.synth_render=true --out runs/<job>/bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401  (sys.path)
from _bootstrap import REPO, guard_gpus

GPU_MAPPING = guard_gpus()

import torch  # noqa: E402
from train import build_data  # noqa: E402

from s4d.config import load_config  # noqa: E402
from s4d.model.render import require_prebuilt_renderer  # noqa: E402
from s4d.train import ddp  # noqa: E402
from s4d.train.loop import build_model, build_optimizer, forward_losses, make_train_loader, move_batch  # noqa: E402
from s4d.train.workers import fork_safe_iter  # noqa: E402


def tree_pss_gb(root: int) -> float:
    """PSS of ``root`` and all its descendants (GB); shared pages are divided among the processes that map them."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            ppid = int(Path(f"/proc/{entry}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (FileNotFoundError, ProcessLookupError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(entry))
    total, stack = 0, [root]
    while stack:
        pid = stack.pop()
        stack.extend(children.get(pid, []))
        try:
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    total += int(line.split()[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total / 2**20


def run_setting(cfg, dataset, model, optimizer, device, prefetch, workers, warmup, steps, sample_every) -> dict:
    cfg["train"]["prefetch_factor"], cfg["train"]["num_workers"] = prefetch, workers
    loader = make_train_loader(dataset, cfg, ddp.init_distributed(), seed=0)
    batches = fork_safe_iter(loader)
    step_times, data_times, gpu_times, pss = [], [], [], []
    start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for step in range(warmup + steps):
        t0 = time.perf_counter()
        batch = move_batch(next(batches), device)
        t1 = time.perf_counter()
        start_evt.record()
        out = forward_losses(model, batch, cfg, 100000)
        optimizer.zero_grad(set_to_none=True)
        out["total"].backward()
        optimizer.step()
        end_evt.record()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if step >= warmup:
            step_times.append(t2 - t0)
            data_times.append(t1 - t0)
            gpu_times.append(start_evt.elapsed_time(end_evt) / 1000.0)
        if step % sample_every == 0:
            pss.append(tree_pss_gb(os.getpid()))
    del batches, loader  # persistent workers shut down with their iterator
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    return {
        "prefetch_factor": prefetch,
        "num_workers": workers,
        "step_time_s": mean(step_times),
        "data_wait_s": mean(data_times),
        "gpu_time_s": mean(gpu_times),
        "pss_peak_gb": max(pss),
        "pss_mean_gb": mean(pss),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--settings", nargs="+", default=["4x8", "2x8", "1x8", "4x6", "2x6", "1x6"],
                    help="prefetch x workers, e.g. 4x8")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--sample-every", type=int, default=25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if REPO.resolve() not in out.parents:
        raise ValueError("benchmark output must stay inside the repository")
    require_prebuilt_renderer()
    cfg = load_config(args.config, args.set)
    device = torch.device("cuda", 0)
    train_ds, _, _, _ = build_data(cfg)
    model = build_model(cfg).to(device).train()
    optimizer, _ = build_optimizer(model, cfg)
    settings = [tuple(int(x) for x in s.split("x")) for s in args.settings]
    rows = []
    for prefetch, workers in settings + settings[::-1]:
        row = run_setting(cfg, train_ds, model, optimizer, device, prefetch, workers, args.warmup, args.steps,
                          args.sample_every)
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {}
    for prefetch, workers in settings:
        both = [r for r in rows if (r["prefetch_factor"], r["num_workers"]) == (prefetch, workers)]
        summary[f"{prefetch}x{workers}"] = {k: sum(r[k] for r in both) / len(both) for k in both[0] if k.endswith(("_s", "_gb"))}
        summary[f"{prefetch}x{workers}"]["pss_peak_gb"] = max(r["pss_peak_gb"] for r in both)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gpus": GPU_MAPPING, "config": args.config, "set": args.set, "rows": rows,
                               "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
