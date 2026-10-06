"""Data-driven workspace anchor statistics from valid world-space depth points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus

GPU_MAPPING = guard_gpus()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from s4d.geometry import lift_depth  # noqa: E402


def dataset_for(regime: str, path: Path):
    if regime == "metaworld":
        from s4d.data.metaworld.dataset import MetaworldWindowDataset, list_episodes

        return MetaworldWindowDataset(path, list_episodes(path), strides=(9,), fixed_stride=9)
    from s4d.data.droid.dataset import DroidCacheDataset

    return DroidCacheDataset(path, split="train")


class Moments:
    """Streaming double-precision population mean/std, with bounded cloud memory."""

    def __init__(self):
        self.count = 0
        self.total = torch.zeros(3, dtype=torch.float64)
        self.square = torch.zeros(3, dtype=torch.float64)

    def add(self, points):
        points = points.double()
        self.count += len(points)
        self.total += points.sum(0)
        self.square += points.square().sum(0)

    def result(self):
        if self.count == 0:
            raise ValueError("no valid depth points for workspace statistics")
        mean = self.total / self.count
        std = (self.square / self.count - mean.square()).clamp_min(0).sqrt()
        return {"mean": mean.tolist(), "std": std.tolist(), "num_points": self.count}


def compute_statistics(dataset, samples: int, seed: int) -> dict:
    if samples <= 0 or len(dataset) == 0:
        raise ValueError("workspace statistics require positive samples and a nonempty dataset")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(samples, len(dataset)), replace=False)
    scene, dynamic, depths = Moments(), Moments(), []
    for index in indices:
        sample = dataset[int(index)]
        depth = sample["depth"][0, :, 0]
        points = lift_depth(depth, sample["K"], sample["c2w"])
        valid = depth > 0
        scene.add(points[valid])
        dynamic.add(points[valid & (sample["motion_score"][0, :, 0] > 0.5)])
        depths.append(depth[valid])
    depth = torch.cat(depths)
    if len(depth) == 0:
        raise ValueError("no valid depth points for workspace statistics")
    # A seeded uniform point sample, reused for every depth percentile.
    if len(depth) > 200000:
        depth = depth[torch.from_numpy(rng.integers(len(depth), size=200000))]
    scene_stats = scene.result()
    dynamic_stats = dynamic.result() if dynamic.count else {**scene_stats, "num_points": 0}
    return {
        "scene": scene_stats,
        "dynamic": dynamic_stats,
        "dynamic_fallback_to_scene": dynamic.count == 0,
        "depth_percentiles": {str(p): float(torch.quantile(depth, p / 100)) for p in (1, 5, 50, 95, 99)},
        "depth_quantile_sample_points": len(depth),
        "samples": len(indices),
        "sample_indices": indices.tolist(),
        "seed": seed,
        "standard_deviation": "population, streaming double precision",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["metaworld", "droid"], required=True)
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    output = args.out.resolve()
    allowed = (REPO, Path("/home/ws/data/metaworld/splatter4d_v1"), Path("/home/ws/data/droid_pointworld_cache_sample"))
    if not any(root in output.parents for root in allowed):
        raise ValueError("workspace statistics must remain within an authorized output root")
    torch.set_num_threads(1)
    stats = compute_statistics(dataset_for(args.regime, args.path), args.samples, args.seed)
    stats.update(source=str(args.path), regime=args.regime, gpu_mapping=GPU_MAPPING)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
