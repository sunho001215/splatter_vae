"""Compare a deterministic single-GPU global batch with two-rank NCCL/DDP.

First launch with one approved UUID and --output outputs/<name>/single.json.
Then launch with both UUIDs using torchrun --nproc_per_node=2 and pass that
file as --reference plus a new --output. No optimizer update or training runs.
The comparison uses FP32, no masking/stochastic depth, and fixed source views.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus

GPU_MAPPING = guard_gpus()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402

from s4d.config import get, load_config  # noqa: E402
from s4d.data.contract import collate  # noqa: E402
from s4d.data.droid.dataset import DroidCacheDataset  # noqa: E402
from s4d.model.render import require_prebuilt_renderer  # noqa: E402
from s4d.train import ddp  # noqa: E402
from s4d.train.loop import build_model, forward_losses, move_batch, seed_everything  # noqa: E402

FORMAT = "splatter4d-step0-v1"
TOLERANCE = 1e-4
GLOBAL_BATCH = 4


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--indices", nargs="+", type=int, default=list(range(GLOBAL_BATCH)))
    parser.add_argument("--output", required=True, help="new repository-local JSON file")
    parser.add_argument("--reference", help="single-GPU JSON from the same config, seed and global batch")
    return parser


def repository_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if REPO.resolve() not in resolved.parents:
        raise ValueError("comparison artifacts must remain inside the new repository")
    return resolved


def partition_indices(indices: list[int], length: int, world_size: int, rank: int) -> list[int]:
    if world_size not in (1, 2) or not 0 <= rank < world_size:
        raise ValueError("comparison supports one or two ranks")
    if len(indices) != GLOBAL_BATCH or len(set(indices)) != GLOBAL_BATCH:
        raise ValueError("comparison requires exactly four distinct training-window indices")
    if any(type(index) is not int or not 0 <= index < length for index in indices):
        raise ValueError("comparison indices must be in-range integers")
    count = GLOBAL_BATCH // world_size
    return indices[rank * count : (rank + 1) * count]


def comparison_config(cfg: dict) -> dict:
    if get(cfg, "data.regime") != "droid":
        raise ValueError("step-zero comparison requires the DROID cache")
    cfg = copy.deepcopy(cfg)
    cfg["train"]["bf16"] = False
    cfg["model"]["encoder"]["drop_path"] = 0.0
    cfg["loss"]["teacher_scale_jitter"] = None
    return cfg


def tensor_fingerprint(values: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        if not torch.is_tensor(value):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def comparison_result(reference: dict, records: list[dict]) -> dict:
    if reference.get("format") != FORMAT or reference.get("world_size") != 1:
        raise ValueError("reference must be an actual single-GPU step-zero record")
    if len(records) != 2 or sorted(record["rank"] for record in records) != [0, 1]:
        raise ValueError("comparison requires exactly two distributed rank records")
    for record in records:
        for name in ("config_sha256", "model_sha256", "batch_sha256", "indices", "source_views"):
            if record[name] != reference[name]:
                raise ValueError(f"single/distributed {name} mismatch")
    values = [float(record["loss"]) for record in records]
    expected = float(reference["loss"])
    if not all(math.isfinite(value) for value in [expected, *values]):
        raise ValueError("non-finite comparison loss")
    actual = sum(values) / len(values)
    error = abs(actual - expected)
    return {
        "single_gpu_global_loss": expected,
        "distributed_rank_losses": values,
        "distributed_mean_loss": actual,
        "absolute_error": error,
        "absolute_tolerance": TOLERANCE,
        "passed": error <= TOLERANCE,
    }


def run(cfg: dict, indices: list[int], output: Path, reference_path: Path | None) -> None:
    # Refuse before reading the cache or importing gsplat's JIT-capable initializer.
    require_prebuilt_renderer()
    ctx = ddp.init_distributed()
    try:
        if len(GPU_MAPPING) != ctx.world_size or ctx.world_size not in (1, 2):
            raise ValueError("use one UUID for the single process or both UUIDs for exactly two NCCL ranks")
        if (ctx.world_size == 2) != (reference_path is not None):
            raise ValueError("--reference is required only for the two-rank invocation")
        cfg = comparison_config(cfg)
        seed_everything(int(get(cfg, "train.seed", 0)))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dataset = DroidCacheDataset(
            get(cfg, "data.root"),
            split="train",
            image_height=get(cfg, "model.encoder.image_height"),
            image_width=get(cfg, "model.encoder.image_width"),
        )
        try:
            local_indices = partition_indices(indices, len(dataset), ctx.world_size, ctx.rank)
            global_batch = collate([dataset[index] for index in indices])
        finally:
            dataset.close()
        model = build_model(cfg).to(ctx.device)
        stats_path = get(cfg, "model.anchor_stats")
        if stats_path:
            model.decoder.set_anchor_statistics(json.loads(Path(stats_path).read_text()))
        # The bundle remains training=True so InfoNCE actually gathers global negatives.
        # Child eval disables stochastic network layers without disabling the objective.
        model.train()
        model.encoder.eval()
        model.decoder.eval()
        wrapped = DistributedDataParallel(model, device_ids=[ctx.local_rank]) if ctx.world_size == 2 else model
        cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        source_views = [index % global_batch["images"].shape[2] for index in indices]
        record = {
            "format": FORMAT,
            "rank": ctx.rank,
            "world_size": ctx.world_size,
            "gpu_mapping": GPU_MAPPING,
            "config_sha256": cfg_hash,
            "model_sha256": tensor_fingerprint(model.state_dict()),
            "batch_sha256": tensor_fingerprint(global_batch),
            "indices": indices,
            "local_indices": local_indices,
            "source_views": source_views,
            "comparison_settings": {"step": 0, "mask_ratio": 0.0, "bf16": False, "stochastic_layers": False},
            "resolved_config": cfg,
        }
        count = GLOBAL_BATCH // ctx.world_size
        start, stop = ctx.rank * count, (ctx.rank + 1) * count
        local_batch = {
            key: ({name: value[start:stop] for name, value in values.items()} if key == "meta" else values[start:stop])
            for key, values in global_batch.items()
        }
        with torch.no_grad():
            out = forward_losses(
                wrapped,
                move_batch(local_batch, ctx.device),
                cfg,
                step=0,
                source=torch.tensor(source_views[start:stop], device=ctx.device),
                mask_ratio=0.0,
            )
        record["loss"] = float(out["total"])
        records = [record]
        if ctx.world_size == 2:
            records = [None, None]
            dist.all_gather_object(records, record)

        def write_record():
            report = dict(record)
            if reference_path is not None:
                reference = json.loads(reference_path.read_text())
                report.update(comparison_result(reference, records))
                report["reference"] = str(reference_path)
                report["rank_records"] = records
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
            if reference_path is not None and not report["passed"]:
                raise AssertionError(f"step-zero loss differs by {report['absolute_error']}, limit {TOLERANCE}")

        ddp.rank_zero_call(write_record if ctx.is_main else None)
    finally:
        ddp.cleanup()


def main() -> None:
    args = argument_parser().parse_args()
    output = repository_path(args.output)
    reference = repository_path(args.reference) if args.reference else None
    if output.exists():
        raise FileExistsError(f"refusing to overwrite comparison artifact {output}")
    run(load_config(args.config, args.set), args.indices, output, reference)


if __name__ == "__main__":
    main()
