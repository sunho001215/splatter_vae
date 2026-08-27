from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

from models.gaussian.parameterization import (
    SplatterConfig,
    WorldSpaceGaussianParameterization,
)
from models.training.config import TrainConfig
from models.training.distributed import DistributedContext, move_to_device
from models.training.losses import cross_view_info_nce
from models.training.online_preprocessing import OnlineTeacherPipeline
from models.training.reconstruction import compute_droid_reconstruction


def _detach_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    return value


@torch.no_grad()
def evaluate_droid(
    model: nn.Module,
    parameterization: WorldSpaceGaussianParameterization,
    splatter_config: SplatterConfig,
    dataloader: DataLoader,
    train_config: TrainConfig,
    context: DistributedContext,
    background_color: torch.Tensor,
    online_preprocessor: OnlineTeacherPipeline,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    model.eval()
    local_sums: dict[str, torch.Tensor] = {}
    local_batches = 0
    visualization: dict[str, Any] | None = None
    for batch_index, cpu_batch in enumerate(dataloader):
        if batch_index >= int(train_config.validation_batches):
            break
        raw_batch = move_to_device(cpu_batch, context.device)
        devices = [context.local_rank] if context.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(
                int(train_config.seed) + 1_000_003 + batch_index + 1009 * context.rank
            )
            batch = online_preprocessor(
                raw_batch,
                novel_enabled=bool(train_config.novel_view_enabled),
                seed=(
                    int(train_config.seed)
                    + 10_000_019
                    + batch_index
                    + 1009 * context.rank
                ),
            )
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if train_config.bf16 and context.device.type == "cuda"
                else nullcontext()
            )
            with autocast:
                prediction = model(
                    batch["representation_histories"],
                    batch["representation_flows"],
                    batch["representation_validity"],
                )
                contrastive, contrast_metrics = cross_view_info_nce(
                    prediction["projected_cls_by_view"],
                    train_config.contrastive_temperature,
                )
            reconstruction = compute_droid_reconstruction(
                parameterization,
                splatter_config,
                prediction,
                batch,
                train_config,
                motion_translation_max=float(
                    (
                        model.module if hasattr(model, "module") else model
                    ).motion_translation_max
                ),
                background_color=background_color,
                return_renders=context.is_main and visualization is None,
                novel_view_enabled=bool(train_config.novel_view_enabled),
            )
        total = (
            reconstruction["loss"]
            + float(train_config.contrastive_weight) * contrastive.float()
        )
        values: dict[str, torch.Tensor] = {
            "loss": total,
            "contrastive_loss": contrastive,
            "rgb_l1_loss": reconstruction["rgb_l1_loss"],
            "dssim_loss": reconstruction["dssim_loss"],
            "metric_depth_loss": reconstruction["metric_depth_loss"],
            "scale_invariant_depth_loss": reconstruction["scale_invariant_depth_loss"],
            "flow_loss": reconstruction["flow_loss"],
            "visibility_loss": reconstruction["visibility_loss"],
            "novel_view_loss": reconstruction["novel_view_loss"],
            "novel_support_fraction": reconstruction["novel_support_fraction"],
            **contrast_metrics,
        }
        for name, value in values.items():
            local_sums[name] = (
                local_sums.get(name, torch.zeros_like(value.detach().float()))
                + value.detach().float()
            )
        local_batches += 1
        if context.is_main and visualization is None:
            visualization = {
                "batch": _detach_to_cpu(batch),
                "reconstruction": _detach_to_cpu(reconstruction),
                "prediction": _detach_to_cpu(prediction),
            }
    count = torch.tensor(float(local_batches), device=context.device)
    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    output: dict[str, float] = {}
    for name, value in local_sums.items():
        value = value.to(context.device)
        if dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        output[f"validation/{name}"] = float((value / count.clamp_min(1.0)).item())
    return output, visualization
