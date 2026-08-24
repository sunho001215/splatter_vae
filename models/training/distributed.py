from __future__ import annotations

import inspect
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    process_group_initialized: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launched_by_torchrun = "RANK" in os.environ and "LOCAL_RANK" in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError("DROID pretraining requires CUDA.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    initialized_here = False
    if launched_by_torchrun and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        initialized_here = True
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        process_group_initialized=initialized_here,
    )


def cuda_device_diagnostics(context: DistributedContext) -> dict[str, Any]:
    return {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "torch_cuda_current_device": torch.cuda.current_device(),
        "torch_cuda_device_name": torch.cuda.get_device_name(context.device),
        "rank": context.rank,
        "local_rank": context.local_rank,
        "world_size": context.world_size,
        "mapped_device": str(context.device),
    }


def seed_distributed(seed: int, context: DistributedContext) -> None:
    rank_seed = int(seed) + 10_003 * context.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)


def wrap_ddp(
    model: nn.Module,
    context: DistributedContext,
    *,
    find_unused_parameters: bool = False,
) -> nn.Module:
    if not dist.is_initialized():
        return model
    kwargs: dict[str, Any] = {
        "device_ids": [context.local_rank],
        "output_device": context.local_rank,
        "find_unused_parameters": bool(find_unused_parameters),
        "gradient_as_bucket_view": True,
    }
    parameters = inspect.signature(DistributedDataParallel).parameters
    if "forward_sync_buffers" in parameters:
        kwargs["forward_sync_buffers"] = False
    else:
        kwargs["broadcast_buffers"] = False
    return DistributedDataParallel(model, **kwargs)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def reduce_scalar_metrics(
    metrics: Mapping[str, torch.Tensor | float | int],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, value in metrics.items():
        tensor = (
            value.detach().float()
            if torch.is_tensor(value)
            else torch.tensor(float(value), device="cuda")
        )
        if tensor.numel() != 1:
            continue
        tensor = tensor.clone()
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= dist.get_world_size()
        output[name] = float(tensor.item())
    return output


def distributed_barrier() -> None:
    if dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def destroy_distributed(context: DistributedContext) -> None:
    if context.process_group_initialized and dist.is_initialized():
        dist.destroy_process_group()
