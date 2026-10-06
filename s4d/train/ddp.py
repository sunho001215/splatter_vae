"""Minimal DDP helpers. Single-process runs never initialise a process group."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistContext:
    """Initialise NCCL from torchrun env vars when present; otherwise a single-process context."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        return DistContext(dist.get_rank(), local_rank, dist.get_world_size(), torch.device("cuda", local_rank))
    torch.cuda.set_device(0)
    return DistContext(0, 0, 1, torch.device("cuda", 0))


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def barrier() -> None:
    if is_distributed():
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()


class _AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local: torch.Tensor) -> torch.Tensor:
        ctx.world = world_size()
        ctx.rank = rank()
        if ctx.world == 1:
            return local
        gathered = [torch.empty_like(local) for _ in range(ctx.world)]
        dist.all_gather(gathered, local.contiguous())
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        if ctx.world == 1:
            return grad
        # Reduce matching GLOBAL rows before selecting this rank's local slice.
        # Selecting first would incorrectly sum different sample rows across ranks.
        reduced = grad.contiguous().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced.chunk(ctx.world, dim=0)[ctx.rank].contiguous()


def all_gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    return _AllGatherWithGrad.apply(x)


def reduce_mean(values: dict[str, torch.Tensor | float]) -> dict[str, float]:
    """Average scalar metrics across ranks (no-op on one process)."""
    out = {}
    for k, v in values.items():
        t = v.detach().float().clone() if torch.is_tensor(v) else torch.tensor(float(v), device="cuda")
        if t.numel() != 1:
            continue
        if is_distributed():
            if dist.get_backend() == "nccl" and not t.is_cuda:
                t = t.cuda()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= world_size()
        out[k] = float(t.item())
    return out


def rank_zero_call(fn) -> None:
    """All ranks participate even when only rank zero has an evaluation callback.

    Broadcast callback errors so peers fail rather than hanging in a barrier.
    """
    barrier()
    error = [None]
    if rank() == 0 and fn is not None:
        try:
            fn()
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if is_distributed():
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"Rank-zero evaluation failed: {error[0]}")
    barrier()
