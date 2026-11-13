"""Checkpoint save/load for encoder + decoder + optimizer + scheduler + RNG state."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def capture_rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def gather_rng_states() -> list[dict]:
    """Called by every rank at checkpoint boundaries, before rank zero writes."""
    local = capture_rng_state()
    if not dist.is_available() or not dist.is_initialized():
        return [local]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, local)
    return states


def save_checkpoint(
    path: str | Path, step: int, encoder, decoder, optimizer, scheduler, cfg: dict, *, rng_by_rank: list[dict] | None = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": int(step),
        "encoder": _unwrap(encoder).state_dict(),
        "decoder": _unwrap(decoder).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": cfg,
        "rng": capture_rng_state(),
        "rng_by_rank": rng_by_rank,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    latest = path.parent / "latest.pt"
    tmp_latest = latest.with_suffix(".tmp")
    torch.save(state, tmp_latest)
    os.replace(tmp_latest, latest)
    return path


def load_checkpoint(
    path: str | Path, encoder, decoder, optimizer=None, scheduler=None, restore_rng: bool = True, *, rank: int = 0
) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    _unwrap(encoder).load_state_dict(state["encoder"])
    _unwrap(decoder).load_state_dict(state["decoder"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng and "rng" in state:
        ranks = state.get("rng_by_rank")
        if ranks is not None and (rank >= len(ranks) or rank < 0):
            raise ValueError("checkpoint RNG state does not contain this distributed rank")
        rng = ranks[rank] if ranks is not None else state["rng"]
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and len(rng["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    return int(state["step"])
