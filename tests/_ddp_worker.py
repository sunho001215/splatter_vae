"""Guarded CPU/Gloo worker for a real cross-rank autograd regression."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from s4d.gpu_guard import enforce_allowed_gpus  # noqa: E402

enforce_allowed_gpus()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from s4d.losses.invariance import multi_positive_info_nce  # noqa: E402
from s4d.train import ddp  # noqa: E402
from s4d.train.checkpoint import gather_rng_states, load_checkpoint, save_checkpoint  # noqa: E402
from s4d.train.ddp import all_gather_with_grad  # noqa: E402


def _rank_zero_regression(out, rank):
    marker = Path(out).parent / "rank_zero_callback_done.txt"
    callback_called, barrier_calls = False, 0
    original_barrier = ddp.barrier

    def tracked_barrier():
        nonlocal barrier_calls
        barrier_calls += 1
        original_barrier()

    def successful_callback():
        nonlocal callback_called
        assert rank == 0
        callback_called = True
        marker.write_text("done")

    def failing_callback():
        raise ValueError("intentional evaluation callback failure")

    ddp.barrier = tracked_barrier
    try:
        ddp.rank_zero_call(successful_callback if rank == 0 else None)
        assert marker.read_text() == "done"
        try:
            ddp.rank_zero_call(failing_callback if rank == 0 else None)
        except RuntimeError as exc:
            callback_error = str(exc)
        else:
            raise AssertionError("Both ranks must receive the callback failure")
    finally:
        ddp.barrier = original_barrier
    return {
        "rank_zero_callback_called": callback_called,
        "rank_zero_callback_completed": True,
        "evaluation_barriers": barrier_calls,
        "callback_error": callback_error,
    }


def _draw_rngs():
    return {
        "python": random.random(),
        "numpy": np.random.normal(size=4).tolist(),
        "torch": torch.rand(4).tolist(),
        "cuda": torch.rand(4, device="cuda").cpu().tolist(),
    }


def _rank_rng_regression(out, rank):
    encoder, decoder = torch.nn.Linear(2, 3), torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    seed = 1300 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    states = gather_rng_states()
    assert len(states) == 2
    distinct = (
        states[0]["python"] != states[1]["python"]
        and not np.array_equal(states[0]["numpy"][1], states[1]["numpy"][1])
        and not torch.equal(states[0]["torch"], states[1]["torch"])
        and not torch.equal(states[0]["cuda"][0], states[1]["cuda"][0])
    )
    checkpoint = Path(out).parent / "distributed_checkpoint.pt"

    def write_checkpoint():
        save_checkpoint(checkpoint, 7, encoder, decoder, optimizer, scheduler, {}, rng_by_rank=states)

    ddp.rank_zero_call(write_checkpoint if rank == 0 else None)
    expected = _draw_rngs()
    random.seed(987)
    np.random.seed(987)
    torch.manual_seed(987)
    torch.cuda.manual_seed_all(987)
    step = load_checkpoint(checkpoint, encoder, decoder, optimizer, scheduler, rank=rank)
    actual = _draw_rngs()
    assert expected == actual
    return {
        "rank_distinct_rng_states": distinct,
        "restored_step": step,
        "expected_rng_draws": expected,
        "restored_rng_draws": actual,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--rendezvous", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(args.rendezvous).resolve().as_uri(),
        rank=args.rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        local = torch.tensor([0.1, 0.2]) + args.rank
        local.requires_grad_()
        gathered = all_gather_with_grad(local)
        coefficients = torch.tensor([1.0, 2.0, 3.0, 4.0]) * (1 if args.rank == 0 else 10)
        (gathered * coefficients).sum().backward()
        global_features = torch.sin(torch.arange(32).reshape(4, 2, 4).float() * 0.7)
        features = global_features[args.rank * 2 : (args.rank + 1) * 2].clone().requires_grad_()
        info_loss, _ = multi_positive_info_nce(features, temperature=0.1)
        info_loss.backward()
        record = {
            "rank": args.rank,
            "gathered": gathered.detach().tolist(),
            "grad": local.grad.tolist(),
            "info_loss": info_loss.item(),
            "info_grad": features.grad.tolist(),
        }
        record.update(_rank_zero_regression(args.out, args.rank))
        record.update(_rank_rng_regression(args.out, args.rank))
        Path(args.out).write_text(json.dumps(record))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
