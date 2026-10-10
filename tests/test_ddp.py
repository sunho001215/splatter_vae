"""Real Gloo ranks catch reduction of different gradient slices as one slice."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from s4d.losses.invariance import multi_positive_info_nce
from s4d.train.ddp import all_gather_with_grad

REPO = Path(__file__).resolve().parents[1]


def test_single_process_gather_is_identity_with_identity_gradient():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    gathered = all_gather_with_grad(x)
    torch.testing.assert_close(gathered, x, atol=0, rtol=0)
    (gathered * torch.tensor([[2.0, 3.0], [4.0, 5.0]])).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[2.0, 3.0], [4.0, 5.0]]), atol=0, rtol=0)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="The installed torch build lacks the CPU Gloo backend.")
def test_two_rank_gather_gradient_and_infonce_equal_global_reference(tmp_path):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    processes = []
    try:
        for rank in range(2):
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-I",
                        str(REPO / "tests" / "_ddp_worker.py"),
                        "--rank",
                        str(rank),
                        "--rendezvous",
                        str(tmp_path / "rendezvous"),
                        "--out",
                        str(tmp_path / f"rank{rank}.json"),
                    ],
                    cwd=REPO,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        for process in processes:
            output, _ = process.communicate(timeout=120)
            assert process.returncode == 0, output
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=15)
    records = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    expected_gradients = ([11.0, 22.0], [33.0, 44.0])
    for rank, record in enumerate(records):
        assert record["rank_zero_callback_called"] == (rank == 0)
        assert record["rank_zero_callback_completed"]
        assert record["evaluation_barriers"] == 3
        assert record["callback_error"] == "Rank-zero evaluation failed: ValueError: intentional evaluation callback failure"
        assert record["rank_distinct_rng_states"]
        assert record["restored_step"] == 7
        assert record["restored_rng_draws"] == record["expected_rng_draws"]
        torch.testing.assert_close(
            torch.tensor(record["gathered"]), torch.tensor([0.1, 0.2, 1.1, 1.2]), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(torch.tensor(record["grad"]), torch.tensor(expected_gradients[rank]), atol=0, rtol=0)
    features = torch.sin(torch.arange(32).reshape(4, 2, 4).float() * 0.7).requires_grad_()
    reference_loss, _ = multi_positive_info_nce(features, temperature=0.1)
    reference_loss.backward()
    torch.testing.assert_close(
        torch.tensor([record["info_loss"] for record in records]).mean(), reference_loss.detach(), atol=1e-6, rtol=1e-6
    )
    distributed_grad = torch.cat([torch.tensor(record["info_grad"]) for record in records]) / 2
    torch.testing.assert_close(distributed_grad, features.grad, atol=2e-6, rtol=2e-5)
