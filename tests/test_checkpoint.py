"""Checkpoint restores the actual next CPU update and every recorded RNG."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler

from s4d.train.checkpoint import gather_rng_states, load_checkpoint, save_checkpoint
from s4d.train.loop import infinite


def _modules():
    encoder = torch.nn.Linear(4, 4)
    decoder = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return encoder, decoder, optimizer, scheduler


def _step(encoder, decoder, optimizer, scheduler):
    x, target = torch.randn(3, 4), torch.randn(3, 2)
    optimizer.zero_grad(set_to_none=True)
    loss = (decoder(torch.tanh(encoder(x))) - target).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach()


def _draw_rngs():
    return random.random(), np.random.normal(size=4), torch.rand(4), torch.rand(4, device="cuda")


def test_checkpoint_next_update_and_rngs_are_identical(tmp_path):
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    torch.cuda.manual_seed_all(19)
    encoder, decoder, optimizer, scheduler = _modules()
    _step(encoder, decoder, optimizer, scheduler)
    checkpoint = save_checkpoint(tmp_path / "step_0000001.pt", 1, encoder, decoder, optimizer, scheduler, {"seed": 19})
    expected_rngs = _draw_rngs()
    expected_loss = _step(encoder, decoder, optimizer, scheduler)
    expected_state = [{k: v.clone() for k, v in module.state_dict().items()} for module in (encoder, decoder)]
    expected_lr = scheduler.get_last_lr()
    random.seed(987)
    np.random.seed(987)
    torch.manual_seed(987)
    torch.cuda.manual_seed_all(987)
    encoder, decoder, optimizer, scheduler = _modules()
    assert load_checkpoint(checkpoint, encoder, decoder, optimizer, scheduler) == 1
    actual_rngs = _draw_rngs()
    assert actual_rngs[0] == expected_rngs[0]
    np.testing.assert_array_equal(actual_rngs[1], expected_rngs[1])
    torch.testing.assert_close(actual_rngs[2], expected_rngs[2], atol=0, rtol=0)
    torch.testing.assert_close(actual_rngs[3], expected_rngs[3], atol=0, rtol=0)
    torch.testing.assert_close(_step(encoder, decoder, optimizer, scheduler), expected_loss, atol=0, rtol=0)
    assert scheduler.get_last_lr() == expected_lr
    for module, expected in zip((encoder, decoder), expected_state):
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    assert (tmp_path / "latest.pt").is_file()


def test_single_process_gathered_rng_overrides_later_save_time_rng(tmp_path):
    encoder, decoder, optimizer, scheduler = _modules()
    random.seed(29)
    np.random.seed(29)
    torch.manual_seed(29)
    torch.cuda.manual_seed_all(29)
    states = gather_rng_states()
    assert len(states) == 1
    expected = _draw_rngs()
    checkpoint = save_checkpoint(tmp_path / "gathered.pt", 2, encoder, decoder, optimizer, scheduler, {}, rng_by_rank=states)
    assert load_checkpoint(checkpoint, encoder, decoder, rank=0) == 2
    actual = _draw_rngs()
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
    torch.testing.assert_close(actual[3], expected[3], atol=0, rtol=0)


@pytest.mark.parametrize("invalid_rank", [-1, 1])
def test_checkpoint_rejects_rank_without_saved_rng_state(tmp_path, invalid_rank):
    encoder, decoder, optimizer, scheduler = _modules()
    checkpoint = save_checkpoint(
        tmp_path / "gathered.pt", 2, encoder, decoder, optimizer, scheduler, {}, rng_by_rank=gather_rng_states()
    )
    with pytest.raises(ValueError, match="does not contain this distributed rank"):
        load_checkpoint(checkpoint, encoder, decoder, rank=invalid_rank)


def _loader(rank=0, world=1):
    dataset = list(range(19))
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=37, drop_last=True)
    return DataLoader(
        dataset, batch_size=2, sampler=sampler, drop_last=True, generator=torch.Generator().manual_seed(37), num_workers=0
    )


def test_sampler_resume_replays_epoch_and_skips_exact_batch_prefix():
    for rank, world in ((0, 1), (0, 2), (1, 2)):
        baseline = infinite(_loader(rank, world))
        expected = [next(baseline).tolist() for _ in range(30)]
        for start in (0, 1, 4, 9, 15, 22):
            loader = _loader(rank, world)
            resumed = infinite(loader, start_step=start)
            actual = [next(resumed).tolist() for _ in range(5)]
            assert actual == expected[start : start + 5], (rank, world, start)
            assert loader.sampler.epoch == (start + 4) // len(loader)


def test_sampler_replay_does_not_consume_model_global_rng():
    torch.manual_seed(47)
    state = torch.get_rng_state().clone()
    resumed = infinite(_loader(), start_step=15)
    next(resumed)
    torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)


def test_distributed_sampler_ranks_are_disjoint_for_same_epoch():
    left, right = _loader(rank=0, world=2), _loader(rank=1, world=2)
    left_stream, right_stream = infinite(left, start_step=len(left)), infinite(right, start_step=len(right))
    left_ids = torch.cat([next(left_stream) for _ in range(len(left))]).tolist()
    right_ids = torch.cat([next(right_stream) for _ in range(len(right))]).tolist()
    assert not set(left_ids) & set(right_ids)
    assert left.sampler.epoch == right.sampler.epoch == 1
