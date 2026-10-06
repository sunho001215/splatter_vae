"""Tube masks share patch identity across time; inference sees all RGB patches."""

from __future__ import annotations

import inspect

import pytest
import torch

from s4d.model.encoder import Encoder, EncoderConfig, sample_tube_mask


def _encoder(depth=1, use_cls_token=False):
    return Encoder(
        EncoderConfig(
            image_height=32,
            image_width=32,
            patch_size=8,
            width=16,
            depth=depth,
            heads=4,
            num_slots=2,
            slot_dim=12,
            num_frames=3,
            mask_ratio=0.5,
            drop_path=0.0,
            use_cls_token=use_cls_token,
        )
    )


def test_visible_mask_retains_highest_motion_half_and_random_remainder():
    scores = torch.linspace(0.1, 1.0, 16).expand(64, -1)
    visible = sample_tube_mask(scores, mask_ratio=0.5, threshold=0.05)
    assert visible.dtype == torch.bool and visible.shape == (64, 16)
    assert torch.equal(visible.sum(1), torch.full((64,), 8))
    assert visible[:, -4:].all()  # Half the eight visible patches are the top four.
    assert torch.equal(visible[:, :-4].sum(1), torch.full((64,), 4))
    assert visible[:, :-4].unique(dim=0).shape[0] > 1, "The remaining visible patches must actually vary."


def test_static_samples_choose_patches_randomly_not_by_location():
    torch.manual_seed(101)
    visible = sample_tube_mask(torch.zeros(2048, 16), mask_ratio=0.5, threshold=0.05)
    assert torch.equal(visible.sum(1), torch.full((2048,), 8))
    frequencies = visible.float().mean(0)
    assert torch.all((frequencies > 0.44) & (frequencies < 0.56)), frequencies


def test_zero_mask_ratio_keeps_all_patches():
    assert sample_tube_mask(torch.rand(3, 16), mask_ratio=0.0, threshold=0.05).all()


def test_tube_mask_uses_identical_patch_identities_in_every_frame():
    encoder = _encoder(depth=0).train()
    with torch.no_grad():
        encoder.temporal_embed.zero_()
    frame = torch.rand(2, 1, 3, 32, 32)
    images = frame.expand(-1, 3, -1, -1, -1).clone()
    score = torch.zeros(2, 3, 1, 32, 32)
    score[..., :16, :16] = 0.9
    output = encoder(images, score)
    assert output["visible"].shape == (2, 16)
    assert torch.equal(output["visible"].sum(1), torch.full((2,), 8))
    tokens = output["patch_tokens"].reshape(2, 3, 8, 16)
    torch.testing.assert_close(tokens[:, 0], tokens[:, 1], atol=0, rtol=0)
    torch.testing.assert_close(tokens[:, 0], tokens[:, 2], atol=0, rtol=0)


@pytest.mark.parametrize("use_cls_token", [False, True])
def test_encoder_returns_projected_slots_and_keeps_cls_separate(use_cls_token):
    encoder = _encoder(use_cls_token=use_cls_token).eval()
    output = encoder(torch.rand(2, 3, 3, 32, 32))
    assert output["slots"].shape == (2, 2, 12)
    assert output["patch_tokens"].shape == (2, 3 * 16, 16)
    assert output["visible"].all()
    assert (output["cls"] is not None) == use_cls_token
    if use_cls_token:
        assert output["cls"].shape == (2, 16)


def test_policy_state_is_flattened_eval_no_grad_and_accepts_uint8():
    encoder = _encoder().train()
    images = torch.randint(0, 256, (2, 3, 3, 32, 32), dtype=torch.uint8)
    state = encoder.policy_state(images)
    assert state.shape == (2, 2 * 12)
    assert state.dtype == torch.float32
    assert not state.requires_grad
    assert encoder.training  # Temporary inference must not mutate the caller's mode.
    encoder.eval()
    expected = encoder(images.float() / 255, mask_ratio=0.0)["slots"].flatten(1)
    torch.testing.assert_close(state, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(encoder.policy_state(images.float() / 255), state, atol=1e-6, rtol=1e-6)
    assert not encoder.training


def test_encoder_accepts_no_camera_parameters():
    parameters = set(inspect.signature(Encoder.forward).parameters)
    assert parameters == {"self", "images", "motion_score", "mask_ratio"}
    with pytest.raises(TypeError):
        _encoder()(torch.rand(1, 3, 3, 32, 32), K=torch.eye(3))


def test_encoder_rejects_incorrect_time_or_image_dimensions():
    with pytest.raises(ValueError, match="expects"):
        _encoder()(torch.rand(1, 1, 3, 32, 32))
    with pytest.raises(ValueError, match="expects"):
        _encoder()(torch.rand(1, 3, 3, 32, 40))
