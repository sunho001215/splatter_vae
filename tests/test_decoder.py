"""Slots are the decoder's only input; only dynamic Gaussians carry motion."""

from __future__ import annotations

import inspect

import pytest
import torch

from s4d.model.decoder import DecoderConfig, GaussianDecoder, GroupConfig, GroupHeads
from s4d.model.gaussians import DYNAMIC_GROUP, SCENE_GROUP


def _config(single_group=False):
    return DecoderConfig(
        slot_dim=12,
        dim=16,
        depth=2,
        heads=4,
        motion_max=0.25,
        scene=GroupConfig(3, 2, 0.15, 0.06, 0.001, 0.08),
        dynamic=GroupConfig(2, 2, 0.4, 0.04, 0.0005, 0.03),
        single_group=single_group,
    )


def test_decoder_forward_requires_only_slots():
    assert set(inspect.signature(GaussianDecoder.forward).parameters) == {"self", "slots"}
    decoder = GaussianDecoder(_config())
    with pytest.raises(TypeError):
        decoder(torch.randn(1, 2, 12), cameras=torch.eye(4))
    with pytest.raises(ValueError, match="slots"):
        decoder(torch.randn(1, 12))


def test_gaussian_budget_attributes_and_group_labels():
    decoder = GaussianDecoder(_config())
    scene = decoder(torch.randn(2, 2, 12))
    assert scene.num_gaussians == 10
    assert scene.xyz.shape == (2, 10, 3)
    assert scene.opacity.shape == (2, 10)
    assert torch.equal(scene.group, torch.tensor([SCENE_GROUP] * 6 + [DYNAMIC_GROUP] * 4))
    for attr in ("xyz", "scales", "quats", "opacity", "rgb", "delta01", "delta12"):
        assert torch.isfinite(getattr(scene, attr)).all(), attr
    assert ((scene.opacity >= 0) & (scene.opacity <= 1)).all()
    assert ((scene.rgb >= 0) & (scene.rgb <= 1)).all()
    torch.testing.assert_close(scene.quats.norm(dim=-1), torch.ones(2, 10), atol=1e-6, rtol=1e-6)
    for group, spec in ((SCENE_GROUP, decoder.cfg.scene), (DYNAMIC_GROUP, decoder.cfg.dynamic)):
        scale = scene.scales[:, scene.group == group]
        assert ((scale >= spec.scale_min) & (scale <= spec.scale_max)).all()


def test_static_group_has_no_motion_head_and_exact_zero_motion_after_dynamic_head_changes():
    decoder = GaussianDecoder(_config())
    assert decoder.groups[0].motion_head is None
    with torch.no_grad():
        decoder.groups[1].motion_head.bias.copy_(torch.tensor([4.0, -4.0, 2.0, -2.0, 3.0, -3.0]))
    scene = decoder(torch.randn(2, 2, 12))
    static = scene.group == SCENE_GROUP
    dynamic = scene.group == DYNAMIC_GROUP
    assert torch.count_nonzero(scene.delta01[:, static]).item() == 0
    assert torch.count_nonzero(scene.delta12[:, static]).item() == 0
    assert scene.delta01[:, dynamic].abs().sum() > 0
    assert scene.delta12[:, dynamic].abs().sum() > 0
    assert scene.delta01.abs().max() <= 0.25
    assert scene.delta12.abs().max() <= 0.25
    torch.testing.assert_close(scene.xyz_at(1), scene.xyz + scene.delta01, atol=0, rtol=0)
    torch.testing.assert_close(scene.xyz_at(2), scene.xyz + scene.delta01 + scene.delta12, atol=0, rtol=0)
    sequence = scene.xyz_sequence()
    assert sequence.shape == (2, 3, 10, 3)
    torch.testing.assert_close(sequence[:, 0, static], sequence[:, 2, static], atol=0, rtol=0)


def test_parent_offsets_are_unbounded_while_child_offsets_are_bounded():
    config = GroupConfig(2, 3, 0.4, 0.04, 0.0005, 0.03)
    head = GroupHeads(config, dim=16, dynamic=True)
    with torch.no_grad():
        head.anchors.zero_()
        head.parent_pos.bias.fill_(100.0)
        head.xyz_head.bias.fill_(100.0)
    result = head(torch.randn(1, 2, 16), motion_max=0.25, scale_act_bias=-1.0)
    torch.testing.assert_close(result["parent_centers"], torch.full((1, 2, 3), 40.0), atol=0, rtol=0)
    offsets = result["xyz"].reshape(1, 2, 3, 3) - result["parent_centers"][:, :, None]
    assert offsets.abs().max() <= config.child_radius + 2e-6
    assert offsets.abs().min() >= config.child_radius - 2e-6


def test_single_group_ablation_preserves_total_gaussian_count():
    decoder = GaussianDecoder(_config(single_group=True))
    scene = decoder(torch.randn(1, 2, 12))
    assert scene.num_gaussians == 10
    assert (scene.group == DYNAMIC_GROUP).all()
    assert len(decoder.groups) == 1 and decoder.groups[0].motion_head is not None


def test_slot_gradients_flow_through_film_and_cross_attention():
    decoder = GaussianDecoder(_config())
    slots = torch.randn(2, 2, 12, requires_grad=True)
    scene = decoder(slots)
    (scene.rgb.sum() + scene.opacity.sum()).backward()
    assert slots.grad is not None and torch.isfinite(slots.grad).all()
    assert slots.grad.abs().sum() > 0
    assert decoder.film[0].weight.grad is not None
    assert decoder.blocks[0].cross_attn.in_proj_weight.grad is not None


def test_anchor_statistics_apply_to_a_cuda_decoder():
    decoder = GaussianDecoder(_config()).cuda()
    stats = {name: {"mean": [0.1, 0.6, 0.2], "std": [1e-6, 1e-6, 1e-6]} for name in ("scene", "dynamic")}
    decoder.set_anchor_statistics(stats)
    for head in decoder.groups:
        assert head.anchors.is_cuda
        expected = torch.tensor([0.1, 0.6, 0.2], device="cuda").expand_as(head.anchors)
        torch.testing.assert_close(head.anchors, expected, atol=1e-4, rtol=0)
