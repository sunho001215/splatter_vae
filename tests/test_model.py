from __future__ import annotations

import pytest
import torch

from models.gaussian.parameterization import NUM_GAUSSIANS, gaussian_params_per_gaussian
from models.splattervae import SplatterVAE, ViTSmallConfig
from models.splattervae.backbones import MultiHeadSelfAttention
from models.splattervae.decoder import MultiHeadCrossAttention


@pytest.fixture(scope="module")
def model() -> SplatterVAE:
    torch.manual_seed(7)
    return SplatterVAE(
        vit_config=ViTSmallConfig(),
        gaussian_parameters_per_gaussian=gaussian_params_per_gaussian(1),
        decoder_config={
            "num_groups": 256,
            "gaussians_per_group": 8,
            "dimension": 256,
            "depth": 2,
            "num_heads": 8,
            "swiglu_hidden_dimension": 512,
            "global_center": (0.5, 0.0, 0.5),
            "anchor_initial_spread": 0.45,
            "parent_displacement_scale": 0.25,
            "child_radius": 0.06,
        },
    ).eval()


def test_inference_representation_shapes(model: SplatterVAE) -> None:
    histories = torch.randn(1, 3, 3, 224, 224)
    with torch.inference_mode():
        features = model.inference_features(histories)
        encoder_only = model.encoder.inference_features(histories)
    assert features["cls_token"].shape == (1, 384)
    assert features["patch_tokens"].shape == (1, 196, 384)
    assert features["patch_validity"].shape == (1, 196)
    assert features["patch_validity"].all()
    assert encoder_only["cls_token"].shape == (1, 384)
    assert encoder_only["patch_tokens"].shape == (1, 196, 384)


def test_masking_projector_and_grouped_decoder_shapes(model: SplatterVAE) -> None:
    histories = torch.randn(1, 3, 3, 224, 224)
    motion = torch.rand(1, 1, 224, 224)
    validity = torch.ones(1, 3, 1, 224, 224, dtype=torch.bool)
    with torch.inference_mode():
        encoded = model.encode_pretraining(histories, motion, validity)
        decoded = model.predict_gaussian_parameters(
            encoded["decoder_tokens"], encoded["decoder_token_validity"]
        )
    assert encoded["visible_patch_ids"].shape == (1, 78)
    assert encoded["patch_mask"].sum().item() == 118
    assert encoded["patch_validity"].sum().item() == 196
    assert encoded["num_visible_patches"].tolist() == [78]
    assert encoded["current_patch_tokens"].shape == (1, 78, 384)
    assert encoded["decoder_tokens"].shape == (1, 1 + 3 * 78, 384)
    assert encoded["projected_cls"].shape == (1, 256)
    torch.testing.assert_close(encoded["projected_cls"].norm(dim=-1), torch.ones(1))
    assert decoded["raw_gaussian_params"].shape == (
        1,
        NUM_GAUSSIANS,
        gaussian_params_per_gaussian(1),
    )
    assert decoded["raw_motion_params"].shape == (1, NUM_GAUSSIANS, 6)
    assert decoded["parent_centers"].shape == (1, 256, 3)
    assert decoded["child_offsets"].shape == (1, 256, 8, 3)


def test_mask_ratio_is_relative_to_valid_patches_and_batch_length_is_padded(
    model: SplatterVAE,
) -> None:
    histories = torch.randn(2, 3, 3, 224, 224)
    motion = torch.rand(2, 1, 224, 224)
    validity = torch.ones(2, 3, 1, 224, 224, dtype=torch.bool)
    validity[1, :, :, 160:] = False
    with torch.inference_mode():
        encoded = model.encode_pretraining(histories, motion, validity)
    # Ten 16px rows remain in sample 1: 10*14 = 140 valid patches.
    assert encoded["patch_validity"].sum(dim=1).tolist() == [196, 140]
    assert encoded["num_visible_patches"].tolist() == [78, 56]
    assert encoded["visible_patch_ids"].shape == (2, 78)
    assert encoded["visible_patch_validity"].sum(dim=1).tolist() == [78, 56]
    assert encoded["patch_mask"].sum(dim=1).tolist() == [118, 84]


def test_partial_patches_are_valid_and_fully_padded_patches_never_enter_tube(
    model: SplatterVAE,
) -> None:
    histories = torch.randn(1, 3, 3, 224, 224)
    motion = torch.rand(1, 1, 224, 224)
    validity = torch.zeros(1, 3, 1, 224, 224, dtype=torch.bool)
    # The second 16-pixel patch row contains one real row; all later rows are padding.
    validity[..., :17, :] = True
    with torch.inference_mode():
        encoded = model.encode_pretraining(histories, motion, validity)
    patch_validity = encoded["patch_validity"].reshape(1, 14, 14)
    assert patch_validity[:, :2].all()
    assert not patch_validity[:, 2:].any()
    assert encoded["num_visible_patches"].tolist() == [11]
    selected = encoded["visible_patch_ids"][0, :11]
    assert encoded["patch_validity"][0, selected].all()
    tube_validity = encoded["decoder_token_validity"][:, 1:].reshape(1, 3, 11)
    assert torch.equal(tube_validity[:, 0], tube_validity[:, 1])
    assert torch.equal(tube_validity[:, 1], tube_validity[:, 2])


def test_attention_and_gaussian_decoder_ignore_dummy_token_values(
    model: SplatterVAE,
) -> None:
    torch.manual_seed(17)
    self_attention = MultiHeadSelfAttention(24, 4).eval()
    values = torch.randn(2, 7, 24)
    validity = torch.tensor([[True, True, True, False, False, False, False]] * 2)
    changed = values.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 10_000.0
    with torch.inference_mode():
        first = self_attention(values, validity)
        second = self_attention(changed, validity)
    torch.testing.assert_close(first[:, :3], second[:, :3], atol=1e-6, rtol=1e-6)

    cross_attention = MultiHeadCrossAttention(24, 4).eval()
    query = torch.randn(2, 5, 24)
    with torch.inference_mode():
        first_cross = cross_attention(query, values, validity)
        second_cross = cross_attention(query, changed, validity)
    torch.testing.assert_close(first_cross, second_cross, atol=1e-6, rtol=1e-6)

    memory = torch.randn(1, 11, 384)
    memory_validity = torch.tensor(
        [[True, True, True, True, True, False, False, False, False, False, False]]
    )
    changed_memory = memory.clone()
    changed_memory[:, 5:] = torch.randn_like(changed_memory[:, 5:]) * 10_000.0
    with torch.inference_mode():
        first_gaussian = model.predict_gaussian_parameters(memory, memory_validity)
        second_gaussian = model.predict_gaussian_parameters(
            changed_memory, memory_validity
        )
    torch.testing.assert_close(
        first_gaussian["raw_gaussian_params"],
        second_gaussian["raw_gaussian_params"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_inference_invalid_image_patches_do_not_contaminate_cls_or_valid_patches(
    model: SplatterVAE,
) -> None:
    histories = torch.randn(1, 3, 3, 224, 224)
    validity = torch.ones(1, 3, 1, 224, 224, dtype=torch.bool)
    validity[..., 112:, :] = False
    changed = histories.clone()
    changed[..., 112:, :] = torch.randn_like(changed[..., 112:, :]) * 1000.0
    with torch.inference_mode():
        first = model.inference_features(histories, validity)
        second = model.inference_features(changed, validity)
    torch.testing.assert_close(
        first["cls_token"], second["cls_token"], atol=1e-5, rtol=1e-5
    )
    valid = first["patch_validity"][0]
    torch.testing.assert_close(
        first["patch_tokens"][0, valid],
        second["patch_tokens"][0, valid],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.count_nonzero(first["patch_tokens"][0, ~valid]) == 0


def test_decoder_refuses_compact_single_vector_conditioning(model: SplatterVAE) -> None:
    with pytest.raises(ValueError, match="multiple encoder tokens"):
        model.predict_gaussian_parameters(torch.randn(1, 384))
    with pytest.raises(ValueError, match="single-token"):
        model.predict_gaussian_parameters(torch.randn(1, 1, 384))


def test_architecture_and_parameter_accounting(model: SplatterVAE) -> None:
    assert model.encoder.config == ViTSmallConfig()
    assert model.masking_ratio == pytest.approx(0.60)
    assert model.num_groups == 256
    assert model.gaussians_per_group == 8
    assert model.num_gaussians == 2048
    counts = model.parameter_counts()
    assert counts == {
        "encoder": 21_672_576,
        "gaussian_decoder": 2_217_760,
        "contrastive_projector": 656_640,
        "total_trainable": 24_546_976,
    }
