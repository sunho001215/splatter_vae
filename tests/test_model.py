from __future__ import annotations

import pytest
import torch

from models.gaussian.parameterization import NUM_GAUSSIANS, gaussian_params_per_gaussian
from models.splattervae import SplatterVAE, ViTSmallConfig


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
    assert encoder_only["cls_token"].shape == (1, 384)
    assert encoder_only["patch_tokens"].shape == (1, 196, 384)


def test_masking_projector_and_grouped_decoder_shapes(model: SplatterVAE) -> None:
    histories = torch.randn(1, 3, 3, 224, 224)
    flows = torch.randn(1, 2, 2, 224, 224)
    validity = torch.ones(1, 3, 1, 224, 224, dtype=torch.bool)
    with torch.inference_mode():
        encoded = model.encode_pretraining(histories, flows, validity)
        decoded = model.predict_gaussian_parameters(encoded["decoder_tokens"])
    assert encoded["visible_patch_ids"].shape == (1, 78)
    assert encoded["patch_mask"].sum().item() == 118
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
