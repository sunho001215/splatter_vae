from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from .backbones import ContrastiveProjector, TemporalViTEncoder, ViTSmallConfig
from .config import SPLATTERVAE_ARCHITECTURE, TEMPORAL_WINDOW
from .decoder import GaussianSlotDecoder


class SplatterVAE(nn.Module):
    """DROID temporal ViT-S plus a multi-token grouped Gaussian decoder."""

    architecture_id = SPLATTERVAE_ARCHITECTURE

    def __init__(
        self,
        *,
        vit_config: ViTSmallConfig,
        gaussian_parameters_per_gaussian: int,
        decoder_config: Mapping[str, Any],
        masking_ratio: float = 0.60,
        motion_visible_fraction: float = 0.50,
        projector_hidden_dimension: int = 1024,
        projector_output_dimension: int = 256,
        motion_translation_max: float = 0.50,
    ):
        super().__init__()
        if vit_config.temporal_window != TEMPORAL_WINDOW:
            raise ValueError("The model requires a three-frame history.")
        if float(motion_translation_max) <= 0.0:
            raise ValueError("motion_translation_max must be positive.")
        self.motion_translation_max = float(motion_translation_max)
        self.encoder = TemporalViTEncoder(
            vit_config,
            masking_ratio=masking_ratio,
            motion_visible_fraction=motion_visible_fraction,
        )
        self.contrastive_projector = ContrastiveProjector(
            input_dimension=vit_config.embed_dimension,
            hidden_dimension=int(projector_hidden_dimension),
            output_dimension=int(projector_output_dimension),
        )
        self.gaussian_decoder = GaussianSlotDecoder(
            encoder_dimension=vit_config.embed_dimension,
            gaussian_parameters_per_gaussian=int(gaussian_parameters_per_gaussian),
            **dict(decoder_config),
        )

    @property
    def num_groups(self) -> int:
        return self.gaussian_decoder.num_groups

    @property
    def gaussians_per_group(self) -> int:
        return self.gaussian_decoder.gaussians_per_group

    @property
    def num_gaussians(self) -> int:
        return self.num_groups * self.gaussians_per_group

    @property
    def masking_ratio(self) -> float:
        return self.encoder.masking_ratio

    def encode_pretraining(
        self,
        histories: torch.Tensor,
        optical_flows: torch.Tensor,
        image_validity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.encoder(
            histories,
            optical_flows=optical_flows,
            image_validity=image_validity,
            apply_mask=True,
        )
        output["projected_cls"] = self.contrastive_projector(output["cls_token"])
        return output

    def predict_gaussian_parameters(
        self, encoder_tokens: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.gaussian_decoder(encoder_tokens)

    def forward(
        self,
        representation_histories: torch.Tensor,
        representation_flows: torch.Tensor,
        representation_validity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if (
            representation_histories.dim() != 6
            or representation_histories.shape[1] != 2
        ):
            raise ValueError(
                "Expected two positive representation views as (B,2,3,3,224,224)."
            )
        batch = representation_histories.shape[0]
        histories = representation_histories.flatten(0, 1)
        flows = representation_flows.flatten(0, 1)
        validity = (
            None
            if representation_validity is None
            else representation_validity.flatten(0, 1)
        )
        encoded = self.encode_pretraining(histories, flows, validity)
        gaussian = self.predict_gaussian_parameters(encoded["decoder_tokens"][0::2])
        output = {
            "cls_tokens_by_view": encoded["cls_token"].view(batch, 2, -1),
            "projected_cls_by_view": encoded["projected_cls"].view(batch, 2, -1),
            "current_patch_tokens_by_view": encoded["current_patch_tokens"].view(
                batch, 2, encoded["current_patch_tokens"].shape[1], -1
            ),
            "patch_masks_by_view": encoded["patch_mask"].view(batch, 2, -1),
            "visible_patch_ids_by_view": encoded["visible_patch_ids"].view(
                batch, 2, -1
            ),
        }
        output.update(gaussian)
        return output

    def inference_features(self, histories: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return raw reusable encoder features; projector and masking are excluded."""
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if histories.device.type == "cuda"
            else nullcontext()
        )
        with context:
            output = self.encoder.inference_features(histories)
        patch_tokens = output["patch_tokens"]
        if patch_tokens.shape[1:] != (196, 384):
            raise RuntimeError(
                f"Inference representation must be (B,196,384), got {tuple(patch_tokens.shape)}."
            )
        return {
            "cls_token": output["cls_token"].float(),
            "patch_tokens": patch_tokens.float(),
        }

    def decoder_configuration(self) -> dict[str, object]:
        return self.gaussian_decoder.configuration()

    def parameter_counts(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )

        encoder = count(self.encoder)
        decoder = count(self.gaussian_decoder)
        projector = count(self.contrastive_projector)
        return {
            "encoder": encoder,
            "gaussian_decoder": decoder,
            "contrastive_projector": projector,
            "total_trainable": encoder + decoder + projector,
        }

    @torch.no_grad()
    def position_initialization_diagnostics(self) -> dict[str, object]:
        anchors = self.gaussian_decoder.group_anchors
        memory = anchors.new_zeros(1, 2, self.encoder.embed_dimension)
        decoded = self.gaussian_decoder(memory)
        xyz = decoded["raw_gaussian_params"][..., :3]
        child_offsets = decoded["child_offsets"]
        return {
            "parent_anchor_mean": tuple(float(value) for value in anchors.mean(0)),
            "parent_anchor_std": tuple(
                float(value) for value in anchors.std(0, unbiased=False)
            ),
            "parent_displacement_max": float(
                decoded["parent_displacements"].abs().max()
            ),
            "child_offset_rms": float(child_offsets.square().mean().sqrt()),
            "child_offset_max": float(
                torch.linalg.vector_norm(child_offsets, dim=-1).max()
            ),
            "xyz_min": tuple(float(value) for value in xyz.amin(dim=(0, 1))),
            "xyz_max": tuple(float(value) for value in xyz.amax(dim=(0, 1))),
        }
