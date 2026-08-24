from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.gaussian.parameterization import (
    SplatterConfig,
    SplatterDataConfig,
    SplatterModelConfig,
    WorldSpaceGaussianParameterization,
    gaussian_params_per_gaussian,
)
from models.splattervae import SplatterVAE, ViTSmallConfig
from models.training.config import TrainConfig
from models.training.distributed import (
    cuda_device_diagnostics,
    destroy_distributed,
    initialize_distributed,
    seed_distributed,
    unwrap_model,
    wrap_ddp,
)
from models.training.loop import (
    TrainingState,
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
)
from models.training.losses import cross_view_info_nce
from models.training.reconstruction import compute_droid_reconstruction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic DROID GPU forward/backward smoke test."
    )
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument(
        "--output-root",
        default="/home/ws/ws/droid_training/derived/smoke",
    )
    return parser.parse_args()


def _model() -> tuple[SplatterVAE, SplatterConfig]:
    splatter = SplatterConfig(
        data=SplatterDataConfig(
            img_height=224,
            img_width=224,
            znear=0.05,
            zfar=3.0,
            white_background=False,
        ),
        model=SplatterModelConfig(
            max_sh_degree=1,
            scale_min=(0.001, 0.001, 0.001),
            scale_max=(0.05, 0.05, 0.05),
        ),
    )
    model = SplatterVAE(
        vit_config=ViTSmallConfig(),
        gaussian_parameters_per_gaussian=gaussian_params_per_gaussian(1),
        decoder_config={
            "global_center": (0.0, 0.0, 1.0),
            "anchor_initial_spread": 0.10,
            "parent_displacement_scale": 0.10,
            "child_radius": 0.02,
        },
    )
    return model, splatter


def _batch(device: torch.device) -> dict[str, torch.Tensor]:
    batch, times, cameras, size = 1, 3, 2, 224
    histories = torch.randn(batch, 2, times, 3, size, size, device=device)
    representation_flows = torch.zeros(batch, 2, 2, 2, size, size, device=device)
    representation_validity = torch.ones(
        batch, 2, times, 1, size, size, dtype=torch.bool, device=device
    )
    K = torch.tensor(
        [[150.0, 0.0, 112.0], [0.0, 150.0, 112.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    c2w = torch.eye(4, device=device).repeat(batch, times, cameras, 1, 1)
    c2w[:, :, 1, 0, 3] = 0.15
    w2c = torch.linalg.inv(c2w.float())
    target_shape = (batch, times, cameras, 1, size, size)
    return {
        "representation_histories": histories,
        "representation_flows": representation_flows,
        "representation_validity": representation_validity,
        "target_rgb": torch.rand(batch, times, cameras, 3, size, size, device=device),
        "target_image_validity": torch.ones(
            target_shape, dtype=torch.bool, device=device
        ),
        "target_depth": torch.ones(target_shape, device=device),
        "target_depth_confidence": torch.ones(target_shape, device=device),
        "target_depth_validity": torch.ones(
            target_shape, dtype=torch.bool, device=device
        ),
        "target_flow": torch.zeros(batch, 3, cameras, 2, size, size, device=device),
        "target_flow_validity": torch.ones(
            batch, 3, cameras, 1, size, size, dtype=torch.bool, device=device
        ),
        "target_K": K.view(1, 1, 1, 3, 3)
        .expand(batch, times, cameras, -1, -1)
        .contiguous(),
        "target_c2w": c2w,
        "target_w2c": w2c,
        "calibration_validity": torch.ones(batch, dtype=torch.bool, device=device),
        "synthetic_available": torch.zeros(batch, dtype=torch.bool, device=device),
    }


def main() -> None:
    args = parse_args()
    context = initialize_distributed()
    try:
        seed_distributed(123, context)
        diagnostics = cuda_device_diagnostics(context)
        model, splatter = _model()
        model.to(context.device)
        batch = _batch(context.device)

        torch.cuda.reset_peak_memory_stats(context.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = model.encode_pretraining(
                batch["representation_histories"][:, 0],
                batch["representation_flows"][:, 0],
                batch["representation_validity"][:, 0],
            )
            encoder_probe = (
                encoded["projected_cls"].square().mean()
                + encoded["current_patch_tokens"].square().mean()
            )
        encoder_probe.backward()
        vit_peak = torch.cuda.max_memory_allocated(context.device) / (1024**3)
        model.zero_grad(set_to_none=True)

        features = model.inference_features(batch["representation_histories"][:, 0])
        assert features["cls_token"].shape == (1, 384)
        assert features["patch_tokens"].shape == (1, 196, 384)
        if args.ddp:
            model = wrap_ddp(model, context)
        config = TrainConfig(
            max_global_steps=1,
            warmup_steps=0,
            warmup_fraction=0.0,
            checkpoint_dir=str(Path(args.output_root) / "checkpoints"),
        )
        optimizer, _learning_rates = build_optimizer(
            unwrap_model(model), config, effective_global_batch=1
        )
        parameterization = WorldSpaceGaussianParameterization(splatter).to(
            context.device
        )
        background = torch.zeros(3, device=context.device)
        torch.cuda.reset_peak_memory_stats(context.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(
                batch["representation_histories"],
                batch["representation_flows"],
                batch["representation_validity"],
            )
            contrastive, _contrast_metrics = cross_view_info_nce(
                prediction["projected_cls_by_view"], config.contrastive_temperature
            )
        reconstruction = compute_droid_reconstruction(
            parameterization,
            splatter,
            prediction,
            batch,
            config,
            motion_translation_max=unwrap_model(model).motion_translation_max,
            background_color=background,
            return_renders=True,
        )
        total = reconstruction["loss"] + contrastive.float()
        if not torch.isfinite(total):
            raise FloatingPointError(f"Synthetic training loss is non-finite: {total}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), 1.0)
        optimizer.step()
        gaussian_peak = torch.cuda.max_memory_allocated(context.device) / (1024**3)
        unwrapped = unwrap_model(model)
        gradient_presence = {
            "encoder": any(
                parameter.grad is not None
                for parameter in unwrapped.encoder.parameters()
            ),
            "gaussian_decoder": any(
                parameter.grad is not None
                for parameter in unwrapped.gaussian_decoder.parameters()
            ),
            "contrastive_projector": any(
                parameter.grad is not None
                for parameter in unwrapped.contrastive_projector.parameters()
            ),
        }
        if not all(gradient_presence.values()):
            raise RuntimeError(f"Missing module gradients: {gradient_presence}")
        mode = "ddp-nproc1" if args.ddp else "single-gpu"
        checkpoint = Path(args.output_root) / f"{mode}-checkpoint.pt"
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            TrainingState(global_step=1),
            config,
            context,
        )
        loaded = load_checkpoint(checkpoint, model, optimizer)
        assert loaded.global_step == 1
        result = {
            "mode": mode,
            "gpu": diagnostics,
            "loss": float(total.detach()),
            "inference_cls_shape": list(features["cls_token"].shape),
            "inference_patch_shape": list(features["patch_tokens"].shape),
            "rendered_rgb_shape": list(reconstruction["rendered_rgb"].shape),
            "rendered_expected_depth_shape": list(
                reconstruction["rendered_expected_depth"].shape
            ),
            "rendered_flow_shape": list(reconstruction["rendered_flow"].shape),
            "gradients": gradient_presence,
            "parameter_counts": unwrapped.parameter_counts(),
            "vit_forward_backward_peak_gib": vit_peak,
            "gaussian_training_step_peak_gib": gaussian_peak,
            "checkpoint_resume": True,
        }
        output = Path(args.output_root) / f"{mode}-result.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if context.is_main:
            print(json.dumps(result, indent=2), flush=True)
    finally:
        destroy_distributed(context)


if __name__ == "__main__":
    main()
