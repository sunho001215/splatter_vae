from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from dataset.droid.cache import HDF5CacheReader
from dataset.droid.dataset import DROIDLogicalDataset, droid_collate
from models.gaussian.parameterization import WorldSpaceGaussianParameterization
from models.training.distributed import move_to_device
from models.training.losses import cross_view_info_nce
from models.training.reconstruction import compute_droid_reconstruction
from models.training.visualization import save_droid_validation_visualization
from preprocessing.common import verify_cuda_device
from scripts.train_droid import (
    _dataset_config,
    _model_and_renderer,
    _train_config,
    _validate_fixed_pipeline_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-DROID forward/backward and Gaussian-rendering smoke test."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--flow-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    _validate_fixed_pipeline_contract(values)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly the authorized GPU before this smoke test.")
    device = torch.device("cuda:0")
    diagnostics = verify_cuda_device()
    dataset_config = replace(
        _dataset_config(values, "train"),
        calibration_manifest=str(Path(args.manifest).expanduser().resolve()),
        require_depth_cache=False,
        require_flow_cache=True,
    )
    dataset = DROIDLogicalDataset(
        dataset_config,
        flow_cache=HDF5CacheReader(args.flow_cache),
    )
    item = dataset[int(args.dataset_index)]
    cpu_batch = droid_collate([item])
    batch = move_to_device(cpu_batch, device)
    model, splatter = _model_and_renderer(values)
    model.to(device).train()
    parameterization = WorldSpaceGaussianParameterization(splatter).to(device)
    train_config = _train_config(values, None)
    background = torch.zeros(3, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(
            batch["representation_histories"],
            batch["representation_flows"],
            batch["representation_validity"],
        )
        contrastive, contrast_metrics = cross_view_info_nce(
            prediction["projected_cls_by_view"],
            train_config.contrastive_temperature,
        )
    reconstruction = compute_droid_reconstruction(
        parameterization,
        splatter,
        prediction,
        batch,
        train_config,
        motion_translation_max=model.motion_translation_max,
        background_color=background,
        return_renders=True,
    )
    total = reconstruction["loss"] + train_config.contrastive_weight * contrastive
    if not torch.isfinite(total):
        raise FloatingPointError(f"Real-DROID loss is non-finite: {total}")
    total.backward()
    gradients = {
        "encoder": any(parameter.grad is not None for parameter in model.encoder.parameters()),
        "gaussian_decoder": any(
            parameter.grad is not None
            for parameter in model.gaussian_decoder.parameters()
        ),
        "contrastive_projector": any(
            parameter.grad is not None
            for parameter in model.contrastive_projector.parameters()
        ),
    }
    if not all(gradients.values()):
        raise RuntimeError(f"Real-DROID smoke test missed gradients: {gradients}")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    grid = save_droid_validation_visualization(
        output_root / "visualization",
        0,
        {
            "batch": cpu_batch,
            "reconstruction": reconstruction,
            "prediction": prediction,
        },
    )
    result = {
        "gpu": diagnostics,
        "episode_id": item["episode_id"],
        "history_indices": item["history_indices"].tolist(),
        "camera_serials": list(item["camera_serials"]),
        "crop_metadata": {
            key: value.tolist() for key, value in item["crop_metadata"].items()
        },
        "model_input_shape": list(batch["representation_histories"].shape),
        "loss": float(total.detach()),
        "contrastive_loss": float(contrastive.detach()),
        "positive_cosine_similarity": float(
            contrast_metrics["positive_cosine_similarity"].detach()
        ),
        "rgb_l1_loss": float(reconstruction["rgb_l1_loss"].detach()),
        "flow_loss": float(reconstruction["flow_loss"].detach()),
        "metric_depth_loss": float(reconstruction["metric_depth_loss"].detach()),
        "scale_invariant_depth_loss": float(
            reconstruction["scale_invariant_depth_loss"].detach()
        ),
        "rendered_rgb_shape": list(reconstruction["rendered_rgb"].shape),
        "rendered_expected_depth_shape": list(
            reconstruction["rendered_expected_depth"].shape
        ),
        "rendered_flow_shape": list(reconstruction["rendered_flow"].shape),
        "rendered_visible_pixel_fraction": float(
            reconstruction["rendered_visible_pixel_fraction"].detach()
        ),
        "out_of_frustum_fraction": float(
            reconstruction["out_of_frustum_fraction"].detach()
        ),
        "gradients": gradients,
        "parameter_counts": model.parameter_counts(),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "visualization": str(grid),
        "xlens_depth_available": False,
    }
    result_path = output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
