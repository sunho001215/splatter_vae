from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from preprocessing.see3d.visualization import (
    colorize_scalar,
    save_see3d_validation_grid,
)


def _uint8_rgb(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().clamp(0.0, 1.0).movedim(-3, -1).numpy()
    return (array * 255.0).round().astype(np.uint8)


def _flow_rgb(flow: torch.Tensor, maximum: float = 16.0) -> np.ndarray:
    value = flow.detach().float().cpu().numpy()
    x, y = value[0], value[1]
    angle = (np.arctan2(y, x) + np.pi) / (2.0 * np.pi)
    magnitude = np.clip(np.sqrt(x * x + y * y) / maximum, 0.0, 1.0)
    # HSV to RGB without an image-library dependency.
    sector = angle * 6.0
    index = np.floor(sector).astype(np.int32) % 6
    fraction = sector - np.floor(sector)
    p = np.zeros_like(magnitude)
    q = magnitude * (1.0 - fraction)
    t = magnitude * fraction
    candidates = (
        (magnitude, t, p),
        (q, magnitude, p),
        (p, magnitude, t),
        (p, q, magnitude),
        (t, p, magnitude),
        (magnitude, p, q),
    )
    rgb = np.zeros((*magnitude.shape, 3), dtype=np.float32)
    for sector_index, channels in enumerate(candidates):
        mask = index == sector_index
        for channel, values in enumerate(channels):
            rgb[..., channel][mask] = values[mask]
    return (rgb * 255.0).round().astype(np.uint8)


def _write_point_cloud(path: Path, xyz: np.ndarray, opacity: np.ndarray) -> None:
    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    alpha = np.clip(np.asarray(opacity).reshape(-1), 0.0, 1.0)
    intensity = (alpha * 255.0).round().astype(np.uint8)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write(
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for point, color in zip(points, intensity, strict=True):
            stream.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} {color} {color} {color}\n"
            )


def save_droid_validation_visualization(
    output_root: str | Path,
    step: int,
    payload: dict[str, Any],
) -> Path:
    destination = Path(output_root) / f"step-{int(step):08d}"
    destination.mkdir(parents=True, exist_ok=True)
    batch = payload["batch"]
    reconstruction = payload["reconstruction"]
    target_rgb = batch["target_rgb"][0]
    teacher_depth = batch["target_depth"][0, 2, 0, 0].numpy()
    rendered_depth = (
        reconstruction["rendered_expected_depth"][0, 2, 0, 0].detach().cpu().numpy()
    )
    teacher_flow = batch["target_flow"][0, 1, 0]
    rendered_flow = reconstruction["rendered_flow"][0, 1, 0]
    panels: list[tuple[str, np.ndarray]] = [
        ("global RGB input A", _uint8_rgb(target_rgb[2, 0])),
        ("local crop B", _uint8_rgb(batch["local_rgb_b"][0])),
        ("paired exterior A", _uint8_rgb(target_rgb[2, 0])),
        ("paired exterior B", _uint8_rgb(target_rgb[2, 1])),
        ("X-Lens metric depth", colorize_scalar(teacher_depth)),
        ("rendered expected depth", colorize_scalar(rendered_depth)),
        ("rendered RGB A", _uint8_rgb(reconstruction["rendered_rgb"][0, 2, 0])),
        ("rendered RGB B", _uint8_rgb(reconstruction["rendered_rgb"][0, 2, 1])),
        ("WAFT teacher flow", _flow_rgb(teacher_flow)),
        ("rendered Gaussian flow", _flow_rgb(rendered_flow)),
        (
            "rendered alpha",
            (
                reconstruction["rendered_alpha"][0, 2, 0, 0].detach().cpu().numpy()
                * 255
            ).astype(np.uint8),
        ),
    ]
    if bool(batch["synthetic_available"][0]):
        panels.extend(
            [
                ("See3D target", _uint8_rgb(batch["synthetic_rgb"][0])),
                (
                    "See3D confidence",
                    colorize_scalar(
                        batch["synthetic_confidence"][0, 0].numpy(),
                        minimum=0,
                        maximum=1,
                    ),
                ),
                (
                    "See3D geometry support",
                    batch["synthetic_geometry_supported"][0, 0].numpy().astype(np.uint8)
                    * 255,
                ),
            ]
        )
    grid_path = destination / "validation_grid.jpg"
    save_see3d_validation_grid(grid_path, panels, columns=4)
    anchor = reconstruction["gaussian_pc_anchor"]
    opacity = anchor["opacity"][0].detach().cpu().numpy()
    _write_point_cloud(
        destination / "gaussians_current_robot_base.ply",
        anchor["xyz"][0].detach().cpu().numpy(),
        opacity,
    )
    sequence_xyz = []
    for time_index, gaussian_pc in enumerate(reconstruction["gaussian_pc_sequence"]):
        xyz = gaussian_pc["xyz"][0].detach().cpu().numpy()
        sequence_xyz.append(xyz)
        _write_point_cloud(
            destination / f"gaussians_history_t{time_index}_robot_base.ply",
            xyz,
            opacity,
        )
    np.savez_compressed(
        destination / "temporal_gaussian_motion.npz",
        xyz=np.stack(sequence_xyz),
        delta_xyz_01=anchor["delta_xyz_01"][0].detach().cpu().numpy(),
        delta_xyz_12=anchor["delta_xyz_12"][0].detach().cpu().numpy(),
    )
    return grid_path
