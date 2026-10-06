from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


def _uint8_rgb(value: torch.Tensor) -> np.ndarray:
    tensor = value.detach().float().cpu()
    if tensor.shape[-3] != 3:
        raise ValueError("RGB visualization tensors must have three channels.")
    if tensor.max() <= 1.0 + 1.0e-6:
        tensor = tensor * 255.0
    return (
        tensor.clamp(0.0, 255.0)
        .movedim(-3, -1)
        .numpy()
        .round()
        .astype(np.uint8)
    )


def _mask_rgb(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = np.asarray(value.detach().cpu() if torch.is_tensor(value) else value)
    array = np.squeeze(array).astype(bool)
    return np.repeat((array.astype(np.uint8) * 255)[..., None], 3, axis=-1)


def _colorize_scalar(
    value: torch.Tensor | np.ndarray,
    *,
    minimum: float,
    maximum: float,
    validity: torch.Tensor | np.ndarray | None = None,
) -> np.ndarray:
    array = np.asarray(value.detach().float().cpu() if torch.is_tensor(value) else value)
    array = np.squeeze(array).astype(np.float32)
    normalized = np.clip(
        (array - float(minimum)) / max(float(maximum - minimum), 1.0e-8), 0, 1
    )
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    valid = np.isfinite(array)
    if validity is not None:
        mask = validity.detach().cpu().numpy() if torch.is_tensor(validity) else validity
        valid &= np.squeeze(np.asarray(mask)).astype(bool)
    rgb[~valid] = 0.0
    return (rgb * 255.0).round().astype(np.uint8)


def _flow_rgb(
    flow: torch.Tensor,
    validity: torch.Tensor | None = None,
    maximum: float = 64.0,
) -> np.ndarray:
    value = flow.detach().float().cpu().numpy()
    x, y = value[0], value[1]
    hue = (np.arctan2(y, x) + np.pi) / (2.0 * np.pi)
    magnitude = np.clip(np.sqrt(x * x + y * y) / float(maximum), 0.0, 1.0)
    sector = hue * 6.0
    index = np.floor(sector).astype(np.int32) % 6
    fraction = sector - np.floor(sector)
    zero = np.zeros_like(magnitude)
    q = magnitude * (1.0 - fraction)
    t = magnitude * fraction
    candidates = (
        (magnitude, t, zero),
        (q, magnitude, zero),
        (zero, magnitude, t),
        (zero, q, magnitude),
        (t, zero, magnitude),
        (magnitude, zero, q),
    )
    rgb = np.zeros((*magnitude.shape, 3), dtype=np.float32)
    for sector_index, channels in enumerate(candidates):
        mask = index == sector_index
        for channel, values in enumerate(channels):
            rgb[..., channel][mask] = values[mask]
    if validity is not None:
        valid = np.squeeze(validity.detach().cpu().numpy()).astype(bool)
        rgb[~valid] = 0.0
    return (rgb * 255.0).round().astype(np.uint8)


def _save_grid(
    path: Path,
    panels: list[tuple[str, np.ndarray]],
    *,
    columns: int,
) -> Path:
    if not panels:
        raise ValueError("A visualization grid requires at least one panel.")
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared: list[tuple[str, Image.Image]] = []
    for title, array in panels:
        image = np.asarray(array)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Panel {title!r} is not HxWx3: {image.shape}.")
        prepared.append((title, Image.fromarray(image.astype(np.uint8), mode="RGB")))
    cell_width = max(image.width for _, image in prepared)
    cell_height = max(image.height for _, image in prepared) + 24
    rows = (len(prepared) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(prepared):
        row, column = divmod(index, columns)
        x = column * cell_width + (cell_width - image.width) // 2
        y = row * cell_height + 24
        canvas.paste(image, (x, y))
        draw.text((column * cell_width + 4, row * cell_height + 5), title, fill="white")
    canvas.save(path, quality=92)
    return path


def _padded_crop_overlay(
    raw_rgb: torch.Tensor,
    *,
    crop_size: int,
    crop_x: int,
    crop_y: int,
    center_x: int,
    center_y: int,
    fallback: bool,
) -> np.ndarray:
    padded = np.zeros((320, 320, 3), dtype=np.uint8)
    padded[:] = np.asarray((124, 116, 104), dtype=np.uint8)
    padded[70:250] = _uint8_rgb(raw_rgb)
    image = Image.fromarray(padded, mode="RGB")
    draw = ImageDraw.Draw(image)
    color = (255, 190, 0) if fallback else (255, 40, 40)
    draw.rectangle(
        (crop_x, crop_y, crop_x + crop_size - 1, crop_y + crop_size - 1),
        outline=color,
        width=3,
    )
    draw.line(
        (center_x - 7, center_y, center_x + 7, center_y),
        fill=(0, 255, 80),
        width=2,
    )
    draw.line(
        (center_x, center_y - 7, center_x, center_y + 7),
        fill=(0, 255, 80),
        width=2,
    )
    return np.asarray(image)


def _write_point_cloud(path: Path, xyz: np.ndarray, opacity: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{color} {color} {color}\n"
            )


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        selected = value.detach().cpu()
        return selected.item() if selected.dim() == 0 else selected.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_droid_validation_visualization(
    output_root: str | Path,
    step: int,
    payload: dict[str, Any],
    *,
    num_samples: int = 2,
    depth_range_m: tuple[float, float] = (0.25, 5.0),
) -> dict[str, Path]:
    """Render diagnostics from cached Stage-0 fields without invoking teachers."""

    root = Path(output_root)
    batch = payload["batch"]
    reconstruction = payload["reconstruction"]
    batch_size = int(batch["raw_histories"].shape[0])
    sample_count = min(int(num_samples), batch_size)
    paths: dict[str, Path] = {}
    for sample in range(sample_count):
        sample_name = f"sample-{sample:02d}"
        raw = batch["raw_histories"][sample]
        input_panels = [
            (f"Cam {camera} t{time} cached RGB", _uint8_rgb(raw[camera, time]))
            for camera in range(2)
            for time in range(3)
        ]
        paths[f"val/input/{sample_name}"] = _save_grid(
            root / "input" / f"step-{step:08d}-{sample_name}.jpg",
            input_panels,
            columns=3,
        )

        flow_panels: list[tuple[str, np.ndarray]] = []
        native_flow = batch["native_megaflow_flow"][sample]
        native_validity = batch["native_megaflow_validity"][sample]
        for camera in range(2):
            for pair, name in enumerate(("t0->t1", "t1->t2")):
                flow = native_flow[pair, camera]
                valid = native_validity[pair, camera]
                flow_panels.extend(
                    (
                        (
                            f"Cam {camera} cached MegaFlow {name}",
                            _flow_rgb(flow, valid),
                        ),
                        (
                            f"Cam {camera} {name} magnitude",
                            _colorize_scalar(
                                torch.linalg.vector_norm(flow.float(), dim=0),
                                minimum=0.0,
                                maximum=64.0,
                                validity=valid,
                            ),
                        ),
                    )
                )
            flow_panels.extend(
                (
                    (
                        f"Cam {camera} middle-grid aggregate",
                        _colorize_scalar(
                            batch["aggregate_motion"][sample, camera],
                            minimum=0.0,
                            maximum=64.0,
                        ),
                    ),
                    (
                        f"Cam {camera} smoothed crop score",
                        _colorize_scalar(
                            batch["smoothed_motion"][sample, camera],
                            minimum=0.0,
                            maximum=64.0,
                        ),
                    ),
                )
            )
        paths[f"val/megaflow/{sample_name}"] = _save_grid(
            root / "megaflow" / f"step-{step:08d}-{sample_name}.jpg",
            flow_panels,
            columns=4,
        )

        crop = batch["crop_metadata"]
        crop_panels: list[tuple[str, np.ndarray]] = []
        for camera in range(2):
            size = int(crop["crop_size"][sample, camera])
            x0 = int(crop["crop_x0"][sample, camera])
            y0 = int(crop["crop_y0"][sample, camera])
            cx = int(crop["crop_center_x"][sample, camera])
            cy = int(crop["crop_center_y"][sample, camera])
            peak = float(crop["flow_peak_value"][sample, camera])
            fallback = bool(crop["low_motion_fallback_used"][sample, camera])
            for time in range(3):
                crop_panels.append(
                    (
                        f"Cam {camera} t{time} S={size} c=({cx},{cy}) peak={peak:.2f}"
                        + (" FALLBACK" if fallback else ""),
                        _padded_crop_overlay(
                            raw[camera, time],
                            crop_size=size,
                            crop_x=x0,
                            crop_y=y0,
                            center_x=cx,
                            center_y=cy,
                            fallback=fallback,
                        ),
                    )
                )
            crop_panels.extend(
                (
                    (
                        f"Cam {camera} cropped middle RGB",
                        _uint8_rgb(batch["target_rgb"][sample, 1, camera]),
                    ),
                    (
                        f"Cam {camera} geometric validity",
                        _mask_rgb(
                            batch["target_image_validity"][sample, 1, camera]
                        ),
                    ),
                )
            )
        paths[f"val/crop/{sample_name}"] = _save_grid(
            root / "crop" / f"step-{step:08d}-{sample_name}.jpg",
            crop_panels,
            columns=5,
        )

        for camera in range(2):
            depth_panels: list[tuple[str, np.ndarray]] = []
            reconstruction_panels: list[tuple[str, np.ndarray]] = []
            for time in range(3):
                teacher_depth = batch["target_depth"][sample, time, camera, 0]
                teacher_valid = batch["target_depth_validity"][
                    sample, time, camera, 0
                ]
                rendered_depth = reconstruction["rendered_expected_depth"][
                    sample, time, camera, 0
                ]
                rendered_valid = (
                    reconstruction["rendered_alpha"][sample, time, camera, 0] > 0.01
                )
                depth_panels.extend(
                    (
                        (
                            f"t{time} cached DA3",
                            _colorize_scalar(
                                teacher_depth,
                                minimum=depth_range_m[0],
                                maximum=depth_range_m[1],
                                validity=teacher_valid,
                            ),
                        ),
                        (
                            f"t{time} GS expected depth",
                            _colorize_scalar(
                                rendered_depth,
                                minimum=depth_range_m[0],
                                maximum=depth_range_m[1],
                                validity=rendered_valid,
                            ),
                        ),
                        (
                            f"t{time} absolute depth error",
                            _colorize_scalar(
                                (rendered_depth - teacher_depth).abs(),
                                minimum=0.0,
                                maximum=1.0,
                                validity=teacher_valid & rendered_valid,
                            ),
                        ),
                    )
                )
                target_rgb = batch["target_rgb"][sample, time, camera]
                rendered_rgb = reconstruction["rendered_rgb"][
                    sample, time, camera
                ]
                reconstruction_panels.extend(
                    (
                        (f"t{time} real RGB", _uint8_rgb(target_rgb)),
                        (f"t{time} GS RGB", _uint8_rgb(rendered_rgb)),
                        (
                            f"t{time} RGB error",
                            _colorize_scalar(
                                (rendered_rgb - target_rgb).abs().mean(dim=0),
                                minimum=0.0,
                                maximum=0.5,
                            ),
                        ),
                    )
                )
            paths[f"val/da3/{sample_name}/cam-{camera}"] = _save_grid(
                root
                / "da3"
                / f"step-{step:08d}-{sample_name}-cam-{camera}.jpg",
                depth_panels,
                columns=3,
            )
            paths[f"val/reconstruction/{sample_name}/cam-{camera}"] = _save_grid(
                root
                / "reconstruction"
                / f"step-{step:08d}-{sample_name}-cam-{camera}.jpg",
                reconstruction_panels,
                columns=3,
            )

            rendered_flow_panels: list[tuple[str, np.ndarray]] = []
            for pair, name in enumerate(("t0->t1", "t1->t2")):
                target = batch["target_flow"][sample, pair, camera]
                target_valid = batch["target_flow_validity"][
                    sample, pair, camera
                ]
                rendered = reconstruction["rendered_flow"][
                    sample, pair, camera
                ]
                rendered_flow_panels.extend(
                    (
                        (f"{name} cached MegaFlow", _flow_rgb(target, target_valid)),
                        (f"{name} GS flow", _flow_rgb(rendered)),
                        (
                            f"{name} endpoint error",
                            _colorize_scalar(
                                torch.linalg.vector_norm(rendered - target, dim=0),
                                minimum=0.0,
                                maximum=32.0,
                                validity=target_valid,
                            ),
                        ),
                        (f"{name} usable pixels", _mask_rgb(target_valid)),
                    )
                )
            paths[f"val/flow_render/{sample_name}/cam-{camera}"] = _save_grid(
                root
                / "flow_render"
                / f"step-{step:08d}-{sample_name}-cam-{camera}.jpg",
                rendered_flow_panels,
                columns=4,
            )

        if "novel_rgb" in batch and "novel_rendered_rgb" in reconstruction:
            for time in range(3):
                lager_panels: list[tuple[str, np.ndarray]] = []
                for view in range(4):
                    teacher = batch["novel_rgb"][sample, time, view]
                    rendered = reconstruction["novel_rendered_rgb"][
                        sample, time, view
                    ]
                    support = batch["novel_support_mask"][sample, time, view]
                    alpha = float(
                        batch["novel_pose_metadata"]["alpha"][
                            sample, time, view
                        ]
                    )
                    lager_panels.extend(
                        (
                            (
                                f"view {view} Lager target alpha={alpha:.3f}",
                                _uint8_rgb(teacher),
                            ),
                            (f"view {view} GS render", _uint8_rgb(rendered)),
                            (
                                f"view {view} RGB error",
                                _colorize_scalar(
                                    (teacher - rendered).abs().mean(dim=0),
                                    minimum=0.0,
                                    maximum=0.5,
                                ),
                            ),
                            (f"view {view} support", _mask_rgb(support)),
                        )
                    )
                paths[f"val/lagernvs/{sample_name}/t{time}"] = _save_grid(
                    root
                    / "lagernvs"
                    / f"step-{step:08d}-{sample_name}-t{time}.jpg",
                    lager_panels,
                    columns=4,
                )
            metadata_path = (
                root / "lagernvs" / f"step-{step:08d}-{sample_name}-poses.json"
            )
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            pose_metadata = {
                key: value[sample]
                for key, value in batch["novel_pose_metadata"].items()
            }
            metadata_path.write_text(
                json.dumps(_jsonable(pose_metadata), indent=2) + "\n",
                encoding="utf-8",
            )

        if sample == 0:
            sequence = reconstruction["gaussian_pc_sequence"]
            opacity = reconstruction["gaussian_pc_anchor"]["opacity"][0].numpy()
            geometry_root = root / "geometry" / f"step-{step:08d}"
            # Full static cloud remains t0-only. Motion is retained separately.
            _write_point_cloud(
                geometry_root / "gaussians_t0_robot_base.ply",
                sequence[0]["xyz"][0].numpy(),
                opacity,
            )
            tracking_root = root / "tracking" / f"step-{step:08d}"
            tracking_root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                tracking_root / "t0_to_t1_to_t2_gaussian_tracking.npz",
                xyz=np.stack([state["xyz"][0].numpy() for state in sequence]),
                delta_xyz_01=reconstruction["gaussian_pc_anchor"][
                    "delta_xyz_01"
                ][0].numpy(),
                delta_xyz_12=reconstruction["gaussian_pc_anchor"][
                    "delta_xyz_12"
                ][0].numpy(),
            )
            geometry_root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                geometry_root / "camera_frames_t0.npz",
                real_exterior_c2w=batch["target_c2w"][0, 0].numpy(),
                lager_target_c2w=batch["novel_c2w"][0, 0].numpy(),
                robot_base_frame=np.eye(4, dtype=np.float32),
            )
    return paths
