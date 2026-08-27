from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from preprocessing.lagernvs.camera import lager_to_display_plane


def _uint8_rgb(value: torch.Tensor) -> np.ndarray:
    tensor = value.detach().float().cpu()
    if tensor.shape[-3] != 3:
        raise ValueError("RGB visualization tensors must have three channels.")
    if tensor.max() <= 1.0 + 1.0e-6:
        tensor = tensor * 255.0
    return tensor.clamp(0.0, 255.0).movedim(-3, -1).numpy().round().astype(np.uint8)


def _colorize_scalar(
    value: torch.Tensor | np.ndarray,
    *,
    minimum: float,
    maximum: float,
    validity: torch.Tensor | np.ndarray | None = None,
) -> np.ndarray:
    array = np.asarray(value.detach().float().cpu() if torch.is_tensor(value) else value)
    array = np.squeeze(array).astype(np.float32)
    normalized = np.clip((array - float(minimum)) / max(float(maximum - minimum), 1e-8), 0, 1)
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


def _flow_rgb(flow: torch.Tensor, maximum: float = 64.0) -> np.ndarray:
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
    return (rgb * 255.0).round().astype(np.uint8)


def _save_grid(
    path: Path,
    panels: list[tuple[str, np.ndarray]],
    *,
    columns: int = 3,
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
    draw.line((center_x - 7, center_y, center_x + 7, center_y), fill=(0, 255, 80), width=2)
    draw.line((center_x, center_y - 7, center_x, center_y + 7), fill=(0, 255, 80), width=2)
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


def _lager_display(
    image: torch.Tensor, K: torch.Tensor, *, threshold: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    display, _display_K, validity = lager_to_display_plane(image.float(), K.float())
    if threshold:
        output = (display[0] >= 0.5).numpy().astype(np.uint8) * 255
        return np.repeat(output[..., None], 3, axis=-1), validity[0].numpy()
    return _uint8_rgb(display), validity[0].numpy()


def _tensor_metadata(metadata: dict[str, Any], sample: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in metadata.items():
        if torch.is_tensor(value):
            selected = value[sample]
            output[key] = selected.tolist() if selected.dim() else selected.item()
        elif isinstance(value, list):
            output[key] = value[sample] if len(value) > sample else value
        else:
            output[key] = value
    return output


def save_droid_validation_visualization(
    output_root: str | Path,
    step: int,
    payload: dict[str, Any],
    *,
    num_samples: int = 2,
    depth_range_m: tuple[float, float] = (0.25, 5.0),
) -> dict[str, Path]:
    """Save deterministic validation panels while reusing online teacher outputs."""

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
            (f"Cam {camera} t{time} raw 320x180", _uint8_rgb(raw[camera, time]))
            for camera in range(2)
            for time in range(3)
        ]
        input_path = root / "input" / f"step-{step:08d}-{sample_name}.jpg"
        paths[f"val/input/{sample_name}"] = _save_grid(
            input_path, input_panels, columns=3
        )

        flow_panels: list[tuple[str, np.ndarray]] = []
        for camera in range(2):
            native_flow = batch["native_memfof_flow"][sample, camera]
            for direction, name in enumerate(("middle->previous", "middle->next")):
                magnitude = torch.linalg.vector_norm(native_flow[direction].float(), dim=0)
                flow_panels.extend(
                    [
                        (f"Cam {camera} {name} MEMFOF", _flow_rgb(native_flow[direction])),
                        (
                            f"Cam {camera} {name} magnitude",
                            _colorize_scalar(magnitude, minimum=0.0, maximum=64.0),
                        ),
                    ]
                )
            aggregate = batch["aggregate_motion"][sample, camera]
            smoothed = batch["smoothed_motion"][sample, camera]
            flow_panels.extend(
                [
                    (
                        f"Cam {camera} aggregate max",
                        _colorize_scalar(aggregate, minimum=0.0, maximum=64.0),
                    ),
                    (
                        f"Cam {camera} smoothed motion",
                        _colorize_scalar(smoothed, minimum=0.0, maximum=64.0),
                    ),
                ]
            )
        memfof_path = root / "memfof" / f"step-{step:08d}-{sample_name}.jpg"
        paths[f"val/memfof/{sample_name}"] = _save_grid(
            memfof_path, flow_panels, columns=4
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
            crop_panels.append(
                (
                    f"Cam {camera} final encoder crop 224x224",
                    _uint8_rgb(batch["target_rgb"][sample, 2, camera]),
                )
            )
        crop_path = root / "crop" / f"step-{step:08d}-{sample_name}.jpg"
        paths[f"val/crop/{sample_name}"] = _save_grid(crop_path, crop_panels, columns=4)

        depth_panels: list[tuple[str, np.ndarray]] = []
        for camera in range(2):
            teacher_depth = batch["target_depth"][sample, 2, camera, 0]
            teacher_valid = batch["target_depth_validity"][sample, 2, camera, 0]
            rendered_depth = reconstruction["rendered_expected_depth"][sample, 2, camera, 0]
            rendered_valid = reconstruction["rendered_alpha"][sample, 2, camera, 0] > 0.01
            error = (rendered_depth - teacher_depth).abs()
            depth_panels.extend(
                [
                    (f"Cam {camera} crop RGB", _uint8_rgb(batch["target_rgb"][sample, 2, camera])),
                    (
                        f"Cam {camera} X-Lens metric depth",
                        _colorize_scalar(
                            teacher_depth,
                            minimum=depth_range_m[0],
                            maximum=depth_range_m[1],
                            validity=teacher_valid,
                        ),
                    ),
                    (
                        f"Cam {camera} X-Lens confidence",
                        _colorize_scalar(
                            batch["target_depth_confidence"][sample, 2, camera, 0],
                            minimum=0.0,
                            maximum=24.0,
                            validity=teacher_valid,
                        ),
                    ),
                    (
                        f"Cam {camera} X-Lens validity",
                        teacher_valid.numpy().astype(np.uint8) * 255,
                    ),
                    (
                        f"Cam {camera} GS expected depth",
                        _colorize_scalar(
                            rendered_depth,
                            minimum=depth_range_m[0],
                            maximum=depth_range_m[1],
                            validity=rendered_valid,
                        ),
                    ),
                    (
                        f"Cam {camera} absolute depth error",
                        _colorize_scalar(
                            error,
                            minimum=0.0,
                            maximum=1.0,
                            validity=teacher_valid & rendered_valid,
                        ),
                    ),
                ]
            )
        depth_path = root / "depth" / f"step-{step:08d}-{sample_name}.jpg"
        paths[f"val/xlens/{sample_name}"] = _save_grid(depth_path, depth_panels, columns=3)

        reconstruction_panels: list[tuple[str, np.ndarray]] = []
        for camera in range(2):
            target_rgb = batch["target_rgb"][sample, 2, camera]
            rendered_rgb = reconstruction["rendered_rgb"][sample, 2, camera]
            rgb_error = (rendered_rgb - target_rgb).abs().mean(dim=0)
            for direction, direction_name in enumerate(("backward", "forward")):
                teacher_flow = batch["target_flow"][sample, direction, camera]
                rendered_flow = reconstruction["rendered_flow"][sample, direction, camera]
                flow_error = torch.linalg.vector_norm(
                    rendered_flow - teacher_flow, dim=0
                )
                reconstruction_panels.extend(
                    [
                        (f"Cam {camera} MEMFOF {direction_name}", _flow_rgb(teacher_flow)),
                        (f"Cam {camera} GS flow {direction_name}", _flow_rgb(rendered_flow)),
                        (
                            f"Cam {camera} flow EPE {direction_name}",
                            _colorize_scalar(flow_error, minimum=0.0, maximum=32.0),
                        ),
                    ]
                )
            reconstruction_panels.extend(
                [
                    (f"Cam {camera} target RGB", _uint8_rgb(target_rgb)),
                    (f"Cam {camera} GS RGB", _uint8_rgb(rendered_rgb)),
                    (
                        f"Cam {camera} RGB absolute error",
                        _colorize_scalar(rgb_error, minimum=0.0, maximum=0.5),
                    ),
                    (
                        f"Cam {camera} alpha / visibility",
                        (reconstruction["rendered_alpha"][sample, 2, camera, 0].numpy() * 255)
                        .clip(0, 255)
                        .astype(np.uint8),
                    ),
                ]
            )
        reconstruction_path = (
            root / "reconstruction" / f"step-{step:08d}-{sample_name}.jpg"
        )
        paths[f"val/reconstruction/{sample_name}"] = _save_grid(
            reconstruction_path, reconstruction_panels, columns=4
        )

        if "novel_rgb" in batch and "novel_rendered_rgb" in reconstruction:
            canonical_K = batch["novel_K"][sample]
            teacher_canonical = batch["novel_rgb"][sample]
            rendered_canonical = reconstruction["novel_rendered_rgb"][sample, 0]
            canonical_error = (teacher_canonical - rendered_canonical).abs()
            support = batch["novel_support_mask"][sample].float()
            teacher_display, _ = _lager_display(teacher_canonical, canonical_K)
            rendered_display, _ = _lager_display(rendered_canonical, canonical_K)
            error_display, _ = _lager_display(canonical_error, canonical_K)
            support_display, _ = _lager_display(support, canonical_K, threshold=True)
            pose_metadata = _tensor_metadata(batch["novel_pose_metadata"], sample)
            title = (
                f"alpha={pose_metadata.get('alpha', 0):.3f} "
                f"coverage={pose_metadata.get('source_coverage', 0):.3f} "
                f"jitter={pose_metadata.get('translation_perturbation_magnitude', 0):.3f}m/"
                f"{pose_metadata.get('rotation_perturbation_degrees', 0):.2f}deg"
            )
            lager_panels = [
                ("Source Cam A canonical", _uint8_rgb(batch["novel_source_rgb"][sample, 0])),
                ("Source Cam B canonical", _uint8_rgb(batch["novel_source_rgb"][sample, 1])),
                (f"LagerNVS canonical 256 ({title})", _uint8_rgb(teacher_canonical)),
                ("GS canonical novel render", _uint8_rgb(rendered_canonical)),
                ("Canonical RGB absolute error", _uint8_rgb(canonical_error)),
                ("Canonical source-support mask", support[0].numpy().astype(np.uint8) * 255),
                ("LagerNVS natural 320x180", teacher_display),
                ("GS natural 320x180", rendered_display),
                ("RGB error natural 320x180", error_display),
                ("Support natural 320x180", support_display),
            ]
            lager_path = root / "lagernvs" / f"step-{step:08d}-{sample_name}.jpg"
            paths[f"val/lagernvs/{sample_name}"] = _save_grid(
                lager_path, lager_panels, columns=3
            )
            metadata_path = (
                root / "lagernvs" / f"step-{step:08d}-{sample_name}-pose.json"
            )
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(pose_metadata, indent=2), encoding="utf-8")

        if sample == 0:
            sequence = reconstruction["gaussian_pc_sequence"]
            opacity = reconstruction["gaussian_pc_anchor"]["opacity"][0].numpy()
            geometry_root = root / "geometry" / f"step-{step:08d}"
            # The only full static cloud is chronological t0.  t1/t2 are retained
            # solely in the compact tracking artifact below.
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
                delta_xyz_01=reconstruction["gaussian_pc_anchor"]["delta_xyz_01"][0].numpy(),
                delta_xyz_12=reconstruction["gaussian_pc_anchor"]["delta_xyz_12"][0].numpy(),
            )
            camera_values: dict[str, np.ndarray] = {
                "source_c2w": batch["raw_c2w"][0].numpy(),
            }
            if "novel_c2w" in batch:
                camera_values["virtual_c2w"] = batch["novel_c2w"][0].numpy()
                camera_values["interpolated_base_c2w"] = batch["novel_base_c2w"][0].numpy()
            np.savez_compressed(geometry_root / "camera_frames_t0.npz", **camera_values)
    return paths
