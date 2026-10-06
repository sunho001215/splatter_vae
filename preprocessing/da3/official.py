from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

DA3_REPOSITORY_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_CHECKPOINT_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DA3_CHECKPOINT_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"


@dataclass(frozen=True)
class DA3InferenceConfig:
    process_resolution: int = 504
    process_resolution_method: str = "upper_bound_resize"
    align_to_input_extrinsic_scale: bool = True
    use_ray_pose: bool = False
    reference_view_strategy: str = "first"
    output_height: int = 180
    output_width: int = 320

    def __post_init__(self) -> None:
        if self.process_resolution <= 0:
            raise ValueError("DA3 processing resolution must be positive.")
        if self.process_resolution_method not in {
            "upper_bound_resize",
            "lower_bound_resize",
        }:
            raise ValueError("DA3 requires an aspect-preserving resize method.")
        if (self.output_height, self.output_width) != (180, 320):
            raise ValueError(
                "Cached DROID depth must remain on the 180x320 source grid."
            )

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


def _resize_valid_depth(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    value = torch.as_tensor(np.asarray(depth), dtype=torch.float32)[:, None]
    valid = torch.isfinite(value) & (value > 0.0)
    numerator = F.interpolate(
        torch.where(valid, value, torch.zeros_like(value)),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    )
    denominator = F.interpolate(
        valid.float(),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    )
    resized = torch.where(
        denominator > 1.0e-6,
        numerator / denominator.clamp_min(1.0e-6),
        torch.zeros_like(numerator),
    )
    return resized[:, 0].cpu().numpy().astype(np.float32, copy=False)


def _homogeneous_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    value = np.asarray(extrinsics, dtype=np.float64)
    if value.shape == (2, 4, 4):
        return value
    if value.shape == (2, 3, 4):
        output = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
        output[:, :3] = value
        return output
    raise ValueError(f"Two-view extrinsics have invalid shape {value.shape}.")


def two_view_baseline_alignment(
    predicted_w2c: np.ndarray,
    input_w2c: np.ndarray,
    *,
    return_aligned: bool = False,
    **_unused: Any,
) -> tuple[np.ndarray, np.ndarray, float] | tuple[
    np.ndarray, np.ndarray, float, np.ndarray
]:
    """Resolve DA3's rank-deficient two-camera Sim(3) scale by baseline.

    Rotation about a line through two camera centers is not identifiable.  The
    official posed path only consumes the scale when
    ``align_to_input_ext_scale=True`` and then replaces predicted poses with the
    supplied poses.  Consequently the exact, identifiable scale is the ratio of
    predicted to calibrated source-camera baselines; no arbitrary rotation is
    introduced.
    """

    predicted = _homogeneous_extrinsics(predicted_w2c)
    supplied = _homogeneous_extrinsics(input_w2c)
    predicted_centers = np.linalg.inv(predicted)[:, :3, 3]
    supplied_centers = np.linalg.inv(supplied)[:, :3, 3]
    predicted_baseline = float(np.linalg.norm(predicted_centers[1] - predicted_centers[0]))
    supplied_baseline = float(np.linalg.norm(supplied_centers[1] - supplied_centers[0]))
    if not np.isfinite(predicted_baseline) or predicted_baseline <= 1.0e-8:
        raise RuntimeError("DA3 predicted a degenerate two-view camera baseline.")
    if not np.isfinite(supplied_baseline) or supplied_baseline <= 1.0e-8:
        raise RuntimeError("DROID supplied a degenerate two-view camera baseline.")
    scale = predicted_baseline / supplied_baseline
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    if return_aligned:
        return rotation, translation, scale, supplied.copy()
    return rotation, translation, scale


class DA3DROIDTeacher:
    """Official refreshed metric any-view DA3 adapter for one synchronized pair.

    Independent timestamps are intentionally not batched together: each official
    inference group contains exactly Cam A and Cam B from the same raw timestep.
    Supplied extrinsics are canonical OpenCV/Colmap world-to-camera matrices.
    """

    def __init__(
        self,
        repository: str | Path,
        *,
        cache_dir: str | Path,
        device: torch.device | str = "cuda:0",
        config: DA3InferenceConfig | None = None,
    ) -> None:
        repository_path = Path(repository).expanduser().resolve()
        if not (repository_path / "src" / "depth_anything_3" / "api.py").is_file():
            raise FileNotFoundError(
                f"Official Depth Anything 3 source is missing: {repository_path}"
            )
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=DA3_CHECKPOINT_ID,
            revision=DA3_CHECKPOINT_REVISION,
            cache_dir=str(Path(cache_dir).expanduser().resolve()),
        )
        source = str(repository_path / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from depth_anything_3.api import DepthAnything3

        model = DepthAnything3.from_pretrained(str(Path(snapshot).resolve()))
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.model.requires_grad_(False)
        self.config = config or DA3InferenceConfig()
        self.repository_path = str(repository_path)
        self.checkpoint_path = str(Path(snapshot).resolve())

    @torch.inference_mode()
    def __call__(
        self,
        synchronized_rgb: np.ndarray,
        intrinsics: np.ndarray,
        w2c: np.ndarray,
    ) -> dict[str, Any]:
        images = np.asarray(synchronized_rgb, dtype=np.uint8)
        K = np.asarray(intrinsics, dtype=np.float32)
        extrinsics = np.asarray(w2c, dtype=np.float32)
        if images.shape != (2, 180, 320, 3):
            raise ValueError(
                f"DA3 expects synchronized RGB (2,180,320,3), got {images.shape}."
            )
        if K.shape != (2, 3, 3) or extrinsics.shape != (2, 4, 4):
            raise ValueError("DA3 expects two aligned K and OpenCV w2c matrices.")
        if not np.isfinite(K).all() or not np.isfinite(extrinsics).all():
            raise ValueError("DA3 camera geometry must be finite.")

        alignment_scales: list[float] = []

        def align_two_views(*args: Any, **kwargs: Any):
            result = two_view_baseline_alignment(*args, **kwargs)
            alignment_scales.append(float(result[2]))
            return result

        # Official Umeyama alignment raises on two camera centers because their
        # covariance has rank one. Patch only the API module's alignment call;
        # the checkpoint, input conditioning, model forward, and depth scaling
        # remain the official inference path.
        with patch(
            "depth_anything_3.api.align_poses_umeyama",
            side_effect=align_two_views,
        ):
            prediction = self.model.inference(
                [images[0], images[1]],
                extrinsics=extrinsics,
                intrinsics=K,
                align_to_input_ext_scale=self.config.align_to_input_extrinsic_scale,
                use_ray_pose=self.config.use_ray_pose,
                ref_view_strategy=self.config.reference_view_strategy,
                process_res=self.config.process_resolution,
                process_res_method=self.config.process_resolution_method,
            )
        if len(alignment_scales) != 1:
            raise RuntimeError("DA3 did not execute exactly one two-view scale alignment.")
        depth = np.asarray(prediction.depth, dtype=np.float32)
        if depth.ndim != 3 or depth.shape[0] != 2:
            raise RuntimeError(f"DA3 returned unexpected depth shape {depth.shape}.")
        depth = _resize_valid_depth(
            depth, self.config.output_height, self.config.output_width
        )
        metric_flag = bool(np.asarray(prediction.is_metric).item())
        if not metric_flag:
            raise RuntimeError(
                "The selected DA3 checkpoint did not report metric output."
            )
        predicted_extrinsics = np.asarray(prediction.extrinsics, dtype=np.float32)
        if predicted_extrinsics.shape == (2, 3, 4):
            expected = extrinsics[:, :3]
        elif predicted_extrinsics.shape == (2, 4, 4):
            expected = extrinsics
        else:
            raise RuntimeError(
                f"DA3 returned unexpected extrinsic shape {predicted_extrinsics.shape}."
            )
        extrinsic_error = float(np.abs(predicted_extrinsics - expected).max())
        if self.config.align_to_input_extrinsic_scale and extrinsic_error > 5.0e-4:
            raise RuntimeError(
                f"DA3 output was not aligned to supplied w2c geometry (max error {extrinsic_error})."
            )
        finite = np.isfinite(depth) & (depth > 0.0)
        if float(finite.mean()) < 0.50:
            raise RuntimeError(
                f"DA3 produced only {float(finite.mean()):.3f} finite positive depth."
            )
        baseline_input = float(
            np.linalg.norm(
                np.linalg.inv(extrinsics[1])[:3, 3]
                - np.linalg.inv(extrinsics[0])[:3, 3]
            )
        )
        aligned_4x4 = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy()
        aligned_4x4[:, : predicted_extrinsics.shape[-2], :4] = predicted_extrinsics
        baseline_output = float(
            np.linalg.norm(
                np.linalg.inv(aligned_4x4[1])[:3, 3]
                - np.linalg.inv(aligned_4x4[0])[:3, 3]
            )
        )
        return {
            "metric_depth": depth,
            "validity": finite,
            "processed_resolution": tuple(
                int(value) for value in prediction.depth.shape[-2:]
            ),
            "output_resolution": (self.config.output_height, self.config.output_width),
            "is_metric": metric_flag,
            "scale_factor": None
            if prediction.scale_factor is None
            else float(prediction.scale_factor),
            "pose_alignment_scale": alignment_scales[0],
            "extrinsic_max_abs_error": extrinsic_error,
            "input_baseline_m": baseline_input,
            "output_baseline_m": baseline_output,
            "processed_intrinsics": np.asarray(prediction.intrinsics, dtype=np.float32),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "repository": "ByteDance-Seed/Depth-Anything-3",
            "repository_revision": DA3_REPOSITORY_REVISION,
            "model_id": DA3_CHECKPOINT_ID,
            "checkpoint_revision": DA3_CHECKPOINT_REVISION,
            "checkpoint_path": self.checkpoint_path,
            "camera_extrinsics": "opencv_colmap_w2c",
            "inference_group": "synchronized_cam_a_cam_b_only",
            "two_view_pose_scale_alignment": "predicted_baseline_over_input_baseline",
            **self.config.metadata(),
        }
