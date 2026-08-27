from __future__ import annotations

import sys
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

XLENS_PATCH_SIZE = 14
XLENS_PREPROCESSING_VERSION = "droid-online-pinhole-symmetric-pad14-v2"
XLENS_CHECKPOINT_SHA256 = (
    "266a0340b53e5cb996cc613a1b0c5966b5bcaeee1ec7c4431e4fc6e7d1e58a0c"
)
XLENS_NATIVE_HEIGHT = 180
XLENS_NATIVE_WIDTH = 320


def pad_pinhole_scene_to_patch_multiple(
    images: Sequence[np.ndarray],
    intrinsics: Sequence[np.ndarray],
    *,
    patch_size: int = XLENS_PATCH_SIZE,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int, int, int]]:
    if not images or len(images) != len(intrinsics):
        raise ValueError("X-Lens requires aligned nonempty image and intrinsic lists.")
    height, width = np.asarray(images[0]).shape[:2]
    target_height = ((height + patch_size - 1) // patch_size) * patch_size
    target_width = ((width + patch_size - 1) // patch_size) * patch_size
    pad_y = target_height - height
    pad_x = target_width - width
    top, bottom = pad_y // 2, pad_y - pad_y // 2
    left, right = pad_x // 2, pad_x - pad_x // 2
    padded_images: list[np.ndarray] = []
    padded_intrinsics: list[np.ndarray] = []
    for image, K in zip(images, intrinsics, strict=True):
        value = np.asarray(image, dtype=np.uint8)
        if value.shape != (height, width, 3):
            raise ValueError(
                "Every synchronized X-Lens image must share one HxWx3 shape."
            )
        padded_images.append(
            np.pad(value, ((top, bottom), (left, right), (0, 0)), mode="edge")
        )
        camera = np.asarray(K, dtype=np.float32).copy()
        if camera.shape != (3, 3):
            raise ValueError("Every X-Lens pinhole intrinsic must be 3x3.")
        camera[0, 2] += left
        camera[1, 2] += top
        padded_intrinsics.append(camera)
    return padded_images, padded_intrinsics, (left, right, top, bottom)


class XLensDROIDTeacher(nn.Module):
    """Frozen online adapter around the official X-Lens implementation.

    A forward call jointly processes both calibrated exterior cameras at every
    physical timestep.  Input images are consumed only once and all returned
    tensors stay on the teacher device for losses, workspace geometry, and
    LagerNVS coverage validation.
    """

    def __init__(
        self,
        repository: str,
        checkpoint: str,
        *,
        architecture_config: str | None = None,
        device: str = "cuda:0",
        amp_dtype: str = "bf16",
    ):
        super().__init__()
        root = Path(repository).expanduser().resolve()
        if not (root / "xlens" / "inference" / "pipeline.py").is_file():
            raise FileNotFoundError(
                f"Official X-Lens was not found at {root}; clone https://github.com/zhouhengamerica/XLens."
            )
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"X-Lens checkpoint does not exist: {checkpoint_path}"
            )
        architecture_path = (
            Path(architecture_config).expanduser().resolve()
            if architecture_config is not None
            else root / "configs" / "xlens_vits.yaml"
        )
        if checkpoint_path.suffix == ".safetensors" and not architecture_path.is_file():
            raise FileNotFoundError(
                "The released X-Lens safetensors checkpoint requires its official "
                f"architecture config: {architecture_path}"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from xlens.inference import XLensInference
        from xlens.inference.preprocess import assemble_batch, pinhole_d_cam

        del assemble_batch, pinhole_d_cam
        inference = XLensInference(
            str(checkpoint_path),
            device=device,
            amp_dtype=amp_dtype,
            config=str(architecture_path),
        )
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            checkpoint_state = {
                key.removeprefix("module."): value
                for key, value in load_file(str(checkpoint_path), device="cpu").items()
            }
            model_state = inference.model.state_dict()
            missing = sorted(set(model_state) - set(checkpoint_state))
            unexpected = sorted(set(checkpoint_state) - set(model_state))
            mismatched = sorted(
                key
                for key in set(model_state) & set(checkpoint_state)
                if model_state[key].shape != checkpoint_state[key].shape
            )
            if missing or unexpected or mismatched:
                raise RuntimeError(
                    "X-Lens checkpoint is not strictly compatible with the official "
                    f"ViT-S config: missing={missing[:5]}, unexpected={unexpected[:5]}, "
                    f"shape_mismatch={mismatched[:5]}."
                )
        self.model = inference.model.eval()
        self.model.requires_grad_(False)
        self.amp_dtype = inference.amp_dtype
        self.checkpoint_path = str(checkpoint_path)
        self.architecture_config = str(architecture_path)
        self.strict_checkpoint_compatible = True

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def _pinhole_rays(K: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if K.shape[-2:] != (3, 3):
            raise ValueError("X-Lens intrinsics must end in 3x3.")
        u = torch.arange(width, device=K.device, dtype=torch.float32) + 0.5
        v = torch.arange(height, device=K.device, dtype=torch.float32) + 0.5
        x = (u.view(1, 1, 1, width) - K[..., 0, 2, None, None]) / K[
            ..., 0, 0, None, None
        ]
        y = (v.view(1, 1, height, 1) - K[..., 1, 2, None, None]) / K[
            ..., 1, 1, None, None
        ]
        x = x.expand(*K.shape[:-2], height, width)
        y = y.expand(*K.shape[:-2], height, width)
        rays = torch.stack((x, y, torch.ones_like(x)), dim=-3)
        return F.normalize(rays, dim=-3, eps=1.0e-6)

    @staticmethod
    def _build_ray_map(d_cam: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
        scenes, views, _, height, width = d_cam.shape
        relative = torch.linalg.inv(c2w[:, :1]) @ c2w
        rotation = relative[..., :3, :3].float()
        translation = relative[..., :3, 3].float()
        world_direction = torch.matmul(
            rotation, d_cam.float().reshape(scenes, views, 3, height * width)
        ).reshape(scenes, views, 3, height, width)
        if views > 1:
            scale = translation[:, 1:].norm(dim=-1).mean(dim=1).clamp_min(1.0e-6)
        else:
            scale = torch.ones(scenes, device=d_cam.device)
        normalized_translation = (
            translation / scale[:, None, None]
        )[..., None, None].expand(-1, -1, -1, height, width)
        return torch.cat((world_direction, normalized_translation), dim=2)

    @torch.inference_mode()
    def prepare_inputs(
        self,
        histories: torch.Tensor,
        intrinsics: torch.Tensor,
        c2w: torch.Tensor,
    ) -> dict[str, Any]:
        """Apply official DROID pinhole preprocessing without model inference."""

        if histories.dim() != 6 or tuple(histories.shape[-3:]) != (
            3,
            XLENS_NATIVE_HEIGHT,
            XLENS_NATIVE_WIDTH,
        ):
            raise ValueError(
                "X-Lens expects raw histories as (B,V,T,3,180,320), got "
                f"{tuple(histories.shape)}."
            )
        batch, views, timesteps = histories.shape[:3]
        if views != 2:
            raise ValueError("Initial DROID X-Lens training uses two exterior cameras.")
        if intrinsics.shape != (batch, views, 3, 3):
            raise ValueError("X-Lens intrinsics must have shape (B,2,3,3).")
        if c2w.shape != (batch, views, 4, 4):
            raise ValueError("X-Lens c2w must have shape (B,2,4,4).")
        images = histories.to(self.device, non_blocking=True)
        images = images.permute(0, 2, 1, 3, 4, 5).reshape(
            batch * timesteps, views, 3, XLENS_NATIVE_HEIGHT, XLENS_NATIVE_WIDTH
        )
        images = images.float().div_(255.0)
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
        images = (images - mean) / std

        target_height = (
            (XLENS_NATIVE_HEIGHT + XLENS_PATCH_SIZE - 1) // XLENS_PATCH_SIZE
        ) * XLENS_PATCH_SIZE
        target_width = (
            (XLENS_NATIVE_WIDTH + XLENS_PATCH_SIZE - 1) // XLENS_PATCH_SIZE
        ) * XLENS_PATCH_SIZE
        pad_y, pad_x = (
            target_height - XLENS_NATIVE_HEIGHT,
            target_width - XLENS_NATIVE_WIDTH,
        )
        top, bottom = pad_y // 2, pad_y - pad_y // 2
        left, right = pad_x // 2, pad_x - pad_x // 2
        flat_images = images.flatten(0, 1)
        flat_images = F.pad(
            flat_images, (left, right, top, bottom), mode="replicate"
        )
        images = flat_images.reshape(
            batch * timesteps, views, 3, target_height, target_width
        )

        K = intrinsics.to(self.device, dtype=torch.float32, non_blocking=True)
        K = K[:, None].expand(-1, timesteps, -1, -1, -1).reshape(
            batch * timesteps, views, 3, 3
        ).clone()
        K[..., 0, 2] += left
        K[..., 1, 2] += top
        scene_c2w = c2w.to(self.device, dtype=torch.float32, non_blocking=True)
        scene_c2w = scene_c2w[:, None].expand(-1, timesteps, -1, -1, -1).reshape(
            batch * timesteps, views, 4, 4
        )
        d_cam = self._pinhole_rays(K, target_height, target_width)
        ray_map = self._build_ray_map(d_cam, scene_c2w)
        cam_types = torch.ones(
            batch * timesteps, views, dtype=torch.long, device=self.device
        )
        return {
            "images": images,
            "ray_map": ray_map,
            "d_cam": d_cam,
            "cam_types": cam_types,
            "batch": batch,
            "views": views,
            "timesteps": timesteps,
            "padding_lrtb": (left, right, top, bottom),
        }

    @torch.inference_mode()
    def infer_prepared(self, prepared: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run the frozen official model and restore the native DROID grid."""

        images = prepared["images"]
        ray_map = prepared["ray_map"]
        d_cam = prepared["d_cam"]
        cam_types = prepared["cam_types"]
        batch = int(prepared["batch"])
        views = int(prepared["views"])
        timesteps = int(prepared["timesteps"])
        left, right, top, bottom = prepared["padding_lrtb"]
        autocast = (
            torch.autocast("cuda", dtype=self.amp_dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            output = self.model(
                images,
                ray_map=ray_map,
                d_cam=d_cam,
                cam_types=cam_types,
            )
        depth = output["depth_metric"].float()[
            ..., top : top + XLENS_NATIVE_HEIGHT, left : left + XLENS_NATIVE_WIDTH
        ]
        confidence = output.get("depth_conf")
        confidence = (
            torch.ones_like(depth)
            if confidence is None
            else confidence.float()[
                ...,
                top : top + XLENS_NATIVE_HEIGHT,
                left : left + XLENS_NATIVE_WIDTH,
            ]
        )
        depth = depth.reshape(
            batch, timesteps, views, XLENS_NATIVE_HEIGHT, XLENS_NATIVE_WIDTH
        ).unsqueeze(3)
        confidence = confidence.reshape_as(depth)
        validity = (
            torch.isfinite(depth)
            & torch.isfinite(confidence)
            & (depth > 0.0)
            & (confidence > 0.0)
        )
        return {
            "metric_depth": torch.where(validity, depth, torch.zeros_like(depth)),
            "confidence": torch.where(
                validity, confidence, torch.zeros_like(confidence)
            ),
            "validity": validity,
            "padding_lrtb": torch.tensor(
                (left, right, top, bottom), device=self.device, dtype=torch.int32
            ),
        }

    @torch.inference_mode()
    def forward(
        self,
        histories: torch.Tensor,
        intrinsics: torch.Tensor,
        c2w: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Infer metric depth for ``(B,V,T,3,180,320)`` raw RGB histories."""

        return self.infer_prepared(self.prepare_inputs(histories, intrinsics, c2w))

    def predict(
        self,
        images: Sequence[np.ndarray],
        intrinsics: Sequence[np.ndarray],
        c2w: Sequence[np.ndarray],
    ) -> dict[str, np.ndarray]:
        if len(images) != 2:
            raise ValueError("The convenience X-Lens API expects two exterior views.")
        history = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
        history = history[:, None].expand(-1, 3, -1, -1, -1)[None]
        output = self(
            history,
            torch.from_numpy(np.asarray(intrinsics, dtype=np.float32))[None],
            torch.from_numpy(np.asarray(c2w, dtype=np.float32))[None],
        )
        depth = output["metric_depth"][0, 0, :, 0].cpu().numpy()
        confidence = output["confidence"][0, 0, :, 0].cpu().numpy()
        validity = output["validity"][0, 0, :, 0].cpu().numpy()
        return {
            "metric_depth": depth.astype(np.float32),
            "confidence": confidence.astype(np.float32),
            "validity": validity,
            "padding_lrtb": output["padding_lrtb"].cpu().numpy(),
        }
