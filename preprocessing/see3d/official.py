from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SEE3D_PREPROCESSING_VERSION = "official-sparse-letterbox512-droid-v1"
SEE3D_SIZE = 512


def letterbox_droid_image(
    image: np.ndarray,
    *,
    is_mask: bool = False,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    value = np.asarray(image)
    height, width = value.shape[:2]
    scale = SEE3D_SIZE / width
    resized_height = int(round(height * scale))
    mode = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    pil = Image.fromarray(value.astype(np.uint8)).resize(
        (SEE3D_SIZE, resized_height), mode
    )
    top = (SEE3D_SIZE - resized_height) // 2
    bottom = SEE3D_SIZE - resized_height - top
    if is_mask:
        canvas = np.zeros((SEE3D_SIZE, SEE3D_SIZE), dtype=np.uint8)
        canvas[top : top + resized_height] = np.asarray(pil)
    else:
        canvas = np.zeros((SEE3D_SIZE, SEE3D_SIZE, 3), dtype=np.uint8)
        canvas[top : top + resized_height] = np.asarray(pil)
    return canvas, (0, 0, top, bottom)


def unletterbox_droid_image(
    image: np.ndarray, padding: tuple[int, int, int, int]
) -> np.ndarray:
    left, right, top, bottom = padding
    value = np.asarray(image, dtype=np.uint8)
    cropped = value[top : SEE3D_SIZE - bottom, left : SEE3D_SIZE - right]
    return np.asarray(
        Image.fromarray(cropped).resize((320, 180), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


class OfficialSee3DCompleter:
    """Use the authors' sparse-view model; target pose comes only from our warp."""

    def __init__(
        self,
        repository: str,
        base_model_path: str,
        *,
        seed: int = 12345,
        super_resolution: bool = False,
    ):
        if super_resolution:
            raise ValueError(
                "The initial DROID See3D pipeline keeps official super-resolution disabled."
            )
        root = Path(repository).expanduser().resolve()
        if (
            not (root / "mv_diffusion.py").is_file()
            or not (root / "inference.py").is_file()
        ):
            raise FileNotFoundError(
                f"Official See3D was not found at {root}; clone https://github.com/baaivision/See3D."
            )
        weights = Path(base_model_path).expanduser().resolve()
        sparse_unet = weights / "unet" / "sparse" / "ema-checkpoint"
        if not sparse_unet.exists():
            raise FileNotFoundError(
                f"Official See3D sparse weights were not found under {weights}."
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from mv_diffusion import mvdream_diffusion_model
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(str(weights), subfolder="tokenizer")
        self.model = mvdream_diffusion_model(
            str(weights), str(sparse_unet), tokenizer, seed=int(seed)
        )

    @staticmethod
    def _batch(
        images: Sequence[np.ndarray], masks: Sequence[np.ndarray]
    ) -> dict[str, torch.Tensor]:
        conditions = []
        observed_masks = []
        for image, mask in zip(images, masks, strict=True):
            conditions.append(
                torch.from_numpy(np.asarray(image, np.float32) / 127.5 - 1.0).permute(
                    2, 0, 1
                )
            )
            observed_masks.append(
                torch.from_numpy((np.asarray(mask) > 127).astype(np.float32))[None]
            )
        return {
            "conditioning_pixel_values": torch.stack(conditions),
            "masks": torch.stack(observed_masks),
        }

    def complete(
        self,
        source_images: Sequence[np.ndarray],
        warped_target: np.ndarray,
        observed_target_mask: np.ndarray,
        *,
        preserve_observed_pixels: bool = True,
    ) -> np.ndarray:
        if not source_images:
            raise ValueError(
                "See3D sparse completion requires at least one real source image."
            )
        frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for source in source_images:
            frame, _ = letterbox_droid_image(source)
            frames.append(frame)
            masks.append(np.ones((SEE3D_SIZE, SEE3D_SIZE), dtype=np.uint8) * 255)
        target, target_padding = letterbox_droid_image(warped_target)
        target_mask, _ = letterbox_droid_image(
            np.asarray(observed_target_mask, dtype=np.uint8) * 255, is_mask=True
        )
        frames.append(target)
        masks.append(target_mask)
        batch = self._batch(frames, masks)
        generated = self.model.inference_next_frame(
            [""],
            batch,
            len(frames),
            SEE3D_SIZE,
            SEE3D_SIZE,
            gt_num_frames=len(source_images),
            output_type="pil",
        )[-1]
        generated_array = np.asarray(generated.convert("RGB"), dtype=np.uint8)
        result = unletterbox_droid_image(generated_array, target_padding)
        if preserve_observed_pixels:
            observed = np.asarray(observed_target_mask, dtype=bool)
            warp = np.asarray(warped_target, dtype=np.uint8)
            result[observed] = warp[observed]
        return result
