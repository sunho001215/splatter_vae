from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

MEMFOF_MODEL_ID = "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"
MEMFOF_MODEL_REVISION = "6c6c9aa3ad64f93aee8efbc2f7a6e4535814ee96"
MEMFOF_REPOSITORY_REVISION = "a51de9fc59c6fe20ba08e079372c7b583d58a712"
MEMFOF_ITERATIONS = 2
MEMFOF_NATIVE_HEIGHT = 180
MEMFOF_NATIVE_WIDTH = 320
MEMFOF_DIRECTIONS = ("middle_to_previous", "middle_to_next")


def _install_native_droid_corr_fix() -> None:
    """Make the official correlation pyramid well-defined on the 180x320 grid.

    At the pinned upstream revision ``CorrBlock`` downsamples ``fmap2`` after
    every pyramid level, including after constructing the final level.  DROID's
    internally padded 192x320 input gives a final 1x2 feature map; the unused
    post-final interpolation consequently requests 0x1 and raises.  Its
    bilinear sampler also normalizes the singleton height by ``height - 1``,
    producing NaN coordinates.  Small CUDA grid-sample kernels happened to
    tolerate those coordinates while larger batches returned entirely NaN
    flow.  The compatibility layer omits the unused downsample and maps a
    singleton axis to normalized coordinate zero.  Images are not resized.
    """

    import memfof.corr as corr_module
    import memfof.model as model_module

    sampler = corr_module.bilinear_sampler
    if not getattr(sampler, "_droid_singleton_grid_fix", False):

        def droid_bilinear_sampler(img, coords, mode="bilinear", mask=False):
            """Normalize singleton pyramid dimensions without dividing by zero."""

            height, width = img.shape[-2:]
            xgrid, ygrid = coords.split([1, 1], dim=-1)
            xgrid = (
                2 * xgrid / (width - 1) - 1
                if width > 1
                else torch.zeros_like(xgrid)
            )
            ygrid = (
                2 * ygrid / (height - 1) - 1
                if height > 1
                else torch.zeros_like(ygrid)
            )
            grid = torch.cat((xgrid, ygrid), dim=-1)
            sampled = F.grid_sample(
                img, grid, mode=mode, align_corners=True
            )
            if mask:
                valid = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
                return sampled, valid.float()
            return sampled

        droid_bilinear_sampler._droid_singleton_grid_fix = True
        corr_module.bilinear_sampler = droid_bilinear_sampler

    existing = model_module.CorrBlock
    if getattr(existing, "_droid_native_grid_fix", False):
        return

    class DROIDNativeCorrBlock(corr_module.CorrBlock):
        _droid_native_grid_fix = True

        def __init__(self, fmap1, fmap2, corr_levels, corr_radius):
            self.num_levels = int(corr_levels)
            self.radius = int(corr_radius)
            self.corr_pyramid = []
            for level in range(self.num_levels):
                corr = corr_module.CorrBlock.corr(fmap1, fmap2, 1)
                batch, h1, w1, channels, h2, w2 = corr.shape
                self.corr_pyramid.append(
                    corr.reshape(batch * h1 * w1, channels, h2, w2)
                )
                if level + 1 < self.num_levels:
                    fmap2 = F.interpolate(
                        fmap2,
                        scale_factor=0.5,
                        mode="bilinear",
                        align_corners=False,
                    )

    model_module.CorrBlock = DROIDNativeCorrBlock


class MEMFOFDROIDTeacher(nn.Module):
    """Frozen official MEMFOF teacher for three native-resolution DROID frames.

    Input layout is ``(B, V, 3, 3, 180, 320)``.  The returned two flows are
    defined on the middle-frame grid and ordered exactly as upstream:
    middle->previous, then middle->next.
    """

    def __init__(
        self,
        *,
        model_id: str = MEMFOF_MODEL_ID,
        revision: str = MEMFOF_MODEL_REVISION,
        iterations: int = MEMFOF_ITERATIONS,
        device: torch.device | str = "cuda:0",
        cache_dir: str | Path | None = None,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if int(iterations) != MEMFOF_ITERATIONS:
            raise ValueError("DROID MEMFOF must use exactly two refinement iterations.")
        _install_native_droid_corr_fix()
        from memfof import MEMFOF

        kwargs: dict[str, Any] = {"revision": str(revision)}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
        model = MEMFOF.from_pretrained(str(model_id), **kwargs)
        self.model = model.eval().to(device)
        self.model.requires_grad_(False)
        self.iterations = MEMFOF_ITERATIONS
        self.model_id = str(model_id)
        self.revision = str(revision)
        self.amp_dtype = amp_dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def _confidence_from_info(info: torch.Tensor) -> torch.Tensor:
        """Turn MEMFOF's two-component Laplace mixture into [0,1] confidence."""

        logits = info[..., :2, :, :].float()
        raw_log_scale = info[..., 2:4, :, :].float()
        mixture = logits.softmax(dim=-3)
        log_scale = torch.stack(
            (
                raw_log_scale[..., 0, :, :].clamp(0.0, 10.0),
                raw_log_scale[..., 1, :, :].clamp(-10.0, 0.0),
            ),
            dim=-3,
        )
        expected_scale = (mixture * log_scale.exp()).sum(dim=-3, keepdim=True)
        return (1.0 / (1.0 + expected_scale)).clamp(0.0, 1.0)

    @torch.inference_mode()
    def forward(self, histories: torch.Tensor) -> dict[str, torch.Tensor]:
        if histories.dim() != 6 or tuple(histories.shape[2:5]) != (3, 3, 180):
            expected = "(B,V,3,3,180,320)"
            raise ValueError(f"MEMFOF expects {expected}, got {tuple(histories.shape)}.")
        if histories.shape[-1] != MEMFOF_NATIVE_WIDTH:
            raise ValueError("MEMFOF must receive the native 320-pixel DROID width.")
        batch, views = histories.shape[:2]
        images = histories.reshape(
            batch * views, 3, 3, MEMFOF_NATIVE_HEIGHT, MEMFOF_NATIVE_WIDTH
        ).to(self.device, non_blocking=True)
        if images.dtype not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"Unsupported MEMFOF image dtype {images.dtype}.")
        autocast = (
            torch.autocast("cuda", dtype=self.amp_dtype)
            if self.device.type == "cuda" and self.amp_dtype is not None
            else nullcontext()
        )
        with autocast:
            output = self.model(
                images,
                iters=self.iterations,
                fmap_cache=[None, None, None],
            )
        flow_predictions = output.get("flow")
        info_predictions = output.get("info")
        if not flow_predictions or not info_predictions:
            raise RuntimeError("Official MEMFOF returned no refined flow prediction.")
        flow = flow_predictions[-1].float().reshape(
            batch,
            views,
            2,
            2,
            MEMFOF_NATIVE_HEIGHT,
            MEMFOF_NATIVE_WIDTH,
        )
        info = info_predictions[-1].float().reshape(
            batch,
            views,
            2,
            4,
            MEMFOF_NATIVE_HEIGHT,
            MEMFOF_NATIVE_WIDTH,
        )
        if not torch.isfinite(flow).all() or not torch.isfinite(info).all():
            raise FloatingPointError(
                "Official MEMFOF returned non-finite flow or uncertainty."
            )
        valid = torch.isfinite(flow).all(dim=3, keepdim=True)
        confidence = self._confidence_from_info(info)
        confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
        flow = torch.where(valid, flow, torch.zeros_like(flow))
        return {
            "flow": flow.contiguous(),
            "confidence": confidence.contiguous(),
            "validity": valid.contiguous(),
            "info": info.contiguous(),
        }
