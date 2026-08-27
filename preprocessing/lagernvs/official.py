from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .camera import (
    LAGERNVS_IMAGE_SIZE,
    camera_tokens,
    canonicalize_droid_views,
    normalize_lagernvs_poses,
    plucker_rays,
)

LAGERNVS_REPOSITORY_REVISION = "665f727aba8298a04ff4c040fd6279a32ef23017"
LAGERNVS_CHECKPOINT_ID = "facebook/lagernvs_dl3dv_2-6_v_256"
LAGERNVS_CHECKPOINT_REVISION = "4026552953a72c5fb037501564dc673dd73c574e"
LAGERNVS_CHECKPOINT_FILENAME = "model.pt"


def resolve_lagernvs_checkpoint(
    checkpoint: str | Path | None,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LagerNVS checkpoint does not exist: {path}")
        return path
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        LAGERNVS_CHECKPOINT_ID,
        filename=LAGERNVS_CHECKPOINT_FILENAME,
        revision=LAGERNVS_CHECKPOINT_REVISION,
        cache_dir=None
        if cache_dir is None
        else str(Path(cache_dir).expanduser().resolve()),
    )
    return Path(downloaded).resolve()


def _install_xformers_sdpa_fallback() -> None:
    try:
        import xformers.ops

        return
    except ImportError:
        pass

    xformers = types.ModuleType("xformers")
    ops = types.ModuleType("xformers.ops")

    def memory_efficient_attention(q, k, v, p=0.0, op=None):
        del op
        output = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            dropout_p=float(p),
        )
        return output.permute(0, 2, 1, 3)

    ops.memory_efficient_attention = memory_efficient_attention
    dummy = type("SDPAFallbackOp", (), {})
    ops.fmha = types.SimpleNamespace(
        flash=types.SimpleNamespace(FwOp=dummy, BwOp=dummy),
        flash3=types.SimpleNamespace(FwOp=dummy, BwOp=dummy),
    )
    xformers.ops = ops
    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = ops


def _construct_official_model(repository: Path) -> nn.Module:
    """Load LagerNVS despite its top-level ``models`` namespace collision."""

    if not (repository / "models" / "encoder_decoder.py").is_file():
        raise FileNotFoundError(
            f"Official LagerNVS source is missing under {repository}."
        )
    _install_xformers_sdpa_fallback()
    saved_models = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "models" or name.startswith("models.")
    }
    for name in saved_models:
        sys.modules.pop(name, None)
    old_path = list(sys.path)
    sys.path.insert(0, str(repository))
    # LagerNVS ships ``models/`` without an ``__init__.py``.  A namespace
    # package on an earlier sys.path entry still loses to this project's regular
    # top-level ``models`` package, so path ordering alone cannot resolve the
    # official module.  Root a temporary namespace explicitly at the upstream
    # directory while its model is constructed, then restore the project
    # modules below.
    models_path = str(repository / "models")
    official_models = types.ModuleType("models")
    official_models.__path__ = [models_path]
    official_models.__package__ = "models"
    official_models.__spec__ = importlib.machinery.ModuleSpec(
        "models", loader=None, is_package=True
    )
    official_models.__spec__.submodule_search_locations = [models_path]
    sys.modules["models"] = official_models
    try:
        module = importlib.import_module("models.encoder_decoder")
        model = module.EncDec_VitB8(
            pretrained_vggt=False,
            pretrained_patch_embed=False,
            attention_to_features_type="bidirectional_cross_attention",
        )
    finally:
        imported_names = [
            name
            for name, value in tuple(sys.modules.items())
            if (name == "models" or name.startswith("models."))
            and (
                name == "models"
                or str(getattr(value, "__file__", "")).startswith(str(repository))
            )
        ]
        for name in imported_names:
            sys.modules.pop(name, None)
        sys.path[:] = old_path
        sys.modules.update(saved_models)
    return model


class LagerNVSDROIDTeacher(nn.Module):
    """Frozen official posed DL3DV LagerNVS teacher at canonical 256x256."""

    def __init__(
        self,
        repository: str | Path,
        checkpoint: str | Path | None = None,
        *,
        cache_dir: str | Path | None = None,
        device: torch.device | str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        microbatch_size: int = 1,
        canonical_focal_px: float = 186.5,
    ) -> None:
        super().__init__()
        if int(microbatch_size) <= 0:
            raise ValueError("LagerNVS teacher microbatch must be positive.")
        repository_path = Path(repository).expanduser().resolve()
        checkpoint_path = resolve_lagernvs_checkpoint(
            checkpoint, cache_dir=cache_dir
        )
        model = _construct_official_model(repository_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state, strict=True)
        self.model = model.eval().to(device=device, dtype=dtype)
        self.model.requires_grad_(False)
        self.microbatch_size = int(microbatch_size)
        self.teacher_dtype = dtype
        self.canonical_focal_px = float(canonical_focal_px)
        self.checkpoint_path = str(checkpoint_path)
        self.repository_path = str(repository_path)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def prepare_inputs(
        self,
        raw_source_images: torch.Tensor,
        source_K: torch.Tensor,
        source_c2w: torch.Tensor,
        target_c2w: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Canonicalize raw DROID cameras and build official LagerNVS inputs."""

        if raw_source_images.dim() != 5 or raw_source_images.shape[1:] != (
            2,
            3,
            180,
            320,
        ):
            raise ValueError(
                "LagerNVS consumes raw current exterior views as (B,2,3,180,320)."
            )
        batch = raw_source_images.shape[0]
        if source_K.shape != (batch, 2, 3, 3):
            raise ValueError("LagerNVS source intrinsics must be (B,2,3,3).")
        if source_c2w.shape != (batch, 2, 4, 4) or target_c2w.shape != (
            batch,
            4,
            4,
        ):
            raise ValueError("LagerNVS source/target c2w shapes are invalid.")
        source_images = raw_source_images.to(self.device, non_blocking=True)
        source_K = source_K.to(self.device, dtype=torch.float32, non_blocking=True)
        source_c2w = source_c2w.to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        target_c2w = target_c2w.to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        canonical_images, canonical_source_K, source_validity = (
            canonicalize_droid_views(
                source_images,
                source_K,
                focal_px=self.canonical_focal_px,
            )
        )
        target_K = canonical_source_K[:, 0].clone()
        all_K = torch.cat((canonical_source_K, target_K[:, None]), dim=1)
        all_c2w = torch.cat((source_c2w, target_c2w[:, None]), dim=1)
        normalized_c2w, camera_scale, scene_scale_ratio = normalize_lagernvs_poses(
            all_c2w, num_conditioning_views=2
        )
        tokens = camera_tokens(normalized_c2w, all_K, camera_scale)
        target_rays = plucker_rays(
            normalized_c2w[:, 2:3],
            all_K[:, 2:3],
            LAGERNVS_IMAGE_SIZE,
            LAGERNVS_IMAGE_SIZE,
        )
        source_rays = torch.zeros(
            batch,
            2,
            6,
            LAGERNVS_IMAGE_SIZE,
            LAGERNVS_IMAGE_SIZE,
            device=self.device,
        )
        rays = torch.cat((source_rays, target_rays), dim=1)
        return {
            "canonical_source_rgb": canonical_images,
            "canonical_source_validity": source_validity,
            "canonical_K": target_K,
            "normalized_c2w": normalized_c2w,
            "camera_tokens": tokens,
            "target_rays": target_rays,
            "rays": rays,
            "camera_scale": camera_scale,
            "scene_scale_ratio": scene_scale_ratio,
        }

    @torch.inference_mode()
    def infer_prepared(
        self, prepared: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run the frozen posed DL3DV model on canonical prepared inputs."""

        canonical_images = prepared["canonical_source_rgb"]
        rays = prepared["rays"]
        tokens = prepared["camera_tokens"]
        batch = int(canonical_images.shape[0])
        generated = []
        autocast = (
            torch.autocast("cuda", dtype=self.teacher_dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )
        for start in range(0, batch, self.microbatch_size):
            stop = min(start + self.microbatch_size, batch)
            target_zeros = torch.zeros_like(canonical_images[start:stop, :1])
            model_images = torch.cat(
                (canonical_images[start:stop], target_zeros), dim=1
            )
            with autocast:
                output = self.model(
                    model_images,
                    rays[start:stop],
                    tokens[start:stop],
                    num_cond_views=2,
                )
            generated.append(output[:, 2].float())
        output = dict(prepared)
        output.pop("rays")
        output["generated_rgb"] = torch.cat(generated, dim=0)
        return output

    @torch.inference_mode()
    def forward(
        self,
        raw_source_images: torch.Tensor,
        source_K: torch.Tensor,
        source_c2w: torch.Tensor,
        target_c2w: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Render every logical target; no probability or sample skipping exists."""

        return self.infer_prepared(
            self.prepare_inputs(
                raw_source_images,
                source_K,
                source_c2w,
                target_c2w,
            )
        )
