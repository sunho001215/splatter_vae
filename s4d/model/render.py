"""gsplat rasterization wrappers. Everything here runs in FP32 regardless of autocast."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch

from s4d.model.gaussians import GaussianSet


@lru_cache(maxsize=1)
def require_prebuilt_renderer() -> str:
    """Load the native binary before gsplat can turn an ABI error into a JIT build.

    Source builds are never attempted by this project. An approved, compatible
    installed extension is required for every rendering or training entry point.
    """
    package = importlib.util.find_spec("gsplat")
    if package is None or not package.submodule_search_locations:
        raise RuntimeError("gsplat is not installed; a compatible prebuilt extension is required")
    directory = Path(next(iter(package.submodule_search_locations)))
    extensions = sorted(directory.glob("csrc*.so"))
    if len(extensions) != 1:
        raise RuntimeError(f"Expected one prebuilt gsplat csrc binary in {directory}; source/JIT builds are disabled")
    binary = extensions[0]
    try:
        compiled = sys.modules.get("gsplat.csrc")
        if compiled is None:
            spec = importlib.util.spec_from_file_location("gsplat.csrc", binary)
            compiled = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(compiled)
            sys.modules["gsplat.csrc"] = compiled
        if Path(compiled.__file__).resolve() != binary.resolve():
            raise ImportError("gsplat.csrc was loaded from an unexpected location")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"Prebuilt gsplat is incompatible with torch {torch.__version__}: {exc}. "
            "Rendering is blocked; source/JIT builds are disabled."
        ) from exc
    return str(binary)


def _rasterize(**kwargs):
    require_prebuilt_renderer()
    from gsplat.rendering import rasterization  # noqa: PLC0415

    with torch.autocast(device_type="cuda", enabled=False):
        return rasterization(**kwargs)


def _expand_cams(w2c: torch.Tensor, K: torch.Tensor, S: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,V,…) camera tensors -> (B,S,V,…) for S rendered Gaussian states."""
    return (
        w2c.float()[:, None].expand(-1, S, -1, -1, -1).contiguous(),
        K.float()[:, None].expand(-1, S, -1, -1, -1).contiguous(),
    )


def _expand_attr(x: torch.Tensor, S: int) -> torch.Tensor:
    return x.float()[:, None].expand(-1, S, *x.shape[1:]).contiguous()


def render_rgbd(
    gs: GaussianSet,
    xyz_seq: torch.Tensor,
    w2c: torch.Tensor,
    K: torch.Tensor,
    height: int,
    width: int,
    near: float,
    far: float,
) -> dict[str, torch.Tensor]:
    """Render RGB + expected depth + alpha for every (state, camera).

    xyz_seq (B,S,N,3) Gaussian centres for S states (timesteps); cameras (B,V,…).
    Returns rgb (B,S,V,3,H,W), depth (B,S,V,1,H,W) expected depth, alpha (B,S,V,1,H,W).
    """
    B, S, N, _ = xyz_seq.shape
    V = w2c.shape[1]
    viewmats, Ks = _expand_cams(w2c, K, S)
    rendered, alpha, _ = _rasterize(
        means=xyz_seq.float().contiguous(),
        quats=_expand_attr(gs.quats, S),
        scales=_expand_attr(gs.scales, S),
        opacities=_expand_attr(gs.opacity, S),
        colors=_expand_attr(gs.rgb, S),
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        packed=False,
        sh_degree=None,
        backgrounds=torch.zeros(B, S, V, 3, device=xyz_seq.device),
        render_mode="RGB+ED",
    )
    rendered = rendered.movedim(-1, -3)  # (B,S,V,4,H,W)
    return {"rgb": rendered[..., :3, :, :], "depth": rendered[..., 3:4, :, :], "alpha": alpha.movedim(-1, -3)}


# GSPLAT_NUM_CHANNELS of the pinned gsplat build (gsplat/cuda/csrc/Config.h, revision d28ee0c).
COMPILED_CHANNELS = (1, 2, 3, 4, 5, 6, 8, 9, 16, 17, 21, 23, 24, 32, 33, 64, 65, 128, 129, 256, 257, 512, 513)


def render_features(
    gs: GaussianSet,
    xyz_states: torch.Tensor,
    features: torch.Tensor,
    w2c: torch.Tensor,
    K: torch.Tensor,
    height: int,
    width: int,
    near: float,
    far: float,
) -> dict[str, torch.Tensor]:
    """Splat per-Gaussian features (B,S,N,F) from the (detached) states xyz_states (B,S,N,3).

    Returns features (B,S,V,F,H,W) = sum_i w_i f_i and alpha (B,S,V,1,H,W) = sum_i w_i.
    """
    B, S, N, F = features.shape
    V = w2c.shape[1]
    viewmats, Ks = _expand_cams(w2c, K, S)
    # The compiled kernels support a fixed set of channel counts (gsplat Config.h); zero channels are exact padding.
    padded = next(c for c in COMPILED_CHANNELS if c >= F)
    colors = torch.nn.functional.pad(features.float(), (0, padded - F)).contiguous()
    rendered, alpha, _ = _rasterize(
        means=xyz_states.float().contiguous(),
        quats=_expand_attr(gs.quats, S),
        scales=_expand_attr(gs.scales, S),
        opacities=_expand_attr(gs.opacity, S),
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        packed=False,
        sh_degree=None,
        backgrounds=torch.zeros(B, S, V, padded, device=features.device),
        render_mode="RGB",
    )
    return {"features": rendered[..., :F].movedim(-1, -3), "alpha": alpha.movedim(-1, -3)}


def render_hard_depth(
    gs: GaussianSet,
    xyz_seq: torch.Tensor,
    w2c: torch.Tensor,
    K: torch.Tensor,
    height: int,
    width: int,
    near: float,
    far: float,
    opacity: float = 0.95,
) -> dict[str, torch.Tensor]:
    """Expected depth with every opacity fixed and scales/rotations detached: only centres get gradient."""
    B, S, N, _ = xyz_seq.shape
    viewmats, Ks = _expand_cams(w2c, K, S)
    rendered, alpha, _ = _rasterize(
        means=xyz_seq.float().contiguous(),
        quats=_expand_attr(gs.quats.detach(), S),
        scales=_expand_attr(gs.scales.detach(), S),
        opacities=torch.full((B, S, N), opacity, device=xyz_seq.device, dtype=torch.float32),
        colors=None,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        packed=False,
        sh_degree=None,
        render_mode="ED",
    )
    return {"depth": rendered.movedim(-1, -3), "alpha": alpha.movedim(-1, -3)}
