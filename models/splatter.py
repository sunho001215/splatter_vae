import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from gsplat.rendering import rasterization

from utils.general_utils import (
    flatten_vector,
    quaternion_raw_multiply,
    transform_rotations
)
from utils.ray_utils import (
    build_ray_dirs_from_intrinsics
)
from utils.sh_utils import (
    init_sh_transform_matrices,
    transform_SHs
)

# -----------------------------------------------------------------------------
# Small config objects (subset of original Hydra cfg)
# -----------------------------------------------------------------------------

@dataclass
class SplatterDataConfig:
    """
    Camera & image config
    """
    img_height: int = 128
    img_width: int = 128
    znear: float = 0.1
    zfar: float = 2.0
    white_background: bool = False
    inverted_x: bool = False         # optional flips for datasets
    inverted_y: bool = False
    category: str = "generic"


@dataclass
class SplatterModelConfig:
    """
    Gaussian parameterization config.

    We *fix* spherical harmonics to L = 1:
      - SH0: 1 coefficient (DC term)
      - SH1: 3 coefficients (linear terms) per color channel
    so that the higher-order block has size 3 (SH1) per color, i.e. 3*3.
    :contentReference[oaicite:3]{index=3}
    """
    max_sh_degree: int = 1            # we assume 0 or 1, see asserts below
    isotropic: bool = False           # if True: same scale for xyz
    num_gaussians_per_pixel: int = 5  # number of splat Gaussians per pixel
    depth_scale: float = 1.0
    depth_bias: float = 0.0
    xyz_scale: float = 1.0
    xyz_bias: float = 0.0
    opacity_scale: float = 1.0
    opacity_bias: float = 0.0
    scale_scale: float = 1.0
    scale_bias: float = 1.0


@dataclass
class SplatterConfig:
    data: SplatterDataConfig
    model: SplatterModelConfig

# -----------------------------------------------------------------------------
# Main class: splatter map (decoder output) -> Gaussian param dict
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Direct multi-Gaussian splatter converter
# -----------------------------------------------------------------------------

class VAESplatterToGaussians(nn.Module):
    """
    Direct converter:
      decoder splatter image -> K depth-ordered Gaussians per pixel

    This keeps the *old* SplatterVAE rendering style:
      one decoded splatter image predicts multiple Gaussians at each pixel,
      sorted by depth, then rendered directly.

    Channel layout for K gaussians / pixel:
        depth(K)
        offset(3K)
        opacity(K)
        scaling(3K)
        rotation(4K)
        features_dc(3K)
        features_rest(K * sh_rest)

    where:
        sh_rest = 0                         if max_sh_degree == 0
        sh_rest = 3 * (((L + 1)^2) - 1)    if max_sh_degree > 0
    """

    def __init__(self, cfg: SplatterConfig):
        super().__init__()
        self.cfg = cfg

        assert cfg.model.max_sh_degree in (0, 1), \
            "This direct converter currently supports max_sh_degree in {0,1}."
        assert cfg.model.num_gaussians_per_pixel >= 1, \
            "num_gaussians_per_pixel must be >= 1."

        self.depth_act = nn.Sigmoid()
        self.opacity_activation = torch.sigmoid
        self.scaling_activation = torch.exp
        self.rotation_activation = nn.functional.normalize

        if self.cfg.model.max_sh_degree > 0:
            sh_to_v, v_to_sh = init_sh_transform_matrices(
                device=torch.device("cpu"),
                max_sh_degree=self.cfg.model.max_sh_degree,
            )
            self.register_buffer("sh_to_v_transform", sh_to_v, persistent=False)
            self.register_buffer("v_to_sh_transform", v_to_sh, persistent=False)
        else:
            self.register_buffer("sh_to_v_transform", None, persistent=False)
            self.register_buffer("v_to_sh_transform", None, persistent=False)

    # ------------------------------------------------------------------
    # Splatter channel bookkeeping
    # ------------------------------------------------------------------
    def get_split_dimensions(self):
        k = int(self.cfg.model.num_gaussians_per_pixel)
        sh_rest = 0 if self.cfg.model.max_sh_degree == 0 else (((self.cfg.model.max_sh_degree + 1) ** 2) - 1) * 3

        dims = (
            k,          # depth
            3 * k,      # offset
            k,          # opacity
            3 * k,      # scaling
            4 * k,      # rotation
            3 * k,      # features_dc
        )
        if sh_rest > 0:
            dims = dims + (k * sh_rest,)
        return dims

    def num_splatter_channels(self) -> int:
        return int(sum(self.get_split_dimensions()))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _split(self, splatter: torch.Tensor):
        return splatter.split(self.get_split_dimensions(), dim=1)

    def _map_to_ordered_tensor(
        self,
        x: torch.Tensor,
        channels_per_gaussian: int,
    ) -> torch.Tensor:
        """
        (B, K*C, H, W) -> (B, H*W, K, C)
        """
        b, _, h, w = x.shape
        k = int(self.cfg.model.num_gaussians_per_pixel)

        x = x.view(b, k, channels_per_gaussian, h, w)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x.view(b, h * w, k, channels_per_gaussian)

    def _compute_xyz_camera(
        self,
        depth_logits: torch.Tensor,   # (B, N, K, 1)
        offset: torch.Tensor,         # (B, N, K, 3)
        intrinsics: torch.Tensor,     # (B, 3, 3)
        activate_output: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build camera-space Gaussian centers for the direct multi-Gaussian path."""
        dcfg = self.cfg.data
        mcfg = self.cfg.model

        # depth_logits: (B, N, K, 1) -> (B, N, K)
        depth_logits = depth_logits.squeeze(-1)

        # Old fixed-layer monotonic depth logic
        if activate_output:
            depth_pre = depth_logits * mcfg.depth_scale + mcfg.depth_bias  # (B, N, K)

            if depth_pre.shape[2] == 1:
                depth_stack = depth_pre
            else:
                base = depth_pre[:, :, :1]           # (B, N, 1)
                # Clamp before exp() to avoid inf gradients from rare large logits.
                inc_pre = depth_pre[:, :, 1:].clamp(min=-10.0, max=10.0)
                inc = torch.exp(inc_pre)             # (B, N, K-1), positive increments
                depth_tail = base + torch.cumsum(inc, dim=2)
                depth_stack = torch.cat([base, depth_tail], dim=2)

            depth = self.depth_act(depth_stack) * (dcfg.zfar - dcfg.znear) + dcfg.znear
        else:
            depth = depth_logits

        # Camera-space offset
        offset = offset * mcfg.xyz_scale + mcfg.xyz_bias  # (B, N, K, 3)
        offset = torch.nan_to_num(offset, nan=0.0, posinf=0.0, neginf=0.0).clamp(-3.0, 3.0)

        # Ray directions from intrinsics come back as (B, 3, H, W)
        ray_dirs = build_ray_dirs_from_intrinsics(
            H=dcfg.img_height,
            W=dcfg.img_width,
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            device=depth.device,
            inverted_x=dcfg.inverted_x,
            inverted_y=dcfg.inverted_y,
        )  # (B, 3, H, W)

        # Convert to flattened per-pixel form: (B, N, 3)
        B, _, H, W = ray_dirs.shape
        ray_dirs = ray_dirs.permute(0, 2, 3, 1).contiguous().view(B, H * W, 3)

        # Add the Gaussian-layer axis: (B, N, 1, 3)
        ray_dirs = ray_dirs.unsqueeze(2)

        # depth is (B, N, K), so depth.unsqueeze(-1) -> (B, N, K, 1)
        xyz_camera = ray_dirs * depth.unsqueeze(-1) + offset  # (B, N, K, 3)

        return xyz_camera, depth.unsqueeze(-1)  # keep depth as (B, N, K, 1)

    def forward(
        self,
        splatter: torch.Tensor,
        source_cameras_view_to_world: torch.Tensor,
        source_cv2wT_quat: torch.Tensor,
        intrinsics: torch.Tensor,
        activate_output: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert decoded splatter image to the Gaussian dictionary expected by render_predicted.
        """
        b, _, h, w = splatter.shape
        if (h, w) != (self.cfg.data.img_height, self.cfg.data.img_width):
            raise ValueError(
                f"Expected splatter of size {(self.cfg.data.img_height, self.cfg.data.img_width)}, "
                f"got {(h, w)}."
            )

        pieces = self._split(splatter)

        if self.cfg.model.max_sh_degree == 0:
            depth_map, offset_map, opacity_map, scaling_map, rotation_map, feat_dc_map = pieces
            feat_rest_map = None
        else:
            depth_map, offset_map, opacity_map, scaling_map, rotation_map, feat_dc_map, feat_rest_map = pieces

        # (B, K*C, H, W) -> (B, N, K, C)
        depth_logits = self._map_to_ordered_tensor(depth_map, 1)
        offset = self._map_to_ordered_tensor(offset_map, 3)
        opacity_logits = self._map_to_ordered_tensor(opacity_map, 1)
        scaling_raw = self._map_to_ordered_tensor(scaling_map, 3)
        rotation_raw = self._map_to_ordered_tensor(rotation_map, 4)
        features_dc = self._map_to_ordered_tensor(feat_dc_map, 3).unsqueeze(-2)  # (B, N, K, 1, 3)

        if feat_rest_map is not None:
            sh_rest = (((self.cfg.model.max_sh_degree + 1) ** 2) - 1) * 3
            features_rest = self._map_to_ordered_tensor(feat_rest_map, sh_rest)
            features_rest = features_rest.view(b, h * w, self.cfg.model.num_gaussians_per_pixel, -1, 3)
        else:
            features_rest = torch.zeros(
                b,
                h * w,
                self.cfg.model.num_gaussians_per_pixel,
                0,
                3,
                device=splatter.device,
                dtype=splatter.dtype,
            )

        # Build camera-space xyz + monotonic depth using old ordering logic
        xyz_camera, depth_cont = self._compute_xyz_camera(
            depth_logits=depth_logits,
            offset=offset,
            intrinsics=intrinsics,
            activate_output=activate_output,
        )

        # ------------------------------------------------------------------
        # Apply output activations and affine parameterization
        # ------------------------------------------------------------------

        # Apply output activations / affine parameterization
        if activate_output:
            mcfg = self.cfg.model

            opacity_pre = opacity_logits * mcfg.opacity_scale + mcfg.opacity_bias
            scaling_pre = scaling_raw * mcfg.scale_scale + mcfg.scale_bias

            opacity_pre = torch.nan_to_num(opacity_pre, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            scaling_pre = torch.nan_to_num(scaling_pre, nan=-10.0, posinf=4.0, neginf=-10.0).clamp(-10.0, 4.0)
            rotation_raw = torch.nan_to_num(rotation_raw, nan=0.0, posinf=0.0, neginf=0.0)

            opacity = self.opacity_activation(opacity_pre)
            scaling = self.scaling_activation(scaling_pre)
            rotation = self.rotation_activation(rotation_raw, dim=-1, eps=1e-6)
        else:
            opacity = opacity_logits
            scaling = scaling_raw
            rotation = rotation_raw

        # ------------------------------------------------------------------
        # Camera -> world transform
        # ------------------------------------------------------------------
        rot_c2w = source_cameras_view_to_world[:, None, None, :3, :3]     # (B,1,1,3,3)
        trans_c2w = source_cameras_view_to_world[:, None, None, :3, 3]    # (B,1,1,3)

        xyz_world = torch.matmul(
            xyz_camera.unsqueeze(-2),
            rot_c2w.transpose(-1, -2),
        ).squeeze(-2) + trans_c2w
        xyz_world = torch.nan_to_num(xyz_world, nan=0.0, posinf=0.0, neginf=0.0)

        # Rotate Gaussian orientation into world frame
        q_world = source_cv2wT_quat[:, None, None, :].expand_as(rotation)
        rotation_world = quaternion_raw_multiply(q_world, rotation)

        # Rotate SH coefficients into world frame if present
        if features_rest.shape[-2] > 0:
            features_rest_flat = features_rest.view(b, h * w * self.cfg.model.num_gaussians_per_pixel, features_rest.shape[-2], 3)
            features_rest_flat = transform_SHs(
                shs=features_rest_flat,
                sh_to_v_transform=self.sh_to_v_transform.to(features_rest_flat.device),
                v_to_sh_transform=self.v_to_sh_transform.to(features_rest_flat.device),
                source_cameras_to_world=source_cameras_view_to_world,
            )
        else:
            features_rest_flat = features_rest.view(b, h * w * self.cfg.model.num_gaussians_per_pixel, 0, 3)

        # Flatten to renderer format
        n_total = h * w * self.cfg.model.num_gaussians_per_pixel
        return {
            "xyz": xyz_world.view(b, n_total, 3).contiguous(),
            "xyz_camera": xyz_camera.view(b, n_total, 3).contiguous(),  # useful for debugging
            "rotation": rotation_world.view(b, n_total, 4).contiguous(),
            "opacity": opacity.view(b, n_total, 1).contiguous(),
            "scaling": scaling.view(b, n_total, 3).contiguous(),
            "features_dc": features_dc.view(b, n_total, 1, 3).contiguous(),
            "features_rest": features_rest_flat.contiguous(),
        }


# -----------------------------------------------------------------------------
# Renderer (intrinsics-based, mostly identical to original render_predicted)
# -----------------------------------------------------------------------------

def render_predicted(
    pc: Dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,   # (B, V, 4, 4) world -> view for each camera
    intrinsics: torch.Tensor,            # (B, V, 3, 3) pinhole Ks for each camera
    bg_color: torch.Tensor,              # (3,) or (B, V, 3) in [0,1]
    cfg: SplatterConfig,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
    packed: bool = False,
    render_mode: str = "RGB",
) -> Dict[str, torch.Tensor]:
    """
    Differentiable Gaussian splat rendering using **gsplat**.

    Assumptions (no shape juggling inside this function):
      - pc["xyz"]          : (B, N, 3)
      - pc["scaling"]      : (B, N, 3)
      - pc["rotation"]     : (B, N, 4)        # quaternions
      - pc["opacity"]      : (B, N, 1)
      - pc["features_dc"]  : (B, N, 1, 3)
      - pc["features_rest"]: (B, N, SH_rest, 3)  # possibly empty, can be missing

      - world_view_transform: (B, V, 4, 4)  world -> view for each of V cameras
      - intrinsics         : (B, V, 3, 3)  camera intrinsics K for each view
      - bg_color           : (3,) (same for all) or (B, V, 3) per view

    gsplat.rasterization is fully differentiable, so gradients flow from the
    rendered images back into the Gaussian parameters and, through them, into
    your network.
    """
    device = pc["xyz"].device
    H = cfg.data.img_height
    W = cfg.data.img_width

    # ------------------------------------------------------------------
    # 1) Basic Gaussian parameters
    # ------------------------------------------------------------------
    means = pc["xyz"]                                # (B, N, 3)
    scales = pc["scaling"] * scaling_modifier        # (B, N, 3)
    quats = pc["rotation"]                           # (B, N, 4)
    opacities = pc["opacity"].squeeze(-1)            # (B, N)
    means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e3, 1e3)
    scales = torch.nan_to_num(scales, nan=1e-4, posinf=1e2, neginf=1e-4).clamp(1e-5, 1e2)
    quats = F.normalize(torch.nan_to_num(quats, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1e-6)
    opacities = torch.nan_to_num(opacities, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # 2) Colors: SH (DC + rest) or override RGB
    # ------------------------------------------------------------------
    if override_color is not None:
        # Use override_color as plain RGB features (no SH)
        # Expected: override_color: (B, N, 3) or (B, N, D)
        colors = override_color
        sh_degree = None
    else:
        features_dc = pc["features_dc"]              # (B, N, 1, 3)
        features_rest = pc.get("features_rest", None)

        if features_rest is not None and features_rest.numel() > 0:
            # Concatenate DC + SH_rest → (B, N, K, 3)
            colors = torch.cat([features_dc, features_rest], dim=2)
            sh_degree = cfg.model.max_sh_degree      # e.g. 1
        else:
            # Only DC term: treat as SH degree 0
            colors = features_dc                     # (B, N, 1, 3)
            sh_degree = 0

    # ------------------------------------------------------------------
    # 3) Backgrounds: broadcast bg_color to (B, V, 3) if needed
    # ------------------------------------------------------------------
    if bg_color.dim() == 1:
        # Single RGB vector → expand to all batches/views
        B, V = world_view_transform.shape[0], world_view_transform.shape[1]
        backgrounds = bg_color.to(device).view(1, 1, 3).expand(B, V, 3)
    else:
        # Assume caller already provided (B, V, 3) or compatible
        backgrounds = bg_color.to(device)

    # ------------------------------------------------------------------
    # 4) Call gsplat rasterization
    # ------------------------------------------------------------------
    # Expected shapes:
    #   means        : (B, N, 3)
    #   quats        : (B, N, 4)
    #   scales       : (B, N, 3)
    #   opacities    : (B, N)
    #   colors       : (B, N, K, 3)   if SH
    #   viewmats     : (B, V, 4, 4)
    #   Ks           : (B, V, 3, 3)
    #
    # Returned:
    #   render_colors: (B, V, H, W, D)  (D = 3 for RGB)
    #   render_alphas: (B, V, H, W, 1)
    #   meta["radii"]: (B, V, N)
    render_colors, render_alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=world_view_transform,
        Ks=intrinsics,
        width=W,
        height=H,
        near_plane=cfg.data.znear,
        far_plane=cfg.data.zfar,
        sh_degree=sh_degree,
        backgrounds=backgrounds,
        packed=packed,
        render_mode=render_mode,
    )

    # ------------------------------------------------------------------
    # 5) Convert render to channel-first: (B, V, 3, H, W)
    # ------------------------------------------------------------------
    # render_colors: (B, V, H, W, 3) → (B, V, 3, H, W)
    rendered_image = render_colors.permute(0, 1, 4, 2, 3).contiguous()

    # ------------------------------------------------------------------
    # 6) Radii and visibility filter
    # ------------------------------------------------------------------
    radii = meta.get("radii", None)  # (B, V, N)
    visibility_filter = radii > 0 if radii is not None else None

    # We don’t need screenspace_points for gsplat; keep a placeholder
    # to satisfy old call sites if needed.
    viewspace_points = None

    return {
        "render": rendered_image,         # (B, V, 3, H, W)
        "viewspace_points": viewspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
    }


# -----------------------------------------------------------------------------
# Helper for decoder construction
# -----------------------------------------------------------------------------


def default_splatter_channels(
    max_sh_degree: int = 1,
    num_gaussians_per_pixel: int = 5,
) -> int:
    """
    Helper for decoder construction:
    return the required decoder channel count for direct K-Gaussian prediction.
    """
    spl_cfg = SplatterConfig(
        data=SplatterDataConfig(),
        model=SplatterModelConfig(
            max_sh_degree=max_sh_degree,
            num_gaussians_per_pixel=num_gaussians_per_pixel,
        ),
    )
    return VAESplatterToGaussians(spl_cfg).num_splatter_channels()
