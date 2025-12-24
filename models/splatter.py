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

class VAESplatterToGaussians(nn.Module):
    """
    Takes the decoder's splatter image tensor and converts it into
    a dictionary of 3D Gaussian parameters, matching what
    `render_predicted` expects.

    Channel layout is assumed to follow the "with offset" network from
    the Splatter Image / GaussianSplatPredictor setup:

        [ depth(1),
          offset(3),
          opacity(1),
          scaling(3),
          rotation(4),
          features_dc(3),
          features_rest(3 * SH_rest)
        ]

    where for L = 1:
        SH_rest = (max_sh_degree + 1)^2 - 1 = 3,
    so features_rest has 3 * 3 = 9 channels.

    The decoder (InvariantDependentSplatterVAE) should be configured
    with:

        splatter_channels = self.num_splatter_channels()
    """

    def __init__(self, cfg: SplatterConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.model.max_sh_degree in (0, 1), \
            "VAESplatterToGaussians currently supports max_sh_degree ∈ {0,1}."
        assert cfg.model.num_gaussians_per_pixel >= 1, \
            "num_gaussians_per_pixel must be >= 1."

        self.depth_act = nn.Sigmoid()
        self.opacity_activation = torch.sigmoid
        self.scaling_activation = torch.exp
        self.rotation_activation = nn.functional.normalize

        # Ray directions will be lazily built from intrinsics and cached
        # if intrinsics are constant. If per-batch intrinsics are used,
        # we rebuild per forward call.
        self.register_buffer("ray_dirs", None, persistent=False)

        # SH transforms (for L = 1)
        if self.cfg.model.max_sh_degree > 0:
            sh_to_v, v_to_sh = init_sh_transform_matrices(
                device=torch.device("cpu"),
                max_sh_degree=self.cfg.model.max_sh_degree,
            )
            self.register_buffer("sh_to_v_transform", sh_to_v)
            self.register_buffer("v_to_sh_transform", v_to_sh)
        else:
            self.register_buffer("sh_to_v_transform", None)
            self.register_buffer("v_to_sh_transform", None)

    # --------- split layout ---------------------------------------------------

    def get_split_dimensions(self) -> Tuple[int, ...]:
        """
        Channel splits for depth-ordered K layers per pixel:
          depth(K), offset(3K), opacity(K), scaling(3K),
          rotation(4K), features_dc(3K), [features_rest(K * 3 * SH_rest)]

        Each pixel has K Gaussians with a canonical (front-to-back) ordering.
        """
        K = int(self.cfg.model.num_gaussians_per_pixel)
        split_dimensions = [
            1 * K,  # depth
            3 * K,  # offset
            1 * K,  # opacity
            3 * K,  # scaling
            4 * K,  # rotation
            3 * K,  # features_dc
        ]

        if self.cfg.model.max_sh_degree != 0:
            sh_num = (self.cfg.model.max_sh_degree + 1) ** 2 - 1  # for L=1 -> 3
            split_dimensions.append(K * sh_num * 3)  # K * SH_rest * 3 (RGB)

        return tuple(split_dimensions)

    def num_splatter_channels(self) -> int:
        return sum(self.get_split_dimensions())

    # --------- ray dirs helper -----------------------------------------------

    def _get_ray_dirs(
        self,
        device: torch.device,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get ray directions in camera coordinates.

        If `intrinsics` is provided:
            - (3,3): single K -> returns (1,3,H,W)
            - (B,3,3): per-sample Ks -> returns (B,3,H,W)

        This implementation keeps gradients w.r.t. intrinsics.
        """
        H = self.cfg.data.img_height
        W = self.cfg.data.img_width

        if intrinsics is not None:
            if intrinsics.dim() == 2:
                # Single K
                Ks = intrinsics.unsqueeze(0)  # (1,3,3)
            elif intrinsics.dim() == 3:
                Ks = intrinsics  # (B,3,3)
            else:
                raise ValueError("intrinsics must be (3,3) or (B,3,3).")

            # Extract fx, fy, cx, cy **as tensors** so autograd works
            fx = Ks[:, 0, 0]  # (B,)
            fy = Ks[:, 1, 1]  # (B,)
            cx = Ks[:, 0, 2]  # (B,)
            cy = Ks[:, 1, 2]  # (B,)

            # This call is now fully differentiable w.r.t. fx, fy, cx, cy
            ray_dirs = build_ray_dirs_from_intrinsics(
                H=H,
                W=W,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                device=device,
                inverted_x=self.cfg.data.inverted_x,
                inverted_y=self.cfg.data.inverted_y,
            )  # (B,3,H,W)
            return ray_dirs

        # Fallback to cached rays (non-parametric case)
        # self.ray_dirs should be (1,3,H,W) and can be broadcast later.
        return self.ray_dirs

    # --------- core forward ---------------------------------------------------

    def forward(
        self,
        splatter: torch.Tensor,
        source_cameras_view_to_world: torch.Tensor,
        source_cv2wT_quat: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        activate_output: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            splatter: (B, C_s, H, W) decoder output from InvariantDependentSplatterVAE
            source_cameras_view_to_world: (B, 4, 4) camera->world extrinsics
            source_cv2wT_quat: (B, 4) camera->world quaternion (rotation part)
            intrinsics: optional camera intrinsics K:
                          - (3,3)  or
                          - (B,3,3)
                        If None, fall back to cfg.data.{fx,fy,cx,cy}.
            activate_output: whether to apply activations (sigmoid/exp/etc)

        Returns:
            out_dict: {
                "xyz":          (B, N, 3),
                "rotation":     (B, N, 4),
                "opacity":      (B, N, 1),
                "scaling":      (B, N, 3),
                "features_dc":  (B, N, 1, 3),
                "features_rest":(B, N, SH_rest, 3)   # if max_sh_degree>0
            }
        """
        B, C, H, W = splatter.shape
        assert H == self.cfg.data.img_height and W == self.cfg.data.img_width, \
            f"Splatter resolution ({H},{W}) does not match cfg.data ({self.cfg.data.img_height},{self.cfg.data.img_width})"

        model_cfg = self.cfg.model

        # Ray directions (1,3,H,W) in camera frame or (B,3,H,W) if per-sample Ks
        ray_dirs_xy = self._get_ray_dirs(splatter.device, intrinsics=intrinsics)

        # Split channels according to layout
        splits = self.get_split_dimensions()
        split_tensors = torch.split(splatter, splits, dim=1)

        # Reshape splite tensors into explicit K-layer structure.
        # Each becomes (B, K, ... , H, W)
        K = int(model_cfg.num_gaussians_per_pixel)
        depth_logits   = split_tensors[0].reshape(B, K, H, W)
        offset         = split_tensors[1].reshape(B, K, 3, H, W)
        opacity_logits = split_tensors[2].reshape(B, K, 1, H, W)
        scaling_logits = split_tensors[3].reshape(B, K, 3, H, W)
        rotation_raw   = split_tensors[4].reshape(B, K, 4, H, W)
        features_dc    = split_tensors[5].reshape(B, K, 3, H, W)
        features_rest_raw = None
        if self.cfg.model.max_sh_degree != 0:
            features_rest_raw = split_tensors[6].reshape(B, K, -1, H, W)

        # ---- depth in [znear, zfar] with affine pre-transform ----------------
        if activate_output:
            # 1) Use the first depth value as the base
            # 2) For k > 0: depth[k] = depth[k-1] + exp(depth_pre[k]) to ensure monotonicity
            depth_pre = depth_logits * model_cfg.depth_scale + model_cfg.depth_bias

            if K == 1:
                depth_stack = depth_pre
            else:
                base = depth_pre[:, :1, ...]  # (B,1,H,W)
                inc = torch.exp(depth_pre[:, 1:, ...])              # (B,K-1,H,W)
                depth_tail = base + torch.cumsum(inc, dim=1)        # (B,K-1,H,W)
                depth_stack = torch.cat([base, depth_tail], dim=1)  # (B,K,H,W)
            
            depth = self.depth_act(depth_stack) * (self.cfg.data.zfar - self.cfg.data.znear) + self.cfg.data.znear
        else:
            depth = depth_logits  # raw logits if you want to regularize differently

        # ---- offset / xyz in camera coordinates ------------------------------
        # Apply xyz_scale / xyz_bias to the offset branch
        offset_cam = offset * model_cfg.xyz_scale + model_cfg.xyz_bias  # (B,K,3,H,W)
        ray_dirs_expanded = ray_dirs_xy.unsqueeze(1)                    # (B,1,3,H,W)
        pos_cam = ray_dirs_expanded * depth.unsqueeze(2) + offset_cam   # (B,K,3,H,W)

        # ---- convert xyz_cam -> xyz_world -----------------------------------
        # Flatten to (B,N,3)
        pos = pos_cam.permute(0, 1, 3, 4, 2).reshape(B, K * H * W, 3)  # (B,N,3)
        pos_h = torch.cat(
            [pos, torch.ones((B, pos.shape[1], 1), device=pos.device, dtype=pos.dtype)],
            dim=2,
        )  # (B,N,4)

        # source_cameras_view_to_world: (B,4,4) row-major
        pos_world_h = torch.bmm(pos_h, source_cameras_view_to_world)  # (B,N,4)
        pos_world = pos_world_h[:, :, :3] / (pos_world_h[:, :, 3:].clamp(min=1e-10))  # (B,N,3)

        # ---- scaling (log-space affine + exp) --------------------------------
        if self.cfg.model.isotropic:
            s = scaling_logits[ :, :, :1, ...]  # (B,K,1,H,W)
            scaling_logits = torch.cat([s, s, s], dim=2)

        if activate_output:
            # log-scale affine: raw -> log_s
            log_scales = scaling_logits * model_cfg.scale_scale + model_cfg.scale_bias
            scaling = self.scaling_activation(log_scales)  # exp(log_s) -> positive scales
        else:
            scaling = scaling_logits
        scaling = scaling.permute(0, 1, 3, 4, 2).reshape(B, K * H * W, 3)  # (B,N,3)

        # ---- opacity (affine + sigmoid) --------------------------------------
        if activate_output:
            opacity_pre = opacity_logits * model_cfg.opacity_scale + model_cfg.opacity_bias
            opacity = self.opacity_activation(opacity_pre)
        else:
            opacity = opacity_logits
        opacity = opacity.permute(0, 1, 3, 4, 2).reshape(B, K * H * W, 1)  # (B,N,1)

        # ---- rotations: normalize quaternions & convert to world frame -------
        rotation = self.rotation_activation(rotation_raw, dim=2)  # (B,K,4,H,W)
        rotation = rotation.permute(0, 1, 3, 4, 2).reshape(B, K * H * W, 4)  # (B,N,4)
        rotation_world = transform_rotations(rotation, source_cv2wT_quat)  # (B,N,4)

        # ---- features_dc & SH rest -------------------------------------------
        features_dc_flat = features_dc.permute(0, 1, 3, 4, 2).reshape(B, K * H * W, 3)  # (B,N,3)
        features_dc_flat = features_dc_flat.unsqueeze(2)  # (B,N,1,3)

        if self.cfg.model.max_sh_degree > 0 and features_rest_raw is not None:
            sh_num = (self.cfg.model.max_sh_degree + 1) ** 2 - 1  # for L=1 -> 3
            fr = features_rest_raw.view(B, K, sh_num, 3, H, W)  # (B,K,SH_rest,3,H,W)
            features_rest = fr.permute(0, 1, 4, 5, 2, 3).reshape(B, K * H * W, sh_num, 3)  # (B,N,SH_rest,3)

            # transform SH from camera to world
            features_rest = transform_SHs(
                features_rest,
                sh_to_v_transform=self.sh_to_v_transform,
                v_to_sh_transform=self.v_to_sh_transform,
                source_cameras_to_world=source_cameras_view_to_world,
            )
        else:
            # no higher-order SH; allocate empty tensor
            features_rest = torch.zeros(
                (B, pos_world.shape[1], 0, 3),
                dtype=features_dc_flat.dtype,
                device=features_dc_flat.device,
            )

        out_dict = {
            "xyz": pos_world,                   # (B,N,3)
            "rotation": rotation_world,         # (B,N,4)
            "opacity": opacity,                 # (B,N,1)
            "scaling": scaling,                 # (B,N,3)
            "features_dc": features_dc_flat,    # (B,N,1,3)
            "features_rest": features_rest,     # (B,N,SH_rest,3)
        }
        return out_dict


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
