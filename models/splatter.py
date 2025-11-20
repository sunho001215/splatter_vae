import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from einops import rearrange

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer
)

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
    max_sh_degree: int = 1           # we assume 0 or 1, see asserts below
    isotropic: bool = True          # if True: same scale for xyz

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
        Channel splits identical to "with offset" network:
          depth(1), offset(3), opacity(1), scaling(3),
          rotation(4), features_dc(3), [features_rest(3*SH_rest)]
        """
        split_dimensions = [1, 3, 1, 3, 4, 3]  # depth, offset, opacity, scaling, rotation, features_dc

        if self.cfg.model.max_sh_degree != 0:
            sh_num = (self.cfg.model.max_sh_degree + 1) ** 2 - 1  # for L=1 -> 3
            split_dimensions.append(sh_num * 3)  # SH_rest * 3 (RGB)

        return tuple(split_dimensions)

    def num_splatter_channels(self) -> int:
        return sum(self.get_split_dimensions())

    # --------- ray dirs helper -----------------------------------------------

    def _get_ray_dirs(
        self,
        device: torch.device,
        intrinsics: torch.Tensor
    ) -> torch.Tensor:
        """
        Get ray directions in camera coordinates.

        If `intrinsics` is provided:
            - (3,3): single K matrix -> returns (1,3,H,W)
            - (B,3,3): per-sample intrinsics -> returns (B,3,H,W)
        Otherwise, fall back to cfg.data.{fx,fy,cx,cy} and cached rays
        (shape (1,3,H,W), later broadcast to batch).
        """
        H = self.cfg.data.img_height
        W = self.cfg.data.img_width

        # Per-call intrinsics override
        if intrinsics is not None:
            if intrinsics.dim() == 2:
                # Single K
                Ks = intrinsics.unsqueeze(0)  # (1,3,3)
            elif intrinsics.dim() == 3:
                # Per-sample Ks
                Ks = intrinsics  # (B,3,3)
            else:
                raise ValueError("intrinsics must be (3,3) or (B,3,3).")

            ray_dirs_list = []
            for K in Ks:
                fx = float(K[0, 0].item())
                fy = float(K[1, 1].item())
                cx = float(K[0, 2].item())
                cy = float(K[1, 2].item())

                ray_dirs_k = build_ray_dirs_from_intrinsics(
                    H, W, fx, fy, cx, cy, device,
                    inverted_x=self.cfg.data.inverted_x,
                    inverted_y=self.cfg.data.inverted_y,
                )  # (1,3,H,W)
                ray_dirs_list.append(ray_dirs_k)

            # (B,3,H,W)
            return torch.cat(ray_dirs_list, dim=0)

        return self.ray_dirs  # (1,3,H,W)

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

        # Ray directions (1,3,H,W) in camera frame
        ray_dirs_xy = self._get_ray_dirs(splatter.device, intrinsics=intrinsics)

        # Split channels according to layout
        splits = self.get_split_dimensions()
        split_tensors = torch.split(splatter, splits, dim=1)

        depth_logits      = split_tensors[0]  # (B,1,H,W)
        offset            = split_tensors[1]  # (B,3,H,W)
        opacity_logits    = split_tensors[2]  # (B,1,H,W)
        scaling_logits    = split_tensors[3]  # (B,3,H,W)
        rotation_raw      = split_tensors[4]  # (B,4,H,W)
        features_dc       = split_tensors[5]  # (B,3,H,W)
        features_rest_raw = split_tensors[6] if len(split_tensors) > 6 else None

        # ---- depth in [znear, zfar] -----------------------------------------
        if activate_output:
            depth = self.depth_act(depth_logits) * (self.cfg.data.zfar - self.cfg.data.znear) + self.cfg.data.znear
        else:
            depth = depth_logits  # raw logits if you want to regularize differently

        # ---- xyz in camera coordinates: ray_dirs * depth + offset -----------
        pos_cam = ray_dirs_xy * depth + offset  # (B,3,H,W)

        # ---- convert xyz_cam -> xyz_world -----------------------------------
        # Flatten to (B,N,3)
        pos = flatten_vector(pos_cam)  # (B,N,3)
        pos_h = torch.cat(
            [pos, torch.ones((B, pos.shape[1], 1), device=pos.device, dtype=pos.dtype)],
            dim=2,
        )  # (B,N,4)

        # source_cameras_view_to_world: (B,4,4) row-major
        pos_world_h = torch.bmm(pos_h, source_cameras_view_to_world)  # (B,N,4)
        pos_world = pos_world_h[:, :, :3] / (pos_world_h[:, :, 3:].clamp(min=1e-10))  # (B,N,3)

        # ---- scaling ---------------------------------------------------------
        if self.cfg.model.isotropic:
            s = scaling_logits[:, :1, ...]
            scaling_logits = torch.cat([s, s, s], dim=1)

        if activate_output:
            scaling = self.scaling_activation(scaling_logits)  # positive scales
        else:
            scaling = scaling_logits
        scaling = flatten_vector(scaling)  # (B,N,3)

        # ---- opacity ---------------------------------------------------------
        if activate_output:
            opacity = self.opacity_activation(opacity_logits)
        else:
            opacity = opacity_logits
        opacity = flatten_vector(opacity)  # (B,N,1)

        # ---- rotations: normalize quaternions & convert to world frame -------
        rotation = self.rotation_activation(rotation_raw, dim=1)  # (B,4,H,W)
        rotation = flatten_vector(rotation)  # (B,N,4)
        rotation_world = transform_rotations(rotation, source_cv2wT_quat)  # (B,N,4)

        # ---- features_dc & SH rest -------------------------------------------
        features_dc_flat = flatten_vector(features_dc)  # (B,N,3)
        features_dc_flat = features_dc_flat.unsqueeze(2)  # (B,N,1,3)

        if self.cfg.model.max_sh_degree > 0 and features_rest_raw is not None:
            features_rest_flat = flatten_vector(features_rest_raw)  # (B,N,rest*3)
            features_rest = features_rest_flat.reshape(
                features_rest_flat.shape[0],
                features_rest_flat.shape[1],
                -1,
                3,
            )  # (B,N,SH_rest,3)

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
    world_view_transform: torch.Tensor,
    full_proj_transform: torch.Tensor,
    camera_center: torch.Tensor,
    bg_color: torch.Tensor,
    cfg: SplatterConfig,
    intrinsics: Optional[torch.Tensor] = None,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Render the scene as specified by the dictionary of Gaussians.

    Args:
        pc: dict with keys:
            "xyz":          (B,N,3)
            "opacity":      (B,N,1)
            "scaling":      (B,N,3)
            "rotation":     (B,N,4)
            "features_dc":  (B,N,1,3)
            "features_rest":(B,N,SH_rest,3)   # may be empty if no SH
        world_view_transform: (4,4) matrix for target camera (world->view)
        full_proj_transform:  (4,4) projection matrix
        camera_center:        (3,) camera center in world space
        bg_color:             (3,) tensor (on any device; moved as needed)
        cfg:                  SplatterConfig
        intrinsics:           optional K for this *target* camera
                              (for tanfovx/tanfovy; see below)
        scaling_modifier:     optional global scale factor
        override_color:       if not None, precomputed RGB instead of SH

    Returns:
        dict with:
          "render":          (B,3,H,W)
          "viewspace_points":(B,N,3) placeholder for gradients
          "visibility_filter":(B,N) visibility mask
          "radii":           (B,N) projected radii
    """
    means3D = pc["xyz"]  # (B,N,3)
    if means3D.dim() == 2:
        means3D = means3D.unsqueeze(0)
    B, N, _ = means3D.shape
    device = means3D.device

    # Screen-space placeholder (for gradient wrt 2D positions)
    screenspace_points = torch.zeros_like(means3D, device=device, requires_grad=True)
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Background
    bg_color = bg_color.to(device)

    H = cfg.data.img_height
    W = cfg.data.img_width

    # --- Intrinsics -> tan(fov) conversion for rasterizer -------------------
    #
    # The GaussianRasterizer expects tan(fov_x/2) and tan(fov_y/2). If we
    # know fx, fy in pixels, then:
    #
    #   tan(fov_x / 2) = (W / 2) / fx
    #   tan(fov_y / 2) = (H / 2) / fy
    #
    if intrinsics is not None:
        if intrinsics.dim() == 2:
            K = intrinsics
        elif intrinsics.dim() == 3:
            K = intrinsics[0]
        else:
            raise ValueError("intrinsics must be (3,3) or (B,3,3).")

        fx = float(K[0, 0].item())
        fy = float(K[1, 1].item())
    else:
        fx = cfg.data.fx
        fy = cfg.data.fy

    tanfovx = (W * 0.5) / fx
    tanfovy = (H * 0.5) / fy

    # Rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=int(H),
        image_width=int(W),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform.to(device),
        projmatrix=full_proj_transform.to(device),
        sh_degree=cfg.model.max_sh_degree,
        campos=camera_center.to(device),
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    scales = pc["scaling"]        # (B,N,3)
    rotations = pc["rotation"]    # (B,N,4)
    opacities = pc["opacity"]     # (B,N,1)

    # SH features
    if override_color is None:
        if "features_rest" in pc and pc["features_rest"].shape[2] > 0:
            shs = torch.cat([pc["features_dc"], pc["features_rest"]], dim=2).contiguous()
        else:
            shs = pc["features_dc"]
        colors_precomp = None
    else:
        shs = None
        colors_precomp = override_color

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    return {
        "render": rendered_image,            # (B,3,H,W)
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }