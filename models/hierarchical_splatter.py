from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gsplat.rendering import rasterization


# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------


@dataclass
class SplatterDataConfig:
    img_height: int = 128
    img_width: int = 128
    znear: float = 0.1
    zfar: float = 2.0
    white_background: bool = False
    inverted_x: bool = False
    inverted_y: bool = False
    category: str = "generic"


@dataclass
class SplatterModelConfig:
    # Parent splatter image parameterization (single Gaussian per pixel)
    max_sh_degree: int = 1
    isotropic: bool = False
    depth_scale: float = 1.0
    depth_bias: float = 0.0
    xyz_scale: float = 1.0
    xyz_bias: float = 0.0
    opacity_scale: float = 1.0
    opacity_bias: float = 0.0
    scale_scale: float = 1.0
    scale_bias: float = 1.0

    # Child Gaussian hierarchy
    num_child_gaussians: int = 3
    child_warmup_steps: int = 10_000

    # CNN child predictor hyperparameters
    child_cnn_hidden_dim: int = 64
    child_cnn_num_blocks: int = 3
    child_cnn_kernel_size: int = 3
    child_cnn_use_depthwise_separable: bool = True


@dataclass
class SplatterConfig:
    data: SplatterDataConfig
    model: SplatterModelConfig


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------


def build_ray_dirs_from_intrinsics(
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    inverted_x: bool = False,
    inverted_y: bool = False,
) -> torch.Tensor:
    """Build differentiable OpenCV-style ray directions from pinhole intrinsics.

    Args:
        intrinsics: (3, 3) or (B, 3, 3)
    Returns:
        ray_dirs: (B, 3, H, W)
    """
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)
    b = intrinsics.shape[0]
    device = intrinsics.device
    dtype = intrinsics.dtype

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    ys = torch.arange(height, device=device, dtype=dtype) + 0.5
    xs = xs.view(1, 1, width).expand(b, height, width)
    ys = ys.view(1, height, 1).expand(b, height, width)

    fx = fx.view(b, 1, 1)
    fy = fy.view(b, 1, 1)
    cx = cx.view(b, 1, 1)
    cy = cy.view(b, 1, 1)

    x = (xs - cx) / fx
    y = (ys - cy) / fy
    if inverted_x:
        x = -x
    if inverted_y:
        y = -y
    z = torch.ones_like(x)
    return torch.stack([x, y, z], dim=1)



def flatten_image_map(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, C)."""
    return x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2).contiguous()



def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product for quaternions in (w, x, y, z) format."""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )



def matrix_to_quaternion(mat: torch.Tensor) -> torch.Tensor:
    """Convert a rotation matrix to a quaternion in (w, x, y, z) format."""
    m00 = mat[..., 0, 0]
    m11 = mat[..., 1, 1]
    m22 = mat[..., 2, 2]
    trace = m00 + m11 + m22

    qw = torch.sqrt(torch.clamp(1.0 + trace, min=1e-8)) / 2.0
    qx = (mat[..., 2, 1] - mat[..., 1, 2]) / (4.0 * qw + 1e-8)
    qy = (mat[..., 0, 2] - mat[..., 2, 0]) / (4.0 * qw + 1e-8)
    qz = (mat[..., 1, 0] - mat[..., 0, 1]) / (4.0 * qw + 1e-8)
    return torch.stack([qw, qx, qy, qz], dim=-1)



def transform_rotations(rotations: torch.Tensor, source_cv2wT_quat: torch.Tensor) -> torch.Tensor:
    """Transform predicted camera-frame quaternions into world-frame quaternions."""
    q = source_cv2wT_quat.unsqueeze(1).expand_as(rotations)
    return quaternion_raw_multiply(q, rotations)



def init_sh_transform_matrices(device: torch.device, max_sh_degree: int):
    if max_sh_degree <= 0:
        return None, None
    v_to_sh_transform = torch.tensor(
        [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=device
    )
    sh_to_v_transform = v_to_sh_transform.transpose(0, 1)
    return sh_to_v_transform.unsqueeze(0), v_to_sh_transform.unsqueeze(0)



def transform_SHs(
    shs: torch.Tensor,
    sh_to_v_transform: torch.Tensor,
    v_to_sh_transform: torch.Tensor,
    source_cameras_to_world: torch.Tensor,
) -> torch.Tensor:
    """Rotate SH1 coefficients from camera frame into world frame."""
    assert shs.shape[2] == 3, "This helper only supports SH degree 1."
    b, n, sh_num, rgb = shs.shape
    shs = shs.reshape(b, n * rgb, sh_num)
    rot = source_cameras_to_world[:, :3, :3]
    transforms = sh_to_v_transform.expand(b, 3, 3) @ rot @ v_to_sh_transform.expand(b, 3, 3)
    shs = shs @ transforms
    return shs.reshape(b, n, sh_num, rgb).contiguous()



def camera_to_world_quaternion(source_cameras_view_to_world: torch.Tensor) -> torch.Tensor:
    """Get (w, x, y, z) quaternions from row-major camera->world matrices."""
    rot = source_cameras_view_to_world[:, :3, :3].transpose(1, 2).contiguous()
    return matrix_to_quaternion(rot)


# -----------------------------------------------------------------------------
# Parent splatter-image converter
# -----------------------------------------------------------------------------


class ParentSplatterToGaussians(nn.Module):
    """Convert a vanilla 24-channel splatter image into one parent Gaussian per pixel.

    Channel layout matches the original Splatter Image parameterization:
        depth(1), offset(3), opacity(1), scaling(3), rotation(4), features_dc(3), features_rest(9)

    There is no depth ordering here: one pixel -> one parent Gaussian.
    """

    def __init__(self, cfg: SplatterConfig):
        super().__init__()
        self.cfg = cfg
        self.depth_act = nn.Sigmoid()
        self.opacity_activation = torch.sigmoid
        self.scaling_activation = torch.exp
        self.rotation_activation = nn.functional.normalize

        sh_to_v, v_to_sh = init_sh_transform_matrices(torch.device("cpu"), cfg.model.max_sh_degree)
        self.register_buffer("sh_to_v_transform", sh_to_v, persistent=False)
        self.register_buffer("v_to_sh_transform", v_to_sh, persistent=False)

    @staticmethod
    def num_parent_channels(max_sh_degree: int = 1) -> int:
        sh_rest = 0 if max_sh_degree == 0 else (((max_sh_degree + 1) ** 2) - 1) * 3
        return 1 + 3 + 1 + 3 + 4 + 3 + sh_rest

    def _split(self, splatter: torch.Tensor):
        if self.cfg.model.max_sh_degree == 0:
            return splatter.split((1, 3, 1, 3, 4, 3), dim=1)
        return splatter.split((1, 3, 1, 3, 4, 3, 9), dim=1)

    def _parent_camera_positions(
        self,
        depth_logits: torch.Tensor,
        offset: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Compute parent Gaussian centers in the source camera coordinate system.

        This now uses the affine scale/bias parameters from cfg.model:
        - depth_scale, depth_bias
        - xyz_scale, xyz_bias
        """
        dcfg = self.cfg.data
        mcfg = self.cfg.model

        # Apply affine transform to the depth logits before mapping them into [znear, zfar].
        depth_pre = depth_logits * mcfg.depth_scale + mcfg.depth_bias
        depth = self.depth_act(depth_pre) * (dcfg.zfar - dcfg.znear) + dcfg.znear

        # Apply affine transform to the predicted xyz offset in camera space.
        # Since xyz_bias is a scalar config value, it is broadcast to x/y/z equally.
        offset = offset * mcfg.xyz_scale + mcfg.xyz_bias

        ray_dirs = build_ray_dirs_from_intrinsics(
            intrinsics=intrinsics,
            height=dcfg.img_height,
            width=dcfg.img_width,
            inverted_x=dcfg.inverted_x,
            inverted_y=dcfg.inverted_y,
        )
        return ray_dirs * depth + offset

    def forward(
        self,
        splatter: torch.Tensor,
        source_cameras_view_to_world: torch.Tensor,
        source_cv2wT_quat: torch.Tensor,
        intrinsics: torch.Tensor,
        activate_output: bool = True,
    ) -> Dict[str, torch.Tensor]:
        b, _, h, w = splatter.shape
        if (h, w) != (self.cfg.data.img_height, self.cfg.data.img_width):
            raise ValueError("ParentSplatterToGaussians got an image with the wrong spatial size.")

        outputs = self._split(splatter)
        depth_logits, offset, opacity_logits, scaling_raw, rotation_raw, features_dc = outputs[:6]
        features_rest_raw = outputs[6] if len(outputs) > 6 else None

        # 1) Predict parent mean in source camera coordinates.
        pos_cam = self._parent_camera_positions(depth_logits, offset, intrinsics)
        pos_cam = flatten_image_map(pos_cam)
        pos_h = torch.cat(
            [
                pos_cam,
                torch.ones(b, pos_cam.shape[1], 1, device=pos_cam.device, dtype=pos_cam.dtype),
            ],
            dim=-1,
        )
        pos_world_h = pos_h @ source_cameras_view_to_world
        pos_world = pos_world_h[..., :3] / (pos_world_h[..., 3:].clamp_min(1e-8))

        # 2) Parent covariance / opacity / appearance.
        if self.cfg.model.isotropic:
            scaling_raw = torch.cat([scaling_raw[:, :1], scaling_raw[:, :1], scaling_raw[:, :1]], dim=1)

        # Apply affine parameterization before the positive / bounded activations.
        scaling_pre = scaling_raw * self.cfg.model.scale_scale + self.cfg.model.scale_bias
        opacity_pre = opacity_logits * self.cfg.model.opacity_scale + self.cfg.model.opacity_bias

        scaling = self.scaling_activation(scaling_pre) if activate_output else scaling_pre
        opacity = self.opacity_activation(opacity_pre) if activate_output else opacity_pre

        rotation = self.rotation_activation(rotation_raw, dim=1)
        rotation = flatten_image_map(rotation)
        rotation = transform_rotations(rotation, source_cv2wT_quat)

        scaling = flatten_image_map(scaling)
        opacity = flatten_image_map(opacity)
        features_dc = flatten_image_map(features_dc).unsqueeze(2)

        if features_rest_raw is not None:
            features_rest = flatten_image_map(features_rest_raw).reshape(b, h * w, 3, 3)
            if self.sh_to_v_transform is not None and self.v_to_sh_transform is not None:
                features_rest = transform_SHs(
                    features_rest,
                    sh_to_v_transform=self.sh_to_v_transform.to(features_rest.device),
                    v_to_sh_transform=self.v_to_sh_transform.to(features_rest.device),
                    source_cameras_to_world=source_cameras_view_to_world,
                )
        else:
            features_rest = torch.zeros(b, h * w, 0, 3, device=splatter.device, dtype=splatter.dtype)

        # Store camera-space parent parameters too.
        parent_camera_xyz = flatten_image_map(pos_cam)
        parent_raw = {
            "depth_logits": flatten_image_map(depth_logits),
            "offset": flatten_image_map(offset),
            "rotation_camera": flatten_image_map(self.rotation_activation(rotation_raw, dim=1)),
            "scaling_raw": flatten_image_map(scaling_raw),
            "opacity_logits": flatten_image_map(opacity_logits),
            "features_dc": features_dc.squeeze(2),
        }

        return {
            "xyz": pos_world,
            "xyz_camera": parent_camera_xyz,
            "rotation": rotation,
            "opacity": opacity,
            "scaling": scaling,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "parent_raw": parent_raw,
            "parent_splatter_flat": flatten_image_map(splatter),
        }


# -----------------------------------------------------------------------------
# Child Gaussian prediction
# -----------------------------------------------------------------------------


class DepthwiseSeparableConv(nn.Module):
    """Small depthwise-separable conv block used to keep the child branch light."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.GroupNorm(num_groups=8 if channels >= 8 else 1, num_channels=channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class StandardConvBlock(nn.Module):
    """Fallback standard conv block if depthwise separable conv is disabled."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.GroupNorm(num_groups=8 if channels >= 8 else 1, num_channels=channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChildGaussianPredictor(nn.Module):
    """Predict target-view-dependent child Gaussians using a lightweight 2D CNN.

    Compared with the previous per-pixel MLP version, this module now reshapes the
    parent conditioning into a spatial feature map of shape (B, C, H, W) and runs
    local 2D convolutions on top of it. This means each pixel's child Gaussians can
    use information from neighboring *parent* Gaussians.

    Input channels per pixel:
        parent_splatter(24) + distance_to_target_camera(1) + direction_to_target_camera(3)
        = 28 channels when max_sh_degree=1.

    Output heads follow the hierarchical paper's factorization:
        - child center offsets      : 3 values per child
        - child covariance params   : 7 values per child = scaling(3) + rotation(4)
        - child RGB color           : 3 values per child
        - child opacity             : 1 value per child
    """

    def __init__(self, cfg: SplatterConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = ParentSplatterToGaussians.num_parent_channels(cfg.model.max_sh_degree) + 1 + 3
        hidden = cfg.model.child_cnn_hidden_dim
        k = cfg.model.num_child_gaussians
        kernel_size = cfg.model.child_cnn_kernel_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=8 if hidden >= 8 else 1, num_channels=hidden),
            nn.SiLU(inplace=True),
        )

        block_cls = DepthwiseSeparableConv if cfg.model.child_cnn_use_depthwise_separable else StandardConvBlock
        self.blocks = nn.Sequential(*[block_cls(hidden, kernel_size) for _ in range(cfg.model.child_cnn_num_blocks)])

        # Separate heads, mirroring the original paper's decomposition.
        self.head_offset = nn.Conv2d(hidden, k * 3, kernel_size=1, stride=1, padding=0)
        self.head_cov = nn.Conv2d(hidden, k * 7, kernel_size=1, stride=1, padding=0)
        self.head_color = nn.Conv2d(hidden, k * 3, kernel_size=1, stride=1, padding=0)
        self.head_opacity = nn.Conv2d(hidden, k * 1, kernel_size=1, stride=1, padding=0)

        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = nn.functional.normalize

        # Initialize the offset head to predict zero offsets
        nn.init.zeros_(self.head_offset.weight)
        nn.init.zeros_(self.head_offset.bias)

    def _reshape_parent_flat_to_map(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H*W, C) -> (B, C, H, W)."""
        b, n, c = x.shape
        h = self.cfg.data.img_height
        w = self.cfg.data.img_width
        if n != h * w:
            raise ValueError(f"Expected {h*w} parent pixels, got {n}")
        return x.transpose(1, 2).reshape(b, c, h, w).contiguous()

    def forward(
        self,
        parent_pc: Dict[str, torch.Tensor],
        target_camera_centers_world: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict K child Gaussians for every parent Gaussian using local 2D context.

        Args:
            parent_pc["xyz"]:                 (B, N, 3)
            parent_pc["parent_splatter_flat"]:(B, N, C_parent)
            target_camera_centers_world:      (B, 3)

        Returns:
            Child Gaussian dict, flattened to (B, N*K, ...).
        """
        b, n, _ = parent_pc["xyz"].shape
        h = self.cfg.data.img_height
        w = self.cfg.data.img_width
        k = self.cfg.model.num_child_gaussians
        eps = 1e-8

        # ------------------------------------------------------------------
        # 1) Build the same target-view conditioning as the hierarchical paper:
        #    distance and relative direction from each parent center to the
        #    target camera center in the world coordinate system.
        # ------------------------------------------------------------------
        parent_xyz = parent_pc["xyz"]
        camera_centers = target_camera_centers_world[:, None, :].expand(b, n, 3)
        rel = parent_xyz - camera_centers
        dst = torch.norm(rel, dim=-1, keepdim=True)
        direction = rel / (dst + eps)

        # ------------------------------------------------------------------
        # 2) Reshape parent features + target-view conditioning into a spatial
        #    map. This is the key modification: the child branch now runs a 2D
        #    CNN on top of these features, so neighboring parent Gaussians in
        #    image space can contribute to each pixel's child prediction.
        # ------------------------------------------------------------------
        child_in_flat = torch.cat([parent_pc["parent_splatter_flat"], dst, direction], dim=-1)
        child_in_map = self._reshape_parent_flat_to_map(child_in_flat)  # (B, C_in, H, W)

        feat = self.stem(child_in_map)
        feat = self.blocks(feat)

        # ------------------------------------------------------------------
        # 3) Predict all child attributes densely over the 2D grid.
        # ------------------------------------------------------------------
        offset_map = self.head_offset(feat)    # (B, K*3, H, W)
        cov_map = self.head_cov(feat)          # (B, K*7, H, W)
        color_map = self.head_color(feat)      # (B, K*3, H, W)
        opacity_map = self.head_opacity(feat)  # (B, K*1, H, W)

        # (B, K*C, H, W) -> (B, H*W, K, C)
        def map_to_child_tensor(x: torch.Tensor, channels_per_child: int) -> torch.Tensor:
            x = x.view(b, k, channels_per_child, h, w)
            x = x.permute(0, 3, 4, 1, 2).contiguous()
            return x.view(b, h * w, k, channels_per_child)

        offset = map_to_child_tensor(offset_map, 3)
        cov = map_to_child_tensor(cov_map, 7)
        color = map_to_child_tensor(color_map, 3)
        opacity = map_to_child_tensor(opacity_map, 1)
        
        # Apply same affine parameterization as the parent to child gaussians
        offset = offset * self.cfg.model.xyz_scale + self.cfg.model.xyz_bias
        child_xyz = parent_xyz.unsqueeze(2) + offset

        child_scaling_raw = cov[..., :3]
        child_rotation_raw = cov[..., 3:]

        child_scaling_pre = child_scaling_raw * self.cfg.model.scale_scale + self.cfg.model.scale_bias
        child_opacity_pre = opacity * self.cfg.model.opacity_scale + self.cfg.model.opacity_bias - 4.0 # Add extra bias to encourage initial child opacity to be near zero

        child_scaling = self.scaling_activation(child_scaling_pre)
        child_rotation = self.rotation_activation(child_rotation_raw, dim=-1)
        child_opacity = self.opacity_activation(child_opacity_pre)

        # As in the prior patch, children are RGB/DC only.
        child_features_dc = color.unsqueeze(-2)  # (B, N, K, 1, 3)
        child_features_rest = torch.zeros(
            b,
            h * w,
            k,
            0 if self.cfg.model.max_sh_degree == 0 else 3,
            3,
            dtype=child_features_dc.dtype,
            device=child_features_dc.device,
        )

        return {
            "xyz": child_xyz.reshape(b, h * w * k, 3).contiguous(),
            "rotation": child_rotation.reshape(b, h * w * k, 4).contiguous(),
            "opacity": child_opacity.reshape(b, h * w * k, 1).contiguous(),
            "scaling": child_scaling.reshape(b, h * w * k, 3).contiguous(),
            "features_dc": child_features_dc.reshape(b, h * w * k, 1, 3).contiguous(),
            "features_rest": child_features_rest.reshape(b, h * w * k, child_features_rest.shape[-2], 3).contiguous(),
        }


# -----------------------------------------------------------------------------
# Hierarchical wrapper
# -----------------------------------------------------------------------------


class HierarchicalSplatterToGaussians(nn.Module):
    """Parent + child hierarchical Gaussian construction.

    Usage pattern:
        1) build parents from the decoded splatter image,
        2) optionally predict children conditioned on a target render camera,
        3) merge both sets of Gaussians and send them to the renderer.
    """

    def __init__(self, cfg: SplatterConfig):
        super().__init__()
        self.cfg = cfg
        self.parent_converter = ParentSplatterToGaussians(cfg)
        self.child_predictor = ChildGaussianPredictor(cfg)

    @property
    def num_parent_channels(self) -> int:
        return ParentSplatterToGaussians.num_parent_channels(self.cfg.model.max_sh_degree)

    def forward_parent(
        self,
        splatter: torch.Tensor,
        source_cameras_view_to_world: torch.Tensor,
        source_cv2wT_quat: torch.Tensor,
        intrinsics: torch.Tensor,
        activate_output: bool = True,
    ) -> Dict[str, torch.Tensor]:
        return self.parent_converter(
            splatter=splatter,
            source_cameras_view_to_world=source_cameras_view_to_world,
            source_cv2wT_quat=source_cv2wT_quat,
            intrinsics=intrinsics,
            activate_output=activate_output,
        )

    def build_hierarchical_scene(
        self,
        parent_pc: Dict[str, torch.Tensor],
        target_camera_centers_world: torch.Tensor,
        include_children: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if not include_children:
            return {
                k: v
                for k, v in parent_pc.items()
                if k in {"xyz", "rotation", "opacity", "scaling", "features_dc", "features_rest"}
            }

        child_pc = self.child_predictor(parent_pc, target_camera_centers_world)
        return merge_gaussian_sets(
            parent={
                k: v
                for k, v in parent_pc.items()
                if k in {"xyz", "rotation", "opacity", "scaling", "features_dc", "features_rest"}
            },
            child=child_pc,
        )



def merge_gaussian_sets(parent: Dict[str, torch.Tensor], child: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Concatenate parent and child Gaussian sets along the point dimension."""
    keys = ["xyz", "rotation", "opacity", "scaling", "features_dc", "features_rest"]
    return {k: torch.cat([parent[k], child[k]], dim=1).contiguous() for k in keys}


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def render_predicted(
    pc: Dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    intrinsics: torch.Tensor,
    bg_color: torch.Tensor,
    cfg: SplatterConfig,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
    packed: bool = False,
    render_mode: str = "RGB",
) -> Dict[str, torch.Tensor]:
    """Differentiable rendering through gsplat.

    Shapes follow the batch-multiview style already used in your current code:
      - world_view_transform: (B, V, 4, 4)
      - intrinsics:          (B, V, 3, 3)
    """
    device = pc["xyz"].device
    h = cfg.data.img_height
    w = cfg.data.img_width

    means = pc["xyz"]
    scales = pc["scaling"] * scaling_modifier
    quats = pc["rotation"]
    opacities = pc["opacity"].squeeze(-1)

    if override_color is not None:
        colors = override_color
        sh_degree = None
    else:
        features_dc = pc["features_dc"]
        features_rest = pc.get("features_rest", None)
        if features_rest is not None and features_rest.numel() > 0:
            colors = torch.cat([features_dc, features_rest], dim=2)
            sh_degree = cfg.model.max_sh_degree
        else:
            colors = features_dc
            sh_degree = 0

    if bg_color.dim() == 1:
        b, v = world_view_transform.shape[0], world_view_transform.shape[1]
        backgrounds = bg_color.to(device).view(1, 1, 3).expand(b, v, 3)
    else:
        backgrounds = bg_color.to(device)

    render_colors, render_alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=world_view_transform,
        Ks=intrinsics,
        width=w,
        height=h,
        near_plane=cfg.data.znear,
        far_plane=cfg.data.zfar,
        sh_degree=sh_degree,
        backgrounds=backgrounds,
        packed=packed,
        render_mode=render_mode,
    )

    rendered_image = render_colors.permute(0, 1, 4, 2, 3).contiguous()
    radii = meta.get("radii", None)
    visibility_filter = radii > 0 if radii is not None else None
    return {
        "render": rendered_image,
        "viewspace_points": None,
        "visibility_filter": visibility_filter,
        "radii": radii,
    }
