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
    depth_parameterization: str = "absolute"
    depth_residual_scale: float = 0.1
    depth_increment_scale: float = 0.1
    depth_increment_min: float = 0.01
    depth_scale: float = 1.0
    depth_bias: float = 0.0
    xyz_scale: float = 1.0
    xyz_bias: float = 0.0
    opacity_scale: float = 1.0
    opacity_bias: float = 0.0
    scale_scale: float = 1.0
    scale_bias: float = 1.0
    scale_max: float = 0.3
    voxelize: bool = False
    voxel_size: float = 0.01


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
        confidence(K)
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
        depth_mode = str(getattr(cfg.model, "depth_parameterization", "absolute")).lower()
        assert depth_mode in ("absolute", "direct", "residual_unidepth", "depth_prior"), \
            f"Unsupported depth_parameterization={cfg.model.depth_parameterization!r}."

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
    def _num_predicted_depth_layers(self) -> int:
        k = int(self.cfg.model.num_gaussians_per_pixel)
        depth_mode = str(getattr(self.cfg.model, "depth_parameterization", "absolute")).lower()
        if depth_mode == "depth_prior":
            return max(k - 1, 0)
        return k

    def get_split_dimensions(self):
        k = int(self.cfg.model.num_gaussians_per_pixel)
        sh_rest = 0 if self.cfg.model.max_sh_degree == 0 else (((self.cfg.model.max_sh_degree + 1) ** 2) - 1) * 3

        dims = (
            self._num_predicted_depth_layers(),  # depth, or K-1 relative layers when using depth_prior
            3 * k,      # offset
            k,          # opacity
            k,          # confidence logits for optional voxel fusion
            (1 if bool(self.cfg.model.isotropic) else 3) * k,      # scaling
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
        num_items: Optional[int] = None,
    ) -> torch.Tensor:
        """
        (B, K*C, H, W) -> (B, H*W, K, C)
        """
        b, _, h, w = x.shape
        k = int(self.cfg.model.num_gaussians_per_pixel if num_items is None else num_items)

        x = x.view(b, k, channels_per_gaussian, h, w)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x.view(b, h * w, k, channels_per_gaussian)

    def _format_depth_prior(
        self,
        depth_prior: torch.Tensor,
        batch_size: int,
        num_pixels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalize a metric depth prior to ``(B, H*W, 1)`` camera-z depth."""
        dcfg = self.cfg.data
        if depth_prior is None:
            raise ValueError("depth_prior is required for depth_parameterization='depth_prior' or 'residual_unidepth'.")

        prior = depth_prior.to(device=device, dtype=dtype)
        if prior.dim() == 4:
            if prior.shape[1] != 1:
                raise ValueError(f"Expected depth_prior channels=1, got shape {tuple(prior.shape)}.")
            if prior.shape[-2:] != (dcfg.img_height, dcfg.img_width):
                prior = F.interpolate(
                    prior,
                    size=(dcfg.img_height, dcfg.img_width),
                    mode="bilinear",
                    align_corners=False,
                )
            prior = prior.flatten(2).transpose(1, 2).contiguous()
        elif prior.dim() == 3 and prior.shape[-2:] == (dcfg.img_height, dcfg.img_width):
            prior = prior.unsqueeze(1).flatten(2).transpose(1, 2).contiguous()
        elif prior.dim() == 3 and prior.shape[1:] == (num_pixels, 1):
            pass
        elif prior.dim() == 2 and prior.shape[1] == num_pixels:
            prior = prior.unsqueeze(-1)
        else:
            raise ValueError(
                "depth_prior must have shape (B,1,H,W), (B,H,W), (B,H*W), or (B,H*W,1); "
                f"got {tuple(prior.shape)}."
            )

        if prior.shape[0] != batch_size or prior.shape[1] != num_pixels:
            raise ValueError(
                f"Expected depth_prior batch/pixels {(batch_size, num_pixels)}, "
                f"got {tuple(prior.shape[:2])}."
            )

        prior = torch.nan_to_num(
            prior,
            nan=0.5 * (dcfg.znear + dcfg.zfar),
            posinf=dcfg.zfar,
            neginf=dcfg.znear,
        )
        return prior.clamp(min=dcfg.znear, max=dcfg.zfar)

    def _compute_xyz_camera(
        self,
        depth_logits: torch.Tensor,   # (B, N, K or K-1, 1)
        offset: torch.Tensor,         # (B, N, K, 3)
        intrinsics: torch.Tensor,     # (B, 3, 3)
        depth_prior: Optional[torch.Tensor] = None,
        activate_output: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build camera-space Gaussian centers for the direct multi-Gaussian path."""
        dcfg = self.cfg.data
        mcfg = self.cfg.model

        # depth_logits: (B, N, K or K-1, 1) -> (B, N, K or K-1)
        depth_logits = depth_logits.squeeze(-1)
        num_layers = int(mcfg.num_gaussians_per_pixel)
        depth_mode = str(getattr(mcfg, "depth_parameterization", "absolute")).lower()

        if activate_output and depth_mode == "depth_prior":
            base_depth = self._format_depth_prior(
                depth_prior=depth_prior,
                batch_size=offset.shape[0],
                num_pixels=offset.shape[1],
                device=offset.device,
                dtype=offset.dtype,
            ).squeeze(-1)
            first_depth = base_depth.unsqueeze(-1)

            if num_layers == 1:
                depth = first_depth
            else:
                expected = num_layers - 1
                if depth_logits.shape[2] != expected:
                    raise ValueError(
                        f"depth_prior mode expects {expected} predicted relative depth layers, "
                        f"got {depth_logits.shape[2]}."
                    )
                inc_pre = depth_logits.clamp(min=-10.0, max=10.0)
                inc = F.softplus(inc_pre) * float(mcfg.depth_increment_scale)
                inc = inc + float(mcfg.depth_increment_min)
                depth_tail = first_depth + torch.cumsum(inc, dim=2)
                depth = torch.cat([first_depth, depth_tail], dim=2)

            depth = torch.nan_to_num(depth, nan=dcfg.znear, posinf=dcfg.zfar, neginf=dcfg.znear)
            depth = depth.clamp(min=dcfg.znear, max=dcfg.zfar)
        elif activate_output and depth_mode == "residual_unidepth":
            base_depth = self._format_depth_prior(
                depth_prior=depth_prior,
                batch_size=depth_logits.shape[0],
                num_pixels=depth_logits.shape[1],
                device=depth_logits.device,
                dtype=depth_logits.dtype,
            ).squeeze(-1)
            residual = torch.tanh(depth_logits[:, :, :1]) * float(mcfg.depth_residual_scale)
            first_depth = base_depth.unsqueeze(-1) + residual
            if depth_logits.shape[2] == 1:
                depth = first_depth
            else:
                inc_pre = depth_logits[:, :, 1:].clamp(min=-10.0, max=10.0)
                inc = F.softplus(inc_pre) * float(mcfg.depth_increment_scale)
                inc = inc + float(mcfg.depth_increment_min)
                depth_tail = first_depth + torch.cumsum(inc, dim=2)
                depth = torch.cat([first_depth, depth_tail], dim=2)
            depth = torch.nan_to_num(depth, nan=dcfg.znear, posinf=dcfg.zfar, neginf=dcfg.znear)
            depth = depth.clamp(min=dcfg.znear, max=dcfg.zfar)
        elif activate_output:
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

        return xyz_camera, depth.unsqueeze(-1), offset  # depth: (B, N, K, 1), offset: (B, N, K, 3)


    def _pad_tensor_list(self, tensors: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
        max_items = max(t.shape[0] for t in tensors)
        padded = []
        for tensor in tensors:
            pad_len = max_items - tensor.shape[0]
            if pad_len > 0:
                pad = tensor.new_full((pad_len, *tensor.shape[1:]), float(pad_value))
                tensor = torch.cat((tensor, pad), dim=0)
            padded.append(tensor)
        return torch.stack(padded, dim=0)

    def _pad_rotation_list(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        max_items = max(t.shape[0] for t in tensors)
        padded = []
        for tensor in tensors:
            pad_len = max_items - tensor.shape[0]
            if pad_len > 0:
                pad = tensor.new_zeros((pad_len, 4))
                pad[:, 0] = 1.0
                tensor = torch.cat((tensor, pad), dim=0)
            padded.append(tensor)
        return torch.stack(padded, dim=0)

    def _voxelize_camera_gaussians(self, gaussian: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Fuse decoded Gaussians inside a source-camera-coordinate voxel grid."""
        voxel_size = float(getattr(self.cfg.model, "voxel_size", 0.01))
        if voxel_size <= 0.0:
            raise ValueError(f"voxel_size must be positive when voxelize=true, got {voxel_size}.")

        batch_size = gaussian["xyz_camera"].shape[0]
        fused: Dict[str, list[torch.Tensor]] = {
            "xyz_camera": [],
            "depth_camera": [],
            "offset_camera": [],
            "rotation_camera": [],
            "opacity": [],
            "scaling": [],
            "features_dc": [],
            "features_rest": [],
            "confidence_logits": [],
            "confidence": [],
        }
        valid_masks: list[torch.Tensor] = []
        voxel_counts: list[torch.Tensor] = []

        for batch_idx in range(batch_size):
            xyz = gaussian["xyz_camera"][batch_idx]
            finite_mask = torch.isfinite(xyz).all(dim=-1)
            if not bool(finite_mask.any()):
                finite_mask = torch.ones_like(finite_mask, dtype=torch.bool)

            xyz_valid = xyz[finite_mask]
            voxel_indices = torch.round(xyz_valid / voxel_size).to(torch.int64)
            _, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)
            num_voxels = int(inverse_indices.max().item()) + 1

            conf_logits = gaussian["confidence_logits"][batch_idx, finite_mask, 0]
            conf_logits = torch.nan_to_num(conf_logits.float(), nan=-30.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
            exp_conf = torch.exp(conf_logits - conf_logits.max()).to(xyz_valid.dtype)
            denom = exp_conf.new_zeros(num_voxels)
            denom.index_add_(0, inverse_indices, exp_conf)
            weights = exp_conf / denom[inverse_indices].clamp_min(1e-8)

            def reduce_tensor(name: str) -> torch.Tensor:
                values = gaussian[name][batch_idx, finite_mask]
                weight_shape = (weights.shape[0],) + (1,) * (values.ndim - 1)
                weighted = values * weights.to(values.dtype).view(weight_shape)
                out = values.new_zeros((num_voxels, *values.shape[1:]))
                out.index_add_(0, inverse_indices, weighted)
                return out

            fused["xyz_camera"].append(reduce_tensor("xyz_camera"))
            fused["depth_camera"].append(reduce_tensor("depth_camera"))
            fused["offset_camera"].append(reduce_tensor("offset_camera"))
            fused["rotation_camera"].append(F.normalize(reduce_tensor("rotation_camera"), dim=-1, eps=1e-6))
            fused["opacity"].append(reduce_tensor("opacity"))
            fused["scaling"].append(reduce_tensor("scaling"))
            fused["features_dc"].append(reduce_tensor("features_dc"))
            fused["features_rest"].append(reduce_tensor("features_rest"))
            fused_conf_logits = reduce_tensor("confidence_logits")
            fused["confidence_logits"].append(fused_conf_logits)
            fused["confidence"].append(torch.sigmoid(fused_conf_logits))
            valid_masks.append(torch.ones(num_voxels, device=xyz.device, dtype=torch.bool))
            count = xyz.new_zeros(num_voxels)
            count.index_add_(0, inverse_indices, torch.ones_like(weights))
            voxel_counts.append(count.unsqueeze(-1))

        return {
            "xyz_camera": self._pad_tensor_list(fused["xyz_camera"], 0.0),
            "depth_camera": self._pad_tensor_list(fused["depth_camera"], self.cfg.data.zfar),
            "offset_camera": self._pad_tensor_list(fused["offset_camera"], 0.0),
            "rotation_camera": self._pad_rotation_list(fused["rotation_camera"]),
            "opacity": self._pad_tensor_list(fused["opacity"], 0.0),
            "scaling": self._pad_tensor_list(fused["scaling"], 1e-4),
            "features_dc": self._pad_tensor_list(fused["features_dc"], 0.0),
            "features_rest": self._pad_tensor_list(fused["features_rest"], 0.0),
            "confidence_logits": self._pad_tensor_list(fused["confidence_logits"], -30.0),
            "confidence": self._pad_tensor_list(fused["confidence"], 0.0),
            "valid_mask": self._pad_tensor_list([m.unsqueeze(-1) for m in valid_masks], 0.0).squeeze(-1).bool(),
            "voxel_count": self._pad_tensor_list(voxel_counts, 0.0),
        }

    def forward(
        self,
        splatter: torch.Tensor,
        source_cameras_view_to_world: torch.Tensor,
        source_cv2wT_quat: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_prior: Optional[torch.Tensor] = None,
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
            depth_map, offset_map, opacity_map, confidence_map, scaling_map, rotation_map, feat_dc_map = pieces
            feat_rest_map = None
        else:
            depth_map, offset_map, opacity_map, confidence_map, scaling_map, rotation_map, feat_dc_map, feat_rest_map = pieces

        # (B, K*C, H, W) -> (B, N, K, C)
        if depth_map.shape[1] == 0:
            depth_logits = splatter.new_empty(b, h * w, 0, 1)
        else:
            depth_logits = self._map_to_ordered_tensor(
                depth_map,
                1,
                num_items=self._num_predicted_depth_layers(),
            )
        offset = self._map_to_ordered_tensor(offset_map, 3)
        opacity_logits = self._map_to_ordered_tensor(opacity_map, 1)
        confidence_logits = self._map_to_ordered_tensor(confidence_map, 1)
        scaling_raw = self._map_to_ordered_tensor(scaling_map, 1 if bool(self.cfg.model.isotropic) else 3)
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
        xyz_camera, depth_cont, offset_cont = self._compute_xyz_camera(
            depth_logits=depth_logits,
            offset=offset,
            intrinsics=intrinsics,
            depth_prior=depth_prior,
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
            confidence_pre = torch.nan_to_num(confidence_logits, nan=-30.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
            scaling_pre = torch.nan_to_num(scaling_pre, nan=-10.0, posinf=4.0, neginf=-10.0).clamp(-10.0, 4.0)
            rotation_raw = torch.nan_to_num(rotation_raw, nan=0.0, posinf=0.0, neginf=0.0)

            opacity = self.opacity_activation(opacity_pre)
            confidence = torch.sigmoid(confidence_pre)
            scaling = self.scaling_activation(scaling_pre)
            if bool(mcfg.isotropic):
                scaling = scaling.expand(*scaling.shape[:-1], 3)
            scale_max = max(float(getattr(mcfg, "scale_max", 0.3)), 1e-5)
            scaling = scaling.clamp(max=scale_max)
            rotation = self.rotation_activation(rotation_raw, dim=-1, eps=1e-6)
        else:
            opacity = opacity_logits
            confidence = confidence_logits
            scaling = scaling_raw
            if bool(self.cfg.model.isotropic):
                scaling = scaling.expand(*scaling.shape[:-1], 3)
            rotation = rotation_raw

        # ------------------------------------------------------------------
        # Optional source-camera voxel fusion, then camera -> world transform.
        # ------------------------------------------------------------------
        n_total = h * w * self.cfg.model.num_gaussians_per_pixel
        gaussian_camera = {
            "xyz_camera": xyz_camera.view(b, n_total, 3).contiguous(),
            "depth_camera": depth_cont.view(b, n_total, 1).contiguous(),
            "offset_camera": offset_cont.view(b, n_total, 3).contiguous(),
            "rotation_camera": rotation.view(b, n_total, 4).contiguous(),
            "opacity": opacity.view(b, n_total, 1).contiguous(),
            "scaling": scaling.view(b, n_total, 3).contiguous(),
            "features_dc": features_dc.view(b, n_total, 1, 3).contiguous(),
            "features_rest": features_rest.view(b, n_total, features_rest.shape[-2], 3).contiguous(),
            "confidence_logits": confidence_logits.view(b, n_total, 1).contiguous(),
            "confidence": confidence.view(b, n_total, 1).contiguous(),
            "valid_mask": torch.ones(b, n_total, device=splatter.device, dtype=torch.bool),
        }

        if bool(getattr(self.cfg.model, "voxelize", False)):
            gaussian_camera = self._voxelize_camera_gaussians(gaussian_camera)

        xyz_camera_flat = gaussian_camera["xyz_camera"]
        rotation_camera = gaussian_camera["rotation_camera"]
        rot_c2w = source_cameras_view_to_world[:, None, :3, :3]     # (B,1,3,3)
        trans_c2w = source_cameras_view_to_world[:, None, :3, 3]    # (B,1,3)

        xyz_world = torch.matmul(
            xyz_camera_flat.unsqueeze(-2),
            rot_c2w.transpose(-1, -2),
        ).squeeze(-2) + trans_c2w
        xyz_world = torch.nan_to_num(xyz_world, nan=0.0, posinf=0.0, neginf=0.0)

        # Rotate Gaussian orientation into world frame
        q_world = source_cv2wT_quat[:, None, :].expand_as(rotation_camera)
        rotation_world = quaternion_raw_multiply(q_world, rotation_camera)

        # Rotate SH coefficients into world frame if present
        features_rest_flat = gaussian_camera["features_rest"]
        if features_rest_flat.shape[-2] > 0:
            features_rest_flat = transform_SHs(
                shs=features_rest_flat,
                sh_to_v_transform=self.sh_to_v_transform.to(features_rest_flat.device),
                v_to_sh_transform=self.v_to_sh_transform.to(features_rest_flat.device),
                source_cameras_to_world=source_cameras_view_to_world,
            )

        return {
            "xyz": xyz_world.contiguous(),
            "xyz_camera": xyz_camera_flat.contiguous(),
            "depth_camera": gaussian_camera["depth_camera"].contiguous(),
            "offset_camera": gaussian_camera["offset_camera"].contiguous(),
            "rotation": rotation_world.contiguous(),
            "opacity": gaussian_camera["opacity"].contiguous(),
            "scaling": gaussian_camera["scaling"].contiguous(),
            "features_dc": gaussian_camera["features_dc"].contiguous(),
            "features_rest": features_rest_flat.contiguous(),
            "confidence": gaussian_camera["confidence"].contiguous(),
            "confidence_logits": gaussian_camera["confidence_logits"].contiguous(),
            "valid_mask": gaussian_camera["valid_mask"].contiguous(),
            **({"voxel_count": gaussian_camera["voxel_count"].contiguous()} if "voxel_count" in gaussian_camera else {}),
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
    if device.type != "cuda":
        raise RuntimeError(
            "gsplat rasterization requires CUDA tensors, but Gaussian tensors are on "
            f"{device}. Run with --device cuda, or skip render-dependent code paths."
        )

    world_view_transform = world_view_transform.to(device=device, dtype=pc["xyz"].dtype)
    intrinsics = intrinsics.to(device=device, dtype=pc["xyz"].dtype)
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
    scale_max = max(float(getattr(cfg.model, "scale_max", 0.3)), 1e-5)
    scales = torch.nan_to_num(scales, nan=1e-4, posinf=scale_max, neginf=1e-4).clamp(1e-5, scale_max)
    quats = F.normalize(torch.nan_to_num(quats, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1e-6)
    opacities = torch.nan_to_num(opacities, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # 2) Colors: SH (DC + rest) or override RGB
    # ------------------------------------------------------------------
    if override_color is not None:
        # Use override_color as plain RGB features (no SH)
        # Expected: override_color: (B, N, 3) or (B, N, D)
        colors = override_color.to(device=device, dtype=pc["xyz"].dtype)
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
    # 5) Convert render to channel-first. Depth-only modes return no image.
    # ------------------------------------------------------------------
    if render_mode in {"D", "ED", "d", "Ed"}:
        rendered_image = None
        rendered_depth = render_colors.permute(0, 1, 4, 2, 3).contiguous()
    elif render_colors.shape[-1] > 3:
        rendered_image = render_colors[..., :3].permute(0, 1, 4, 2, 3).contiguous()
        rendered_depth = render_colors[..., 3:4].permute(0, 1, 4, 2, 3).contiguous()
    else:
        rendered_image = render_colors.permute(0, 1, 4, 2, 3).contiguous()
        rendered_depth = None

    rendered_alpha = render_alphas.permute(0, 1, 4, 2, 3).contiguous()

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
        "depth": rendered_depth,          # (B, V, 1, H, W) for RGB+D modes
        "alpha": rendered_alpha,          # (B, V, 1, H, W)
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
    isotropic: bool = False,
    depth_parameterization: str = "absolute",
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
            isotropic=bool(isotropic),
            depth_parameterization=str(depth_parameterization),
        ),
    )
    return VAESplatterToGaussians(spl_cfg).num_splatter_channels()
