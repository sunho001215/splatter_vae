from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import resnet18

from agents.drqv2.drqv2_metaworld import VisionEncoderAdapter, weight_init


def _as_pair(value: Sequence[int] | int | None) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError(f"Expected pair, got {value}")
    return int(value[0]), int(value[1])


class SinusoidalPosEmb(nn.Module):
    """Standard diffusion timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class Conv1dBlock(nn.Module):
    """Conv1d + GroupNorm + Mish block used throughout the DP U-Net."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__()
        padding = kernel_size // 2
        groups = min(int(n_groups), int(out_channels))
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    Residual temporal block with FiLM conditioning.

    The conditioning vector is global for the action sequence. It includes the
    diffusion timestep embedding plus projected visual/proprioceptive features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.cond_predict_scale = bool(cond_predict_scale)
        self.out_channels = int(out_channels)
        cond_channels = out_channels * 2 if self.cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.cond_predict_scale:
            scale, bias = embed.chunk(2, dim=1)
            out = out * (1.0 + scale) + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """
    UNet action denoiser from Diffusion Policy's CNN/UNet variant.

    Input and output shape is (B, horizon, action_dim). Conditioning is provided
    globally through FiLM at every residual block, not through a transformer.
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: Sequence[int] = (512, 1024, 2048),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ) -> None:
        super().__init__()
        all_dims = [int(input_dim)] + [int(v) for v in down_dims]
        dsed = int(diffusion_step_embed_dim)
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + int(global_cond_dim)

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale
                ),
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale
                ),
            ]
        )

        self.down_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= len(in_out) - 1
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale
                        ),
                        ConditionalResidualBlock1D(
                            dim_in, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        start_dim = all_dims[1]
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        self.apply(weight_init)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        x = rearrange(sample, "b h d -> b d h")
        if not torch.is_tensor(timestep):
            timestep = torch.full((x.shape[0],), int(timestep), device=x.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep[None].expand(x.shape[0]).to(x.device)
        else:
            timestep = timestep.to(device=x.device, dtype=torch.long)

        t_feature = self.diffusion_step_encoder(timestep)
        cond = torch.cat([t_feature, global_cond], dim=-1)

        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, cond)
            x = resnet2(x, cond)
            skips.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, cond)

        for resnet, resnet2, upsample in self.up_modules:
            skip = skips.pop()
            # Odd horizons can differ by one after stride-2 operations; crop to
            # the shared temporal length so the skip path stays well-defined.
            if x.shape[-1] != skip.shape[-1]:
                target = min(x.shape[-1], skip.shape[-1])
                x = x[..., :target]
                skip = skip[..., :target]
            x = torch.cat([x, skip], dim=1)
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)

        x = self.final_conv(x)
        if x.shape[-1] != sample.shape[1]:
            x = F.interpolate(x, size=sample.shape[1], mode="linear", align_corners=False)
        return rearrange(x, "b d h -> b h d")


def _betas_for_alpha_bar(num_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    """Cosine schedule used by squaredcos_cap_v2 in common DP configs."""

    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DDPMScheduler:
    """Small self-contained DDPM scheduler for action trajectories."""

    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        variance_type: str = "fixed_small",
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.prediction_type = str(prediction_type)
        self.clip_sample = bool(clip_sample)
        self.variance_type = str(variance_type)
        if beta_schedule == "linear":
            betas = torch.linspace(float(beta_start), float(beta_end), self.num_train_timesteps)
        elif beta_schedule == "squaredcos_cap_v2":
            betas = _betas_for_alpha_bar(self.num_train_timesteps)
        else:
            raise ValueError(f"Unsupported beta_schedule={beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0, dtype=torch.float32)
        self.timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1, dtype=torch.long)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DDPMScheduler":
        return cls(**dict(cfg))

    def _to(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        return tensor.to(device=device)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alphas_cumprod = self._to(self.alphas_cumprod, original_samples.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1)
        sqrt_one_minus = (1.0 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1)
        return sqrt_alpha_prod * original_samples + sqrt_one_minus * noise

    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> None:
        steps = np.linspace(0, self.num_train_timesteps - 1, int(num_inference_steps), dtype=np.int64)
        self.timesteps = torch.from_numpy(steps[::-1].copy()).to(device=device, dtype=torch.long)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if torch.is_tensor(timestep):
            t = int(timestep.flatten()[0].item())
        else:
            t = int(timestep)
        prev_t = max(t - self.num_train_timesteps // max(len(self.timesteps), 1), -1)

        alphas_cumprod = self._to(self.alphas_cumprod, sample.device)
        betas = self._to(self.betas, sample.device)
        alpha_prod_t = alphas_cumprod[t]
        alpha_prod_t_prev = alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod.to(sample.device)
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev

        if self.prediction_type == "epsilon":
            pred_original = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.prediction_type == "sample":
            pred_original = model_output
        else:
            raise ValueError(f"Unsupported prediction_type={self.prediction_type}")

        if self.clip_sample:
            pred_original = pred_original.clamp(-1.0, 1.0)

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t
        pred_original_coeff = alpha_prod_t_prev.sqrt() * current_beta_t / beta_prod_t
        current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t
        prev_sample = pred_original_coeff * pred_original + current_sample_coeff * sample

        if t > 0:
            variance = (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * betas[t]
            noise = torch.randn(sample.shape, device=sample.device, dtype=sample.dtype, generator=generator)
            prev_sample = prev_sample + variance.clamp_min(1e-20).sqrt() * noise
        return prev_sample


class RandomCrop(nn.Module):
    """Random crop during training, center crop during eval."""

    def __init__(self, crop_shape: Optional[Tuple[int, int]]) -> None:
        super().__init__()
        self.crop_shape = crop_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.crop_shape is None:
            return x
        crop_h, crop_w = self.crop_shape
        _, _, h, w = x.shape
        if crop_h > h or crop_w > w:
            raise ValueError(f"crop_shape={self.crop_shape} exceeds image shape {(h, w)}.")
        if crop_h == h and crop_w == w:
            return x
        if self.training:
            max_y = h - crop_h
            max_x = w - crop_w
            ys = torch.randint(0, max_y + 1, (x.shape[0],), device=x.device)
            xs = torch.randint(0, max_x + 1, (x.shape[0],), device=x.device)
        else:
            ys = torch.full((x.shape[0],), (h - crop_h) // 2, device=x.device, dtype=torch.long)
            xs = torch.full((x.shape[0],), (w - crop_w) // 2, device=x.device, dtype=torch.long)
        crops = []
        for i in range(x.shape[0]):
            y = int(ys[i].item())
            x0 = int(xs[i].item())
            crops.append(x[i : i + 1, :, y : y + crop_h, x0 : x0 + crop_w])
        return torch.cat(crops, dim=0)


class SpatialSoftmax(nn.Module):
    """Robomimic-style spatial softmax keypoint pooling."""

    def __init__(self, in_channels: int, num_keypoints: int) -> None:
        super().__init__()
        self.num_keypoints = int(num_keypoints)
        self.keypoint_conv = nn.Conv2d(in_channels, self.num_keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        heatmap = self.keypoint_conv(x).reshape(b, self.num_keypoints, h * w)
        heatmap = F.softmax(heatmap, dim=-1)
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
        keypoints = heatmap @ coords
        return keypoints.reshape(b, self.num_keypoints * 2)


def replace_batchnorm_with_groupnorm(module: nn.Module, channels_per_group: int = 16) -> nn.Module:
    """Replace ResNet BatchNorm with GroupNorm, matching common DP image configs."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            channels = int(child.num_features)
            groups = max(1, min(32, channels // max(1, int(channels_per_group))))
            setattr(module, name, nn.GroupNorm(groups, channels))
        else:
            replace_batchnorm_with_groupnorm(child, channels_per_group)
    return module


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        layers = []
        prev = int(in_dim)
        for _ in range(max(0, int(num_layers) - 1)):
            layers.append(nn.Linear(prev, int(hidden_dim)))
            if use_layer_norm:
                layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.Mish())
            prev = int(hidden_dim)
        layers.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*layers)
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet18ImageConditioner(nn.Module):
    """
    End-to-end CNN baseline: random crop + ResNet18 + optional SpatialSoftmax.

    Multi-view handling happens in the dataloader: each optimizer step receives
    one randomly selected camera view, so this encoder remains single-view like
    the original DP image baseline.
    """

    def __init__(self, cfg: Dict[str, Any], proprio_dim: int, output_dim: int) -> None:
        super().__init__()
        policy_cfg = dict(cfg["policy"])
        cnn_cfg = dict(cfg.get("cnn_encoder", {}))
        self.n_obs_steps = int(policy_cfg["n_obs_steps"])
        self.proprio_dim = int(proprio_dim)
        self.crop = RandomCrop(_as_pair(cnn_cfg.get("crop_shape", None)))

        resnet = resnet18(weights=None)
        if bool(cnn_cfg.get("use_group_norm", True)):
            resnet = replace_batchnorm_with_groupnorm(resnet, int(cnn_cfg.get("channels_per_group", 16)))
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.use_spatial_softmax = bool(cnn_cfg.get("use_spatial_softmax", True))
        if self.use_spatial_softmax:
            num_keypoints = int(cnn_cfg.get("num_keypoints", 32))
            self.pool = SpatialSoftmax(512, num_keypoints)
            image_feat_dim = num_keypoints * 2
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            image_feat_dim = 512

        raw_dim = self.n_obs_steps * (image_feat_dim + self.proprio_dim)
        self.proj = MLP(
            raw_dim,
            int(output_dim),
            int(cnn_cfg.get("projection_hidden_dim", output_dim)),
            int(cnn_cfg.get("projection_layers", 2)),
        )

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        return image

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        b, to, c, h, w = image.shape
        if to != self.n_obs_steps:
            raise ValueError(f"Expected n_obs_steps={self.n_obs_steps}, got {to}")
        x = self._normalize_image(image.reshape(b * to, c, h, w))
        x = self.crop(x)
        x = self.backbone(x)
        x = self.pool(x)
        if not self.use_spatial_softmax:
            x = x.flatten(1)
        x = x.reshape(b, to, -1)
        cond_parts = [x.flatten(1)]
        if proprio.shape[-1] > 0:
            cond_parts.append(proprio.reshape(b, -1))
        return self.proj(torch.cat(cond_parts, dim=-1))


class PretrainedEncoderConditioner(nn.Module):
    """Conditioner for SinCro, ReViWo, and SplatterVAE frozen visual encoders."""

    def __init__(self, cfg: Dict[str, Any], proprio_dim: int, output_dim: int) -> None:
        super().__init__()
        self.n_obs_steps = int(cfg["policy"]["n_obs_steps"])
        # VisionEncoderAdapter expects the DrQ-v2 naming for stacked frames.
        adapter_cfg = copy.deepcopy(cfg)
        adapter_cfg.setdefault("env", {})
        adapter_cfg["env"]["frame_stack"] = self.n_obs_steps
        if "vit" not in adapter_cfg.get("vision", {}) and "vit" in adapter_cfg:
            adapter_cfg["vision"]["vit"] = adapter_cfg["vit"]
        adapter_cfg.setdefault("agent", {})
        adapter_cfg["agent"]["feature_dim"] = int(cfg.get("pretrained_encoder", {}).get("feature_dim", output_dim))
        self.adapter = VisionEncoderAdapter(adapter_cfg)
        raw_dim = int(self.adapter.repr_dim) + self.n_obs_steps * int(proprio_dim)
        head_cfg = dict(cfg.get("pretrained_encoder", {}))
        self.proj = MLP(
            raw_dim,
            int(output_dim),
            int(head_cfg.get("conditioning_hidden_dim", output_dim)),
            int(head_cfg.get("conditioning_layers", 2)),
        )

    def train(self, mode: bool = True) -> "PretrainedEncoderConditioner":
        super().train(mode)
        self.adapter.set_update_mode() if mode else self.adapter.set_act_mode()
        return self

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        b, to, c, h, w = image.shape
        if to != self.n_obs_steps:
            raise ValueError(f"Expected n_obs_steps={self.n_obs_steps}, got {to}")
        stacked = image.reshape(b, to * c, h, w).contiguous()
        visual = self.adapter(stacked, is_feature=False)
        cond_parts = [visual]
        if proprio.shape[-1] > 0:
            cond_parts.append(proprio.reshape(b, -1))
        return self.proj(torch.cat(cond_parts, dim=-1))


def build_conditioner(cfg: Dict[str, Any], proprio_dim: int, output_dim: int) -> nn.Module:
    enc_type = str(cfg["vision"].get("encoder_type", cfg["vision"].get("name", "cnn"))).lower()
    if enc_type in {"cnn", "convnet", "resnet18", "end_to_end_cnn"}:
        return ResNet18ImageConditioner(cfg, proprio_dim=proprio_dim, output_dim=output_dim)
    if enc_type in {"sincro", "reviwo", "splatter_vae", "splattervae"}:
        return PretrainedEncoderConditioner(cfg, proprio_dim=proprio_dim, output_dim=output_dim)
    raise ValueError(f"Unsupported vision.encoder_type={enc_type}")


class Normalizer(nn.Module):
    """Mean/std normalizer for action trajectories and low-dimensional observations."""

    def __init__(self, action_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        self.register_buffer("proprio_mean", torch.zeros(proprio_dim))
        self.register_buffer("proprio_std", torch.ones(proprio_dim))

    def load_stats(self, stats: Dict[str, torch.Tensor]) -> None:
        self.action_mean.copy_(stats["action_mean"].to(self.action_mean))
        self.action_std.copy_(stats["action_std"].to(self.action_std).clamp_min(1e-6))
        if self.proprio_mean.numel() > 0:
            self.proprio_mean.copy_(stats["proprio_mean"].to(self.proprio_mean))
            self.proprio_std.copy_(stats["proprio_std"].to(self.proprio_std).clamp_min(1e-6))

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std + self.action_mean

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        if proprio.shape[-1] == 0:
            return proprio
        return (proprio - self.proprio_mean) / self.proprio_std


@dataclass
class LossOutput:
    loss: torch.Tensor
    pred_mse: torch.Tensor


class DiffusionPolicy(nn.Module):
    """UNet-based action diffusion policy with FiLM visual conditioning."""

    def __init__(self, cfg: Dict[str, Any], action_dim: int, proprio_dim: int) -> None:
        super().__init__()
        policy_cfg = dict(cfg["policy"])
        self.horizon = int(policy_cfg["horizon"])
        self.n_obs_steps = int(policy_cfg["n_obs_steps"])
        self.n_action_steps = int(policy_cfg["n_action_steps"])
        self.action_dim = int(action_dim)
        cond_dim = int(policy_cfg.get("global_cond_dim", 256))

        self.normalizer = Normalizer(action_dim=action_dim, proprio_dim=proprio_dim)
        self.conditioner = build_conditioner(cfg, proprio_dim=proprio_dim, output_dim=cond_dim)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=cond_dim,
            diffusion_step_embed_dim=int(policy_cfg.get("diffusion_step_embed_dim", 128)),
            down_dims=tuple(int(v) for v in policy_cfg.get("down_dims", [512, 1024, 2048])),
            kernel_size=int(policy_cfg.get("kernel_size", 5)),
            n_groups=int(policy_cfg.get("n_groups", 8)),
            cond_predict_scale=bool(policy_cfg.get("cond_predict_scale", True)),
        )

    def compute_loss(self, batch: Dict[str, torch.Tensor], noise_scheduler: DDPMScheduler) -> LossOutput:
        image = batch["image"]
        action = self.normalizer.normalize_action(batch["action"])
        proprio = self.normalizer.normalize_proprio(batch["proprio"])
        global_cond = self.conditioner(image, proprio)

        noise = torch.randn_like(action)
        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (action.shape[0],),
            device=action.device,
            dtype=torch.long,
        )
        noisy_action = noise_scheduler.add_noise(action, noise, timesteps)
        pred = self.noise_pred_net(noisy_action, timesteps, global_cond=global_cond)

        if noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.prediction_type == "sample":
            target = action
        else:
            raise ValueError(f"Unsupported prediction_type={noise_scheduler.prediction_type}")
        loss = F.mse_loss(pred, target)
        return LossOutput(loss=loss, pred_mse=loss.detach())

    @torch.no_grad()
    def predict_action(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        noise_scheduler: DDPMScheduler,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """Sample an action chunk for deployment/evaluation."""
        self.eval()
        proprio = self.normalizer.normalize_proprio(proprio)
        global_cond = self.conditioner(image, proprio)
        sample = torch.randn(
            image.shape[0],
            self.horizon,
            self.action_dim,
            device=image.device,
            dtype=torch.float32,
        )
        noise_scheduler.set_timesteps(num_inference_steps, device=image.device)
        for t in noise_scheduler.timesteps:
            model_output = self.noise_pred_net(sample, t, global_cond=global_cond)
            sample = noise_scheduler.step(model_output, t, sample)
        action_pred = self.normalizer.unnormalize_action(sample)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return action_pred[:, start:end]


class EMAModel:
    """Exponential moving average wrapper for stable validation/checkpointing."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.averaged_model = copy.deepcopy(model).eval()
        for p in self.averaged_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_param, param in zip(self.averaged_model.parameters(), model.parameters()):
            ema_param.mul_(self.decay).add_(param, alpha=1.0 - self.decay)
        for ema_buffer, buffer in zip(self.averaged_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "averaged_model": self.averaged_model.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.averaged_model.load_state_dict(state["averaged_model"])

