from __future__ import annotations

from typing import Any, Dict, Sequence

import os

import torch
import torch.nn as nn
from einops import rearrange


def _weight_init(m: nn.Module) -> None:
    """Initialize weights for linear and convolutional layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            m.bias.data.zero_()


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove 'module.' prefix from state dict keys if present."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        out[k[len("module."):] if k.startswith("module.") else k] = v
    return out


def _select_checkpoint_subdict(state: Dict[str, Any], preferred_keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    """Select and strip the state dict from checkpoint using preferred keys."""
    for key in preferred_keys:
        if key in state and isinstance(state[key], dict):
            return _strip_module_prefix(state[key])
    return _strip_module_prefix(state)


def _flatten_feature_output(x: torch.Tensor) -> torch.Tensor:
    """Flatten feature output to 2D if necessary."""
    return x.contiguous() if x.dim() == 2 else x.flatten(1).contiguous()


class ConvNet(nn.Module):
    """
    Official DrQ-v2 style encoder: stacked frames are channels and there is no
    internal augmentation.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        frame_stack = int(cfg["env"].get("frame_stack", 1))
        in_channels = 3 * frame_stack
        h = int(cfg["env"]["image_height"])
        w = int(cfg["env"]["image_width"])
        if h != w:
            raise ValueError(f"Official DrQ-v2 RandomShiftsAug assumes square images, got H={h}, W={w}")

        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.apply(_weight_init)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            self.repr_dim = int(self.convnet(dummy).flatten(1).shape[-1])

        self.is_trainable = True
        self.is_perturbable = True

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float() / 255.0 - 0.5
        return self.convnet(obs).flatten(1).contiguous()


class SplatterVAEInvariantEncoder(nn.Module):
    """Encoder using SplatterVAE for invariant features."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        from models.vae import SplatterVAE, CodebookConfig, default_parent_splatter_channels

        sv_cfg = dict(cfg["vision"]["splatter_vae"])
        vit_cfg = dict(cfg["vision"]["vit"])
        model_cfg = dict(sv_cfg.get("model", {}))
        cb_cfg = dict(sv_cfg["codebook"])

        self.feature_source = str(sv_cfg.get("feature_source", "codebook")).lower()
        img_h = int(cfg["vision"]["img_height"])
        img_w = int(cfg["vision"]["img_width"])

        inv_cb = CodebookConfig(**cb_cfg["invariant"])
        dep_cb = CodebookConfig(**cb_cfg["dependent"])
        max_sh_degree = int(sv_cfg.get("max_sh_degree", 1))
        splatter_channels = int(sv_cfg.get("splatter_channels", default_parent_splatter_channels(max_sh_degree=max_sh_degree)))

        self.vae = SplatterVAE(
            vit_cfg=vit_cfg,
            invariant_cb_config=inv_cb,
            dependent_cb_config=dep_cb,
            img_height=img_h,
            img_width=img_w,
            splatter_channels=splatter_channels,
            fusion_style=str(model_cfg.get("fusion_style", "cat")),
            use_dependent_vq=bool(model_cfg.get("use_dependent_vq", True)),
            is_dependent_ae=bool(model_cfg.get("is_dependent_ae", True)),
            use_invariant_vq=bool(model_cfg.get("use_invariant_vq", True)),
            is_invariant_ae=bool(model_cfg.get("is_invariant_ae", True)),
            dep_input_mask_ratio=float(model_cfg.get("dep_input_mask_ratio", 0.95)),
            dep_mask_eval=bool(model_cfg.get("dep_mask_eval", True)),
            dpt_features=int(vit_cfg.get("dpt_features", 256)),
        )

        state = torch.load(str(sv_cfg["checkpoint_path"]), map_location="cpu")
        state_dict = _select_checkpoint_subdict(state, ("vae_state_dict", "model_state_dict", "state_dict"))
        self.vae.load_state_dict(state_dict, strict=True)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.is_trainable = False
        self.is_perturbable = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 - 1.0
        h_inv_tokens, _, _ = self.vae.invariant_encoder(x)
        if self.feature_source in {"encoder", "vit", "tokens"}:
            feat = h_inv_tokens
        else:
            h_proj = self.vae.invariant_encoder_output_proj(h_inv_tokens)
            if self.feature_source == "pre_vq":
                feat = h_proj
            elif self.feature_source == "codebook":
                feat, *_ = self.vae.invariant_output_head(h_proj)
            else:
                raise ValueError(f"Unknown SplatterVAE feature_source={self.feature_source}")
        return _flatten_feature_output(feat)


class ReViWoInvariantEncoder(nn.Module):
    """Encoder using ReViWo for invariant features."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        from baselines.ReViWo.ReViWo.common.models.multiview_vae import MultiViewBetaVAE
        from models.transformer import STTransConfig
        from models.vae import CodebookConfig

        rv_cfg = dict(cfg["vision"]["reviwo"])
        img_size = int(cfg["vision"]["img_height"])
        self.model = MultiViewBetaVAE(
            view_encoder_config=STTransConfig(**dict(rv_cfg["view_encoder"])),
            latent_encoder_config=STTransConfig(**dict(rv_cfg["latent_encoder"])),
            decoder_config=STTransConfig(**dict(rv_cfg["decoder"])),
            view_cb_config=CodebookConfig(**dict(rv_cfg["view_codebook"])),
            latent_cb_config=CodebookConfig(**dict(rv_cfg["latent_codebook"])),
            img_size=img_size,
            patch_size=int(rv_cfg.get("patch_size", 16)),
            fusion_style=str(rv_cfg.get("fusion_style", "plus")),
            use_latent_vq=bool(rv_cfg.get("use_latent_vq", True)),
            is_latent_ae=bool(rv_cfg.get("is_latent_ae", False)),
            use_view_vq=bool(rv_cfg.get("use_view_vq", True)),
            is_view_ae=bool(rv_cfg.get("is_view_ae", False)),
        )
        state = torch.load(str(rv_cfg["checkpoint_path"]), map_location="cpu")
        state_dict = _select_checkpoint_subdict(state, ("model_state_dict", "state_dict"))
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.is_trainable = False
        self.is_perturbable = False
        for sub in self.model.modules():
            if hasattr(sub, "init_kmeans"):
                try:
                    sub.init_kmeans = False
                except Exception:
                    pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 - 1.0
        _, _, z_l, _, _ = self.model.encode(x)
        return z_l.flatten(1).contiguous()


class SinCroSceneEncoder(nn.Module):
    """Encoder using SinCro for scene encoding."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        from baselines.SinCro.sincro.MV_run_nerf import create_nerf

        sc_cfg = dict(cfg["vision"]["sincro"])
        self.time_interval = int(sc_cfg.get("time_interval", 3))
        self.num_ref_views = int(sc_cfg.get("num_ref_views", 2))

        class _Args:
            def __init__(self, m: Dict[str, Any]):
                self.netdepth = int(m.get("netdepth", 8))
                self.netwidth = int(m.get("netwidth", 256))
                self.netdepth_fine = int(m.get("netdepth_fine", 8))
                self.netwidth_fine = int(m.get("netwidth_fine", 256))
                self.N_rand = int(m.get("N_rand", 2048))
                self.N_samples = int(m.get("N_samples", 64))
                self.N_importance = int(m.get("N_importance", 64))
                self.multires = int(m.get("multires", 10))
                self.multires_views = int(m.get("multires_views", 4))
                self.i_embed = int(m.get("i_embed", 0))
                self.use_viewdirs = bool(m.get("use_viewdirs", True))
                self.raw_noise_std = float(m.get("raw_noise_std", 0.0))
                self.white_bkgd = bool(m.get("white_bkgd", True))
                self.perturb = float(m.get("perturb", 1.0))
                self.lindisp = bool(m.get("lindisp", False))
                self.img_size = int(m.get("img_size", 224))
                self.patch_size = int(m.get("patch_size", 16))
                self.embed_dim = int(m.get("embed_dim", 384))
                self.vit_depth = int(m.get("vit_depth", 8))
                self.vit_num_heads = int(m.get("vit_num_heads", 6))
                self.vit_mlp_dim = int(m.get("vit_mlp_dim", 1024))
                self.decoder_depth = int(m.get("decoder_depth", 2))
                self.decoder_num_heads = int(m.get("decoder_num_heads", 2))
                self.decoder_mlp_dim = int(m.get("decoder_mlp_dim", 1024))
                self.decoder_output_dim = int(m.get("decoder_output_dim", 64))
                self.vit_encoder_mlp_dim = self.vit_mlp_dim
                self.vit_decoder_mlp_dim = self.decoder_mlp_dim
                self.time_interval = int(m.get("time_interval", 3))
                self.mask_ratio = float(m.get("mask_ratio", 0.75))
                self.num_view = int(m.get("num_views", 6))
                self.batch_size = 1
                self.lrate = float(m.get("lrate", 5e-4))
                self.no_reload = True
                self.ft_path = None
                self.dataset_type = "rl_single_view_sincro"
                self.N_rgb = 0
                self.no_ndc = True
                self.render_only = False
                self.render_test = False
                self.render_factor = 1
                self.precrop_iters = 0
                self.precrop_frac = 1.0
                self.N_iters = 1
                self.i_embed_views = 0
                self.i_embed_state = -1
                self.chunk = int(m.get("chunk", 1024 * 32))
                self.netchunk = int(m.get("netchunk", 1024 * 64))
                self.lr_decay = int(m.get("lrate_decay", 250))
                self.use_mae = True
                self.gamma = 1.0
                self.log_wandb = False
                self.render_pose_path = None
                self.render_episode = None
                self.basedir = str(m.get("basedir", "./logs_sincro_rl"))
                self.expname = str(m.get("expname", "sincro_rl_encoder"))

        args = _Args(sc_cfg)
        os.makedirs(args.basedir, exist_ok=True)
        os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
        (_, _, _, _, _, latent_embed) = create_nerf(args, args.basedir, args.expname)
        self.encoder = latent_embed
        state = torch.load(str(sc_cfg["checkpoint_path"]), map_location="cpu")
        state_dict = _select_checkpoint_subdict(state, ("latent_embed_state_dict", "model_state_dict", "state_dict"))
        self.encoder.load_state_dict(state_dict, strict=True)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.is_trainable = False
        self.is_perturbable = False

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim != 5:
            raise ValueError(f"Expected x_seq with shape (B,T,3,H,W), got {tuple(x_seq.shape)}")
        B, T, C, H, W = x_seq.shape
        if C != 3:
            raise ValueError(f"Expected RGB input with C=3, got C={C}")
        if T != self.time_interval:
            raise ValueError(f"SinCro expects T == time_interval == {self.time_interval}, got T={T}.")

        primary_images = x_seq.permute(0, 1, 3, 4, 2).contiguous()
        try:
            primary_images = primary_images.to(next(self.encoder.parameters()).device, non_blocking=True)
        except StopIteration:
            pass

        latent, mask, ids_restore = self.encoder.SinCro_image_encoder(primary_images, mask_ratio=0.0, T=T, is_ref=False)
        ref_for_encoder = primary_images.repeat(self.num_ref_views, 1, 1, 1, 1)
        ref_latent, _, _ = self.encoder.SinCro_image_encoder(ref_for_encoder, mask_ratio=0.0, T=T, is_ref=True)
        ref_latent = rearrange(ref_latent[:, 1:, :], 'b (t hw) d -> b t hw d', t=T)[:, -1]
        ref_latent = rearrange(ref_latent, '(v b) hw d -> b (v hw) d', v=self.num_ref_views, b=B)
        latent, _, _ = self.encoder.SinCro_state_encoder(latent, ref_latent, mask, ids_restore)
        return latent.view(B, T, -1)[:, -1].contiguous()


def build_vision_encoder(cfg: Dict[str, Any]) -> nn.Module:
    """Build vision encoder based on configuration."""
    enc_type = str(cfg["vision"].get("encoder_type", cfg["vision"].get("name", "convnet"))).lower()
    if enc_type == "convnet":
        return ConvNet(cfg)
    if enc_type == "splatter_vae":
        return SplatterVAEInvariantEncoder(cfg)
    if enc_type == "reviwo":
        return ReViWoInvariantEncoder(cfg)
    if enc_type == "sincro":
        return SinCroSceneEncoder(cfg)
    raise ValueError(f"Unknown vision.encoder_type={enc_type}")
