from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

from pathlib import Path
import os

from einops import rearrange

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _weight_init(m: nn.Module) -> None:
    """Orthogonal init like your current DrM / DrQ-style code."""
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
    """Remove a leading 'module.' prefix if the checkpoint came from DDP."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def _select_checkpoint_subdict(state: Dict[str, Any], preferred_keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    """Pick the most likely state_dict field from a loaded checkpoint dict."""
    for key in preferred_keys:
        if key in state and isinstance(state[key], dict):
            return _strip_module_prefix(state[key])
    return _strip_module_prefix(state)  # fallback: assume raw state_dict


def _get_attr_any(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    """Try multiple attribute names, since local forks often rename modules."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _flatten_feature_output(x: torch.Tensor) -> torch.Tensor:
    """Convert tokens / maps / vectors into one flat feature vector per sample."""
    if x.dim() == 2:
        return x.contiguous()
    return x.flatten(1).contiguous()


# -----------------------------------------------------------------------------
# ConvNet encoder (same as before)
# -----------------------------------------------------------------------------

class ConvNet(nn.Module):
    """
    Same ConvNet visual encoder as before.

    - Input expected in [0, 1]
    - train(): random crop
    - eval(): center crop
    - normalize by subtracting 0.5
    - output: flat vector (B, D)
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.crop_height = int(cfg.get("crop_height", 100))
        self.crop_width = int(cfg.get("crop_width", 100))

        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.apply(_weight_init)

        self.is_trainable = True
        self.is_perturbable = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            i, j, h, w = torchvision.transforms.RandomCrop.get_params(
                x, output_size=(self.crop_height, self.crop_width)
            )
            x = TF.crop(x, i, j, h, w)
        else:
            x = TF.center_crop(x, (self.crop_height, self.crop_width))

        x = x - 0.5
        h = self.convnet(x)
        return h.view(h.shape[0], -1).contiguous()


# -----------------------------------------------------------------------------
# Frozen SplatterVAE invariant encoder
# -----------------------------------------------------------------------------

class SplatterVAEInvariantEncoder(nn.Module):
    """
    Frozen invariant encoder from the NEW hierarchical SplatterVAE.

    feature_source:
      - "encoder" / "vit" / "tokens": final invariant ViT tokens
      - "pre_vq": invariant tokens after projection, before VQ
      - "codebook": quantized invariant tokens
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        from models.vae import SplatterVAE, CodebookConfig, default_parent_splatter_channels

        sv_cfg = dict(cfg["splatter_vae"])
        vit_cfg = dict(cfg["vit"])
        model_cfg = dict(sv_cfg.get("model", {}))
        cb_cfg = dict(sv_cfg["codebook"])

        self.feature_source = str(sv_cfg.get("feature_source", "codebook")).lower()

        img_h = int(cfg["img_height"])
        img_w = int(cfg["img_width"])

        inv_cb = CodebookConfig(**cb_cfg["invariant"])
        dep_cb = CodebookConfig(**cb_cfg["dependent"])

        max_sh_degree = int(sv_cfg.get("max_sh_degree", 1))
        splatter_channels = int(
            sv_cfg.get(
                "splatter_channels",
                default_parent_splatter_channels(max_sh_degree=max_sh_degree),
            )
        )

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

        ckpt_path = str(sv_cfg["checkpoint_path"])
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = _select_checkpoint_subdict(
            state,
            preferred_keys=("vae_state_dict", "model_state_dict", "state_dict"),
        )
        self.vae.load_state_dict(state_dict, strict=True)

        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        self.is_trainable = False
        self.is_perturbable = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SplatterVAE was trained on [-1, 1]
        x = x * 2.0 - 1.0

        # New hierarchical SplatterVAE invariant encoder returns tokens.
        h_inv_tokens, _, _ = self.vae.invariant_encoder(x)

        if self.feature_source in {"encoder", "vit", "tokens"}:
            feat = h_inv_tokens
        else:
            h_proj = self.vae.invariant_encoder_output_proj(h_inv_tokens)

            if self.feature_source == "pre_vq":
                feat = h_proj
            elif self.feature_source == "codebook":
                z_inv, *_ = self.vae.invariant_output_head(h_proj)
                feat = z_inv
            else:
                raise ValueError(
                    f"Unknown SplatterVAE feature_source='{self.feature_source}'. "
                    f"Use one of: encoder, vit, tokens, pre_vq, codebook"
                )

        return _flatten_feature_output(feat)


# -----------------------------------------------------------------------------
# Frozen ReViWo invariant / latent encoder
# -----------------------------------------------------------------------------

class ReViWoInvariantEncoder(nn.Module):
    """
    Frozen ReViWo encoder for downstream RL.

    IMPORTANT:
    - Your local ReViWo implementation expects raw images to go through the full
      MultiViewBetaVAE.encode(x) path.
    - The latent / view-invariant representation for downstream control is z_l.

    Supported feature_source:
      - "codebook": use z_l returned by model.encode(x)  [recommended]
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        from baselines.ReViWo.ReViWo.common.models.multiview_vae import MultiViewBetaVAE
        from models.transformer import STTransConfig
        from models.vae import CodebookConfig

        rv_cfg = dict(cfg["reviwo"])
        img_size = int(cfg["img_height"])

        view_enc_cfg = STTransConfig(**dict(rv_cfg["view_encoder"]))
        latent_enc_cfg = STTransConfig(**dict(rv_cfg["latent_encoder"]))
        decoder_cfg = STTransConfig(**dict(rv_cfg["decoder"]))
        view_cb_cfg = CodebookConfig(**dict(rv_cfg["view_codebook"]))
        latent_cb_cfg = CodebookConfig(**dict(rv_cfg["latent_codebook"]))

        self.model = MultiViewBetaVAE(
            view_encoder_config=view_enc_cfg,
            latent_encoder_config=latent_enc_cfg,
            decoder_config=decoder_cfg,
            view_cb_config=view_cb_cfg,
            latent_cb_config=latent_cb_cfg,
            img_size=img_size,
            patch_size=int(rv_cfg.get("patch_size", 16)),
            fusion_style=str(rv_cfg.get("fusion_style", "plus")),
            use_latent_vq=bool(rv_cfg.get("use_latent_vq", True)),
            is_latent_ae=bool(rv_cfg.get("is_latent_ae", False)),
            use_view_vq=bool(rv_cfg.get("use_view_vq", True)),
            is_view_ae=bool(rv_cfg.get("is_view_ae", False)),
        )

        ckpt_path = str(rv_cfg["checkpoint_path"])
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = _select_checkpoint_subdict(
            state,
            preferred_keys=("model_state_dict", "state_dict"),
        )
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
        # ReViWo training uses images in [-1, 1].
        x = x * 2.0 - 1.0

        # ReViWo forward pass
        z_v, view_embed_loss, z_l, latent_embed_loss, encoding_indices = self.model.encode(x)
        return z_l.flatten(1).contiguous()

# -----------------------------------------------------------------------------
# Frozen SinCro encoder
# -----------------------------------------------------------------------------

class SinCroSceneEncoder(nn.Module):
    """
    Paper-faithful SinCro downstream encoder for RL.

    Unlike the earlier simplified wrapper, this uses the FULL frozen SinCro
    scene encoder Ωθ:
      1) primary history -> SinCro_image_encoder(..., is_ref=False)
      2) replicated reference histories -> SinCro_image_encoder(..., is_ref=True)
      3) fuse them with SinCro_state_encoder(...)
      4) take the final timestep scene latent

    This mirrors both:
      - the SinCro paper's downstream RL description
      - your local SinCro training code path

    Expected input:
        x_seq: (B, T, 3, H, W), float in [0, 1]

    Returns:
        z_t: (B, D), flat vector
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        from baselines.SinCro.sincro.MV_run_nerf import create_nerf

        sc_cfg = dict(cfg["sincro"])

        self.time_interval = int(sc_cfg.get("time_interval", 3))
        self.num_ref_views = int(sc_cfg.get("num_ref_views", 2))

        # ------------------------------------------------------------------
        # Build the original SinCro latent encoder through create_nerf(...)
        # so the architecture matches your local SinCro implementation.
        # ------------------------------------------------------------------
        class _SimpleSinCroArgs:
            def __init__(self, model_cfg: Dict[str, Any]):
                # NeRF / rendering args expected by create_nerf
                self.netdepth = int(model_cfg.get("netdepth", 8))
                self.netwidth = int(model_cfg.get("netwidth", 256))
                self.netdepth_fine = int(model_cfg.get("netdepth_fine", 8))
                self.netwidth_fine = int(model_cfg.get("netwidth_fine", 256))
                self.N_rand = int(model_cfg.get("N_rand", 2048))
                self.N_samples = int(model_cfg.get("N_samples", 64))
                self.N_importance = int(model_cfg.get("N_importance", 64))
                self.multires = int(model_cfg.get("multires", 10))
                self.multires_views = int(model_cfg.get("multires_views", 4))
                self.i_embed = int(model_cfg.get("i_embed", 0))
                self.use_viewdirs = bool(model_cfg.get("use_viewdirs", True))
                self.raw_noise_std = float(model_cfg.get("raw_noise_std", 0.0))
                self.white_bkgd = bool(model_cfg.get("white_bkgd", True))
                self.perturb = float(model_cfg.get("perturb", 1.0))
                self.lindisp = bool(model_cfg.get("lindisp", False))

                # SinCro MAE / ViT args
                self.img_size = int(model_cfg.get("img_size", 224))
                self.patch_size = int(model_cfg.get("patch_size", 16))
                self.embed_dim = int(model_cfg.get("embed_dim", 384))
                self.vit_depth = int(model_cfg.get("vit_depth", 8))
                self.vit_num_heads = int(model_cfg.get("vit_num_heads", 6))
                self.vit_mlp_dim = int(model_cfg.get("vit_mlp_dim", 1024))

                self.decoder_depth = int(model_cfg.get("decoder_depth", 2))
                self.decoder_num_heads = int(model_cfg.get("decoder_num_heads", 2))
                self.decoder_mlp_dim = int(model_cfg.get("decoder_mlp_dim", 1024))
                self.decoder_output_dim = int(model_cfg.get("decoder_output_dim", 64))

                # IMPORTANT:
                # Your local create_nerf(...) expects these exact alias names.
                # They are present in your own SinCro SimpleArgs shim.
                self.vit_encoder_mlp_dim = self.vit_mlp_dim
                self.vit_decoder_mlp_dim = self.decoder_mlp_dim

                # SinCro-specific
                self.time_interval = int(model_cfg.get("time_interval", 3))
                self.mask_ratio = float(model_cfg.get("mask_ratio", 0.75))
                self.num_view = int(model_cfg.get("num_views", 6))
                self.batch_size = 1

                # Misc placeholders expected by create_nerf
                self.lrate = float(model_cfg.get("lrate", 5e-4))
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
                self.chunk = int(model_cfg.get("chunk", 1024 * 32))
                self.netchunk = int(model_cfg.get("netchunk", 1024 * 64))
                self.lr_decay = int(model_cfg.get("lrate_decay", 250))
                self.use_mae = True
                self.gamma = 1.0
                self.log_wandb = False

                self.render_pose_path = None
                self.render_episode = None

                # create_nerf scans basedir/expname; make them safe
                self.basedir = str(model_cfg.get("basedir", "./logs_sincro_rl"))
                self.expname = str(model_cfg.get("expname", "sincro_rl_encoder"))

        build_args = _SimpleSinCroArgs(sc_cfg)
        os.makedirs(build_args.basedir, exist_ok=True)
        os.makedirs(os.path.join(build_args.basedir, build_args.expname), exist_ok=True)

        (
            _render_kwargs_train,
            _render_kwargs_test,
            _start_step,
            _grad_vars,
            _optimizer,
            latent_embed,
        ) = create_nerf(build_args, build_args.basedir, build_args.expname)

        self.encoder = latent_embed

        ckpt_path = str(sc_cfg["checkpoint_path"])
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = _select_checkpoint_subdict(
            state,
            preferred_keys=("latent_embed_state_dict", "model_state_dict", "state_dict"),
        )
        self.encoder.load_state_dict(state_dict, strict=True)

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.is_trainable = False
        self.is_perturbable = False

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (B, T, 3, H, W), float in [0, 1]

        Returns:
            scene_latent: (B, D)
        """
        if x_seq.ndim != 5:
            raise ValueError(f"Expected x_seq with shape (B,T,3,H,W), got {tuple(x_seq.shape)}")

        B, T, C, H, W = x_seq.shape
        if C != 3:
            raise ValueError(f"Expected RGB input with C=3, got C={C}")
        if T != self.time_interval:
            raise ValueError(
                f"SinCro expects T == time_interval == {self.time_interval}, but got T={T}. "
                f"Set env.frame_stack to {self.time_interval} for paper-faithful SinCro RL."
            )

        # SinCro expects channel-last temporal image blocks: [B, T, H, W, 3]
        primary_images = x_seq.permute(0, 1, 3, 4, 2).contiguous()

        # Ensure input is on the same device as the frozen encoder.
        try:
            encoder_device = next(self.encoder.parameters()).device
            primary_images = primary_images.to(encoder_device, non_blocking=True)
        except StopIteration:
            encoder_device = primary_images.device

        # --------------------------------------------------------------
        # 1) Primary branch: NO masking during downstream RL
        # --------------------------------------------------------------
        latent, mask, ids_restore = self.encoder.SinCro_image_encoder(
            primary_images,
            mask_ratio=0.0,
            T=T,
            is_ref=False,
        )

        # --------------------------------------------------------------
        # 2) Reference branch: replicate the SAME single-view history K times
        #    This matches the SinCro paper's downstream RL inference rule.
        # --------------------------------------------------------------
        ref_for_encoder = primary_images.repeat(self.num_ref_views, 1, 1, 1, 1)

        ref_latent, _, _ = self.encoder.SinCro_image_encoder(
            ref_for_encoder,
            mask_ratio=0.0,
            T=T,
            is_ref=True,
        )

        # Match the original SinCro training path:
        #   - drop CLS
        #   - reshape into [num_ref_view * B, T, HW, D]
        #   - keep only the latest timestep for reference views
        #   - merge reference views into one token bank
        ref_latent = rearrange(
            ref_latent[:, 1:, :],
            "b (t hw) d -> b t hw d",
            t=T,
        )[:, -1]

        ref_latent = rearrange(
            ref_latent,
            "(v b) hw d -> b (v hw) d",
            v=self.num_ref_views,
            b=B,
        )

        # --------------------------------------------------------------
        # 3) State encoder fusion
        # --------------------------------------------------------------
        latent, mask, ids_restore = self.encoder.SinCro_state_encoder(
            latent,
            ref_latent,
            mask,
            ids_restore,
        )

        # Same extraction pattern as your SinCro training code:
        # reshape to [B, T, D_eff] and take the final timestep scene latent.
        latent_seq = latent.view(B, T, -1)
        scene_latent = latent_seq[:, -1]

        return scene_latent.contiguous()

# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def build_vision_encoder(cfg: Dict[str, Any]) -> nn.Module:
    enc_type = str(
        cfg["vision"].get("encoder_type", cfg["vision"].get("name", "convnet"))
    ).lower()

    if enc_type == "convnet":
        return ConvNet(cfg["vision"]["convnet"])
    if enc_type == "splatter_vae":
        return SplatterVAEInvariantEncoder(cfg["vision"])
    if enc_type == "reviwo":
        return ReViWoInvariantEncoder(cfg["vision"])
    if enc_type == "sincro":
        return SinCroSceneEncoder(cfg["vision"])

    raise ValueError(f"Unknown vision.encoder_type={enc_type}")