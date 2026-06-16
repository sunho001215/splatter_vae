from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from models.splatter import render_predicted
from utils.general_utils import image_to_tensor as repo_image_to_tensor
from visualize.metaworld_camera_utils import image_to_tensor, load_yaml, tensor_to_uint8_image
from visualize.splattervae_common import build_visualization_models


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def select_state_dict(state: Any, preferred: Sequence[str]) -> Dict[str, torch.Tensor]:
    if isinstance(state, dict):
        for key in preferred:
            if key in state and isinstance(state[key], dict):
                return strip_module_prefix(state[key])
        return strip_module_prefix(state)
    raise TypeError(f"Unsupported checkpoint payload type: {type(state)!r}")


def latest_checkpoint(path_or_dir: str | Path | None, suffixes: Sequence[str]) -> str | None:
    if path_or_dir is None:
        return None
    path = Path(path_or_dir)
    if path.is_file():
        return str(path)
    if not path.is_dir():
        return None
    candidates = []
    for suffix in suffixes:
        candidates.extend(path.glob(f"*{suffix}"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    return str(sorted(candidates)[-1])


def infer_splatter_ckpt(cfg: Mapping[str, Any], explicit: str | None = None) -> str:
    if explicit:
        return explicit
    direct = cfg.get("checkpoint_path")
    if direct:
        return str(direct)
    train_dir = cfg.get("train", {}).get("ckpt_dir")
    ckpt = latest_checkpoint(train_dir, (".pth", ".pt"))
    if ckpt is None:
        raise ValueError("No SplatterVAE checkpoint was provided and no checkpoint was found in train.ckpt_dir.")
    return ckpt


class SplatterVAERenderer:
    def __init__(self, config_path: str | Path, dataset_path: str | Path, demo_key: str, ckpt_path: str | Path | None, device: torch.device):
        self.cfg = load_yaml(config_path)
        self.ckpt_path = infer_splatter_ckpt(self.cfg, None if ckpt_path is None else str(ckpt_path))
        self.device = device
        self.vae, self.converter, self.spl_cfg = build_visualization_models(
            self.cfg,
            str(dataset_path),
            demo_key,
            self.ckpt_path,
            device,
        )
        self.bg = torch.ones(3, device=device) if self.spl_cfg.data.white_background else torch.zeros(3, device=device)

    @torch.no_grad()
    def encode_source(self, image_u8: np.ndarray, source_k: np.ndarray, source_c2w: np.ndarray) -> Dict[str, torch.Tensor]:
        x = image_to_tensor(image_u8).unsqueeze(0).to(self.device)
        z_inv, _, z_dep, _, _ = self.vae.encode(
            x,
            deterministic_invariant=True,
            deterministic_dependent=True,
        )
        splatter = self.vae.decode(z_inv.contiguous(), z_dep.contiguous())
        k = torch.from_numpy(np.asarray(source_k, dtype=np.float32)).unsqueeze(0).to(self.device)
        c2w = torch.from_numpy(np.asarray(source_c2w, dtype=np.float32)).unsqueeze(0).to(self.device)
        return self.converter(
            proposal_map=splatter,
            z_inv=z_inv.contiguous(),
            source_cameras_view_to_world=c2w,
            intrinsics=k,
            activate_output=True,
        )

    @torch.no_grad()
    def render_pc(self, pc: Dict[str, torch.Tensor], target_k: np.ndarray, target_w2c: np.ndarray) -> np.ndarray:
        k = torch.from_numpy(np.asarray(target_k, dtype=np.float32)).view(1, 1, 3, 3).to(self.device)
        w2c = torch.from_numpy(np.asarray(target_w2c, dtype=np.float32)).view(1, 1, 4, 4).to(self.device)
        out = render_predicted(
            pc=pc,
            world_view_transform=w2c,
            intrinsics=k,
            bg_color=self.bg,
            cfg=self.spl_cfg,
        )["render"][0, 0]
        return tensor_to_uint8_image(out)

    @torch.no_grad()
    def render_from_source(self, image_u8: np.ndarray, source_k: np.ndarray, source_c2w: np.ndarray, target_k: np.ndarray, target_w2c: np.ndarray) -> np.ndarray:
        pc = self.encode_source(image_u8, source_k, source_c2w)
        return self.render_pc(pc, target_k, target_w2c)


class DrQSinCroArgs:
    def __init__(self, cfg: Mapping[str, Any]):
        self.netdepth = int(cfg.get("netdepth", 8))
        self.netwidth = int(cfg.get("netwidth", 256))
        self.netdepth_fine = int(cfg.get("netdepth_fine", 8))
        self.netwidth_fine = int(cfg.get("netwidth_fine", 256))
        self.N_rand = int(cfg.get("N_rand", 2048))
        self.N_samples = int(cfg.get("N_samples", 64))
        self.N_importance = int(cfg.get("N_importance", 64))
        self.multires = int(cfg.get("multires", 10))
        self.multires_views = int(cfg.get("multires_views", 4))
        self.i_embed = int(cfg.get("i_embed", 0))
        self.use_viewdirs = bool(cfg.get("use_viewdirs", True))
        self.raw_noise_std = float(cfg.get("raw_noise_std", 0.0))
        self.white_bkgd = bool(cfg.get("white_bkgd", False))
        self.perturb = float(cfg.get("perturb", 1.0))
        self.lindisp = bool(cfg.get("lindisp", False))
        self.img_size = int(cfg.get("img_size", 128))
        self.patch_size = int(cfg.get("patch_size", 16))
        self.embed_dim = int(cfg.get("embed_dim", 256))
        self.vit_depth = int(cfg.get("vit_depth", 4))
        self.vit_num_heads = int(cfg.get("vit_num_heads", 4))
        self.vit_mlp_dim = int(cfg.get("vit_mlp_dim", 1024))
        self.decoder_depth = int(cfg.get("decoder_depth", 2))
        self.decoder_num_heads = int(cfg.get("decoder_num_heads", 2))
        self.decoder_mlp_dim = int(cfg.get("decoder_mlp_dim", 1024))
        self.decoder_output_dim = int(cfg.get("decoder_output_dim", 256))
        self.vit_encoder_mlp_dim = self.vit_mlp_dim
        self.vit_decoder_mlp_dim = self.decoder_mlp_dim
        self.time_interval = int(cfg.get("time_interval", 3))
        self.mask_ratio = float(cfg.get("mask_ratio", 0.75))
        self.num_view = int(cfg.get("num_views", 6))
        self.num_ref_views = int(cfg.get("num_ref_views", 2))
        self.batch_size = 1
        self.lrate = float(cfg.get("lrate", 5e-4))
        self.no_reload = True
        self.ft_path = None
        self.dataset_type = "metaworld"
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
        self.chunk = int(cfg.get("chunk", 32768))
        self.netchunk = int(cfg.get("netchunk", 65536))
        self.lr_decay = int(cfg.get("lrate_decay", 250))
        self.use_mae = True
        self.gamma = 1.0
        self.log_wandb = False
        self.enc_contrastive_margin = float(cfg.get("enc_contrastive_margin", 0.2))
        self.render_pose_path = None
        self.render_episode = None
        self.basedir = str(cfg.get("basedir", "./logs_sincro_vis"))
        self.expname = str(cfg.get("expname", "sincro_vis"))


def infer_sincro_full_ckpt(config_ckpt: str | None, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if config_ckpt is None:
        raise ValueError("No SinCro checkpoint was provided.")
    ckpt = Path(config_ckpt)
    name = ckpt.name
    if name.endswith("_encoder.tar"):
        candidate = ckpt.with_name(name.replace("_encoder.tar", ".tar"))
        if candidate.exists():
            return str(candidate)
    return str(ckpt)


class SinCroRenderer:
    def __init__(self, drq_config_path: str | Path, ckpt_path: str | Path | None, device: torch.device):
        from baselines.SinCro.sincro.MV_run_nerf import create_nerf

        self.cfg = load_yaml(drq_config_path)
        self.sc_cfg = dict(self.cfg["vision"]["sincro"])
        self.args = DrQSinCroArgs(self.sc_cfg)
        os.makedirs(Path(self.args.basedir) / self.args.expname, exist_ok=True)
        self.render_kwargs_train, self.render_kwargs_test, _start, _vars, _opt, self.latent_embed = create_nerf(
            self.args,
            self.args.basedir,
            self.args.expname,
        )
        self.device = device
        self.ckpt_path = infer_sincro_full_ckpt(self.sc_cfg.get("checkpoint_path"), None if ckpt_path is None else str(ckpt_path))
        state = torch.load(self.ckpt_path, map_location=device)
        if not isinstance(state, dict) or "network_fn_state_dict" not in state:
            raise ValueError(
                f"SinCro rendering needs a full NeRF checkpoint with network_fn_state_dict. Got {self.ckpt_path}. "
                "If your DrQ-v2 config points to *_encoder.tar, pass the matching full step_*.tar with --sincro_ckpt."
            )
        self.render_kwargs_train["network_fn"].load_state_dict(state["network_fn_state_dict"])
        self.render_kwargs_test["network_fn"].load_state_dict(state["network_fn_state_dict"])
        if self.render_kwargs_train.get("network_fine") is not None and "network_fine_state_dict" in state:
            self.render_kwargs_train["network_fine"].load_state_dict(state["network_fine_state_dict"])
            self.render_kwargs_test["network_fine"].load_state_dict(state["network_fine_state_dict"])
        self.latent_embed.load_state_dict(select_state_dict(state, ("latent_embed_state_dict", "model_state_dict", "state_dict")), strict=True)
        self.latent_embed.to(device).eval()
        for value in self.render_kwargs_test.values():
            if hasattr(value, "eval"):
                value.eval()

    @torch.no_grad()
    def encode_single_view(self, image_or_sequence: np.ndarray) -> torch.Tensor:
        arr = np.asarray(image_or_sequence)
        if arr.ndim == 3:
            arr = np.repeat(arr[None], self.args.time_interval, axis=0)
        if arr.ndim != 4:
            raise ValueError(f"Expected image (H,W,3) or sequence (T,H,W,3), got {arr.shape}.")
        if arr.shape[0] != self.args.time_interval:
            if arr.shape[0] > self.args.time_interval:
                arr = arr[-self.args.time_interval :]
            else:
                pad = np.repeat(arr[:1], self.args.time_interval - arr.shape[0], axis=0)
                arr = np.concatenate([pad, arr], axis=0)
        x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        b, t, c, h, w = x.shape
        primary_images = x.permute(0, 1, 3, 4, 2).contiguous()
        latent, mask, ids_restore = self.latent_embed.SinCro_image_encoder(primary_images, mask_ratio=0.0, T=t, is_ref=False)
        ref_for_encoder = primary_images.repeat(self.args.num_ref_views, 1, 1, 1, 1)
        ref_latent, _, _ = self.latent_embed.SinCro_image_encoder(ref_for_encoder, mask_ratio=0.0, T=t, is_ref=True)
        ref_latent = rearrange(ref_latent[:, 1:, :], "b (t hw) d -> b t hw d", t=t)[:, -1]
        ref_latent = rearrange(ref_latent, "(v b) hw d -> b (v hw) d", v=self.args.num_ref_views, b=b)
        latent, _, _ = self.latent_embed.SinCro_state_encoder(latent, ref_latent, mask, ids_restore)
        return latent.reshape(b, t, 1, -1)[:, -1, 0]

    @torch.no_grad()
    def render_latent(self, latent: torch.Tensor, target_k: np.ndarray, target_c2w_gl: np.ndarray, height: int, width: int) -> np.ndarray:
        from baselines.SinCro.train import render_full_image

        k = torch.from_numpy(np.asarray(target_k, dtype=np.float32)).to(self.device)
        c2w = torch.from_numpy(np.asarray(target_c2w_gl, dtype=np.float32)).to(self.device)
        return render_full_image(
            int(height),
            int(width),
            k,
            c2w,
            latent=latent.to(self.device),
            args=self.args,
            render_kwargs=self.render_kwargs_test,
        )

    @torch.no_grad()
    def render_from_source(self, image_or_sequence: np.ndarray, target_k: np.ndarray, target_c2w_gl: np.ndarray, height: int, width: int) -> np.ndarray:
        latent = self.encode_single_view(image_or_sequence)
        return self.render_latent(latent, target_k, target_c2w_gl, height, width)


class ReViWoRenderer:
    def __init__(self, drq_config_path: str | Path, ckpt_path: str | Path | None, device: torch.device):
        from baselines.ReViWo.ReViWo.common.models.multiview_vae import MultiViewBetaVAE
        from models.transformer import STTransConfig
        from models.vae import CodebookConfig

        self.cfg = load_yaml(drq_config_path)
        rv_cfg = dict(self.cfg["vision"]["reviwo"])
        if ckpt_path is not None:
            rv_cfg["checkpoint_path"] = str(ckpt_path)
        state = torch.load(str(rv_cfg["checkpoint_path"]), map_location="cpu")
        state_dict = select_state_dict(state, ("model_state_dict", "state_dict"))
        view_proj = state_dict.get("view_encoder_output_proj.weight")
        latent_proj = state_dict.get("latent_encoder_output_proj.weight")
        if view_proj is not None:
            rv_cfg.setdefault("view_codebook", {})["embed_dim"] = int(view_proj.shape[0])
        if latent_proj is not None:
            rv_cfg.setdefault("latent_codebook", {})["embed_dim"] = int(latent_proj.shape[0])
        self.model = MultiViewBetaVAE(
            view_encoder_config=STTransConfig(**dict(rv_cfg["view_encoder"])),
            latent_encoder_config=STTransConfig(**dict(rv_cfg["latent_encoder"])),
            decoder_config=STTransConfig(**dict(rv_cfg["decoder"])),
            view_cb_config=CodebookConfig(**dict(rv_cfg["view_codebook"])),
            latent_cb_config=CodebookConfig(**dict(rv_cfg["latent_codebook"])),
            img_size=int(rv_cfg.get("img_size", self.cfg.get("env", {}).get("image_height", 128))),
            patch_size=int(rv_cfg.get("patch_size", 16)),
            fusion_style=str(rv_cfg.get("fusion_style", "plus")),
            use_latent_vq=bool(rv_cfg.get("use_latent_vq", True)),
            is_latent_ae=bool(rv_cfg.get("is_latent_ae", False)),
            use_view_vq=bool(rv_cfg.get("use_view_vq", True)),
            is_view_ae=bool(rv_cfg.get("is_view_ae", False)),
        )
        self.model.load_state_dict(state_dict, strict=True)
        for module in self.model.modules():
            if hasattr(module, "init_kmeans"):
                module.init_kmeans = False
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def reconstruct_with_target_view(self, source_image_u8: np.ndarray, target_view_image_u8: np.ndarray) -> np.ndarray:
        source = repo_image_to_tensor(source_image_u8).unsqueeze(0).to(self.device)
        target = repo_image_to_tensor(target_view_image_u8).unsqueeze(0).to(self.device)
        _z_v_source, _, z_l_source, _, _ = self.model.encode(
            source,
            deterministic_view=True,
            deterministic_latent=True,
        )
        z_v_target, _, _z_l_target, _, _ = self.model.encode(
            target,
            deterministic_view=True,
            deterministic_latent=True,
        )
        pred = self.model.decode(z_v_target, z_l_source)[0]
        return tensor_to_uint8_image(pred)


def mse_uint8(pred: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float32) / 255.0
    t = np.asarray(target, dtype=np.float32) / 255.0
    return float(np.mean((p - t) ** 2))


def ssim_torch(pred: np.ndarray, target: np.ndarray, device: torch.device | None = None) -> float:
    device = device or torch.device("cpu")
    p = torch.from_numpy(np.asarray(pred, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    t = torch.from_numpy(np.asarray(target, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    window = torch.ones((3, 1, 11, 11), device=device, dtype=torch.float32) / 121.0
    mu_p = F.conv2d(p, window, padding=5, groups=3)
    mu_t = F.conv2d(t, window, padding=5, groups=3)
    sigma_p = F.conv2d(p * p, window, padding=5, groups=3) - mu_p * mu_p
    sigma_t = F.conv2d(t * t, window, padding=5, groups=3) - mu_t * mu_t
    sigma_pt = F.conv2d(p * t, window, padding=5, groups=3) - mu_p * mu_t
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2.0 * mu_p * mu_t + c1) * (2.0 * sigma_pt + c2)) / ((mu_p ** 2 + mu_t ** 2 + c1) * (sigma_p + sigma_t + c2) + 1e-8)
    return float(ssim_map.mean().detach().cpu().item())


class LPIPSMetric:
    def __init__(self, device: torch.device, net: str = "alex", required: bool = False):
        self.device = device
        self.model = None
        try:
            import lpips

            self.model = lpips.LPIPS(net=net).to(device).eval()
        except Exception as exc:
            if required:
                raise RuntimeError("LPIPS is required but could not be imported. Install lpips>=0.1.4.") from exc
            self.error = exc

    @torch.no_grad()
    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float | None:
        if self.model is None:
            return None
        p = torch.from_numpy(np.asarray(pred, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t = torch.from_numpy(np.asarray(target, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        p = p * 2.0 - 1.0
        t = t * 2.0 - 1.0
        return float(self.model(p, t).mean().detach().cpu().item())
