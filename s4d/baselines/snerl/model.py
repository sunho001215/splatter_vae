"""SNeRL networks, training step, rendering and encoder export, around the official ``nerf_pretrain`` code.

The networks, optimiser, renderer and loss terms are the official ones (``create_nerf``, ``render``, ``img2mse``,
``get_rays``, ``PixelEncoder``, ``ViTExtractor``). The official training loop is inline in ``train()``
(run_nerf.py:668-886), so its body is reproduced here step by step with the upstream line numbers; tests execute those
upstream source lines on fixed inputs and compare. Rendering code runs under ``default_tensor_type`` (see upstream.py).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from s4d.baselines.snerl.data import POSE_SCALE, normalise_c2w, official_K
from s4d.baselines.snerl.upstream import default_tensor_type, official
from s4d.data.metaworld.cameras import LOOKAT

EXPORT_FORMAT = "splatter4d-baseline-snerl-v1"
# ``PixelEncoder()`` defaults, exactly as ``create_nerf`` builds the encoder (run_nerf.py:188; multiview_encoder.py:21).
ENCODER_CFG = {"obs_shape": [3, 128, 128], "feature_dim": 63, "num_layers": 4, "num_filters": 32, "output_logits": True}


@dataclass
class SNeRL:
    """The objects ``create_nerf`` returns, plus the scene bounds ``train()`` adds (run_nerf.py:550-551, 595-600)."""

    encoder: nn.Module
    render_kwargs_train: dict
    render_kwargs_test: dict
    optimizer: torch.optim.Optimizer
    args: argparse.Namespace
    device: torch.device

    def state_dict(self) -> dict:
        return {
            "latent_embed_state_dict": self.encoder.state_dict(),
            "network_fn_state_dict": self.render_kwargs_train["network_fn"].state_dict(),
            "network_fine_state_dict": self.render_kwargs_train["network_fine"].state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.encoder.load_state_dict(state["latent_embed_state_dict"])
        self.render_kwargs_train["network_fn"].load_state_dict(state["network_fn_state_dict"])
        self.render_kwargs_train["network_fine"].load_state_dict(state["network_fine_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

    def train(self, mode: bool = True) -> None:
        for module in (self.encoder, self.render_kwargs_train["network_fn"], self.render_kwargs_train["network_fine"]):
            module.train(mode)


def official_args(nerf_cfg: dict) -> argparse.Namespace:
    """The ``args`` the official ``create_nerf``/``train`` read, from the config's ``nerf`` section."""
    return argparse.Namespace(**nerf_cfg, basedir="", expname="")


def create(nerf_cfg: dict, device: torch.device | str) -> SNeRL:
    """Official ``create_nerf(args)`` (encoder, coarse and fine NeRF, one Adam over all three) on ``device``."""
    device = torch.device(device)
    run_nerf = official("run_nerf")["run_nerf"]
    run_nerf.device = device  # the official module renders and places models on its global ``device``
    args = official_args(nerf_cfg)
    with default_tensor_type(device):  # the official process creates everything under the CUDA default type
        train_kwargs, test_kwargs, _, _, optimizer, encoder = run_nerf.create_nerf(args)
    bounds = {"near": float(nerf_cfg["near"]), "far": float(nerf_cfg["far"])}
    train_kwargs.update(bounds)
    test_kwargs.update(bounds)
    return SNeRL(encoder, train_kwargs, test_kwargs, optimizer, args, device)


def update_learning_rate(optimizer: torch.optim.Optimizer, lrate: float, lrate_decay: int, global_step: int) -> float:
    """run_nerf.py:775-779, applied after ``optimizer.step()`` with the step count before the increment."""
    decay_rate = 0.1
    decay_steps = lrate_decay * 1000
    new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lrate
    return new_lrate


def dino_target(extractor, target: torch.Tensor) -> torch.Tensor:
    """run_nerf.py:711-715: DINO ViT-S/8 (stride 4) layer-11 key descriptors of the (H, W, 3) target in [0, 1], with
    SNeRL's scalar normalisation, bilinearly resized to the image (31x31 tokens -> 128x128 for our images)."""
    with torch.no_grad():
        target_dino = (target - 0.45) / 0.225
        target_dino = extractor.extract_descriptors(target_dino.unsqueeze(dim=0).permute(0, 3, 1, 2), 11, "key", False)
        target_dino = target_dino.view(1, *extractor.num_patches, -1)
        size = tuple(target.shape[:2])
        return F.interpolate(target_dino.permute(0, 3, 1, 2), size, mode="bilinear").permute(0, 2, 3, 1)[0]


def ray_coords(H: int, W: int, precrop: bool, precrop_frac: float) -> torch.Tensor:
    """run_nerf.py:723-736 (call under ``default_tensor_type``): (N, 2) row/column coordinates of the candidate rays,
    the central crop while ``i < precrop_iters``."""
    if precrop:
        dH = int(H // 2 * precrop_frac)
        dW = int(W // 2 * precrop_frac)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                indexing="ij",
            ),
            -1,
        )
    else:
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W), indexing="ij"), -1)
    return torch.reshape(coords, [-1, 2])


def select_rays(rays_o, rays_d, target, target_semantic, target_dino, coords, select_inds):
    """run_nerf.py:738-745: the chosen rays and their RGB, semantic and DINO targets."""
    select_coords = coords[select_inds].long()
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
    batch_rays = torch.stack([rays_o, rays_d], 0)
    target_s = target[select_coords[:, 0], select_coords[:, 1]]
    target_semantic_s = target_semantic[select_coords[:, 0], select_coords[:, 1]]
    target_dino_s = target_dino[select_coords[:, 0], select_coords[:, 1]]
    return batch_rays, target_s, target_semantic_s, target_dino_s


def snerl_loss(rgb, semantic, dino, extras, target_s, target_semantic_s, target_dino_s, semantic_weight: float = 0.004,
               dino_weight: float = 0.04) -> tuple[torch.Tensor, dict]:
    """run_nerf.py:754-768: MSE(RGB) + 0.004 CE(semantic) + 0.04 MSE(DINO), for the fine and the coarse renders."""
    helpers = official("run_nerf_helpers")["run_nerf_helpers"]
    img2mse, mse2psnr = helpers.img2mse, helpers.mse2psnr
    img_loss = img2mse(rgb, target_s)
    dino_loss = img2mse(dino, target_dino_s)
    semantic_loss = nn.CrossEntropyLoss()(semantic, target_semantic_s)
    loss = img_loss + semantic_weight * semantic_loss + dino_weight * dino_loss
    psnr = mse2psnr(img_loss)
    terms = {"img_loss": img_loss, "semantic_loss": semantic_loss, "dino_loss": dino_loss, "psnr": psnr}
    if "rgb0" in extras:
        img_loss0 = img2mse(extras["rgb0"], target_s)
        semantic_loss0 = nn.CrossEntropyLoss()(extras["semantic0"], target_semantic_s)
        dino_loss_0 = img2mse(extras["dino0"], target_dino_s)
        loss = loss + img_loss0 + semantic_weight * semantic_loss0 + dino_weight * dino_loss_0
        terms.update(img_loss0=img_loss0, semantic_loss0=semantic_loss0, dino_loss0=dino_loss_0, psnr0=mse2psnr(img_loss0))
    return loss, {"loss": loss, **terms}


def to_device_scene(scene: dict, device: torch.device) -> dict:
    """uint8 images -> float in [0, 1] (as ``load_blender_data``: ``imgs / 255.``) on ``device``."""
    out = {}
    for key, value in scene.items():
        if key in ("episode", "t", "target", "target_K"):  # K stays a host array, as in the official loop
            out[key] = value
            continue
        tensor = torch.as_tensor(value)
        if key in ("obs_rgb", "target_rgb"):
            tensor = tensor.float() / 255.0
        out[key] = tensor.to(device, non_blocking=True)
    return out


def training_step(model: SNeRL, extractor, scene: dict, global_step: int, rng: np.random.Generator,
                  semantic_weight: float = 0.004, dino_weight: float = 0.04) -> tuple[torch.Tensor, dict]:
    """The official loop body for one scene (run_nerf.py:687-768) up to the loss; the caller runs backward, the
    optimiser step and ``update_learning_rate``. ``scene`` is a ``SNeRLScenes`` item on the device (``to_device_scene``);
    ``rng`` replaces NumPy's global RNG for the ray choice (same distribution: ``choice(..., replace=False)``)."""
    args = model.args
    obs_img = scene["obs_rgb"].permute(0, 3, 1, 2)  # :697-700
    obs_poses = scene["obs_c2w"].reshape(-1, 16)  # :698
    latent = model.encoder(obs_img, obs_poses)  # :703
    target = scene["target_rgb"]  # :707-708
    pose = scene["target_c2w"][:3, :4]  # :709
    target_dino = dino_target(extractor, target)  # :711-715
    target_semantic = scene["target_semantic"].long()  # :717-718
    H, W = target.shape[:2]
    K = np.asarray(scene["target_K"], dtype=np.float64)
    run_nerf = official("run_nerf")["run_nerf"]
    with default_tensor_type(model.device):
        rays_o, rays_d = run_nerf.get_rays(H, W, K, pose)  # :721
        coords = ray_coords(H, W, global_step + 1 < args.precrop_iters, args.precrop_frac)  # :723-736, i = step + 1
        select_inds = rng.choice(coords.shape[0], size=[args.N_rand], replace=False)  # :737
        batch_rays, target_s, target_semantic_s, target_dino_s = select_rays(
            rays_o, rays_d, target, target_semantic, target_dino, coords, select_inds
        )
        rgb, disp, acc, semantic, dino, extras = run_nerf.render(  # :749-751
            H, W, K, chunk=args.chunk, rays=batch_rays, verbose=global_step + 1 < 10, retraw=True, latent=latent,
            **model.render_kwargs_train,
        )
        loss, terms = snerl_loss(rgb, semantic, dino, extras, target_s, target_semantic_s, target_dino_s,
                                 semantic_weight, dino_weight)
    return loss, terms


@torch.no_grad()
def live_density_fractions(model: SNeRL, scene: dict, stride: int = 8) -> tuple[float, float]:
    """Fraction of points with positive raw density (``raw2alpha`` uses ``relu(sigma)``, run_nerf.py:259) for the
    coarse and the fine network, on the coarse samples of every ``stride``-th pixel of the scene's target camera.
    A network at zero receives no gradient through any rendered quantity and never recovers (about half of the
    initialisations do this with the official code and default PyTorch init)."""
    run_nerf = official("run_nerf")["run_nerf"]
    kwargs = model.render_kwargs_train
    H, W = scene["target_rgb"].shape[:2]
    obs = scene["obs_rgb"].permute(0, 3, 1, 2)
    latent = model.encoder(obs, scene["obs_c2w"].reshape(-1, 16))
    with default_tensor_type(model.device):
        K = np.asarray(scene["target_K"], dtype=np.float64)
        rays_o, rays_d = run_nerf.get_rays(H, W, K, scene["target_c2w"][:3, :4])
        rays_o, rays_d = rays_o[::stride, ::stride].reshape(-1, 3), rays_d[::stride, ::stride].reshape(-1, 3)
        z_vals = torch.linspace(kwargs["near"], kwargs["far"], kwargs["N_samples"])
        pts = rays_o[:, None] + rays_d[:, None] * z_vals[None, :, None]
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        fractions = []
        for network in (kwargs["network_fn"], kwargs["network_fine"]):
            raw = kwargs["network_query_fn"](pts, viewdirs, network, latent)
            fractions.append(float((raw[..., 3] > 0).float().mean()))
    return fractions[0], fractions[1]


def encode_batch(encoder: nn.Module, rgb: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """Batched ``PixelEncoder.forward`` (multiview_encoder.py:81-107) over N scenes, as the official RL encoder does it
    (snerl/encoder.py:320-364, ``MeanEncoder``): rgb (N, V, 3, H, W) in [0, 1], poses (N, V, 16) normalised flattened
    OpenGL c2w -> the LayerNorm output (N, 63) (``output_logits=True``; the official actor adds tanh, the critic not)."""
    n, v = rgb.shape[:2]
    conv = torch.relu(encoder.convs[0](rgb.reshape(n * v, *rgb.shape[2:])))  # forward_conv without its output log
    for i in range(1, encoder.num_layers):
        conv = torch.relu(encoder.convs[i](conv))
    h = encoder.fc(conv.reshape(conv.size(0), -1)).view(n, v, -1)
    h = encoder.mlp1(torch.cat([h, poses], dim=2))
    h_fc = encoder.mlp2(torch.mean(h, dim=1))
    h_norm = encoder.ln(h_fc)
    return h_norm if encoder.output_logits else torch.tanh(h_norm)


def encode_views(model: SNeRL, rgb: torch.Tensor, c2w_cv: torch.Tensor, lookat=LOOKAT, scale: float = POSE_SCALE):
    """Latent (1, 63) of one scene from V input views: rgb (V, H, W, 3) in [0, 1], OpenCV c2w (V, 4, 4)."""
    poses = normalise_c2w(c2w_cv.to(model.device).double(), lookat, scale).float().reshape(-1, 16)
    return model.encoder(rgb.to(model.device).permute(0, 3, 1, 2), poses)


@torch.no_grad()
def render_camera(model: SNeRL, latent: torch.Tensor, K_cv, c2w_cv, H: int, W: int, chunk: int = 4096,
                  lookat=LOOKAT, scale: float = POSE_SCALE) -> dict[str, torch.Tensor]:
    """Full image of an arbitrary camera (our OpenCV K, c2w) through the official ``render`` (test settings: no
    jitter, no noise; render_path, run_nerf.py:150). ``chunk`` only bounds memory. Depth: ``1 / disp_map``, the
    weight-normalised expected camera-z depth, in metres (divided by the pose scale)."""
    run_nerf = official("run_nerf")["run_nerf"]
    K = official_K(np.asarray(K_cv, dtype=np.float64))
    c2w = normalise_c2w(torch.as_tensor(c2w_cv, dtype=torch.float64), lookat, scale).float().to(model.device)
    with default_tensor_type(model.device):
        rgb, disp, acc, semantic, dino, _ = run_nerf.render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], latent=latent,
                                                            **model.render_kwargs_test)
    return {"rgb": rgb, "depth": 1.0 / disp / scale, "acc": acc, "semantic": semantic, "dino": dino}


def build_encoder(encoder_cfg: dict | None = None) -> nn.Module:
    """The official ``PixelEncoder`` with the exported constructor arguments."""
    cfg = dict(ENCODER_CFG if encoder_cfg is None else encoder_cfg)
    pixel_encoder = official("multiview_encoder")["multiview_encoder"].PixelEncoder
    return pixel_encoder(obs_shape=tuple(cfg["obs_shape"]), feature_dim=cfg["feature_dim"], num_layers=cfg["num_layers"],
                         num_filters=cfg["num_filters"], output_logits=cfg["output_logits"])


def save_export(path: str | Path, encoder: nn.Module, step: int, num_views: int, lookat=LOOKAT,
                scale: float = POSE_SCALE, random_init: bool = False) -> None:
    """Encoder-only export for the frozen RL encoder: weights, constructor config, the pose normalisation and the
    number of input views. SNeRL pretrains on single time steps, so there is no pretraining frame spacing."""
    payload = {
        "format": EXPORT_FORMAT,
        "encoder_cfg": dict(ENCODER_CFG),
        "pose": {"lookat": [float(x) for x in lookat], "scale": float(scale),
                 "convention": "c2w_gl = c2w_cv @ diag(1,-1,-1,1); translation (t - lookat) * scale; flattened 4x4"},
        "num_views": int(num_views),
        "image_size": int(ENCODER_CFG["obs_shape"][-1]),
        "frame_spacing": None,
        "random_init": bool(random_init),
        "step": int(step),
        "state_dict": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
    }
    tmp = Path(path).with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_export(path: str | Path) -> tuple[dict, nn.Module]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != EXPORT_FORMAT:
        raise ValueError(f"{path} is not a SNeRL encoder export")
    encoder = build_encoder(payload["encoder_cfg"])
    encoder.load_state_dict(payload["state_dict"], strict=True)
    return payload, encoder.eval().requires_grad_(False)
