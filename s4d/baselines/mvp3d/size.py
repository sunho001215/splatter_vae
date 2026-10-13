"""Encoder size of 3D-MVP configurations, measured on the real modules (CPU ``FlopCounterMode``, a multiply-add counts
2 FLOPs; large configurations on the meta device, which counts the same operators without computing them).

RL time: one RGB-D observation is rendered into the five virtual views (reported separately; FlopCounterMode counts
only its matrix products, the lift and the view transforms) and encoded by E; the per-view max-pool costs no FLOPs.
"""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch.utils.flop_counter import FlopCounterMode

from s4d.baselines.mvp3d.model import ORIGINAL, RVT1, SCENE_BOUNDS, MAEDecoder, MVP3DConfig, build_encoder
from s4d.baselines.mvp3d.renderer import RADIUS, render_rgbd

SINCRO_PARAMS = 9_064_704  # SinCro's RL-time encoder (parameters reached by the RL forward)
SINCRO_FLOPS_PER_FRAME = 1_786_078_891


def _flops(fn, *args) -> int:
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        fn(*args)
    return int(counter.get_total_flops())


def encoder_cost(cfg: MVP3DConfig, device: str = "cpu") -> dict:
    """Parameters and FLOPs of E (and of the pretraining-only decoder) on one observation's five virtual views."""
    with torch.device(device):
        encoder = build_encoder(cfg).eval()
        decoder = MAEDecoder(cfg).eval()
        images = torch.zeros(1, cfg.num_img, 10, cfg.img_size, cfg.img_size)
        z = encoder(images)
        flops = _flops(encoder, images)
        decoder_flops = _flops(decoder, z)
    tokens = cfg.num_img * (cfg.img_size // cfg.img_patch_size) ** 2
    return {
        "config": asdict(cfg),
        "tokens": tokens,
        "encoder_params": sum(p.numel() for p in encoder.parameters()),
        "encoder_flops_per_frame": flops,
        "decoder_params": sum(p.numel() for p in decoder.parameters()),
        "decoder_flops_per_frame": decoder_flops,
        "latent_size": tokens * cfg.attn_dim,
        "feature_size": cfg.num_img * cfg.attn_dim,
    }


def rl_cost(encoder, img_size: int, image_hw: int = 128, bounds=SCENE_BOUNDS, radius: float = RADIUS) -> dict:
    """RL-time cost of one frame (CPU): parameters reached by the forward (those receiving a gradient), encoder FLOPs,
    rendering FLOPs (counted operators only) and the output size."""
    gen = torch.Generator().manual_seed(0)
    rgb = torch.rand(1, 1, 3, image_hw, image_hw, generator=gen)
    depth = 0.8 + 0.4 * torch.rand(1, 1, image_hw, image_hw, generator=gen)
    from s4d.data.metaworld.cameras import camera_rig  # noqa: PLC0415

    rig = camera_rig(image_hw, image_hw)
    K, c2w = torch.from_numpy(rig["K"][:1])[None], torch.from_numpy(rig["c2w"][:1])[None]
    render_flops = _flops(render_rgbd, rgb, depth, K, c2w, bounds, img_size, radius)
    images = render_rgbd(rgb, depth, K, c2w, bounds, img_size, radius)
    encoder.requires_grad_(True)
    feature = encoder.global_feature(encoder(images))
    feature.sum().backward()
    used = sum(p.numel() for p in encoder.parameters() if p.grad is not None)
    encoder.zero_grad(set_to_none=True)
    encoder.requires_grad_(False)
    return {
        "rl_params": used,
        "encoder_flops_per_frame": _flops(lambda x: encoder.global_feature(encoder(x)), images),
        "render_flops_per_frame": render_flops,
        "feature_size": int(feature.shape[-1]),
    }


def size_table() -> dict:
    """Original configurations (computed) and the size-matched one (measured), with SinCro's budget."""
    rows = {
        "RVT-1 MVT encoder part (220 px, patch 11, 512 wide, 8x64 heads)": encoder_cost(RVT1, "meta"),
        "3D-MVP paper (220 px, patch 10, 1024 wide, 8x128 heads)": encoder_cost(ORIGINAL, "meta"),
        "3D-MVP size-matched (48 px, patch 12, 256 wide, 8x32 heads)": encoder_cost(MVP3DConfig(), "cpu"),
    }
    return {"budget": {"params": SINCRO_PARAMS, "flops_per_frame": SINCRO_FLOPS_PER_FRAME}, "rows": rows}
