# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see third_party/rvt/LICENSE for details].
#
# Modified for splatter4d (3D-MVP port) from github.com/NVlabs/RVT @ 367995a1a2169b6352bf4e8b0ed405890462a3a0:
#   rvt/mvt/mvt_single.py  MVT.__init__ (input_preprocess, patchify, pos_encoding, fc_bef_attn, layers, pixel_loc)
#                          and MVT.forward (tokenisation, within-view / cross-view attention, per-view max-pool);
#   rvt/mvt/mvt.py         MVT.render (channel order of the virtual images);
#   rvt/mvt/renderer.py    PointsRendererWithDepth.forward (per-view depth normalisation);
#   rvt/utils/rvt_utils.py move_pc_in_bound (scene-bounds crop), returning a mask for fixed-size batches.
# The layers themselves are RVT's own classes, imported from the pinned submodule (s4d/baselines/mvp3d/rvt.py).
"""RVT-1 parts of the 3D-MVP encoder E and renderer R (Qi et al., CVPR 2025, §3.1-3.2).

``MVTEncoder`` is RVT-1's ``MVTSingle`` up to and including its attention layers, with image tokens only
(``add_proprio=False``, ``add_lang=False``: 3D-MVP pretrains E without language or proprioception), ``pe_fix=True``,
``self_cross_ver=1`` (first half of the layers within each view, second half across views) and
``inp_pre_pro=True``. Attribute names are MVTSingle's, so its state dict loads into an ``MVTSingle`` built with the
same arguments (checked in tests/_mvp3d_rvt_worker.py). Its output is 3D-MVP's latent ``z`` (5N tokens x hidden
size H, Eq. 3); RVT's action head (``fc_aft_attn``, up-sampling, heatmaps, ``feat_fc``) is 3D-MVP's action decoder
D and is not part of E.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from s4d.baselines.mvp3d import rvt


def pixel_loc(num_img: int, img_size: int) -> torch.Tensor:
    """(num_img, 3, img_size, img_size) constant pixel-location channels (MVTSingle, ``add_pixel_loc``)."""
    loc = torch.zeros((num_img, 3, img_size, img_size))
    loc[:, 0, :, :] = torch.linspace(-1, 1, num_img).unsqueeze(-1).unsqueeze(-1)
    loc[:, 1, :, :] = torch.linspace(-1, 1, img_size).unsqueeze(0).unsqueeze(-1)
    loc[:, 2, :, :] = torch.linspace(-1, 1, img_size).unsqueeze(0).unsqueeze(0)
    return loc


def normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    """(..., h, w) view-space depth with -1 at empty pixels -> mean-centred per view over the covered pixels, -1 at
    empty pixels (PointsRendererWithDepth.forward, ``with_depth``; reductions over the last two axes)."""
    depth = depth.clone()
    h, w = depth.shape[-2:]
    depth_0 = depth == -1
    depth_sum = torch.sum(depth, (-2, -1)) + torch.sum(depth_0, (-2, -1))
    depth_mean = depth_sum / ((h * w) - torch.sum(depth_0, (-2, -1)))
    depth -= depth_mean.unsqueeze(-1).unsqueeze(-1)
    depth[depth_0] = -1
    return depth


def bounds_mask(pc: torch.Tensor, bounds) -> torch.Tensor:
    """(..., 3) points -> bool (...): inside ``bounds`` = [x_min, y_min, z_min, x_max, y_max, z_max] and not NaN
    (move_pc_in_bound's ``~inv_pnt``; it removes the other points, here they are masked)."""
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    inv_pnt = (
        (pc[..., 0] < x_min)
        | (pc[..., 0] > x_max)
        | (pc[..., 1] < y_min)
        | (pc[..., 1] > y_max)
        | (pc[..., 2] < z_min)
        | (pc[..., 2] > z_max)
        | torch.isnan(pc[..., 0])
        | torch.isnan(pc[..., 1])
        | torch.isnan(pc[..., 2])
    )
    return ~inv_pnt


def assemble_virtual_images(feat_img: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Rendered point features (bs, num_img, 6, h, w) = [xyz (cube), rgb] and normalised depth (bs, num_img, h, w)
    -> RVT-1's 10-channel virtual images (bs, num_img, 10, h, w): xyz, rgb, depth, pixel location (MVT.render:
    features ``cat((pc, img_feat))``, depth appended by the renderer, then ``pixel_loc``)."""
    bs, num_img, _, h, w = feat_img.shape
    img = torch.cat((feat_img, depth.unsqueeze(2)), dim=2)
    loc = pixel_loc(num_img, h).to(img)
    return torch.cat((img, loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2)


def multiview_layers(layers: nn.ModuleList, x: torch.Tensor, bs: int, num_img: int, num_tok: int) -> torch.Tensor:
    """MVTSingle's ``self_cross_ver == 1`` block without language tokens: the first half of ``layers`` attends within
    each view, the second half across all views. x: (bs, num_img * num_tok, dim)."""
    imgx = x.reshape(bs * num_img, num_tok, -1)
    for self_attn, self_ff in layers[: len(layers) // 2]:
        imgx = self_attn(imgx) + imgx
        imgx = self_ff(imgx) + imgx
    x = imgx.view(bs, num_img * num_tok, -1)
    for self_attn, self_ff in layers[len(layers) // 2 :]:
        x = self_attn(x) + x
        x = self_ff(x) + x
    return x


def rvt_layers(dim: int, heads: int, dim_head: int, dropout: float, depth: int) -> nn.ModuleList:
    """``depth`` x [PreNorm(Attention), PreNorm(FeedForward)] exactly as MVTSingle builds them (no weight tying)."""
    attn = rvt.attn()
    get_attn_attn = lambda: attn.PreNorm(
        dim,
        attn.Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, use_fast=False),
    )
    get_attn_ff = lambda: attn.PreNorm(dim, attn.FeedForward(dim))
    get_attn_attn, get_attn_ff = map(attn.cache_fn, (get_attn_attn, get_attn_ff))
    cache_args = {"_cache": False}
    return nn.ModuleList([nn.ModuleList([get_attn_attn(**cache_args), get_attn_ff(**cache_args)]) for _ in range(depth)])


class MVTEncoder(nn.Module):
    """3D-MVP encoder E (§3.2, Eq. 3/4): RVT-1 MVT tokenisation + ``depth`` attention layers (half within-view, half
    cross-view). Input: (bs, num_img, 10, img_size, img_size) virtual images; output z: (bs, num_img * N, attn_dim)."""

    def __init__(
        self,
        img_size: int,
        img_patch_size: int,
        im_channels: int,
        attn_dim: int,
        attn_heads: int,
        attn_dim_head: int,
        depth: int,
        attn_dropout: float,
        activation: str = "lrelu",
        num_img: int = 5,
        inp_img_feat_dim: int = 10,
    ):
        super().__init__()
        if img_size % img_patch_size:
            raise ValueError(f"img_size {img_size} is not a multiple of the patch size {img_patch_size}")
        if depth % 2:
            raise ValueError("the within-view / cross-view split needs an even number of layers")
        attn = rvt.attn()
        self.depth = depth
        self.img_size = img_size
        self.img_patch_size = img_patch_size
        self.im_channels = im_channels
        self.attn_dim = attn_dim
        self.num_img = num_img
        spatial_size = img_size // self.img_patch_size
        self.input_dim_before_seq = self.im_channels  # add_proprio=False
        num_pe_token = spatial_size**2 * self.num_img  # pe_fix=True
        self.pos_encoding = nn.Parameter(torch.randn(1, num_pe_token, self.input_dim_before_seq))
        self.input_preprocess = attn.Conv2DBlock(
            inp_img_feat_dim,
            self.im_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation=activation,
        )
        self.patchify = attn.Conv2DBlock(
            self.im_channels,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=activation,
            padding=0,
        )
        self.fc_bef_attn = attn.DenseBlock(self.input_dim_before_seq, attn_dim, norm=None, activation=None)
        self.layers = rvt_layers(attn_dim, attn_heads, attn_dim_head, attn_dropout, depth)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        bs, num_img, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.img_patch_size
        assert num_img == self.num_img
        assert h == w == self.img_size

        img = img.reshape(bs * num_img, img_feat_dim, h, w)
        d0 = self.input_preprocess(img)
        ins = self.patchify(d0)
        ins = ins.view(bs, num_img, self.im_channels, num_pat_img, num_pat_img).transpose(1, 2).clone()
        ins = rearrange(ins, "b d ... -> b ... d")  # [B, num_img, np, np, im_channels]
        ins = rearrange(ins, "b ... d -> b (...) d")  # [B, num_img * np * np, im_channels]
        ins += self.pos_encoding
        x = self.fc_bef_attn(ins)
        return multiview_layers(self.layers, x, bs, num_img, num_pat_img * num_pat_img)

    def global_feature(self, z: torch.Tensor) -> torch.Tensor:
        """RVT's global image feature, a spatial max-pool per view, applied to z: (bs, attn_dim * num_img), flattened
        channel-major as in MVTSingle (``_feat.view(bs, -1)`` of a (bs, dim, num_img) tensor)."""
        bs = z.shape[0]
        num_pat_img = self.img_size // self.img_patch_size
        x = z.view(bs, self.num_img, num_pat_img, num_pat_img, z.shape[-1])
        x = rearrange(x, "b ... d -> b d ...")  # [B, dim, num_img, np, np]
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        return _feat.view(bs, -1)
