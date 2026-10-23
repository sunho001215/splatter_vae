"""Bidirectional 3D Chamfer loss between decoded Gaussian centres and the training cameras' fused depth.

Variant ``chamfer_lean`` (registered 2026-10-21; off by default, ``loss.chamfer`` = 0).

Target cloud per (sample, timestep), from training data only (the batch's training cameras; never a held-out or
moving-camera set): every training-camera pixel lifted to world coordinates with its stored depth and calibration
(``s4d.geometry.lift_depth``), kept under the reference validity rule (depth > 0, and depth < far when
``loss.depth_valid_far``), deduplicated on a voxel grid (one point per occupied voxel: the voxel mean), then sampled
uniformly over voxels to a fixed budget (without replacement when enough voxels exist, otherwise every voxel once and
then cycled). Voxel deduplication equalises the point density per surface area, so near surfaces, which cover many
pixels per square metre, do not outnumber distant ones.

    L = 0.5 * mean_g min_p ||g - p||^2 + 0.5 * mean_p min_g ||p - g||^2

over all Gaussian centres g of the sample at that timestep (no opacity or contribution filter), in metric
coordinates, without alignment. Nearest neighbours are found without gradient (a fused Triton kernel on CUDA, an exact
chunked fallback elsewhere; bounded memory, no dense distance matrix kept), then the squared distances to the selected
neighbours are recomputed with gradient: the gradient of a minimum is the gradient at its argmin.
"""

from __future__ import annotations

import warnings

import torch

from s4d.geometry import lift_depth

_FALLBACK_ELEMENTS = 16 * 2**20  # distance-matrix entries per chunk of the exact fallback
_KEY_BITS = 16  # voxel coordinates per axis are stored in 16 bits (+-32768 voxels)

try:  # the fused kernel needs Triton (bundled with CUDA builds of torch); the fallback is exact
    import triton
    import triton.language as tl

    @triton.jit
    def _nearest_kernel(src, dst, out_d, out_i, N, M, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
        """For every source point of batch b, the squared distance to and index of its nearest destination point.
        Exact fp32 squared differences; ties resolve to the lowest index."""
        pid = tl.program_id(0)
        b = tl.program_id(1)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        base_s = src + b * N * 3 + offs_n * 3
        sx = tl.load(base_s + 0, mask=mask_n, other=0.0)
        sy = tl.load(base_s + 1, mask=mask_n, other=0.0)
        sz = tl.load(base_s + 2, mask=mask_n, other=0.0)
        best_d = tl.full((BLOCK_N,), float("inf"), tl.float32)
        best_i = tl.zeros((BLOCK_N,), tl.int32)
        for m0 in range(0, M, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < M
            base_d = dst + b * M * 3 + offs_m * 3
            dx = tl.load(base_d + 0, mask=mask_m, other=0.0)
            dy = tl.load(base_d + 1, mask=mask_m, other=0.0)
            dz = tl.load(base_d + 2, mask=mask_m, other=0.0)
            ex = sx[:, None] - dx[None, :]
            ey = sy[:, None] - dy[None, :]
            ez = sz[:, None] - dz[None, :]
            d = ex * ex + ey * ey + ez * ez
            d = tl.where(mask_m[None, :], d, float("inf"))
            dmin = tl.min(d, axis=1)
            imin = tl.argmin(d, axis=1) + m0
            upd = dmin < best_d
            best_d = tl.where(upd, dmin, best_d)
            best_i = tl.where(upd, imin, best_i)
        tl.store(out_d + b * N + offs_n, best_d, mask=mask_n)
        tl.store(out_i + b * N + offs_n, best_i, mask=mask_n)

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - CPU-only installs
    HAVE_TRITON = False

_FUSED_OK: dict[str, bool] = {}


def fused_available(device) -> bool:
    """Whether the Triton kernel builds and launches on ``device`` (probed once per device). Triton compiles a small CUDA
    helper with the system C compiler at its first launch; an image without one (the remote campaign image, 2026-10-23)
    uses the exact chunked fallback, which returns the same nearest neighbours (ties to the lowest index in both)."""
    key = str(device)
    if key not in _FUSED_OK:
        ok = HAVE_TRITON and torch.device(device).type == "cuda"
        if ok:
            try:
                probe = torch.zeros(1, 2, 3, device=device)
                d = torch.empty(1, 2, device=device)
                i = torch.empty(1, 2, device=device, dtype=torch.int32)
                _nearest_kernel[(1, 1)](probe, probe, d, i, 2, 2, BLOCK_N=64, BLOCK_M=128)
            except RuntimeError as exc:
                warnings.warn(f"Triton nearest-neighbour kernel unavailable ({exc}); using the exact chunked fallback")
                ok = False
        _FUSED_OK[key] = ok
    return _FUSED_OK[key]


@torch.no_grad()
def nearest_neighbours(src: torch.Tensor, dst: torch.Tensor, fused: bool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """src (B,N,3), dst (B,M,3) -> squared distance (B,N) to and index (B,N) of each source point's nearest destination
    point (M >= 1). ``fused`` None: the Triton kernel on CUDA when it builds there (``fused_available``), else the exact
    chunked fallback."""
    src, dst = src.detach().float().contiguous(), dst.detach().float().contiguous()
    B, N, _ = src.shape
    M = dst.shape[1]
    if M == 0:
        raise ValueError("nearest_neighbours needs a non-empty destination cloud")
    if fused is None:
        fused = src.is_cuda and fused_available(src.device)
    if fused:
        d = torch.empty(B, N, device=src.device, dtype=torch.float32)
        i = torch.empty(B, N, device=src.device, dtype=torch.int32)
        if N > 0:
            _nearest_kernel[(triton.cdiv(N, 64), B)](src, dst, d, i, N, M, BLOCK_N=64, BLOCK_M=128)
        return d, i.long()
    # tiles over batch, query and destination, each holding at most _FALLBACK_ELEMENTS pairs (x3 coordinates of
    # temporaries); running minima keep global indices, ties resolving to the lowest index as in the kernel
    best_d = torch.full((B, N), float("inf"), device=src.device)
    best_i = torch.zeros(B, N, dtype=torch.long, device=src.device)
    tile = max(1, int(_FALLBACK_ELEMENTS ** 0.5))
    for b0 in range(B):
        for n0 in range(0, N, tile):
            for m0 in range(0, M, tile):
                diff = src[b0, n0 : n0 + tile, None, :] - dst[b0, None, m0 : m0 + tile, :]
                d2 = (diff * diff).sum(-1)
                v, k = d2.min(-1)
                better = v < best_d[b0, n0 : n0 + tile]
                best_d[b0, n0 : n0 + tile] = torch.where(better, v, best_d[b0, n0 : n0 + tile])
                best_i[b0, n0 : n0 + tile] = torch.where(better, k + m0, best_i[b0, n0 : n0 + tile])
    return best_d, best_i


def target_clouds(
    depth: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    far: float,
    valid_far: bool,
    voxel: float,
    budget: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Training-camera target clouds (module docstring).

    depth (B,T,V,H,W) metres, K (B,V,3,3), c2w (B,V,4,4) -> points (B,T,budget,3), valid (B,T) (False when no pixel
    passes the validity rule) and statistics (pixel fractions excluded by each rule, voxels per cloud, and the depth
    composition of raw valid pixels vs sampled points in bands < 0.9 m, 0.9-1.3 m, >= 1.3 m).
    """
    B, T, V, H, W = depth.shape
    device = depth.device
    xyz = lift_depth(depth.float(), K[:, None].expand(B, T, V, 3, 3).float(), c2w[:, None].expand(B, T, V, 4, 4).float())
    zero = ~(depth > 0)  # also NaN: only finite positive depth is valid, as in the reference rule
    beyond = (depth >= far) if valid_far else torch.zeros_like(zero)
    keep = (depth > 0) & ((depth < far) if valid_far else torch.isfinite(depth))
    BT = B * T
    group = torch.arange(BT, device=device)[:, None].expand(BT, V * H * W)[keep.reshape(BT, -1)]
    pts = xyz.reshape(BT, V * H * W, 3)[keep.reshape(BT, -1)]
    pix_depth = depth.reshape(BT, -1)[keep.reshape(BT, -1)].float()
    stats = {
        "chamfer_excluded_no_depth": zero.float().mean(),
        "chamfer_excluded_beyond_far": beyond.float().mean(),
    }
    bands = torch.tensor([0.9, 1.3], device=device)
    if len(pts) == 0:
        stats["chamfer_voxels_per_cloud"] = depth.new_zeros(())
        return depth.new_zeros(B, T, budget, 3), torch.zeros(B, T, dtype=torch.bool, device=device), stats
    cells = torch.floor(pts / voxel).long() + (1 << (_KEY_BITS - 1))
    if (cells.min() < 0) or (cells.max() >= (1 << _KEY_BITS)):
        raise ValueError("chamfer target points fall outside the voxel key range")
    key = (((group << _KEY_BITS) | cells[:, 0]) << _KEY_BITS | cells[:, 1]) << _KEY_BITS | cells[:, 2]
    uniq, inverse = torch.unique(key, return_inverse=True)
    n_vox = len(uniq)
    count = torch.zeros(n_vox, device=device).index_add_(0, inverse, torch.ones_like(pix_depth))
    centroid = torch.zeros(n_vox, 3, device=device).index_add_(0, inverse, pts) / count[:, None]
    vox_depth = torch.zeros(n_vox, device=device).index_add_(0, inverse, pix_depth) / count
    vox_group = uniq >> (3 * _KEY_BITS)
    per_group = torch.bincount(vox_group, minlength=BT)
    # uniform random order of the voxels inside each cloud; slot j takes the j-th voxel (cycled if fewer voxels)
    prio = torch.rand(n_vox, device=device, generator=generator)
    order = torch.argsort(vox_group.double() + prio.double())
    starts = torch.cumsum(per_group, 0) - per_group
    slots = torch.arange(budget, device=device)
    valid = per_group > 0
    take = starts[:, None] + slots[None, :] % per_group.clamp_min(1)[:, None]
    take = torch.where(valid[:, None], take, torch.zeros_like(take))
    chosen = order[take]
    points = centroid[chosen].view(B, T, budget, 3)
    sampled_depth = vox_depth[chosen][valid]
    stats.update({
        "chamfer_voxels_per_cloud": per_group.float().mean(),
        "chamfer_raw_near_fraction": (pix_depth < bands[0]).float().mean(),
        "chamfer_raw_far_fraction": (pix_depth >= bands[1]).float().mean(),
        "chamfer_sampled_near_fraction": (sampled_depth < bands[0]).float().mean(),
        "chamfer_sampled_far_fraction": (sampled_depth >= bands[1]).float().mean(),
    })
    return points, valid.view(B, T), stats


def chamfer_loss(centres: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    """centres (B,S,N,3) Gaussian centres at each reconstructed timestep, targets (B,S,M,3), valid (B,S) ->
    per-timestep (S,) means over the valid samples of the loss and of both directions:
    ``pred_to_target`` = mean_g min_p ||g - p||^2 and ``target_to_pred`` = mean_p min_g ||p - g||^2 (m^2)."""
    B, S, N, _ = centres.shape
    M = targets.shape[2]
    g = centres.float().reshape(B * S, N, 3)
    p = targets.float().reshape(B * S, M, 3).detach()
    _, i_gp = nearest_neighbours(g, p)
    _, i_pg = nearest_neighbours(p, g)
    pred_to_target = ((g - torch.gather(p, 1, i_gp[..., None].expand(-1, -1, 3))) ** 2).sum(-1).mean(-1)
    target_to_pred = ((p - torch.gather(g, 1, i_pg[..., None].expand(-1, -1, 3))) ** 2).sum(-1).mean(-1)
    w = valid.reshape(B * S).float()
    out = {}
    for name, v in (("pred_to_target", pred_to_target), ("target_to_pred", target_to_pred),
                    ("loss", 0.5 * pred_to_target + 0.5 * target_to_pred)):
        per = (v * w).view(B, S).sum(0) / w.view(B, S).sum(0).clamp_min(1.0)
        out[name] = per
    return out
