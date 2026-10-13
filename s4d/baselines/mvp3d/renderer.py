"""3D-MVP renderer R (Qi et al., CVPR 2025, §3.2 Eq. 2): posed RGB-D images -> point cloud -> RVT-1's five
orthographic virtual RGB-D views (Goyal et al., CoRL 2023, §3).

Pure-PyTorch replacement for RVT-1's renderer (``rvt/mvt/renderer.py`` ``BoxRenderer``: PyTorch3D point rasteriser
+ ``NormWeightedCompositor``), which cannot be built here (no PyTorch3D, no CUDA point-renderer extension).
Conventions are RVT-1's:

* **Cameras** (``get_cube_R_T``): orthographic cameras at distance 1 from the centre of the cube [-1, 1]^3, looking at
  it, in the order top, front, back, left, right. Their PyTorch3D world->view transforms are hard-coded here
  (``VIEW_AXES`` rows are the view x/y/z axes in cube coordinates, ``VIEW_CENTRES`` the camera centres;
  view = axes @ (p - centre)). Tests check them against a transcription of PyTorch3D's ``look_at_view_transform``
  evaluated with the arguments of RVT's own ``get_cube_R_T``.
* **Image plane** (``FoVOrthographicCameras``, unit scale, PyTorch3D's pixel grid with +x_ndc to the left and +y_ndc
  up): pixel (row r, column c) of an S x S image is centred at view (x, y) = (1 - (2c+1)/S, 1 - (2r+1)/S), so a
  point at view (x, y) lies in column floor(S (1 - x) / 2) and row floor(S (1 - y) / 2).
* **Channels** (``rvt_derived.assemble_virtual_images``): xyz in cube coordinates, RGB in [0, 1], depth = view-space
  z of the front point (in [0, 2]) mean-centred per view with -1 at empty pixels, pixel-location code; empty pixels
  have xyz = RGB = 0.
* **Cloud**: lifted with our OpenCV K / c2w (``s4d.geometry.lift_depth``), cropped to the scene bounds
  (``rvt_derived.bounds_mask`` = RVT's ``move_pc_in_bound``) and placed in the cube by those bounds (RVT's own
  ``place_pc_in_cube(..., with_mean_or_bounds=False)``, as RVT's ``place_with_mean: False``).

**Rasterisation differs from RVT-1** (docs/baselines_mvp3d.md): PyTorch3D draws each point as a disc of radius
0.012 (NDC) and composites the 5 nearest points per pixel with weights 1 - d^2/r^2. Here, as in RVT-2's point
renderer, each point is binned to the pixel that contains it, a z-buffer keeps the nearest one
(``scatter_reduce`` amin on a packed (quantised depth << 32 | point index) key), and a depth-tested screen-space
splat with the same 0.012 NDC radius (radius_px = 0.012 * S / 2) lets each pixel take the nearest covering
neighbour. At S = 220 the splat covers the 4-neighbourhood; at S = 48 (the size-matched configuration) it covers only
the pixel itself, so the render is a pure nearest-point binning.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from s4d.baselines.mvp3d import rvt
from s4d.baselines.mvp3d.rvt_derived import assemble_virtual_images, bounds_mask, normalize_depth
from s4d.geometry import lift_depth

VIEWS = ("top", "front", "back", "left", "right")
# Rows: view x, y, z axes in cube coordinates (PyTorch3D R^T for RVT-1's get_cube_R_T cameras).
VIEW_AXES = torch.tensor(
    [
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],  # top:   camera at +z, image up = +y
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],  # front: camera at +y, image up = +z
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],  # back:  camera at -y, image up = +z
        [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],  # left:  camera at +x, image up = +z
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],  # right: camera at -x, image up = +z
    ]
)
VIEW_CENTRES = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
RADIUS = 0.012  # RVT's BoxRenderer / RVTBoxRenderer radius, NDC units
_DEPTH_KEY_SCALE = 2.0**24  # view depth in [0, 2] -> 25-bit integer, packed above a 32-bit point index
_INDEX_MASK = 0xFFFFFFFF
_EMPTY = torch.iinfo(torch.int64).max


def to_view(points: torch.Tensor) -> torch.Tensor:
    """(N, P, 3) cube coordinates -> (N, 5, P, 3) view coordinates of the five cameras."""
    axes = VIEW_AXES.to(points)
    centres = VIEW_CENTRES.to(points)
    return torch.einsum("vij,nvpj->nvpi", axes, points[:, None] - centres[None, :, None])


def pixel_of(view: torch.Tensor, img_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """View coordinates (..., 3) -> integer (row, col) of the pixel containing the point (may be off-image)."""
    col = torch.floor(img_size * (1.0 - view[..., 0]) / 2.0)
    row = torch.floor(img_size * (1.0 - view[..., 1]) / 2.0)
    return row, col


def _splat(feat: torch.Tensor, z: torch.Tensor, radius_px: float) -> tuple[torch.Tensor, torch.Tensor]:
    """RVT-2-style depth-tested screen-space splat (orthographic: equal splat size for all points). Every pixel takes
    the nearest point among itself and the neighbours whose splat (radius ``radius_px``) covers it.
    feat (M, F, S, S), z (M, S, S) with +inf at empty pixels."""
    k = min(2 * math.ceil(radius_px) + 1, 51)  # RVT-2 PointRenderer.splat_filter kernel size
    h = k // 2
    offsets = [(0, 0)] + [
        (dy, dx)
        for dy in range(-h, h + 1)
        for dx in range(-h, h + 1)
        if (dy, dx) != (0, 0) and math.sqrt(dx * dx + dy * dy) < radius_px
    ]
    if len(offsets) == 1:
        return feat, z
    S = z.shape[-1]
    zp = F.pad(z, (h, h, h, h), value=math.inf)
    fp = F.pad(feat, (h, h, h, h))
    cand_z = torch.stack([zp[:, h + dy : h + dy + S, h + dx : h + dx + S] for dy, dx in offsets])  # (K, M, S, S)
    best = cand_z.argmin(0)  # ties keep the earliest candidate, i.e. the pixel itself
    cand_f = torch.stack([fp[:, :, h + dy : h + dy + S, h + dx : h + dx + S] for dy, dx in offsets])  # (K,M,F,S,S)
    z = cand_z.gather(0, best[None])[0]
    feat = cand_f.gather(0, best[None, :, None].expand(1, *feat.shape))[0]
    return feat, z


def render_points(
    points: torch.Tensor, features: torch.Tensor, valid: torch.Tensor, img_size: int, radius: float = RADIUS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-buffered render of N clouds into the five cube views.

    points (N, P, 3) cube coordinates, features (N, P, F), valid (N, P) bool ->
    feature images (N, 5, F, S, S) (0 at empty pixels), view-space depth (N, 5, S, S) (+inf at empty pixels) and
    covered (N, 5, S, S) bool."""
    N, P, _ = points.shape
    V, S, Fd = len(VIEWS), int(img_size), features.shape[-1]
    if P >= 2**32:
        raise ValueError("too many points for the packed z-buffer key")
    device = points.device
    points = torch.where(valid[..., None], points, torch.zeros_like(points))
    view = to_view(points)  # (N, V, P, 3)
    row, col = pixel_of(view, S)
    z = view[..., 2]
    keep = valid[:, None, :] & (z >= 0) & (col >= 0) & (col < S) & (row >= 0) & (row < S)  # PyTorch3D culls z < 0
    image = torch.arange(N * V, device=device).view(N, V, 1)
    flat = (image * S + row.long().clamp(0, S - 1)) * S + col.long().clamp(0, S - 1)
    key = ((z.clamp(0.0, 2.0) * _DEPTH_KEY_SCALE).long() << 32) | torch.arange(P, device=device).view(1, 1, P)
    zbuf = torch.full((N * V * S * S,), _EMPTY, dtype=torch.int64, device=device)
    zbuf.scatter_reduce_(0, flat[keep], key[keep], reduce="amin")
    covered = zbuf != _EMPTY
    pix = covered.nonzero().squeeze(1)
    winner = zbuf[pix] & _INDEX_MASK  # point index of the front point
    img_index = pix // (S * S)  # n * V + v
    feat_img = torch.zeros(N * V * S * S, Fd, dtype=features.dtype, device=device)
    feat_img[pix] = features.reshape(N * P, Fd)[(img_index // V) * P + winner]
    depth = torch.full((N * V * S * S,), math.inf, dtype=points.dtype, device=device)
    depth[pix] = z.reshape(N * V * P)[img_index * P + winner]
    feat_img = feat_img.view(N * V, S, S, Fd).permute(0, 3, 1, 2)
    depth = depth.view(N * V, S, S)
    feat_img, depth = _splat(feat_img, depth, radius * S / 2.0)
    covered = torch.isfinite(depth)
    return feat_img.reshape(N, V, Fd, S, S).contiguous(), depth.view(N, V, S, S), covered.view(N, V, S, S)


def virtual_images(
    points: torch.Tensor, colors: torch.Tensor, valid: torch.Tensor, img_size: int, radius: float = RADIUS
) -> torch.Tensor:
    """Cube-placed clouds (N, P, 3), RGB in [0, 1] (N, P, 3), valid (N, P) -> RVT-1 virtual images (N, 5, 10, S, S)."""
    feats = torch.cat((points, colors.to(points.dtype)), dim=-1)  # MVT.render: cat((pc, img_feat))
    feat_img, depth, covered = render_points(points, feats, valid, img_size, radius)
    depth = normalize_depth(torch.where(covered, depth, torch.full_like(depth, -1.0)))
    return assemble_virtual_images(feat_img, depth)


def rgbd_clouds(
    rgb: torch.Tensor, depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor, view_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Posed RGB-D images -> one world-frame cloud per sample (the union of its views).

    rgb (N, V, 3, H, W) in [0, 1], depth (N, V, H, W) metres (0 = invalid), K (N, V, 3, 3), c2w (N, V, 4, 4) OpenCV,
    view_mask (N, V) bool (views that belong to the cloud) -> points (N, V*H*W, 3), colors (N, V*H*W, 3), valid."""
    N, V, H, W = depth.shape
    xyz = lift_depth(depth.float(), K.float(), c2w.float())  # (N, V, H, W, 3)
    valid = depth > 0
    if view_mask is not None:
        valid = valid & view_mask[:, :, None, None]
    colors = rgb.float().permute(0, 1, 3, 4, 2)
    return xyz.reshape(N, V * H * W, 3), colors.reshape(N, V * H * W, 3), valid.reshape(N, V * H * W)


def place_in_cube(points: torch.Tensor, bounds) -> torch.Tensor:
    """World points (..., 3) -> cube coordinates by the scene bounds: RVT's ``place_pc_in_cube(pc,
    with_mean_or_bounds=False, scene_bounds=bounds)`` (imported from the pinned RVT commit; it broadcasts over
    leading axes)."""
    return rvt.mvt_utils().place_pc_in_cube(points, with_mean_or_bounds=False, scene_bounds=list(bounds))[0]


def render_rgbd(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    bounds,
    img_size: int,
    radius: float = RADIUS,
    view_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """R (Eq. 2): posed RGB-D images (see ``rgbd_clouds``) -> virtual images (N, 5, 10, S, S)."""
    points, colors, valid = rgbd_clouds(rgb, depth, K, c2w, view_mask)
    valid = valid & bounds_mask(points, bounds)
    points = torch.where(valid[..., None], points, torch.zeros_like(points))
    return virtual_images(place_in_cube(points, bounds), colors, valid, img_size, radius)
