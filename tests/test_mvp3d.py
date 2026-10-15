"""3D-MVP port: fidelity to RVT (layer classes, MVTSingle trunk and read-out, cube cameras, depth normalisation, crop
and cube placement), renderer conventions, masking, Eq. 6, data adapter, export, RL backbone, trainer resume and the
size budget."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from s4d.baselines.mvp3d import rvt
from s4d.baselines.mvp3d.data import EpochBatches, MVPTimeSteps, batch_to_device, fixed_items
from s4d.baselines.mvp3d.model import (
    MVP3D,
    SCENE_BOUNDS,
    MVP3DConfig,
    build_encoder,
    load_export,
    mask_rgb,
    random_view_mask,
    reconstruction_loss,
    save_export,
)
from s4d.baselines.mvp3d.renderer import (
    VIEW_AXES,
    VIEW_CENTRES,
    VIEWS,
    place_in_cube,
    render_points,
    render_rgbd,
    virtual_images,
)
from s4d.baselines.mvp3d.rvt_derived import bounds_mask, normalize_depth, pixel_loc
from s4d.data.metaworld.cameras import camera_rig
from s4d.geometry import lift_depth, project
from s4d.third_party import source_tree

REPO = Path(__file__).resolve().parents[1]
SIZE = 32
TINY = MVP3DConfig(img_size=24, img_patch_size=6, im_channels=16, attn_dim=32, attn_heads=2, attn_dim_head=16, depth=4,
                   decoder_dim=32, decoder_heads=2, decoder_dim_head=16)
RENDER = {"bounds": list(SCENE_BOUNDS), "radius": 0.012}
TASKS = ("hammer", "pick-place", "bin-picking", "coffee-push", "door-open", "peg-insert-side", "peg-unplug-side",
         "stick-push")


# ------------------------------------------------------------------------------------------------ official sources
def _rvt_node(relative: str, qualname: str):
    node = ast.parse((source_tree("rvt") / relative).read_text())
    for part in qualname.split("."):
        node = next(n for n in ast.iter_child_nodes(node) if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                    and n.name == part)
    return node


def official_function(relative: str, qualname: str, namespace: dict):
    """An RVT function's own source (pinned tree), executed without importing its module (whose other imports, e.g.
    PyTorch3D or PerAct, are unavailable)."""
    node = _rvt_node(relative, qualname)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), namespace)
    return namespace[node.name]


def official_depth_block(zbuf: torch.Tensor) -> torch.Tensor:
    """The ``if with_depth:`` block of RVT's PointsRendererWithDepth.forward, run on a PyTorch3D-style zbuf."""
    node = _rvt_node("rvt/mvt/renderer.py", "PointsRendererWithDepth.forward")
    block = next(n for n in node.body if isinstance(n, ast.If) and getattr(n.test, "id", None) == "with_depth")
    namespace = {"torch": torch, "fragments": type("Fragments", (), {"zbuf": zbuf})}
    exec(compile(ast.Module(body=block.body, type_ignores=[]), "renderer.py", "exec"), namespace)
    return namespace["depth"]


def pytorch3d_look_at_view_transform(dist, elev, azim, up, at=((0, 0, 0),)):
    """Transcription of PyTorch3D's look_at_view_transform (BSD-3-Clause; pytorch3d/renderer/cameras.py with
    camera_position_from_spherical_angles and look_at_rotation, convert_to_tensors_and_broadcast/format_tensor),
    float32 as there."""

    def fmt(x):
        t = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
        return t.view(1) if t.dim() == 0 else t

    def bcast(*args):
        args = [fmt(a) for a in args]
        n = max(a.shape[0] for a in args)
        return [a.expand(n, *a.shape[1:]) for a in args]

    dist, elev, azim, at, up = bcast(dist, elev, azim, at, up)
    d, e, a = bcast(dist, elev, azim)
    e, a = math.pi / 180.0 * e, math.pi / 180.0 * a
    C = torch.stack([d * torch.cos(e) * torch.sin(a), d * torch.sin(e), d * torch.cos(e) * torch.cos(a)], 1).view(-1, 3)
    C, at, up = bcast(C + at, at, up)
    z = F.normalize(at - C, eps=1e-5)
    x = F.normalize(torch.cross(up, z, dim=1), eps=1e-5)
    y = F.normalize(torch.cross(z, x, dim=1), eps=1e-5)
    close = torch.isclose(x, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if close.any():
        x = torch.where(close, F.normalize(torch.cross(y, z, dim=1), eps=1e-5), x)
    R = torch.cat((x[:, None, :], y[:, None, :], z[:, None, :]), dim=1).transpose(1, 2)
    T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]
    return R, T


# --------------------------------------------------------------------------------------------- synthetic data
def table_scene(rig: dict, size: int, box_x: float = 0.0):
    """Analytic RGB-D of every rig camera: the plane z = 0 (colour from world x, y) and a raised square (top at
    z = 0.1, |x - box_x| < 0.1, |y - 0.6| < 0.1). Returns depth (V, H, W) metres and rgb (V, H, W, 3) uint8."""
    K, c2w = torch.from_numpy(rig["K"]).double(), torch.from_numpy(rig["c2w"]).double()
    v, u = torch.meshgrid(torch.arange(size) + 0.5, torch.arange(size) + 0.5, indexing="ij")
    rays = torch.stack([(u - K[:, 0, 2, None, None]) / K[:, 0, 0, None, None],
                        (v - K[:, 1, 2, None, None]) / K[:, 1, 1, None, None], torch.ones(len(K), size, size)], -1)
    d = torch.einsum("vij,vhwj->vhwi", c2w[:, :3, :3], rays)
    o = c2w[:, None, None, :3, 3]
    t_box = (0.1 - o[..., 2]) / d[..., 2]
    p_box = o + t_box[..., None] * d
    in_box = (t_box > 0) & ((p_box[..., 0] - box_x).abs() < 0.1) & ((p_box[..., 1] - 0.6).abs() < 0.1)
    t = torch.where(in_box, t_box, -o[..., 2] / d[..., 2])
    p = o + t[..., None] * d
    color = torch.stack([(p[..., 0] + 0.6) / 1.2, p[..., 1] - 0.15, torch.where(in_box, 1.0, 0.2)], -1)
    return t.float(), (color.clamp(0, 1) * 255).round().to(torch.uint8)


def write_dataset(tmp_path: Path, episodes: int = 3, length: int = 6) -> tuple[Path, Path]:
    rig = camera_rig(SIZE, SIZE)
    path = tmp_path / "task.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["depth_unit_m"] = 1e-4
        cams = f.create_group("cameras")
        for key in ("K", "c2w", "w2c", "is_train"):
            cams.create_dataset(key, data=rig[key])
        group = f.create_group("episodes")
        for e in range(episodes):
            ep = group.create_group(f"ep{e:03d}")
            ep.attrs["length"] = length
            scenes = [table_scene(rig, SIZE, box_x=-0.2 + 0.05 * t + 0.01 * e) for t in range(length)]
            ep.create_dataset("rgb", data=np.stack([s[1].numpy() for s in scenes]))
            ep.create_dataset("depth", data=np.stack([np.round(s[0].numpy() / 1e-4).astype(np.uint16) for s in scenes]))
    manifest = tmp_path / "split.json"
    names = [f"ep{e:03d}" for e in range(episodes)]
    manifest.write_text(json.dumps({"train": names[:-1], "validation": names[-1:]}))
    return path, manifest


# ---------------------------------------------------------------------------------------------- RVT imports
def test_patched_rvt_tree_only_drops_the_unused_matplotlib_import():
    tree, checkout = source_tree("rvt"), REPO / "third_party/rvt"
    assert tree != checkout and (tree / ".s4d-patched.json").is_file()
    original = (checkout / "rvt/mvt/utils.py").read_text()
    assert "import matplotlib.pyplot as plt\n" in original and "plt." not in original
    assert (tree / "rvt/mvt/utils.py").read_text() == original.replace("import matplotlib.pyplot as plt\n", "")
    for relative in ("rvt/mvt/attn.py", "rvt/mvt/mvt_single.py", "rvt/mvt/renderer.py", "rvt/utils/rvt_utils.py"):
        assert (tree / relative).read_bytes() == (checkout / relative).read_bytes(), relative
    assert subprocess.run(["git", "-C", str(checkout), "status", "--porcelain"], capture_output=True,
                          text=True).stdout == ""


def test_encoder_and_decoder_are_built_from_rvt_layer_classes():
    attn = rvt.attn()
    assert Path(inspect.getsourcefile(attn.PreNorm)) == source_tree("rvt") / "rvt/mvt/attn.py"
    model = MVP3D(TINY)
    enc = model.encoder
    assert type(enc.input_preprocess) is attn.Conv2DBlock and type(enc.patchify) is attn.Conv2DBlock
    assert type(enc.fc_bef_attn) is attn.DenseBlock
    for layers, depth in ((enc.layers, TINY.depth), (model.decoder.layers, TINY.decoder_depth)):
        assert len(layers) == depth
        for self_attn, self_ff in layers:
            assert type(self_attn) is attn.PreNorm and type(self_attn.fn) is attn.Attention
            assert type(self_ff) is attn.PreNorm and type(self_ff.fn) is attn.FeedForward
            assert type(self_ff.fn.net[1]) is attn.GEGLU and self_attn.fn.heads == TINY.attn_heads
    assert enc.patchify.conv2d.kernel_size == (6, 6) and enc.patchify.conv2d.stride == (6, 6)
    assert type(enc.patchify.norm) is torch.nn.GroupNorm and enc.input_preprocess.conv2d.in_channels == 10


def test_encoder_matches_rvt_mvt_single_trunk_readout_and_pixel_loc(tmp_path):
    """Our encoder's state dict loads into RVT's own MVTSingle (same names); on the same input its attention-trunk
    output equals our z, and RVT's per-view max-pool equals our global feature."""
    torch.manual_seed(0)
    chosen, square = MVP3DConfig(), MVP3DConfig(**{**asdict(TINY), "im_channels": TINY.attn_dim})
    cases, ours = [], []
    for cfg, identity_head in ((chosen, False), (square, True)):
        enc = build_encoder(cfg).eval()
        img = torch.randn(2, 5, 10, cfg.img_size, cfg.img_size)
        with torch.no_grad():
            z = enc(img)
        ours.append((enc, z, enc.global_feature(z)))
        cases.append({"cfg": asdict(cfg), "state_dict": enc.state_dict(), "img": img, "identity_head": identity_head})
    torch.save(cases, tmp_path / "cases.pt")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]}
    result = subprocess.run([sys.executable, "-I", str(REPO / "tests/_mvp3d_rvt_worker.py"), str(tmp_path / "cases.pt"),
                             str(tmp_path / "out.pt")], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    out = torch.load(tmp_path / "out.pt", weights_only=False)
    head = ("fc_aft_attn.", "up0.", "final.", "trans_decoder.", "feat_fc.")
    for (_, z, feat), res, (cfg, identity_head) in zip(ours, out, ((chosen, False), (square, True))):
        assert res["unexpected"] == [] and res["missing"] and all(k.startswith(head) for k in res["missing"])
        assert Path(res["attn_file"]) == source_tree("rvt") / "rvt/mvt/attn.py"
        assert torch.equal(res["trunk"], z)
        assert torch.equal(res["pixel_loc"], pixel_loc(5, cfg.img_size))
        if identity_head:
            assert torch.equal(res["feat_fc_in"][:, : feat.shape[1]], feat)


# ------------------------------------------------------------------------------------------- renderer
def test_cube_cameras_are_rvt_get_cube_R_T_through_pytorch3d():
    namespace = {"torch": torch, "look_at_view_transform": pytorch3d_look_at_view_transform}
    R, T = official_function("rvt/mvt/renderer.py", "get_cube_R_T", namespace)()
    names = [k.value for k in next(n for n in ast.walk(_rvt_node("rvt/mvt/renderer.py", "get_cube_R_T"))
                                   if isinstance(n, ast.Dict)).keys]
    assert tuple(names) == VIEWS
    torch.testing.assert_close(R.transpose(1, 2), VIEW_AXES, atol=1e-6, rtol=0)  # PyTorch3D: X_view = X R + T
    torch.testing.assert_close(T, -torch.einsum("vij,vj->vi", VIEW_AXES, VIEW_CENTRES), atol=1e-6, rtol=0)
    for axes in VIEW_AXES:  # proper rotations
        torch.testing.assert_close(axes @ axes.T, torch.eye(3))
        assert torch.isclose(torch.det(axes), torch.tensor(1.0))


def test_known_point_lands_on_the_expected_pixel_of_every_view():
    S = 48
    p = torch.tensor([[[0.5, -0.25, 0.1]]])
    rgb = torch.tensor([[[0.2, 0.4, 0.6]]])
    # PyTorch3D pixel grid: column floor(S (1 - x_view) / 2), row floor(S (1 - y_view) / 2); depth = view z
    expected = {"top": (30, 36, 0.9), "front": (21, 12, 1.25), "back": (21, 36, 0.75), "left": (21, 18, 0.5),
                "right": (21, 30, 1.5)}
    feat, z, covered = render_points(p, torch.cat([p, rgb], -1), torch.ones(1, 1, dtype=torch.bool), S)
    img = virtual_images(p, rgb, torch.ones(1, 1, dtype=torch.bool), S)[0]
    for v, name in enumerate(VIEWS):
        row, col, depth = expected[name]
        assert covered[0, v].sum() == 1 and covered[0, v, row, col], name
        assert math.isclose(float(z[0, v, row, col]), depth, abs_tol=1e-6)
        torch.testing.assert_close(img[v, :3, row, col], p[0, 0])
        torch.testing.assert_close(img[v, 3:6, row, col], rgb[0, 0])
        assert abs(float(img[v, 6, row, col])) < 1e-3, "one point: depth minus its own mean (RVT's float sums)"
        empty = ~covered[0, v]
        assert (img[v, 6][empty] == -1).all() and (img[v, :6][:, empty] == 0).all()
        torch.testing.assert_close(img[v, 7:], pixel_loc(5, S)[v])


def test_zbuffer_keeps_the_nearest_point_and_breaks_ties_by_index():
    S = 48
    points = torch.tensor([[[0.0, 0.0, 0.5], [0.0, 0.0, -0.5], [0.3, 0.5, 0.2], [0.3, -0.5, 0.2], [0.3, 0.5, 0.2]]])
    colors = torch.eye(5, 3)[None]  # points 0-2 red/green/blue, 3-4 black
    colors[0, 3] = torch.tensor([0.5, 0.5, 0.5])
    colors[0, 4] = torch.tensor([0.9, 0.9, 0.9])  # duplicate of point 2: loses the tie (higher index)
    img = virtual_images(points, colors, torch.ones(1, 5, dtype=torch.bool), S)[0]
    feat, z, covered = render_points(points, torch.cat([points, colors], -1), torch.ones(1, 5, dtype=torch.bool), S)
    top, front, back = VIEWS.index("top"), VIEWS.index("front"), VIEWS.index("back")
    r, c = S // 2, S // 2  # (0, 0, z) is on the boundary pixel (row S/2, col S/2) of every view's centre
    torch.testing.assert_close(img[top, 3:6, r, c], colors[0, 0])  # higher z is nearer the top camera
    assert math.isclose(float(z[0, top, r, c]), 0.5, abs_tol=1e-6)
    row, col = int(S * (1 - 0.2) / 2), int(S * (1 - 0.3) / 2)
    torch.testing.assert_close(img[front, 3:6, row, col], colors[0, 2])  # front camera at +y: y = 0.5 is nearer
    assert math.isclose(float(z[0, front, row, col]), 0.5, abs_tol=1e-6)
    torch.testing.assert_close(img[back, 3:6, row, int(S * (1 + 0.3) / 2)], colors[0, 3])  # back camera at -y


def test_splat_covers_neighbours_within_rvt_radius_and_is_depth_tested():
    for S, neighbours in ((220, 4), (48, 0)):  # radius_px = 0.012 * S / 2 = 1.32 (4-neighbourhood) or 0.29
        i = S // 2 - 3
        x = 1 - (2 * i + 1) / S  # view x / y of the centre of column / row i
        p = torch.tensor([[[-x, x, 0.0]]])  # top view (x_view = -X, y_view = Y): pixel (i, i)
        feat, z, covered = render_points(p, torch.ones(1, 1, 1), torch.ones(1, 1, dtype=torch.bool), S)
        assert covered[0, 0, i, i] and int(covered[0, 0].sum()) == 1 + neighbours, S
        if neighbours:
            assert covered[0, 0, i - 1, i] and covered[0, 0, i, i + 1] and not covered[0, 0, i + 1, i + 1]
    S = 220
    x = 1 - (2 * 100 + 1) / S
    x2 = 1 - (2 * 101 + 1) / S
    near, far = [-x, x, 0.5], [-x2, x, -0.5]  # top view: same row, adjacent columns 100 / 101
    pts = torch.tensor([[near, far]])
    feat, z, covered = render_points(pts, torch.tensor([[[1.0], [2.0]]]), torch.ones(1, 2, dtype=torch.bool), S)
    assert float(feat[0, 0, 0, 100, 101]) == 1.0, "the nearer neighbour's splat overwrites the far point"
    assert math.isclose(float(z[0, 0, 100, 101]), 0.5, abs_tol=1e-6)
    assert float(feat[0, 0, 0, 100, 102]) == 2.0, "beyond the near splat, the far point's splat remains"


def test_depth_normalisation_is_rvts_block():
    gen = torch.Generator().manual_seed(0)
    depth = 2 * torch.rand(5, 16, 16, generator=gen)
    depth[torch.rand(5, 16, 16, generator=gen) < 0.6] = -1
    depth[3] = -1  # an empty view
    expected = official_depth_block(depth.clone()[..., None])
    assert torch.equal(normalize_depth(depth), expected)
    assert torch.equal(normalize_depth(depth[None])[0], expected)


def test_crop_and_cube_placement_match_rvt():
    gen = torch.Generator().manual_seed(1)
    pc = torch.rand(2, 400, 3, generator=gen) * torch.tensor([1.6, 1.4, 1.2]) - torch.tensor([0.8, -0.05, 0.25])
    pc[0, 5, 1] = float("nan")
    feat = torch.rand(2, 400, 3, generator=gen)
    move_pc_in_bound = official_function("rvt/utils/rvt_utils.py", "move_pc_in_bound", {"torch": torch})
    pcs, feats = move_pc_in_bound(pc, feat, list(SCENE_BOUNDS))
    mask = bounds_mask(pc, SCENE_BOUNDS)
    assert 0 < int(mask.sum()) < mask.numel() and not mask[0, 5]
    place_pc_in_cube = rvt.mvt_utils().place_pc_in_cube
    cube = place_in_cube(pc, SCENE_BOUNDS)
    for i in range(2):
        assert torch.equal(pc[i][mask[i]], pcs[i]) and torch.equal(feat[i][mask[i]], feats[i])
        official = place_pc_in_cube(pcs[i], with_mean_or_bounds=False, scene_bounds=list(SCENE_BOUNDS))[0]
        assert torch.equal(cube[i][mask[i]], official)
    lo, hi = torch.tensor(SCENE_BOUNDS[:3]), torch.tensor(SCENE_BOUNDS[3:])
    corners = place_in_cube(torch.stack([lo, hi]), SCENE_BOUNDS)
    assert torch.allclose(corners[:, 0], torch.tensor([-1.0, 1.0])) and corners.abs().max() <= 1 + 1e-6


def test_render_rgbd_of_the_table_plane():
    """Our K/c2w lift + RVT crop/cube + renderer: the table plane z = 0 renders at the same top-view depth
    everywhere; the side views see it edge-on."""
    rig = camera_rig(SIZE, SIZE)
    depth, rgb = table_scene(rig, SIZE, box_x=10.0)  # box outside: plane only
    K, c2w = torch.from_numpy(rig["K"][:6])[None], torch.from_numpy(rig["c2w"][:6])[None]
    img = render_rgbd(rgb[None, :6].permute(0, 1, 4, 2, 3).float() / 255, depth[None, :6], K, c2w, SCENE_BOUNDS, 48)[0]
    covered = img[0, 6] != -1
    assert covered.float().mean() > 0.3
    torch.testing.assert_close(img[0, 2][covered], torch.full_like(img[0, 2][covered], img[0, 2][covered][0]),
                               atol=2e-3, rtol=0)  # cube z of the plane
    assert img[0, 6][covered].abs().max() < 2e-3  # constant depth, centred
    rows = torch.nonzero((img[1, 6] != -1).any(1)).squeeze(1)
    assert len(rows) <= 2, "front view: the plane is a line"


# --------------------------------------------------------------------------------------------- MAE parts
def test_masking_is_75_percent_per_view_and_only_rgb():
    gen = torch.Generator().manual_seed(0)
    mask = random_view_mask(64, 5, 4, 0.75, gen)
    assert mask.shape == (64, 5, 4, 4) and (mask.flatten(2).sum(-1) == 12).all()
    assert not torch.equal(mask[:, 0], mask[:, 1]) and not torch.equal(mask[0], mask[1])
    freq = random_view_mask(4000, 5, 4, 0.75, gen).float().mean((0, 1))
    assert (freq - 0.75).abs().max() < 0.03, "every patch position is masked with probability 0.75"
    assert (random_view_mask(2, 5, 22, 0.75, gen).flatten(2).sum(-1) == 484 - int(484 * 0.25)).all()
    images = torch.rand(3, 5, 10, 24, 24, generator=gen)
    m = random_view_mask(3, 5, 4, 0.75, gen)
    out = mask_rgb(images, m, 6, fill=0.0)
    pix = m.repeat_interleave(6, -2).repeat_interleave(6, -1)
    assert (out[:, :, 3:6].permute(0, 1, 3, 4, 2)[pix] == 0).all()
    assert torch.equal(out[:, :, 3:6].permute(0, 1, 3, 4, 2)[~pix], images[:, :, 3:6].permute(0, 1, 3, 4, 2)[~pix])
    assert torch.equal(out[:, :, :3], images[:, :, :3]) and torch.equal(out[:, :, 6:], images[:, :, 6:])


def test_loss_is_eq6():
    gen = torch.Generator().manual_seed(0)
    pred, target = torch.randn(2, 5, 4, 6, 6, generator=gen), torch.randn(2, 5, 4, 6, 6, generator=gen)
    per_sample = []
    for b in range(2):
        total = 0.0
        for i in range(5):
            for y in range(6):
                for x in range(6):
                    total += float(((target[b, i, :, y, x] - pred[b, i, :, y, x]) ** 2).sum())
        per_sample.append(total / (5 * 6 * 6))
    assert math.isclose(float(reconstruction_loss(pred, target)), sum(per_sample) / 2, rel_tol=1e-6)


def test_pretraining_step_is_finite_and_updates_encoder_and_decoder():
    torch.manual_seed(0)
    model = MVP3D(TINY).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    images = torch.rand(3, 5, 10, 24, 24, device="cuda")
    mask = random_view_mask(3, 5, 4, 0.75, device="cuda")
    before = copy.deepcopy(model.state_dict())
    out = model(images, mask)
    assert out["pred"].shape == (3, 5, 4, 24, 24) and torch.equal(out["masked"][:, :, 6:], images[:, :, 6:])
    torch.testing.assert_close(out["loss"], out["loss_rgb"] + out["loss_depth"])
    opt.zero_grad()
    out["loss"].backward()
    opt.step()
    assert math.isfinite(float(out["loss"].detach()))
    prefixes = ("encoder.layers.0.", "encoder.layers.3.", "encoder.patchify.", "decoder.layers.1.", "decoder.decoder_pred")
    for prefix in prefixes:
        assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items() if k.startswith(prefix)), prefix


# --------------------------------------------------------------------------------------------- data adapter
def test_data_adapter_fuses_all_six_cameras_with_our_camera_convention(tmp_path):
    path, manifest = write_dataset(tmp_path)
    episodes = json.loads(manifest.read_text())["train"]
    ds = MVPTimeSteps(path, episodes, num_views=6)
    assert len(ds) == 2 * 6
    s = ds[(4, 17)]
    assert s["rgb"].shape == (6, SIZE, SIZE, 3) and s["rgb"].dtype == torch.uint8 and s["depth"].shape == (6, SIZE, SIZE)
    sel = s["view_mask"]
    assert sel.all() and (s["depth"] > 0).all(), "every sample is the fused cloud of all six training cameras"
    for k in range(50):  # no random camera subsets, independent of the sample key
        other = ds[(4, k)]
        assert other["view_mask"].all() and torch.equal(other["rgb"], s["rgb"]) and torch.equal(other["depth"], s["depth"])
    with pytest.raises(ValueError, match="fuses all 6 training cameras"):
        MVPTimeSteps(path, episodes, num_views=3)
    rig = camera_rig(SIZE, SIZE)
    assert torch.allclose(s["K"], torch.from_numpy(rig["K"][:6]))
    assert torch.allclose(s["c2w"], torch.from_numpy(rig["c2w"][:6]))
    v = int(torch.nonzero(sel)[0])
    xyz = lift_depth(s["depth"][v], s["K"][v], s["c2w"][v])
    on_plane = (xyz[..., 2].abs() < 2e-3) | ((xyz[..., 2] - 0.1).abs() < 2e-3)
    assert on_plane.float().mean() > 0.99, "lifted with our OpenCV K/c2w, the pixels land on the scene planes"
    uv, z = project(xyz.reshape(-1, 3), s["K"][v], torch.linalg.inv(s["c2w"][v]))
    centres = torch.stack(torch.meshgrid(torch.arange(SIZE) + 0.5, torch.arange(SIZE) + 0.5, indexing="xy"), -1)
    torch.testing.assert_close(uv, centres.reshape(-1, 2), atol=1e-3, rtol=0)
    batch = batch_to_device(torch.utils.data.default_collate([ds[(i, i)] for i in range(3)]), "cpu")
    assert batch["rgb"].shape == (3, 6, 3, SIZE, SIZE) and 0 <= batch["rgb"].min() and batch["rgb"].max() <= 1


def test_epoch_batches_are_seeded_permutations_and_resume_mid_epoch():
    full = list(EpochBatches(10, 3, seed=1, epoch=2))
    assert len(full) == 3 and len({i for b in full for i, _ in b}) == 9
    assert list(EpochBatches(10, 3, seed=1, epoch=2, start=1)) == full[1:]
    assert full != list(EpochBatches(10, 3, seed=1, epoch=3))
    assert [k for b in full for _, k in b] == list(range(20, 29))
    assert all(k >= 2**40 for _, k in fixed_items(10, 4))


# ----------------------------------------------------------------------------------------- export and RL
def test_export_reload_identity(tmp_path):
    torch.manual_seed(0)
    enc = build_encoder(TINY).eval()
    save_export(tmp_path / "e.pt", enc, asdict(TINY), RENDER, 7)
    payload, loaded = load_export(tmp_path / "e.pt")
    assert payload["step"] == 7 and payload["render_cfg"] == RENDER and payload["frame_spacing"] is None
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    for key, value in enc.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    x = torch.rand(2, 5, 10, 24, 24)
    with torch.no_grad():
        assert torch.equal(enc(x), loaded(x))


def rl_cfg(export: Path, **env) -> dict:
    base = yaml.safe_load((REPO / "configs/rl/base.yaml").read_text())
    return {
        "env": {"frame_stack": 3, "image_size": SIZE, "views": 1, "depth": True, "cameras": True, **env},
        "vision": {"encoder_type": "mvp3d", "export_path": str(export)},
        "agent": {**base["agent"], "feature_dim": 8, "hidden_dim": 16, "dormant_perturb_interval": 4},
    }


def dict_observation(n: int, rng: np.random.Generator, views: int = 1) -> dict[str, np.ndarray]:
    """(n, T=3, V=views, ...) RGB-D + camera observations of the synthetic scene; each episode's ``views`` distinct
    training cameras are drawn at random and fixed over the history (the RL camera protocol)."""
    rig = camera_rig(SIZE, SIZE)
    obs = {"rgb": [], "depth": [], "K": [], "c2w": []}
    for _ in range(n):
        frames = {k: [] for k in obs}
        cams = rng.choice(6, size=views, replace=False)
        for _t in range(3):
            depth, rgb = table_scene(rig, SIZE, box_x=float(rng.uniform(-0.3, 0.3)))
            frames["rgb"].append(rgb[cams].permute(0, 3, 1, 2).numpy())
            frames["depth"].append(depth[cams].numpy())
            frames["K"].append(rig["K"][cams])
            frames["c2w"].append(rig["c2w"][cams])
        for k in obs:
            obs[k].append(np.stack(frames[k]))
    return {k: np.stack(v) for k, v in obs.items()}


@pytest.mark.parametrize("views", [1, 3])
def test_rl_backbone_reads_the_dict_observation_and_matches_the_pipeline(tmp_path, views):
    from s4d.baselines.mvp3d.rl import CONDITIONS
    from s4d.rl.encoders import VisionEncoderAdapter

    torch.manual_seed(0)
    enc = build_encoder(TINY).eval()
    save_export(tmp_path / "e.pt", enc, asdict(TINY), RENDER, 0)
    adapter = VisionEncoderAdapter(rl_cfg(tmp_path / "e.pt", views=views))
    assert adapter.backbone.observation == {"views": views, "depth": True, "cameras": True}
    assert adapter.backbone.label == CONDITIONS[views]
    assert adapter.replay_atom_shape == (5 * TINY.attn_dim,) and adapter.replay_atom_is_feature
    assert not adapter.backbone.training and not any(p.requires_grad for p in adapter.backbone.parameters())
    obs = dict_observation(2, np.random.default_rng(0), views)
    feat = adapter.extract_cacheable_feature({k: torch.as_tensor(v) for k, v in obs.items()})
    assert feat.shape == (2, 3, 5 * TINY.attn_dim) and feat.dtype == torch.float16
    for b, t in ((0, 0), (1, 2)):
        frame = {k: torch.as_tensor(v[b, t][None]).float() for k, v in obs.items()}
        frame["rgb"] = frame["rgb"] / 255
        with torch.no_grad():
            images = render_rgbd(frame["rgb"], frame["depth"], frame["K"], frame["c2w"], SCENE_BOUNDS, TINY.img_size)
            expected = enc.global_feature(enc(images))[0]
        assert (images[0, :, 6] != -1).any(), "the fused cloud of the episode's cameras is rendered"
        torch.testing.assert_close(feat[b, t].float(), expected, atol=2e-2, rtol=2e-3)
    assert adapter(feat, is_feature=True).shape == (2, adapter.repr_dim)
    with pytest.raises(ValueError, match="needs env observation"):
        VisionEncoderAdapter(rl_cfg(tmp_path / "e.pt", views=views, depth=False))
    with pytest.raises(ValueError, match="input conditions use"):
        VisionEncoderAdapter(rl_cfg(tmp_path / "e.pt", views=2))


def test_rl_condition_configs_share_the_encoder_and_differ_only_in_views():
    a = yaml.safe_load((REPO / "configs/rl/encoders/mvp3d_multiview.yaml").read_text())
    b = yaml.safe_load((REPO / "configs/rl/encoders/mvp3d_single.yaml").read_text())
    assert a["env"] == {"views": 3, "depth": True, "cameras": True}
    assert b["env"] == {"views": 1, "depth": True, "cameras": True}
    assert a["vision"] == b["vision"] == {"encoder_type": "mvp3d", "export_path": None}
    assert a["agent"] == b["agent"]
    assert not (REPO / "configs/rl/encoders/mvp3d.yaml").exists(), "each run names its input condition"


def test_drm_end_to_end_with_the_frozen_mvp3d_encoder(tmp_path):
    from s4d.rl.agent import DrMAgent
    from s4d.rl.evaluate import policy_inputs
    from s4d.rl.replay import Replay, replay_iterator

    device = torch.device("cuda")
    torch.manual_seed(0)
    save_export(tmp_path / "e.pt", build_encoder(TINY), asdict(TINY), RENDER, 0)
    agent = DrMAgent(rl_cfg(tmp_path / "e.pt"), 4, 4, device)
    enc = agent.encoder
    assert not agent.augment_pixels and enc.replay_atom_is_feature and not enc.replay_atom_is_stack_feature
    replay = Replay(None, enc.replay_atom_shape, np.float16, 4, 4, 1000, enc.replay_atom_frame_stack, 10)
    rng = np.random.default_rng(0)
    for _ in range(3):
        replay.add_initial(policy_inputs(agent, dict_observation(1, rng))[0][-1].cpu().numpy(), rng.normal(size=4))
        for t in range(12):
            atom = policy_inputs(agent, dict_observation(1, rng))[0][-1].cpu().numpy()
            replay.add(rng.uniform(-1, 1, 4), rng.normal(), 1.0, atom, rng.normal(size=4), t == 11)
    frozen = copy.deepcopy(enc.backbone.state_dict())
    batches = replay_iterator(replay, 16, 0.97, seed=0, device=device)
    for step in range(2, 6, 2):
        metrics = agent.update(batches, step)
        assert all(math.isfinite(v) for v in metrics.values()), metrics
    for key, value in enc.backbone.state_dict().items():
        assert torch.equal(value, frozen[key]), f"frozen encoder changed: {key}"
    action = agent.act(policy_inputs(agent, dict_observation(2, rng)), np.zeros((2, 4)), 5000, True)
    assert action.shape == (2, 4)


# ------------------------------------------------------------------------------------------ trainer, configs
def run_trainer(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]}
    result = subprocess.run([sys.executable, "-I", str(REPO / "scripts/train_mvp3d.py"), *args], env=env,
                            capture_output=True, text=True, timeout=900)
    if check:
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    return result


def test_pretraining_entry_point_exports_and_resumes_losslessly(tmp_path):
    path, manifest = write_dataset(tmp_path)  # 2 training episodes x 6 steps = 12 samples = 4 batches of 3
    cfg = yaml.safe_load((REPO / "configs/baselines/mvp3d/hammer.yaml").read_text())
    cfg["dataset"].update(hdf5_path=str(path), num_workers=1)
    cfg["data_interface"]["split_manifest"] = str(manifest)
    cfg["model"].update(asdict(TINY))
    cfg["train"].update(max_steps=6, log_every=1, eval_every=3, save_every=2, val_samples=4)
    config = tmp_path / "mvp3d.yaml"
    config.write_text(yaml.safe_dump(cfg))
    root = tmp_path / "runs"
    common = ["--config", str(config), "--output-root", str(root), "--no-wandb"]
    run_trainer(*common, "--name", "straight")
    run_trainer(*common, "--name", "resumed", "--set", "train.max_steps=3")
    run_trainer(*common, "--name", "resumed")  # continues at step 3, mid-epoch, then crosses the epoch boundary

    def losses(name):
        rows = [json.loads(line) for line in (root / name / "metrics.jsonl").read_text().splitlines()]
        return {r["step"]: r["train/loss"] for r in rows if "train/loss" in r}

    assert sorted(losses("straight")) == list(range(6)) and losses("straight") == losses("resumed")
    for name in ("straight", "resumed"):
        assert json.loads((root / name / "completion.json").read_text())["steps"] == 6
        assert any((root / name / "eval").iterdir())
    a, _ = load_export(root / "straight" / "encoder.pt")
    b, _ = load_export(root / "resumed" / "encoder.pt")
    assert a["step"] == b["step"] == 6 and a["model_cfg"] == asdict(TINY)
    for key, value in a["state_dict"].items():
        assert torch.equal(value, b["state_dict"][key]), key
    init, _ = load_export(root / "straight" / "encoder_init.pt")
    assert init["random_init"] and init["step"] == 0
    assert any(not torch.equal(v, a["state_dict"][k]) for k, v in init["state_dict"].items())
    init_b, _ = load_export(root / "resumed" / "encoder_init.pt")
    assert all(torch.equal(v, init_b["state_dict"][k]) for k, v in init["state_dict"].items())
    refused = run_trainer(*common, "--name", "straight", "--resume", "never", check=False)
    assert refused.returncode != 0 and "--resume never" in refused.stderr


def test_task_configs_share_the_recipe_and_the_size_matched_model():
    recipes = {}
    for task in TASKS:
        cfg = yaml.safe_load((REPO / f"configs/baselines/mvp3d/{task}.yaml").read_text())
        assert cfg["dataset"]["hdf5_path"].endswith(f"/{task}.hdf5")
        assert cfg["data_interface"]["split_manifest"].endswith(f"/{task}_seed0.json")
        assert cfg["model"] == asdict(MVP3DConfig()) and cfg["render"] == RENDER
        recipes[task] = {k: v for k, v in cfg.items() if k not in ("dataset", "data_interface")}
    assert all(r == recipes["hammer"] for r in recipes.values())
    train = recipes["hammer"]["train"]
    assert (train["batch_size"], train["epochs"], train["lr"], train["weight_decay"]) == (3, 15, 1e-4, 0.01)


def test_size_matched_encoder_fits_sincros_budget():
    from s4d.baselines.mvp3d.size import SINCRO_FLOPS_PER_FRAME, SINCRO_PARAMS, encoder_cost, rl_cost

    cost = encoder_cost(MVP3DConfig())
    assert cost["encoder_params"] <= SINCRO_PARAMS and cost["encoder_flops_per_frame"] <= SINCRO_FLOPS_PER_FRAME
    assert cost["encoder_params"] > 0.99 * SINCRO_PARAMS, "as close to the budget as the knobs allow"
    rl = rl_cost(build_encoder(MVP3DConfig()).eval(), MVP3DConfig().img_size)
    assert rl["rl_params"] == cost["encoder_params"] and rl["encoder_flops_per_frame"] == cost["encoder_flops_per_frame"]
    assert rl["feature_size"] == 5 * 256
    wider = encoder_cost(MVP3DConfig(attn_dim=264, attn_dim_head=33, decoder_dim=264, decoder_dim_head=33))
    assert wider["encoder_params"] > SINCRO_PARAMS, "the next width step (8 heads x 33) exceeds the budget"
    finer = encoder_cost(MVP3DConfig(img_size=60))
    assert finer["encoder_flops_per_frame"] > SINCRO_FLOPS_PER_FRAME, "5 x 5 tokens per view exceed the budget"
