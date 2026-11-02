"""Square crop of the DROID ViT-S-sized profile (s4d/data/droid/largescale/crop.py): geometry, resampling, the
border-patch rule, the eligible-token guarantee, the motion-informed sampler, the dataset fields, the single-stage
crop masking and the exclusion of padding from every supervision term (CPU only)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import droid_largescale_fixtures as F  # noqa: E402

from s4d.data.contract import ContractError, collate, validate_batch  # noqa: E402
from s4d.data.droid.largescale import crop as CR  # noqa: E402
from s4d.data.droid.largescale import geometry as G  # noqa: E402
from s4d.data.droid.largescale.dataset import DroidLargeScaleDataset  # noqa: E402
from s4d.model.encoder import sample_eligible_tokens, visible_count  # noqa: E402

H, W, S = 180, 320, 224
BOXES = [CR.CropBox(70, 0, 180), CR.CropBox(0, -70, 320), CR.CropBox(-20, -10, 200), CR.CropBox(40, -30, 250),
         CR.CropBox(140, -5, 190), CR.CropBox(100, 0, 180)]
SETTINGS = CR.CropSettings()


def _camera(seed=0):
    rng = np.random.default_rng(seed)
    K = np.array([[131.7, 0.0, 161.3], [0.0, 130.9, 88.6], [0.0, 0.0, 1.0]])  # no skew (s4d.geometry.project has none)
    c2w = F.lookat_c2w((0.45, -0.75, 0.55), (0.45, 0.0, 0.0))
    X = np.stack([rng.uniform(0.0, 0.9, 500), rng.uniform(-0.6, 0.6, 500), rng.uniform(0.0, 0.3, 500)], -1)
    return K, G.invert(c2w), X


# ------------------------------------------------------------------------------------------------ geometry
@pytest.mark.parametrize("box", BOXES)
def test_transformed_intrinsics_reproduce_the_analytic_crop_resize_mapping(box):
    """Projecting with K' equals mapping the original projection by (u - x0) S/s, including negative / out-of-image
    offsets; the continuous pixel-centre convention of s4d/geometry.py (torch) agrees with the numpy projection."""
    from s4d.geometry import project as torch_project

    K, w2c, X = _camera()
    uv, z = G.project(X, K, w2c)
    K2 = CR.crop_intrinsics(K, box, S)
    uv2, z2 = G.project(X, K2, w2c)
    expected = (uv - np.array([box.x0, box.y0])) * (S / box.side)
    assert np.abs(uv2 - expected).max() < 1e-4 and np.array_equal(z, z2)  # metric depth unchanged
    np.testing.assert_allclose(CR.map_pixels(uv, box, S), expected, atol=1e-9)
    uv_t, _ = torch_project(torch.as_tensor(X), torch.as_tensor(K2), torch.as_tensor(w2c))
    assert np.abs(uv_t.numpy() - expected).max() < 1e-4
    a = S / box.side
    assert K2[0, 0] == pytest.approx(a * K[0, 0]) and K2[1, 1] == pytest.approx(a * K[1, 1])
    assert K2[0, 2] == pytest.approx(a * (K[0, 2] - box.x0)) and K2[1, 2] == pytest.approx(a * (K[1, 2] - box.y0))
    # the pixel centres themselves: native centre (i + 0.5) lands at output (i + 0.5 - x0) S/s
    i = np.arange(W) + 0.5
    np.testing.assert_allclose(CR.map_pixels(np.stack([i, np.full_like(i, 0.5)], -1), box, S)[:, 0], (i - box.x0) * a)


@pytest.mark.parametrize("box", BOXES)
def test_rgb_resampling_follows_the_same_continuous_convention(box):
    """A blob at native continuous (u, v) has its output centroid at the analytically mapped point."""
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    checked = 0
    for u, v in ((100.3, 77.7), (180.0, 90.0), (160.2, 50.9), (205.6, 101.1)):
        blob = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * 4.0**2))
        image = np.repeat(np.round(blob * 255).astype(np.uint8)[..., None], 3, -1)
        target = CR.map_pixels([u, v], box, S)
        if not ((target > 20) & (target < S - 20)).all() or not (5 * 4.0 <= u - box.x0 <= box.side - 5 * 4.0):
            continue  # the blob must lie well inside the crop for its centroid to be unbiased
        out = CR.crop_rgb(image, box, S)[..., 0].astype(float)
        oy, ox = np.mgrid[0:S, 0:S] + 0.5
        centroid = np.array([(out * ox).sum(), (out * oy).sum()]) / out.sum()
        assert np.abs(centroid - target).max() < 0.05
        checked += 1
    assert checked


def test_padding_fill_observed_mask_and_border_patch_rule():
    box = CR.CropBox(0, -10, 190)  # 10 padded native rows on top -> 11.8 output rows; nothing padded elsewhere
    image = np.full((H, W, 3), 200, np.uint8)
    out = CR.crop_rgb(image, box, S)
    observed = CR.observed_mask(box, H, W, S)
    assert (out[~observed] < 200).all() and (out[observed] == 200).all()  # padding (fill 0) never leaks into observed
    assert (out[:11] == CR.PAD_RGB).all()  # pure padding rows hold the documented constant
    rows = observed.all(1)
    assert not rows[:12].any() and rows[13:].all()  # the boundary row blended with padding is not observed
    padded = CR.token_padding(observed, 16).reshape(14, 14)
    assert padded[0].all() and not padded[1:].any()  # patch row 0 is only PARTIALLY padded (rows 0..11 of 16): masked
    one = observed.copy()
    one[100, 37] = False  # a single non-observed pixel makes its patch padded
    assert CR.token_padding(one, 16).reshape(14, 14)[6, 2] and CR.token_padding(one, 16).sum() == padded.sum() + 1
    assert CR.observed_mask(CR.CropBox(70, 0, 180), H, W, S).all()  # no padding: everything observed


def test_depth_is_nearest_never_fabricated_and_zero_on_padding():
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.3, 2.5, (H, W)).astype(np.float32)
    depth[rng.random((H, W)) < 0.1] = 0
    for box in BOXES:
        observed = CR.observed_mask(box, H, W, S)
        out = CR.crop_depth(depth, box, S, observed)
        assert (out[~observed] == 0).all()
        assert set(np.unique(out[out > 0])) <= set(np.unique(depth))  # every value is a native measurement
        c = (np.arange(S) + 0.5) * box.side / S
        j, i = 150, 100
        r, q = int(np.floor(box.y0 + c[j])), int(np.floor(box.x0 + c[i]))
        if observed[j, i]:
            assert out[j, i] == depth[r, q]


def test_every_crop_side_leaves_the_kept_tokens_unpadded():
    """Worst case s = 320 pads (320 - 180) / 320 of the rows; with the any-pixel rule at least 84 of 196 tokens per frame
    stay unpadded (252 of 588 temporal tokens >= 147 kept) for every side and vertical placement."""
    need = visible_count(588, 0.75)
    assert need == 147
    worst = 196
    for side in range(180, 321):
        for y0 in sorted({180 - side, (180 - side) // 2, 0}):
            eligible = (~CR.token_padding(CR.observed_mask(CR.CropBox(0, y0, side), H, W, S), 16)).sum()
            worst = min(worst, int(eligible))
    assert worst == 84 and 3 * worst >= need


# ------------------------------------------------------------------------------------------------ sampler
def _scene(n_moving=200, n_static=2000, centre=(220.0, 110.0), spread=25.0, seed=0, dispersed=False):
    """Native pixel tracks over 3 frames: a moving cluster (world motion 5 cm) and static background."""
    rng = np.random.default_rng(seed)
    n = n_moving + n_static
    uv = np.empty((3, n, 2))
    base = np.stack([rng.uniform(0, W, n), rng.uniform(0, H, n)], -1)
    if dispersed:
        base[:n_moving] = np.stack([rng.uniform(0, W, n_moving), rng.uniform(0, H, n_moving)], -1)
    else:
        base[:n_moving] = np.array(centre) + rng.normal(0, spread, (n_moving, 2))
    for t in range(3):
        uv[t] = base + np.array([4.0 * t, 0.0]) * (np.arange(n) < n_moving)[:, None]
    motion = np.where(np.arange(n) < n_moving, 0.05, 0.002)
    return uv, np.ones((3, n), bool), motion, np.ones(n, bool)


def test_motion_crop_contains_the_moving_extent_with_context_and_stays_stochastic():
    uv, vis, motion, rel = _scene()
    boxes = set()
    for seed in range(40):
        box, info = CR.sample_crop(np.random.default_rng(seed), SETTINGS, H, W, uv=uv, visible=vis, motion=motion,
                                   reliable=rel, min_eligible_per_frame=49)
        assert info["fallback"] is None and info["moving_tracks"] == 200
        x_lo, y_lo, x_hi, y_hi = info["extent"]
        m = info["margin"]
        assert 0 <= box.x0 and box.x0 + box.side <= W  # s <= 320: clamped inside the image horizontally
        assert box.x0 <= max(0, x_lo - m) + 1e-6 and box.x0 + box.side >= min(W, x_hi + m) - 1e-6
        assert SETTINGS.side_min <= box.side <= SETTINGS.side_max
        if box.side > H:  # only the axis where s exceeds the image extent pads, never losing observed rows
            assert H - box.side <= box.y0 <= 0
        else:
            assert 0 <= box.y0 <= H - box.side
        boxes.add((box.x0, box.y0, box.side))
    assert len(boxes) > 20  # random side and position, never a deterministic argmax
    again = CR.sample_crop(np.random.default_rng(7), SETTINGS, H, W, uv=uv, visible=vis, motion=motion, reliable=rel)
    assert again == CR.sample_crop(np.random.default_rng(7), SETTINGS, H, W, uv=uv, visible=vis, motion=motion,
                                   reliable=rel)  # deterministic given the seed


@pytest.mark.parametrize("case,reason", [
    ("few", "few_reliable_tracks"), ("weak", "weak_motion"), ("dispersed", "dispersed"), ("none", "no_geometry"),
    ("failures", "weak_motion"),
])
def test_broad_fallbacks(case, reason):
    uv, vis, motion, rel = _scene()
    kw = {"uv": uv, "visible": vis, "motion": motion, "reliable": rel}
    if case == "few":
        kw["reliable"] = np.zeros_like(rel)
        kw["reliable"][:20] = True
    elif case == "weak":
        kw["motion"] = np.full_like(motion, 0.01)  # below 2 cm everywhere
    elif case == "dispersed":
        uv, vis, motion, rel = _scene(dispersed=True)
        kw = {"uv": uv, "visible": vis, "motion": motion, "reliable": rel}
    elif case == "none":
        kw = {}
    else:
        kw["motion"] = np.where(motion > 0.04, 0.9, motion)  # 0.9 m jumps are track failures, not motion
    sides = []
    for seed in range(20):
        box, info = CR.sample_crop(np.random.default_rng(seed), SETTINGS, H, W, **kw)
        assert info["fallback"] == reason and SETTINGS.broad_side_min <= box.side <= SETTINGS.side_max
        assert 0 <= box.x0 <= W - box.side and H - box.side <= box.y0 <= 0
        sides.append(box.side)
    assert len(set(sides)) > 5


def test_camera_ego_motion_never_counts_as_motion():
    """A moving (wrist-like) camera sees static world points at changing pixels: world-frame motion is zero, so the
    crop falls back instead of following the apparent image motion."""
    K, w2c, X = _camera()
    poses = [w2c @ F.small_rigid(0, 0, 0.05 * t, 0.03 * t, 0, 0) for t in range(3)]
    points = np.broadcast_to(X, (3, *X.shape)).copy()
    uv = np.stack([G.project(points[t], K, poses[t])[0] for t in range(3)])
    assert np.abs(uv[2] - uv[0]).max() > 5  # large apparent image motion
    motion, reliable = CR.track_motion(points, np.ones((3, len(X)), bool))
    assert motion.max() == 0 and reliable.all()
    _, info = CR.sample_crop(np.random.default_rng(0), SETTINGS, H, W, uv=uv, visible=np.ones((3, len(X)), bool),
                             motion=motion, reliable=reliable)
    assert info["fallback"] == "weak_motion"
    vis = np.ones((3, 3), bool)
    vis[1, 0] = vis[2, 0] = False  # visible once only: unreliable
    m, r = CR.track_motion(np.stack([np.zeros((3, 3)), np.ones((3, 3)), 2 * np.ones((3, 3))]), vis)
    assert not r[0] and r[1:].all() and np.allclose(m[1:], 2 * np.sqrt(3))


def test_eligibility_shrink_and_center_policy():
    box, info = CR.sample_crop(np.random.default_rng(0), SETTINGS, H, W, policy="center")
    assert box == CR.CropBox(70, 0, 180) and info["fallback"] == "center"
    strict = CR.CropSettings(broad_side_min=320)
    box, info = CR.sample_crop(np.random.default_rng(0), strict, H, W, min_eligible_per_frame=150)
    eligible = (~CR.token_padding(CR.observed_mask(box, H, W, S), 16)).sum()
    assert info["shrunk"] > 0 and eligible >= 150 and box.side < 320


def test_inference_center_crop_contract():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)  # a RealSense colour frame
    K = np.array([[615.0, 0.0, 321.5], [0.0, 615.0, 242.3], [0.0, 0.0, 1.0]])
    out, K2, box = CR.inference_center_crop(image, K)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8 and box == CR.CropBox(80, 0, 480)
    a = 224 / 480
    np.testing.assert_allclose(K2, [[615 * a, 0, (321.5 - 80) * a], [0, 615 * a, 242.3 * a], [0, 0, 1]])
    np.testing.assert_array_equal(out, CR.crop_rgb(image, box, 224))
    _, K3, box3 = CR.inference_center_crop(np.zeros((180, 320, 3), np.uint8), np.eye(3))
    assert box3 == CR.CropBox(70, 0, 180)


def test_settings_validation():
    assert CR.CropSettings.from_config(None) is None
    with pytest.raises(ValueError, match="unknown data.crop keys"):
        CR.CropSettings.from_config({"side": 3})
    with pytest.raises(ValueError, match="eval_policy"):
        CR.CropSettings.from_config({"eval_policy": "motion_argmax"})
    assert CR.CropSettings.from_config({"extent_quantiles": [0.1, 0.9]}).extent_quantiles == (0.1, 0.9)


# ------------------------------------------------------------------------------------------------ masking
def test_eligible_token_masking_is_single_stage_and_fixed_count():
    from s4d.train.loop import crop_visible_tokens

    torch.manual_seed(0)
    padded = torch.zeros(3, 2, 196, dtype=torch.bool)
    padded[0, 0, :56] = True  # 4 padded patch rows
    padded[1, 1, -112:] = True  # 8 padded rows (84 left per frame)
    padded[2, 0, 7::14] = True  # a column
    visible, stats = crop_visible_tokens(padded, 3, visible_count(588, 0.75))
    assert visible.shape == (6, 588) and (visible.sum(1) == 147).all()  # 441 masked in every row, whatever the crop
    eligible = (~padded).reshape(6, 1, 196).expand(-1, 3, -1).reshape(6, 588)
    assert not (visible & ~eligible).any()  # padded tokens are always masked
    np.testing.assert_allclose(stats["padded_tokens"].numpy(), [168, 0, 0, 336, 42, 0])
    np.testing.assert_allclose(stats["masked_fraction_observed"].numpy()[[0, 1, 3]], [1 - 147 / 420, 0.75, 1 - 147 / 252])
    counts = torch.zeros(588)
    for _ in range(200):  # uniform among eligible tokens: no motion or position priority
        counts += crop_visible_tokens(padded[:1, :1], 3, 147)[0][0].float()
    frac = counts[eligible[0]] / 200
    assert abs(float(frac.mean()) - 147 / 420) < 0.01 and float(frac.std()) < 0.06
    with pytest.raises(ValueError, match="fewer than 147 eligible"):
        sample_eligible_tokens(torch.zeros(1, 588, dtype=torch.bool), 147)


def test_ssim_support_and_masked_pixel_weights():
    from s4d.losses.rgb import pixel_weights, ssim_support

    observed = torch.ones(2, 1, 40, 30, dtype=torch.bool)
    observed[0, :, :10] = False
    support = ssim_support(observed)
    assert (support[0, :, :15] == 0).all() and (support[0, :, 15:] == 1).all() and (support[1] == 1).all()
    score = torch.rand(2, 1, 40, 30)
    w = pixel_weights(score, 1.0, observed)
    assert (w[0, :, :10] == 0).all() and torch.allclose(w.mean(dim=(1, 2, 3)), torch.ones(2))
    torch.testing.assert_close(pixel_weights(score, 1.0, torch.ones_like(observed)), pixel_weights(score, 1.0))


# ------------------------------------------------------------------------------------------------ dataset
CROP = {"out_size": 32, "min_moving_tracks": 4, "margin_min_px": 8}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return F.build_all(tmp_path_factory.mktemp("droid_crop"))


def make_ds(env, **kw):
    base = {"image_height": 32, "image_width": 32, "crop": CROP, "patch_size": 4, "min_eligible_tokens": 48}
    base.update(kw)
    return DroidLargeScaleDataset(env["manifest"], **F.dataset_kwargs(env, **base))


def test_dataset_crop_fields_geometry_and_contract(env):
    ds, plain = make_ds(env), DroidLargeScaleDataset(env["manifest"], **F.dataset_kwargs(env))
    gt = env["gt"]
    index = ds.windows.index((0, "5:16", 5))
    item, ref = ds[index], plain[index]
    assert item["images"].shape == (3, 2, 3, 32, 32) and item["observed"].shape == (2, 1, 32, 32)
    assert item["token_padded"].shape == (2, 64) and item["depth"].shape == (3, 2, 1, 32, 32)
    validate_batch({k: v if k == "meta" else v.unsqueeze(0) for k, v in item.items()})
    for v, role in enumerate(("ext1", "ext2")):
        x0, y0, s = (int(c) for c in item["crop_box"][v])
        box = CR.CropBox(x0, y0, s)
        np.testing.assert_allclose(item["K"][v].numpy(), CR.crop_intrinsics(gt["K_ext"], box, 32), atol=1e-4)
        assert torch.equal(item["w2c"][v], ref["w2c"][v])  # extrinsics unchanged
        obs = item["observed"][v, 0].numpy()
        assert (item["depth"][:, v, 0].numpy()[:, ~obs] == 0).all()
        np.testing.assert_array_equal(item["token_padded"][v].numpy(), CR.token_padding(obs, 4))
        # tracks: the cropped weights are the native ones restricted to the crop's observed pixels at the source time
        n = int(item["track_count"][v])
        w, w_ref = item["track_weight"][:, v, :n].numpy(), ref["track_weight"][:, v, :n].numpy()
        assert (w <= w_ref).all()
        for p, (a, _, _) in enumerate(((0, 1, 0), (1, 2, 1), (0, 2, 0))):
            src = ref["track_src"][p, v, :n].numpy()
            uv, _ = G.project(src, item["K"][v].numpy().astype(np.float64), item["w2c"][v].numpy().astype(np.float64))
            inside = CR.in_observed(uv, obs)
            np.testing.assert_array_equal(w[p] > 0, (w_ref[p] > 0) & inside)
        assert torch.equal(item["track_src"][:, v][item["track_weight"][:, v] > 0],
                           ref["track_src"][:, v][item["track_weight"][:, v] > 0])  # world positions unchanged
        # the same crop for every history frame: each frame is the crop of its own native frame (uniform fixture colour)
        for t, step in enumerate(item["meta"]["raw_indices"]):
            pix = item["images"][t, v].permute(1, 2, 0).numpy()[obs].astype(float)
            np.testing.assert_allclose(pix.mean(0), F.frame_color(step, v), atol=3)
    assert item["wrist_images"].shape == (3, 3, 32, 32) and item["wrist_observed"].shape == (1, 32, 32)
    wrist_obs = item["wrist_observed"][0].numpy()
    assert (item["wrist_depth"][:, 0].numpy()[:, ~wrist_obs] == 0).all()


def test_dataset_crop_is_deterministic_per_window_and_camera(env):
    ds = make_ds(env)
    a, b = ds[4], pickle.loads(pickle.dumps(ds))[4]
    for key in a:
        if key != "meta":
            assert torch.equal(a[key], b[key]), key
    boxes = {tuple(ds[i]["crop_box"][v].tolist()) for i in range(len(ds)) for v in range(2)}
    assert len(boxes) > 3  # different windows / cameras draw different crops
    other = make_ds(env, seed=1)
    assert any(not torch.equal(other[i]["crop_box"], ds[i]["crop_box"]) for i in range(len(ds)))  # seed enters
    moving = [int(c) for i in range(len(ds)) for c in ds[i]["crop_moving_tracks"]]
    assert max(moving) >= 4  # the fixture's sliding patch is motion
    center = make_ds(env, crop_policy="center")[4]
    assert center["crop_box"].tolist() == [[70, 0, 180]] * 2 and not center["token_padded"].any()
    assert (center["crop_fallback"] == CR.FALLBACK_CODES["center"]).all()


def test_dataset_crop_config_wiring_and_guards(env):
    from s4d.config import load_config
    from s4d.data.droid.largescale import config as DC

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config([repo / "configs/droid/vits_largescale.yaml"], [
        f"data.manifest={env['manifest']}", f"data.pointworld_root={env['tmp'] / 'pw'}",
        f"data.rlds_index_dir={env['tmp'] / 'rlds_index'}", f"data.rlds_data_root={env['tmp'] / 'rlds'}"])
    kw = DC.dataset_kwargs(cfg)
    assert kw["patch_size"] == 16 and kw["min_eligible_tokens"] == 147 and kw["crop"]["out_size"] == 224
    plain = load_config([repo / "configs/droid/csw4_largescale.yaml"], [
        f"data.manifest={env['manifest']}", f"data.pointworld_root={env['tmp'] / 'pw'}",
        f"data.rlds_index_dir={env['tmp'] / 'rlds_index'}", f"data.rlds_data_root={env['tmp'] / 'rlds'}"])
    assert not {"crop", "patch_size", "min_eligible_tokens"} & set(DC.dataset_kwargs(plain))  # csw4: no new keys
    with pytest.raises(ValueError, match="must equal the encoder image size"):
        make_ds(env, image_height=36, image_width=64)
    cfg["model"]["encoder"]["mask_mode"] = "tube"
    with pytest.raises(ValueError, match="mask_mode random"):
        DC.dataset_kwargs(cfg)


def test_contract_rejects_inconsistent_crop_fields(env):
    batch = collate([make_ds(env)[0], make_ds(env)[3]])
    bad = dict(batch, token_padded=~batch["token_padded"])
    with pytest.raises(ContractError, match="token_padded must mark"):
        validate_batch(bad)
    obs = batch["observed"].clone()
    obs[0, 0, 0, 10:20, 10:20] = False  # claims padding where depth is valid
    with pytest.raises(ContractError, match="depth must be 0 on padded pixels"):
        validate_batch(dict(batch, observed=obs))
    with pytest.raises(ContractError, match="given together"):
        validate_batch({k: v for k, v in batch.items() if k != "observed"})


# ------------------------------------------------------------------------------------------------ losses
def test_padding_is_never_supervision_nor_encoder_input(env, monkeypatch):
    """Changing the RGB of padded pixels changes neither the loss nor any gradient (training mode: the crop masking
    hides padded tokens; RGB / coverage / depth / tracks ignore padded pixels)."""
    from s4d.config import load_config
    from s4d.losses import rgb
    from s4d.model import render
    from s4d.model.decoder import DecoderConfig, GaussianDecoder, GroupConfig
    from s4d.model.encoder import Encoder, EncoderConfig
    from s4d.train.loop import Model, forward_losses
    from test_review_options import _fake_rasterizer

    monkeypatch.setattr(render, "_rasterize", _fake_rasterizer([]))
    monkeypatch.setattr(rgb, "_ssim_map", lambda x, y: 1 - (x - y).square())
    ds = make_ds(env, max_tracks=32, crop={**CROP, "side_min": 300, "broad_side_min": 300})  # large sides: padding
    batch = collate([ds[0], ds[7]])
    assert batch["token_padded"].any() and (~batch["observed"]).any()
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config([repo / "configs/droid/vits_largescale.yaml"], ["model.anchor_stats=null"])
    cfg["train"]["bf16"] = False
    cfg["aug"]["synth_views"] = 0
    group = GroupConfig(parents=2, children=2, offset_scale=0.2, child_radius=0.05, scale_min=0.001, scale_max=0.08)
    torch.manual_seed(0)
    enc = Encoder(EncoderConfig(image_height=32, image_width=32, patch_size=4, width=16, depth=1, heads=4, num_slots=1,
                                slot_dim=8, drop_path=0, mask_ratio=0.75, mask_mode="random"))
    model = Model(enc, GaussianDecoder(DecoderConfig(slot_dim=8, dim=16, depth=1, heads=4, scene=group, dynamic=group,
                                                     num_slots=1))).train()

    def run(b):
        model.zero_grad()
        torch.manual_seed(3)
        out = forward_losses(model, b, cfg, 10, source=torch.tensor([0, 1]))
        out["total"].backward()
        grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        return out, grads

    out, grads = run(batch)
    assert torch.isfinite(out["total"]) and (out["metrics"]["crop_padded_tokens"] > 0)
    assert out["metrics"]["crop_masked_fraction_observed"] < 0.75
    noisy = dict(batch)
    pad = ~batch["observed"][:, None].expand_as(batch["images"][:, :, :, :1]).expand_as(batch["images"])
    noisy["images"] = torch.where(pad, torch.randint_like(batch["images"], 0, 256), batch["images"])
    assert not torch.equal(noisy["images"], batch["images"])
    out2, grads2 = run(noisy)
    assert torch.equal(out2["total"].detach(), out["total"].detach())
    assert all(torch.equal(a, b) for a, b in zip(grads, grads2))
    model.eval()
    with torch.no_grad():  # evaluation (mask_ratio 0): no crop masking, every token visible
        ev = forward_losses(model, batch, cfg, 10, source=torch.tensor([0, 1]), mask_ratio=0.0)
    assert "crop_padded_tokens" not in ev["metrics"] and torch.isfinite(ev["total"])


def test_crop_varies_across_visits_and_resume_replays_it(env):
    """The trainer sampler tags crop-dataset indices with the epoch: a window draws a new crop on every visit, and a
    resumed stream (persistent workers) reproduces the same (epoch, step) batches exactly. Plain datasets get ints."""
    from s4d.train.ddp import DistContext
    from s4d.train.loop import ResumableDistributedSampler, infinite, make_train_loader

    ds = make_ds(env, max_tracks=16)
    assert ds.epoch_tagged_indices and not make_ds(env, crop_policy="center").epoch_tagged_indices
    plain = DroidLargeScaleDataset(env["manifest"], **F.dataset_kwargs(env))
    assert not getattr(plain, "epoch_tagged_indices", False)
    s = ResumableDistributedSampler(plain, num_replicas=1, rank=0, shuffle=True, seed=3, drop_last=True)
    assert all(isinstance(i, int) for i in s)
    boxes = {e: ds[(4, e)]["crop_box"] for e in range(4)}
    assert torch.equal(boxes[0], ds[4]["crop_box"])  # a plain index is epoch 0
    assert len({tuple(b.flatten().tolist()) for b in boxes.values()}) > 1  # same window, different epochs
    assert torch.equal(ds[(4, 2)]["crop_box"], boxes[2])  # deterministic per (window, epoch)

    cfg = {"train": {"num_workers": 2, "batch_size": 2, "prefetch_factor": 2}}
    ctx = DistContext(0, 0, 1, torch.device("cpu"))
    key = lambda b: (tuple(b["meta"]["t_indices"]), tuple(b["meta"]["clip"]))  # noqa: E731
    stream = infinite(make_train_loader(ds, cfg, ctx, seed=3), start_step=0)
    steps = len(make_train_loader(ds, cfg, ctx, seed=3))
    first = [next(stream) for _ in range(2 * steps)]  # two epochs: every window visited twice
    by_window = {}
    for b in first:
        for j in range(2):
            by_window.setdefault((b["meta"]["clip"][j], b["meta"]["t_indices"][j]), []).append(b["crop_box"][j])
    assert any(not torch.equal(v[0], v[1]) for v in by_window.values() if len(v) == 2)
    resumed = infinite(make_train_loader(ds, cfg, ctx, seed=3), start_step=steps + 1)
    for b in first[steps + 1:]:
        r = next(resumed)
        assert key(r) == key(b)
        for name in ("crop_box", "images", "track_weight", "token_padded"):
            assert torch.equal(r[name], b[name]), name


def test_padding_does_not_change_the_relative_weight_of_depth_and_coverage(monkeypatch):
    """Uniform errors: RGB, coverage, depth L1 (and the enabled-on-demand gradient / hard-depth terms) give the same loss
    for the padding-free s = 180 crop and the most padded s = 320 crop."""
    from s4d.losses import rgb
    from s4d.losses.depth import depth_gradient_loss, depth_l1
    from s4d.losses.rgb import coverage_loss, pixel_weights, rgb_loss

    monkeypatch.setattr(rgb, "_ssim_map", lambda x, y: 1 - (x - y).square())
    values = {}
    for box in (CR.CropBox(70, 0, 180), CR.CropBox(0, -70, 320)):
        obs = torch.from_numpy(CR.observed_mask(box, H, W, S))[None, None]
        weights = pixel_weights(torch.zeros(1, 1, S, S), 1.0, obs)
        geo = weights * obs.float().mean(dim=(1, 2, 3), keepdim=True)  # as loop.forward_losses
        yy, xx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
        target = 1.0 + 0.002 * (xx + yy).float()[None, None]
        values[box.side] = [
            float(rgb_loss(torch.full((1, 3, S, S), 0.7), torch.full((1, 3, S, S), 0.2), weights, 0.0)),
            float(coverage_loss(torch.full((1, 1, S, S), 0.5), obs, geo)),
            float(depth_l1(target + 0.5, target, obs, geo)),
            float(depth_gradient_loss(target * torch.exp(0.01 * xx.float()), target, obs, levels=1, weights=geo)),
        ]
        # the gradient term's value depends on the region's shape, so check its weights: 1 on every observed pixel
        torch.testing.assert_close(geo[obs], torch.ones(int(obs.sum())))
        assert values[box.side][3] == pytest.approx(
            float(depth_gradient_loss(target * torch.exp(0.01 * xx.float()), target, obs, levels=1)), rel=1e-5)
    np.testing.assert_allclose(values[180][:3], values[320][:3], rtol=1e-4, atol=1e-7)
    assert values[180][2] == pytest.approx(0.5) and values[180][1] == pytest.approx(np.log(2), rel=1e-5)
