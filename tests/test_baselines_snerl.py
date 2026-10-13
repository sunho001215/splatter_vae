"""SNeRL baseline: the patched official code, our adapters against verbatim official functions and loop lines, the
data adapter (cameras, semantic classes, scene draws), export/reload, the frozen RL encoder and trainer resume."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from s4d.baselines.data import split_episodes
from s4d.baselines.snerl import data as D
from s4d.baselines.snerl import model as M
from s4d.baselines.snerl import upstream as U
from s4d.data.metaworld.cameras import LOOKAT, RADIUS, camera_rig, heldout_camera_set, orbit_rig
from s4d.third_party import source_tree

REPO = Path(__file__).resolve().parents[1]
SIZE = 128  # the official encoder (fc 32*57*57) and the official DINO lines (31x31 tokens -> 128x128) need 128^2
COMMON_BODIES = [
    "world", "tablelink", "RetainingWall", "base", "controller_box", "pedestal_feet", "torso", "pedestal",
    "right_arm_base_link", "right_l0", "head", "screen", "head_camera", "right_torso_itb", "right_l1", "right_l2",
    "right_l3", "right_l4", "right_arm_itb", "right_l5", "right_hand_camera", "right_wrist", "right_l6", "right_hand",
    "hand", "rightclaw", "rightpad", "leftclaw", "leftpad", "right_l4_2", "right_l2_2", "right_l1_2", "mocap",
]  # fmt: skip
TASK_TAILS = {  # body_names attributes of /home/ws/data/metaworld/splatter4d_v1/<task>.hdf5 after COMMON_BODIES
    "hammer": ["hammerbody", "hammer", "box", "hammerblock", "nail_link"],
    "pick-place": ["obj"],
    "bin-picking": ["obj", "objA", "bin_start", "binA", "bin_goal", "binB"],
    "door-open": ["door", "doorlockB", "door_link"],
    "peg-insert-side": ["peg", "box", "body35"],
    "peg-unplug-side": ["plug1", "plug", "box", "plug_wall"],
    "stick-push": ["stick", "body34", "object", "body36"],
}
TASKS = ("hammer", "pick-place", "bin-picking", "coffee-push", "door-open", "peg-insert-side", "peg-unplug-side",
         "stick-push")
TINY_NERF = {"netdepth": 2, "netwidth": 32, "netdepth_fine": 2, "netwidth_fine": 32, "N_samples": 8, "N_importance": 8,
             "N_rand": 64, "chunk": 1024, "netchunk": 4096, "precrop_iters": 2}


def body_names(task: str) -> list[str]:
    if task == "coffee-push":  # the mug sits between the scene and the robot in this model
        return COMMON_BODIES[:3] + ["obj", "mug"] + COMMON_BODIES[3:] + ["coffee_machine", "coffeemachine", "cm_link",
                                                                         "cmbutton"]
    return COMMON_BODIES + TASK_TAILS[task]


def nerf_cfg(**overrides) -> dict:
    cfg = yaml.safe_load((REPO / "configs/baselines/snerl/hammer.yaml").read_text())["nerf"]
    return {**cfg, **overrides}


def write_task_file(tmp_path: Path, episodes: int = 3, length: int = 5, seed: int = 0) -> tuple[Path, Path]:
    """Our HDF5 layout (hammer bodies, the real camera rig at 128^2) with random content."""
    rng = np.random.default_rng(seed)
    rig = camera_rig(SIZE, SIZE)
    names = body_names("hammer")
    path = tmp_path / "hammer.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs.update(task="hammer", body_names=json.dumps(names), background_body=65535, depth_unit_m=1e-4)
        cams = f.create_group("cameras")
        for key in ("K", "c2w", "w2c", "is_train"):
            cams.create_dataset(key, data=rig[key])
        group = f.create_group("episodes")
        for e in range(episodes):
            ep = group.create_group(f"ep{e:03d}")
            ep.attrs["length"] = length
            ep.create_dataset("rgb", data=rng.integers(0, 256, (length, 10, SIZE, SIZE, 3), dtype=np.uint8))
            ids = np.append(np.arange(len(names)), 65535).astype(np.uint16)
            ep.create_dataset("body_id", data=rng.choice(ids, (length, 6, SIZE, SIZE)))
            ep.create_dataset("depth", data=rng.integers(4000, 16000, (length, 10, SIZE, SIZE), dtype=np.uint16))
    manifest = tmp_path / "split.json"
    eps = [f"ep{e:03d}" for e in range(episodes)]
    manifest.write_text(json.dumps({"train": eps[:-1], "validation": eps[-1:]}))
    return path, manifest


def official_lines(first: int, last: int, first_text: str, last_text: str) -> str:
    """Lines ``first``..``last`` (1-based) of the official run_nerf.py (patched copy, same numbering), dedented."""
    lines = (source_tree("snerl") / "nerf_pretrain" / "run_nerf.py").read_text().split("\n")[first - 1: last]
    assert first_text in lines[0] and last_text in lines[-1], (lines[0], lines[-1])
    return textwrap.dedent("\n".join(lines))


# ------------------------------------------------------------------------------------------------ official code
def test_patched_copy_only_drops_unused_imports_and_keeps_line_numbers():
    checkout, patched = REPO / "third_party/snerl/nerf_pretrain", source_tree("snerl") / "nerf_pretrain"
    assert patched != checkout
    changed = {"run_nerf.py": {12, 16, 536}, "dino.py": {8, 72}}
    for path in sorted(checkout.rglob("*")):
        if path.is_file():
            ours = patched / path.relative_to(checkout)
            if path.name not in changed:
                assert ours.read_bytes() == path.read_bytes(), path.name
                continue
            a, b = path.read_text().split("\n"), ours.read_text().split("\n")
            assert len(a) == len(b)
            assert {i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y} == changed[path.name]
            assert all("s4d patch 0001" in b[i - 1] for i in changed[path.name])
    run_nerf = (patched / "run_nerf.py").read_text().split("\n")
    assert run_nerf[535].strip().startswith("from load_blender import load_blender_data")  # inside train()
    for module in ("snerl", "dino"):
        status = subprocess.run(["git", "-C", str(REPO / "third_party" / module), "status", "--porcelain",
                                 "--untracked-files=all"], capture_output=True, text=True, check=True).stdout
        assert status == "", f"third_party/{module} must stay a clean checkout"


def test_official_modules_load_under_unique_names_without_side_effects():
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    modules = U.official()
    assert np.random.rand() == expected, "the import-time np.random.seed(0) of run_nerf.py is undone"
    assert {m.__name__ for m in modules.values()} == {f"s4d_snerl_{n}" for n in U.NERF_MODULES}
    assert all(Path(m.__file__).is_relative_to(source_tree("snerl")) for m in modules.values())
    assert not any(n in sys.modules for n in (*U.NERF_MODULES, "load_blender", "vision_transformer", "hubconf"))
    assert modules["run_nerf"].PixelEncoder is modules["multiview_encoder"].PixelEncoder


# --------------------------------------------------------------------------------------------- cameras and rays
def all_cameras_cv() -> np.ndarray:
    rig = camera_rig(SIZE, SIZE)["c2w"]
    poses = [(c["azimuth"], c["elevation"], c["distance"]) for name in ("near", "traj") for c in heldout_camera_set(name)]
    return np.concatenate([rig, orbit_rig(poses, SIZE, SIZE)["c2w"]]).astype(np.float64)


def test_pose_normalisation_round_trip_and_official_scene_geometry():
    c2w_cv = torch.from_numpy(all_cameras_cv())
    c2w = D.normalise_c2w(c2w_cv)
    torch.testing.assert_close(D.denormalise_c2w(c2w), c2w_cv, atol=1e-12, rtol=0)
    R, t = c2w[:, :3, :3], c2w[:, :3, 3]
    torch.testing.assert_close(R.transpose(1, 2) @ R, torch.eye(3, dtype=R.dtype).expand_as(R), atol=1e-6, rtol=0)
    assert torch.allclose(torch.linalg.det(R), torch.ones(len(R), dtype=R.dtype), atol=1e-6)
    rig = slice(0, 10)
    torch.testing.assert_close(t[rig].norm(dim=-1), torch.full((10,), 4.0 * RADIUS, dtype=t.dtype), atol=1e-6, rtol=0)
    # OpenGL: the camera looks down -z; every rig camera looks at the normalised origin (the rig look-at point)
    forward = -R[rig, :, 2]
    torch.testing.assert_close(forward, -t[rig] / t[rig].norm(dim=-1, keepdim=True), atol=1e-6, rtol=0)
    assert D.POSE_SCALE * 0.5 == 2.0 and D.POSE_SCALE * 1.5 == 6.0, "official near/far 2/6 = 0.5 m / 1.5 m"


def project_cv(K: np.ndarray, c2w_cv: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    w2c = np.linalg.inv(c2w_cv)
    cam = w2c[:3, :3] @ x + w2c[:3, 3]
    return np.array([K[0, 0] * cam[0] / cam[2] + K[0, 2], K[1, 1] * cam[1] / cam[2] + K[1, 2]]), float(cam[2])


def test_official_get_rays_with_our_cameras_pass_through_pixel_centres_at_camera_z_depth():
    """Official ``get_rays`` with our converted K and pose: the ray of pixel (i, j) hits our OpenCV pixel centre
    (i + 0.5, j + 0.5); its parameter is camera-z depth in normalised units (metres x 4), so near/far bracket
    0.5-1.5 m; the rays at the image centre pass through the look-at point."""
    get_rays = U.official("run_nerf")["run_nerf"].get_rays
    rig = camera_rig(SIZE, SIZE)
    cams = all_cameras_cv()
    pixels = [(0, 0), (SIZE - 1, 0), (17, 90), (SIZE // 2, SIZE // 2), (SIZE - 1, SIZE - 1)]
    for v in range(len(cams)):
        K_cv = rig["K"][0].astype(np.float64)
        c2w = D.normalise_c2w(torch.from_numpy(cams[v])).float()
        rays_o, rays_d = get_rays(SIZE, SIZE, D.official_K(K_cv), c2w[:3, :4])
        for i, j in pixels:
            s = 3.0
            x = (rays_o[j, i] + s * rays_d[j, i]).double()
            world = (x / D.POSE_SCALE + torch.tensor(LOOKAT, dtype=torch.float64)).numpy()
            uv, z = project_cv(K_cv, cams[v], world)
            assert np.allclose(uv, [i + 0.5, j + 0.5], atol=2e-3), (v, i, j, uv)
            assert math.isclose(z, s / D.POSE_SCALE, rel_tol=1e-5), "ray parameter = camera-z depth x pose scale"
        if v < 10:  # rig cameras aim at the look-at point: it lies between the four central pixel rays
            centre = rays_d[SIZE // 2 - 1: SIZE // 2 + 1, SIZE // 2 - 1: SIZE // 2 + 1].reshape(4, 3).mean(0)
            o = rays_o[0, 0]
            closest = o - (o @ centre) / (centre @ centre) * centre
            assert float(closest.norm()) < 1e-3


# ---------------------------------------------------------------------------------------------- semantic classes
def test_semantic_class_table_covers_every_body_of_every_task_exactly_once():
    assert set(D.TASK_BODIES) == set(TASKS)
    for task in TASKS:
        names = body_names(task)
        table = D.class_table(task, names)
        assert table.shape == (len(names),) and set(np.unique(table)) == {0, 1, 2, 3}
        assert table[names.index("world")] == D.BACKGROUND and table[names.index("tablelink")] == D.FIXTURE
        assert table[names.index("rightpad")] == D.ROBOT and (table == D.ROBOT).sum() == len(D.ROBOT_BODIES)
        assert {names[i] for i in np.where(table == D.OBJECT)[0]} == set(D.TASK_BODIES[task]["object"])
        lut = D.class_lookup(table, 65535)
        ids = np.array([[names.index("hammer") if task == "hammer" else 0, 65535]], dtype=np.uint16)
        assert lut[ids].tolist() == [[D.OBJECT if task == "hammer" else D.BACKGROUND, D.BACKGROUND]]
    with pytest.raises(ValueError, match="without a semantic class"):
        D.class_table("hammer", body_names("hammer") + ["new_body"])
    with pytest.raises(ValueError, match="not in the model"):
        D.class_table("hammer", body_names("pick-place"))
    with pytest.raises(ValueError, match="collides"):
        D.class_lookup(np.zeros(40, np.uint8), 12)
    assert len(D.SEMANTIC_CLASSES) == 4 < D.NUM_SEMANTIC_OUTPUTS == 5  # the official 5-way head, class 4 unused


# ------------------------------------------------------------------------------------------------- scene draws
def test_scenes_follow_the_official_draws_deterministically(tmp_path):
    path, manifest = write_task_file(tmp_path)
    episodes = split_episodes(manifest, "train")
    ds = D.SNeRLScenes(path, episodes, seed=7, num_inputs=3, length=600)
    item = ds[11]
    assert item["obs_rgb"].shape == (3, SIZE, SIZE, 3) and item["obs_rgb"].dtype == np.uint8
    assert item["target_rgb"].shape == (SIZE, SIZE, 3) and item["target_semantic"].dtype == np.int64
    assert set(np.unique(item["target_semantic"])) <= {0, 1, 2, 3}
    with h5py.File(path, "r") as f:
        ep = f["episodes"][item["episode"]]
        np.testing.assert_array_equal(item["obs_rgb"], np.stack([ep["rgb"][item["t"], v] for v in item["inputs"]]))
        np.testing.assert_array_equal(item["target_rgb"], ep["rgb"][item["t"], item["target"]])
        np.testing.assert_array_equal(item["target_semantic"], ds.lut[ep["body_id"][item["t"], item["target"]]])
        c2w_cv = torch.from_numpy(f["cameras"]["c2w"][:].astype(np.float64))
    torch.testing.assert_close(item["obs_c2w"], D.normalise_c2w(c2w_cv[item["inputs"]]).float())
    torch.testing.assert_close(item["target_K"].numpy(), D.official_K(camera_rig(SIZE, SIZE)["K"][0]))
    again = D.SNeRLScenes(path, episodes, seed=7, num_inputs=3, length=600)
    for i in (0, 11, 599):
        assert ds.draw(i) == again.draw(i), "item i depends only on (seed, i)"
    draws = [ds.draw(i) for i in range(600)]
    assert all(len(set(inputs)) == 3 and set(inputs) <= set(range(6)) for _, _, inputs, _ in draws)
    frames = {(ep, t) for ep, t, _, _ in draws}
    assert frames == {(ep, t) for ep in episodes for t in range(5)}, "uniform over every frame of the split"
    targets = np.bincount([target for *_, target in draws], minlength=6)
    assert targets.min() > 60, "target uniform over all six training cameras (inputs included)"
    assert any(target in inputs for _, _, inputs, target in draws) and ds.draw(1) != D.SNeRLScenes(
        path, episodes, seed=8, num_inputs=3).draw(1)


# --------------------------------------------------------------------------------- encoder, DINO, loop, render
def tiny_model(device: str = "cpu", seed: int = 0, live: bool = False) -> M.SNeRL:
    """Small official networks; ``live`` shifts the density bias up so no ray is empty (full-image render tests)."""
    torch.manual_seed(seed)
    model = M.create(nerf_cfg(**TINY_NERF), device)
    if live:
        with torch.no_grad():
            for name in ("network_fn", "network_fine"):
                model.render_kwargs_train[name].alpha_linear.bias.add_(1.0)
    return model


def test_batched_encoder_and_rl_backbone_match_the_official_pixel_encoder(tmp_path):
    from s4d.baselines.snerl.encoder import SNeRLFrameEncoder

    torch.manual_seed(0)
    encoder = M.build_encoder().eval()
    assert sum(p.numel() for p in encoder.parameters()) == 6_587_885
    rgb = torch.rand(4, 3, 3, SIZE, SIZE)
    c2w_cv = torch.from_numpy(all_cameras_cv()[[0, 3, 5, 1, 2, 4, 7, 8, 9, 12, 15, 20]]).float().view(4, 3, 4, 4)
    poses = D.normalise_c2w(c2w_cv).flatten(-2)
    with torch.no_grad():
        official = torch.cat([encoder(rgb[i], poses[i]) for i in range(4)])  # PixelEncoder.forward, one scene each
        torch.testing.assert_close(M.encode_batch(encoder, rgb, poses), official)
        M.save_export(tmp_path / "e.pt", encoder, 0, 3)
        backbone = SNeRLFrameEncoder({"vision": {"export_path": str(tmp_path / "e.pt")}})
        assert backbone.observation == {"views": 3, "depth": False, "cameras": True}
        torch.testing.assert_close(backbone({"rgb": rgb, "c2w": c2w_cv, "K": torch.eye(3).expand(4, 3, 3, 3)}), official)
        perm = backbone({"rgb": rgb[:, [2, 0, 1]], "c2w": c2w_cv[:, [2, 0, 1]]})
        torch.testing.assert_close(perm, official, atol=1e-5, rtol=1e-5)  # permutation-invariant over views
        with pytest.raises(ValueError, match="views"):
            backbone({"rgb": rgb[:, :2], "c2w": c2w_cv[:, :2]})


def test_dino_teacher_is_the_official_extractor_with_the_official_weights():
    """Official weights (sha256), the ViT-S/8 from the pinned hub code with stride 4, our ``dino_target`` against
    the verbatim official lines and against an independent layer-11 key computation."""
    path = U.dino_weights_path()
    assert U.sha256_file(path) == U.DINO_WEIGHTS_SHA256
    extractor = U.dino_extractor("cpu")
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert all(torch.equal(v, extractor.model.state_dict()[k]) for k, v in state.items())
    assert extractor.model.patch_embed.proj.stride == (4, 4) and extractor.p == 8
    torch.manual_seed(0)
    target = torch.rand(SIZE, SIZE, 3).cumsum(0).cumsum(1) / (SIZE * SIZE * 0.25) % 1.0  # smooth structured image
    ours = M.dino_target(extractor, target)
    ns = {"torch": torch, "F": F, "target": target, "dino_extractor": extractor, "device": torch.device("cpu")}
    exec(official_lines(711, 715, "with torch.no_grad():", "F.interpolate"), ns)
    assert torch.equal(ours, ns["target_dino"]) and ours.shape == (SIZE, SIZE, 384)
    with torch.no_grad():  # independent: keys of block 11's attention input, heads interleaved as d*h + h
        vit = extractor.model
        x = vit.prepare_tokens(((target - 0.45) / 0.225).permute(2, 0, 1)[None])
        for block in vit.blocks[:11]:
            x = block(x)
        qkv = vit.blocks[11].attn.qkv(vit.blocks[11].norm1(x)).reshape(1, x.shape[1], 3, 6, 64)
        keys = qkv[0, 1:, 1].permute(0, 2, 1).reshape(31, 31, 384)  # (tokens, d, heads) -> 384
        expected = F.interpolate(keys.permute(2, 0, 1)[None], (SIZE, SIZE), mode="bilinear")[0].permute(1, 2, 0)
    torch.testing.assert_close(ours, expected, atol=1e-4, rtol=1e-4)


def official_namespace(model: M.SNeRL, extractor, scene: dict, ds: D.SNeRLScenes, step: int) -> dict:
    """Variables of the official loop (run_nerf.py:687-768) for one of our scenes."""
    run_nerf, helpers = U.official("run_nerf", "run_nerf_helpers").values()
    local_images = np.zeros((6, SIZE, SIZE, 3), np.float32)
    local_images[scene["inputs"].numpy()] = scene["obs_rgb"].numpy()
    local_images[scene["target"]] = scene["target_rgb"].numpy()
    local_semantics = np.zeros((6, SIZE, SIZE), np.float32)
    local_semantics[scene["target"]] = scene["target_semantic"].numpy()
    return {
        "np": np, "torch": torch, "nn": nn, "F": F, "device": torch.device("cpu"), "args": model.args,
        "latent_embed": model.encoder, "dino_extractor": extractor, "optimizer": model.optimizer,
        "local_images": local_images, "local_poses": ds.c2w.numpy(), "local_semantics": local_semantics,
        "obs_index": scene["inputs"].numpy(), "img_i": scene["target"], "i": step + 1, "start": 1,
        "N_rand": model.args.N_rand, "H": SIZE, "W": SIZE, "K": ds.K[scene["target"]],
        "get_rays": run_nerf.get_rays, "render": run_nerf.render, "render_kwargs_train": model.render_kwargs_train,
        "img2mse": helpers.img2mse, "mse2psnr": helpers.mse2psnr,
    }


@pytest.mark.parametrize("step", [0, 5])  # 0: central-crop rays (i = 1 < precrop_iters = 2); 5: full image
def test_training_step_is_the_official_loop_body(tmp_path, step):
    path, manifest = write_task_file(tmp_path)
    ds = D.SNeRLScenes(path, split_episodes(manifest, "train"), seed=0)
    raw = ds[3]
    scene = M.to_device_scene(raw, torch.device("cpu"))
    model = tiny_model()
    extractor = U.dino_extractor("cpu", pretrained=False)  # the loop is compared, not the teacher weights
    torch.manual_seed(1)
    np.random.seed(2)
    ours, terms = M.training_step(model, extractor, scene, step, np.random)  # np.random: the official ray RNG
    ours.backward()
    grads = [p.grad.clone() for p in model.encoder.parameters()]
    torch.manual_seed(1)
    np.random.seed(2)
    ns = official_namespace(model, extractor, scene, ds, step)
    for block in [(697, 703, "obs_img = local_images", "latent = latent_embed"),
                  (707, 718, "target = local_images", "target_semantic = torch.Tensor"),
                  (720, 745, "if N_rand is not None:", "target_dino_s ="),
                  (749, 751, "rgb, disp, acc, semantic, dino, extras = render", "**render_kwargs_train)"),
                  (753, 768, "optimizer.zero_grad()", "psnr0 = mse2psnr")]:
        exec(official_lines(*block), ns)
    assert torch.equal(ours, ns["loss"]) and torch.equal(terms["psnr0"], ns["psnr0"])
    assert ns["coords"].shape[0] == ((SIZE // 2) ** 2 if step == 0 else SIZE * SIZE)  # precrop_frac 0.5
    for key in ("img_loss", "semantic_loss", "dino_loss", "img_loss0", "semantic_loss0"):
        assert torch.equal(terms[key], ns[key]), key
    ns["loss"].backward()
    assert all(torch.equal(a, p.grad) for a, p in zip(grads, model.encoder.parameters()))
    assert all(math.isfinite(float(v)) for v in terms.values())


def test_official_compositing_matches_the_volume_rendering_equation():
    raw2outputs = U.official("run_nerf")["run_nerf"].raw2outputs
    gen = torch.Generator().manual_seed(0)
    raw = torch.randn(5, 7, 4 + 5 + 384, generator=gen, requires_grad=True)
    z_vals = torch.sort(2 + 4 * torch.rand(5, 7, generator=gen), -1).values
    rays_d = torch.randn(5, 3, generator=gen)
    rgb_map, disp_map, acc_map, weights, depth_map, semantic_map, dino_map = raw2outputs(raw, z_vals, rays_d)
    r, z, d = raw.detach().double().numpy(), z_vals.double().numpy(), rays_d.double().numpy()
    dists = np.concatenate([np.diff(z, axis=-1), np.full((5, 1), 1e10)], -1) * np.linalg.norm(d, axis=-1)[:, None]
    alpha = 1 - np.exp(-np.maximum(r[..., 3], 0) * dists)
    trans = np.cumprod(np.concatenate([np.ones((5, 1)), 1 - alpha + 1e-10], -1), -1)[:, :-1]
    w = alpha * trans
    sigmoid = 1 / (1 + np.exp(-r[..., :3]))
    for got, want in [(weights, w), (rgb_map, (w[..., None] * sigmoid).sum(1)), (acc_map, w.sum(1)),
                      (depth_map, (w * z).sum(1)), (disp_map, 1 / np.maximum(1e-10, (w * z).sum(1) / w.sum(1))),
                      (semantic_map, (w[..., None] * r[..., 4:9]).sum(1)), (dino_map, (w[..., None] * r[..., 9:]).sum(1))]:
        np.testing.assert_allclose(got.detach().double().numpy(), want, rtol=1e-5, atol=1e-6)
    (semantic_map.sum() + dino_map.sum()).backward()
    assert torch.count_nonzero(raw.grad[..., 3]) == 0, "semantic/DINO renders use detached weights"
    assert torch.count_nonzero(raw.grad[..., 4:]) > 0


def test_render_camera_is_the_official_render_path(tmp_path):
    model = tiny_model(live=True)
    torch.manual_seed(3)
    rgb, c2w_cv = torch.rand(3, SIZE, SIZE, 3), torch.from_numpy(camera_rig(SIZE, SIZE)["c2w"][[0, 2, 4]])
    with torch.no_grad():
        latent = M.encode_views(model, rgb, c2w_cv)
        reference = model.encoder(rgb.permute(0, 3, 1, 2), D.normalise_c2w(c2w_cv.double()).float().reshape(-1, 16))
    assert torch.equal(latent, reference) and latent.shape == (1, 63)
    small = camera_rig(32, 32)
    for v in (1, 7):  # a training and a held-out camera
        out = M.render_camera(model, latent, small["K"][v], small["c2w"][v], 32, 32, chunk=300)
        run_nerf = U.official("run_nerf")["run_nerf"]
        pose = D.normalise_c2w(torch.from_numpy(small["c2w"][v]).double()).float()
        with torch.no_grad():  # as the official calls (run_nerf.py:802-803, 834-835)
            rgbs, disps = run_nerf.render_path(pose[None], [32, 32, float(small["K"][v][0, 0])],
                                               D.official_K(small["K"][v]), 1024, model.render_kwargs_test, latent=latent)
        np.testing.assert_allclose(out["rgb"].numpy(), rgbs[0], rtol=1e-5, atol=1e-6)  # chunking changes nothing
        np.testing.assert_allclose(out["depth"].numpy(), 1.0 / disps[0] / D.POSE_SCALE, rtol=1e-5)
        assert out["semantic"].shape == (32, 32, 5) and out["dino"].shape == (32, 32, 384)
        assert out["depth"].min() >= 0.5 - 1e-6 and out["depth"].max() <= 1.5 + 1e-6, "inside near/far"
    assert model.render_kwargs_test["perturb"] is False and model.render_kwargs_test["raw_noise_std"] == 0.0


def test_gpu_rendering_under_the_default_type_context_matches_cpu_and_restores_it():
    cpu, gpu = tiny_model("cpu", live=True), tiny_model("cuda")
    for name in ("network_fn", "network_fine"):
        gpu.render_kwargs_test[name].load_state_dict(cpu.render_kwargs_test[name].state_dict())
    latent = torch.randn(1, 63, generator=torch.Generator().manual_seed(0))
    small = camera_rig(32, 32)
    a = M.render_camera(cpu, latent, small["K"][2], small["c2w"][2], 32, 32)
    b = M.render_camera(gpu, latent.cuda(), small["K"][2], small["c2w"][2], 32, 32)
    for key in ("rgb", "depth", "semantic"):
        torch.testing.assert_close(b[key].cpu(), a[key], atol=2e-4, rtol=2e-4)
    assert b["rgb"].is_cuda and torch.Tensor().type() == "torch.FloatTensor" and torch.ones(1).device.type == "cpu"


def test_dead_density_initialisations_are_detected(tmp_path):
    path, manifest = write_task_file(tmp_path)
    scene = M.to_device_scene(D.SNeRLScenes(path, split_episodes(manifest, "train"), seed=0)[0], torch.device("cpu"))
    model = tiny_model()
    with torch.no_grad():
        model.render_kwargs_train["network_fine"].alpha_linear.bias.fill_(-1e4)
        model.render_kwargs_train["network_fn"].alpha_linear.bias.fill_(1e4)
    assert M.live_density_fractions(model, scene) == (1.0, 0.0)


# ---------------------------------------------------------------------------------- export, sizes, RL backbone
def test_export_reload_identity_and_random_init_flag(tmp_path):
    model = tiny_model()
    M.save_export(tmp_path / "init.pt", model.encoder, 0, 3, random_init=True)
    payload, loaded = M.load_export(tmp_path / "init.pt")
    assert payload["random_init"] and payload["num_views"] == 3 and payload["frame_spacing"] is None
    assert payload["pose"]["lookat"] == list(LOOKAT) and payload["pose"]["scale"] == D.POSE_SCALE
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.encoder.state_dict().items())
    assert not any(p.requires_grad for p in loaded.parameters()) and not loaded.training
    rgb, poses = torch.rand(2, 3, 3, SIZE, SIZE), torch.randn(2, 3, 16)
    with torch.no_grad():
        assert torch.equal(M.encode_batch(loaded, rgb, poses), M.encode_batch(model.encoder.eval(), rgb, poses))
    torch.save({**payload, "format": "other"}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="not a SNeRL encoder export"):
        M.load_export(tmp_path / "bad.pt")


def test_rl_time_encoder_size_and_flops_per_frame(tmp_path):
    """One RL observation at one time step: 3 views of 128^2 RGB + 3 cameras -> 63 floats. Every parameter is reached;
    FLOPs (multiply-add = 2) on CPU with FlopCounterMode, as for our other encoders."""
    from torch.utils.flop_counter import FlopCounterMode

    from s4d.baselines.snerl.encoder import SNeRLFrameEncoder

    M.save_export(tmp_path / "e.pt", M.build_encoder(), 0, 3)
    backbone = SNeRLFrameEncoder({"vision": {"export_path": str(tmp_path / "e.pt")}}).requires_grad_(True)
    obs = {"rgb": torch.rand(1, 3, 3, SIZE, SIZE), "c2w": torch.from_numpy(camera_rig(SIZE, SIZE)["c2w"][None, :3])}
    with FlopCounterMode(display=False) as counter:
        out = backbone(obs)
    assert out.shape == (1, 63)
    (out * torch.randn_like(out)).sum().backward()  # the sum of a LayerNorm output is constant
    reached = sum(p.numel() for p in backbone.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert reached == sum(p.numel() for p in backbone.parameters()) == 6_587_885
    convs = 63 * 63 * 32 * 27 + (61 * 61 + 59 * 59 + 57 * 57) * 32 * 288
    macs = 3 * (convs + 32 * 57 * 57 * 63) + 3 * 79 * 63 + 63 * 63
    assert counter.get_total_flops() == 2 * macs == 637_811_496


def drm_cfg(export: Path) -> dict:
    base = yaml.safe_load((REPO / "configs/rl/base.yaml").read_text())
    enc = yaml.safe_load((REPO / "configs/rl/encoders/snerl.yaml").read_text())
    return {
        "env": {"frame_stack": 3, "image_size": SIZE, **enc["env"]},
        "vision": {**enc["vision"], "export_path": str(export)},
        "agent": {**base["agent"], "feature_dim": 8, "hidden_dim": 16, "dormant_perturb_interval": 4},
    }


def test_drm_end_to_end_with_the_frozen_snerl_encoder(tmp_path):
    from s4d.rl.agent import DrMAgent
    from s4d.rl.evaluate import policy_inputs
    from s4d.rl.replay import Replay, replay_iterator

    device = torch.device("cuda")
    torch.manual_seed(0)
    M.save_export(tmp_path / "export.pt", M.build_encoder(), 0, 3)
    agent = DrMAgent(drm_cfg(tmp_path / "export.pt"), 4, 4, device)
    enc = agent.encoder
    assert not agent.augment_pixels and enc.replay_atom_is_feature and not enc.replay_atom_is_stack_feature
    assert enc.replay_atom_shape == (63,) and enc.observation == {"views": 3, "depth": False, "cameras": True}
    assert not any(p.requires_grad for p in enc.backbone.parameters())
    rig = camera_rig(SIZE, SIZE)
    rng = np.random.default_rng(0)

    def observation(n: int) -> dict:
        views = np.stack([rng.choice(6, 3, replace=False) for _ in range(n)])
        return {"rgb": rng.integers(0, 255, (n, 3, 3, 3, SIZE, SIZE), dtype=np.uint8),
                "K": np.broadcast_to(rig["K"][views][:, None], (n, 3, 3, 3, 3)).copy(),
                "c2w": np.broadcast_to(rig["c2w"][views][:, None], (n, 3, 3, 4, 4)).copy()}

    obs = observation(2)
    features = policy_inputs(agent, obs)
    assert features.shape == (2, 3, 63) and features.dtype == torch.float16
    frame = {k: torch.as_tensor(v[1:2, 2], device=device).float() for k, v in obs.items()}
    frame["rgb"] = frame["rgb"] / 255.0
    torch.testing.assert_close(features[1, 2].float(), enc.backbone(frame)[0], atol=2e-3, rtol=1e-3)
    replay = Replay(None, enc.replay_atom_shape, np.float16, 4, 4, 1000, enc.replay_atom_frame_stack, 10)
    for _ in range(3):
        replay.add_initial(policy_inputs(agent, observation(1))[0][-1].cpu().numpy(), rng.normal(size=4))
        for t in range(12):
            atom = policy_inputs(agent, observation(1))[0][-1].cpu().numpy()
            replay.add(rng.uniform(-1, 1, 4), rng.normal(), 1.0, atom, rng.normal(size=4), t == 11)
    frozen = copy.deepcopy(enc.backbone.state_dict())
    batches = replay_iterator(replay, 16, 0.97, seed=0, device=device)
    for step in range(2, 8, 2):
        metrics = agent.update(batches, step)
        assert all(math.isfinite(v) for v in metrics.values()), metrics
    assert all(torch.equal(v, frozen[k]) for k, v in enc.backbone.state_dict().items()), "encoder frozen"
    assert agent.act(policy_inputs(agent, observation(2)), np.zeros((2, 4)), 5000, True).shape == (2, 4)


# ------------------------------------------------------------------------------------------- trainer and resume
def run_trainer(config: Path, root: Path, name: str, *overrides: str) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]}
    cmd = [sys.executable, "-I", str(REPO / "scripts/train_snerl.py"), "--config", str(config), "--name", name,
           "--output-root", str(root), "--no-wandb", "--set", *overrides]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]


def test_trainer_runs_exports_and_resumes_losslessly(tmp_path):
    path, manifest = write_task_file(tmp_path)
    cfg = yaml.safe_load((REPO / "configs/baselines/snerl/hammer.yaml").read_text())
    cfg["dataset"].update(hdf5_path=str(path), num_workers=1)
    cfg["data_interface"]["split_manifest"] = str(manifest)
    cfg["nerf"].update(TINY_NERF)
    cfg["train"].update(device="cpu", max_steps=4, i_print=1, eval_every=2, save_every=2, eval_scenes=1,
                        render_chunk=4096)
    cfg["dino"]["pretrained"] = False
    config = tmp_path / "snerl.yaml"
    config.write_text(yaml.safe_dump(cfg))
    root = tmp_path / "runs"  # conftest keeps tmp_path inside the repository, as the script requires
    run_trainer(config, root, "straight")
    run_trainer(config, root, "resumed", "train.max_steps=2")
    run_trainer(config, root, "resumed")  # --resume auto from step 2
    straight, resumed = root / "straight", root / "resumed"
    for run in (straight, resumed):
        assert json.loads((run / "completion.json").read_text())["steps"] == 4
        assert (run / "eval" / "step_00000002.png").is_file()
        init = torch.load(run / "encoder_init.pt", map_location="cpu", weights_only=True)
        assert init["random_init"] and init["step"] == 0
    a, b = (torch.load(r / "checkpoints/latest.pt", map_location="cpu", weights_only=False) for r in (straight, resumed))
    for part in ("latent_embed_state_dict", "network_fn_state_dict", "network_fine_state_dict"):
        assert all(torch.equal(v, b["model"][part][k]) for k, v in a["model"][part].items()), part
    rows = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    rows_a = [json.loads(line) for line in (straight / "metrics.jsonl").read_text().splitlines()]
    loss = lambda rs: {r["step"]: r["train/loss"] for r in rs if "train/loss" in r}  # noqa: E731
    assert loss(rows) == loss(rows_a) and set(loss(rows)) == {0, 1, 2, 3}
    export = torch.load(resumed / "encoder.pt", map_location="cpu", weights_only=True)
    assert export["step"] == 4 and not export["random_init"]
    attempts = json.loads((straight / "init.json").read_text())["attempts"]
    assert min(attempts[-1]["live_coarse"], attempts[-1]["live_fine"]) > 0
