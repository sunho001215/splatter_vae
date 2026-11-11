"""Our method on REAL compact-store episodes (calibration release calrel-6c3b9ec): gate evidence, windows, geometry of
depth / tracks / gripper points under the released calibration, and every loss on a real batch with the full Original
model. Defined only where a verified local sample exists (``S4D_OURS_RW_SAMPLE``, default the main checkout's
``.cache/realworld-sample-oursrw``: episodes + learning inputs fetched byte-identically from the remote, sha256 in
``REMOTE_SHA256.json``), as the other real-data tests; the GPU test additionally needs CUDA."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from s4d.data.contract import collate, validate_batch
from s4d.rwlearn import ours

SAMPLE = Path(os.environ.get("S4D_OURS_RW_SAMPLE", "/home/ws/ws/splatter4d/.cache/realworld-sample-oursrw"))
RELEASE = "calrel-6c3b9ec"
EPISODES = {"drawer-open": ("000078", "000017", "000007"), "mug-hanging": ("000075", "000064")}


def _episode_files(task):
    return [SAMPLE / f"processed/{RELEASE}/{task}/{task}-{e}/train_data/{task}-{e}/episode.h5" for e in EPISODES[task]]


def _available() -> bool:
    manifest = SAMPLE / "REMOTE_SHA256.json"
    if not manifest.is_file():
        return False
    want = json.loads(manifest.read_text())
    return all((SAMPLE / rel).is_file() for rel in want)


def _inputs(task, name):
    return SAMPLE / f"learning_inputs/{RELEASE}/{task}/{name}"


def _project(points, K, w2c):
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    uv = cam[:, :2] / cam[:, 2:3] @ K[:2, :2].T + K[:2, 2]
    return uv, cam[:, 2]


def _dataset(task, tmp_path, **kw):
    split = json.loads(_inputs(task, "split_seed0.json").read_text())
    store_yaml = tmp_path / "store.yaml"
    store_yaml.write_text(yaml.safe_dump({"kind": "compact", "paths": [str(p.parent) for p in _episode_files(task)]}))
    store = ours.open_compact_store(store_yaml)
    eps = store.episodes()
    train = [e for e in eps if e in split["train"]] or eps
    return ours._windows_class()(store, episodes=train, split="all", strides=(1, 2, 3), **kw), store, split


if _available():

    def test_sample_is_byte_identical_to_the_remote_release():
        want = json.loads((SAMPLE / "REMOTE_SHA256.json").read_text())
        for rel, digest in want.items():
            assert hashlib.sha256((SAMPLE / rel).read_bytes()).hexdigest() == digest, rel

    @pytest.mark.parametrize("task", sorted(EPISODES))
    def test_released_gate_admits_ours_and_binds_the_sample_calibration(task, tmp_path):
        from s4d.rwlearn import gate

        report = json.loads(_inputs(task, "gate_report.json").read_text())
        assert gate.task_passed(report, task, "ours"), gate.task_problems(report, task, "ours")
        _ds, store, split = _dataset(task, tmp_path)
        eps = store.episodes()
        assert set(eps) <= set(split["train"]) | set(split["validation"])
        # every sample episode was built from its validated calibration, only validated cameras
        gate.check_store_calibration(report, {task: eps}, store.upstream, store.cameras)

    @pytest.mark.parametrize("task", sorted(EPISODES))
    def test_real_windows_satisfy_the_contract(task, tmp_path):
        ds, store, _ = _dataset(task, tmp_path)
        strides = {ds.window(i)["stride"] for i in range(len(ds))}
        assert strides == {1, 2, 3} and len(ds) > 100
        idx = np.linspace(0, len(ds) - 1, 6).round().astype(int)
        batch = collate([ds[int(i)] for i in idx])
        validate_batch(batch)
        assert batch["images"].shape[1:4] == (3, 3, 3) and batch["images"].shape[-1] == 128
        assert (batch["track_count"] > 0).all()
        assert float((batch["depth"] > 0).float().mean()) > 0.8  # FoundationStereo depth: dense
        assert float(batch["depth"][batch["depth"] > 0].median()) < 2.0
        assert all(0.06 < dt < 0.075 for dt in batch["meta"]["dt_seconds"])  # 15 Hz cameras

    @pytest.mark.parametrize("task", sorted(EPISODES))
    def test_tracks_and_gripper_points_agree_with_depth_under_the_released_calibration(task, tmp_path):
        """Base-frame track positions (lifted at native resolution) re-projected with the stored 128 px K and the
        released w2c land on stored depth of the same surface; gripper FK points that the store marks visible do too."""
        ds, store, _ = _dataset(task, tmp_path)
        errs, gerrs, inside = [], [], []
        for i in np.linspace(0, len(ds) - 1, 12).round().astype(int):
            s = ds[int(i)]
            for v in range(s["images"].shape[1]):
                K, w2c, depth = s["K"][v].numpy(), s["w2c"][v].numpy(), s["depth"][0, v, 0].numpy()
                n = int(s["track_count"][v])
                pts = s["track_src"][0, v, :n].numpy()
                ok = s["track_weight"][0, v, :n].numpy() > 0
                uv, z = _project(pts[ok], K, w2c)
                px = np.floor(uv).astype(int)
                good = (px >= 0).all(1) & (px < 128).all(1) & (z > 0)
                inside.append(good.mean() if len(good) else 1.0)
                d = depth[px[good, 1], px[good, 0]]
                errs.append(np.abs(d - z[good])[d > 0])
                gw = s["gripper_weight"][0, v].numpy() > 0
                if gw.any():
                    guv, gz = _project(s["gripper_points"][0].numpy()[gw], K, w2c)
                    gpx = np.floor(guv).astype(int)
                    gin = (gpx >= 0).all(1) & (gpx < 128).all(1)
                    gd = depth[gpx[gin, 1], gpx[gin, 0]]
                    gerrs.append(np.abs(gd - gz[gin])[gd > 0])
        err = np.concatenate(errs)
        assert min(inside) == 1.0  # supervised track sources are inside the stored image by construction
        assert len(err) > 500 and np.median(err) < 0.02 and np.mean(err < 0.05) > 0.8, np.percentile(err, [50, 80, 95])
        gerr = np.concatenate(gerrs)
        assert len(gerr) > 20 and np.median(gerr) < 0.03, np.percentile(gerr, [50, 80, 95])

    @pytest.mark.parametrize("task", sorted(EPISODES))
    def test_cross_camera_depth_is_consistent_under_the_released_calibration(task, tmp_path):
        """Depth of one camera lifted to the base frame and projected into another camera of the same M5 set agrees
        with that camera's depth where the surface is visible to both (front-most points)."""
        from s4d.geometry import lift_depth

        ds, store, _ = _dataset(task, tmp_path)
        agree = []
        for i in np.linspace(0, len(ds) - 1, 8).round().astype(int):
            s = ds[int(i)]
            V = s["images"].shape[1]
            for a in range(V):
                xyz = lift_depth(s["depth"][0, a, 0][None], s["K"][a][None], s["c2w"][a][None])[0].reshape(-1, 3)
                valid = (s["depth"][0, a, 0].reshape(-1) > 0).numpy()
                pts = xyz.numpy()[valid]
                for b in range(V):
                    if b == a:
                        continue
                    uv, z = _project(pts, s["K"][b].numpy(), s["w2c"][b].numpy())
                    px = np.floor(uv).astype(int)
                    ok = (px >= 0).all(1) & (px < 128).all(1) & (z > 0)
                    db = s["depth"][0, b, 0].numpy()[px[ok, 1], px[ok, 0]]
                    zb = z[ok]
                    front = (db > 0) & (zb < 1.5) & (zb <= db + 0.03)  # not occluded in b (workspace depth range)
                    agree.append(np.abs(db[front] - zb[front]) < 0.03)
        frac = float(np.concatenate(agree).mean())
        assert frac > 0.7, frac

    if torch.cuda.is_available():

        @pytest.mark.gpu
        @pytest.mark.parametrize("task", sorted(EPISODES))
        def test_every_original_loss_is_finite_on_a_real_batch_with_the_full_model(task, tmp_path):
            from s4d.train.loop import build_model, forward_losses, move_batch

            ds, _store, _ = _dataset(task, tmp_path)
            cfg = yaml.safe_load(ours.render(task))
            torch.manual_seed(0)
            model = build_model(cfg).cuda().train()
            idx = np.linspace(0, len(ds) - 1, 4).round().astype(int)
            batch = move_batch(collate([ds[int(i)] for i in idx]), torch.device("cuda"))
            out = forward_losses(model, batch, cfg, 25000, return_renders=True)
            out["total"].backward()
            assert torch.isfinite(out["total"])
            for key in ("render", "motion", "infonce", "consistency", "visibility", "synth_render", "depth_grad_t0",
                        "depth_hard_t0", "rgb_t0", "coverage_t0", "depth_l1_t0"):
                assert torch.isfinite(out["losses"][key]), key
            assert out["rgb"].shape == batch["images"].shape[:3] + (3, 128, 128)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
