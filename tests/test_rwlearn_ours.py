"""Our method (Original Splatter4D) on the real-world compact store: configs, data path, gate, losses (synthetic)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from s4d.data.contract import collate, validate_batch
from s4d.rwlearn import ours
from tests.test_realworld_train_store import built, pipeline_named_store  # noqa: F401  (module fixtures)

REPO = Path(__file__).resolve().parents[1]


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


# ----------------------------------------------------------------------------------------------- configs
@pytest.mark.parametrize("target", ours.TARGETS)
def test_committed_configs_equal_the_generator(target):
    assert ours.config_path(target).read_text() == ours.render(target)


@pytest.mark.parametrize("target", ours.TARGETS)
def test_config_differs_from_the_simulation_original_in_exactly_the_listed_keys(target):
    sim = _flat(ours.sim_original())
    real = _flat(yaml.safe_load(ours.config_path(target).read_text()))
    listed = {key: value for key, value, _ in ours.overrides(target)}
    diff = {k for k in set(sim) | set(real) if sim.get(k, "<absent>") != real.get(k, "<absent>")}
    # a listed key whose value equals the simulation value (none today) would simply not appear in the diff
    assert diff <= set(listed) and {k for k, v in listed.items() if sim.get(k, "<absent>") != v} == diff
    for key, value in listed.items():
        assert real.get(key, "<absent>") == ("<absent>" if value is ours.DELETE else value), key
    # the method itself is untouched: model, losses, optimiser/schedule/budget, render planes, synthetic views
    for section in ("model.encoder.", "model.decoder.", "loss.", "train.", "render."):
        assert {k: v for k, v in sim.items() if k.startswith(section)} == \
               {k: v for k, v in real.items() if k.startswith(section)}, section
    assert real["aug.synth_views"] == 2 and real["aug.synth_render"] is True


def test_simulation_original_is_the_arm4_recommended_configuration():
    cfg = ours.sim_original()
    enc, loss = cfg["model"]["encoder"], cfg["loss"]
    assert (enc["backbone"], enc["patch_size"], enc["width"], enc["depth"], enc["heads"], enc["drop_path"]) == \
        ("scratch", 16, 128, 8, 8, 0.1)
    assert (enc["num_slots"], enc["slot_dim"], enc["num_frames"], enc["image_height"]) == (1, 256, 3, 128)
    assert (enc["mask_ratio"], enc["motion_patch_threshold"]) == (0.5, 0.05) and "mask_mode" not in enc  # tube
    assert {k: loss[k] for k in ("rgb", "ssim_weight", "lambda_dyn", "coverage", "depth_l1", "depth_grad",
                                 "depth_hard", "visibility", "motion", "motion_huber", "invariance", "consistency",
                                 "temperature")} == {
        "rgb": 1.0, "ssim_weight": 0.2, "lambda_dyn": 1.0, "coverage": 0.1, "depth_l1": 1.0, "depth_grad": 0.5,
        "depth_hard": 0.5, "visibility": 1.0, "motion": 5.0, "motion_huber": 0.01, "invariance": 1.0,
        "consistency": 0.5, "temperature": 0.1}
    assert "chamfer" not in loss and loss["motion_space"] == "gaussian" and loss["depth_valid_far"] is True
    assert (cfg["train"]["steps"], cfg["train"]["seed"], cfg["train"]["batch_size"]) == (200000, 0, 16)


def test_simulation_original_matches_the_registered_reference_runs():
    """The resolved stack equals the method-frozen-v1 reference (s3-m3d-seed0-door-open-200k) up to data.task and
    model.anchor_stats (the per-task fields every real-world target replaces)."""
    ref = REPO / "tests/fixtures/ours/s3-m3d-seed0-door-open-200k.config.yaml"  # copied from the run (226d8ae)
    registered = _flat(yaml.safe_load(ref.read_text()))
    sim = _flat(ours.sim_original())
    keys = {k for k in set(registered) | set(sim) if not k.startswith("run.")} - {"data.task", "model.anchor_stats"}
    assert {k: registered.get(k) for k in keys} == {k: sim.get(k) for k in keys}


def test_metaworld_training_modules_never_import_the_realworld_learning_package():
    code = ("import sys; import s4d.train.loop, s4d.train.evaluate, s4d.data.metaworld.dataset; "
            "assert not [m for m in sys.modules if m.startswith('s4d.rwlearn')]")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


# ----------------------------------------------------------------------------------------------- data path
def _cfg(tmp_path, paths, split, *, gate_report=None, steps=200000, smoke=False, target="drawer-open"):
    store = tmp_path / "store.yaml"
    store.write_text(yaml.safe_dump({"kind": "compact", "paths": [str(p) for p in paths]}))
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    cfg = yaml.safe_load(ours.render("drawer-open"))
    cfg["data"].update(store=str(store), split_manifest=str(split_path), target=target,
                       gate_report=None if gate_report is None else str(gate_report))
    if smoke:
        cfg["data"]["smoke_without_calibration"] = True
    cfg["train"]["steps"] = steps
    return cfg


def _loader(dataset, shuffle):
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=2, shuffle=shuffle, collate_fn=collate, num_workers=0)



def test_full_gate_path_builds_train_validation_and_probe_loaders(pipeline_named_store, tmp_path):
    store, report, paths = pipeline_named_store
    eps = store.episodes()
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(report))
    cfg = _cfg(tmp_path, paths, {"train": [eps[0]], "validation": [eps[1]]}, gate_report=gate_path)
    with pytest.warns(UserWarning, match="MOCK"):
        train_ds, val, probe, n_cams = ours.build_realworld(cfg, _loader)
    assert train_ds.gate["smoke"] is False and train_ds.gate["geometry_dependent"] is True
    assert n_cams == 2 and set(val) == set(probe) == {1, 3}
    assert {train_ds.window(i)["episode"] for i in range(len(train_ds))} == {eps[0]}
    assert {val[1].dataset.window(i)["episode"] for i in range(len(val[1].dataset))} == {eps[1]}
    assert {train_ds.window(i)["stride"] for i in range(len(train_ds))} == {1, 2, 3}
    sample = train_ds[0]
    raw = super(type(train_ds), train_ds).__getitem__(0)
    assert torch.equal(sample["probe_state"], raw["probe_state"][:, [0, 1, 2, 6, 3, 4, 5]])
    assert 0.01 < sample["meta"]["dt_seconds"] < 1.0
    batch = next(iter(val[1]))
    validate_batch(batch)
    assert "track_src" in batch and "motion3d" not in batch


def test_gate_refusals(pipeline_named_store, tmp_path):
    store, report, paths = pipeline_named_store
    eps = store.episodes()
    split = {"train": [eps[0]], "validation": [eps[1]]}
    with pytest.raises(RuntimeError, match="no passed full-data gate"):  # long run without evidence
        ours.build_realworld(_cfg(tmp_path, paths, split), _loader)
    with pytest.raises(RuntimeError, match="calibration hold"):  # geometry consumer: short run needs the opt-in
        ours.build_realworld(_cfg(tmp_path, paths, split, steps=100), _loader)
    cfg = _cfg(tmp_path, paths, split, steps=100, smoke=True)
    cfg["data"]["consumer"] = "reviwo"
    with pytest.raises(ValueError, match="conflicts with the trainer's consumer"):
        ours.build_realworld(cfg, _loader)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="target"):  # the store's task is drawer-open
        ours.build_realworld(_cfg(tmp_path, paths, split, gate_report=gate_path, target="mug-hanging"), _loader)
    with pytest.raises(RuntimeError, match="partition"):
        ours.build_realworld(_cfg(tmp_path, paths, {"train": [eps[0]], "validation": [eps[0]]},
                                  gate_report=gate_path), _loader)
    native = tmp_path / "native.yaml"
    native.write_text(yaml.safe_dump({"kind": "compact", "paths": [str(p) for p in paths],
                                      "depth_source": "native_realsense"}))
    with pytest.raises(ValueError, match="FoundationStereo"):
        ours.open_compact_store(native)


def test_samples_carry_each_episodes_own_calibration(pipeline_named_store, tmp_path):
    """Per-episode (e.g. covering's per-period) calibration: every sample uses its episode's stored w2c."""
    import h5py
    import numpy as np

    store, _report, paths = pipeline_named_store
    store.close()
    cam = None
    with h5py.File(paths[1], "r+") as f:
        cam = sorted(f["cameras"])[0]
        w2c = f[f"cameras/{cam}/w2c"][:]
        shift = np.eye(4)
        shift[0, 3] = 0.01  # a different (period) calibration: 1 cm
        del f[f"cameras/{cam}/w2c"]
        f[f"cameras/{cam}/w2c"] = shift @ w2c
    eps = sorted(store.episodes())
    cfg = _cfg(tmp_path, paths, {"train": [eps[0]], "validation": [eps[1]]}, steps=100, smoke=True)
    with pytest.warns(UserWarning, match="MOCK"):
        train_ds, val, _probe, _ = ours.build_realworld(cfg, _loader)
    seen = {}
    for ds in (train_ds, val[1].dataset):  # one batch of the run mixes nothing here; each loader keeps its episode's K/w2c
        s = ds[0]
        seen[ds.window(0)["episode"]] = s["w2c"][list(s["meta"]["camera_ids"]).index(cam)]
    assert set(seen) == set(eps)
    assert not torch.allclose(seen[eps[0]], seen[eps[1]])
    assert torch.allclose(seen[eps[1]], torch.as_tensor(shift @ w2c, dtype=torch.float32), atol=1e-6)


def test_workspace_statistics_use_training_episodes_and_record_inputs(pipeline_named_store, tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    store, _report, paths = pipeline_named_store
    eps = store.episodes()
    cfg = _cfg(tmp_path, paths, {"train": [eps[0]], "validation": [eps[1]]}, steps=100, smoke=True)
    import importlib

    stats_mod = importlib.import_module("compute_workspace_stats")
    args = type("A", (), {"split": cfg["data"]["split_manifest"]})()
    with pytest.warns(UserWarning, match="MOCK"):
        ds = stats_mod.dataset_for("realworld", Path(cfg["data"]["store"]), args)
    assert {ds.window(i)["episode"] for i in range(len(ds))} == {eps[0]}
    assert {ds.window(i)["stride"] for i in range(len(ds))} == {3}
    stats = stats_mod.compute_statistics(ds, samples=4, seed=0)
    assert stats["scene"]["num_points"] > 0 and len(stats["scene"]["mean"]) == 3
    with pytest.raises(ValueError, match="--split"):
        stats_mod.dataset_for("realworld", Path(cfg["data"]["store"]), None)


def test_forward_losses_with_the_realworld_config_on_cpu(built, tmp_path, monkeypatch):  # noqa: F811
    """The generated configuration (synthetic views around the training cameras, track-form M3D, every Original loss)
    on a synthetic real-world batch: finite total, gradients, every Original loss term present."""
    from s4d.losses import rgb
    from s4d.model import render
    from s4d.model.decoder import DecoderConfig, GaussianDecoder, GroupConfig
    from s4d.model.encoder import Encoder, EncoderConfig
    from s4d.train.loop import Model, forward_losses
    from tests.test_review_options import _fake_rasterizer

    monkeypatch.setattr(render, "_rasterize", _fake_rasterizer([]))
    monkeypatch.setattr(rgb, "_ssim_map", lambda x, y: 1 - (x - y).square())
    from s4d.realworld.train_windows import RealworldWindowDataset

    with pytest.warns(UserWarning, match="MOCK"):
        ds = RealworldWindowDataset([built["tmp"] / "store"], split="all", strides=(1,))
    batch = collate([ds[0], ds[len(ds) - 1]])
    cfg = yaml.safe_load(ours.render("drawer-open"))
    cfg["train"]["bf16"] = False
    group = GroupConfig(parents=2, children=2, offset_scale=0.2, child_radius=0.05, scale_min=0.001, scale_max=0.08)
    torch.manual_seed(0)
    enc = Encoder(EncoderConfig(image_height=128, image_width=128, patch_size=16, width=16, depth=1, heads=4,
                                num_slots=1, slot_dim=8, drop_path=0, mask_ratio=0.5, motion_patch_threshold=0.05))
    model = Model(enc, GaussianDecoder(DecoderConfig(slot_dim=8, dim=16, depth=1, heads=4, scene=group, dynamic=group,
                                                     num_slots=1))).train()
    out = forward_losses(model, batch, cfg, 30000, source=torch.tensor([0, 1]))
    out["total"].backward()
    assert torch.isfinite(out["total"])
    losses = out["losses"]
    for key in ("render", "motion", "infonce", "consistency", "visibility", "synth_render", "depth_grad_t0",
                "depth_hard_t0", "rgb_t0", "coverage_t0", "depth_l1_t0"):
        assert key in losses and torch.isfinite(losses[key]), key
    assert not any(k.startswith("chamfer") for k in losses)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_window_sampling_rules_stride_uniformity_and_spread_validation_order(pipeline_named_store, tmp_path):
    from collections import Counter

    store, _report, paths = pipeline_named_store
    eps = store.episodes()
    windows = ours._windows_class()
    with pytest.warns(UserWarning, match="MOCK"):
        base = windows.__mro__[1](paths, split="all", strides=(1, 2, 3))
    with pytest.warns(UserWarning, match="MOCK"):
        ds = windows(paths, split="all", strides=(1, 2, 3))
    starts = Counter((w["episode"], w["canonical_indices"][0]) for w in map(ds.window, range(len(ds))))
    assert set(starts.values()) == {3}  # every kept start frame has all three strides: stride exactly uniform
    assert len(ds) == 3 * len(starts) <= len(base)
    per_stride = Counter(ds.window(i)["stride"] for i in range(len(ds)))
    assert len(set(per_stride.values())) == 1
    with pytest.warns(UserWarning, match="MOCK"):
        val = windows(paths, split="all", strides=(1,), spread_episodes=True)
    order = [val.window(i)["episode"] for i in range(len(val))]
    assert set(order[:2]) == set(eps)  # round-robin over episodes from the first windows on
    t0 = [val.window(i)["canonical_indices"][0] for i in range(len(val)) if order[i] == eps[0]]
    assert t0 == sorted(t0)  # evenly over each episode's time
    assert sorted(map(tuple, val.table.tolist())) == sorted(map(tuple, windows(paths, split="all",
                                                                             strides=(1,)).table.tolist()))


def test_split_must_partition_the_store_also_for_smoke_runs(pipeline_named_store, tmp_path):
    store, _report, paths = pipeline_named_store
    eps = store.episodes()
    for split in ({"train": [eps[0]], "validation": [eps[0]]}, {"train": [eps[0]], "validation": []},
                  {"train": [eps[0], eps[0]], "validation": [eps[1]]}):
        with pytest.raises((RuntimeError, ValueError), match="partition|validation episodes"):
            ours.build_realworld(_cfg(tmp_path, paths, split, steps=100, smoke=True), _loader)
