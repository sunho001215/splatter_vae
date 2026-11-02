"""The DROID csw4 profile keeps the accepted csw4 architecture and method objective (user decision 2026-11-02).

configs/droid/csw4_largescale.yaml resolved on its own must equal the resolved Meta-World csw4 composition (base + task +
chamfer_lean + chamfer_lean_randmask + enc_cs + enc_w4) in every method key; every key that differs must be one of the
documented dataset-specific keys below (docs/droid_largescale_validation/csw4_effective_config.md)."""

from __future__ import annotations

from pathlib import Path

import pytest

from s4d.config import load_config

REPO = Path(__file__).resolve().parents[1]
CSW4_VARIANTS = ("chamfer_lean", "chamfer_lean_randmask", "enc_cs", "enc_w4")  # runs/setup/encx-definition-20261026.json
PROFILE = REPO / "configs/droid/csw4_largescale.yaml"

# key -> why it may differ from Meta-World csw4 (dataset geometry, timing, units, image handling, loader, bookkeeping)
DATASET_SPECIFIC = {
    "data.*": "the data regime and its sources (droid_largescale: manifest, PointWorld, RLDS index, RLDS release)",
    "model.encoder.image_height": "image handling: 0.8 x the native 180x320 DROID grid, aspect kept",
    "model.encoder.image_width": "image handling: 0.8 x the native 180x320 DROID grid, aspect kept",
    "model.anchor_stats": "geometry: the dataset's own workspace statistics",
    "render.near": "geometry: dataset depth range",
    "render.far": "geometry: dataset depth range",
    "aug.synth_camera_rig": "geometry: synthetic cameras around the episode's own cameras (not the Meta-World rig)",
    "train.steps": "run scale: fixed at launch",
    "train.num_workers": "loader: measured DROID throughput",
    "train.prefetch_factor": "loader",
    "train.log_every": "bookkeeping cadence",
    "train.eval_every": "bookkeeping cadence",
    "train.save_every": "bookkeeping cadence",
    "eval.orbit_center": "geometry: diagnostic orbit renders in the DROID robot-base frame",
    "eval.orbit_distance": "geometry: diagnostic orbit renders",
    "eval.orbit_elevation": "geometry: diagnostic orbit renders",
    "eval.probe_every": "evaluation protocol (the DROID regime has no probe loaders)",
    "eval.probe_batches": "evaluation protocol (the DROID regime has no probe loaders)",
    "wandb.project": "bookkeeping",
}
TRAIN_METHOD = ("batch_size", "lr", "min_lr", "weight_decay", "warmup_steps", "grad_clip", "bf16", "seed")


def flatten(cfg: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[prefix + key] = value
    return out


def dataset_specific(key: str) -> bool:
    return key in DATASET_SPECIFIC or any(p.endswith(".*") and key.startswith(p[:-1]) for p in DATASET_SPECIFIC)


def metaworld_csw4(task: str = "hammer") -> dict:
    paths = [REPO / "configs/metaworld/base.yaml", REPO / f"configs/metaworld/tasks/{task}.yaml"]
    return load_config(paths + [REPO / f"configs/metaworld/variants/{v}.yaml" for v in CSW4_VARIANTS])


def test_method_keys_equal_metaworld_csw4():
    droid, csw4 = flatten(load_config([PROFILE])), flatten(metaworld_csw4())
    def method(cfg: dict, section: str) -> dict:
        return {k: v for k, v in cfg.items() if k.startswith(section) and not dataset_specific(k)}

    for section in ("model.", "loss.", "aug."):
        assert method(droid, section) == method(csw4, section), section
    for key in TRAIN_METHOD:
        assert droid[f"train.{key}"] == csw4[f"train.{key}"], key
    # the architecture and objective that distinguish csw4 from the DROID base are present explicitly
    assert droid["model.encoder.stem"] == "patch_cnn" and droid["model.encoder.width"] == 256
    assert droid["model.encoder.mask_mode"] == "random" and droid["model.encoder.mask_ratio"] == 0.75
    assert droid["model.decoder.dim"] == 128 and droid["model.decoder.scene.parents"] == 768
    assert droid["model.encoder.num_slots"] == 1 and droid["model.encoder.slot_dim"] == 256
    assert droid["loss.consistency"] == 0.5 and droid["loss.depth_align"] == "none" and droid["loss.chamfer"] == 5.0


@pytest.mark.parametrize("task", ["hammer", "pick-place", "bin-picking"])
def test_every_difference_is_a_documented_dataset_key(task):
    droid, csw4 = flatten(load_config([PROFILE])), flatten(metaworld_csw4(task))
    differ = {k for k in set(droid) | set(csw4) if droid.get(k, "<absent>") != csw4.get(k, "<absent>")}
    unclassified = sorted(k for k in differ if not dataset_specific(k))
    assert not unclassified, f"keys differing from Meta-World csw4 without a dataset-specific justification: {unclassified}"
    assert droid["data.regime"] == "droid_largescale" and droid["aug.synth_camera_rig"] == "training_cameras"
    for key in ("data.manifest", "data.pointworld_root", "data.rlds_index_dir", "data.rlds_data_root", "model.anchor_stats"):
        assert droid[key] is None, key  # fail fast until the run gives its sources and workspace statistics


def test_profile_passes_the_geometry_guards():
    from s4d.train.augment import synth_camera_rig
    from s4d.train.loop import anchor_statistics_path

    cfg = load_config([PROFILE])
    assert synth_camera_rig(cfg) == "training_cameras"
    with pytest.raises(ValueError, match="needs model.anchor_stats"):
        anchor_statistics_path(cfg)
    assert anchor_statistics_path(load_config([PROFILE], ["model.anchor_stats=/x/stats.json"])) == "/x/stats.json"


def test_legacy_droid_base_composition_is_not_csw4():
    """pretrain.yaml + largescale.yaml (+ the csw4 variant files) is the LEGACY DROID-base smoke profile: it inherits
    the DROID base's decoder size, slot count and loss weights and must not be used as csw4."""
    legacy = flatten(load_config(
        [REPO / "configs/droid/pretrain.yaml", REPO / "configs/droid/largescale.yaml"]
        + [REPO / f"configs/metaworld/variants/{v}.yaml" for v in CSW4_VARIANTS]
    ))
    csw4 = flatten(metaworld_csw4())
    differ = {k for k in csw4 if k.startswith(("model.", "loss.")) and not dataset_specific(k) and legacy.get(k) != csw4[k]}
    assert {"model.decoder.dim", "model.encoder.num_slots", "model.encoder.slot_dim", "loss.consistency",
            "loss.depth_align"} <= differ
    from s4d.train.augment import synth_camera_rig

    with pytest.raises(ValueError, match="synth_camera_rig training_cameras"):  # the Meta-World rig is refused on DROID
        synth_camera_rig(load_config(
            [REPO / "configs/droid/pretrain.yaml", REPO / "configs/droid/largescale.yaml"]
            + [REPO / f"configs/metaworld/variants/{v}.yaml" for v in CSW4_VARIANTS]
        ))
