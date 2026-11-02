"""The DROID ViT-S-sized profile (user decision 2026-11-02): a ViT-S-sized transformer body (384 / 12 / 6, MLP ratio 4) on
our patch-local convolutional stem, 224 x 224 motion-informed square crops, one 75 % masking budget over the full token
grid. Every other key equals the DROID csw4 profile; each intended difference is registered here with its justification
(docs/droid_largescale_validation/vits_effective_config.md). configs/droid/csw4_largescale.yaml and its tests are
unchanged."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_droid_csw4_profile import TRAIN_METHOD, dataset_specific, flatten, metaworld_csw4  # noqa: E402

from s4d.config import load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "configs/droid/vits_largescale.yaml"
CSW4 = REPO / "configs/droid/csw4_largescale.yaml"

# key -> (csw4 value, this profile's value, why it differs)
INTENDED = {
    "model.encoder.image_height": (144, 224, "image: square crops of the native 180x320 grid resized to 224 (user)"),
    "model.encoder.image_width": (256, 224, "image: square crops resized to 224 x 224 (user)"),
    "model.encoder.width": (256, 384, "architecture: ViT-S-sized transformer width (user)"),
    "model.encoder.depth": (4, 12, "architecture: ViT-S-sized depth (user)"),
    "model.encoder.heads": (4, 6, "architecture: ViT-S-sized heads, head dimension 64 as in csw4 (user)"),
    "model.encoder.mlp_ratio": ("<absent>", 4.0, "architecture: explicit MLP ratio 4 (the EncoderConfig default; user)"),
}
INTENDED_PREFIX = {
    "data.crop.": "image: the motion-informed stochastic square crop and its evaluation policy (user)",
}


def intended(key: str) -> bool:
    return key in INTENDED or any(key.startswith(p) for p in INTENDED_PREFIX)


def test_every_difference_from_droid_csw4_is_registered():
    vits, csw4 = flatten(load_config([PROFILE])), flatten(load_config([CSW4]))
    differ = {k for k in set(vits) | set(csw4) if vits.get(k, "<absent>") != csw4.get(k, "<absent>")}
    assert sorted(k for k in differ if not intended(k)) == []
    for key, (before, after, _) in INTENDED.items():
        assert csw4.get(key, "<absent>") == before and vits[key] == after, key
    assert {k for k in differ if k.startswith("data.crop.")} == {k for k in vits if k.startswith("data.crop.")}


def test_kept_method_keys_and_the_csw4_relation():
    """Everything that is not an intended difference is the csw4 method: Meta-World csw4 values for method keys
    outside the registered architecture / image keys, the same temporal processing, slots, masking and objective."""
    vits, mw = flatten(load_config([PROFILE])), flatten(metaworld_csw4())
    for section in ("model.", "loss.", "aug."):
        def method(cfg, section=section):
            return {k: v for k, v in cfg.items() if k.startswith(section) and not dataset_specific(k) and not intended(k)}

        assert method(vits) == method(mw), section
    for key in TRAIN_METHOD:
        assert vits[f"train.{key}"] == mw[f"train.{key}"], key
    assert vits["model.encoder.stem"] == "patch_cnn" and vits["model.encoder.patch_size"] == 16
    assert vits["model.encoder.num_frames"] == 3 and vits["model.encoder.num_slots"] == 1
    assert vits["model.encoder.slot_dim"] == 256 and vits["model.decoder.dim"] == 128
    assert vits["model.encoder.mask_mode"] == "random" and vits["model.encoder.mask_ratio"] == 0.75
    assert vits["data.crop.out_size"] == vits["model.encoder.image_height"] == vits["model.encoder.image_width"]
    assert vits["data.crop.eval_policy"] == "center"


def test_model_dimensions_token_budget_and_unchanged_decoder():
    from s4d.model.encoder import PatchCNNStem, visible_count
    from s4d.train.loop import build_model

    vits = build_model(load_config([PROFILE], ["model.anchor_stats=/x.json"]))
    csw4 = build_model(load_config([CSW4], ["model.anchor_stats=/x.json"]))
    enc = vits.encoder
    assert isinstance(enc.patch_embed, PatchCNNStem) and enc.patch_embed.proj.out_channels == 384
    assert len(enc.blocks) == 12 and enc.blocks[0].attn.heads == 6 and enc.blocks[0].mlp.fc1.out_features == 1536
    assert enc.cfg.num_patches == 196 and enc.pos_embed.shape == (1, 196, 384)
    assert enc.temporal_embed.shape == (1, 3, 384) and enc.state_tokens.shape == (1, 1, 384)
    assert tuple(enc.slot_proj.weight.shape) == (256, 384) and enc.state_dim == csw4.encoder.state_dim == 256
    assert visible_count(3 * 196, enc.cfg.mask_ratio) == 147  # 441 of 588 temporal tokens masked
    # the decoder (its Gaussian budget and every connecting projection) is identical: slot_dim is unchanged
    assert [(n, p.shape) for n, p in vits.decoder.named_parameters()] == [
        (n, p.shape) for n, p in csw4.decoder.named_parameters()]
    n_enc = sum(p.numel() for p in enc.parameters())
    assert 21_000_000 < n_enc < 22_500_000  # ViT-S-sized (the stem replaces the 0.3 M patchify projection)
    torch.manual_seed(0)
    enc.train()
    x = torch.rand(2, 3, 3, 224, 224)
    visible = torch.zeros(2, 588, dtype=torch.bool)
    visible[:, :147] = True
    out = enc.forward_visible(x, visible)
    assert out["slots"].shape == (2, 1, 256) and out["patch_tokens"].shape == (2, 147, 384)
    assert enc.eval().policy_state((x * 255).to(torch.uint8)).shape == (2, 256)


def test_profile_passes_the_geometry_guards():
    from s4d.train.augment import synth_camera_rig
    from s4d.train.loop import anchor_statistics_path

    cfg = load_config([PROFILE])
    assert synth_camera_rig(cfg) == "training_cameras"
    with pytest.raises(ValueError, match="needs model.anchor_stats"):
        anchor_statistics_path(cfg)
    for key in ("data.manifest", "data.pointworld_root", "data.rlds_index_dir", "data.rlds_data_root", "model.anchor_stats"):
        assert flatten(cfg)[key] is None, key


def test_export_records_the_inference_contract(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("export_encoder", REPO / "scripts/export_encoder.py")
    export_encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_encoder)
    assert export_encoder.input_contract(load_config([CSW4])) is None  # unchanged exports without data.crop
    contract = export_encoder.input_contract(load_config([PROFILE]))
    assert contract == {"crop": "center_square", "side": "min(H, W)", "size": [224, 224], "resample": "bilinear",
                        "function": "s4d.data.droid.largescale.crop.inference_center_crop"}
