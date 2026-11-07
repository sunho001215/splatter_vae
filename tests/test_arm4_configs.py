"""Four-arm local campaign (arm4, user directive 2026-11-06): the arms' resolved pretraining configurations differ from
their reference configurations by exactly the intended keys, the combinations run through the training path with the
intended loss terms, initialisation is matched, and the local driver reads the registration.

  A ocsw4  Original (method-frozen-v1 row arguments) + CSW4 encoder (enc_cs.yaml + enc_w4.yaml)
  B ochf   Original + the Chamfer loss at the chamfer_lean settings (hard depth, depth gradient, visibility kept)
  C c50dg  cst50 + Original's depth-gradient coefficient (hard depth, visibility off)
  D c50vis cst50 + Original's visibility coefficient (hard depth, depth gradient off)
The comparison against the reference runs' STORED configurations, the cross-commit fingerprints and the RL configs is
runs/setup/tools/arm4_config_audit_20261106.py (it reads run directories, which a clean checkout does not have).
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest
import torch

from s4d.config import load_config
from s4d.model.encoder import EncoderConfig, sample_tube_mask, visible_count
from s4d.train.loop import build_model, forward_losses, seed_everything
from tests import _enc_golden as golden

REPO = Path(__file__).resolve().parents[1]
TASKS = ("hammer", "pick-place", "bin-picking", "coffee-push", "door-open", "peg-insert-side", "peg-unplug-side",
         "stick-push")
V = "configs/metaworld/variants/"
COMMON = ["train.steps=200000", "train.seed=0"]
ORIGINAL_SET = ["aug.synth_views=2", "aug.synth_render=true", "loss.depth_valid_far=true", "loss.motion_space=gaussian"]
CST50_FILES = [V + "chamfer_lean.yaml", V + "chamfer_lean_randmask.yaml", V + "enc_cs.yaml", V + "enc_w4.yaml"]
CST50_SET = ["model.encoder.mask_mode=tube", "model.encoder.mask_ratio=0.5"]
ARMS = {
    "A": ("original", [V + "enc_cs.yaml", V + "enc_w4.yaml"], []),
    "B": ("original", [], ["loss.chamfer=5.0", "loss.chamfer_points=4096", "loss.chamfer_voxel_m=0.01"]),
    "C": ("cst50", [], ["loss.depth_grad=0.5"]),
    "D": ("cst50", [], ["loss.visibility=1.0"]),
}
ABSENT = "<absent>"  # a key missing from a resolved configuration (distinct from an explicit null)
EXPECTED = {  # flattened key: (reference, arm)
    "A": {"model.encoder.stem": (ABSENT, "patch_cnn"), "model.encoder.width": (128, 256), "model.encoder.depth": (8, 4),
          "model.encoder.heads": (8, 4)},
    "B": {"loss.chamfer": (ABSENT, 5.0), "loss.chamfer_points": (ABSENT, 4096), "loss.chamfer_voxel_m": (ABSENT, 0.01)},
    "C": {"loss.depth_grad": (0.0, 0.5)},
    "D": {"loss.visibility": (0.0, 1.0)},
}


def stack(family_or_arm: str, task: str):
    base = ["configs/metaworld/base.yaml", f"configs/metaworld/tasks/{task}.yaml"]
    if family_or_arm == "original":
        return base, COMMON + ORIGINAL_SET
    if family_or_arm == "cst50":
        return base + CST50_FILES, COMMON + CST50_SET
    family, files, sets = ARMS[family_or_arm]
    ref_files, ref_sets = stack(family, task)
    return ref_files + files, ref_sets + sets


def resolve(name: str, task: str = "hammer") -> dict:
    files, sets = stack(name, task)
    return load_config([REPO / f for f in files], sets)


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def keydiff(ref, new):
    fa, fb = flat(ref), flat(new)
    return {k: (fa.get(k, ABSENT), fb.get(k, ABSENT)) for k in set(fa) | set(fb) if fa.get(k, ABSENT) != fb.get(k, ABSENT)}


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_each_arm_differs_from_its_reference_only_by_the_intended_keys(arm, task):
    assert keydiff(resolve(ARMS[arm][0], task), resolve(arm, task)) == EXPECTED[arm]


def test_encoder_variant_files_carry_only_their_encoder_keys():
    assert flat(load_config([REPO / V / "enc_cs.yaml"])) == {"model.encoder.stem": "patch_cnn"}
    assert flat(load_config([REPO / V / "enc_w4.yaml"])) == {"model.encoder.width": 256, "model.encoder.depth": 4,
                                                            "model.encoder.heads": 4}


def test_arm_a_has_the_cst50_encoder_with_the_original_50_percent_tube():
    a, cst = resolve("A")["model"]["encoder"], resolve("cst50")["model"]["encoder"]
    # cst50 states mask_mode 'tube' explicitly (overriding chamfer_lean_randmask); A inherits the dataclass default
    assert keydiff({"e": cst}, {"e": a}) == {"e.mask_mode": ("tube", ABSENT)}
    assert EncoderConfig().mask_mode == "tube" and EncoderConfig(**a).mask_mode == "tube"
    assert (a["stem"], a["width"], a["depth"], a["heads"], a["mask_ratio"], a["patch_size"]) == \
        ("patch_cnn", 256, 4, 4, 0.5, 16)
    keep = visible_count(64, a["mask_ratio"])
    assert (keep, keep // 2) == (32, 16)  # 32 visible spatial positions, a motion-priority quota of 16 (as Original)
    scores = torch.zeros(2, 64)
    scores[:, :40] = torch.linspace(1.0, 0.5, 40)
    torch.manual_seed(0)
    visible = sample_tube_mask(scores, a["mask_ratio"], a["motion_patch_threshold"])
    assert (visible.sum(1) == 32).all() and visible[:, :16].all()


def test_arm_b_chamfer_is_the_chamfer_lean_recipe_and_keeps_the_original_geometry_losses():
    b, orig = resolve("B")["loss"], resolve("original")["loss"]
    recipe = load_config([REPO / V / "chamfer_lean.yaml"])["loss"]
    assert {k: b[k] for k in ("chamfer", "chamfer_points", "chamfer_voxel_m")} == \
        {k: recipe[k] for k in ("chamfer", "chamfer_points", "chamfer_voxel_m")} == \
        {k: resolve("cst50")["loss"][k] for k in ("chamfer", "chamfer_points", "chamfer_voxel_m")}
    assert (b["depth_hard"], b["depth_grad"], b["visibility"]) == (orig["depth_hard"], orig["depth_grad"],
                                                                   orig["visibility"]) == (0.5, 0.5, 1.0)
    assert "occlusion" not in b and "motion_feature_render" not in b  # no other chamfer_lean key is dragged in


def test_arms_c_and_d_restore_exactly_one_original_coefficient():
    orig = resolve("original")["loss"]
    c, d = resolve("C")["loss"], resolve("D")["loss"]
    assert c["depth_grad"] == orig["depth_grad"] and (c["depth_hard"], c["visibility"]) == (0.0, 0.0)
    assert d["visibility"] == orig["visibility"] and (d["depth_hard"], d["depth_grad"]) == (0.0, 0.0)
    assert c["chamfer"] == d["chamfer"] == 5.0


def _state_sha(model) -> str:
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode())
        h.update(p.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def test_initialisation_is_matched_to_the_reference_of_the_same_architecture():
    """train.seed 0 -> seed_everything(0) -> build_model (as s4d.train.loop.train): A, C and D start from cst50's
    exact parameters (the CSW4 architecture), B from Original's."""
    inits = {}
    for name in ("original", "cst50", "A", "B", "C", "D"):
        seed_everything(0)
        inits[name] = _state_sha(build_model(resolve(name)))
    assert inits["A"] == inits["C"] == inits["D"] == inits["cst50"]
    assert inits["B"] == inits["original"] != inits["cst50"]


FAMILIES = {  # loss-term families each arm's training step reports (forward_losses), at test sizes
    "original": {"depth_hard", "depth_grad", "visibility"},
    "cst50": {"chamfer"},
    "A": {"depth_hard", "depth_grad", "visibility"},
    "B": {"depth_hard", "depth_grad", "visibility", "chamfer"},
    "C": {"chamfer", "depth_grad"},
    "D": {"chamfer", "visibility"},
}


def _small_step(cfg, tmp_path):
    """One training-mode forward_losses at step 0 (test sizes, synthetic rasterizer) -> (model, result)."""
    from s4d.data.contract import collate
    from s4d.data.metaworld.dataset import MetaworldWindowDataset

    cfg["train"]["bf16"] = False
    enc = cfg["model"]["encoder"]
    # 16 x 16 fixture; the patch-local stem needs 16 px patches (one token per frame then), patchify uses 4 px
    enc.update(image_height=16, image_width=16, patch_size=16 if enc.get("stem") == "patch_cnn" else 4, width=16,
               depth=2, heads=4, slot_dim=8)
    cfg["model"]["decoder"].update(dim=16, depth=1, heads=4)
    for group in ("scene", "dynamic"):
        cfg["model"]["decoder"][group].update(parents=4, children=2)
    cfg["model"]["anchor_stats"] = None
    if "chamfer_points" in cfg["loss"]:
        cfg["loss"]["chamfer_points"] = 64
    path = tmp_path / "f.hdf5"
    if not path.is_file():
        golden.write_fixture(path)
    ds = MetaworldWindowDataset(path, ["ep000", "ep001"], strides=(2, 4, 6), seed=0)
    batch = collate([ds[0], ds[7]])
    saved = golden._patched()
    try:
        torch.manual_seed(0)
        model = build_model(cfg).train()
        return model, forward_losses(model, batch, cfg, 0, source=torch.tensor([0, 1]))
    finally:
        golden._restore(saved)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_each_combination_trains_with_exactly_its_geometry_terms(name, tmp_path):
    model, res = _small_step(resolve(name), tmp_path)
    res["total"].backward()
    terms = {k.rsplit("_t", 1)[0] for k in res["losses"]}
    geometry = {"depth_hard", "depth_grad", "visibility", "chamfer", "occlusion"}
    assert terms & geometry == FAMILIES[name]
    assert torch.isfinite(res["total"]) and all(torch.isfinite(torch.as_tensor(v)).all() for v in res["losses"].values())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters())


@pytest.mark.parametrize("arm, key, term", [("B", "chamfer", "chamfer_t0"), ("C", "depth_grad", "depth_grad_t0"),
                                            ("D", "visibility", "visibility_t0")])
def test_the_restored_term_enters_the_total_with_its_coefficient(arm, key, term, tmp_path):
    """At step 0 (temporal ramp 0) the term of frame t0 enters the objective linearly with its weight: doubling the
    arm's coefficient adds exactly coefficient x term (the same random draws: the weight is no random input)."""
    base, doubled = resolve(arm), resolve(arm)
    w = float(base["loss"][key])
    doubled["loss"][key] = 2 * w
    _, r1 = _small_step(base, tmp_path)
    _, r2 = _small_step(doubled, tmp_path)
    assert float(r1["losses"][term]) == float(r2["losses"][term]) > 0
    torch.testing.assert_close(r2["total"].detach() - r1["total"].detach(), w * r1["losses"][term].detach().float(),
                               rtol=1e-4, atol=1e-6)


def _registrations() -> list[str]:
    """REGISTRATIONS of ops/campaign/baseline_local_driver.py, read from the source (importing the driver runs it)."""
    tree = ast.parse((REPO / "ops/campaign/baseline_local_driver.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "REGISTRATIONS")
    return [ast.literal_eval(e) for e in node.value.elts]


def test_the_local_driver_reads_the_arm4_registration_after_the_earlier_campaigns():
    regs = _registrations()
    assert regs[-1] == "arm4-registration-eight-task-20261106.json" and regs.count(regs[-1]) == 1
    # equal-priority ties resolve by registration order: the authorized CSW4 / hybrid closeout rows come first
    assert regs.index("cst-registration-eight-task-20261104.json") < regs.index(regs[-1])
    assert regs.index("ana-chlhm-registration-20261104.json") < regs.index(regs[-1])
