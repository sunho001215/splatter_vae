"""Optional baseline stages: SNeRL DINO spec pinned to the consumer code, SAM 3 instance -> semantic conversion
with explicit absent detections, and stage plumbing through the flagged MOCK worker (no real model runs)."""

import inspect
import json
import sys

import h5py
import numpy as np
import pytest

from s4d.realworld.baseline_stages import SNERL_CLASSES, SNERL_DINO, dino_snerl, instances_to_semantic, sam3_masks
from s4d.realworld.workers import WorkerSpec


def test_dino_spec_matches_the_snerl_consumer_source():
    from s4d.baselines.snerl import data as snerl_data
    from s4d.baselines.snerl import model as snerl_model
    from s4d.baselines.snerl import upstream as snerl_upstream

    src = inspect.getsource(snerl_model.dino_target)
    assert "(target - 0.45) / 0.225" in src and '11, "key", False' in src and 'mode="bilinear"' in src
    assert 'vit_extractor("dino_vits8", 4' in inspect.getsource(snerl_upstream.dino_extractor)
    assert tuple(snerl_data.SEMANTIC_CLASSES) == SNERL_CLASSES
    assert SNERL_DINO["layer"] == 11 and SNERL_DINO["facet"] == "key" and SNERL_DINO["token_grid"] == (31, 31)


def test_instances_to_semantic_resolves_overlaps_and_reports_absence():
    H, W = 4, 5
    masks = np.zeros((3, H, W), bool)
    masks[0, :2] = True  # robot, frame 0, score .9
    masks[1, 1:3] = True  # object, frame 0, score .8 (overlaps row 1 -> robot wins)
    masks[2, :, :1] = True  # fixture, frame 1, score .3 (below threshold -> absent)
    sem, present = instances_to_semantic(masks, np.array([0.9, 0.8, 0.3]), np.array([0, 0, 1]), np.array([0, 1, 2]),
                                         ["robot", "object", "fixture"], 2, (H, W))
    assert (sem[0, :2] == 1).all() and (sem[0, 2] == 2).all() and (sem[0, 3] == 0).all()
    assert (sem[1] == 0).all()
    assert present.tolist() == [[True, True, True, False], [True, False, False, False]]


def test_baseline_stages_plumbing_with_mock_worker(tmp_path):
    from s4d.realworld.adapters import get_adapter
    from s4d.realworld.pipeline import ingest

    adapter = get_adapter("synthetic", {"asset_dir": str(tmp_path / "a"), "frames": 3})
    root = tmp_path / "out"
    ingest(adapter, "synthetic_ep000", root)
    spec = WorkerSpec("mock", sys.executable, timeout_s=120)
    with pytest.raises(Exception, match="mock output refused"):
        sam3_masks(root, "synthetic_ep000", spec, cameras=["rig0/left"], prompts={"robot arm": "robot"})
    rec = sam3_masks(root, "synthetic_ep000", spec, cameras=["rig0/left"],
                     prompts={"robot arm": "robot", "red mug": "object"}, allow_mock=True)
    assert rec["summary"]["rig0/left"]["absent_detections"] == 4  # frames 1, 2 x two prompts
    with h5py.File(root / "episodes/synthetic_ep000/derived/sam3_masks/rig0__left.h5", "r") as f:
        assert json.loads(f.attrs["prompts"]) == ["robot arm", "red mug"]
        assert f["semantic"].shape == (3, *adapter.hw) and f["class_present"][0].tolist() == [True, True, True, False]
    # DINO stage on a tiny fake export cache
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(json.dumps({"image_modality": {"c": "rgb"}, "windows": [{}]}))
    with h5py.File(cache / "scratch.h5", "w") as f:
        f.create_dataset("windows/00000/images", data=np.full((3, 2, 3, 144, 256), 128, np.uint8))
    rec = dino_snerl(cache, tmp_path / "dino", spec, allow_mock=True)
    with h5py.File(tmp_path / "dino" / "dino_tokens.h5", "r") as f:
        assert f["windows/00000/tokens"].shape == (3, 2, 31, 31, 384)
    assert rec["summary"]["model"]["is_mock"] is True
