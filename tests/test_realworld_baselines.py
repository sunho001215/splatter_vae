"""Optional baseline stage: SNeRL DINO spec pinned to the consumer code (targets computed online), SAM 3 instance ->
semantic conversion with explicit absent detections, and stage plumbing through the flagged MOCK worker."""

import inspect
import json
import sys

import h5py
import numpy as np
import pytest

from s4d.realworld.stages.semantics import SNERL_CLASSES, SNERL_DINO, instances_to_semantic, sam3_masks
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
    from s4d.realworld.pipeline import ingest
    from tests.test_realworld_trackeverything import ColourEpisode

    adapter = ColourEpisode(tmp_path / "a", frames=3)
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
