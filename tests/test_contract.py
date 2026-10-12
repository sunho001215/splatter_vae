"""One batch contract is shared by both data regimes and every loader."""

from __future__ import annotations

import pytest
import torch

from s4d.data.contract import TRACK_MOTION, ContractError, collate, motion_targets, validate_batch


def test_training_batch_valid_without_heldout(batch):
    assert validate_batch(batch) == {"B": 2, "T": 3, "V": 2, "H": 16, "W": 16}


def test_complete_optional_heldout_fields_valid_in_both_modes(heldout_batch):
    assert validate_batch(heldout_batch) == validate_batch(heldout_batch, training=False)


def test_validation_droid_batch_may_omit_optional_heldout_views(batch):
    # DROID exterior cameras are all training cameras. The shared contract makes
    # eval_* optional, not a mandatory condition for using a validation split.
    assert validate_batch(batch, training=False)["V"] == 2


@pytest.mark.parametrize("field", ["images", "K", "w2c", "c2w", "depth", "motion3d", "motion_weight", "motion_score"])
def test_required_fields_rejected_when_missing(batch, field):
    del batch[field]
    with pytest.raises(ContractError):
        validate_batch(batch)


@pytest.mark.parametrize("field", ["K", "w2c", "c2w", "depth", "motion3d", "motion_weight", "motion_score"])
def test_required_floats_reject_nonfinite_values(batch, field):
    batch[field][(0,) * batch[field].ndim] = float("nan")
    with pytest.raises(ContractError, match="non-finite"):
        validate_batch(batch)


def test_probe_state_rejects_nonfinite_values(batch):
    batch["probe_state"][0, 1, 4] = float("inf")
    with pytest.raises(ContractError):
        validate_batch(batch)


@pytest.mark.parametrize("field", ["images", "depth", "motion3d"])
def test_wrong_dtype_rejected(batch, field):
    batch[field] = batch[field].to(torch.float64)
    with pytest.raises(ContractError, match="dtype"):
        validate_batch(batch)


def test_three_time_steps_are_required(batch):
    batch["images"] = batch["images"][:, :1]
    with pytest.raises(ContractError):
        validate_batch(batch)


@pytest.mark.parametrize(
    "field,value",
    [("depth", -0.1), ("motion_weight", -0.1), ("motion_weight", 1.1), ("motion_score", -0.1), ("motion_score", 1.1)],
)
def test_negative_depth_and_out_of_range_motion_rejected(batch, field, value):
    batch[field].flatten()[0] = value
    with pytest.raises(ContractError):
        validate_batch(batch)


def test_camera_inverse_mismatch_rejected(batch):
    batch["c2w"][0, 0, 0, 3] += 0.2
    with pytest.raises(ContractError, match="identity"):
        validate_batch(batch)


def test_camera_rotation_requires_proper_not_reflected_frame(batch):
    batch["w2c"][0, 0, 0, 0] = -1
    batch["c2w"][0, 0, 0, 0] = -1
    with pytest.raises(ContractError):
        validate_batch(batch)


def test_focal_lengths_must_be_positive(batch):
    batch["K"][0, 0, 0, 0] = 0
    with pytest.raises(ContractError, match="focal"):
        validate_batch(batch)


def test_intrinsics_are_pixel_units(batch):
    batch["K"][0, 0, 0, 2] = 16
    with pytest.raises(ContractError, match="principal point"):
        validate_batch(batch)


@pytest.mark.parametrize("field", ["eval_images", "eval_K", "eval_w2c", "eval_depth"])
def test_partial_heldout_fields_rejected(heldout_batch, field):
    del heldout_batch[field]
    with pytest.raises(ContractError, match="together"):
        validate_batch(heldout_batch)


def test_heldout_camera_dtype_checked(heldout_batch):
    heldout_batch["eval_K"] = heldout_batch["eval_K"].double()
    with pytest.raises(ContractError, match="dtype"):
        validate_batch(heldout_batch)


def test_meta_must_be_a_dictionary(batch):
    batch["meta"] = ["not-a-dictionary"]
    with pytest.raises(ContractError, match="meta"):
        validate_batch(batch)


def test_track_batch_valid_without_pixel_grid_targets(track_batch):
    assert motion_targets(track_batch) == "tracks"
    assert validate_batch(track_batch) == {"B": 2, "T": 3, "V": 2, "H": 16, "W": 16}


@pytest.mark.parametrize("field", TRACK_MOTION)
def test_partial_track_fields_rejected(track_batch, field):
    del track_batch[field]
    with pytest.raises(ContractError, match="together"):
        validate_batch(track_batch)


def test_dense_and_track_targets_cannot_be_mixed(track_batch):
    track_batch["motion3d"] = torch.zeros(2, 3, 2, 3, 16, 16)
    with pytest.raises(ContractError, match="cannot accompany"):
        validate_batch(track_batch)


@pytest.mark.parametrize(
    "field,index,value,match",
    [
        ("track_weight", (0, 0, 0, 2), 1.0, "padding"),
        ("track_src", (0, 0, 0, 2, 0), 0.5, "padding"),
        ("track_disp", (0, 1, 1, 2, 2), 0.5, "padding"),
        ("track_weight", (0, 0, 0, 0), 1.5, r"\[0,1\]"),
        ("gripper_weight", (0, 0, 0, 0), -0.1, r"\[0,1\]"),
        ("track_src", (0, 0, 0, 0, 0), float("nan"), "non-finite"),
        ("gripper_points", (0, 0, 0, 0), float("inf"), "non-finite"),
    ],
)
def test_track_values_checked(track_batch, field, index, value, match):
    track_batch[field][index] = value
    with pytest.raises(ContractError, match=match):
        validate_batch(track_batch)


def test_track_count_range_and_dtype(track_batch):
    track_batch["track_count"][0, 1] = 4
    with pytest.raises(ContractError, match="track_count"):
        validate_batch(track_batch)
    track_batch["track_count"] = track_batch["track_count"].clamp_max(2).to(torch.int32)
    with pytest.raises(ContractError, match="dtype"):
        validate_batch(track_batch)


def test_track_shapes_must_agree_across_fields(track_batch):
    track_batch["track_weight"] = track_batch["track_weight"][..., :2]
    with pytest.raises(ContractError, match="shape"):
        validate_batch(track_batch)


def test_collate_stacks_track_arrays(track_batch):
    samples = []
    for i in range(2):
        sample = {key: value[i] for key, value in track_batch.items() if key != "meta"}
        sample["meta"] = {key: value[i] for key, value in track_batch["meta"].items()}
        samples.append(sample)
    result = collate(samples)
    assert result["track_count"].shape == (2, 2) and result["track_count"].dtype == torch.int64
    torch.testing.assert_close(result["track_src"], track_batch["track_src"])


def test_collate_preserves_axes_and_metadata(heldout_batch):
    samples = []
    for i in range(2):
        sample = {key: value[i] for key, value in heldout_batch.items() if key != "meta"}
        sample["meta"] = {key: value[i] for key, value in heldout_batch["meta"].items()}
        samples.append(sample)
    result = collate(samples)
    assert validate_batch(result, training=False)["B"] == 2
    assert result["meta"] == heldout_batch["meta"]
    torch.testing.assert_close(result["motion3d"], heldout_batch["motion3d"])
