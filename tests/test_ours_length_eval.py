"""CPU checks of grasp-site labels, camera projection, flow sampling, and probe split discipline."""

import numpy as np
import pytest
import torch

from s4d.diag.ours_length_eval import grasp_anchor, handle_roi, probe_labels, score_probes, track_errors


def test_grasp_uses_increasing_closure_and_valid_fk():
    robot = {"probe_state": np.zeros((7, 7)), "gripper_points": np.zeros((7, 4, 3)), "ok": np.ones(7, bool)}
    robot["probe_state"][:, 6] = [0, 0, 0.3, 0.3, 0.3, 0.1, 0]
    robot["gripper_points"][:, :, 2] = 1
    anchor = grasp_anchor(robot)
    assert anchor["grasp"] == 2 and anchor["release"] == 5
    assert anchor["xyz"] == [0.0, 0.0, 1.0]
    robot["probe_state"][:, 6] = 0
    with pytest.raises(ValueError, match="closing edge"):
        grasp_anchor(robot)


def test_roi_and_labels_are_fixed_grasp_proxies():
    batch = {
        "images": torch.zeros(1, 3, 3, 3, 16, 16, dtype=torch.uint8),
        "K": torch.tensor([[100.0, 0, 8.0], [0, 100.0, 8.0], [0, 0, 1.0]]).expand(1, 3, 3, 3),
        "w2c": torch.eye(4).expand(1, 3, 4, 4).clone(),
        "probe_state": torch.zeros(1, 3, 7),
        "meta": {"episode": ["demo"], "t_indices": [(0, 1, 2)]},
    }
    anchors = {"demo": {"xyz": [0, 0, 1.0], "grasp": 1, "release": 5}}
    roi = handle_roi(batch, anchors)
    assert roi.shape == (1, 3, 3, 1, 16, 16)
    assert roi[0, 0, 0, 0, 8, 8] and not roi[0, 0, 0, 0, 0, 0]
    batch["w2c"][0, 2, 2, 3] = -2  # behind camera: no ROI support
    assert not handle_roi(batch, anchors)[0, :, 2].any()
    batch["probe_state"][0, 2, 0] = 0.02
    labels = probe_labels(batch, anchors)
    assert labels["phase_proxy"].tolist() == [[0, 1, 0]]
    assert labels["gripper_open_fraction"].item() == 1
    assert labels["future_ee_displacement_m"][0, 0] == 0.02
    assert labels["ee_initial_handle_relative_m"].tolist() == [[0.0, 0.0, -1.0]]


def test_dynamic_motion_excludes_static_and_invalid_tracks():
    src = torch.tensor([[[[[0.0, 0, 1.0], [0.0, 0, 1.0], [0.0, 0, 1.0]]]]]).expand(1, 3, 3, 3, 3).clone()
    disp = torch.zeros_like(src)
    disp[..., 0, 0] = 0.02
    disp[..., 1, 0] = 0.04
    weight = torch.ones(1, 3, 3, 3)
    weight[..., 1] = 0
    batch = dict(
        track_src=src,
        track_disp=disp,
        track_weight=weight,
        K=torch.tensor([[10.0, 0, 8.0], [0, 10.0, 8.0], [0, 0, 1.0]]).expand(1, 3, 3, 3),
        w2c=torch.eye(4).expand(1, 3, 4, 4),
    )
    pred = torch.zeros(1, 3, 3, 3, 16, 16)
    pred[..., 0, :, :] = 0.02
    errors = track_errors(batch, {"pred_disp": pred})
    assert errors["dynamic_track_epe_m"].numel() == 9
    torch.testing.assert_close(errors["dynamic_track_epe_m"], torch.zeros(9), atol=1e-7, rtol=0)
    torch.testing.assert_close(errors["dynamic_flow_epe_px"], torch.zeros(9), atol=1e-6, rtol=0)


def test_frozen_probes_train_only_no_encoder_gradients():
    torch.manual_seed(3)
    x, vx = torch.randn(50, 4), torch.randn(20, 4)
    y, vy = {"relative": x[:, :2] * 2}, {"relative": vx[:, :2] * 2}
    result = score_probes((x, y), (vx, vy))
    assert result["r2_relative_validation"] > 0.99
    assert result["mlp_r2_relative_validation"] > 0.8
    assert not x.requires_grad and not vx.requires_grad
