"""CPU checks of grasp-site labels, camera projection, flow sampling, and probe split discipline."""

import numpy as np
import pytest
import torch

from s4d.diag.ours_task_length_eval import grasp_anchor, handle_roi, probe_labels, score_probes, track_errors


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


@pytest.fixture
def roi_batch():
    return {
        "images": torch.zeros(2, 3, 3, 3, 32, 32, dtype=torch.uint8),
        "depth": torch.ones(2, 3, 3, 1, 32, 32),
        "K": torch.tensor([[100., 0, 16.], [0, 100., 16.], [0, 0, 1.]]).expand(2, 3, 3, 3),
        "w2c": torch.eye(4).expand(2, 3, 4, 4).clone(),
        "probe_state": torch.zeros(2, 3, 7),
        "meta": {"task_name": ["covering"] * 2, "episode": ["covering-000004", "other"],
                 "t_indices": [(0, 1, 2)] * 2},
    }


def test_store_keys_and_batch_episode_ids_resolve_same_grasp(roi_batch):
    from types import SimpleNamespace

    from s4d.diag.ours_task_length_eval import collect_grasp_anchors

    robot = {"probe_state": np.zeros((3, 7)), "gripper_points": np.zeros((3, 5, 3))}
    robot["probe_state"][:, 6] = [0, 1, 0]
    robot["gripper_points"][..., 2] = 1
    calls = []
    def episode(key):
        calls.append(key)
        return SimpleNamespace(task="covering", episode_id="covering-000004", reference_camera="cam",
                               robot=lambda _: robot)
    loader = SimpleNamespace(dataset=SimpleNamespace(episodes=["covering/covering-000004"],
                                                     store=SimpleNamespace(episode=episode)))
    anchors = collect_grasp_anchors([loader, loader])
    assert set(anchors) == {"covering/covering-000004"}
    mask = handle_roi(roi_batch, anchors)
    assert mask[0].any() and not mask[1].any()
    labels = probe_labels(roi_batch, anchors)
    assert torch.isfinite(labels["ee_initial_handle_relative_m"][0]).all()
    assert torch.isnan(labels["ee_initial_handle_relative_m"][1]).all()
    assert labels["phase_proxy"][0].tolist() == [0, 1, 0]
    # Identical episode ids in another task must not borrow this task's site.
    roi_batch["meta"]["task_name"][0] = "pick-and-place"
    assert not handle_roi(roi_batch, anchors).any()


@pytest.mark.parametrize("task,mean", [("covering", [0.04, 0, 1]), ("pick-and-place", [-0.04, 0, 1]),
                                        ("mug-hanging", [0, 0.04, 1])])
def test_task_region_uses_own_dynamic_workspace(tmp_path, roi_batch, task, mean):
    import json

    from s4d.diag.ours_task_length_eval import detail_region, detail_roi
    from s4d.losses.rgb import masked_psnr

    stats = {"source": f"/calibrated/{task}/store.yaml", "dynamic_fallback_to_scene": False,
             "dynamic": {"mean": mean, "std": [0.01, 0.01, 0.01], "num_points": 123}}
    path = tmp_path / "workspace_stats.json"
    path.write_text(json.dumps(stats))
    region = detail_region({"data": {"task": task}, "model": {"anchor_stats": str(path)}})
    assert region["mean"] == mean and region["kind"] == "dynamic_workspace"
    roi = detail_roi(roi_batch, {}, region)
    assert roi.shape == (2, 3, 3, 1, 32, 32) and roi.any()
    cx, cy = round(16 + mean[0] * 100), round(16 + mean[1] * 100)
    assert roi[0, :, :, :, cy, cx].all()
    assert not roi[0, :, :, :, 0, 0].any()
    assert torch.isfinite(masked_psnr(torch.zeros(18, 3, 32, 32), torch.ones(18, 3, 32, 32),
                                    roi.flatten(0, 2))).all()
    roi_batch["w2c"][0, 2, 2, 3] = -2
    assert not detail_roi(roi_batch, {}, region)[0, :, 2].any()


@pytest.mark.parametrize("patch", [
    {"dynamic_fallback_to_scene": True}, {"dynamic": {}},
    {"dynamic": {"mean": [0, 0, 1], "std": [0, 1, 1], "num_points": 2}},
    {"source": "/calibrated/drawer-open/store.yaml"},
])
def test_no_fabricated_region_from_scene_or_wrong_task(tmp_path, roi_batch, patch):
    import json

    from s4d.diag.ours_task_length_eval import detail_region, detail_roi

    stats = {"source": "/calibrated/covering/store.yaml", "dynamic_fallback_to_scene": False,
             "scene": {"mean": [0, 0, 1], "std": [1, 1, 1]},
             "dynamic": {"mean": [0, 0, 1], "std": [0.1, 0.1, 0.1], "num_points": 2}, **patch}
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    region = detail_region({"data": {"task": "covering"}, "model": {"anchor_stats": str(path)}})
    assert region["value"] is None and region["reason"]
    assert detail_roi(roi_batch, {}, region) is None
    assert json.loads(json.dumps(region, allow_nan=False))["value"] is None


def test_scene_parameters_alone_do_not_define_small_detail_region(roi_batch):
    from s4d.diag.ours_task_length_eval import detail_region, detail_roi
    from s4d.rwlearn.configs import SCENE_PARAMETERS

    for task in SCENE_PARAMETERS:
        if task == "drawer-open":
            continue
        region = detail_region({"data": {"task": task}})
        assert region["value"] is None and region["reason"]
        assert detail_roi(roi_batch, {}, region) is None


def test_missing_grasp_does_not_suppress_defined_probes(roi_batch):
    import json
    from types import SimpleNamespace

    from s4d.diag.ours_task_length_eval import collect_grasp_anchors

    robot = {"probe_state": np.zeros((3, 7)), "gripper_points": np.zeros((3, 5, 3))}
    ef = SimpleNamespace(task="covering", episode_id="covering-000004", reference_camera="cam", robot=lambda _: robot)
    loader = SimpleNamespace(dataset=SimpleNamespace(episodes=["covering/covering-000004"],
                                                     store=SimpleNamespace(episode=lambda _: ef)))
    anchors = collect_grasp_anchors([loader])
    assert anchors["covering/covering-000004"]["value"] is None
    assert "closing edge" in anchors["covering/covering-000004"]["reason"]
    labels = probe_labels(roi_batch, anchors)
    assert torch.isnan(labels["phase_proxy"]).all()
    assert torch.isfinite(labels["gripper_open_fraction"]).all()
    assert torch.isfinite(labels["future_ee_displacement_m"]).all()
    x = torch.randn(2, 3)
    scores = score_probes((x, labels), (x, labels))
    assert scores["r2_ee_initial_handle_relative_m_validation"] is None
    assert scores["mlp_phase_accuracy_validation"] is None
    assert scores["not_applicable"]["mlp_phase_accuracy_validation"]
    assert np.isfinite(scores["r2_gripper_open_fraction_validation"])
    assert np.isfinite(scores["mlp_r2_future_ee_displacement_m_validation"])
    json.dumps(scores, allow_nan=False)


@pytest.mark.parametrize("task,workspace", [("covering", True), ("pick-and-place", True), ("unmeasured", False)])
def test_entrypoint_completes_all_defined_metrics(tmp_path, monkeypatch, roi_batch, task, workspace):
    """Exercise the report-writing orchestration with tiny CPU data and deterministic render/model doubles."""
    import importlib.util
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from torch.utils.data import DataLoader, Dataset

    import ops.jobs.run_identity as identity
    import s4d.config as config
    import s4d.diag.local_log as local_log
    import s4d.diag.ours_task_length_eval as diag
    import s4d.losses.rgb as rgb_loss
    import s4d.model.render as render
    import s4d.train.checkpoint as checkpoint
    import s4d.train.evaluate as evaluate
    import s4d.train.loop as loop

    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo / "scripts"))
    import train

    spec = importlib.util.spec_from_file_location("length_eval_test_entry", repo / "scripts/evaluate_ours_task_length.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setattr(entry, "require_passed_tests", lambda: None)
    robot = {"probe_state": np.zeros((3, 7)), "gripper_points": np.zeros((3, 5, 3))}
    ef = SimpleNamespace(task=task, episode_id="episode", reference_camera="cam", robot=lambda _: robot)

    class Windows(Dataset):
        episodes = [f"{task}/episode"]
        store = SimpleNamespace(episode=lambda _: ef)

        def __len__(self):
            return 2

        def __getitem__(self, i):
            sample = {k: v[i] for k, v in roi_batch.items() if k != "meta"}
            sample["meta"] = {"task_name": task, "episode": "episode", "t_indices": (0, 1, 2)}
            return sample

    from s4d.data.contract import collate

    roi_batch.update(c2w=roi_batch["w2c"].clone(), motion3d=torch.zeros(2, 3, 3, 3, 32, 32),
                     motion_weight=torch.ones(2, 3, 3, 1, 32, 32), motion_score=torch.zeros(2, 3, 3, 1, 32, 32))
    loader = DataLoader(Windows(), batch_size=2, collate_fn=collate)
    monkeypatch.setattr(train, "build_data", lambda _: (None, {1: loader}, {1: loader}, 3))
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps({"source": f"/calibrated/{task}/store.yaml", "dynamic_fallback_to_scene": False,
                                "dynamic": {"mean": [0, 0, 1], "std": [.01, .01, .01], "num_points": 10}}))
    cfg = {"data": {"task": task}, "model": {"anchor_stats": str(path)} if workspace else {},
           "train": {"batch_size": 2}, "eval": {"max_batches": 1, "probe_batches": 1},
           "render": {"near": .05, "far": 3.}}
    monkeypatch.setattr(config, "load_config", lambda *a: cfg)
    monkeypatch.setattr(identity, "record_run_identity", lambda *a: None)
    model = SimpleNamespace(encoder=None, decoder=None, eval=lambda: model, to=lambda _: model)
    monkeypatch.setattr(loop, "build_model", lambda _: model)
    monkeypatch.setattr(loop, "move_batch", lambda batch, _: batch)
    gs = SimpleNamespace(xyz_sequence=lambda: None)
    monkeypatch.setattr(loop, "forward_losses", lambda model, batch, *a, **kw:
                        {"rgb": batch["images"].float() / 255, "depth": batch["depth"], "gs": gs})
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *a, **kw: 400000)
    monkeypatch.setattr(render, "require_prebuilt_renderer", lambda: None)
    monkeypatch.setattr(render, "render_rgbd", lambda *a: {"rgb": roi_batch["images"].float() / 255})
    monkeypatch.setattr(evaluate, "Evaluator", lambda *a: lambda *a: {"standard_metric": 1.})
    monkeypatch.setattr(evaluate, "encode_states", lambda model, images: torch.ones(len(images), 3, 1, 2))
    monkeypatch.setattr(diag, "track_errors", lambda *a: {"dynamic_track_epe_m": torch.tensor([.01])})
    monkeypatch.setattr(rgb_loss, "masked_ssim", rgb_loss.masked_psnr)
    monkeypatch.setattr(local_log, "RunLogger", lambda _: SimpleNamespace(close=lambda: None))
    source = tmp_path / "source.pt"
    source.write_bytes(b"immutable checkpoint")
    monkeypatch.setattr(sys, "argv", [str(entry.__file__), "--checkpoint", str(source), "--config", "fixture",
                                     "--name", "result", "--output-root", str(tmp_path)])
    entry.main()
    report = json.loads((tmp_path / "result/evaluation.json").read_text())
    assert report["step"] == 400000
    assert report["standard"]["standard_metric"] == 1.
    metrics = report["extended"]["1"]
    assert metrics["heldout/psnr"]["count"] > 0
    assert metrics["dynamic_track_epe_m"]["value"] == pytest.approx(.01)
    roi_metric = metrics["task_relevant_roi/psnr"]
    assert (roi_metric["value"] is not None) == workspace
    if not workspace:
        assert roi_metric["count"] == 0 and roi_metric["reason"]
    assert report["probes"]["1"]["r2_gripper_open_fraction_validation"] is not None
    assert report["probes"]["1"]["r2_ee_initial_handle_relative_m_validation"] is None
    assert json.loads((tmp_path / "result/completion.json").read_text())["status"] == "completed"
    assert source.read_bytes() == b"immutable checkpoint"


def test_task_evaluator_has_identical_launch_input_and_output_guards():
    from ops.containers.launch import validate_output_arguments
    from ops.jobs import host_roles, realworld_local

    script = "scripts/evaluate_ours_task_length.py"
    row = {"id": "eval", "script": script, "deps": ["pretrain"], "args": [
        "--name", "eval", "--output-root", "runs", "--checkpoint", "runs/pretrain/checkpoints/step_0400000.pt",
        "--config", "configs/realworld/ours_ext/baseline200k-covering-S.yaml", "--set",
        "data.store=/remote/covering/store.yaml", "data.split_manifest=/remote/covering/split.json",
        "data.gate_report=/remote/covering/gate.json", "model.anchor_stats=runs/stats/workspace_stats.json"]}
    assert host_roles._activity(script) == "pretrain"
    assert realworld_local.RESOLVERS[script] == realworld_local.RESOLVERS[realworld_local.OURS_EVAL]
    assert script in realworld_local.CLOSEOUT_CONSUMERS
    actual = vars(realworld_local._arguments(row))
    legacy = vars(realworld_local._arguments({**row, "script": realworld_local.OURS_EVAL}))
    assert actual == legacy
    root = realworld_local.REPO
    paths = realworld_local._resolved_paths(row, root, remote_prefix="/remote")
    assert paths == realworld_local._resolved_paths({**row, "script": realworld_local.OURS_EVAL},
                                                    root, remote_prefix="/remote")
    assert "/remote/covering/store.yaml" in paths
    bad = {**row, "args": [*row["args"][:-1], "eval.max_batches=1"]}
    with pytest.raises(realworld_local.LocalInputError, match="four verified"):
        realworld_local._arguments(bad)
    with pytest.raises(realworld_local.LocalInputError, match="declared dependency"):
        realworld_local._arguments({**row, "deps": []})
    with pytest.raises(realworld_local.LocalInputError, match="own id"):
        realworld_local._arguments({**row, "id": "another"})
    validate_output_arguments(script, ["--name", "eval", "--output-root", "runs"], "eval")
    with pytest.raises(ValueError, match="artifacts"):
        validate_output_arguments(script, ["--name", "another", "--output-root", "runs"], "eval")
