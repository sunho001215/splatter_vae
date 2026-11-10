"""infer.py (directive K): camera names, contract checks, --list, per-run output folders (CPU, fakes; nothing real)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ops.robot_pc import config as C
from ops.robot_pc.safety import ExecutionRefused

REPO = Path(__file__).resolve().parents[1]
NAMES = {"left": "135122070988", "middle": "135122076054", "right": "135122073201"}
ALL3 = [["135122070988", "135122073201", "135122076054"]]


def _infer():
    from scripts.robot_pc import infer

    return infer


def test_camera_names_file_is_the_rig_geometry_mapping(tmp_path):
    m = yaml.safe_load((REPO / "configs/robot_pc/camera_names.yaml").read_text())
    assert m == NAMES
    cfg = C.load(root=tmp_path)  # no copy under root: the repo template is used
    assert _infer().camera_names(cfg) == NAMES
    (tmp_path / "config").mkdir()
    (tmp_path / "config/camera_names.yaml").write_text("front: '135122076054'\n")
    assert _infer().camera_names(cfg) == {"front": "135122076054"}  # the user's edited copy wins
    (tmp_path / "config/camera_names.yaml").write_text("a: '1'\nb: '1'\n")
    with pytest.raises(SystemExit, match="every serial used once"):
        _infer().camera_names(cfg)


def test_rig_documents_put_left_at_plus_y_middle_at_zero_right_at_minus_y():
    docs = sorted((REPO / "assets").rglob("rig-loma-*.json")) if (REPO / "assets").is_dir() else []
    if not docs:
        pytest.skip("rig documents ship with the bundle (assets/rig), not the repo")
    for doc in docs:
        cams = json.loads(doc.read_text())["cameras"]
        y = {k.split("/")[0]: np.linalg.inv(np.array(v["T_cam_base"]).reshape(4, 4))[1, 3] for k, v in cams.items()}
        assert y[NAMES["left"]] > 0.3 and abs(y[NAMES["middle"]]) < 0.05 and y[NAMES["right"]] < -0.3


def test_resolve_cameras_names_count_and_training_contract():
    resolve = _infer().resolve_cameras
    single = {"views": 1, "trained_camera_orders": ALL3}
    assert resolve(single, NAMES, None) == ["135122070988"]  # default: the contract's first trained camera
    assert resolve(single, NAMES, "right") == ["135122073201"]
    assert resolve(single, NAMES, None, "135122076054") == ["135122076054"]
    with pytest.raises(ExecutionRefused, match="single-view: give exactly ONE camera"):
        resolve(single, NAMES, "left,right")
    with pytest.raises(ExecutionRefused, match="unknown camera name"):
        resolve(single, NAMES, "front")
    only_left = {"views": 1, "trained_camera_orders": [["135122070988"]]}
    with pytest.raises(ExecutionRefused, match="not a trained camera"):
        resolve(only_left, NAMES, "middle")
    multi = {"views": 3, "trained_camera_orders": [["135122070988", "135122076054", "135122073201"]]}
    with pytest.raises(ExecutionRefused, match="needs 3 cameras in training order: left,middle,right"):
        resolve(multi, NAMES, "left,right")
    with pytest.raises(ExecutionRefused, match="multi-view .* not implemented"):
        resolve(multi, NAMES, "left,middle,right")


def test_list_shows_task_method_seed_eligibility(monkeypatch, capsys, tmp_path):
    from ops.robot_pc import cli

    offered = {
        "e2e_cnn-drawer-open-s0": {"info": {"task": "drawer-open", "condition": "e2e_cnn", "seed": 0,
                                            "status": "deployable", "execution_eligible": True},
                                   "input_spec": {"views": 1, "modalities": ["rgb"], "trained_camera_orders": ALL3}},
        "dynamo-drawer-open-s0": {"info": {"task": "drawer-open", "condition": "dynamo", "seed": 0,
                                           "status": "restricted", "execution_eligible": False},
                                  "input_spec": {"views": 1, "modalities": ["rgb"], "trained_camera_orders": ALL3}},
    }
    monkeypatch.setattr(cli, "welcome", lambda cfg, args: {"policies": offered})
    monkeypatch.setattr(cli, "load_config", lambda args: C.load(root=tmp_path))
    assert _infer().main(["--list"]) == 0
    out = capsys.readouterr().out.splitlines()
    rows = {r.split()[3]: r.split() for r in out[1:] if r.strip() and not r.startswith("camera names")}
    assert rows["e2e_cnn-drawer-open-s0"][:5] == ["drawer-open", "e2e_cnn", "0", "e2e_cnn-drawer-open-s0", "yes"]
    assert rows["dynamo-drawer-open-s0"][4] == "no" and rows["dynamo-drawer-open-s0"][-1] == "left,right,middle"


def test_run_folders_are_named_numbered_and_never_reused(tmp_path):
    from ops.robot_pc.recorder import SessionRecorder

    cams = {"135122073201": {"native_hw": [4, 6], "K_native": np.eye(3).tolist()}}
    dirs = []
    for _ in range(3):
        r = SessionRecorder(tmp_path, meta={"x": 1}, cameras=cams, run_prefix="drawer-open_e2e_cnn_seed0_right",
                            camera_names={"135122073201": "right"}, files={"config.json": {"policy": "p"}})
        r.frame("135122073201", 1, 10, np.zeros((4, 6, 3), np.uint8), None, None, {})
        summary = r.close()
        dirs.append(Path(summary["session_dir"]).name)
        d = Path(summary["session_dir"])
        assert json.loads((d / "config.json").read_text()) == {"policy": "p"}
        assert (d / "inference.h5").is_file() and (d / "summary.json").is_file()
        video = summary["videos"]["135122073201"]
        assert video["status"] != "available" or Path(video["file"]).name == "right.mp4"
    assert dirs == [f"drawer-open_e2e_cnn_seed0_right_run{i:03d}" for i in (1, 2, 3)]
    (tmp_path / "inference_outputs/drawer-open_e2e_cnn_seed0_right_run007").mkdir()  # a gap: next is 1 + max
    r = SessionRecorder(tmp_path, meta={}, cameras=cams, run_prefix="drawer-open_e2e_cnn_seed0_right")
    assert r.session_dir.name.endswith("_run008")
    r.close()
    with pytest.raises(ValueError):
        SessionRecorder(tmp_path, meta={}, cameras=cams, run_prefix="../escape")


def test_infer_cli_refuses_conflicting_camera_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        _infer().main(["--cameras", "left", "--camera", "135122070988", "--dry-run"])
    assert exc.value.code == 2 and "not both" in capsys.readouterr().err
