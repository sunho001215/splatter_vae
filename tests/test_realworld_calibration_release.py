"""Calibration release 2026-11-08 (docs/realworld_calibration_release.md): release rig documents (rig_loma_publish),
rig-mode fixedness rule + fixedness exception, calibration_validated records (calibration_release), per-camera depth
scale from the accepted calibration record, corrected Robotiq 2F-85 gripper map, fail-closed visual meshes and the
add-only visual-mesh tool. Synthetic fixtures only."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from s4d.realworld import io as rio
from s4d.realworld.robot import GRIPPER_MAP_2F85, GRIPPER_MAP_LEGACY, gripper_joint_value

REPO = Path(__file__).resolve().parents[1]
CAMS = ("135122070988/color", "135122073201/color", "135122076054/color")


# ------------------------------------------------------------------------------------------------ gripper map
def test_gripper_map_corrected_for_the_2f85_register_and_legacy_unchanged():
    g = lambda r, m: gripper_joint_value(r, "robotiq_position_register", "robotiq_2f_85", mapping=m)  # noqa: E731
    # legacy values are bit-identical to the pre-release formula
    for r in (0, 3, 49, 211, 230, 255):
        assert g(r, GRIPPER_MAP_LEGACY) == float(np.clip(r, 0, 255)) / 255.0 * 0.8
        assert gripper_joint_value(r, "robotiq_position_register", "robotiq_2f_85") == g(r, GRIPPER_MAP_LEGACY)
    # corrected: open plateau register 3 -> 0, close command register 230 -> pads touching (0.8 rad)
    assert g(3, GRIPPER_MAP_2F85) == 0.0 and g(0, GRIPPER_MAP_2F85) == 0.0
    assert g(230, GRIPPER_MAP_2F85) == pytest.approx(0.8) and g(255, GRIPPER_MAP_2F85) == pytest.approx(0.8)
    assert g(211, GRIPPER_MAP_2F85) == pytest.approx(0.7330396, abs=1e-6)  # mug-grasp plateau (robot_model_check)
    assert g(211, GRIPPER_MAP_2F85) - g(211, GRIPPER_MAP_LEGACY) == pytest.approx(0.0711, abs=1e-3)
    with pytest.raises(ValueError, match="2F-85"):
        gripper_joint_value(100, "robotiq_position_register", "robotiq_2f_140", mapping=GRIPPER_MAP_2F85)
    with pytest.raises(ValueError, match="2F-85"):
        gripper_joint_value(0.5, "normalized_closure", "robotiq_2f_85", mapping=GRIPPER_MAP_2F85)
    with pytest.raises(ValueError, match="unknown gripper map"):
        gripper_joint_value(3, "robotiq_position_register", "robotiq_2f_85", mapping="reg/254")


def test_robot_geometry_poses_with_the_map_but_keeps_the_legacy_closure(monkeypatch):
    from types import SimpleNamespace

    from s4d.realworld import pipeline as P

    class FakeModel:
        joints, links = {}, {}

    monkeypatch.setattr(P.URDFModel, "from_file", staticmethod(lambda path: FakeModel()))
    monkeypatch.setattr(P.RobotSurface, "build", classmethod(lambda cls, *a, **k: "collision-surface"))
    desc = SimpleNamespace(urdf_path="x.urdf", gripper_model="robotiq_2f_85", mesh_root=None,
                           gripper_state_semantics="robotiq_position_register", gripper_joint="finger_joint",
                           joint_names=[f"j{i}" for i in range(7)])
    meta = SimpleNamespace(robot=desc)
    legacy, fixed = P.RobotGeometry(meta), P.RobotGeometry(meta, gripper_map=GRIPPER_MAP_2F85)
    assert legacy.robot_model == P.LEGACY_ROBOT_MODEL
    assert fixed.robot_model == {"gripper_map": GRIPPER_MAP_2F85, "surface": "collision"}
    assert fixed.q(np.zeros(7), 230.0)["finger_joint"] == pytest.approx(0.8)
    assert legacy.q(np.zeros(7), 230.0)["finger_joint"] == pytest.approx(230 / 255 * 0.8)
    assert fixed.closure(230.0) == legacy.closure(230.0) == pytest.approx(230 / 255)  # proprioception unchanged
    synthetic = SimpleNamespace(robot=SimpleNamespace(**{**vars(desc), "gripper_model": "synthetic_two_finger",
                                                         "gripper_state_semantics": "normalized_closure"}))
    with pytest.raises(ValueError, match="2F-85"):
        P.RobotGeometry(synthetic, gripper_map=GRIPPER_MAP_2F85, closed_value=0.04)
    with pytest.raises(ValueError, match="surface"):
        P.RobotGeometry(meta, surface="hull")


# ------------------------------------------------------------------------------------------------ visual meshes
DAE = """<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
 <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
 <library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array id="pa" count="12">0 0 0 0.1 0 0 0 0.1 0 0 0 0.1</float_array>
   <technique_common><accessor source="#pa" count="4" stride="3"/></technique_common></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <triangles count="4"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2 0 1 3 0 2 3 1 2 3</p></triangles>
 </mesh></geometry></library_geometries>
 <library_visual_scenes><visual_scene id="s"><node id="n"><instance_geometry url="#g"/></node></visual_scene>
 </library_visual_scenes><scene><instance_visual_scene url="#s"/></scene>
</COLLADA>"""


def _mesh_urdf(tmp_path, with_visual: bool):
    import struct

    from s4d.realworld.robot import URDFModel

    col = tmp_path / "pkg" / "meshes" / "collision"
    col.mkdir(parents=True)
    tri = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], np.float32)
    (col / "link.stl").write_bytes(b"\0" * 80 + struct.pack("<I", 1) + np.zeros(3, np.float32).tobytes()
                                   + tri.tobytes() + b"\0\0")
    if with_visual:
        (tmp_path / "pkg" / "meshes" / "visual").mkdir()
        (tmp_path / "pkg" / "meshes" / "visual" / "link.dae").write_text(DAE)
    urdf = f"""<robot name="r"><link name="base"><collision><geometry><mesh filename="{col / 'link.stl'}"/>
    </geometry></collision></link></robot>"""
    return URDFModel.from_string(urdf, use="collision")


def test_visual_surfaces_fail_closed_without_visual_meshes(tmp_path):
    from s4d.realworld.rig_loma_render import build_robot_mesh
    from s4d.realworld.robot import RobotSurface
    from s4d.realworld.robot_visual import MissingVisualMeshError, VisualSurface

    model = _mesh_urdf(tmp_path / "a", with_visual=False)
    with pytest.raises(MissingVisualMeshError, match="no collision fallback"):
        RobotSurface.build_visual(model, 50)
    with pytest.raises(MissingVisualMeshError):
        VisualSurface.build(model, spacing_m=0.01)
    with pytest.raises(MissingVisualMeshError):
        build_robot_mesh(model)
    # an explicit opt-in still records the fallback (diagnostics / synthetic tests only)
    assert VisualSurface.build(model, spacing_m=0.01, allow_collision_fallback=True).fallback == ["base"]
    assert RobotSurface.build(model, 50).provenance == {"surface": "collision"}
    good = _mesh_urdf(tmp_path / "b", with_visual=True)
    surf = RobotSurface.build_visual(good, 60, seed=0)
    assert surf.provenance["surface"] == "visual" and surf.link_points["base"].shape == (60, 3)
    assert surf.provenance["meshes"]["base"][0]["file"].endswith("visual/link.dae")
    assert surf.link_points["base"][:, 2].max() > 0.01  # sampled the visual tetrahedron, not the flat STL triangle
    assert VisualSurface.build(good, spacing_m=0.01).fallback == []


def test_add_visual_meshes_tool_is_add_only(tmp_path):
    tool = REPO / "scripts" / "realworld_add_visual_meshes.py"
    src, dst = tmp_path / "src", tmp_path / "dst"
    rel = ["robot_assets/src/pkg/meshes/visual/a.dae", "robot_assets/src/pkg/meshes/visual/b.dae"]
    for i, r in enumerate(rel):
        (src / r).parent.mkdir(parents=True, exist_ok=True)
        (src / r).write_text(f"mesh {i}")
    (dst / "robot_assets/src/pkg/meshes/collision").mkdir(parents=True)
    os.chmod(dst / "robot_assets/src/pkg/meshes", 0o555)  # the asset copy is write-protected
    import hashlib

    sha = {r: hashlib.sha256((src / r).read_bytes()).hexdigest() for r in rel}
    man = tmp_path / "m.sha256"
    man.write_text("".join(f"{sha[r]}  {r}\n" for r in rel))

    def run(*extra, report="r.json"):
        return subprocess.run([sys.executable, "-I", str(tool), "--manifest", str(man), "--source", str(src),
                               "--target", str(dst), "--report", str(tmp_path / report), *extra],
                              capture_output=True, text=True)

    dry = run(report="dry.json")
    assert dry.returncode == 0 and not (dst / rel[0]).exists()
    assert json.loads((tmp_path / "dry.json").read_text())["counts"]["add"] == 2
    assert run(report="dry.json").returncode != 0  # never overwrites its own report either
    out = run("--apply", report="apply1.json")
    assert out.returncode == 0, out.stderr
    meshes = dst / "robot_assets/src/pkg/meshes"
    assert stat.S_IMODE(os.stat(meshes).st_mode) == 0o555  # restored exactly
    assert stat.S_IMODE(os.stat(meshes / "visual").st_mode) == 0o555
    assert stat.S_IMODE(os.stat(dst / rel[0]).st_mode) == 0o444
    rep = json.loads((tmp_path / "apply1.json").read_text())
    assert rep["counts"] == {"add": 0, "added": 2, "present": 0} and all(f["verified_sha256"] for f in rep["files"])
    assert rep["directories_created"] == ["robot_assets/src/pkg/meshes/visual"]
    assert (dst / rel[0]).read_text() == "mesh 0" and not list(dst.rglob(".*adding*"))
    again = run("--apply", report="apply.json")  # idempotent: identical files are present, nothing written
    assert again.returncode == 0
    rep = json.loads((tmp_path / "apply.json").read_text())
    assert rep["counts"] == {"add": 0, "added": 0, "present": 2}
    os.chmod(meshes / "visual", 0o755)
    os.chmod(dst / rel[0], 0o644)
    # a conflicting target file aborts before any write; a source that does not match the manifest aborts
    (dst / rel[0]).write_text("other")
    conflict = run("--apply", report="c.json")
    assert conflict.returncode != 0 and "never overwritten" in conflict.stderr and (dst / rel[0]).read_text() == "other"
    (src / rel[0]).write_text("tampered")
    assert run(report="t.json").returncode != 0


# ------------------------------------------------------------------------------------------------ depth scale
def _write_cal(root, ep, cam, status="accepted", **extra):
    d = rio.stage_dir(root, ep, "calibrate")
    d.mkdir(parents=True, exist_ok=True)
    rio.atomic_write_json(d / f"{rio.safe_name(cam)}.json", {"status": status, "T_cam_base": np.eye(4).tolist(),
                                                             **extra})


def test_depth_scale_comes_only_from_the_accepted_calibration_record(tmp_path):
    from s4d.realworld.pipeline import (
        LEGACY_ROBOT_MODEL,
        calibrated_version,
        calibration_depth_scale,
        calibration_robot_model,
    )

    root, ep = tmp_path, "e"
    _write_cal(root, ep, "a/left")  # legacy record: no depth_scale -> 1.0 (unchanged consumers)
    assert calibration_depth_scale(root, ep, "a/left", "foundation_stereo_depth") == 1.0
    _write_cal(root, ep, "b/left", depth_scale={"foundation_stereo_depth": 1.02, "sensor_depth": 1.01})
    assert calibration_depth_scale(root, ep, "b/left", "foundation_stereo_depth") == 1.02
    with pytest.raises(ValueError, match="not for 's2m2_depth'"):  # never borrowed from another depth source
        calibration_depth_scale(root, ep, "b/left", "s2m2_depth")
    _write_cal(root, ep, "c/left", status="rejected", depth_scale={"foundation_stereo_depth": 1.5})
    assert calibration_depth_scale(root, ep, "c/left", "foundation_stereo_depth") == 1.0  # no accepted pose
    _write_cal(root, ep, "d/left", depth_scale={"foundation_stereo_depth": 1.5})
    with pytest.raises(ValueError, match="outside"):
        calibration_depth_scale(root, ep, "d/left", "foundation_stereo_depth")
    assert calibration_depth_scale(root, ep, "missing/left", "foundation_stereo_depth") == 1.0
    assert calibrated_version("2", {}) == "2" and calibrated_version("2", {"depth_scale": 1}) == "2+cal1"

    class Meta:  # camera_views needs only views()
        def views(self):
            from types import SimpleNamespace
            return [SimpleNamespace(stream_id=c, kind="rgbd") for c in ("a/left", "b/left")]

    assert calibration_robot_model(root, ep, Meta()) == LEGACY_ROBOT_MODEL
    _write_cal(root, ep, "b/left", robot_model={"gripper_map": GRIPPER_MAP_2F85, "surface": "visual"})
    with pytest.raises(ValueError, match="disagree"):
        calibration_robot_model(root, ep, Meta())


def test_trackeverything_receives_scaled_depth_and_records_the_scale(tmp_path):
    from s4d.realworld.trackeverything_stage import VERSION, track_everything
    from tests.test_realworld_trackeverything import SyntheticRunner, episode

    a, root, ep = episode(tmp_path, frames=8)
    TE = {"grid_stride": 4, "model_hw": tuple(a.hw)}

    class Capture(SyntheticRunner):
        def worker(self, spec, task, params, arrays, *, chunk_id, allow_mock):
            self.depths = getattr(self, "depths", {})
            self.depths.setdefault(chunk_id.split("/")[1], arrays["depth"].copy())
            return super().worker(spec, task, params, arrays, chunk_id=chunk_id, allow_mock=allow_mock)

    from tests.test_realworld_trackeverything import MOCK_SPEC

    r0 = Capture(tmp_path / "w0", a)
    legacy = track_everything(root, ep, MOCK_SPEC, r0, depth_stage="sensor_depth", **TE)
    assert legacy["track2d"]["version"] == VERSION and "depth_scale" not in legacy["track2d"]["config"]
    cal = rio.stage_dir(root, ep, "calibrate") / "rig0__left.json"
    doc = json.loads(cal.read_text())
    cal.write_text(json.dumps({**doc, "depth_scale": {"sensor_depth": 1.04}}))
    r1 = Capture(tmp_path / "w1", a)
    scaled = track_everything(root, ep, MOCK_SPEC, r1, depth_stage="sensor_depth", **TE)
    assert scaled["track2d"]["version"] == VERSION + "+cal1" and scaled["lift3d"]["version"] == VERSION + "+cal1"
    factors = scaled["track2d"]["config"]["depth_scale"]["factors"]
    assert factors["rig0/left"] == 1.04 and factors["rig1/left"] == 1.0
    assert scaled["track2d"]["fingerprint"] != legacy["track2d"]["fingerprint"]
    np.testing.assert_allclose(r1.depths["rig0"], r0.depths["rig0"] * np.float32(1.04), rtol=1e-6)
    np.testing.assert_array_equal(r1.depths["rig1"], r0.depths["rig1"])


def test_train_store_scales_geometry_depth_only_and_keeps_proprioception_and_sets(tmp_path):
    pytest.importorskip("torch")
    from s4d.realworld import train_store as TS
    from s4d.realworld.pipeline import calibrated_view_depth, load_meta, view_depth
    from s4d.realworld.track_motion import MotionFilterConfig, track_motion
    from tests.test_realworld_pipeline import synthetic_rgbd_episode

    run = synthetic_rgbd_episode(tmp_path / "rw")
    root, ep = run["root"], run["ep"]
    track_motion(root, ep, cfg=MotionFilterConfig(min_bin_samples=10, min_segment_samples=4,
                                                  reference_robot_distance_m=0.05))
    kw = dict(task="synthetic", depth_stage="sensor_depth", allow_mock=True, semantic=False)
    legacy = TS.write_episode(root, ep, tmp_path / "legacy", **kw)
    scaled_root = tmp_path / "scaled"
    shutil.copytree(root, scaled_root)
    for rig in ("rig0", "rig1", "rig2"):
        p = rio.stage_dir(scaled_root, ep, "calibrate") / f"{rig}__left.json"
        d = json.loads(p.read_text())
        p.write_text(json.dumps({**d, "depth_scale": {"sensor_depth": 1.03, "foundation_stereo_depth": 1.02}}))
    meta = load_meta(scaled_root, ep)
    d0, _ = view_depth(scaled_root, ep, meta, "rig0/left", [0], "sensor_depth")
    d1, _ = calibrated_view_depth(scaled_root, ep, meta, "rig0/left", [0], "sensor_depth")
    if json.loads((rio.stage_dir(scaled_root, ep, "calibrate") / "rig0__left.json").read_text())["status"] == "accepted":
        np.testing.assert_allclose(d1, d0 * np.float32(1.03), rtol=1e-6)
    rec = TS.write_episode(scaled_root, ep, tmp_path / "store", **kw)
    assert rec["version"] == TS.WRITER_VERSION + "+cal1" and rec["config"]["depth_scale"]["stage"] == "sensor_depth"
    assert legacy["version"] == TS.WRITER_VERSION and "depth_scale" not in legacy["config"]
    with h5py.File(tmp_path / "legacy" / TS.FILE) as a, h5py.File(tmp_path / "store" / TS.FILE) as b:
        assert json.loads(b.attrs["calibration_derived"])["depth_scale"]["factors"]
        assert "calibration_derived" not in a.attrs
        np.testing.assert_array_equal(a["sync/surface_disp_m"][:], b["sync/surface_disp_m"][:])
        np.testing.assert_array_equal(a["sync/frames"][:], b["sync/frames"][:])
        for cam in json.loads(a.attrs["cameras"]):
            g, h = a[f"cameras/{cam}"], b[f"cameras/{cam}"]
            np.testing.assert_array_equal(g["robot/probe_state"][:], h["robot/probe_state"][:])
            np.testing.assert_array_equal(g["rgb"][:], h["rgb"][:])
            if "depth" in g:
                da, db = g["depth"][:], h["depth"][:]
                np.testing.assert_allclose(db[da > 0], da[da > 0] * np.float32(1.03), rtol=1e-6)


# ------------------------------------------------------------------------------------------------ release rule
def _release_doc(cams, ep, **over):
    from s4d.realworld.calibration_rig import FIXEDNESS_RULE_G5AB, FORMAT

    doc = {"format": FORMAT, "name": "rel-test", "task": "synthetic", "gate_version": "test-gate/1",
           "fixedness_rule": FIXEDNESS_RULE_G5AB,
           "fixedness_config": {"patch": 24, "stride": 12, "min_inlier_patches": 4, "max_dynamic_fraction": 0.5,
                                "min_gradient": 1.0, "search_px": 4, "segments": 1},
           "fixedness_evidence": {c: {ep: {"registration": "fixed", "epipolar": "ok"}} for c in cams},
           "fixedness_exception": {"eligible_cameras": [], "cameras": [], "decision": None, "evidence": [],
                                   "decided_by": None, "decided_at": None, "reference": None},
           "episode_checks": {"max_heldout_depth_median_m": 10.0, "min_explained_fraction": 0.0},
           "robot_model": {"gripper_map": GRIPPER_MAP_LEGACY, "surface": "collision"},
           "provenance": {"frozen_commit": "f" * 40, "verdict": {"sha256": "a" * 64}}, "pool_frames": {},
           "cameras": {c: {"status": "validated", "held_by": [], "reasons": [], "T_cam_base": np.eye(4).tolist(),
                           "depth_scale": {"sensor_depth": 1.01, "foundation_stereo_depth": 1.02},
                           "provenance": {"rig": "rel-test"}} for c in cams}}
    doc.update(over)
    return doc


def test_release_fixedness_rule_and_exception_validity():
    from s4d.realworld.calibration_rig import (
        FixednessConfig,
        fixedness_exception_problem,
        release_fixedness,
        rig_camera_status,
    )
    from s4d.realworld.rig_loma_publish import fill_exception

    cfg = FixednessConfig(segments=1)
    doc = _release_doc(["c0", "c1"], "ep")
    doc["cameras"]["c1"].update(status="held", held_by=["G5a"], reasons=["G5a: registration detects motion in 1"])
    doc["fixedness_exception"]["eligible_cameras"] = ["c1"]
    fixed = {"fixed": True, "segments": [{"patches": 40, "inlier_patches": 38, "shift_px": 0.01, "fixed": True}]}
    moved = {"fixed": False, "segments": [{"patches": 40, "inlier_patches": 38, "shift_px": 0.4, "fixed": False,
                                           "reason": "image shift 0.4 px > 0.25"}]}
    unver = {"fixed": False, "segments": [{"patches": 40, "inlier_patches": 10, "shift_px": 0.1, "fixed": False,
                                           "reason": "10/40 consistent patches"}]}
    assert rig_camera_status(doc, "c0")[0] == "validated"
    st, why, ex = rig_camera_status(doc, "c1")
    assert st == "held" and ex is None and "registration-fixedness check G5a" in why[0]
    # G5a: only a VERIFIABLE registration detects motion; an unverifiable one is reported, never counted as moved
    assert release_fixedness(doc, "c0", "ep", fixed, cfg)["fixed"]
    r = release_fixedness(doc, "c0", "ep", unver, cfg)
    assert r["fixed"] and not r["registration"]["verifiable"] and r["strict_segment_test_fixed"] is False
    r = release_fixedness(doc, "c0", "ep", moved, cfg)
    assert not r["fixed"] and "G5a" in r["reason"] and r["registration"]["moved"]
    # held-by-G5a camera without an exception: rejected on the registration-fixedness check, metrics still reported
    r = release_fixedness(doc, "c1", "ep", fixed, cfg)
    assert not r["fixed"] and "G5a" in r["reason"] and r["registration"]["verifiable"]
    # stored G5b evidence: a flagged or missing episode is rejected
    bad = json.loads(json.dumps(doc))
    bad["fixedness_evidence"]["c0"]["ep"]["epipolar"] = "moved"
    assert "G5b" in release_fixedness(bad, "c0", "ep", fixed, cfg)["reason"]
    assert "not covered" in release_fixedness(doc, "c0", "other-ep", fixed, cfg)["reason"]
    # exception validity
    assert "empty" in fixedness_exception_problem(doc, "c1") and "not eligible" in fixedness_exception_problem(doc, "c0")
    with pytest.raises(SystemExit, match="not eligible"):
        fill_exception(doc, cameras=["c0"], evidence=[("f", "b" * 64)], decision="accepted", decided_by="x",
                       reference="y")
    filled = fill_exception(doc, cameras=["c1"], evidence=[("fixreg/report.json", "b" * 64)], decision="accepted",
                            decided_by="coordinator", reference="HANDOFF 2026-11-08 FIXREG")
    assert doc["fixedness_exception"]["decision"] is None  # the published document is never modified
    st, why, ex = rig_camera_status(filled, "c1")
    assert st == "validated" and ex["decision"] == "accepted" and filled["name"].endswith("-fx-accepted")
    r = release_fixedness(filled, "c1", "ep", moved, cfg)
    assert r["fixed"] and r["exception"] and "covered by the fixedness exception" in r["notes"][0]
    with pytest.raises(SystemExit, match="already filled"):
        fill_exception(filled, cameras=["c1"], evidence=[("f", "b" * 64)], decision="accepted", decided_by="x",
                       reference="y")
    rejected = fill_exception(doc, cameras=["c1"], evidence=[("f", "b" * 64)], decision="rejected",
                              decided_by="x", reference="y")
    assert rig_camera_status(rejected, "c1")[0] == "held"
    nosha = json.loads(json.dumps(filled))
    nosha["fixedness_exception"]["evidence"] = [{"path": "f", "sha256": "short"}]
    assert "sha256" in fixedness_exception_problem(nosha, "c1")


@pytest.fixture(scope="module")
def rig_episode(tmp_path_factory):
    from s4d.realworld.calibration_rig import FixednessConfig, static_image
    from s4d.realworld.pipeline import camera_views, load_meta, read_frames
    from tests.test_realworld_pipeline import synthetic_rgbd_episode

    tmp = tmp_path_factory.mktemp("rel")
    r = synthetic_rgbd_episode(tmp)
    root, ep = r["root"], r["ep"]
    meta = load_meta(root, ep)
    cams = [v.stream_id for v in camera_views(meta)]

    def build_masks(t):
        for cam in cams:
            n = len(meta.stream_timestamps_ns[cam])
            hw = (meta.stream(cam).intrinsics.height, meta.stream(cam).intrinsics.width)
            sem = np.zeros((n, *hw), np.uint8)
            sem[:, 30:60, 50:90] = 1
            with h5py.File(t / f"{rio.safe_name(cam)}.h5", "w") as f:
                f.create_dataset("semantic", data=sem)
                f.attrs["class_names"] = json.dumps(["background", "robot", "object", "fixture"])
        return {}

    rio.run_stage(rio.stage_dir(root, ep, "masks_test"), stage="masks_test", version="1", config={}, inputs={},
                  build=build_masks)
    fcfg = FixednessConfig(**_release_doc(cams, ep)["fixedness_config"])

    def reference(dx=0):
        ref = {}
        for cam in cams:
            n = len(meta.stream_timestamps_ns[cam])
            pick = np.linspace(0, n - 1, min(fcfg.frames, n)).round().astype(int)
            with h5py.File(rio.stage_dir(root, ep, "masks_test") / f"{rio.safe_name(cam)}.h5") as f:
                sem = f["semantic"][()]
            grey, dyn = static_image(read_frames(root, ep, cam, pick), sem == 1, fcfg)
            ref[f"{rio.safe_name(cam)}__grey"] = np.roll(grey, dx, axis=1) if dx else grey
            ref[f"{rio.safe_name(cam)}__dynamic"] = dyn
        return ref

    return {"root": root, "ep": ep, "cams": cams, "adapter": r["adapter"], "reference": reference, "tmp": tmp}


def _calibrate_with(rig_episode, doc, ref, name):
    from s4d.realworld.calibration import CalibrationConfig
    from s4d.realworld.calibration_rig import publish_rig
    from s4d.realworld.pipeline import calibrate
    from tests.test_realworld_pipeline import CALIBRATION

    path = rig_episode["tmp"] / "rigs" / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    adapter = rig_episode["adapter"]
    for cam in doc["cameras"]:
        doc["cameras"][cam]["T_cam_base"] = adapter.T_cam_base(cam.split("/")[0], "left").tolist()
    publish_rig(path, doc, ref)
    cfg = CalibrationConfig(**{**CALIBRATION.__dict__, "evidence": "rgbd", "min_visible_fraction": 0.0})
    rec = calibrate(rig_episode["root"], rig_episode["ep"], None, landmark_links=[], cfg=cfg,
                    robot_mask_stage="masks_test", rig_calibration=str(path), depth_stage="sensor_depth")
    outs = {cam: json.loads((rio.stage_dir(rig_episode["root"], rig_episode["ep"], "calibrate") /
                             f"{rio.safe_name(cam)}.json").read_text()) for cam in doc["cameras"]}
    return rec, outs, path


def test_rig_mode_release_document_fail_closed_exception_and_record_fields(rig_episode):
    from s4d.realworld.calibration_rig import load_rig, publish_rig
    from s4d.realworld.pipeline import accepted_calibration_doc, calibration_depth_scale
    from s4d.realworld.rig_loma_publish import fill_exception

    root, ep, cams = rig_episode["root"], rig_episode["ep"], rig_episode["cams"]
    ref = rig_episode["reference"]()
    doc = _release_doc(cams, ep)
    doc["cameras"][cams[1]].update(status="held", held_by=["G5a"], reasons=["G5a: registration detects motion in 1"])
    doc["fixedness_exception"]["eligible_cameras"] = [cams[1]]
    rec, outs, path = _calibrate_with(rig_episode, json.loads(json.dumps(doc)), ref, "held")
    assert rec["config"]["rig_calibration"]["fixedness"]["segments"] == 1  # the document's G5a configuration
    held = outs[cams[1]]
    assert held["status"] == "rejected" and any("G5a" in x and "no valid fixedness exception" in x
                                                for x in held["reasons"])
    fx = held["diagnostics"]["fixedness"]
    assert fx["rule"] == "g5ab/1" and not fx["fixed"] and fx["registration"]["segments"] == 1
    assert "shift_px" in fx["segments"][0]  # registration metrics are reported for the rejected camera
    assert held["diagnostics"]["release"]["effective_rig_status"] == "held"
    accepted = [c for c in cams if outs[c]["status"] == "accepted"]
    assert accepted and cams[1] not in accepted
    for cam in accepted:
        out = outs[cam]
        assert out["depth_scale"] == {"sensor_depth": 1.01, "foundation_stereo_depth": 1.02}
        assert out["robot_model"] == doc["robot_model"] and out["diagnostics"]["fixedness"]["fixed"]
        assert out["diagnostics"]["depth_scale_applied"] == {"stage": "sensor_depth", "factor": 1.01}
        assert accepted_calibration_doc(root, ep, cam)["depth_scale"]["sensor_depth"] == 1.01
        assert calibration_depth_scale(root, ep, cam, "foundation_stereo_depth") == 1.02
    # coordinator fills the exception from FIXREG evidence -> a NEW document -> new calibrate fingerprint
    got, arrays, fp = load_rig(path)
    filled = fill_exception(got, cameras=[cams[1]], evidence=[("fixreg/verdict.json", "c" * 64)],
                            decision="accepted", decided_by="coordinator", reference="FIXREG report")
    rec2, outs2, _ = _calibrate_with(rig_episode, filled, ref, "filled")
    assert rec2["fingerprint"] != rec["fingerprint"] and rec2["inputs"]["rig_calibration"] != fp["rig_calibration"]
    ok = outs2[cams[1]]
    assert ok["diagnostics"]["fixedness"]["fixed"] and ok["diagnostics"]["release"]["fixedness_exception"]
    assert not any("G5a" in x for x in ok["reasons"])
    # a camera moved against the reference (verifiable 2 px shift): rejected unless the exception covers it
    rec3, outs3, _ = _calibrate_with(rig_episode, json.loads(json.dumps(filled)), rig_episode["reference"](dx=2),
                                     "moved")
    for cam in cams:
        fx = outs3[cam]["diagnostics"]["fixedness"]
        if fx["registration"]["moved"]:
            assert (cam == cams[1]) == fx["fixed"]
    assert any(outs3[c]["diagnostics"]["fixedness"]["registration"]["moved"] for c in (cams[0], cams[2]))
    # G5b evidence flags the episode / the episode is not covered: rejected
    flagged = json.loads(json.dumps(doc))
    flagged["fixedness_evidence"][cams[0]][ep]["epipolar"] = "moved"
    flagged["fixedness_evidence"][cams[2]] = {}
    _, outs4, _ = _calibrate_with(rig_episode, flagged, ref, "flagged")
    assert outs4[cams[0]]["status"] == "rejected" and "G5b" in outs4[cams[0]]["diagnostics"]["fixedness"]["reason"]
    assert "not covered" in outs4[cams[2]]["diagnostics"]["fixedness"]["reason"]
    # a release document must carry the scale of the depth stage the checks use
    noscale = json.loads(json.dumps(doc))
    for c in cams:
        noscale["cameras"][c]["depth_scale"] = {"foundation_stereo_depth": 1.02}
    with pytest.raises(ValueError, match="no depth scale"):
        _calibrate_with(rig_episode, noscale, ref, "noscale")
    # visual meshes requested by the document but absent from the (synthetic, primitive) robot: fail closed
    visual = json.loads(json.dumps(doc))
    visual["robot_model"] = {"gripper_map": GRIPPER_MAP_LEGACY, "surface": "visual"}
    from s4d.realworld.robot_visual import MissingVisualMeshError

    with pytest.raises(MissingVisualMeshError):
        _calibrate_with(rig_episode, visual, ref, "visual")
    publish_rig  # noqa: B018 (imported for the helper)


# ------------------------------------------------------------------------------------------------ publisher
def _fake_results(tmp, *, sam_fail=("roty+",), samfree_fail=(), moved=None, n_ep=10):
    """rig_loma result files of one task with the fields the release verdict reads."""
    from s4d.realworld.rig_loma_publish import CHALLENGE

    rng = np.random.default_rng(0)
    eps = [f"t-{k:06d}" for k in range(n_ep)]
    T = {c: np.eye(4).tolist() for c in CAMS}

    def state(jitter=0.0):
        Ts = {}
        for c in CAMS:
            M = np.eye(4)
            M[:3, 3] = rng.normal(0, jitter, 3)
            Ts[c] = M.tolist()
        return {"T_cam_base": Ts, "alpha": {c: 0.01 * i for i, c in enumerate(CAMS)},
                "tau_s": {c: 0.005 for c in CAMS}}

    def challenge(fail):
        out = {}
        for name in CHALLENGE:
            bad = name in fail
            out[name] = {"frames_worse": 200 if bad else 600, "frames": 445, "episodes_worse": 18, "episodes": 20,
                         "p_value": 0.03 if bad else 1e-9, "p_value_episode": 1e-4, "rejected": not bad}
        out["all_rejected"] = not fail
        return out

    pairs = ["135122070988-135122073201", "135122070988-135122076054", "135122073201-135122076054"]
    full = {"task": "t", "variant": "full", "fit_episodes": eps[:8], "heldout_episodes": eps[8:], "state": state(),
            "heldout": {"corr_challenge": challenge(samfree_fail)},
            "heldout_sam": {"extext_sam_static": {"all": {"line_median_px": 0.25, "transfer_ab_median_px": 2.0,
                                                          "transfer_ba_median_px": 1.5}},
                            "robot": {c: {"contour_median_px": 1.1} for c in CAMS},
                            "robot_alpha_depth": {c: {"depth_median_mm": 7.0, "explained_fraction": 0.97,
                                                      "o2m_fixed_median_px": 1.3} for c in CAMS},
                            "robot_raw_depth": {c: {} for c in CAMS}, "common_challenge_sam": challenge(sam_fail)},
            "per_episode_extext": {e: {p: {"median_px": 0.25, "se_px": 0.01} for p in pairs} for e in eps}}
    tmp.mkdir(parents=True)
    (tmp / "full.json").write_text(json.dumps(full))
    names = [f"start-{k}" for k in range(1, 9)] + [f"half-{k}" for k in range(5)] + ["group-0", "group-1",
                                                                                     "block-0", "block-1"]
    names += ["loco-135122070988", "loco-135122073201", "loco-135122076054"]
    for n in names:
        (tmp / f"{n}.json").write_text(json.dumps({"state": state(0.0005)}))
    allr = {"task": "t", "variant": "all", "fit_episodes": eps, "heldout_episodes": [], "state": state(),
            "config": {"x": 1}, "mount_dM": [0.0] * 6}
    seg = {"patches": 80, "inlier_patches": 78, "shift_px": 0.01, "fixed": True}
    fixed = {"reference_episode": eps[0], "config": {"segments": 1, "min_inlier_patches": 6},
             "episodes": {e: {c: {"segments": [dict(seg)]} for c in CAMS} for e in eps}}
    if moved:
        fixed["episodes"][eps[3]][moved]["segments"][0].update(fixed=False, shift_px=0.4)
    (tmp / "all.json").write_text(json.dumps(allr))
    return T, fixed, allr


def _frozen(results_dir, fixed):
    from s4d.realworld.rig_dw_publish import verdict
    from s4d.realworld.rig_loma_publish import load_results

    R = load_results(results_dir)
    cc = R["full"]["heldout"]["corr_challenge"]
    return {"rig_dw_gates": verdict(R, fixed, "full"),
            "samfree": {c: {"corr_challenge_all_rejected": cc["all_rejected"]} for c in CAMS}}


def _publish_args(tmp, rdir, fixed, frozen, out, eligible=()):
    from types import SimpleNamespace

    (tmp / "fixed.json").write_text(json.dumps(fixed))
    (tmp / "verdict.json").write_text(json.dumps(frozen))
    (tmp / "FREEZE.md").write_text("freeze")
    np.savez(tmp / "ref.reference.npz", **{f"{c.replace('/', '__')}__{k}": np.zeros((4, 4), np.float32 if k == "grey"
                                                                                     else bool)
                                          for c in CAMS for k in ("grey", "dynamic")})
    return SimpleNamespace(task="t", results=str(rdir), final=str(rdir / "all.json"), fixedness=str(tmp / "fixed.json"),
                           reference=str(tmp / "ref.reference.npz"), verdict=str(tmp / "verdict.json"),
                           freeze=str(tmp / "FREEZE.md"), commit="a935aef" + "0" * 33, depth_result=[],
                           exception_eligible=list(eligible), out=str(out))


def test_publisher_applies_the_release_gate_and_documents_it(tmp_path):
    from s4d.realworld.calibration_rig import load_rig, rig_camera_status
    from s4d.realworld.rig_loma_publish import GATE_VERSION, publish

    # drawer-like: SAM G6 fails roty+, SAM-free challenge rejects all 12 -> validated, SAM G6 kept as diagnostic
    rdir = tmp_path / "a" / "res"
    _, fixed, _ = _fake_results(rdir)
    a = _publish_args(tmp_path / "a", rdir, fixed, _frozen(rdir, fixed), tmp_path / "a" / "rig.json")
    doc = publish(a)
    assert "G6 = SAM-free corr_challenge primary; SAM G6 diagnostic" in GATE_VERSION
    assert doc["gate_version"] == GATE_VERSION and doc["provenance"]["frozen_commit"].startswith("a935aef")
    assert doc["provenance"]["verdict"]["sha256"] and "USER decision update" in doc["provenance"]["user_decision"]
    for c in CAMS:
        cam = doc["cameras"][c]
        assert cam["status"] == "validated" and cam["checks"]["G6"]
        assert cam["diagnostics"]["g6_sam"]["passed"] is False and "roty+" in cam["diagnostics"]["g6_sam"]["reasons"][0]
        assert cam["depth_scale"]["foundation_stereo_depth"] == pytest.approx(np.exp(0.01 * CAMS.index(c)))
    assert doc["robot_model"] == {"gripper_map": GRIPPER_MAP_2F85, "surface": "visual"}
    assert doc["fixedness_exception"]["eligible_cameras"] == [] and doc["fixedness_exception"]["decision"] is None
    assert set(doc["fixedness_evidence"][CAMS[0]]) == set(json.loads((rdir / "all.json").read_text())["fit_episodes"])
    assert load_rig(a.out)[0]["name"] == doc["name"]
    with pytest.raises(SystemExit, match="never replaced"):
        publish(a)
    # mug-like: G5a moved for 070988 -> held by G5a alone, eligible, exception empty -> effective status held
    rdir = tmp_path / "b" / "res"
    _, fixed, _ = _fake_results(rdir, sam_fail=(), moved=CAMS[0])
    frozen = _frozen(rdir, fixed)
    b = _publish_args(tmp_path / "b", rdir, fixed, frozen, tmp_path / "b" / "rig.json", eligible=[CAMS[0]])
    doc = publish(b)
    assert doc["cameras"][CAMS[0]]["held_by"] == ["G5a"] and doc["cameras"][CAMS[0]]["status"] == "held"
    assert rig_camera_status(doc, CAMS[0])[0] == "held" and rig_camera_status(doc, CAMS[1])[0] == "validated"
    assert doc["fixedness_evidence"][CAMS[0]]["t-000003"]["registration"] == "moved"
    # an eligible camera that is held by something else, a verdict that differs from the frozen one, an
    # inconsistent challenge record: refused
    c = _publish_args(tmp_path / "b", rdir, fixed, frozen, tmp_path / "b" / "rig2.json", eligible=[CAMS[1]])
    with pytest.raises(SystemExit, match="held by"):
        publish(c)
    bad = json.loads(json.dumps(frozen))
    bad["rig_dw_gates"][CAMS[2]]["reasons"] = ["G2: something"]
    d = _publish_args(tmp_path / "b", rdir, fixed, bad, tmp_path / "b" / "rig3.json", eligible=[CAMS[0]])
    with pytest.raises(SystemExit, match="frozen"):
        publish(d)
    full = json.loads((rdir / "full.json").read_text())
    full["heldout"]["corr_challenge"]["rotx+"]["rejected"] = False
    (rdir / "full.json").write_text(json.dumps(full))
    e = _publish_args(tmp_path / "b", rdir, fixed, frozen, tmp_path / "b" / "rig4.json", eligible=[CAMS[0]])
    with pytest.raises(SystemExit, match="disagrees"):
        publish(e)
    # SAM-free challenge failing -> held by G6 (the SAM evidence never rescues it either)
    rdir = tmp_path / "c" / "res"
    _, fixed, _ = _fake_results(rdir, sam_fail=(), samfree_fail=("roty+",))
    f = _publish_args(tmp_path / "c", rdir, fixed, _frozen(rdir, fixed), tmp_path / "c" / "rig.json")
    doc = publish(f)
    assert all(doc["cameras"][c]["held_by"] == ["G6"] for c in CAMS)


# ------------------------------------------------------------------------------------------------ records
def _fake_calibrate_outputs(root, ep, doc, rig_sha, *, status="accepted", build_id="b" * 32):
    d = rio.stage_dir(root, ep, "calibrate")
    d.mkdir(parents=True, exist_ok=True)
    rio.atomic_write_json(d / rio.RECORD, {"stage": "calibrate", "build_id": build_id, "fingerprint": "f" * 64,
                                              "config": {"rig_calibration": {"name": doc["name"]}},
                                              "inputs": {"rig_calibration": rig_sha,
                                                         "rig_reference": doc["reference_sha256"]}})
    for cam, c in doc["cameras"].items():
        rio.atomic_write_json(d / f"{rio.safe_name(cam)}.json", {
            "status": status, "reasons": [], "T_cam_base": c["T_cam_base"], "init_method": "rig_shared",
            "depth_scale": c["depth_scale"], "robot_model": doc["robot_model"],
            "diagnostics": {"release": {"rig": doc["name"]}, "heldout_depth_abs_median_m": 0.007,
                            "fixedness": {"fixed": True, "rule": "g5ab/1", "registration": {"moved": False},
                                          "exception": None}}})


def test_calibration_validated_records_satisfy_the_gate_and_refuse_failures(tmp_path):
    from s4d.realworld.calibration_release import OUTPUT_FORMAT, apply, plan
    from s4d.realworld.calibration_rig import load_rig, publish_rig
    from s4d.rwlearn.gate import calibration_record_problem

    cams = list(CAMS)
    doc = _release_doc(cams, "t-000000", task="drawer-open")
    path = tmp_path / "rig.json"
    publish_rig(path, doc, {f"{c.replace('/', '__')}__{k}": np.zeros((2, 2)) for c in cams for k in ("grey", "dynamic")})
    doc, _, fp = load_rig(path)
    manifest = {"schema": "rw-ingest-manifest", "schema_version": 2, "episodes": {}}
    sha = {}
    for k in range(3):
        ep = f"drawer-open-{k:06d}"
        key = f"drawer-open/episode_{k:06d}.h5"
        root = tmp_path / "remote" / ep
        _fake_calibrate_outputs(root, ep, doc, fp["rig_calibration"], build_id=f"{k}" * 32)
        sha[key] = f"{k}" * 64
        manifest["episodes"][key] = {"task": "drawer-open", "sha256": sha[key], "validation": {"status": "valid"},
                                     "preprocessing": {"schema": "rw-preprocess", "schema_version": 1,
                                                       "status": "done", "stages": {"calibrate": {
                                         "status": "done", "input_sha256": sha[key], "metrics": {"build_id": f"{k}" * 32},
                                         "outputs": str(rio.episode_dir(root, ep)).replace(str(tmp_path / "remote"),
                                                                                           "/remote")}}}}
    manifest["episodes"]["mug-hanging/episode_000000.h5"] = {"task": "mug-hanging", "validation": {"status": "valid"}}
    p = plan(manifest, {"drawer-open": str(path)}, path_map=[("/remote", str(tmp_path / "remote"))],
             code_commit="c" * 40)
    t = p["tasks"]["drawer-open"]
    assert not t["refused"], t["problems"]
    assert len(t["records"]) == 3
    for key, rec in t["records"].items():
        m = rec["metrics"]
        assert rec["output_format"] == OUTPUT_FORMAT and m["decision"] == "accepted"
        assert m["cameras"] == sorted(c.split("/")[0] for c in cams) and m["gate_version"] == "test-gate/1"
        assert m["shared_calibration"]["sha256"] == fp["rig_calibration"] and m["evidence"]["verdict_sha256"]
        stages = {"calibrate": manifest["episodes"][key]["preprocessing"]["stages"]["calibrate"],
                  "calibration_validated": rec}
        assert calibration_record_problem(stages, sha[key]) is None
    # dry run by default; the live manifest must still match; set_stage is the only writer
    state = tmp_path / "state"
    state.mkdir()
    (state / "manifest.json").write_text(json.dumps(manifest))
    calls = []
    rep = apply(p, state, set_stage=lambda *a: calls.append(a))
    assert not calls and rep["tasks"]["drawer-open"]["would_write"] == 3
    rep = apply(p, state, write=True, set_stage=lambda *a: calls.append(a))
    assert len(calls) == 3 and {c[2] for c in calls} == {"calibration_validated"}
    # the real ingest-worker writer accepts the records (field whitelist, sha256 check) and the gate releases them;
    # inside the release containers the runs/ bind mount hides the tracked worker (as tests/test_ops_layout.py
    # tool_path): that part runs on every host checkout, and its absence is only accepted inside a container
    from s4d.realworld.calibration_release import INGEST_WORKER

    if not INGEST_WORKER.is_file():
        assert Path("/.dockerenv").exists(), f"{INGEST_WORKER} is missing"
    else:
        real = tmp_path / "state-real"
        real.mkdir()
        (real / "manifest.json").write_text(json.dumps(manifest))
        rep = apply(p, real, write=True)
        assert rep["tasks"]["drawer-open"]["written"] == 3
        written = json.loads((real / "manifest.json").read_text())
        for key in t["records"]:
            stages = written["episodes"][key]["preprocessing"]["stages"]
            assert stages["calibration_validated"]["status"] == "done"
            assert calibration_record_problem(stages, sha[key]) is None
    changed = json.loads(json.dumps(manifest))
    changed["episodes"]["drawer-open/episode_000001.h5"]["preprocessing"]["stages"]["calibrate"]["metrics"][
        "build_id"] = "z" * 32
    (state / "manifest.json").write_text(json.dumps(changed))
    calls.clear()
    rep = apply(p, state, write=True, set_stage=lambda *a: calls.append(a))
    assert not calls and "changed since the plan" in rep["tasks"]["drawer-open"]["refused"]
    # any failed check refuses the whole task: a rejected camera, a different document, a held document camera
    root1 = tmp_path / "remote" / "drawer-open-000001"
    _fake_calibrate_outputs(root1, "drawer-open-000001", doc, fp["rig_calibration"], status="rejected",
                            build_id="1" * 32)
    p2 = plan(manifest, {"drawer-open": str(path)}, path_map=[("/remote", str(tmp_path / "remote"))])
    assert p2["tasks"]["drawer-open"]["refused"] and not p2["tasks"]["drawer-open"]["records"]
    _fake_calibrate_outputs(root1, "drawer-open-000001", doc, "0" * 64, build_id="1" * 32)
    p3 = plan(manifest, {"drawer-open": str(path)}, path_map=[("/remote", str(tmp_path / "remote"))])
    assert any("not built with the release document" in x for x in p3["tasks"]["drawer-open"]["problems"])
    _fake_calibrate_outputs(root1, "drawer-open-000001", doc, fp["rig_calibration"], build_id="1" * 32)
    held = json.loads(json.dumps(doc))
    held.pop("reference_sha256")
    held["cameras"][cams[0]].update(status="held", held_by=["G5a"])
    held["fixedness_exception"]["eligible_cameras"] = [cams[0]]
    hpath = tmp_path / "held.json"
    publish_rig(hpath, held, {f"{c.replace('/', '__')}__{k}": np.zeros((2, 2)) for c in cams
                              for k in ("grey", "dynamic")})
    p4 = plan(manifest, {"drawer-open": str(hpath)}, path_map=[("/remote", str(tmp_path / "remote"))])
    assert p4["tasks"]["drawer-open"]["refused"]
    assert any("no valid fixedness exception" in x for x in p4["tasks"]["drawer-open"]["problems"])


def test_check_episode_reads_the_completion_record_written_by_run_stage(tmp_path):
    """Regression (OPS 2026-11-09): the builder must read the calibrate stage's completion record under the name the
    stage writer uses (``rio.RECORD``, written by ``rio.run_stage``), not a fixture-only ``record.json``."""
    from s4d.realworld.calibration_release import check_episode
    from s4d.realworld.calibration_rig import load_rig, publish_rig

    cams = list(CAMS)
    doc = _release_doc(cams, "t-000000", task="drawer-open")
    path = tmp_path / "rig.json"
    publish_rig(path, doc, {f"{c.replace('/', '__')}__{k}": np.zeros((2, 2)) for c in cams for k in ("grey", "dynamic")})
    doc, _, fp = load_rig(path)
    ep = "drawer-open-000000"
    root = tmp_path / "root"

    def build(tmp):
        for cam, c in doc["cameras"].items():
            rio.atomic_write_json(tmp / f"{rio.safe_name(cam)}.json", {
                "status": "accepted", "reasons": [], "T_cam_base": c["T_cam_base"], "init_method": "rig_shared",
                "depth_scale": c["depth_scale"], "robot_model": doc["robot_model"],
                "diagnostics": {"release": {"rig": doc["name"]}, "heldout_depth_abs_median_m": 0.007,
                                "fixedness": {"fixed": True, "rule": "g5ab/1", "registration": {"moved": False},
                                              "exception": None}}})
        return {}

    rec = rio.run_stage(rio.stage_dir(root, ep, "calibrate"), stage="calibrate", version="1",
                        config={"rig_calibration": {"name": doc["name"]}},
                        inputs={"rig_calibration": fp["rig_calibration"], "rig_reference": doc["reference_sha256"]},
                        build=build)
    assert not (rio.stage_dir(root, ep, "calibrate") / "record.json").exists()
    entry = {"sha256": "a" * 64, "preprocessing": {"stages": {"calibrate": {
        "status": "done", "input_sha256": "a" * 64, "metrics": {"build_id": rec["build_id"]},
        "outputs": str(rio.episode_dir(root, ep))}}}}
    record, problems = check_episode("drawer-open/episode_000000.h5", entry, doc, fp["rig_calibration"])
    assert problems == [] and record["metrics"]["calibrate_build_id"] == rec["build_id"]
