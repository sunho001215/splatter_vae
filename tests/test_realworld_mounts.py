import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from s4d.realworld.mounts import MOUNTS, T_parent_gripper_root, get_mount
from s4d.realworld.robot import URDFModel
from s4d.realworld.synthetic import synthetic_panda_urdf

ROOT = Path(__file__).resolve().parents[1]


def _compose_script():
    spec = importlib.util.spec_from_file_location("compose", ROOT / "scripts/realworld_compose_robot_urdf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _robotiq_like_urdf(root_to_base_z: float = 0.0) -> str:
    """PickNik-style topology: a ``world`` root joined to ``robotiq_85_base_link`` (identity upstream)."""
    return (
        '<robot name="g"><link name="world"/>'
        '<link name="robotiq_85_base_link"><visual><origin xyz="0 0 0.045"/><geometry><box size="0.075 0.085 0.09"/>'
        "</geometry></visual></link>"
        '<joint name="robotiq_85_base_joint" type="fixed"><parent link="world"/><child link="robotiq_85_base_link"/>'
        f'<origin xyz="0 0 {root_to_base_z}" rpy="0 0 0"/></joint></robot>'
    )


def test_droid_mount_is_the_documented_robotiq_iso50_stack():
    m = get_mount("droid_robotiq_2f85_iso50_coupling")
    assert m.status == "documented" and m.gripper_frame == "robotiq_85_base_link"
    # 16.9 mm coupling - 3.0 mm robot-side recess - 2.9 mm gripper-side recess (Robotiq drawings)
    assert sum(e.dz_m for e in m.stack) == pytest.approx(0.011, abs=1e-12)
    T = m.T()
    np.testing.assert_allclose(T[:3, 3], [0, 0, 0.011], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)  # +90 deg about flange z
    assert "Panda" in m.applies_to and "Research 3" in m.applies_to
    # Robotiq TCP-table cross-check: 174.0 mm (coupling included) - 162.8 mm closed height ~ the stack
    assert abs((0.174 - 0.1628) - T[2, 3]) < 0.0005


def test_every_mount_stack_matches_its_translation_and_is_rigid():
    for m in MOUNTS.values():
        T = m.T()
        assert sum(e.dz_m for e in m.stack) == pytest.approx(m.xyz[2], abs=1e-12)
        np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12)
        assert m.status in ("documented", "assumption")
    assert MOUNTS["franka_external_gripper_example"].status == "assumption"
    with pytest.raises(KeyError):
        get_mount("droid_unknown")


def test_mount_targets_the_named_gripper_frame_even_with_an_offset_root():
    m = get_mount("droid_robotiq_2f85_iso50_coupling")
    grip = URDFModel.from_string(_robotiq_like_urdf(0.02))
    fk = grip.fk({})
    T_joint = T_parent_gripper_root(m, fk["robotiq_85_base_link"])
    np.testing.assert_allclose(T_joint @ fk["robotiq_85_base_link"], m.T(), atol=1e-12)


@pytest.mark.parametrize("root_z", [0.0, 0.02])
def test_compose_script_mount_config_records_provenance(tmp_path, root_z):
    arm, grip = tmp_path / "arm.urdf", tmp_path / "grip.urdf"
    arm.write_text(synthetic_panda_urdf())
    grip.write_text(_robotiq_like_urdf(root_z))
    out = tmp_path / "robot.urdf"
    C = _compose_script()
    args = ["--arm", str(arm), "--gripper", str(grip), "--parent-link", "panda_link8", "--gripper-root", "world"]
    assert C.main(args + ["--mount-config", "droid_robotiq_2f85_iso50_coupling", "--out", str(out)]) == 0
    man = json.loads(out.with_suffix(".manifest.json").read_text())["mount"]
    assert man["config"] == "droid_robotiq_2f85_iso50_coupling" and man["status"] == "documented"
    assert man["stack_dz_m"] == pytest.approx(0.011) and len(man["stack"]) == 3
    np.testing.assert_allclose(np.asarray(man["T_parent_gripper"])[:3, 3], [0, 0, 0.011 - root_z], atol=1e-12)
    model = URDFModel.from_file(out)
    fk = model.fk({})
    T_flange_base = np.linalg.inv(fk["panda_link8"]) @ fk["robotiq_85_base_link"]
    np.testing.assert_allclose(T_flange_base, get_mount("droid_robotiq_2f85_iso50_coupling").T(), atol=1e-9)


def test_compose_script_explicit_mount_still_works_and_modes_are_exclusive(tmp_path):
    arm, grip = tmp_path / "arm.urdf", tmp_path / "grip.urdf"
    arm.write_text(synthetic_panda_urdf())
    grip.write_text(_robotiq_like_urdf())
    out = tmp_path / "robot.urdf"
    C = _compose_script()
    base = ["--arm", str(arm), "--gripper", str(grip), "--parent-link", "panda_link8", "--gripper-root", "world",
            "--out", str(out)]
    explicit = ["--mount-xyz", "0", "0", "0", "--mount-rpy", "0", "0", "1.5707963267948966", "--mount-source", "test"]
    assert C.main(base + explicit) == 0
    man = json.loads(out.with_suffix(".manifest.json").read_text())["mount"]
    assert man["source"] == "test" and "config" not in man
    with pytest.raises(SystemExit):
        C.main(base + explicit + ["--mount-config", "droid_robotiq_2f85_iso50_coupling"])
    with pytest.raises(SystemExit):
        C.main(base + ["--mount-xyz", "0", "0", "0"])
