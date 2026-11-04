"""Decode a composed arm/gripper manifest shared by recording adapters."""

import json
from pathlib import Path

from s4d.realworld.contract import RobotDescription


def _composed_robot(rc, *, gripper_state_semantics):
    manifest = json.loads(Path(rc["urdf_path"]).with_suffix(".manifest.json").read_text())
    robot = RobotDescription(
        arm_model="franka_panda", gripper_model=rc.get("gripper_model", "robotiq_2f_85"),
        urdf_path=str(rc["urdf_path"]), mesh_root=None, base_frame=rc["base_frame"],
        flange_frame=rc["flange_frame"], gripper_base_frame=manifest["mount"]["gripper_root"],
        T_flange_gripper=tuple(map(tuple, manifest["mount"]["T_parent_gripper"])),
        mount_source=manifest["mount"]["source"], joint_names=tuple(rc["joint_names"]),
        gripper_state_semantics=gripper_state_semantics, gripper_joint=rc["gripper_joint"],
        asset_sha256={"urdf": manifest["urdf_sha256"]})
    return robot, manifest
