"""Flange -> gripper mounting transforms (``T_flange_gripper``), one entry per hardware configuration, with provenance.

A mount is the rigid transform from the arm flange frame (Franka ``panda_link8`` / ``fr3_link8``: the flange mounting
face, DH ``d = 0.107`` m beyond joint 7) to a named frame of the gripper URDF. It is used only to compose the arm and
gripper URDFs (``scripts/realworld_compose_robot_urdf.py``), i.e. for robot rendering and gripper surface points in the
real-world pipeline. It is NOT:

* the DROID large-scale approximate gripper-point offset ``data.gripper_offset_m`` (``s4d/data/droid/gripper.py``): a
  constant fitted shift of a hand-made 32-point model expressed in DROID's reported EE frame (= flange), which never
  reads this transform;
* the DROID-3D wrist hand-eye ``T_gw_static`` / ``dT_gw`` (camera <- flange): the ZED Mini bracket is screwed to the
  link-7 ring beside the flange, so the gripper coupling is not part of that chain.

Each entry records the stack of parts along the flange z axis with the source of every dimension and the robots it
applies to. ``status``:

* ``measured``: measured on the actual hardware (none yet);
* ``provisional``: derived from vendor drawings whose part identity / interfaces are inferred (e.g. from photos), and
  corroborated but not measured; applied as a default until a measurement or the actual part's drawing exists;
* ``assumption``: a convention without a hardware basis.

``missing_measurement`` names what would turn a provisional entry into a measured one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from s4d.realworld.geometry import make_T

ROBOTIQ_GEN_MANUAL = (
    "Robotiq 2F-85 & 2F-140 Instruction Manual (generic, HTML 2019-05-24), section 6 Specifications: "
    "https://assets.robotiq.com/website-assets/support_documents/document/online/"
    "2F-85_2F-140_Instruction_Manual_Gen_HTML_20190524.zip/2F-85_2F-140_Instruction_Manual_Gen_HTML/Content/"
    "6.%20Specifications.htm"
)


@dataclass(frozen=True)
class StackElement:
    part: str
    dz_m: float  # contribution along flange +z to the gripper frame origin
    source: str


@dataclass(frozen=True)
class CouplingGeometry:
    """Shape of the coupling part itself (a link fixed to the flange), for rendering and robot-surface sampling. It
    carries no kinematics: the gripper frame is placed by ``MountConfig.xyz/rpy`` alone. Mesh URIs are
    ``package://`` paths resolved by the composer; ``xyz/rpy`` place the mesh frame in the flange frame."""

    link: str
    visual_mesh: str
    collision_mesh: str
    xyz: tuple
    rpy: tuple
    status: str  # "nominal" (vendor model of the documented part) | "measured" | "assumption"
    source: str
    caveats: tuple = ()

    def manifest(self) -> dict:
        return {"link": self.link, "visual_mesh": self.visual_mesh, "collision_mesh": self.collision_mesh,
                "xyz": list(self.xyz), "rpy": list(self.rpy), "status": self.status, "source": self.source,
                "caveats": list(self.caveats)}


@dataclass(frozen=True)
class MountConfig:
    name: str
    gripper_frame: str  # gripper-URDF link whose pose in the flange frame is given by (xyz, rpy)
    xyz: tuple
    rpy: tuple  # URDF fixed-axis roll-pitch-yaw, radians
    status: str  # "measured" | "provisional" | "assumption"
    stack: tuple  # StackElement, flange face -> gripper frame
    yaw_source: str
    applies_to: str
    cross_checks: tuple = ()
    caveats: tuple = ()
    missing_measurement: str = ""
    # optional shape of the coupling part (scripts/realworld_compose_robot_urdf.py --coupling-geometry)
    coupling_geometry: CouplingGeometry | None = None

    def T(self) -> np.ndarray:
        return make_T(Rotation.from_euler("xyz", self.rpy).as_matrix(), np.asarray(self.xyz, float))

    def source_text(self) -> str:
        return f"{self.name} ({self.status}): " + "; ".join(f"{e.part} {e.dz_m * 1e3:+.1f} mm" for e in self.stack)

    def manifest(self) -> dict:
        return {
            "config": self.name, "status": self.status, "gripper_frame": self.gripper_frame,
            "stack": [{"part": e.part, "dz_m": e.dz_m, "source": e.source} for e in self.stack],
            "stack_dz_m": float(sum(e.dz_m for e in self.stack)), "yaw_source": self.yaw_source,
            "applies_to": self.applies_to, "cross_checks": list(self.cross_checks), "caveats": list(self.caveats),
            "missing_measurement": self.missing_measurement,
        }


# PROVISIONAL. Robotiq ISO 9409-1-50-4-M6 coupling (GRP-CPL-062 drawing). The DROID guide names only a "gripper mount"
# from the gripper box; its photos show a Robotiq pigtail coupling (AGC-CPL-062-002 family) directly on the flange,
# but the exact part/revision and its gripper-side recess are not established. Section A-A: 16.9 mm overall, robot
# side has a 3 mm deep Ø63 F8 recess that takes the Ø63 ISO 9409-1-A50 flange; the gripper side has the Ø71 recess
# (2.9 mm deep on the dimensioned blank coupling AGC-CPL-BLANK-002, 13.9 mm overall) that takes the gripper's Ø71 h8
# boss. The PickNik/ROS-Industrial 2F-85 base frame (robotiq_85_base_link == robotiq_arg2f_base_link) is the face of
# that boss: the URDF fingertip tops sit at 149.3 mm (open) and 162.8 mm (closed) above it, Robotiq's documented
# gripper heights.
_ISO50_STACK = (
    StackElement("Robotiq ISO 9409-1-50-4-M6 coupling, overall thickness (GRP-CPL-062 section A-A)", 0.0169,
                 ROBOTIQ_GEN_MANUAL + " (6.1.1, Coupling_C-062 drawing: 16.9)"),
    StackElement("minus robot-side Ø63 F8 recess seated on the ISO 9409-1-A50 flange (flange face = link8)", -0.0030,
                 ROBOTIQ_GEN_MANUAL + " (6.1.1, Coupling_C-062 drawing: 3)"),
    StackElement("minus gripper-side Ø71 recess holding the gripper's Ø71 h8 boss (= gripper frame origin)", -0.0029,
                 ROBOTIQ_GEN_MANUAL + " (6.1.1, AGC-CPL-BLANK-002 drawing: 13.9 overall, 2.9 recess; "
                 "2F-85 general dimensions: Ø71 h8)"),
)

MOUNTS = {
    m.name: m for m in (
        MountConfig(
            name="droid_robotiq_2f85_iso50_coupling",
            gripper_frame="robotiq_85_base_link",
            xyz=(0.0, 0.0, 0.011), rpy=(0.0, 0.0, np.pi / 2), status="provisional", stack=_ISO50_STACK,
            yaw_source="Franka external_gripper_example fe4f871 robotiq_mounted_on_panda.urdf.xacro (rpy 0 0 +pi/2); "
                       "+/-90 deg confirmed on DROID images (item5 mount_check); 180 deg is not identifiable for the 2F-85",
            applies_to=('PROVISIONAL default for DROID robots built per the DROID assembly guide '
                "(droid-dataset/droid docs/hardware-setup/assembly.md 'Mounting Robotiq Gripper on Robot'):"
                ' Robotiq 2F-85 on the Robotiq ISO 50 coupling screwed directly to the Franka flange; '
                'Franka Emika Panda and Franka Research 3 alike (same ISO 9409-1-A50 flange, same flange '
                "frame: DROID's polymetis ee_link is panda_link8 for both); the ZED Mini bracket screws to "
                'the link-7 ring and is not in this stack'),
            cross_checks=(
                "Robotiq TCP table (6.2.3, coupling included): 2F-85 TCP z 174.0 mm from the robot flange; closed "
                "height 162.8 mm above the gripper frame -> 11.2 mm",
                "PickNik ros2_robotiq_gripper a74d007 ur_to_robotiq_adapter.urdf.xacro (same ISO 50 coupling on UR): "
                "gripper_side_joint z 0.011; adapter mesh z -3.0 .. +13.9 mm",
            ),
            caveats=(
                "derived from Robotiq drawings, not measured on a DROID robot",
                "16.9 mm and 3.0 mm come from the GRP-CPL-062 drawing; the 2.9 mm gripper-side recess comes from the "
                "blank coupling AGC-CPL-BLANK-002 and is inferred to be the same on the DROID coupling",
                "coupling identity (Robotiq ISO-50 coupling with pigtail, no spacer) is inferred from DROID guide photos; "
                "part number/revision unknown",
                "assumes the Franka flange spigot fully seats in the coupling's 3 mm Ø63 recess",
                "the TCP table and the PickNik UR adapter corroborate the magnitude but do not identify DROID's part",
                "single-episode image fits (11 mm mask vs 17 mm depth optima) do not discriminate; see docs",
            ),
            missing_measurement="on a guide-built DROID robot: distance along flange z from the Franka flange face "
                                "(link8) to the Robotiq 2F-85 Ø71 boss face (robotiq_85_base_link); the coupling's "
                                "part number/revision (or its dimensioned drawing/CAD); the gripper yaw relative to "
                                "the flange dowel-pin hole",
            coupling_geometry=CouplingGeometry(
                link="robotiq_iso50_coupling",
                visual_mesh="package://robotiq_description/meshes/visual/2f_85/ur_to_robotiq_adapter.dae",
                collision_mesh="package://robotiq_description/meshes/collision/2f_85/ur_to_robotiq_adapter.stl",
                xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, np.pi / 2), status="nominal",
                source="PickNik ros2_robotiq_gripper a74d007 ur_to_robotiq_adapter.urdf.xacro: Robotiq's ISO "
                       "9409-1-50-4-M6 coupling mesh, mesh frame on the robot flange face (mesh z -3.0 .. +13.9 mm), "
                       "same yaw as the gripper (the gripper sits on the coupling without extra rotation)",
                caveats=("vendor model of the plain ISO-50 coupling; the pigtail variant seen in the DROID guide "
                         "photos has a side cable exit and spring guard that are not modelled",
                         "its axial placement follows the provisional stack above (not measured)")),
        ),
        MountConfig(
            name="franka_external_gripper_example",
            gripper_frame="robotiq_85_base_link",
            xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, np.pi / 2), status="assumption",
            stack=(StackElement("no coupling (gripper base on the flange face)", 0.0,
                                "Franka external_gripper_example fe4f871 robotiq_mounted_on_panda.urdf.xacro"),),
            yaw_source="Franka external_gripper_example fe4f871 robotiq_mounted_on_panda.urdf.xacro (rpy 0 0 +pi/2)",
            applies_to="legacy item-5 baseline asset only; omits the Robotiq coupling",
            caveats=("omits the 11.0 mm Robotiq ISO 50 coupling stack",),
        ),
    )
}


def get_mount(name: str) -> MountConfig:
    if name not in MOUNTS:
        raise KeyError(f"unknown mount config {name!r}; known: {sorted(MOUNTS)}")
    return MOUNTS[name]


def T_parent_gripper_root(mount: MountConfig, T_root_frame: np.ndarray) -> np.ndarray:
    """Transform to put on the composing joint (flange -> gripper URDF root) so that ``mount.gripper_frame`` lands at
    ``mount.T()``; ``T_root_frame`` is that frame's pose in the gripper URDF root (identity for PickNik's ``world``)."""
    return mount.T() @ np.linalg.inv(np.asarray(T_root_frame, float))


