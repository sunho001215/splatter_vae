"""Compose the arm + gripper URDF used by the real-world pipeline's robot-grounded stages (calibration, export).

Inputs are xacro-EXPANDED URDFs of the official descriptions (expand in an asset environment: the pipeline never runs
xacro). The gripper is attached with a mounting transform, recorded with its source: either a hardware configuration
from ``s4d.realworld.mounts`` (``--mount-config NAME``; its status - measured / provisional / assumption - part stack,
sources, scope and missing measurement go into the manifest) or explicit ``--mount-xyz/--mount-rpy/--mount-source``.
``package://NAME/`` mesh URIs are resolved to directories; ``--visual-from-collision`` replaces visual geometry by the
collision geometry (the pipeline loads STL/OBJ only, and many visual meshes are Collada). Writes the composed URDF and
a manifest with the sha256 of every input, then loads the result with ``s4d.realworld.robot.URDFModel`` and checks
every mesh.

``--coupling-geometry`` also adds the mount's coupling part (``MountConfig.coupling_geometry``) as a link fixed to the
parent link, so robot rendering and surface sampling include the coupling; it never changes the gripper transform.

    python scripts/realworld_compose_robot_urdf.py --arm fr3_arm.urdf --gripper robotiq_2f_85.urdf \
        --parent-link fr3_link8 --gripper-root world --mount-config droid_robotiq_2f85_iso50_coupling \
        --package franka_description=DIR --package robotiq_description=DIR --visual-from-collision --out robot.urdf
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from s4d.realworld.geometry import make_T  # noqa: E402
from s4d.realworld.robotics.model import RobotSurface, URDFModel, compose_urdf  # noqa: E402
from s4d.realworld.robotics.mounts import MOUNTS, T_parent_gripper_root, get_mount  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_packages(text: str, packages: dict) -> str:
    for name, directory in packages.items():
        text = text.replace(f"package://{name}/", f"{Path(directory).resolve()}/")
    if "package://" in text:
        raise ValueError("unresolved package:// URI; pass --package NAME=DIR")
    return text


def coupling_link_xml(parent_link: str, geometry) -> tuple[str, str]:
    """(<link> xml, <joint> xml) of a coupling part fixed to ``parent_link`` (``s4d.realworld.mounts.CouplingGeometry``)."""
    xyz = " ".join(f"{v:.9g}" for v in geometry.xyz)
    rpy = " ".join(f"{v:.12g}" for v in geometry.rpy)
    link = (f'<link name="{geometry.link}"><visual><geometry><mesh filename="{geometry.visual_mesh}"/></geometry>'
            f'</visual><collision><geometry><mesh filename="{geometry.collision_mesh}"/></geometry></collision></link>')
    joint = (f'<joint name="{geometry.link}_joint" type="fixed"><parent link="{parent_link}"/>'
             f'<child link="{geometry.link}"/><origin xyz="{xyz}" rpy="{rpy}"/></joint>')
    return link, joint


def add_coupling(arm_text: str, parent_link: str, geometry) -> str:
    root = ET.fromstring(arm_text)
    if any(e.get("name") == geometry.link for e in root if e.tag in ("link", "joint")):
        raise ValueError(f"name clash: {geometry.link!r} already in the arm URDF")
    link, joint = coupling_link_xml(parent_link, geometry)
    root.append(ET.fromstring(link))
    root.append(ET.fromstring(joint))
    return ET.tostring(root, encoding="unicode")


def visual_from_collision(text: str) -> str:
    root = ET.fromstring(text)
    for link in root.iter("link"):
        for v in link.findall("visual"):
            link.remove(v)
        for c in link.findall("collision"):
            v = copy.deepcopy(c)
            v.tag = "visual"
            link.append(v)
    return ET.tostring(root, encoding="unicode")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--gripper", type=Path, required=True)
    ap.add_argument("--parent-link", required=True)
    ap.add_argument("--gripper-root", required=True)
    ap.add_argument("--mount-config", choices=sorted(MOUNTS), help="mount configuration (s4d/realworld/mounts.py)")
    ap.add_argument("--mount-xyz", type=float, nargs=3)
    ap.add_argument("--mount-rpy", type=float, nargs=3, help="fixed-axis xyz (URDF rpy), radians")
    ap.add_argument("--mount-source", help="where an explicit mounting transform comes from")
    ap.add_argument("--package", action="append", default=[], help="NAME=DIR for package:// URIs")
    ap.add_argument("--visual-from-collision", action="store_true")
    ap.add_argument("--coupling-geometry", action="store_true",
                    help="add the mount config's coupling part as a link fixed to --parent-link (shape only)")
    ap.add_argument("--provenance", default="{}", help="JSON with upstream repos/commits of the descriptions")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    explicit = (args.mount_xyz, args.mount_rpy, args.mount_source)
    if (args.mount_config is not None and any(v is not None for v in explicit)) or (
        args.mount_config is None and any(v is None for v in explicit)
    ):
        ap.error("give either --mount-config or all of --mount-xyz/--mount-rpy/--mount-source")
    packages = dict(p.split("=", 1) for p in args.package)
    arm_text = Path(args.arm).read_text()
    coupling = None
    if args.coupling_geometry:
        coupling = get_mount(args.mount_config).coupling_geometry if args.mount_config else None
        if coupling is None:
            ap.error("--coupling-geometry needs a --mount-config that defines coupling_geometry")
        arm_text = add_coupling(arm_text, args.parent_link, coupling)
    arm, grip = (resolve_packages(text, packages) for text in (arm_text, Path(args.gripper).read_text()))
    if args.visual_from_collision:
        arm, grip = visual_from_collision(arm), visual_from_collision(grip)
    if args.mount_config:
        mount = get_mount(args.mount_config)
        fk = URDFModel.from_string(grip).fk({})
        if mount.gripper_frame not in fk or args.gripper_root not in fk:
            raise SystemExit(f"gripper URDF lacks {mount.gripper_frame!r} or root {args.gripper_root!r}")
        T_root_frame = np.linalg.inv(fk[args.gripper_root]) @ fk[mount.gripper_frame]
        T = T_parent_gripper_root(mount, T_root_frame)
        mount_doc = {"xyz": list(mount.xyz), "rpy": list(mount.rpy), "source": mount.source_text(),
                     **mount.manifest(), "T_root_gripper_frame": T_root_frame.tolist()}
    else:
        T = make_T(Rotation.from_euler("xyz", args.mount_rpy).as_matrix(), np.asarray(args.mount_xyz, float))
        mount_doc = {"xyz": args.mount_xyz, "rpy": args.mount_rpy, "source": args.mount_source,
                     "gripper_frame": args.gripper_root}
    composed = compose_urdf(arm, grip, args.parent_link, args.gripper_root, T)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(composed)
    model = URDFModel.from_file(args.out)
    surface = RobotSurface.build(model, points_per_link=50, seed=0)  # loads every mesh
    manifest = {
        "urdf": str(args.out), "urdf_sha256": sha256(args.out),
        "inputs": {"arm": {"path": str(args.arm), "sha256": sha256(args.arm)},
                   "gripper": {"path": str(args.gripper), "sha256": sha256(args.gripper)}},
        "mount": {"parent_link": args.parent_link, "gripper_root": args.gripper_root, **mount_doc,
                  "T_parent_gripper": T.tolist()},
        "packages": packages, "visual_from_collision": args.visual_from_collision,
        **({"coupling_geometry": {"parent_link": args.parent_link, **coupling.manifest()}} if coupling else {}),
        "provenance": json.loads(args.provenance),
        "links": sorted(model.links), "sampled_links": sorted(surface.link_points),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("urdf", "urdf_sha256", "mount")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
