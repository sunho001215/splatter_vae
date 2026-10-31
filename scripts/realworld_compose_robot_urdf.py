"""Compose the arm + gripper URDF used by the real-world pipeline's robot-grounded stages (calibration, export).

Inputs are xacro-EXPANDED URDFs of the official descriptions (expand in an asset environment: the pipeline never runs
xacro). The gripper is attached with an explicit mounting transform, recorded with its source; ``package://NAME/``
mesh URIs are resolved to directories; ``--visual-from-collision`` replaces visual geometry by the collision geometry
(the pipeline loads STL/OBJ only, and many visual meshes are Collada). Writes the composed URDF and a manifest with
the sha256 of every input, then loads the result with ``s4d.realworld.robot.URDFModel`` and checks every mesh.

    python scripts/realworld_compose_robot_urdf.py --arm fr3_arm.urdf --gripper robotiq_2f_85.urdf \
        --parent-link fr3_link8 --gripper-root world --mount-xyz 0 0 0 --mount-rpy 0 0 1.5707963267948966 \
        --mount-source "..." --package franka_description=DIR --package robotiq_description=DIR \
        --visual-from-collision --out robot.urdf
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
from s4d.realworld.robot import RobotSurface, URDFModel, compose_urdf  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_packages(text: str, packages: dict) -> str:
    for name, directory in packages.items():
        text = text.replace(f"package://{name}/", f"{Path(directory).resolve()}/")
    if "package://" in text:
        raise ValueError("unresolved package:// URI; pass --package NAME=DIR")
    return text


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--gripper", type=Path, required=True)
    ap.add_argument("--parent-link", required=True)
    ap.add_argument("--gripper-root", required=True)
    ap.add_argument("--mount-xyz", type=float, nargs=3, required=True)
    ap.add_argument("--mount-rpy", type=float, nargs=3, required=True, help="fixed-axis xyz (URDF rpy), radians")
    ap.add_argument("--mount-source", required=True, help="where the mounting transform comes from")
    ap.add_argument("--package", action="append", default=[], help="NAME=DIR for package:// URIs")
    ap.add_argument("--visual-from-collision", action="store_true")
    ap.add_argument("--provenance", default="{}", help="JSON with upstream repos/commits of the descriptions")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    packages = dict(p.split("=", 1) for p in args.package)
    arm, grip = (resolve_packages(Path(p).read_text(), packages) for p in (args.arm, args.gripper))
    if args.visual_from_collision:
        arm, grip = visual_from_collision(arm), visual_from_collision(grip)
    T = make_T(Rotation.from_euler("xyz", args.mount_rpy).as_matrix(), np.asarray(args.mount_xyz, float))
    composed = compose_urdf(arm, grip, args.parent_link, args.gripper_root, T)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(composed)
    model = URDFModel.from_file(args.out)
    surface = RobotSurface.build(model, points_per_link=50, seed=0)  # loads every mesh
    manifest = {
        "urdf": str(args.out), "urdf_sha256": sha256(args.out),
        "inputs": {"arm": {"path": str(args.arm), "sha256": sha256(args.arm)},
                   "gripper": {"path": str(args.gripper), "sha256": sha256(args.gripper)}},
        "mount": {"parent_link": args.parent_link, "gripper_root": args.gripper_root, "xyz": args.mount_xyz,
                  "rpy": args.mount_rpy, "T_parent_gripper": T.tolist(), "source": args.mount_source},
        "packages": packages, "visual_from_collision": args.visual_from_collision,
        "provenance": json.loads(args.provenance),
        "links": sorted(model.links), "sampled_links": sorted(surface.link_points),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("urdf", "urdf_sha256", "mount")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
