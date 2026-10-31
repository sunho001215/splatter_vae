"""Robot geometry from URDF: forward kinematics, mimic joints, mounting, and surface/landmark sampling.

Robot geometry for calibration comes from the full articulated model (arm links + the actual Robotiq gripper
meshes driven by the recorded gripper state), never from the EE pose alone.

What must be supplied for the real robot (see docs/realworld_preprocessing.md, "Robot assets"):

* a Franka Panda URDF (``franka_description``; joint8/flange = panda_link8 at +0.107 m z of panda_link7);
* the URDF + meshes of the *exact* Robotiq model on the robot (2F-85 or 2F-140 - the planned "Robotiq adaptive
  gripper" is not further specified); STL or OBJ meshes (Collada .dae visual meshes must be converted, or the
  collision STLs used);
* the measured mounting transform ``T_flange_gripper`` (panda_link8 -> gripper base link) including the coupling
  thickness and the yaw (Franka's ``external_gripper_example`` uses xyz 0 0 0, rpy 0 0 +pi/2; DROID's code
  comments the same +pi/2 yaw; the coupling offset must be measured on the real robot);
* the mapping from the recorded gripper state to the URDF gripper joint (``gripper_joint_value``).

``compose_urdf`` attaches the gripper URDF to the arm with that transform. Without these assets the calibration
stage refuses to run. Mesh files are parsed with numpy only (STL binary/ascii, OBJ).
"""

from __future__ import annotations

import ast
import hashlib
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s4d.realworld.geometry import make_T


def _number(token: str) -> float:
    """A float, or a tiny arithmetic expression over numbers and ``pi`` (some URDFs write "pi/2"); no eval."""
    try:
        return float(token)
    except ValueError:
        pass

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return float(np.pi)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -ev(node.operand) if isinstance(node.op, ast.USub) else ev(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = ev(node.left), ev(node.right)
            return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b, ast.Div: a / b if b else np.nan}[type(node.op)]
        raise ValueError(f"unsupported URDF number {token!r}")

    return ev(ast.parse(token, mode="eval").body)


def _floats(text: str | None, n: int, default: float = 0.0) -> np.ndarray:
    if text is None:
        return np.full(n, default, float)
    vals = [_number(v) for v in text.split()]
    if len(vals) != n:
        raise ValueError(f"expected {n} numbers, got {text!r}")
    return np.array(vals)


def origin_T(elem) -> np.ndarray:
    if elem is None:
        return np.eye(4)
    xyz = _floats(elem.get("xyz"), 3)
    rpy = _floats(elem.get("rpy"), 3)
    return make_T(Rotation.from_euler("xyz", rpy).as_matrix(), xyz)  # URDF fixed-axis roll, pitch, yaw


@dataclass
class Geometry:
    kind: str  # box / cylinder / sphere / mesh
    T_link_geom: np.ndarray
    size: tuple = ()  # box: (x,y,z); cylinder: (radius, length); sphere: (radius,)
    mesh: str | None = None
    scale: tuple = (1.0, 1.0, 1.0)


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    T_parent_joint: np.ndarray
    axis: np.ndarray
    mimic: tuple | None = None  # (joint, multiplier, offset)
    limits: tuple | None = None


@dataclass
class URDFModel:
    name: str
    links: dict  # name -> list[Geometry]
    joints: dict  # name -> Joint
    root: str
    source: str = ""
    sha256: str = ""
    _children: dict = field(default_factory=dict)

    @classmethod
    def from_string(cls, text: str, *, source: str = "", use: str = "visual") -> URDFModel:
        root = ET.fromstring(text)
        links: dict = {}
        for link in root.findall("link"):
            geoms = []
            for item in link.findall(use) or (link.findall("collision") if use == "visual" else []):
                g = item.find("geometry")
                if g is None:
                    continue
                T = origin_T(item.find("origin"))
                if (b := g.find("box")) is not None:
                    geoms.append(Geometry("box", T, tuple(_floats(b.get("size"), 3))))
                elif (c := g.find("cylinder")) is not None:
                    geoms.append(Geometry("cylinder", T, (float(c.get("radius")), float(c.get("length")))))
                elif (s := g.find("sphere")) is not None:
                    geoms.append(Geometry("sphere", T, (float(s.get("radius")),)))
                elif (m := g.find("mesh")) is not None:
                    scale = tuple(_floats(m.get("scale"), 3, 1.0)) if m.get("scale") else (1.0, 1.0, 1.0)
                    geoms.append(Geometry("mesh", T, mesh=m.get("filename"), scale=scale))
            links[link.get("name")] = geoms
        joints = {}
        for j in root.findall("joint"):
            mimic = j.find("mimic")
            lim = j.find("limit")
            joints[j.get("name")] = Joint(
                name=j.get("name"),
                kind=j.get("type"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                T_parent_joint=origin_T(j.find("origin")),
                axis=_floats(j.find("axis").get("xyz"), 3) if j.find("axis") is not None else np.array([1.0, 0, 0]),
                mimic=(mimic.get("joint"), float(mimic.get("multiplier", 1)), float(mimic.get("offset", 0)))
                if mimic is not None else None,
                limits=(float(lim.get("lower", 0)), float(lim.get("upper", 0))) if lim is not None else None,
            )
        children = {j.child for j in joints.values()}
        roots = [name for name in links if name not in children]
        if len(roots) != 1:
            raise ValueError(f"URDF must be a single tree, roots={roots}")
        model = cls(root.get("name", ""), links, joints, roots[0], source=source,
                    sha256=hashlib.sha256(text.encode()).hexdigest())
        for j in joints.values():
            model._children.setdefault(j.parent, []).append(j)
        return model

    @classmethod
    def from_file(cls, path: str | Path, use: str = "visual") -> URDFModel:
        path = Path(path)
        if path.suffix == ".xacro":
            raise ValueError("expand xacro first (xacro robot.urdf.xacro > robot.urdf) in the asset environment")
        return cls.from_string(path.read_text(), source=str(path), use=use)

    def joint_value(self, q: dict, joint: Joint) -> float:
        if joint.mimic is not None:
            src, mult, off = joint.mimic
            return mult * self.joint_value(q, self.joints[src]) + off
        return float(q.get(joint.name, 0.0))

    def fk(self, q: dict) -> dict:
        """{link: T_root_link} for joint values ``q`` (missing actuated joints default to 0; mimics follow)."""
        out = {self.root: np.eye(4)}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            for j in self._children.get(parent, []):
                T = out[parent] @ j.T_parent_joint
                if j.kind in ("revolute", "continuous"):
                    T = T @ make_T(Rotation.from_rotvec(j.axis / np.linalg.norm(j.axis) * self.joint_value(q, j))
                                   .as_matrix(), np.zeros(3))
                elif j.kind == "prismatic":
                    T = T @ make_T(np.eye(3), j.axis / np.linalg.norm(j.axis) * self.joint_value(q, j))
                elif j.kind != "fixed":
                    raise NotImplementedError(f"joint type {j.kind}")
                out[j.child] = T
                stack.append(j.child)
        return out


def compose_urdf(arm_text: str, gripper_text: str, parent_link: str, gripper_root: str, T_parent_gripper: np.ndarray,
                 joint_name: str = "gripper_mount") -> str:
    """Attach a gripper URDF under ``parent_link`` with a fixed mounting joint (coupling included in the transform)."""
    arm, grip = ET.fromstring(arm_text), ET.fromstring(gripper_text)
    names = {e.get("name") for e in arm if e.tag in ("link", "joint")}
    for e in grip:
        if e.tag in ("link", "joint"):
            if e.get("name") in names:
                raise ValueError(f"name clash {e.get('name')!r} between arm and gripper URDFs")
            arm.append(e)
    rpy = Rotation.from_matrix(np.asarray(T_parent_gripper)[:3, :3]).as_euler("xyz")
    xyz = np.asarray(T_parent_gripper)[:3, 3]
    j = ET.SubElement(arm, "joint", name=joint_name, type="fixed")
    ET.SubElement(j, "parent", link=parent_link)
    ET.SubElement(j, "child", link=gripper_root)
    ET.SubElement(j, "origin", xyz=" ".join(f"{v:.9g}" for v in xyz), rpy=" ".join(f"{v:.12g}" for v in rpy))
    return ET.tostring(arm, encoding="unicode")


# ---------------------------------------------------------------- Panda DH cross-check (Franka documentation)

PANDA_DH = {  # Craig (modified) DH: T_i = RotX(alpha_{i-1}) TransX(a_{i-1}) RotZ(theta_i) TransZ(d_i)
    "a": (0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0),
    "d": (0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0, 0.107),
    "alpha": (0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0.0),
}


def panda_dh_fk(q: np.ndarray) -> list[np.ndarray]:
    """T_link0_link{i} for i = 1..8 (8 = flange) from Franka's published modified-DH parameters."""
    theta = list(np.asarray(q, float)) + [0.0]
    T = np.eye(4)
    out = []
    for a, d, alpha, th in zip(PANDA_DH["a"], PANDA_DH["d"], PANDA_DH["alpha"], theta):
        rx = make_T(Rotation.from_euler("x", alpha).as_matrix(), [a, 0, 0])
        rz = make_T(Rotation.from_euler("z", th).as_matrix(), [0, 0, d])
        T = T @ rx @ rz
        out.append(T.copy())
    return out


# ---------------------------------------------------------------- gripper state -> URDF joint

ROBOTIQ_FINGER_JOINT_MAX = {"robotiq_2f_85": 0.8, "robotiq_2f_140": 0.7}  # finger_joint upper limits in the URDFs


def gripper_joint_value(state: float, semantics: str, gripper_model: str, *, closed_value: float | None = None) -> float:
    """Recorded gripper state -> URDF gripper joint value (assumed linear; VERIFY on the real gripper).

    ``closed_value`` overrides the fully-closed joint value (e.g. the synthetic stand-in gripper).
    """
    top = closed_value if closed_value is not None else ROBOTIQ_FINGER_JOINT_MAX.get(gripper_model)
    if semantics == "finger_joint_rad":
        return float(state)
    if top is None:
        raise ValueError(f"no joint range known for gripper {gripper_model!r}; supply closed_value")
    if semantics == "normalized_closure":
        return float(np.clip(state, 0, 1)) * top
    if semantics == "robotiq_position_register":
        return float(np.clip(state, 0, 255)) / 255.0 * top
    raise ValueError(f"gripper semantics {semantics!r} needs a measured mapping (e.g. opening_width_m)")


# ---------------------------------------------------------------- surface sampling


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (V,3), faces (F,3)) from STL (binary or ascii) or OBJ."""
    path = Path(path)
    data = path.read_bytes()
    if path.suffix.lower() == ".stl":
        if data[:5] == b"solid" and b"facet" in data[:1024]:
            verts = [list(map(float, ln.split()[1:4])) for ln in data.decode(errors="ignore").splitlines()
                     if ln.strip().startswith("vertex")]
            v = np.array(verts, float)
        else:
            n = struct.unpack("<I", data[80:84])[0]
            rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
            v = rec["v"].reshape(-1, 3).astype(float)
        return v, np.arange(len(v)).reshape(-1, 3)
    if path.suffix.lower() == ".obj":
        verts, faces = [], []
        for ln in data.decode(errors="ignore").splitlines():
            parts = ln.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append(list(map(float, parts[1:4])))
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
        return np.array(verts, float), np.array(faces, int)
    raise ValueError(f"mesh format {path.suffix} unsupported (convert .dae to .stl/.obj, or use collision meshes)")


def _sample_triangles(v: np.ndarray, f: np.ndarray, n: int, rng) -> np.ndarray:
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    idx = rng.choice(len(f), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return (1 - s)[:, None] * tri[idx, 0] + (s * (1 - r2))[:, None] * tri[idx, 1] + (s * r2)[:, None] * tri[idx, 2]


def sample_geometry(g: Geometry, n: int, rng, mesh_root: Path | None = None) -> np.ndarray:
    if g.kind == "box":
        x, y, z = g.size
        faces_area = np.array([y * z, y * z, x * z, x * z, x * y, x * y])
        k = rng.choice(6, size=n, p=faces_area / faces_area.sum())
        p = (rng.random((n, 3)) - 0.5) * np.array([x, y, z])
        axis = k // 2
        p[np.arange(n), axis] = np.where(k % 2 == 0, -0.5, 0.5) * np.array([x, y, z])[axis]
    elif g.kind == "cylinder":
        r, length = g.size
        ang = rng.uniform(0, 2 * np.pi, n)
        p = np.stack([r * np.cos(ang), r * np.sin(ang), rng.uniform(-length / 2, length / 2, n)], 1)
    elif g.kind == "sphere":
        d = rng.normal(size=(n, 3))
        p = g.size[0] * d / np.linalg.norm(d, axis=1, keepdims=True)
    else:
        name = g.mesh or ""
        for prefix in ("package://", "file://"):
            if name.startswith(prefix):
                name = name[len(prefix):]
        path = Path(name) if Path(name).is_absolute() else (mesh_root or Path(".")) / name
        if not path.is_file():
            raise FileNotFoundError(f"robot mesh {g.mesh!r} not found under {mesh_root}; supply the exact assets")
        v, f = load_mesh(path)
        p = _sample_triangles(v * np.asarray(g.scale), f, n, rng)
    return p @ g.T_link_geom[:3, :3].T + g.T_link_geom[:3, 3]


@dataclass
class RobotSurface:
    """Fixed per-link surface samples; ``points(q)`` poses them with FK."""

    model: URDFModel
    link_points: dict  # link -> (n,3) in link frame

    @classmethod
    def build(cls, model: URDFModel, points_per_link: int = 200, seed: int = 0, mesh_root: str | Path | None = None):
        rng = np.random.default_rng(seed)
        out = {}
        for link, geoms in sorted(model.links.items()):
            if not geoms:
                continue
            per = max(1, points_per_link // len(geoms))
            out[link] = np.concatenate([sample_geometry(g, per, rng, Path(mesh_root) if mesh_root else None)
                                        for g in geoms])
        return cls(model, out)

    def points(self, q: dict) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) base-frame points and (N,) link indices (into sorted link names)."""
        fk = self.model.fk(q)
        pts, ids = [], []
        for i, (link, local) in enumerate(sorted(self.link_points.items())):
            T = fk[link]
            pts.append(local @ T[:3, :3].T + T[:3, 3])
            ids.append(np.full(len(local), i))
        return np.concatenate(pts), np.concatenate(ids)


def landmarks(model: URDFModel, q: dict, names: list[str]) -> np.ndarray:
    """Base-frame positions of named link origins (used as 2D-3D landmarks for visual initialisation)."""
    fk = model.fk(q)
    return np.stack([fk[n][:3, 3] for n in names])


def joint_dict(description_joint_names, q: np.ndarray, gripper_joint: str | None = None,
               gripper_value: float | None = None) -> dict:
    out = {name: float(v) for name, v in zip(description_joint_names, q)}
    if gripper_joint is not None and gripper_value is not None:
        out[gripper_joint] = float(gripper_value)
    return out
