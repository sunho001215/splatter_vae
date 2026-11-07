"""Visual (render) meshes of the robot for the board-free rig calibration (``rig_dw``).

The composed Panda + Robotiq URDF of the pipeline references the COLLISION meshes (STL, 86-976 triangles per link,
because ``robot.load_mesh`` cannot read COLLADA). They deviate from the visual meshes (COLLADA, 1.2k-29k triangles) by a
median 2-4 mm and a 90th percentile of 8-23 mm per Panda link (2026-11-07, ``tests/test_realworld_robot_visual.py``
documents the comparison); a calibration that fits a camera to the observed robot surface absorbs that model error
into the extrinsics. This module loads the visual meshes next to the collision files of the same package
(``.../collision/<x>.stl`` -> ``.../visual/<x>.dae``) with a minimal COLLADA reader and samples the robot surface with
normals. The URDF itself (kinematics, origins, sha256) is unchanged; only the surface geometry differs, and every mesh
file used is hashed into the provenance."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _tag(el) -> str:
    return el.tag.split("}", 1)[-1]


def _children(el, name):
    return [c for c in el if _tag(c) == name]


def _node_matrix(node) -> np.ndarray:
    T = np.eye(4)
    for ch in node:
        t = _tag(ch)
        vals = np.array(ch.text.split(), float) if ch.text else None
        if t == "matrix":
            T = T @ vals.reshape(4, 4)
        elif t == "translate":
            M = np.eye(4)
            M[:3, 3] = vals
            T = T @ M
        elif t == "scale":
            T = T @ np.diag([*vals, 1.0])
        elif t == "rotate":
            axis, ang = vals[:3] / np.linalg.norm(vals[:3]), np.radians(vals[3])
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            M = np.eye(4)
            M[:3, :3] = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
            T = T @ M
    return T


def load_collada(path: str | Path) -> np.ndarray:
    """Triangles (F, 3, 3) in metres of every ``<triangles>``/``<polylist>``/``<polygons>`` primitive of a COLLADA file,
    instanced through the ``<visual_scene>`` node hierarchy (nested node transforms), scaled by ``<unit meter>``."""
    root = ET.parse(path).getroot()
    unit = 1.0
    for el in root.iter():
        if _tag(el) == "unit" and el.get("meter"):
            unit = float(el.get("meter"))
            break
    geoms = {}
    for g in root.iter():
        if _tag(g) != "geometry":
            continue
        meshes = _children(g, "mesh")
        if not meshes:
            continue
        mesh = meshes[0]
        sources = {}
        for s in _children(mesh, "source"):
            fa = _children(s, "float_array")
            if not fa:
                continue
            acc = [a for a in s.iter() if _tag(a) == "accessor"]
            stride = int(acc[0].get("stride", 3)) if acc else 3
            sources[s.get("id")] = np.array(fa[0].text.split(), float).reshape(-1, stride)
        positions = {}
        for v in _children(mesh, "vertices"):
            for inp in _children(v, "input"):
                if inp.get("semantic") == "POSITION":
                    positions[v.get("id")] = sources[inp.get("source")[1:]]
        tris = []
        for prim in mesh:
            kind = _tag(prim)
            if kind not in ("triangles", "polylist", "polygons"):
                continue
            inputs = _children(prim, "input")
            stride = max(int(i.get("offset", 0)) for i in inputs) + 1
            vin = [i for i in inputs if i.get("semantic") == "VERTEX"][0]
            pos = positions[vin.get("source")[1:]]
            off = int(vin.get("offset", 0))
            polys = []
            if kind == "polygons":
                for p in _children(prim, "p"):
                    polys.append(np.array(p.text.split(), int).reshape(-1, stride)[:, off])
            else:
                p_el = _children(prim, "p")
                if not p_el or not p_el[0].text:
                    continue
                idx = np.array(p_el[0].text.split(), int).reshape(-1, stride)[:, off]
                if kind == "triangles":
                    tris.append(pos[idx.reshape(-1, 3)])
                    continue
                vcount = np.array(_children(prim, "vcount")[0].text.split(), int)
                starts = np.concatenate([[0], np.cumsum(vcount)[:-1]])
                polys = [idx[s:s + n] for s, n in zip(starts, vcount)]
            for poly in polys:  # fan triangulation
                if len(poly) >= 3:
                    fan = np.stack([np.full(len(poly) - 2, poly[0]), poly[1:-1], poly[2:]], 1)
                    tris.append(pos[fan])
        geoms[g.get("id")] = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
    out = []

    def walk(node, T):
        T = T @ _node_matrix(node)
        for ch in node:
            t = _tag(ch)
            if t == "instance_geometry":
                tri = geoms.get(ch.get("url", "#")[1:])
                if tri is not None and len(tri):
                    out.append(tri @ T[:3, :3].T + T[:3, 3])
            elif t == "node":
                walk(ch, T)

    for scene in root.iter():
        if _tag(scene) == "visual_scene":
            for node in _children(scene, "node"):
                walk(node, np.eye(4))
    if not out:
        raise ValueError(f"{path}: no instanced triangle geometry")
    return np.concatenate(out) * unit


def visual_mesh_for(collision_path: str) -> Path | None:
    """The visual COLLADA mesh of a ``.../collision/<name>.stl`` file of the same description package, if present."""
    name = collision_path
    for prefix in ("package://", "file://"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    p = Path(name)
    parts = list(p.parts)
    if "collision" not in parts:
        return None
    k = len(parts) - 1 - parts[::-1].index("collision")
    cand = Path(*parts[:k], "visual", *parts[k + 1:]).with_suffix(".dae")
    return cand if cand.is_file() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def triangle_samples(tri: np.ndarray, spacing_m: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Area-uniform surface samples (about one per spacing^2) with the triangle normals."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    cr = np.cross(e1, e2)
    area2 = np.linalg.norm(cr, axis=1)
    keep = area2 > 1e-14
    tri, cr, area2 = tri[keep], cr[keep], area2[keep]
    n = max(1, int(round(0.5 * area2.sum() / spacing_m ** 2)))
    idx = rng.choice(len(tri), size=n, p=area2 / area2.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * tri[idx, 0] + (s * (1 - r2))[:, None] * tri[idx, 1] + (s * r2)[:, None] * tri[idx, 2]
    return pts, cr[idx] / area2[idx, None]


@dataclass
class VisualSurface:
    """Per-link surface samples + normals in the link frame from the visual meshes (collision geometry where no
    visual mesh exists, recorded in ``fallback``)."""

    link_points: dict
    link_normals: dict
    meshes: dict = field(default_factory=dict)  # link -> [{"file", "sha256", "kind"}]
    fallback: list = field(default_factory=list)

    @classmethod
    def build(cls, model, spacing_m: float = 0.004, seed: int = 0, links=None) -> "VisualSurface":
        from s4d.realworld.robot import load_mesh

        rng = np.random.default_rng(seed)
        pts, nrm, meshes, fallback = {}, {}, {}, []
        for link, geoms in sorted(model.links.items()):
            if links is not None and link not in links:
                continue
            P, N, M = [], [], []
            for g in geoms:
                if g.kind != "mesh":
                    continue  # primitive collision proxies are not part of the visual surface
                vis = visual_mesh_for(g.mesh or "")
                if vis is not None:
                    tri = load_collada(vis)
                    M.append({"file": str(vis), "sha256": _sha256(vis), "kind": "visual_collada"})
                else:
                    name = (g.mesh or "").removeprefix("package://").removeprefix("file://")
                    v, f = load_mesh(name)
                    tri = v[f]
                    M.append({"file": name, "sha256": _sha256(Path(name)), "kind": "collision_fallback"})
                    fallback.append(link)
                tri = tri * np.asarray(g.scale)
                tri = tri @ g.T_link_geom[:3, :3].T + g.T_link_geom[:3, 3]
                p, n = triangle_samples(tri, spacing_m, rng)
                P.append(p)
                N.append(n)
            if P:
                pts[link], nrm[link], meshes[link] = np.concatenate(P), np.concatenate(N), M
        return cls(pts, nrm, meshes, sorted(set(fallback)))

    def posed(self, fk: dict, links=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(points (N,3), normals (N,3), link index (N,)) in the root frame for link poses ``fk``."""
        P, N, I = [], [], []
        for i, link in enumerate(sorted(self.link_points)):
            if links is not None and link not in links:
                continue
            T = fk[link]
            P.append(self.link_points[link] @ T[:3, :3].T + T[:3, 3])
            N.append(self.link_normals[link] @ T[:3, :3].T)
            I.append(np.full(len(self.link_points[link]), i, np.int16))
        return np.concatenate(P), np.concatenate(N), np.concatenate(I)

    def provenance(self) -> dict:
        return {"meshes": self.meshes, "fallback_links": self.fallback,
                "points": {k: int(len(v)) for k, v in self.link_points.items()}}
