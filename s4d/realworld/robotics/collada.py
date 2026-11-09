"""Private COLLADA parsing steps shared by the geometry and coloured-mesh readers."""

from __future__ import annotations

import numpy as np


def _tag(el) -> str:
    return el.tag.split("}", 1)[-1]


def _children(el, name):
    return [c for c in el if _tag(c) == name]


def _mesh_positions(mesh):
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
    return positions


def _triangle_fan(poly):
    return np.stack([np.full(len(poly) - 2, poly[0]), poly[1:-1], poly[2:]], 1)
