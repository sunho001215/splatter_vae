"""Robot template renderer for render-to-image robot 2D-3D correspondences (``rig_loma``).

DepthWorld (arXiv 2610.08780, App. B.1) renders the URDF robot into each external view (pyrender, current joint
angles, current extrinsics), with *well-lit, non-PBR shading*, crops render and real image *tightly around the rendered
arm*, matches them with LoMa-G and turns every match into a 2D-3D correspondence: the render pixel's robot surface point
(known from the URDF + forward kinematics) <-> the real-image pixel. pyrender is not available in our containers (no
network, no OpenGL stack in the matcher environment), so this module re-implements the protocol with an exact triangle
rasterizer in torch (CPU or GPU):

* geometry: the Franka / Robotiq VISUAL meshes (COLLADA) with their material diffuse colours (``load_collada_colored``),
  posed by FK (link transforms in the robot base frame), camera T_cam_base, pinhole K (continuous pixel coordinates,
  pixel (i, j) centre at (j + .5, i + .5), the convention of ``rig_dw``);
* shading: Lambert (non-PBR) with a camera-attached key light + a fill light + ambient, smooth corner normals with a
  crease angle, two-sided; ``ss`` x ``ss`` supersampling, box-filtered to the output resolution;
* per output pixel of the supersampled buffers: triangle id, perspective-correct barycentrics -> the exact surface point
  in its LINK frame (+ link id) and the camera depth.  ``surface_points`` maps continuous keypoint coordinates of the
  render to surface points by an exact ray / triangle-plane intersection (not a depth-buffer unprojection) and flags
  keypoints near silhouettes / depth discontinuities.

Nothing here reads masks or images; the renderer depends only on numpy (+ torch for rasterization)."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from s4d.realworld.robotics.collada import _children, _mesh_positions, _tag, _triangle_fan


# ---------------------------------------------------------------------------------------------- COLLADA with colours
def _effect_colors(root) -> dict:
    """effect id -> RGB diffuse colour (None for textured / missing diffuse)."""
    out = {}
    for eff in root.iter():
        if _tag(eff) != "effect":
            continue
        col = None
        for d in eff.iter():
            if _tag(d) == "diffuse":
                for c in d:
                    if _tag(c) == "color" and c.text:
                        col = np.array(c.text.split(), float)[:3]
                break
        out[eff.get("id")] = col
    return out


def load_collada_colored(path: str | Path, default_color=(0.6, 0.6, 0.6)) -> tuple[np.ndarray, np.ndarray]:
    """Triangles (F, 3, 3) in metres and their diffuse colours (F, 3) of every instanced ``<triangles>``/``<polylist>``/
    ``<polygons>`` primitive (node transforms, ``<unit meter>``, ``bind_material``). Same geometry as
    ``robot_visual.load_collada`` (tested for equality); the colour comes from material symbol -> material ->
    effect diffuse colour (``default_color`` if textured or missing)."""
    from s4d.realworld.robotics.visual import _node_matrix

    root = ET.parse(path).getroot()
    unit = 1.0
    for el in root.iter():
        if _tag(el) == "unit" and el.get("meter"):
            unit = float(el.get("meter"))
            break
    effects = _effect_colors(root)
    materials = {}
    for m in root.iter():
        if _tag(m) == "material":
            ie = _children(m, "instance_effect")
            materials[m.get("id")] = effects.get(ie[0].get("url", "#")[1:]) if ie else None
    geoms = {}
    for g in root.iter():
        if _tag(g) != "geometry":
            continue
        meshes = _children(g, "mesh")
        if not meshes:
            continue
        mesh = meshes[0]
        positions = _mesh_positions(mesh)
        prims = []
        for prim in mesh:
            kind = _tag(prim)
            if kind not in ("triangles", "polylist", "polygons"):
                continue
            inputs = _children(prim, "input")
            stride = max(int(i.get("offset", 0)) for i in inputs) + 1
            vin = [i for i in inputs if i.get("semantic") == "VERTEX"][0]
            pos = positions[vin.get("source")[1:]]
            off = int(vin.get("offset", 0))
            tris, polys = [], []
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
                else:
                    vcount = np.array(_children(prim, "vcount")[0].text.split(), int)
                    starts = np.concatenate([[0], np.cumsum(vcount)[:-1]])
                    polys = [idx[s:s + n] for s, n in zip(starts, vcount)]
            for poly in polys:
                if len(poly) >= 3:
                    fan = _triangle_fan(poly)
                    tris.append(pos[fan])
            if tris:
                prims.append((np.concatenate(tris), prim.get("material")))
        geoms[g.get("id")] = prims
    out_t, out_c = [], []

    def walk(node, T):
        T = T @ _node_matrix(node)
        for ch in node:
            t = _tag(ch)
            if t == "instance_geometry":
                binding = {}
                for im in ch.iter():
                    if _tag(im) == "instance_material":
                        binding[im.get("symbol")] = materials.get(im.get("target", "#")[1:])
                for tri, sym in geoms.get(ch.get("url", "#")[1:], []):
                    col = binding.get(sym)
                    if col is None:
                        col = materials.get(sym)  # some exporters reference the material id directly
                    col = np.asarray(default_color if col is None else col, float)
                    out_t.append(tri @ T[:3, :3].T + T[:3, 3])
                    out_c.append(np.repeat(col[None], len(tri), 0))
            elif t == "node":
                walk(ch, T)

    for scene in root.iter():
        if _tag(scene) == "visual_scene":
            for node in _children(scene, "node"):
                walk(node, np.eye(4))
    if not out_t:
        raise ValueError(f"{path}: no instanced triangle geometry")
    return np.concatenate(out_t) * unit, np.concatenate(out_c)


def corner_normals(tri: np.ndarray, crease_deg: float = 35.0, weld_m: float = 1e-6) -> np.ndarray:
    """Smooth per-corner normals (F, 3, 3): area-weighted mean of the face normals around each welded vertex whose angle
    to the corner's own face normal is below ``crease_deg`` (sharp edges stay sharp)."""
    F = len(tri)
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cr, axis=1)
    fn = cr / np.maximum(area2, 1e-30)[:, None]
    key = np.round(tri.reshape(-1, 3) / weld_m).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.reshape(-1)
    corner_face = np.repeat(np.arange(F), 3)
    order = np.argsort(vid, kind="stable")
    vs, fs = vid[order], corner_face[order]
    start = np.r_[0, np.flatnonzero(np.diff(vs)) + 1]
    counts = np.diff(np.r_[start, len(vs)])
    out = np.zeros((F * 3, 3))
    cos_t = np.cos(np.radians(crease_deg))
    # pairs (corner c, face g) sharing the vertex, grouped by valence to stay vectorised
    for val in np.unique(counts):
        grp = start[counts == val]
        idx = grp[:, None] + np.arange(val)[None]  # (G, val) positions in the sorted list
        faces = fs[idx]  # (G, val)
        nf = fn[faces]  # (G, val, 3)
        wa = area2[faces][..., None] * nf  # area-weighted
        dots = np.einsum("gai,gbi->gab", nf, nf)  # (G, val, val)
        acc = np.einsum("gab,gbi->gai", (dots >= cos_t).astype(float), wa)
        out[order[idx.reshape(-1)]] = acc.reshape(-1, 3)
    out = out.reshape(F, 3, 3)
    n = np.linalg.norm(out, axis=2, keepdims=True)
    return np.where(n > 1e-30, out / np.maximum(n, 1e-30), fn[:, None, :]).astype(np.float32)


# ---------------------------------------------------------------------------------------------- robot render mesh
@dataclass
class RobotMesh:
    """Render mesh of the robot: link-frame triangles with colours and corner normals."""

    links: list
    tri: np.ndarray  # (F, 3, 3) float32, link frame
    link: np.ndarray  # (F,) int16 index into ``links``
    color: np.ndarray  # (F, 3) float32 in [0, 1]
    normal: np.ndarray  # (F, 3, 3) float32 corner normals, link frame
    provenance: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        import json

        np.savez_compressed(path, links=np.array(self.links), tri=self.tri, link=self.link, color=self.color,
                            normal=self.normal, provenance=json.dumps(self.provenance))

    @classmethod
    def load(cls, path: str | Path) -> RobotMesh:
        import json

        with np.load(path) as z:
            return cls([str(x) for x in z["links"]], z["tri"], z["link"], z["color"], z["normal"],
                       json.loads(str(z["provenance"])))

    def posed(self, link_T: np.ndarray, link_names: list) -> tuple[np.ndarray, np.ndarray]:
        """Base-frame triangles + corner normals for link transforms ``link_T`` (L, 4, 4) ordered as ``link_names``."""
        li = {l: i for i, l in enumerate(link_names)}
        Ts = np.stack([link_T[li[l]] for l in self.links]).astype(np.float64)
        R, t = Ts[self.link, :3, :3], Ts[self.link, :3, 3]
        V = np.einsum("fij,fkj->fki", R, self.tri.astype(np.float64)) + t[:, None, :]
        N = np.einsum("fij,fkj->fki", R, self.normal.astype(np.float64))
        return V, N


def build_robot_mesh(model, crease_deg: float = 35.0, links=None, allow_collision_fallback: bool = False) -> RobotMesh:
    """Render mesh from a ``robot.URDFModel`` (composed Panda + Robotiq URDF): every mesh geometry is replaced by the
    visual COLLADA mesh next to its collision STL (``robot_visual.visual_mesh_for``). Fails closed
    (``robot_visual.MissingVisualMeshError``) when a visual mesh is missing, unless ``allow_collision_fallback`` (then
    the STL is rendered grey and listed in ``provenance["fallback"]``)."""
    from s4d.realworld.robotics.model import load_mesh
    from s4d.realworld.robotics.visual import require_visual_meshes, visual_mesh_for

    if not allow_collision_fallback:
        require_visual_meshes(model, links=links)
    names, T, L, C, N, prov = [], [], [], [], [], {"meshes": {}, "fallback": []}
    for link, geoms in sorted(model.links.items()):
        if links is not None and link not in links:
            continue
        parts = []
        for g in geoms:
            if g.kind != "mesh":
                continue
            vis = visual_mesh_for(g.mesh or "")
            if vis is not None:
                tri, col = load_collada_colored(vis)
                src = {"file": str(vis), "sha256": hashlib.sha256(Path(vis).read_bytes()).hexdigest(),
                       "kind": "visual_collada"}
            else:
                name = (g.mesh or "").removeprefix("package://").removeprefix("file://")
                v, f = load_mesh(name)
                tri, col = v[f], np.full((len(f), 3), 0.6)
                src = {"file": name, "sha256": hashlib.sha256(Path(name).read_bytes()).hexdigest(),
                       "kind": "collision_fallback"}
                prov["fallback"].append(link)
            tri = tri * np.asarray(g.scale)
            tri = tri @ g.T_link_geom[:3, :3].T + g.T_link_geom[:3, 3]
            parts.append((tri, col))
            prov["meshes"].setdefault(link, []).append(src)
        if not parts:
            continue
        k = len(names)
        names.append(link)
        tri = np.concatenate([p[0] for p in parts])
        T.append(tri)
        C.append(np.concatenate([p[1] for p in parts]))
        L.append(np.full(len(tri), k, np.int16))
        N.append(corner_normals(tri, crease_deg))
    prov["crease_deg"] = crease_deg
    return RobotMesh(names, np.concatenate(T).astype(np.float32), np.concatenate(L), np.concatenate(C).astype(np.float32),
                     np.concatenate(N).astype(np.float32), prov)


# ---------------------------------------------------------------------------------------------- rasterizer
@dataclass
class RenderConfig:
    ss: int = 2  # supersampling factor
    near: float = 0.05
    ambient: float = 0.35
    key: float = 0.65  # camera-attached (head) light
    fill: float = 0.25  # light from above (camera -y)
    background: tuple = (0.0, 0.0, 0.0)
    albedo_scale: float = 0.85  # white materials -> 0.85 (avoid clipping)
    margin_px: int = 16  # tight crop: rendered-arm bounding box + margin (DepthWorld: crop tightly around the arm)
    max_fragments: int = 30_000_000


@dataclass
class Raster:
    """Supersampled buffers of one window. Window (x0, y0, w, h) in output pixels of the full image; buffers are
    (h * ss, w * ss); buffer pixel (i, j) centre at full-image continuous coords (x0 + (j + .5) / ss, y0 + (i + .5) / ss)."""

    window: tuple
    ss: int
    fid: np.ndarray  # (H, W) int64 triangle id, -1 empty
    z: np.ndarray  # (H, W) camera z (inf empty)
    bary: np.ndarray  # (H, W, 3) perspective-correct barycentrics
    rgb: np.ndarray | None = None  # (h, w, 3) uint8 shaded output (box-filtered)
    mask: np.ndarray | None = None  # (h, w) bool coverage >= 50 %


def _torch():
    import torch

    return torch


def rasterize(Vc: np.ndarray, K: np.ndarray, window: tuple, ss: int = 2, near: float = 0.05, device: str = "cpu",
              max_fragments: int = 30_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact z-buffered rasterization of camera-frame triangles ``Vc`` (F, 3, 3) on a window (x0, y0, w, h) of the
    image with intrinsics K, supersampled ``ss`` x. Returns (fid, z, bary) on the (h*ss, w*ss) buffer. A pixel centre
    is covered when its screen-space barycentrics are all >= 0 (shared edges: the nearer triangle wins). Triangles
    with a vertex behind ``near`` are dropped (the robot is always >= 0.5 m in front of the cameras)."""
    torch = _torch()
    x0, y0, w, h = (int(v) for v in window)
    H, W = h * ss, w * ss
    dev = torch.device(device)
    V = torch.as_tensor(np.asarray(Vc), dtype=torch.float64, device=dev)
    Z = V[..., 2]
    ok = (Z > near).all(1)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    zc = Z.clamp(min=near)
    u = fx * V[..., 0] / zc + cx
    v = fy * V[..., 1] / zc + cy
    # buffer coordinates: pixel index j has its centre at s = j
    s = (u - x0) * ss - 0.5
    t = (v - y0) * ss - 0.5
    jmin = torch.ceil(s.min(1).values).clamp(min=0)
    jmax = torch.floor(s.max(1).values).clamp(max=W - 1)
    imin = torch.ceil(t.min(1).values).clamp(min=0)
    imax = torch.floor(t.max(1).values).clamp(max=H - 1)
    area = (s[:, 1] - s[:, 0]) * (t[:, 2] - t[:, 0]) - (s[:, 2] - s[:, 0]) * (t[:, 1] - t[:, 0])
    ok &= (jmax >= jmin) & (imax >= imin) & (area.abs() > 1e-12)
    ids = torch.nonzero(ok).flatten()
    INF = torch.iinfo(torch.int64).max
    zbuf = torch.full((H * W,), INF, dtype=torch.int64, device=dev)
    if len(ids):
        bw = (jmax[ids] - jmin[ids] + 1).long()
        bh = (imax[ids] - imin[ids] + 1).long()
        size = torch.maximum(bw, bh)
        bucket = torch.ceil(torch.log2(size.double())).long().clamp(min=0)
        for b in torch.unique(bucket).tolist():
            S = 1 << b
            sel = ids[bucket == b]
            per = max(1, max_fragments // (S * S))
            for c0 in range(0, len(sel), per):
                f = sel[c0:c0 + per]
                dy, dx = torch.meshgrid(torch.arange(S, device=dev), torch.arange(S, device=dev), indexing="ij")
                jj = jmin[f, None] + dx.reshape(1, -1)
                ii = imin[f, None] + dy.reshape(1, -1)
                inb = (jj <= jmax[f, None]) & (ii <= imax[f, None])
                s0, s1, s2 = s[f, 0, None], s[f, 1, None], s[f, 2, None]
                t0, t1, t2 = t[f, 0, None], t[f, 1, None], t[f, 2, None]
                A = area[f, None]
                w0 = ((s1 - jj) * (t2 - ii) - (s2 - jj) * (t1 - ii)) / A
                w1 = ((s2 - jj) * (t0 - ii) - (s0 - jj) * (t2 - ii)) / A
                w2 = 1.0 - w0 - w1
                inside = inb & (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
                iz = w0 / zc[f, 0, None] + w1 / zc[f, 1, None] + w2 / zc[f, 2, None]
                zz = 1.0 / iz
                key = (torch.clamp(zz * 1e5, 0, 2**40).long() << 22) | f[:, None]
                pix = (ii * W + jj).long()
                zbuf.scatter_reduce_(0, pix[inside], key[inside], reduce="amin")
    hit = zbuf != INF
    fid = torch.full((H * W,), -1, dtype=torch.int64, device=dev)
    fid[hit] = zbuf[hit] & ((1 << 22) - 1)
    z = torch.full((H * W,), float("inf"), dtype=torch.float64, device=dev)
    bary = torch.zeros((H * W, 3), dtype=torch.float64, device=dev)
    if hit.any():
        p = torch.nonzero(hit).flatten()
        f = fid[p]
        jj = (p % W).double()
        ii = (p // W).double()
        s0, s1, s2 = s[f, 0], s[f, 1], s[f, 2]
        t0, t1, t2 = t[f, 0], t[f, 1], t[f, 2]
        A = area[f]
        w0 = ((s1 - jj) * (t2 - ii) - (s2 - jj) * (t1 - ii)) / A
        w1 = ((s2 - jj) * (t0 - ii) - (s0 - jj) * (t2 - ii)) / A
        w2 = 1.0 - w0 - w1
        q = torch.stack([w0 / zc[f, 0], w1 / zc[f, 1], w2 / zc[f, 2]], 1)
        iz = q.sum(1)
        z[p] = 1.0 / iz
        bary[p] = q / iz[:, None]
    return (fid.reshape(H, W).cpu().numpy(), z.reshape(H, W).cpu().numpy(), bary.reshape(H, W, 3).cpu().numpy())


def shade(Vc: np.ndarray, Nc: np.ndarray, color: np.ndarray, fid: np.ndarray, bary: np.ndarray, ss: int,
          cfg: RenderConfig) -> tuple[np.ndarray, np.ndarray]:
    """Non-PBR Lambert shading of the supersampled buffers (two-sided smooth normals; key light at the camera, fill
    from above, ambient), box-filtered to the output resolution. Returns (rgb uint8 (h, w, 3), coverage mask)."""
    H, W = fid.shape
    out = np.empty((H, W, 3))
    out[:] = np.asarray(cfg.background, float)
    hit = fid >= 0
    if hit.any():
        f = fid[hit]
        b = bary[hit]
        P = np.einsum("nk,nki->ni", b, Vc[f])
        n = np.einsum("nk,nki->ni", b, Nc[f])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        view = -P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)  # towards the camera
        n = np.where((np.einsum("ni,ni->n", n, view) < 0)[:, None], -n, n)  # two-sided
        fill_dir = np.array([0.0, -1.0, -0.3]) / np.linalg.norm([0.0, -1.0, -0.3])
        L = cfg.ambient + cfg.key * np.clip(np.einsum("ni,ni->n", n, view), 0, 1) \
            + cfg.fill * np.clip(n @ fill_dir, 0, 1)
        out[hit] = np.clip(color[f] * cfg.albedo_scale * L[:, None], 0, 1)
    h, w = H // ss, W // ss
    rgb = out.reshape(h, ss, w, ss, 3).mean((1, 3))
    cov = hit.reshape(h, ss, w, ss).mean((1, 3))
    return np.round(rgb * 255).astype(np.uint8), cov >= 0.5


def project_bbox(Vc: np.ndarray, K: np.ndarray, hw: tuple, near: float = 0.05):
    """Integer bounding box (x0, y0, x1, y1) of the projected vertices in front of the camera, clipped to the image;
    None if nothing projects into it."""
    P = Vc.reshape(-1, 3)
    P = P[P[:, 2] > near]
    if not len(P):
        return None
    u = K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2]
    v = K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]
    H, W = hw
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not inside.any():
        return None
    return (int(max(0, np.floor(u.min()))), int(max(0, np.floor(v.min()))),
            int(min(W, np.ceil(u.max()))), int(min(H, np.ceil(v.max()))))


def render_view(mesh: RobotMesh, link_T: np.ndarray, link_names: list, T_cam_base: np.ndarray, K: np.ndarray,
                hw: tuple, cfg: RenderConfig | None = None, window: tuple | None = None, device: str = "cpu"):
    """Render the posed robot. Without ``window`` the whole image is rasterized first at ss = 1 to find the rendered arm,
    then the tight window (bounding box of the rendered pixels + ``margin_px``) is rendered at ``cfg.ss``. Returns
    (Raster, Vc, Nc) with camera-frame triangles / normals (for ``surface_points``)."""
    cfg = cfg or RenderConfig()
    Vb, Nb = mesh.posed(link_T, link_names)
    R, t = T_cam_base[:3, :3], T_cam_base[:3, 3]
    Vc = Vb @ R.T + t
    Nc = Nb @ R.T
    H, W = hw
    if window is None:
        bb = project_bbox(Vc, K, hw, cfg.near)
        if bb is None:
            return None, Vc, Nc
        x0, y0, x1, y1 = bb
        fid0, _, _ = rasterize(Vc, K, (x0, y0, x1 - x0, y1 - y0), 1, cfg.near, device, cfg.max_fragments)
        ys, xs = np.nonzero(fid0 >= 0)
        if not len(ys):
            return None, Vc, Nc
        m = cfg.margin_px
        wx0, wy0 = max(0, x0 + xs.min() - m), max(0, y0 + ys.min() - m)
        wx1, wy1 = min(W, x0 + xs.max() + 1 + m), min(H, y0 + ys.max() + 1 + m)
        window = (wx0, wy0, wx1 - wx0, wy1 - wy0)
    fid, z, bary = rasterize(Vc, K, window, cfg.ss, cfg.near, device, cfg.max_fragments)
    rgb, mask = shade(Vc, Nc, mesh.color.astype(np.float64), fid, bary, cfg.ss, cfg)
    return Raster(tuple(int(v) for v in window), cfg.ss, fid, z, bary, rgb, mask), Vc, Nc


def surface_points(r: Raster, Vc: np.ndarray, mesh: RobotMesh, uv: np.ndarray, K: np.ndarray, nbhd: int = 2):
    """Robot surface points of continuous FULL-IMAGE render coordinates ``uv`` (N, 2): the triangle covering the nearest
    supersampled pixel, exact intersection of the pixel ray with that triangle's plane. Returns a dict with
    ``valid`` (a triangle was hit), ``link`` (index into mesh.links), ``X_link`` (N, 3) link-frame point, ``X_cam``
    (N, 3), ``z_range`` (m: depth range of the +-``nbhd`` supersampled neighbourhood, inf where it touches the
    background -> silhouette) and ``bary_out`` (how far outside the triangle the intersection lies, barycentric units)."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    x0, y0, w, h = r.window
    ss = r.ss
    H, W = r.fid.shape
    j = np.clip(np.round((uv[:, 0] - x0) * ss - 0.5).astype(int), 0, W - 1)
    i = np.clip(np.round((uv[:, 1] - y0) * ss - 0.5).astype(int), 0, H - 1)
    f = r.fid[i, j]
    valid = f >= 0
    n = len(uv)
    X_cam = np.full((n, 3), np.nan)
    X_link = np.full((n, 3), np.nan)
    link = np.full(n, -1, np.int64)
    bary_out = np.full(n, np.nan)
    if valid.any():
        fv = f[valid]
        T = Vc[fv]
        nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        d = np.stack([(uv[valid, 0] - K[0, 2]) / K[0, 0], (uv[valid, 1] - K[1, 2]) / K[1, 1], np.ones(valid.sum())], 1)
        lam = np.einsum("ni,ni->n", nrm, T[:, 0]) / np.einsum("ni,ni->n", nrm, d)
        P = d * lam[:, None]
        # barycentrics of P in the camera-frame triangle (least squares on the plane)
        e1, e2, ep = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0], P - T[:, 0]
        d11, d12, d22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
        dp1, dp2 = (ep * e1).sum(1), (ep * e2).sum(1)
        den = d11 * d22 - d12 ** 2
        b1 = (d22 * dp1 - d12 * dp2) / den
        b2 = (d11 * dp2 - d12 * dp1) / den
        b = np.stack([1 - b1 - b2, b1, b2], 1)
        TL = mesh.tri[fv].astype(np.float64)
        X_link[valid] = np.einsum("nk,nki->ni", b, TL)
        X_cam[valid] = P
        link[valid] = mesh.link[fv]
        bary_out[valid] = np.clip(-b, 0, None).sum(1)
    # depth range of the neighbourhood (silhouette / occlusion boundary indicator)
    zr = np.full(n, np.inf)
    zz = np.where(np.isfinite(r.z), r.z, np.nan)
    k = nbhd
    for q in range(n):
        if not valid[q]:
            continue
        win = zz[max(0, i[q] - k):i[q] + k + 1, max(0, j[q] - k):j[q] + k + 1]
        if np.isnan(win).any():
            continue
        zr[q] = float(win.max() - win.min())
    return {"valid": valid, "link": link, "X_link": X_link, "X_cam": X_cam, "z_range": zr, "bary_out": bary_out}


