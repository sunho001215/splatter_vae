"""Geometry of the render-to-image robot template renderer (``rig_loma_render``): COLLADA colours, exact rasterization
(coverage, depth, z-order, barycentrics), surface-point lookup (render pixel -> exact 3D point), crease normals."""

import numpy as np
import pytest

from s4d.realworld import rig_loma_render as rr

K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])


def _quad(z, half=0.2, cx=0.0, cy=0.0):
    a = np.array([[cx - half, cy - half, z], [cx + half, cy - half, z], [cx + half, cy + half, z]])
    b = np.array([[cx - half, cy - half, z], [cx + half, cy + half, z], [cx - half, cy + half, z]])
    return np.stack([a, b])


def test_rasterize_coverage_and_depth_exact():
    tri = _quad(2.0)
    fid, z, bary = rr.rasterize(tri, K, (0, 0, 640, 480), ss=1)
    # the quad spans u in [320 - 60, 320 + 60): pixel centres j + .5 inside -> j = 260..379
    cols = np.flatnonzero((fid >= 0).any(0))
    rows = np.flatnonzero((fid >= 0).any(1))
    assert cols.min() == 260 and cols.max() == 379
    assert rows.min() == 180 and rows.max() == 299
    assert np.allclose(z[fid >= 0], 2.0)
    assert np.allclose(bary[fid >= 0].sum(-1), 1.0)


def test_rasterize_zorder_and_perspective_depth():
    near, far = _quad(1.0, half=0.05), _quad(3.0, half=0.5)
    tri = np.concatenate([far, near])
    fid, z, _ = rr.rasterize(tri, K, (0, 0, 640, 480), ss=2)
    H, W = fid.shape
    assert fid[H // 2, W // 2] in (2, 3)  # the near quad wins in the centre
    assert np.isclose(z[H // 2, W // 2], 1.0)
    # slanted triangle: perspective-correct depth equals the ray/plane intersection
    T = np.array([[[-0.3, -0.2, 1.5], [0.3, -0.2, 2.5], [0.0, 0.3, 2.0]]])
    fid, z, bary = rr.rasterize(T, K, (0, 0, 640, 480), ss=1)
    i, j = np.argwhere(fid >= 0)[len(np.argwhere(fid >= 0)) // 2]
    P = np.einsum("k,ki->i", bary[i, j], T[0])
    u, v = K[0, 0] * P[0] / P[2] + K[0, 2], K[1, 1] * P[1] / P[2] + K[1, 2]
    assert np.allclose([u, v], [j + 0.5, i + 0.5], atol=1e-9)
    assert np.isclose(z[i, j], P[2])


def test_surface_points_round_trip():
    rng = np.random.default_rng(0)
    T = np.array([[[-0.3, -0.2, 1.5], [0.3, -0.2, 2.5], [0.0, 0.3, 2.0]]])
    mesh = rr.RobotMesh(["l0"], T.astype(np.float32), np.zeros(1, np.int16), np.ones((1, 3), np.float32),
                        np.zeros((1, 3, 3), np.float32))
    fid, z, bary = rr.rasterize(T, K, (200, 150, 240, 200), ss=2)
    r = rr.Raster((200, 150, 240, 200), 2, fid, z, bary)
    # random interior points of the triangle -> project -> lookup -> same 3D point
    w = rng.dirichlet([3, 3, 3], 50)
    P = w @ T[0]
    uv = np.stack([K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2], K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]], 1)
    sp = rr.surface_points(r, T, mesh, uv, K, nbhd=0)
    assert sp["valid"].all()
    assert np.abs(sp["X_cam"] - P).max() < 1e-9
    assert np.abs(sp["X_link"] - P).max() < 1e-6  # link frame == camera frame here (float32 mesh)


def test_surface_points_flags_silhouette():
    tri = _quad(2.0)
    mesh = rr.RobotMesh(["l0"], tri.astype(np.float32), np.zeros(2, np.int16), np.ones((2, 3), np.float32),
                        np.zeros((2, 3, 3), np.float32))
    fid, z, bary = rr.rasterize(tri, K, (0, 0, 640, 480), ss=2)
    r = rr.Raster((0, 0, 640, 480), 2, fid, z, bary)
    sp = rr.surface_points(r, tri, mesh, np.array([[320.0, 240.0], [260.3, 240.0]]), K, nbhd=2)
    assert np.isfinite(sp["z_range"][0]) and sp["z_range"][0] < 1e-9
    assert np.isinf(sp["z_range"][1])  # touches the background


def test_corner_normals_keep_creases():
    # unit cube: every corner normal must equal its face normal (90 deg edges > crease angle)
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
             (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    tri = v[np.array(faces)]
    n = rr.corner_normals(tri, crease_deg=35)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    assert np.allclose(n, fn[:, None, :], atol=1e-6)
    # a gently folded strip (10 deg) is smoothed
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    c, s = np.cos(np.radians(10)), np.sin(np.radians(10))
    b = np.array([[1, 0, 0], [1 + c, 0, s], [0, 1, 0]], float)
    n2 = rr.corner_normals(np.stack([a, b]), crease_deg=35)
    assert np.allclose(n2[0, 1], n2[1, 0], atol=1e-6)


COLLADA = """<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
 <asset><unit meter="0.001"/></asset>
 <library_effects>
  <effect id="red-fx"><profile_COMMON><technique sid="c"><phong><diffuse><color>1 0 0 1</color></diffuse></phong>
  </technique></profile_COMMON></effect>
 </library_effects>
 <library_materials><material id="red-mat"><instance_effect url="#red-fx"/></material></library_materials>
 <library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array id="pa" count="9">0 0 0 10 0 0 0 10 0</float_array>
   <technique_common><accessor source="#pa" count="3" stride="3"/></technique_common></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <triangles material="sym" count="1"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2</p></triangles>
 </mesh></geometry></library_geometries>
 <library_visual_scenes><visual_scene id="s"><node><translate>0 0 5</translate>
  <instance_geometry url="#g"><bind_material><technique_common>
   <instance_material symbol="sym" target="#red-mat"/></technique_common></bind_material></instance_geometry>
 </node></visual_scene></library_visual_scenes>
</COLLADA>"""


def test_collada_colours_and_units(tmp_path):
    p = tmp_path / "t.dae"
    p.write_text(COLLADA)
    tri, col = rr.load_collada_colored(p)
    from s4d.realworld.robot_visual import load_collada

    assert np.allclose(tri, load_collada(p))
    assert np.allclose(tri[0], [[0, 0, 0.005], [0.01, 0, 0.005], [0, 0.01, 0.005]])
    assert np.allclose(col, [[1, 0, 0]])


def test_render_view_window_contains_robot():
    tri = _quad(2.0, half=0.1)
    mesh = rr.RobotMesh(["l0"], tri.astype(np.float32), np.zeros(2, np.int16), np.full((2, 3), 0.5, np.float32),
                        np.tile(np.array([0, 0, -1.0], np.float32), (2, 3, 1)))
    r, Vc, _ = rr.render_view(mesh, np.eye(4)[None], ["l0"], np.eye(4), K, (480, 640), rr.RenderConfig(margin_px=5))
    x0, y0, w, h = r.window
    assert (x0, y0, w, h) == (290 - 5, 210 - 5, 60 + 10, 60 + 10)
    assert r.rgb.shape == (h, w, 3) and r.mask.sum() == 60 * 60
    assert r.rgb[r.mask].min() > 0 and r.rgb[~r.mask].max() == 0
