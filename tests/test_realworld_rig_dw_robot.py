"""Robot factors of the board-free rig calibration (rig_dw): conventions, Jacobians, COLLADA loading, and a synthetic
rig recovery from silhouettes + depth of a known articulated-like object seen by three cameras."""

import numpy as np
import pytest

from s4d.realworld import rig_dw_robot as rr
from s4d.realworld.rig_dw_geom import pose_distance, se3_exp


def look_at(C, target, up=(0, 0, 1)):
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ C
    return T


K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])


def box_samples(center, size, spacing, rot=np.eye(3)):
    """Dense surface samples + outward normals of a box."""
    pts, nrm = [], []
    for ax in range(3):
        for sgn in (-1, 1):
            o = [a for a in range(3) if a != ax]
            u = np.arange(-size[o[0]] / 2, size[o[0]] / 2, spacing)
            v = np.arange(-size[o[1]] / 2, size[o[1]] / 2, spacing)
            U, V = np.meshgrid(u, v)
            p = np.zeros((U.size, 3))
            p[:, o[0]], p[:, o[1]], p[:, ax] = U.ravel(), V.ravel(), sgn * size[ax] / 2
            n = np.zeros_like(p)
            n[:, ax] = sgn
            pts.append(p)
            nrm.append(n)
    P, N = np.concatenate(pts) @ rot.T + center, np.concatenate(nrm) @ rot.T
    return P, N


def scene(seed=0, spacing=0.0012):
    rng = np.random.default_rng(seed)
    P, N = [], []
    for k in range(4):  # an L-shaped "arm" of boxes
        R = se3_exp(np.r_[rng.normal(size=3) * 0.3, 0, 0, 0])[:3, :3]
        p, n = box_samples(np.array([0.5 + 0.08 * k, -0.1 + 0.07 * k, 0.4 + 0.1 * (k % 2)]),
                           np.array([0.08, 0.1, 0.2 - 0.03 * k]), spacing, R)
        P.append(p)
        N.append(n)
    return np.concatenate(P), np.concatenate(N)


def render_obs(T, P, crop, fine=None):
    """Observed mask with a coverage >= 50 % rule (4x4 supersampled render of ``fine`` samples, majority vote, like
    a thresholded segmentation) and the z-buffer depth."""
    rend = rr.render(P @ T[:3, :3].T + T[:3, 3], K, crop)
    depth = np.where(rend.mask, rend.zbuf, 0.0)
    if fine is None:
        return rend.mask.astype(np.uint8), depth
    S = 4
    Ks = np.diag([S, S, 1.0]) @ K
    y0, y1, x0, x1 = crop
    hi = rr.render(fine @ T[:3, :3].T + T[:3, 3], Ks, (S * y0, S * y1, S * x0, S * x1))
    cover = hi.mask.reshape(y1 - y0, S, x1 - x0, S).mean((1, 3))
    return (cover >= 0.5).astype(np.uint8), depth


def test_sdf_half_pixel_convention():
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    s = rr.sdf_of_mask(m)
    assert s[10, 10] < 0 and s[0, 0] > 0
    # pixel (row 10, col 4) is just outside the left edge (x = 5.0): centre at x = 4.5 -> distance 0.5
    assert abs(s[10, 4] - 0.5) < 1e-6 and abs(s[10, 5] + 0.5) < 1e-6
    val, gu, gv = rr.bilinear(s, np.array([[5.0, 10.5]]))  # exactly on the boundary
    assert abs(val[0]) < 1e-6 and gu[0] < 0


def test_render_bins_continuous_pixels():
    P = np.array([[0.0, 0.0, 1.0]])  # projects to (320, 240): pixel row 240 col 320
    r = rr.render(P, K, (0, 480, 0, 640), closing=0)
    assert r.mask[240, 320] and r.mask.sum() == 1


@pytest.mark.parametrize("kind", [0, 1, 2])
def test_packed_jacobians_finite_difference(kind):
    P, N = scene()
    T = look_at(np.array([1.6, 0.6, 0.9]), [0.6, 0.0, 0.45])
    crop = (0, 480, 0, 640)
    sem, depth = render_obs(se3_exp(np.r_[0.004, -0.003, 0.002, 0.004, 0.002, -0.003]) @ T, P, crop)
    V = np.tile([0.05, -0.02, 0.01], (len(P), 1))
    ev = {"crop": crop, "sem": sem, "class_names": ["background", "robot", "object", "fixture"], "depth": depth,
          "hw": (480, 640)}
    cfg = rr.RobotTermConfig(depth_points=200)
    ff, st = rr.frame_factors("c", P, N, V, T, K, ev, cfg, np.random.default_rng(0))
    ff.sdf = ff.sdf.astype(np.float64)
    pk = rr.pack([ff])["c"]
    keep = pk.kind == kind
    assert keep.sum() > 20
    alpha, tau, focal = 0.01, 0.005, 0.004

    def Kf(f):
        Ke = K.copy()
        Ke[0, 0] *= np.exp(f)
        Ke[1, 1] *= np.exp(f)
        return Ke

    r0, J = rr.packed_residuals(pk, T, Kf(focal), alpha, tau, cfg, focal=focal)
    eps = 1e-6
    for j in range(9):
        d = np.zeros(9)
        d[j] = eps

        def res(dd):
            return rr.packed_residuals(pk, se3_exp(dd[:6]) @ T, Kf(focal + dd[8]), alpha + dd[6], tau + dd[7], cfg,
                                       jac=False, focal=focal + dd[8])[0]

        num = (res(d) - res(-d)) / (2 * eps)
        a, b = num[keep], J[keep, j]
        if kind == 0:  # bilinear SDF: piecewise-linear, compare where the gradient is smooth
            ok = np.abs(a - b) < 1e-3 * (1 + np.abs(b)) + 1e-6
            assert ok.mean() > 0.95, (j, ok.mean())
        else:
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


def test_three_camera_recovery_from_robot_factors():
    """Three cameras see the object at several 'configurations' (rigid motions); starting 1 deg / 15 mm off, the robot
    factors alone (silhouettes both ways + point-to-plane) recover every camera to < 0.1 deg / 3 mm. The residual
    ~2 mm comes from the sampled (not rasterised) model contour: the extreme sample of a boundary pixel lies on
    average a fraction of the sample spacing inside the true outline (documented; probed on real data by the
    spacing variant)."""
    from s4d.realworld.rig_dw_frames import robot_normal_equations
    from s4d.realworld.rig_dw_extext import solve_step

    P0, N0 = scene()
    F0, _ = scene(spacing=0.0003)
    cams = {"a": look_at(np.array([1.7, 0.7, 0.9]), [0.6, 0.0, 0.45]),
            "b": look_at(np.array([1.7, -0.6, 0.8]), [0.6, 0.0, 0.45]),
            "c": look_at(np.array([1.9, 0.05, 0.6]), [0.6, 0.0, 0.45])}
    rng = np.random.default_rng(3)
    configs = []
    for k in range(5):
        G = se3_exp(np.r_[rng.normal(size=3) * 0.25, rng.normal(size=3) * 0.05])
        configs.append((P0 @ G[:3, :3].T + G[:3, 3], N0 @ G[:3, :3].T, F0 @ G[:3, :3].T + G[:3, 3]))
    crop = (0, 480, 0, 640)
    names = ["background", "robot", "object", "fixture"]
    obs = {(c, k): render_obs(T, P, crop, fine=F) for c, T in cams.items() for k, (P, _, F) in enumerate(configs)}
    est = {c: se3_exp(np.r_[rng.normal(size=3) * 0.012, rng.normal(size=3) * 0.01]) @ T for c, T in cams.items()}
    cfg = rr.RobotTermConfig(depth_points=150, sigma_sil_px=1.0)
    cols = {c: list(range(9 * i, 9 * i + 9)) for i, c in enumerate(cams)}
    state = {"T": est, "alpha": {c: 0.0 for c in cams}, "tau": {c: 0.0 for c in cams}}
    for outer in range(6):
        factors = []
        for (c, k), (sem, depth) in obs.items():
            P, N, _ = configs[k]
            ev = {"crop": crop, "sem": sem, "class_names": names, "depth": depth, "hw": (480, 640)}
            ff, _ = rr.frame_factors(c, P, N, np.zeros_like(P), state["T"][c], K, ev, cfg, np.random.default_rng(k))
            factors.append(ff)
        pk = rr.pack(factors)
        for _ in range(5):
            H, g, cost, _ = robot_normal_equations(pk, state, {c: K for c in cams}, cfg, lambda c: cols[c], 27)
            free = [j for j in range(27) if j % 9 < 6]  # poses only (alpha, tau, focal fixed)
            d = np.zeros(27)
            d[free] = solve_step(H[np.ix_(free, free)], g[free], 1e-6)
            state["T"] = {c: se3_exp(d[cols[c][:6]]) @ state["T"][c] for c in cams}
    for c in cams:
        rot, ctr = pose_distance(cams[c], state["T"][c])
        assert rot < 0.1 and ctr < 0.003, (c, rot, ctr)


def test_collada_loader_units_and_node_transform(tmp_path):
    from s4d.realworld.robot_visual import load_collada

    dae = """<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
 <asset><unit name="millimeter" meter="0.001"/><up_axis>Z_UP</up_axis></asset>
 <library_geometries><geometry id="g"><mesh>
  <source id="p"><float_array id="pa" count="9">0 0 0 10 0 0 0 10 0</float_array>
   <technique_common><accessor source="#pa" count="3" stride="3"/></technique_common></source>
  <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
  <triangles count="1"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2</p></triangles>
 </mesh></geometry></library_geometries>
 <library_visual_scenes><visual_scene id="s"><node id="n"><translate>0 0 5</translate>
  <instance_geometry url="#g"/></node></visual_scene></library_visual_scenes>
 <scene><instance_visual_scene url="#s"/></scene>
</COLLADA>"""
    p = tmp_path / "t.dae"
    p.write_text(dae)
    tri = load_collada(p)
    np.testing.assert_allclose(tri[0], np.array([[0, 0, 5], [10, 0, 5], [0, 10, 5]]) * 1e-3)
