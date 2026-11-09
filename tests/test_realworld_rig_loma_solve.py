"""2D-3D robot reprojection term of ``rig_loma_solve``: analytic Jacobians vs finite differences, Cauchy weighting,
and recovery of a perturbed 3-camera rig from synthetic robot 2D-3D correspondences + static RGB-D matches."""

import numpy as np

from s4d.realworld.calib.dw import extext as ee
from s4d.realworld.calib.dw.geom import pose_distance, project, se3_exp
from s4d.realworld.calib.dw.solve import NPC, SolveConfig, State
from s4d.realworld.calib.loma.solve import (
    CorrSet,
    LomaConfig,
    LomaProblem,
    corr_depth_residuals,
    corr_residuals,
    mount_information,
)

K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])


def look_at(C, target, up=(0, 0, 1)):
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ C
    return T


def rig():
    tgt = np.array([0.5, 0.0, 0.4])
    return {"c0": look_at(np.array([0.6, 1.3, 0.9]), tgt), "c1": look_at(np.array([0.7, -1.3, 0.9]), tgt),
            "c2": look_at(np.array([1.9, 0.1, 0.8]), tgt)}


def corr_for(T, rng, n=400, noise=0.3):
    X = np.array([0.5, 0.0, 0.4]) + rng.uniform(-0.3, 0.3, (n, 3))
    V = rng.normal(0, 0.05, (n, 3))
    uv, _ = project(X @ T[:3, :3].T + T[:3, 3], K)
    d = (X @ T[:3, :3].T + T[:3, 3])[:, 2]
    return CorrSet(X, V, uv + rng.normal(0, noise, uv.shape), np.zeros(n, int), np.arange(n) // 50,
                   rng.integers(0, 8, n), d * np.exp(rng.normal(0, 0.003, n)))


def test_corr_jacobians_match_finite_differences():
    rng = np.random.default_rng(1)
    T = rig()["c0"]
    cs = corr_for(T, rng, 20)
    tau, alpha = 0.012, 0.01
    r0, J = corr_residuals(T, K, cs, tau, 0.7)
    rd0, Jd, ok = corr_depth_residuals(T, cs, alpha, tau, 0.01)
    eps = 1e-6
    for j in range(NPC):
        d = np.zeros(NPC)
        d[j] = eps
        T1 = se3_exp(d[:6]) @ T
        K1 = K.copy()
        K1[0, 0] *= np.exp(d[8])
        K1[1, 1] *= np.exp(d[8])
        r1, _ = corr_residuals(T1, K1, cs, tau + d[7], 0.7, jac=False)
        assert np.allclose((r1 - r0) / eps, J[:, :, j], atol=1e-4, rtol=1e-4)
        rd1, _, _ = corr_depth_residuals(T1, cs, alpha + d[6], tau + d[7], 0.01, jac=False)
        if j != 8:
            assert np.allclose((rd1 - rd0) / eps, Jd[:, j], atol=1e-3, rtol=1e-4)


def test_rig_recovered_from_corr_and_static_matches():
    rng = np.random.default_rng(0)
    Tt = rig()
    cams = list(Tt)
    corr = {c: corr_for(Tt[c], rng) for c in cams}
    # static scene points seen by all pairs, with depth
    P = np.array([1.0, 0.0, 0.2]) + rng.uniform(-1.0, 1.0, (600, 3)) * np.array([0.6, 1.2, 0.3])
    ca, cb, ua, ub, da, db = [], [], [], [], [], []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        Pa = P @ Tt[cams[a]][:3, :3].T + Tt[cams[a]][:3, 3]
        Pb = P @ Tt[cams[b]][:3, :3].T + Tt[cams[b]][:3, 3]
        u1, _ = project(Pa, K)
        u2, _ = project(Pb, K)
        ok = (Pa[:, 2] > 0.3) & (Pb[:, 2] > 0.3)
        ca += [cams[a]] * ok.sum()
        cb += [cams[b]] * ok.sum()
        ua.append(u1[ok] + rng.normal(0, 0.2, (ok.sum(), 2)))
        ub.append(u2[ok] + rng.normal(0, 0.2, (ok.sum(), 2)))
        da.append(Pa[ok, 2])
        db.append(Pb[ok, 2])
    M = ee.MatchSet(np.array(ca), np.array(cb), np.concatenate(ua), np.concatenate(ub), np.concatenate(da),
                    np.concatenate(db), 0.5, 0.01)
    # common + individual perturbation of the start (2 deg / 3 cm)
    s0 = State({c: se3_exp(np.r_[rng.normal(0, 0.02, 3), rng.normal(0, 0.02, 3)]) @ Tt[c] for c in cams},
               {c: 0.0 for c in cams}, {c: 0.0 for c in cams})
    prob = LomaProblem(cams, {c: K for c in cams}, M, corr, LomaConfig(), SolveConfig(processes=1, inner_iters=10),
                       pivot=np.array([0.5, 0.0, 0.4]))
    s = prob.solve(s0)
    for c in cams:
        rot, cen = pose_distance(s.T[c], Tt[c])
        assert rot < 0.05 and cen < 0.002, (c, rot, cen)
    st = prob.corr_stats(s)
    assert all(v["median_px"] < 0.6 for v in st.values())
    # leave-one-camera-out: camera c2 from the static matches only, still close
    prob2 = LomaProblem(cams, {c: K for c in cams}, M, corr, LomaConfig(corr_cameras=("c0", "c1")),
                        SolveConfig(processes=1, inner_iters=10), pivot=np.array([0.5, 0.0, 0.4]))
    s2 = prob2.solve(s0)
    rot, cen = pose_distance(s2.T["c2"], Tt["c2"])
    assert rot < 0.1 and cen < 0.004
    info = mount_information(s, {c: K for c in cams}, corr, {5, 6, 7},
                             {c: np.tile(np.eye(4), (len(corr[c]), 1, 1)) for c in cams})
    assert len(info["marginal_over_cameras"]["eig"]) == 6


def test_mount_estimate_recovers_known_offset():
    from s4d.realworld.calib.loma.solve import apply_mount, estimate_mount

    rng = np.random.default_rng(3)
    Tt = rig()
    cams = list(Tt)
    dM = np.array([0.01, -0.02, 0.015, 0.002, -0.003, 0.004])  # rad, m
    corr = {}
    for c in cams:
        cs = corr_for(Tt[c], rng, 600, noise=0.0)
        n = len(cs)
        T7 = np.tile(np.eye(4), (n, 1, 1))
        # random link-7 frames (different configurations), gripper links = 5..7
        for i in range(n):
            T7[i] = se3_exp(np.r_[rng.normal(0, 0.5, 3), np.array([0.5, 0.0, 0.5]) + rng.normal(0, 0.1, 3)])
        cs.T7 = T7
        cs.link = rng.integers(0, 8, n)
        corr[c] = cs
    gl = {5, 6, 7}
    # observations generated with the TRUE mount (dM applied), model points without it
    truth = apply_mount(corr, gl, dM)
    obs = {}
    for c in cams:
        uv, _ = project(truth[c].X @ Tt[c][:3, :3].T + Tt[c][:3, 3], K)
        cs = corr[c]
        obs[c] = CorrSet(cs.X, np.zeros_like(cs.V), uv, cs.episode, cs.frame, cs.link, None, None, {}, cs.T7)
    s = State(dict(Tt), {c: 0.0 for c in cams}, {c: 0.0 for c in cams})
    est = estimate_mount(s, {c: K for c in cams}, obs, gl, iters=10)
    assert np.allclose(est["dM_rad_m"], dM, atol=2e-4)


def test_disparity_offset_fit_recovers_beta():
    from s4d.realworld.calib.loma.run import fit_disparity_offset, offset_matches

    rng = np.random.default_rng(5)
    Tt = rig()
    cams = list(Tt)
    P = np.array([1.0, 0.0, 0.2]) + rng.uniform(-1.0, 1.0, (3000, 3)) * np.array([1.5, 2.0, 0.5])
    beta = {"c0": 0.004, "c1": -0.002, "c2": 0.006}
    ca, cb, da, db, keep = [], [], [], [], []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        za = (P @ Tt[cams[a]][:3, :3].T + Tt[cams[a]][:3, 3])[:, 2]
        zb = (P @ Tt[cams[b]][:3, :3].T + Tt[cams[b]][:3, 3])[:, 2]
        ok = (za > 0.5) & (zb > 0.5)
        # observed d with true z = d / (1 + beta d)  ->  d = z / (1 - beta z)
        ca += [cams[a]] * ok.sum()
        cb += [cams[b]] * ok.sum()
        da.append(za[ok] / (1 - beta[cams[a]] * za[ok]))
        db.append(zb[ok] / (1 - beta[cams[b]] * zb[ok]))
        keep.append(P[ok])
    n = sum(len(k) for k in keep)
    M = ee.MatchSet(np.array(ca), np.array(cb), np.zeros((n, 2)), np.zeros((n, 2)), np.concatenate(da),
                    np.concatenate(db), 0.5, 0.01)
    s = State(dict(Tt), {c: 0.0 for c in cams}, {c: 0.0 for c in cams})
    s.P = np.concatenate(keep)
    est = fit_disparity_offset(s, M, cams)
    for c in cams:
        assert abs(est[c] - beta[c]) < 3e-4, (c, est[c], beta[c])
    M2 = offset_matches(M, beta)
    z_true = (s.P @ Tt["c0"][:3, :3].T + Tt["c0"][:3, 3])[:, 2]
    sel = M2.cam_a == "c0"
    assert np.allclose(M2.d_a[sel], z_true[sel], rtol=1e-9)
