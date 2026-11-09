"""Synthetic numerical checks for board-free two-view rig calibration.

No recording, learned matcher, robot model, or board observations are used.
The noisy recovery checks validate this likelihood on independent synthetic
observations; they do not establish accuracy on real recordings.
"""

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from s4d.realworld.calib.dw.extext import (
    MatchSet,
    ParamIndex,
    _linearize,
    apply_delta,
    fit_extext,
    init_points,
    normal_equations,
    reduced_residual_stats,
    refine_points,
    solve_step,
)
from s4d.realworld.calib.dw.geom import (
    camera_centre,
    pose_distance,
    project,
    scaled_info_svd,
    se3_exp,
    se3_log,
    skew,
    so3_exp,
    so3_log,
)


def _scene(n=3000, seed=382, noisy=True, outliers=0.0):
    rng = np.random.default_rng(seed)
    T, K = {}, {}
    centres = [[-.36, 0, 0], [.36, .02, .01], [.02, .42, -.04]]
    for c, centre in enumerate(centres):
        cam, centre = f"c{c}", np.asarray(centre)
        z = np.array([0, 0, 1.6]) - centre
        z /= np.linalg.norm(z)
        x = np.cross([0, 1, 0], z)
        x /= np.linalg.norm(x)
        R = np.stack((x, np.cross(z, x), z))
        T[cam] = np.eye(4)
        T[cam][:3, :3], T[cam][:3, 3] = R, -R @ centre
        K[cam] = np.array([[609.0 + c, 0, 320.5], [0, 605.0 - c, 240.5], [0, 0, 1]])
    alpha = dict(zip(T, np.log([1, 1.01, .985])))
    arrays = {name: [] for name in ('cam_a', 'cam_b', 'uv_a', 'uv_b', 'd_a', 'd_b')}
    points, bad_ids = [], []
    for pair, (a, b) in enumerate((('c0', 'c1'), ('c0', 'c2'), ('c1', 'c2'))):
        P = rng.uniform([-.26, -.23, .7], [.26, .23, 2.5], (n, 3))
        points.append(P)
        arrays['cam_a'].append(np.full(n, a))
        arrays['cam_b'].append(np.full(n, b))
        for side, cam in (('a', a), ('b', b)):
            pc = P @ T[cam][:3, :3].T + T[cam][:3, 3]
            uv, _ = project(pc, K[cam])
            raw = pc[:, 2] * np.exp(-alpha[cam])
            if noisy:
                uv += rng.normal(0, .3, uv.shape)
                raw *= np.exp(rng.normal(0, .01, n))
            arrays[f'uv_{side}'].append(uv)
            arrays[f'd_{side}'].append(raw)
        bad = rng.choice(n, int(n * outliers), replace=False)
        # Gross mismatches and wrong depths, not just heavy-tailed sensor noise.
        arrays['uv_b'][-1][bad] += rng.uniform(30, 100, (len(bad), 2)) * rng.choice([-1, 1], (len(bad), 2))
        arrays['d_b'][-1][bad] *= np.exp(rng.uniform(-.4, .4, len(bad)))
        bad_ids.extend(pair * n + bad)
    M = MatchSet(**{name: np.concatenate(parts) for name, parts in arrays.items()}, sigma_px=.3, sigma_logd=.01)
    return T, alpha, K, M, np.concatenate(points), np.asarray(bad_ids, dtype=int)


def _perturb(T, seed=23):
    rng = np.random.default_rng(seed)
    out = {cam: pose.copy() for cam, pose in T.items()}
    for cam in ('c1', 'c2'):
        rotation, translation = rng.normal(size=(2, 3))
        xi = np.r_[rotation / np.linalg.norm(rotation) * np.deg2rad(2),
                   translation / np.linalg.norm(translation) * .030]
        out[cam] = se3_exp(xi) @ out[cam]
    return out


def _index():
    return ParamIndex({'c0': (False, False), 'c1': (True, True), 'c2': (True, True)})


def _subset(M, ids, **overrides):
    fields = ('cam_a', 'cam_b', 'uv_a', 'uv_b', 'd_a', 'd_b', 'sigma_px', 'sigma_logd')
    args = {name: getattr(M, name)[ids] for name in fields}
    args.update(overrides)
    return MatchSet(**args)


def _observed_residual(T, alpha, K, M, P):
    """Independent literal residual, including masks and per-match sigmas."""
    r = np.zeros((len(M), 6))
    for i in range(len(M)):
        for cam, uv, depth, row, dr in ((M.cam_a[i], M.uv_a[i], M.d_a[i], 0, 4),
                                       (M.cam_b[i], M.uv_b[i], M.d_b[i], 2, 5)):
            pc = T[cam][:3, :3] @ P[i] + T[cam][:3, 3]
            q = K[cam] @ pc
            r[i, row:row + 2] = (q[:2] / q[2] - uv) / M.sigma_px[i]
            if np.isfinite(depth) and depth > 0:
                r[i, dr] = (np.log(pc[2]) - np.log(depth) - alpha[cam]) / M.sigma_logd[i]
    return r


@pytest.mark.parametrize('angle', [0, 1e-10, 1e-5, .7, np.pi - 1e-9, np.pi])
def test_lie_roundtrip_and_matrix_exponential(angle):
    axis = np.array([.2, -.7, .3])
    axis /= np.linalg.norm(axis)
    omega = axis * angle
    xi = np.r_[omega, [.13, -.04, .22]]
    T = se3_exp(xi)
    algebra = np.zeros((4, 4))
    algebra[:3, :3], algebra[:3, 3] = skew(omega), xi[3:]
    np.testing.assert_allclose(T, expm(algebra), atol=2e-12)
    np.testing.assert_allclose(se3_exp(se3_log(T)), T, atol=2e-9)
    np.testing.assert_allclose(so3_exp(so3_log(T[:3, :3])), T[:3, :3], atol=2e-9)
    if angle < np.pi - 1e-8:
        np.testing.assert_allclose(se3_log(T), xi, atol=1e-11)
    batch = np.stack((xi, -xi, np.zeros(6)))
    np.testing.assert_allclose(se3_exp(se3_log(se3_exp(batch))), se3_exp(batch), atol=2e-9)
    np.testing.assert_allclose(T @ se3_exp(-xi), np.eye(4), atol=2e-12)


def test_projection_centres_information_and_parameter_order():
    rng = np.random.default_rng(4)
    pc = rng.uniform([-.4, -.4, .7], [.4, .4, 2.5], (11, 3))
    K = np.array([[600., 0, 320.5], [0, 610, 240.5], [0, 0, 1]])
    uv, jac = project(pc, K)
    np.testing.assert_allclose(project([[0., 0., 1]], K)[0], [[320.5, 240.5]])
    for j in range(3):
        dp = np.eye(3)[j] * 1e-6
        numeric = (project(pc + dp, K)[0] - project(pc - dp, K)[0]) / 2e-6
        np.testing.assert_allclose(jac[:, :, j], numeric, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(project(pc, np.broadcast_to(K, (11, 3, 3)))[0], uv)
    invalid_uv, invalid_jac = project([[0, 0, -1], [0, 0, 0]], K)
    assert np.isnan(invalid_uv).all() and np.isnan(invalid_jac).all()
    Ta = se3_exp([.1, -.2, .3, .1, .2, -.3])
    C = camera_centre(Ta)
    np.testing.assert_allclose(Ta[:3, :3] @ C + Ta[:3, 3], 0, atol=1e-15)
    Tb = Ta.copy()
    Tb[:3, :3] = so3_exp([0, np.deg2rad(.3), 0]) @ Ta[:3, :3]
    Tb[:3, 3] = -Tb[:3, :3] @ (C + [.008, 0, 0])
    np.testing.assert_allclose(pose_distance(Ta, Tb), [.3, .008], atol=1e-12)
    J, w, d = rng.normal(size=(20, 8)), rng.uniform(.1, 1, 20), rng.uniform(.01, .1, 8)
    for W, D in ((w, d), (np.diag(w), np.diag(d))):
        s, vt = scaled_info_svd(J, W, D)
        expected = d[:, None] * (J.T @ (w[:, None] * J)) * d
        np.testing.assert_allclose((vt.T * s**2) @ vt, expected, atol=1e-14)
    Q = Rotation.from_rotvec([.2, -.3, .1]).as_matrix()
    W = Q @ np.diag([.3, 1, 2]) @ Q.T
    s, vt = scaled_info_svd(J[:3], W, d)
    np.testing.assert_allclose((vt.T * s**2) @ vt, d[:, None] * (J[:3].T @ W @ J[:3]) * d, atol=1e-14)
    assert s.shape == (8,) and vt.shape == (8, 8) and np.count_nonzero(s == 0) == 5
    with pytest.raises(ValueError):
        scaled_info_svd(J, -w, d)
    idx = ParamIndex({'b': (True, False), 'a': (False, True), 'c': (False, False)}, order=['a', 'b', 'c'])
    assert idx.n == 7 and idx.alpha['a'] == 0 and idx.pose['b'] == slice(1, 7)
    assert idx.pose_offset['b'] == 1 and idx.pose['a'] is None and idx.alpha['c'] is None
    with pytest.raises(ValueError, match='exactly once'):
        ParamIndex({'a': (True, False), 'b': (False, False)}, order=['a', 'b', 'a'])


@pytest.mark.parametrize('depth_mode', ['both', 'mixed', 'none'])
def test_whitened_residual_jacobians_central_difference(depth_mode):
    T, alpha, K, M, P, _ = _scene(n=5)
    M = _subset(M, slice(0, 5), sigma_px=np.linspace(.2, .6, 5), sigma_logd=np.linspace(.008, .02, 5))
    P = P[:5] + [.002, -.001, .004]
    if depth_mode == 'mixed':
        M = _subset(M, slice(None), d_a=[np.nan, 0, -1, 1.4, 1.7], d_b=[1.2, np.nan, -1, 0, 1.9])
    if depth_mode == 'none':
        M = _subset(M, slice(None), d_a=np.nan, d_b=0)
    idx = ParamIndex({'c0': (True, True), 'c1': (True, True)})
    ids = np.arange(len(M))
    r, jp, jx, cols = _linearize(T, alpha, K, M, P, 'c0', 'c1', ids, idx)
    np.testing.assert_array_equal(cols, np.arange(14))
    np.testing.assert_allclose(r, _observed_residual(T, alpha, K, M, P), atol=1e-12)
    eps = 2e-7
    numeric_x = np.empty_like(jx)
    for j in range(idx.n):
        delta = np.eye(idx.n)[j] * eps
        tp, ap = apply_delta(T, alpha, idx, delta)
        tm, am = apply_delta(T, alpha, idx, -delta)
        numeric_x[:, :, j] = (_observed_residual(tp, ap, K, M, P) - _observed_residual(tm, am, K, M, P)) / (2 * eps)
    numeric_p = np.empty_like(jp)
    for j in range(3):
        dp = np.eye(3)[j] * eps
        numeric_p[:, :, j] = (_observed_residual(T, alpha, K, M, P + dp)
                               - _observed_residual(T, alpha, K, M, P - dp)) / (2 * eps)
    for analytic, numeric in ((jx, numeric_x), (jp, numeric_p)):
        relative = np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric)
        assert relative < 1e-5
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize('cauchy_c', [3., np.inf])
def test_schur_equals_full_joint_marginal_and_chunking(cauchy_c):
    T, alpha, K, M, P, _ = _scene(n=4)
    # Include both depth-missing and one-sided factors across all three pairs.
    da, db = M.d_a.copy(), M.d_b.copy()
    da[[0, 2, 6]], db[[0, 1, 7]] = np.nan, 0
    M = _subset(M, slice(None), d_a=da, d_b=db)
    idx = ParamIndex({'c0': (False, False), 'c1': (True, True), 'c2': (True, True)},
                     order=['c2', 'c0', 'c1'])
    result = normal_equations(T, alpha, K, M, P, idx, cauchy_c, chunk_size=2)
    full_J = np.zeros((len(M) * 6, idx.n + len(M) * 3))
    full_r = np.zeros(len(M) * 6)
    for i in range(len(M)):
        r, jp, jx, cols = _linearize(T, alpha, K, M, P[i:i + 1], M.cam_a[i], M.cam_b[i], [i], idx)
        rows = np.arange(6 * i, 6 * i + 6)
        full_J[np.ix_(rows, cols)] = jx[0]
        full_J[rows, idx.n + 3 * i:idx.n + 3 * i + 3] = jp[0]
        full_r[rows] = r[0]
    weights = np.repeat(np.sqrt(result['weight']), 6)
    full_J *= weights[:, None]
    full_r *= weights
    H, g = full_J.T @ full_J, full_J.T @ full_r
    A, B, C = H[:idx.n, :idx.n], H[:idx.n, idx.n:], H[idx.n:, idx.n:]
    expected_H = A - B @ np.linalg.solve(C, B.T)
    expected_g = g[:idx.n] - B @ np.linalg.solve(C, g[idx.n:])
    np.testing.assert_allclose(result['H'], expected_H, rtol=1e-9, atol=2e-7)
    np.testing.assert_allclose(result['g'], expected_g, rtol=1e-9, atol=2e-9)
    np.testing.assert_allclose(result['chi2'], np.sum(_observed_residual(T, alpha, K, M, P)**2, axis=1))
    expected_w = 1 / (1 + result['chi2'] / cauchy_c**2)
    np.testing.assert_allclose(result['weight'], expected_w)
    rho = result['chi2'] if np.isinf(cauchy_c) else cauchy_c**2 * np.log1p(result['chi2'] / cauchy_c**2)
    assert result['cost'] == pytest.approx(np.sum(rho))
    np.testing.assert_array_equal(result['r_dim'], 4 + (da > 0).astype(int) + (db > 0))
    all_at_once = normal_equations(T, alpha, K, M, P, idx, cauchy_c, chunk_size=100)
    np.testing.assert_allclose(result['H'], all_at_once['H'], rtol=1e-9, atol=2e-7)
    np.testing.assert_allclose(result['g'], all_at_once['g'], rtol=1e-9, atol=2e-9)
    # Left poses and fixed blocks, with no mutation of either input dictionary.
    delta = np.linspace(-.001, .001, idx.n)
    updated_T, updated_alpha = apply_delta(T, alpha, idx, delta)
    np.testing.assert_array_equal(updated_T['c0'], T['c0'])
    assert updated_alpha['c0'] == alpha['c0']
    np.testing.assert_allclose(updated_T['c1'], se3_exp(delta[idx.pose['c1']]) @ T['c1'])
    Hsmall = np.array([[2., .2], [.2, 3.]])
    gsmall = np.array([.1, -.4])
    step = solve_step(Hsmall, gsmall, .1)
    np.testing.assert_allclose((Hsmall + .1 * np.diag(np.diag(Hsmall))) @ step, -gsmall)


def test_initialisation_point_refinement_and_validation_metrics():
    T, alpha, K, M, P, _ = _scene(n=8, noisy=False)
    # Replace raw depths by true z so depth-based initialisation is exact.
    da = M.d_a * np.exp([alpha[c] for c in M.cam_a])
    db = M.d_b * np.exp([alpha[c] for c in M.cam_b])
    da[[0, 1, 2]], db[[0, 1, 3]] = np.nan, -1
    zero = dict.fromkeys(T, 0.)
    exact_M = _subset(M, slice(None), d_a=da, d_b=db)
    initial = init_points(T, K, exact_M)
    np.testing.assert_allclose(initial, P, atol=1e-12)
    displaced = P + [.015, -.01, .03]
    refined = refine_points(T, zero, K, exact_M, displaced, iters=6, chunk_size=3)
    np.testing.assert_allclose(refined, P, atol=2e-11)
    np.testing.assert_allclose(displaced, P + [.015, -.01, .03])
    stats = reduced_residual_stats(T, alpha, K, M)
    np.testing.assert_allclose(stats['epipolar_px'], 0, atol=1e-12)
    np.testing.assert_allclose(stats['transfer_px'], 0, atol=1e-12)
    missing = reduced_residual_stats(T, zero, K, exact_M)
    assert np.isnan(missing['transfer_px'][[0, 1, 2]]).all()
    assert np.isfinite(missing['epipolar_px']).all()
    # An epipolar-normal displacement is visible without depth or triangulation.
    moved_uv = M.uv_b.copy()
    moved_uv[:, 1] += 2
    moved = _subset(M, slice(None), uv_b=moved_uv)
    assert np.median(reduced_residual_stats(T, alpha, K, moved)['epipolar_px']) > .8


@pytest.mark.parametrize('outlier_fraction', [0., .10])
def test_three_camera_noisy_recovery(outlier_fraction):
    T, alpha, K, M, _, bad = _scene(outliers=outlier_fraction)
    T0, alpha0 = _perturb(T), dict.fromkeys(T, 0.)
    original_T0 = {cam: pose.copy() for cam, pose in T0.items()}
    idx = _index()
    fitted, scales, P, info = fit_extext(T0, alpha0, K, M, idx)
    errors = {cam: pose_distance(T[cam], fitted[cam]) for cam in T}
    # 3000 matches per pair, 0.3 px image noise, 1% log-depth noise.
    for cam, (rotation, centre) in errors.items():
        assert rotation < .02, (cam, errors, info['termination'])
        assert centre < .001, (cam, errors, info['termination'])
        assert abs(scales[cam] - alpha[cam]) < 1e-3, (cam, scales, alpha)
        np.testing.assert_array_equal(T0[cam], original_T0[cam])
        assert alpha0[cam] == 0
    np.testing.assert_array_equal(fitted['c0'], T['c0'])
    assert scales['c0'] == alpha['c0']
    assert info['iterations'] > 0
    assert np.all(np.diff(info['cost_history']) < 0)
    # Gross mismatches retain an irreducible Cauchy cost even at the true pose.
    assert info['cost_history'][-1] < info['cost_history'][0] * (.25 if outlier_fraction else .1)
    assert info['H'].shape == (14, 14)
    assert np.linalg.eigvalsh(info['H'])[0] > 0
    final = normal_equations(fitted, scales, K, M, P, idx)
    np.testing.assert_array_equal(info['H'], final['H'])
    if outlier_fraction:
        good = np.ones(len(M), dtype=bool)
        good[bad] = False
        assert np.median(info['weight'][bad]) < .005
        assert np.median(info['weight'][good]) > .5


def test_image_only_matches_constrain_rotation_with_scale_gauge():
    T, alpha, K, M, _, _ = _scene(n=700, seed=394, noisy=False)
    M = _subset(M, slice(0, 700), d_a=np.nan, d_b=0)
    idx = ParamIndex({'c0': (False, False), 'c1': (True, False), 'c2': (False, False)})
    T0 = _perturb(T)
    initial = init_points(T0, K, M)
    initial = refine_points(T0, alpha, K, M, initial, iters=5)
    before = normal_equations(T0, alpha, K, M, initial, idx)
    eigen = np.linalg.eigvalsh(before['H'])
    # Image-only calibrated two-view data have five observable pose directions.
    assert abs(eigen[0]) < eigen[-1] * 1e-10 and eigen[1] > eigen[-1] * 1e-5
    fitted, _, _, info = fit_extext(T0, alpha, K, M, idx, cauchy_c=np.inf)
    assert pose_distance(T['c1'], fitted['c1'])[0] < 1e-5
    assert info['cost'] < before['cost'] * 1e-10
    Ca = camera_centre(T['c0'])
    true_direction, fitted_direction = camera_centre(T['c1']) - Ca, camera_centre(fitted['c1']) - Ca
    np.testing.assert_allclose(true_direction / np.linalg.norm(true_direction),
                               fitted_direction / np.linalg.norm(fitted_direction), atol=1e-7)


def test_singular_points_empty_matches_and_input_validation():
    T = {'a': np.eye(4), 'b': np.eye(4)}
    K = dict.fromkeys(T, np.eye(3))
    alpha = dict.fromkeys(T, 0.)
    M = MatchSet(['a'], ['b'], [[.1, .2]], [[.1, .2]], [np.nan], [0])
    P = init_points(T, K, M)
    assert np.isfinite(P).all()
    refined = refine_points(T, alpha, K, M, P + [.01, -.01, .02])
    assert np.isfinite(refined).all()
    idx = ParamIndex({'a': (False, False), 'b': (True, False)})
    normal = normal_equations(T, alpha, K, M, refined, idx)
    assert np.isfinite(normal['H']).all()
    assert np.isnan(reduced_residual_stats(T, alpha, K, M)['epipolar_px']).all()
    empty = _subset(M, slice(0, 0))
    fixed = ParamIndex(dict.fromkeys(T, (False, False)))
    _, _, points, info = fit_extext(T, alpha, K, empty, fixed)
    assert points.shape == (0, 3) and info['cost'] == 0 and info['H'].shape == (0, 0)
    with pytest.raises(ValueError, match='different cameras'):
        MatchSet(['a'], ['a'], [[0, 0]], [[0, 0]], [1], [1])
    with pytest.raises(ValueError, match='positive'):
        _subset(M, slice(None), sigma_px=0)
    with pytest.raises(ValueError, match='front'):
        refine_points(T, alpha, K, M, [[0, 0, -1]])


@pytest.mark.parametrize('dtype', [int, object])
def test_integer_and_object_camera_arrays(dtype):
    T, alpha, K, M, P, _ = _scene(n=3, noisy=False)
    cameras = list(T)
    alias = {cam: i if dtype is int else cam for i, cam in enumerate(cameras)}
    remapped = _subset(M, slice(None), cam_a=np.array([alias[c] for c in M.cam_a], dtype=dtype),
                      cam_b=np.array([alias[c] for c in M.cam_b], dtype=dtype))
    new_T, new_K, new_alpha = ({alias[c]: values[c] for c in cameras} for values in (T, K, alpha))
    idx = ParamIndex({alias[c]: (c != 'c0', c != 'c0') for c in cameras})
    normal = normal_equations(new_T, new_alpha, new_K, remapped, P, idx)
    original = normal_equations(T, alpha, K, M, P, _index())
    np.testing.assert_allclose(normal['H'], original['H'], atol=1e-8)
    np.testing.assert_allclose(normal['g'], original['g'], atol=1e-9)



def test_focal_column_finite_difference():
    """The optional log focal multiplier column of the ext-ext factor (image rows: d u / d eps = u - c)."""
    from s4d.realworld.calib.dw.extext import ParamIndex, _linearize, init_points

    T, alpha, K, M, _, _ = _scene(outliers=0.0)
    cams = sorted(T)
    idx = ParamIndex({c: (True, True, True) for c in cams}, order=cams)
    P = init_points(T, K, M)
    a, b, ids = M._groups[0]
    ids = ids[:50]
    r0, jp, jx, cols = _linearize(T, alpha, K, M, P[ids], a, b, ids, idx)
    for cam in (a, b):
        col = list(cols).index(idx.focal[cam])
        e = 1e-6
        def rr(sgn):
            Kp = {c: k.copy() for c, k in K.items()}
            Kp[cam][0, 0] *= np.exp(sgn * e)
            Kp[cam][1, 1] *= np.exp(sgn * e)
            return _linearize(T, alpha, Kp, M, P[ids], a, b, ids)[0]
        num = (rr(1) - rr(-1)) / (2 * e)
        np.testing.assert_allclose(num, jx[:, :, col], rtol=1e-5, atol=1e-6)
