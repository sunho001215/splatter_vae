"""Two-view RGB-D factors with locally eliminated base-frame points.

For each match the whitened residual is [pi_a-u_a, pi_b-u_b,
log(z_a)-log(d_a)-alpha_a, log(z_b)-log(d_b)-alpha_b], with the image
and log-depth rows divided by their respective sigmas. Missing depths have
zero rows. Depth corrections satisfy true camera-z = exp(alpha) * raw depth.

Poses map base to camera and use camera-frame LEFT twists (omega, v).
The connected ext-ext problem has a seven-dimensional similarity gauge:
fix one camera pose and one depth scale. Image-only data retain a metric
scale ambiguity even if every alpha is fixed; depth or a baseline constraint
elsewhere is required. No priors or damping enter the reported information.

Only NumPy is required. Match operations are batched within camera-pair
chunks; a global N-by-number-of-parameters Jacobian is never allocated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rig_dw_geom import camera_centre, project, se3_exp, skew


@dataclass
class MatchSet:
    """Continuous pixels, raw camera-z depths, and positive noise sigmas.

    Camera arrays contain dict keys (string ids or integer indices). Sigmas
    may be scalars or (N,) arrays. Nonfinite or nonpositive depths are missing.
    Observed pixels must be finite. Inputs are not modified by the solver.
    """

    cam_a: np.ndarray
    cam_b: np.ndarray
    uv_a: np.ndarray
    uv_b: np.ndarray
    d_a: np.ndarray
    d_b: np.ndarray
    sigma_px: np.ndarray = .5
    sigma_logd: np.ndarray = .01
    _groups: list = field(init=False, repr=False)

    def __post_init__(self):
        self.cam_a, self.cam_b = np.asarray(self.cam_a), np.asarray(self.cam_b)
        n = len(self.cam_a)
        if self.cam_a.shape != (n,) or self.cam_b.shape != (n,):
            raise ValueError("camera ids must have shape (N,)")
        if np.any(self.cam_a == self.cam_b):
            raise ValueError("a match must involve two different cameras")
        for name in ("uv_a", "uv_b"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (n, 2) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape (N,2)")
            setattr(self, name, value)
        for name in ("d_a", "d_b", "sigma_px", "sigma_logd"):
            value = np.broadcast_to(np.asarray(getattr(self, name), dtype=float), (n,))
            if name.startswith("sigma") and (not np.isfinite(value).all() or np.any(value <= 0)):
                raise ValueError(f"{name} must be positive and finite")
            setattr(self, name, value)
        # Encode first so object arrays of string ids also support axis-wise unique.
        cameras, codes = np.unique(np.concatenate((self.cam_a, self.cam_b)), return_inverse=True)
        pairs, inverse = np.unique(np.stack((codes[:n], codes[n:]), axis=1), axis=0, return_inverse=True)
        self._groups = [(cameras[a], cameras[b], np.flatnonzero(inverse == k)) for k, (a, b) in enumerate(pairs)]

    def __len__(self):
        return len(self.cam_a)


class ParamIndex:
    """Pack free camera blocks in mapping insertion order (or explicit order).

    ``ParamIndex({'c0': (False, False), 'c1': (True, True)})`` fixes c0's
    pose and alpha. ``pose[cam]`` is a six-column slice or None;
    ``alpha[cam]`` is a scalar column index or None. Offsets are also exposed
    as ``pose_offset``. Each camera's pose precedes its alpha column.
    """

    def __init__(self, free, order=None):
        self.order = tuple(free if order is None else order)
        if len(self.order) != len(free) or set(self.order) != set(free):
            raise ValueError("order must contain each camera exactly once")
        self.pose, self.alpha, self.pose_offset = {}, {}, {}
        self.n = 0
        for cam in self.order:
            pose_free, alpha_free = free[cam]
            self.pose_offset[cam] = self.n if pose_free else None
            self.pose[cam] = slice(self.n, self.n + 6) if pose_free else None
            self.n += 6 * bool(pose_free)
            self.alpha[cam] = self.n if alpha_free else None
            self.n += bool(alpha_free)
        self.n_params = self.n


def _chunks(M, chunk_size):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for a, b, ids in M._groups:
        for start in range(0, len(ids), chunk_size):
            yield a, b, ids[start:start + chunk_size]


def _rays(uv, K):
    return np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(K).T


def _valid_depth(d):
    return np.isfinite(d) & (d > 0)


def init_points(T, K, M):
    """Raw depth lifts in base coordinates, averaged if both are available.

    With no depth, use the midpoint of the closest points on the two unit
    bearing lines. Parallel or non-cheiral triangulations fall back to the
    mean of points one metre along each bearing. Alpha is intentionally not
    part of this initialisation; point refinement applies its correction.
    """
    P = np.empty((len(M), 3))
    for a, b, ids in _chunks(M, 50_000):
        Ra, Rb = T[a][:3, :3], T[b][:3, :3]
        Ca, Cb = camera_centre(T[a]), camera_centre(T[b])
        qa, qb = _rays(M.uv_a[ids], K[a]) @ Ra, _rays(M.uv_b[ids], K[b]) @ Rb
        da, db = M.d_a[ids], M.d_b[ids]
        va, vb = _valid_depth(da), _valid_depth(db)
        pa = qa * np.where(va, da, 0)[:, None] + Ca
        pb = qb * np.where(vb, db, 0)[:, None] + Cb
        points = (pa * va[:, None] + pb * vb[:, None]) / np.maximum(va.astype(int) + vb, 1)[:, None]
        missing = ~(va | vb)
        if np.any(missing):
            u, v = qa[missing], qb[missing]
            u = u / np.linalg.norm(u, axis=1)[:, None]
            v = v / np.linalg.norm(v, axis=1)[:, None]
            dot = np.einsum("ni,ni->n", u, v)
            gap = Cb - Ca
            du, dv = u @ gap, v @ gap
            denom = 1 - dot**2
            sa = (du - dot * dv) / np.maximum(denom, 1e-12)
            sb = (dot * du - dv) / np.maximum(denom, 1e-12)
            good = (denom > 1e-10) & (sa > 0) & (sb > 0)
            midpoint = (Ca + u * sa[:, None] + Cb + v * sb[:, None]) / 2
            points[missing] = np.where(good[:, None], midpoint, (Ca + u + Cb + v) / 2)
        P[ids] = points
    return P


def _linearize(T, alpha, K, M, P, a, b, ids, idx=None):
    """Padded six-row residuals, base-point Jacobians, local pose/scale columns.

    Returned tuple is (r, JP, Jx, global_column_ids). Missing depth rows are
    identically zero, including their Jacobians. This is also the analytic
    residual/Jacobian interface used by the finite-difference tests.
    """
    n = len(ids)
    r, jp = np.zeros((n, 6)), np.zeros((n, 6, 3))
    blocks, columns = [], []
    for cam, uv, depth, image_rows, depth_row in (
        (a, M.uv_a, M.d_a, slice(0, 2), 4), (b, M.uv_b, M.d_b, slice(2, 4), 5)
    ):
        R, t = T[cam][:3, :3], T[cam][:3, 3]
        pc = P @ R.T + t
        pixels, dpi = project(pc, K[cam])
        jc = np.zeros((n, 6, 3))
        r[:, image_rows] = (pixels - uv[ids]) / M.sigma_px[ids, None]
        jc[:, image_rows] = dpi / M.sigma_px[ids, None, None]
        valid = _valid_depth(depth[ids])
        z = np.where(pc[:, 2] > 0, pc[:, 2], np.nan)
        r[valid, depth_row] = (np.log(z[valid]) - np.log(depth[ids][valid]) - alpha[cam]) / M.sigma_logd[ids][valid]
        jc[valid, depth_row, 2] = 1 / (z[valid] * M.sigma_logd[ids][valid])
        jp += jc @ R
        if idx is not None and idx.pose[cam] is not None:
            blocks.append(np.concatenate((jc @ -skew(pc), jc), axis=2))
            columns.extend(range(idx.pose[cam].start, idx.pose[cam].stop))
        if idx is not None and idx.alpha[cam] is not None:
            ja = np.zeros((n, 6, 1))
            ja[valid, depth_row, 0] = -1 / M.sigma_logd[ids][valid]
            blocks.append(ja)
            columns.append(idx.alpha[cam])
    jx = np.concatenate(blocks, axis=2) if blocks else np.empty((n, 6, 0))
    return r, jp, jx, np.asarray(columns, dtype=int)


def refine_points(T, alpha, K, M, P, iters=3, *, chunk_size=50_000):
    """Independent, vectorised 3x3 Gauss-Newton point solves (no input mutation).

    Tiny diagonal Levenberg damping handles singular point blocks; per-point
    backtracking keeps cost nonincreasing and both camera depths positive.
    A common factor IRLS weight cancels from each point's minimisation.
    Initial points must be finite and in front of both cameras.
    """
    out = np.asarray(P, dtype=float).copy()
    if out.shape != (len(M), 3):
        raise ValueError("P must have shape (N,3)")
    for a, b, ids in _chunks(M, chunk_size):
        points = out[ids]
        for _ in range(iters):
            r, jp, _, _ = _linearize(T, alpha, K, M, points, a, b, ids)
            if not np.isfinite(r).all():
                raise ValueError("points must be finite and in front of both cameras")
            C = np.einsum("nri,nrj->nij", jp, jp)
            grad = np.einsum("nri,nr->ni", jp, r)
            diag = np.arange(3)
            C[:, diag, diag] += 1e-8 * np.maximum(C[:, diag, diag], 1)
            step = -np.linalg.solve(C, grad[..., None])[..., 0]
            old_cost = np.sum(r * r, axis=1)
            pending = np.ones(len(ids), dtype=bool)
            for _ in range(12):
                trial = points + step
                rr, _, _, _ = _linearize(T, alpha, K, M, trial, a, b, ids)
                trial_cost = np.sum(rr * rr, axis=1)
                accept = pending & np.isfinite(trial_cost) & (trial_cost <= old_cost + 1e-12)
                points[accept] = trial[accept]
                pending &= ~accept
                if not np.any(pending):
                    break
                step[pending] *= .5
        out[ids] = points
    return out


def _point_inverse(C):
    # Exact inverse on identifiable directions, Moore-Penrose on singular ones.
    eigen, vectors = np.linalg.eigh(C)
    cutoff = 1e-12 * np.maximum(eigen[:, -1:], 1e-300)
    inv = np.zeros_like(eigen)
    np.divide(1, eigen, out=inv, where=eigen > cutoff)
    return (vectors * inv[:, None, :]) @ np.swapaxes(vectors, 1, 2)


def normal_equations(T, alpha, K, M, P, idx, cauchy_c=3.0, *, chunk_size=50_000):
    """Accumulate the undamped point-marginalised IRLS H and g in float64.

    chi2 is the squared whitened factor norm (profiled chi2 when P has been
    refined), without division by residual dimension. The factor loss is
    rho=c^2 log1p(chi2/c^2), weight=1/(1+chi2/c^2). Use c=inf for least
    squares. H=sum w(A-B C^+ B.T), g=sum w(ax-B C^+ ap), where C=JP.T JP.
    r_dim counts observed scalar rows (4, 5, or 6), before point elimination.
    """
    if cauchy_c <= 0 or np.isnan(cauchy_c):
        raise ValueError("cauchy_c must be positive")
    H, g = np.zeros((idx.n, idx.n)), np.zeros(idx.n)
    chi2, weight = np.empty(len(M)), np.empty(len(M))
    r_dim = 4 + _valid_depth(M.d_a).astype(int) + _valid_depth(M.d_b)
    cost = 0.0
    for a, b, ids in _chunks(M, chunk_size):
        r, jp, jx, cols = _linearize(T, alpha, K, M, np.asarray(P)[ids], a, b, ids, idx)
        if not np.isfinite(r).all():
            raise ValueError("points must be finite and in front of both cameras")
        chi = np.einsum("nr,nr->n", r, r)
        if np.isinf(cauchy_c):
            w, rho = np.ones(len(ids)), chi
        else:
            w = 1 / (1 + chi / cauchy_c**2)
            rho = cauchy_c**2 * np.log1p(chi / cauchy_c**2)
        chi2[ids], weight[ids] = chi, w
        cost += float(np.sum(rho))
        if not len(cols):
            continue
        Cinv = _point_inverse(np.einsum("nri,nrj->nij", jp, jp))
        B = np.einsum("nri,nrj->nij", jx, jp)
        BC = B @ Cinv
        local_H = (np.einsum("nri,nrj,n->ij", jx, jx, w, optimize=True)
                   - np.einsum("nik,njk,n->ij", BC, B, w, optimize=True))
        ap = np.einsum("nri,nr->ni", jp, r)
        local_g = (np.einsum("nri,nr,n->i", jx, r, w, optimize=True)
                   - np.einsum("nik,nk,n->i", BC, ap, w, optimize=True))
        H[np.ix_(cols, cols)] += local_H
        g[cols] += local_g
    return dict(H=(H + H.T) / 2, g=g, cost=cost, chi2=chi2, weight=weight, r_dim=r_dim)


def solve_step(H, g, lm_lambda):
    """Solve ``(H + lambda diag(max(diag(H), floor))) delta = -g``.

    Rank-deficient undamped systems use a minimum-norm least-squares solve;
    callers still need to choose an appropriate gauge for interpretation.
    """
    if lm_lambda < 0:
        raise ValueError("lm_lambda must be nonnegative")
    H, g = np.asarray(H, dtype=float), np.asarray(g, dtype=float)
    if not len(g):
        return g.copy()
    scale = np.maximum(np.diag(H), max(1e-12, np.max(np.diag(H)) * 1e-12))
    A = (H + H.T) / 2 + lm_lambda * np.diag(scale)
    try:
        return np.linalg.solve(A, -g)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, -g, rcond=1e-12)[0]


def apply_delta(T, alpha, idx, delta):
    """Copy dictionaries and apply camera-frame left pose/log-scale updates."""
    poses = {cam: np.asarray(pose, dtype=float).copy() for cam, pose in T.items()}
    scales = dict(alpha)
    for cam in idx.order:
        if idx.pose[cam] is not None:
            poses[cam] = se3_exp(delta[idx.pose[cam]]) @ poses[cam]
        if idx.alpha[cam] is not None:
            scales[cam] += delta[idx.alpha[cam]]
    return poses, scales


def fit_extext(T0, alpha0, K, M, idx, max_iters=30, *, cauchy_c=3.0, lm_lambda=1e-3,
               point_iters=5, cost_tol=1e-9, step_tol=1e-10, chunk_size=50_000, P0=None):
    """Variable-projection LM; reoptimise points for every global trial.

    Only cost-decreasing trials are accepted. Rejected trials leave all state
    intact and increase LM damping. Stops on relative cost change or maximum
    absolute parameter step. ``info`` includes monotone ``cost_history``,
    final undamped ``H``, final factor statistics, and a termination reason.
    The returned points belong to the returned poses/scales. No gauge priors
    are added: configure ParamIndex to remove the similarity gauge.
    """
    if point_iters < 1 or max_iters < 0 or lm_lambda < 0:
        raise ValueError("invalid iteration count or LM damping")
    T, alpha = apply_delta(T0, alpha0, idx, np.zeros(idx.n))
    P = init_points(T, K, M) if P0 is None else np.asarray(P0, dtype=float).copy()
    P = refine_points(T, alpha, K, M, P, iters=max(5, point_iters), chunk_size=chunk_size)
    state = normal_equations(T, alpha, K, M, P, idx, cauchy_c, chunk_size=chunk_size)
    history, reason, attempts = [state['cost']], 'max_iters', 0
    for _ in range(max_iters):
        accepted = False
        for _ in range(12):
            attempts += 1
            delta = solve_step(state['H'], state['g'], lm_lambda)
            if np.max(np.abs(delta), initial=0) < step_tol:
                reason = 'step_tol'
                break
            trial_T, trial_alpha = apply_delta(T, alpha, idx, delta)
            try:
                trial_P = refine_points(trial_T, trial_alpha, K, M, P, iters=point_iters, chunk_size=chunk_size)
                trial = normal_equations(trial_T, trial_alpha, K, M, trial_P, idx, cauchy_c,
                                         chunk_size=chunk_size)
            except (ValueError, np.linalg.LinAlgError):
                trial = None
            if trial is not None and trial['cost'] < state['cost']:
                relative = (state['cost'] - trial['cost']) / max(1.0, abs(state['cost']))
                T, alpha, P, state = trial_T, trial_alpha, trial_P, trial
                history.append(state['cost'])
                lm_lambda *= .3
                accepted = True
                if relative < cost_tol:
                    reason = 'cost_tol'
                break
            lm_lambda = max(1e-12, lm_lambda * 10)
        if reason in ('cost_tol', 'step_tol'):
            break
        if not accepted:
            reason = 'no_acceptable_step'
            break
    info = dict(state, cost_history=history, iterations=len(history) - 1, attempts=attempts,
                termination=reason, lm_lambda=lm_lambda)
    return T, alpha, P, info


def reduced_residual_stats(T, alpha, K, M):
    """Depth-free Sampson distance and corrected depth-lift a->b error, in px.

    Returns arrays ``epipolar_px`` and ``transfer_px``. Sampson distance is
    abs(xb.T F xa)/sqrt(||(F xa)[:2]||^2+||(F.T xb)[:2]||^2). Coincident
    centres give NaN epipolar distances; missing a-depth or a transfer behind
    camera b gives NaN transfer error. These are validation metrics only.
    """
    epi, transfer = np.full(len(M), np.nan), np.full(len(M), np.nan)
    for a, b, ids in _chunks(M, 50_000):
        R = T[b][:3, :3] @ T[a][:3, :3].T
        t = T[b][:3, 3] - R @ T[a][:3, 3]
        F = np.linalg.inv(K[b]).T @ skew(t) @ R @ np.linalg.inv(K[a])
        xa = np.column_stack((M.uv_a[ids], np.ones(len(ids))))
        xb = np.column_stack((M.uv_b[ids], np.ones(len(ids))))
        fa, fb = xa @ F.T, xb @ F
        denom = np.sqrt(np.sum(fa[:, :2]**2 + fb[:, :2]**2, axis=1))
        values = np.full(len(ids), np.nan)
        np.divide(np.abs(np.sum(xb * fa, axis=1)), denom, out=values, where=denom > 1e-15)
        epi[ids] = values
        valid = _valid_depth(M.d_a[ids])
        pc = _rays(M.uv_a[ids][valid], K[a]) * (np.exp(alpha[a]) * M.d_a[ids][valid])[:, None]
        uv, _ = project(pc @ R.T + t, K[b])
        transfer[ids[valid]] = np.linalg.norm(uv - M.uv_b[ids][valid], axis=1)
    return dict(epipolar_px=epi, transfer_px=transfer)
