"""Calibration release criteria v2 (CALSYNC 2026-11-12, s4d/realworld/calib/criteria.py): pairwise-nearest RoMa
pairing + supplement of the frozen matches, G1 stability radius + image-space check, G5b completeness / coverage,
G5a materiality, the static re-evaluation (bitwise release/1 reproduction) and the publisher's --criteria v2.
Synthetic fixtures only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_realworld_calibration_release import CAMS, _fake_results, _frozen, _publish_args

REPO = Path(__file__).resolve().parents[1]
PERIOD = 66_715_000  # ns, measured 15 Hz period of the D435 colour streams


def _worker():
    spec = importlib.util.spec_from_file_location("romav2_match_worker",
                                                  REPO / "s4d/realworld/workers/romav2_match_worker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stamps(offsets_ms, n=150, drop=None):
    out = {}
    for c, off in zip(CAMS, offsets_ms):
        t = (np.arange(n) * PERIOD + int(off * 1e6) + 10**12).astype(np.int64)
        if drop and c in drop:
            t = np.delete(t, drop[c])
        out[c] = t
    return out


def _frozen_loop(stamps, cameras, times, max_dt_ms=20.0):
    """The release/1 worker loop's pair selection, verbatim (before CALSYNC)."""
    ref = cameras[0]
    n0 = len(stamps[ref])
    t_ref = stamps[ref][np.linspace(0, n0 - 1, times + 2).round().astype(int)[1:-1]]
    pairs = [(i, j) for i in range(len(cameras)) for j in range(i + 1, len(cameras))]
    frames_used, rows = {}, []
    for k, t in enumerate(t_ref):
        idx = {c: int(np.argmin(np.abs(stamps[c] - t))) for c in cameras}
        frames_used[int(k)] = {c: idx[c] for c in cameras}
        for (pa, pb) in pairs:
            ca, cb = cameras[pa], cameras[pb]
            dt = int(stamps[cb][idx[cb]] - stamps[ca][idx[ca]])
            if abs(dt) > max_dt_ms * 1e6:
                continue
            rows.append((int(k), pa, pb, idx[ca], idx[cb], dt))
    return frames_used, rows


@pytest.mark.parametrize("offsets", [(0, 6.5, -12.5), (0, -28.0, 30.0), (0, 27.0, 31.0), (0, -33.0, 33.0)])
def test_reference_pairing_is_the_frozen_rule_bitwise(offsets):
    w = _worker()
    st = _stamps(offsets)
    fu, rows = _frozen_loop(st, list(CAMS), 6)
    plan = w.plan_pairs(st, list(CAMS), 6, "reference", 20.0)
    assert plan["frames_used"] == fu and plan["pairs"] == rows


def test_pairwise_pairing_uses_each_pairs_nearest_frames_and_records_dt():
    w = _worker()
    st = _stamps((0, -28.0, 30.0))  # pick-and-place 000009: 073201 -28 ms, 076054 +30 ms vs 070988
    ref = w.plan_pairs(st, list(CAMS), 6, "reference", 20.0)
    assert ref["pairs"] == [] and len(ref["skipped"]) == 18  # the frozen rule: no pair at all
    pw = w.plan_pairs(st, list(CAMS), 6, "pairwise")
    assert len(pw["pairs"]) == 18 and not pw["skipped"]
    dts = {(pa, pb): abs(dt) / 1e6 for _, pa, pb, _, _, dt in pw["pairs"]}
    assert dts[(0, 1)] == pytest.approx(28.0, abs=0.01) and dts[(0, 2)] == pytest.approx(30.0, abs=0.01)
    assert dts[(1, 2)] == pytest.approx(66.715 - 58.0, abs=0.01)  # the pair's own nearest frames, not 58 ms apart
    assert all(abs(dt) <= PERIOD / 2 for *_, dt in pw["pairs"])
    assert set(pw["max_dt_ns"].values()) == {PERIOD // 2}


def test_pairwise_skips_a_missing_frame_instead_of_pairing_across_a_gap():
    w = _worker()
    full = w.plan_pairs(_stamps((0, 10.0, 20.0)), list(CAMS), 6, "pairwise")
    k, pa, pb, ia, ib, _ = next(r for r in full["pairs"] if r[1:3] == (0, 2) and r[0] == 2)
    st = _stamps((0, 10.0, 20.0), drop={CAMS[2]: [ib - 1, ib, ib + 1]})  # three frames missing around that time
    pw = w.plan_pairs(st, list(CAMS), 6, "pairwise")
    skipped = [r for r in pw["skipped"] if r[0] == 2 and r[1:3] == (0, 2)]
    assert len(skipped) == 1 and abs(skipped[0][5]) > PERIOD / 2 and "missing" in skipped[0][6]


def test_supplement_plan_adds_exactly_the_missing_pairs_and_checks_the_existing_frames():
    w = _worker()
    st = _stamps((0, 12.0, 31.0))  # 0-1 (12 ms) and 1-2 (19 ms) matched by the frozen rule, 0-2 (31 ms) skipped
    fu, rows = _frozen_loop(st, list(CAMS), 6)
    meta = {"cameras": list(CAMS), "times": 6, "max_dt_ms": 20.0, "frames": {str(k): v for k, v in fu.items()}}
    sup = w.supplement_plan(st, list(CAMS), 6, meta)
    pw = w.plan_pairs(st, list(CAMS), 6, "pairwise")
    union = sorted([tuple(r) for r in rows] + [tuple(r) for r in sup["pairs"]])
    assert union == sorted(tuple(r) for r in pw["pairs"]) and sup["existing_pairs"] == len(rows)
    assert {r[1:3] for r in sup["pairs"]} == {(0, 2)}
    bad = json.loads(json.dumps(meta))
    bad["frames"]["0"][CAMS[1]] += 1
    with pytest.raises(ValueError, match="frames differ"):
        w.supplement_plan(st, list(CAMS), 6, bad)
    with pytest.raises(ValueError, match="not a reference-pairing"):
        w.supplement_plan(st, list(CAMS), 6, {**meta, "pairing": "pairwise"})
    with pytest.raises(ValueError, match="outside the pairwise plan"):  # a supplement tolerance below the existing one
        w.supplement_plan(st, list(CAMS), 6, meta, max_dt_ms=15.0)
    assert w.pair_seed(0, 3, 1, 2) != w.pair_seed(0, 3, 0, 2) and w.pair_seed(0, 3, 1, 2) == w.pair_seed(0, 3, 1, 2)


# ------------------------------------------------------------------------------------------------------------- G1
def _K():
    return np.array([[609.6, 0, 324.1], [0, 608.4, 246.3], [0, 0, 1.0]])


def _look_at(centre, target=(0.32, -0.03, 0.35), up=(0, 0, 1)):
    """base -> camera pose of a camera at ``centre`` looking at ``target`` (OpenCV axes)."""
    c, t = np.asarray(centre, float), np.asarray(target, float)
    z = (t - c) / np.linalg.norm(t - c)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ c
    return T


def _stability(R_all, moves):
    """R dict of solves: 'all' and 'full' = the published pose; the listed solves = moved copies; others = published."""
    from s4d.realworld.calib.dw.publish import EXPECTED

    def st(T):
        return {"state": {"T_cam_base": {c: np.asarray(T).tolist() for c in CAMS}}}

    R = {"all": st(R_all), "full": st(R_all)}
    for n in sum(EXPECTED.values(), []):
        R[n] = st(moves.get(n, R_all))
        if n.startswith("start-"):
            R[n]["cost_on_ref_support"] = {"solution": 1.0, "reference": 1.0}
    return R


CENTRE = np.array([0.32 + 0.75, -0.03 - 0.55, 0.35 + 0.45])  # ~1.03 m from the look-at point, like the rig


def test_g1_scene_box_is_the_measured_3d_mvp_bounds():
    from s4d.realworld.calib.criteria import G1_V2, SCENE_BOX, probe_points
    from s4d.rwlearn.configs import MVP3D_BOUNDS

    assert list(SCENE_BOX) == list(MVP3D_BOUNDS) and G1_V2["image_p95_px"] == 2 * G1_V2["image_p50_px"] == 7.5
    X = probe_points(_look_at(CENTRE), _K())
    assert len(X) >= G1_V2["min_probe_points"]


def test_g1_radius_measures_the_published_estimate_and_the_image_effect():
    from s4d.realworld.calib.criteria import g1_radius
    from s4d.realworld.calib.dw.geom import so3_exp

    T0 = _look_at(CENTRE)
    # pivot-compensated motion (the camera stays aimed at the workspace): two solves 4.4 mm either side of the
    # published pose -> release/1 diameter 8.8 mm > 8 mm, radius 4.4 mm, small image effect
    side = np.cross(CENTRE - [0.32, -0.03, 0.35], [0, 0, 1])
    side /= np.linalg.norm(side)
    R = _stability(T0, {"group-0": _look_at(CENTRE + 0.0044 * side), "loco-135122073201": _look_at(CENTRE - 0.0044 * side)})
    g = g1_radius(R, CAMS[1], _K())
    assert g["diameter_release1"]["max_pair_centre_m"] == pytest.approx(0.0088, abs=1e-4)
    assert g["radius_centre_m"] == pytest.approx(0.0044, abs=1e-4) and g["reasons"] == []
    assert g["image_p50_px"] < 1.5 and g["worst"]["centre_m"] in ("group-0", "loco-135122073201")
    # uncompensated lateral translation of 7.5 mm (passes the 8 mm centre radius and release/1): fails the image p50
    T75 = T0.copy()
    T75[:3, 3] += np.array([0.0075, 0.0, 0.0])  # camera-frame x translation, rotation unchanged
    g = g1_radius(_stability(T0, {"half-2": T75}), CAMS[0], _K())
    assert g["diameter_release1"]["max_pair_centre_m"] < 0.008 and g["radius_centre_m"] < 0.008
    assert any("image displacement p50" in r for r in g["reasons"]) and g["image_p50_px"] > 3.75
    # a 0.5 deg rotation about the camera centre: fails rotation and image
    Tr = T0.copy()
    Tr[:3, :3] = so3_exp(np.radians([0.0, 0.5, 0.0])) @ T0[:3, :3]
    Tr[:3, 3] = -Tr[:3, :3] @ CENTRE
    g = g1_radius(_stability(T0, {"block-1": Tr}), CAMS[2], _K())
    assert any("rotation radius" in r for r in g["reasons"]) and any("image" in r for r in g["reasons"])
    # incomplete / non-finite evidence or no published solve never passes
    R = _stability(T0, {})
    del R["loco-135122070988"]
    assert any("incomplete" in r for r in g1_radius(R, CAMS[0], _K())["reasons"])
    R = _stability(T0, {})
    del R["all"]
    assert any("published solution" in r for r in g1_radius(R, CAMS[0], _K())["reasons"])
    Tn = T0.copy()
    Tn[0, 3] = np.nan
    assert any("not finite" in r for r in g1_radius(_stability(Tn, {}), CAMS[0], _K())["reasons"])
    assert any("not finite" in r for r in g1_radius(_stability(T0, {"half-1": Tn}), CAMS[0], _K())["reasons"])
    assert any("not finite" in r for r in g1_radius(_stability(T0, {}), CAMS[0], np.full((3, 3), np.nan))["reasons"])
    away = _look_at(CENTRE, target=(3.0, -5.0, 0.8))  # looking away from the scene: no probe
    assert any("probe" in r for r in g1_radius(_stability(away, {}), CAMS[0], _K())["reasons"])


# ------------------------------------------------------------------------------------------------------------ G5b
PAIRS = ["135122070988-135122073201", "135122070988-135122076054", "135122073201-135122076054"]


def _per_ep(eps, drop=(), bad=None, pairs=PAIRS):
    out = {}
    for e in eps:
        row = {p: {"median_px": 0.25, "se_px": 0.01, "n": 300} for p in pairs if (e, p) not in drop}
        if bad and e in bad:
            for p in pairs:
                if bad[e] in p and p in row:
                    row[p]["median_px"] = 1.4
        if row:
            out[e] = row
    return out


def test_g5b_v2_requires_complete_evidence_and_coverage_and_keeps_the_release1_rule():
    from s4d.realworld.calib.criteria import g5b_v2

    eps = [f"t-{k:06d}" for k in range(20)]
    assert all(g5b_v2(_per_ep(eps), eps, c)["reasons"] == [] for c in CAMS)
    r = g5b_v2(_per_ep(eps[2:]), eps, CAMS[0])
    assert any("incomplete" in x for x in r["reasons"]) and r["missing"] == eps[:2]
    r = g5b_v2({**_per_ep(eps), eps[3]: {}}, eps, CAMS[0])  # an empty evidence dict is no evidence
    assert r["missing"] == [eps[3]]
    # one pair of 070988 missing in 3 of 20 episodes: verifiable 85 % < 90 % -> held; 073201 unaffected
    drop = {(e, PAIRS[1]) for e in eps[:3]}
    r0, r1 = g5b_v2(_per_ep(eps, drop), eps, CAMS[0]), g5b_v2(_per_ep(eps, drop), eps, CAMS[1])
    assert r0["verifiable_fraction"] == pytest.approx(0.85) and any("verifiable" in x for x in r0["reasons"])
    assert r1["reasons"] == []
    # a pair absent from EVERY episode (the release/1 failure mode): both of its cameras unverifiable everywhere
    one = _per_ep(eps, pairs=[PAIRS[0], PAIRS[2]])
    assert g5b_v2(one, eps, CAMS[2])["verifiable_fraction"] == 0.0 and g5b_v2(one, eps, CAMS[0])["reasons"]
    assert g5b_v2(one, eps, CAMS[1])["reasons"] == []
    for field, bad in (("median_px", float("nan")), ("se_px", -0.01), ("n", 0), ("n", 29)):
        mal = _per_ep(eps)
        mal[eps[0]][PAIRS[0]][field] = bad
        assert eps[0] in g5b_v2(mal, eps, CAMS[0])["epipolar"]["not_verifiable"], (field, bad)
    r = g5b_v2(_per_ep(eps, bad={eps[5]: "135122076054"}), eps, CAMS[2])
    assert r["epipolar"]["moved"] == [eps[5]] and any("flags 1 episodes" in x for x in r["reasons"])


def _dt_rows(eps, long_pair=1, dt_long=30.0, n_long=12):
    """Pair ``long_pair`` is long-offset in the last ``n_long`` episodes (the supplement), short elsewhere."""
    longs = set(eps[-n_long:])
    return [{"episode": e, "pair": q, "file": int(q == long_pair and e in longs),
             "abs_dt_ms": dt_long if (q == long_pair and e in longs) else 8.0} for e in eps for q in (1, 2, 12)]


def test_dt_extension_is_validated_per_pair_with_tails_and_reconciliation():
    from s4d.realworld.calib.criteria import dt_extension, g5b_v2

    eps = [f"t-{k:06d}" for k in range(24)]
    rng = np.random.default_rng(3)
    pe = _per_ep(eps)
    for e in eps:
        for p in PAIRS:
            pe[e][p]["median_px"] = float(0.25 + rng.normal(0, 0.01))
    se = {"per_episode_extext": pe, "dt": {"rows": _dt_rows(eps)}}
    d = dt_extension(se)
    assert d["ok"] and d["n_long"] == 12 and d["pairs"][PAIRS[0]]["n_short"] == 12
    # four long-offset measurements degraded (0.25 -> 0.90 px): per-episode rule against the SHORT baseline
    bad = json.loads(json.dumps(pe))
    for e in eps[-4:]:
        bad[e][PAIRS[0]]["median_px"] = 0.90
    d = dt_extension({"per_episode_extext": bad, "dt": {"rows": _dt_rows(eps)}})
    assert not d["ok"] and d["pairs"][PAIRS[0]]["long_violations"] == eps[-4:]
    assert any("dt extension" in x for x in g5b_v2(bad, eps, CAMS[0], d)["reasons"])
    # a uniform 0.08 px degradation of the long-offset measurements: median check
    worse = json.loads(json.dumps(pe))
    for e in eps[-12:]:
        worse[e][PAIRS[0]]["median_px"] += 0.08
    assert not dt_extension({"per_episode_extext": worse, "dt": {"rows": _dt_rows(eps)}})["ok"]
    # Codex review 3: 9 same-pair short measurements, other pairs degraded -> no pooled fallback, unresolved = fail
    few = _dt_rows(eps, n_long=15)
    hid = json.loads(json.dumps(pe))
    for e in eps:
        for p in PAIRS:
            hid[e][p]["median_px"] = 0.90 if (p != PAIRS[0] or e in eps[-15:]) else 0.25
    d = dt_extension({"per_episode_extext": hid, "dt": {"rows": few}})
    assert not d["ok"] and "cannot be assessed" in d["problem"]
    # fail closed: no table, a table that does not reconcile, a duplicate row
    assert not dt_extension({"per_episode_extext": pe})["ok"]
    assert not dt_extension({"per_episode_extext": pe, "dt": {"rows": []}})["ok"]
    assert not dt_extension({"per_episode_extext": pe, "dt": {"rows": _dt_rows(eps)[1:]}})["ok"]
    assert not dt_extension({"per_episode_extext": pe, "dt": {"rows": _dt_rows(eps) + _dt_rows(eps)[:1]}})["ok"]
    assert dt_extension({"per_episode_extext": {}, "dt": {"rows": []}})["ok"]  # nothing measured, nothing extended
    # Codex review 4: negative |dt|, a release/1 row beyond 20 ms, a file index beyond the inputs -> fail closed
    neg = [{**r, "abs_dt_ms": -r["abs_dt_ms"]} if r["abs_dt_ms"] > 20 else r for r in _dt_rows(eps)]
    assert not dt_extension({"per_episode_extext": bad, "dt": {"rows": neg}})["ok"]
    r1 = [{**r, "file": 0} for r in _dt_rows(eps)]
    assert not dt_extension({"per_episode_extext": pe, "dt": {"rows": r1}})["ok"]
    one = {e: [{"path": "x", "sha256": "a" * 64}] for e in eps}
    assert not dt_extension({"per_episode_extext": pe, "dt": {"rows": _dt_rows(eps)}, "inputs": one})["ok"]


# ------------------------------------------------------------------------------------------------------------ G5a
def _fixreg(task, cam, eps, *, D=0.1, se=0.05, over=None, events=(), sens=True, flagged=None):
    rows = [{"episode": e, "verifiable_B": True, "D_B": D, "se_D_B": se} for e in eps]
    if over:
        for r in rows:
            if r["episode"] in over:
                r.update(over[r["episode"]])
    sensitivity = {"rows": [{"kind": "zero", "val": 0.0, "verifiable": True, "recovered_D": 0.0, "truth_D": 0.0},
                            {"kind": "shift", "val": 0.2, "verifiable": True, "recovered_D": 0.16, "truth_D": 0.2},
                            {"kind": "rot", "val": 0.03, "verifiable": True, "recovered_D": 0.18, "truth_D": 0.25}]}
    verdict = {"verdict": "INCONCLUSIVE"}
    if events is not None:
        verdict["events"] = list(events)
    return {"series": {f"x/{cam[:12]}": {"cam": cam[:12], "episodes": rows, "verdict": verdict,
                                          **({"sensitivity": sensitivity} if sens else {})}},
            "g5a_flagged": {cam[:12]: [{"episode": e} for e in (flagged or [])]}}


def test_g5a_materiality_is_decision_support_only():
    from s4d.realworld.calib.criteria import G5A_BUDGETS_PX, g5a_materiality

    cam = CAMS[1]
    eps = [f"pnp-{k:06d}" for k in range(2, 12)]
    r = g5a_materiality(_fixreg("pnp", cam, eps), cam, [eps[3]], "pnp", eps)
    assert "eligible" not in r and "decision support" in r["role"] and r["problems"] == []
    assert r["recovery"]["min_of_medians"] == pytest.approx(0.72) and r["recovery"]["min_observed"] == pytest.approx(0.72)
    assert r["episodes"][eps[0]]["upper_corr_median_px"] == pytest.approx((0.1 + 2.58 * 0.05) / 0.72)
    assert list(G5A_BUDGETS_PX) == [0.375, 0.9375] and set(r["over_budget"]) == {"0.3750", "0.9375"}
    fx = _fixreg("pnp", cam, eps, over={eps[7]: {"D_B": 0.40, "se_D_B": 0.12}})
    assert g5a_materiality(fx, cam, [eps[3]], "pnp", eps)["over_budget"]["0.9375"]["upper_corr_median_px"] == 1
    fx = _fixreg("pnp", cam, eps, over={eps[3]: {"D_B": float("nan")}})
    assert g5a_materiality(fx, cam, [eps[3]], "pnp", eps)["problems"]
    assert g5a_materiality(None, cam, [eps[3]], "pnp", eps)["problems"] == ["no FIXREG evidence"]


# ------------------------------------------------------------------------------------------------ static evaluation
def _write_matches(path, ep, pairs_times, rng, n=400, *, dt_ns=8_000_000, meta=None):
    rows = {k: [] for k in ("pair", "time", "dt_ns", "uv_a", "uv_b", "overlap", "cycle_px", "dyn_dist_a",
                            "dyn_dist_b", "fs_a", "fs_b", "sd_a", "sd_b", "fs_range_a", "fs_range_b", "magsac_inlier")}
    for pair, k in pairs_times:
        uv = rng.uniform([20, 20], [620, 460], (n, 2)).astype(np.float32)
        rows["pair"].append(np.full(n, pair, np.int16))
        rows["time"].append(np.full(n, k, np.int16))
        rows["dt_ns"].append(np.full(n, dt_ns, np.int64))
        rows["uv_a"].append(uv)
        rows["uv_b"].append((uv + rng.normal(0, 0.4, (n, 2))).astype(np.float32))
        rows["overlap"].append(np.full(n, 0.9, np.float32))
        rows["cycle_px"].append(np.full(n, 0.2, np.float32))
        for key in ("dyn_dist_a", "dyn_dist_b"):
            rows[key].append(np.full(n, 50.0, np.float32))
        for key in ("fs_a", "fs_b", "sd_a", "sd_b"):
            rows[key].append(np.full(n, 1.0, np.float32))
        for key in ("fs_range_a", "fs_range_b"):
            rows[key].append(np.full(n, 0.001, np.float32))
        rows["magsac_inlier"].append(np.ones(n, bool))
    arr = {k: np.concatenate(v) if v else np.zeros(0) for k, v in rows.items()}  # as the worker writes them
    np.savez(path, meta=json.dumps({"episode_id": ep, "cameras": list(CAMS), "max_dt_ms": 20.0, **(meta or {})}), **arr)


def _supplement(m1, m2, ep, pairs_times, rng, dt_ns=30_000_000):
    from s4d.realworld.calib.criteria import _sha256

    plan = [[k, q // 10, q % 10, 0, 0, dt_ns] for q, k in pairs_times]
    _write_matches(m2 / f"{ep}.npz", ep, pairs_times, rng, dt_ns=dt_ns,
                   meta={"pairing": "pairwise", "max_dt_ms": None, "pair_frames": plan,
                         "max_dt_ns": {"01": 33_357_500, "02": 33_357_500, "12": 33_357_500},
                         "supplement_of": {"path": str(m1 / f"{ep}.npz"), "sha256": _sha256(m1 / f"{ep}.npz")}})


def test_static_per_episode_reproduces_release1_and_adds_supplement_pairs_only(tmp_path):
    from s4d.realworld.calib.criteria import _per_episode, match_inputs
    from s4d.realworld.calib.dw.run import per_episode_extext
    from s4d.realworld.calib.loma.run import state_from_json

    rng = np.random.default_rng(0)
    m1, m2 = tmp_path / "m1", tmp_path / "m2"
    m1.mkdir(), m2.mkdir()
    eps = ["t-000001", "t-000002"]
    _write_matches(m1 / f"{eps[0]}.npz", eps[0], [(1, k) for k in range(6)], rng)
    _write_matches(m1 / f"{eps[1]}.npz", eps[1], [], rng)  # the frozen rule matched nothing (pick-and-place 000009)
    _supplement(m1, m2, eps[0], [(2, k) for k in range(6)] + [(12, k) for k in range(6)], rng)
    _supplement(m1, m2, eps[1], [(p, k) for p in (1, 2, 12) for k in range(6)], rng)
    T = {}
    for i, c in enumerate(CAMS):
        M = np.eye(4)
        M[0, 3] = 0.05 * i
        T[c] = M.tolist()
    (tmp_path / "s.json").write_text(json.dumps({"state": {"T_cam_base": T}}))
    s = state_from_json(str(tmp_path / "s.json"))
    K = {c: _K() for c in CAMS}
    ref = per_episode_extext(s, K, m1, eps, 200)
    one = _per_episode(s, K, {e: [m1 / f"{e}.npz"] for e in eps}, 200)
    assert json.dumps(one, sort_keys=True) == json.dumps(ref, sort_keys=True) and set(ref) == {eps[0]}
    files, manifest = match_inputs([m1, m2], eps)
    assert [len(v) for v in files.values()] == [2, 2] and manifest[eps[0]][1]["pairing"] == "pairwise"
    both = _per_episode(s, K, files, 200)
    assert both[eps[0]][PAIRS[0]] == ref[eps[0]][PAIRS[0]]
    assert set(both[eps[0]]) == set(both[eps[1]]) == set(PAIRS)


def test_match_inputs_refuse_foreign_duplicate_or_out_of_plan_supplements(tmp_path):
    from s4d.realworld.calib.criteria import match_inputs

    rng = np.random.default_rng(1)
    m1, m2, m3 = tmp_path / "m1", tmp_path / "m2", tmp_path / "m3"
    for d in (m1, m2, m3):
        d.mkdir()
    ep = "t-000001"
    _write_matches(m1 / f"{ep}.npz", ep, [(1, k) for k in range(6)], rng)
    _supplement(m1, m2, ep, [(2, k) for k in range(6)], rng)
    assert match_inputs([m1, m2], [ep])[1][ep][1]["combinations"] == 6
    with pytest.raises(ValueError, match="two match files"):  # the same supplement twice
        _supplement(m1, m3, ep, [(2, k) for k in range(6)], rng)
        match_inputs([m1, m2, m3], [ep])
    with pytest.raises(ValueError, match="release/1"):  # supplement given first
        match_inputs([m2, m1], [ep])
    _write_matches(m3 / f"{ep}.npz", ep, [(12, k) for k in range(6)], rng,
                   meta={"pairing": "pairwise", "supplement_of": {"sha256": "0" * 64}})
    with pytest.raises(ValueError, match="supplement_of"):
        match_inputs([m1, m3], [ep])
    _supplement(m1, m3, ep, [(12, k) for k in range(6)], rng, dt_ns=40_000_000)  # beyond half a period
    with pytest.raises(ValueError, match="tolerance"):
        match_inputs([m1, m3], [ep])
    _write_matches(m3 / f"{ep}.npz", ep, [(2, k) for k in range(6)], rng, dt_ns=25_000_000)  # release/1 file > 20 ms
    with pytest.raises(ValueError, match="max_dt_ms"):
        match_inputs([m3], [ep])


def _timestamped_episode(root, ep, st):
    d = root / ep / "episodes" / ep / "normalized"
    d.mkdir(parents=True)
    np.savez(d / "arrays.npz", **{f"timestamps/{c}": st[c] for c in CAMS})


def _write_plan(path, ep, rows, rng, meta, n=50):
    arr = {k: [] for k in ("pair", "time", "frame_a", "frame_b", "dt_ns")}
    for k, pa, pb, ia, ib, dt in rows:
        arr["pair"].append(np.full(n, pa * 10 + pb, np.int16))
        arr["time"].append(np.full(n, k, np.int16))
        arr["frame_a"].append(np.full(n, ia, np.int32))
        arr["frame_b"].append(np.full(n, ib, np.int32))
        arr["dt_ns"].append(np.full(n, dt, np.int64))
    out = {k: np.concatenate(v) if v else np.zeros(0) for k, v in arr.items()}
    np.savez(path, meta=json.dumps({"episode_id": ep, "cameras": list(CAMS), "times": 6, **meta}), **out)


def test_match_inputs_validate_frames_against_the_episode_timestamps(tmp_path):
    from s4d.realworld.calib.criteria import _sha256, match_inputs

    w = _worker()
    rng = np.random.default_rng(2)
    root, m1, m2 = tmp_path / "view", tmp_path / "m1", tmp_path / "m2"
    for d in (root, m1, m2):
        d.mkdir()
    ep = "t-000001"
    st = _stamps((0, -28.0, 30.0))  # the frozen rule matched nothing
    _timestamped_episode(root, ep, st)
    ref = w.plan_pairs(st, list(CAMS), 6, "reference", 20.0)
    fr = {str(k): v for k, v in ref["frames_used"].items()}
    _write_plan(m1 / f"{ep}.npz", ep, ref["pairs"], rng, {"max_dt_ms": 20.0, "frames": fr})
    sup = w.supplement_plan(st, list(CAMS), 6, {"cameras": list(CAMS), "times": 6, "max_dt_ms": 20.0, "frames": fr})
    smeta = {"pairing": "pairwise", "max_dt_ms": None, "max_dt_ns": sup["max_dt_ns"], "frames": fr,
             "pair_frames": [list(r) for r in sup["pairs"]],
             "supplement_of": {"path": str(m1 / f"{ep}.npz"), "sha256": _sha256(m1 / f"{ep}.npz")}}
    _write_plan(m2 / f"{ep}.npz", ep, sup["pairs"], rng, smeta)
    files, manifest = match_inputs([m1, m2], [ep], root)
    assert manifest[ep][1]["combinations"] == 18 and manifest[ep][0]["combinations"] == 0
    # wrong frames (a self-consistent plan/rows/tolerance that the timestamps do not support): refused
    bad_rows = [[k, pa, pb, ia, ib + 1, dt + 66_715_000] for k, pa, pb, ia, ib, dt in sup["pairs"]]
    _write_plan(m2 / f"{ep}.npz", ep, bad_rows, rng, {**smeta, "pair_frames": bad_rows,
                                                       "max_dt_ns": {"01": 10**9, "02": 10**9, "12": 10**9}})
    with pytest.raises(ValueError, match="pair_frames differ|planned combination"):
        match_inputs([m1, m2], [ep], root)
    _write_plan(m2 / f"{ep}.npz", ep, sup["pairs"], rng, {**smeta, "cameras": list(CAMS)[::-1]})
    with pytest.raises(ValueError, match="camera order"):
        match_inputs([m1, m2], [ep], root)
    fr_bad = json.loads(json.dumps(fr))
    fr_bad["0"][CAMS[2]] += 1
    _write_plan(m1 / f"{ep}.npz", ep, ref["pairs"], rng, {"max_dt_ms": 20.0, "frames": fr_bad})
    with pytest.raises(ValueError, match="reference rule"):
        match_inputs([m1], [ep], root)


# ------------------------------------------------------------------------------------------------------- publisher
def _se(rdir, per_ep=None):
    from s4d.realworld.calib.criteria import _sha256

    full = json.loads((rdir / "full.json").read_text())
    eps = sorted(set(full["fit_episodes"]) | set(full["heldout_episodes"]))
    v2 = json.loads(json.dumps(full["per_episode_extext"] if per_ep is None else per_ep))
    g4 = json.loads(json.dumps(full["heldout_sam"]["extext_sam_static"]))
    g4.update({p: {"n": 500} for p in PAIRS})
    for rows in v2.values():
        for r in rows.values():
            r.setdefault("n", 300)
    codes = {p: q for q, p in ((1, PAIRS[0]), (2, PAIRS[1]), (12, PAIRS[2]))}
    dt = [{"episode": e, "pair": codes[p], "file": 0, "abs_dt_ms": 8.0} for e, rows in v2.items() for p in rows]
    ev = rdir.parent / "evidence"  # real (tiny) source files: the publisher re-hashes them
    ev.mkdir(exist_ok=True)
    inputs = {}
    for e in eps:
        f = ev / f"{e}.npz"
        if not f.exists():
            np.savez(f, meta=json.dumps({"episode_id": e, "cameras": list(CAMS)}), pair=np.zeros(0))
        inputs[e] = [{"path": str(f), "sha256": _sha256(f), "pairing": "reference"}]
    for name in ("corr.npz", "ev.npz"):
        if not (ev / name).exists():
            np.savez(ev / name, x=np.zeros(1))
    return {"format": "s4d-calib-static-eval/2", "full": {"path": str(rdir / "full.json"),
                                                          "sha256": _sha256(rdir / "full.json")},
            "frames_validated": True, "inputs": inputs,
            "K_sources": {"per_episode": {"path": str(ev / "corr.npz"), "sha256": _sha256(ev / "corr.npz")},
                          "heldout": {"path": str(ev / "ev.npz"), "sha256": _sha256(ev / "ev.npz")}},
            "code": {k: "d" * 64 for k in ("criteria.py", "dw/run.py", "dw/matches.py", "dw/common.py")},
            "episodes": eps, "heldout_episodes": full["heldout_episodes"],
            "per_episode_extext": v2, "heldout_extext_sam_static": g4, "dt": {"rows": dt},
            "release1_reproduction": {"per_episode_extext": full["per_episode_extext"],
                                      "heldout_extext_sam_static": full["heldout_sam"]["extext_sam_static"]},
            "K_nominal": {c: _K().tolist() for c in CAMS}}


def _realistic_poses(rdir):
    """Give the fake batch a rig-like published pose (the scene probe needs the scene in view)."""
    for p in rdir.glob("*.json"):
        d = json.loads(p.read_text())
        if "state" in d:
            d["state"]["T_cam_base"] = {c: _look_at(CENTRE).tolist() for c in CAMS}
            p.write_text(json.dumps(d))


def test_publisher_criteria_v2_keeps_release1_and_documents_both(tmp_path):
    from s4d.realworld.calib.criteria import GATE_VERSION_V2, V2
    from s4d.realworld.calib.loma.publish import GATE_VERSION, publish
    from s4d.realworld.calib.rig import rig_camera_status

    # release/1 holds every camera (one episode without G5b evidence) -> release/2 with complete pairwise evidence
    rdir = tmp_path / "a" / "res"
    _, fixed, _ = _fake_results(rdir, sam_fail=())
    _realistic_poses(rdir)
    full = json.loads((rdir / "full.json").read_text())
    complete = json.loads(json.dumps(full["per_episode_extext"]))
    del full["per_episode_extext"]["t-000004"]
    (rdir / "full.json").write_text(json.dumps(full))
    se = _se(rdir, per_ep=complete)
    frozen = _frozen(rdir, fixed)
    assert all("G5b: per-episode consistency incomplete" in frozen["rig_dw_gates"][c]["reasons"] for c in CAMS)
    (tmp_path / "a" / "se.json").write_text(json.dumps(se))
    a = _publish_args(tmp_path / "a", rdir, fixed, frozen, tmp_path / "a" / "rig.json")
    a.criteria, a.static_eval, a.fixreg = "v2", str(tmp_path / "a" / "se.json"), None
    doc = publish(a)
    assert doc["gate_version"] == GATE_VERSION_V2 and doc["name"].endswith("-c2")
    assert doc["criteria"]["version"] == V2 and doc["criteria"]["gate_version_release1"] == GATE_VERSION
    assert all(doc["criteria"]["release1_verdict"][c]["held_by"] == ["G5b"] for c in CAMS)
    assert all(doc["cameras"][c]["status"] == "validated" and rig_camera_status(doc, c)[0] == "validated" for c in CAMS)
    assert doc["fixedness_evidence"][CAMS[0]]["t-000004"]["epipolar"] == "ok"
    assert doc["criteria"]["release1_fixedness_evidence"][CAMS[0]]["t-000004"]["epipolar"] == "not_verifiable"
    assert doc["provenance"]["static_eval"]["sha256"] and "CALSYNC" in doc["provenance"]["user_decision"]
    # static evaluation of another full.json, or one whose embedded release/1 measurements differ: refused
    b = _publish_args(tmp_path / "a", rdir, fixed, frozen, tmp_path / "a" / "rig2.json")
    b.criteria, b.fixreg = "v2", None
    for name, bad, msg in (("se2", {**se, "full": {**se["full"], "sha256": "0" * 64}}, "another full.json"),
                           ("se3", {**se, "release1_reproduction": {**se["release1_reproduction"],
                                                                    "per_episode_extext": complete}}, "reproduce"),
                           ("se5", {k: v for k, v in se.items() if k != "inputs"}, "manifest"),
                           ("se6", {**se, "frames_validated": False}, "timestamp-validated"),
                           ("se7", {k: v for k, v in se.items() if k != "K_sources"}, "K-source"),
                           ("se8", {**se, "inputs": {e: [{"sha256": "not-a-hash", "pairing": "reference"}]
                                                    for e in se["episodes"]}}, "manifest"),
                           ("se9", {**se, "heldout_episodes": []}, "held-out"),
                           ("se10", {**se, "code": {"criteria.py": "x"}}, "code identity"),
                           ("se11", {**se, "code": {"criteria.py": "d" * 64}}, "code identity"),
                           ("se12", {**se, "inputs": {e: [{**v[0], "sha256": "e" * 64}] for e, v in se["inputs"].items()}},
                            "sources not verified"),
                           ("se13", {**se, "K_sources": {**se["K_sources"], "heldout": {"path": "/nonexistent/x.npz",
                                                                                         "sha256": "f" * 64}}},
                            "sources not verified")):
        (tmp_path / "a" / f"{name}.json").write_text(json.dumps(bad))
        b.static_eval, b.out = str(tmp_path / "a" / f"{name}.json"), str(tmp_path / "a" / f"rig-{name}.json")
        with pytest.raises(SystemExit, match=msg):
            publish(b)
    # impossible G4 distances or missing support -> held by G4
    for name, mut in (("g4neg", {"line_median_px": -1.0, "transfer_ab_median_px": -1.0, "transfer_ba_median_px": -1.0}),):
        bad = json.loads(json.dumps(se))
        bad["heldout_extext_sam_static"]["all"].update(mut)
        (tmp_path / "a" / f"{name}.json").write_text(json.dumps(bad))
        b.static_eval, b.out = str(tmp_path / "a" / f"{name}.json"), str(tmp_path / "a" / f"rig-{name}.json")
        held = publish(b)["cameras"]
        assert all("G4" in held[c]["held_by"] for c in CAMS)
    nos = json.loads(json.dumps(se))
    for p in PAIRS:
        nos["heldout_extext_sam_static"].pop(p)
    (tmp_path / "a" / "g4nos.json").write_text(json.dumps(nos))
    b.static_eval, b.out = str(tmp_path / "a" / "g4nos.json"), str(tmp_path / "a" / "rig-g4nos.json")
    held = publish(b)["cameras"]
    assert all("G4" in held[c]["held_by"] for c in CAMS)
    # v2 evidence still missing an episode -> held by G5b (never passes incomplete evidence)
    (tmp_path / "a" / "se4.json").write_text(json.dumps(_se(rdir)))
    b.static_eval, b.out = str(tmp_path / "a" / "se4.json"), str(tmp_path / "a" / "rig4.json")
    doc = publish(b)
    assert all(doc["cameras"][c]["held_by"] == ["G5b"] for c in CAMS)


def test_publisher_criteria_v2_keeps_the_release1_g5a_exception_rule_with_decision_support(tmp_path):
    from s4d.realworld.calib.loma.publish import publish
    from s4d.realworld.calib.rig import rig_camera_status

    rdir = tmp_path / "b" / "res"
    _, fixed, _ = _fake_results(rdir, sam_fail=(), moved=CAMS[0])
    _realistic_poses(rdir)
    se = _se(rdir)
    (tmp_path / "b" / "se.json").write_text(json.dumps(se))
    frozen = _frozen(rdir, fixed)
    (tmp_path / "b" / "fx.json").write_text(json.dumps(_fixreg("t", CAMS[0], se["episodes"], over={
        se["episodes"][6]: {"D_B": 0.6}})))  # material-looking FIXREG numbers do not change eligibility (release/1)
    a = _publish_args(tmp_path / "b", rdir, fixed, frozen, tmp_path / "b" / "rig.json", eligible=[CAMS[0]])
    a.criteria, a.static_eval, a.fixreg = "v2", str(tmp_path / "b" / "se.json"), str(tmp_path / "b" / "fx.json")
    doc = publish(a)
    c0 = doc["cameras"][CAMS[0]]
    assert c0["held_by"] == ["G5a"] and c0["diagnostics"]["criteria_v2"]["exception_eligible"]
    ds = c0["diagnostics"]["criteria_v2"]["g5a_decision_support"]
    assert "decision support" in ds["role"] and ds["over_budget"]["0.9375"]["upper_corr_median_px"] == 1
    assert doc["fixedness_exception"]["eligible_cameras"] == [CAMS[0]] and rig_camera_status(doc, CAMS[0])[0] == "held"
    b = _publish_args(tmp_path / "b", rdir, fixed, frozen, tmp_path / "b" / "rig2.json", eligible=[CAMS[1]])
    b.criteria, b.static_eval, b.fixreg = "v2", str(tmp_path / "b" / "se.json"), None
    with pytest.raises(SystemExit, match="G5a alone"):
        publish(b)


def test_release1_default_is_unchanged(tmp_path):
    """No --criteria: the frozen release/1 publication (gate version, name, no criteria block)."""
    from s4d.realworld.calib.loma.publish import GATE_VERSION, publish

    rdir = tmp_path / "c" / "res"
    _, fixed, _ = _fake_results(rdir, sam_fail=())
    doc = publish(_publish_args(tmp_path / "c", rdir, fixed, _frozen(rdir, fixed), tmp_path / "c" / "rig.json"))
    assert doc["gate_version"] == GATE_VERSION and "criteria" not in doc and not doc["name"].endswith("-c2")
