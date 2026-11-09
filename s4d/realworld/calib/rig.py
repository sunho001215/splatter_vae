"""Shared ("rig") extrinsics for cameras verified to be physically fixed across episodes (calibration study 2026-11-06).

Why: on the own-rig recordings the per-episode robot-grounded solve is ambiguous - 2-11 of 13 starts reach the same
fit (within 0.5 mm) at poses 0.5-2 deg / 2-22 mm apart, ~10x the local covariance - and its result depends on the
episode's arm workspace (per-task medians of the per-episode poses of a camera that did not move differ by up to
2 deg / 29 mm). The cameras themselves did not move: the median images of every episode register to the reference
to <= 0.1 px (rotation <= 0.03 deg). One pose per camera estimated from robot frames POOLED over many episodes (both
tasks: more configuration diversity) is therefore used for every episode whose camera is verified unmoved.

Two parts:

* ``build_rig_calibration``: the unchanged ``calibrate_camera`` kernel on frames pooled across episodes (evenly
  spaced per episode), plus replicates on random halves of the episodes whose pairwise pose spread is the EMPIRICAL
  uncertainty, and a held-out-episode validation. Writes ``<name>.json`` (poses, evidence, provenance) and
  ``<name>.reference.npz`` (per camera: median grey image + dynamic mask of the reference episode).
* ``validate_episode``: per episode, (a) camera fixedness - the episode's median grey image registered to the
  reference on static textured patches (phase correlation, robust similarity fit), (b) the shared pose against this
  episode's own robot evidence (robot-depth residual, explained fraction). Only then is the shared pose accepted for
  the episode; otherwise the camera is "rejected" for that episode (geometry consumers are held).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from s4d.realworld.calib.core import (
    CalibrationConfig,
    CalibrationFrame,
    CalibrationResult,
    _frame_stats,
    explained_fraction,
    motion_coverage,
    pose_distance,
)

FORMAT = "s4d.realworld.rig_calibration/1"


@dataclass
class FixednessConfig:
    """Image-registration test of an unmoved camera. Thresholds from the 2026-11-06 measurements on 62 episodes of
    fixed cameras (patch registration of median images): |shift| median 0.006-0.05 px, max 0.10 px; in-plane rotation
    max 0.03 deg; scale max ~500 ppm (fx ~609 px: 0.1 deg of camera rotation = ~1 px; 5 mm of translation = 1-3 px for
    a scene at 1-3 m). Every temporal segment of the episode must pass (a camera bumped mid-episode is not fixed)."""

    frames: int = 40
    segments: int = 4  # the sampled frames are split into this many consecutive segments, each registered separately
    dynamic_dilate_px: int = 8
    patch: int = 48
    stride: int = 24
    search_px: int = 6  # integer SSD search radius of the block matching
    max_dynamic_fraction: float = 0.10
    min_gradient: float = 2.0  # mean |intensity gradient| (grey levels / px) of a usable patch
    max_shift_px: float = 0.25
    max_rot_deg: float = 0.03
    max_scale_ppm: float = 1000.0
    min_inlier_patches: int = 6
    min_inlier_fraction: float = 0.6


def block_shift(a: np.ndarray, b: np.ndarray, search: int) -> tuple[float, float, float] | None:
    """Translation (dx, dy) of image ``b`` relative to ``a`` by exhaustive integer search of the mean-removed sum of
    squared differences over +-``search`` px on the patch interior, refined by a parabola through the SSD minimum and
    its neighbours. Returns None when the minimum lies on the search border or is not a strict minimum. Third value:
    the SSD minimum relative to the patch energy."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    H, W = a.shape
    core = a[search:H - search, search:W - search]
    core = core - core.mean()
    ssd = np.full((2 * search + 1, 2 * search + 1), np.inf)
    for j, dy in enumerate(range(-search, search + 1)):
        for i, dx in enumerate(range(-search, search + 1)):
            win = b[search + dy:H - search + dy, search + dx:W - search + dx]
            ssd[j, i] = np.mean((core - (win - win.mean())) ** 2)
    j, i = np.unravel_index(np.argmin(ssd), ssd.shape)
    if j in (0, 2 * search) or i in (0, 2 * search):
        return None

    def sub(m1, c, p1):
        den = m1 - 2 * c + p1
        return 0.5 * (m1 - p1) / den if den > 0 else None

    sy = sub(ssd[j - 1, i], ssd[j, i], ssd[j + 1, i])
    sx = sub(ssd[j, i - 1], ssd[j, i], ssd[j, i + 1])
    if sx is None or sy is None:
        return None
    return float(i - search + sx), float(j - search + sy), float(ssd[j, i] / max(core.var(), 1e-9))


def static_images(rgb: np.ndarray, dynamic: np.ndarray, cfg: FixednessConfig) -> tuple[list, np.ndarray]:
    """([median grey image of each consecutive segment of the sampled frames], dilated dynamic mask) of an episode:
    ``rgb`` (n,H,W,3) evenly spaced frames, ``dynamic`` (m,H,W) bool per-frame masks of everything that moves."""
    from scipy.ndimage import binary_dilation

    grey = np.asarray(rgb, np.float32).mean(-1)
    parts = np.array_split(np.arange(len(grey)), max(1, min(cfg.segments, len(grey))))
    dyn = binary_dilation(np.asarray(dynamic, bool).any(0), iterations=cfg.dynamic_dilate_px)
    return [np.median(grey[p], 0).astype(np.float32) for p in parts], dyn


def static_image(rgb: np.ndarray, dynamic: np.ndarray, cfg: FixednessConfig) -> tuple[np.ndarray, np.ndarray]:
    """(median grey image of all sampled frames, dilated dynamic mask): the rig REFERENCE image."""
    from scipy.ndimage import binary_dilation

    grey = np.median(np.asarray(rgb, np.float32).mean(-1), 0).astype(np.float32)
    return grey, binary_dilation(np.asarray(dynamic, bool).any(0), iterations=cfg.dynamic_dilate_px)


def image_motion(ref_grey, ref_dyn, grey, dyn, K, cfg: FixednessConfig) -> dict:
    """Block matching on static textured patches -> robust (Huber 0.1 px IRLS) similarity about the principal point:
    shift t (px), in-plane rotation (deg), scale (ppm), consistent patches. ``fixed`` applies the thresholds; a patch
    whose SSD minimum is not interior (motion beyond the search radius) counts as inconsistent, never as zero."""
    dyn_any = np.asarray(ref_dyn, bool) | np.asarray(dyn, bool)
    H, W = ref_grey.shape
    s, st, r = cfg.patch, cfg.stride, cfg.search_px
    P, lost = [], 0
    gy, gx = np.gradient(np.asarray(ref_grey, np.float64))
    grad = np.hypot(gx, gy)
    for y in range(8, H - s - 8, st):
        for x in range(8, W - s - 8, st):
            if dyn_any[y:y + s, x:x + s].mean() > cfg.max_dynamic_fraction:
                continue
            if grad[y:y + s, x:x + s].mean() < cfg.min_gradient:
                continue
            got = block_shift(ref_grey[y:y + s, x:x + s], grey[y:y + s, x:x + s], r)
            if got is None:
                lost += 1
                continue
            P.append((x + s / 2, y + s / 2, got[0], got[1]))
    out = {"patches": len(P) + lost, "patches_out_of_search": lost}
    if len(P) < cfg.min_inlier_patches:
        return {**out, "fixed": False, "reason": f"only {len(P)} registrable static textured patches ({lost} beyond the "
                                                 f"+-{r} px search)"}
    P = np.asarray(P)
    d = P[:, :2] - np.asarray(K)[:2, 2]
    A = np.zeros((2 * len(P), 4))
    b = np.zeros(2 * len(P))
    A[0::2, 0] = 1
    A[1::2, 1] = 1
    A[0::2, 2], A[1::2, 2] = d[:, 0], d[:, 1]
    A[0::2, 3], A[1::2, 3] = -d[:, 1], d[:, 0]
    b[0::2], b[1::2] = P[:, 2], P[:, 3]
    w = np.ones(len(b))
    for _ in range(10):
        sw = np.sqrt(w)
        x, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        res = b - A @ x
        rp = np.hypot(res[0::2], res[1::2])
        w = np.repeat(np.where(rp < 0.1, 1.0, 0.1 / np.maximum(rp, 1e-9)), 2)
    inl = int((rp < 0.3).sum())
    out.update(shift_px=float(np.hypot(x[0], x[1])), shift_xy_px=[float(x[0]), float(x[1])],
               rot_deg=float(np.degrees(x[3])), scale_ppm=float(1e6 * x[2]), inlier_patches=inl,
               residual_median_px=float(np.median(rp)))
    problems = []
    if not np.isfinite(x).all():
        problems.append("registration not finite")
    if out["shift_px"] > cfg.max_shift_px:
        problems.append(f"image shift {out['shift_px']:.3f} px > {cfg.max_shift_px}")
    if abs(out["rot_deg"]) > cfg.max_rot_deg:
        problems.append(f"in-plane rotation {out['rot_deg']:.4f} deg > {cfg.max_rot_deg}")
    if abs(out["scale_ppm"]) > cfg.max_scale_ppm:
        problems.append(f"scale {out['scale_ppm']:.0f} ppm > {cfg.max_scale_ppm}")
    total = len(P) + lost
    if inl < cfg.min_inlier_patches or inl < cfg.min_inlier_fraction * total:
        problems.append(f"{inl}/{total} consistent patches")
    out["fixed"] = not problems
    if problems:
        out["reason"] = "; ".join(problems)
    return out


def episode_motion(ref_grey, ref_dyn, rgb, dynamic, K, cfg: FixednessConfig) -> dict:
    """Fixedness of a camera over a whole episode: every temporal segment registered to the reference."""
    segs, dyn = static_images(rgb, dynamic, cfg)
    per = [image_motion(ref_grey, ref_dyn, g, dyn, K, cfg) for g in segs]
    bad = [f"segment {k}: {m.get('reason')}" for k, m in enumerate(per) if not m["fixed"]]
    shifts = [m.get("shift_px") for m in per]
    worst = max(shifts) if shifts and all(x is not None and np.isfinite(x) for x in shifts) else None
    return {"fixed": not bad, "segments": per, "max_shift_px": worst, **({"reason": "; ".join(bad)} if bad else {})}


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def load_rig(path: str | Path) -> tuple[dict, dict, dict]:
    """(rig document, reference arrays {camera: {"grey", "dynamic"}}, fingerprint inputs). The document records the
    sha256 of its reference arrays; both are read ONCE, hashed from the exact bytes parsed and cross-checked, so a
    concurrent republish can never pair a fingerprint with other content. Refuses unknown formats."""
    import io

    path = Path(path)
    raw = path.read_bytes()
    doc = json.loads(raw)
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} document")
    ref_raw = path.with_suffix(".reference.npz").read_bytes()
    ref_sha = _sha256_bytes(ref_raw)
    if doc.get("reference_sha256") != ref_sha:
        raise ValueError(f"{path}: reference arrays do not match the document (sha256 {ref_sha[:12]} != "
                         f"{str(doc.get('reference_sha256'))[:12]}); republish both")
    ref = {}
    with np.load(io.BytesIO(ref_raw)) as z:
        for cam in doc["cameras"]:
            key = cam.replace("/", "__")
            ref[cam] = {"grey": z[f"{key}__grey"], "dynamic": z[f"{key}__dynamic"]}
    return doc, ref, {"rig_calibration": _sha256_bytes(raw), "rig_reference": ref_sha}


def publish_rig(out_path: str | Path, doc: dict, ref_arrays: dict) -> dict:
    """Write the reference arrays, then the document that names their sha256 (the document is the commit point)."""
    import io

    from s4d.realworld import io as rio

    out_path = Path(out_path)
    buf = io.BytesIO()
    np.savez_compressed(buf, **ref_arrays)
    data = buf.getvalue()
    doc = {**doc, "reference_sha256": _sha256_bytes(data)}
    text = json.dumps(doc, indent=1, allow_nan=False)  # refuse non-finite values BEFORE replacing either file
    rio.atomic_write_bytes(out_path.with_suffix(".reference.npz"), data)
    rio.atomic_write_bytes(out_path, text.encode())
    return doc


def validate_episode(camera_id: str, frames: list[CalibrationFrame], K: np.ndarray, cfg: CalibrationConfig,
                     rig_camera: dict, motion: dict, *, pooled_frames=()) -> CalibrationResult:
    """The shared pose of ``camera_id`` judged on THIS episode: fixedness (``motion`` from ``episode_motion``) and the
    episode's own robot evidence. ``pooled_frames``: this episode's frame indices that entered the rig solve; they are
    reported as fit frames and excluded from the held-out statistics and gates (every other frame is independent of
    the rig solve)."""
    if rig_camera.get("T_cam_base") is None or not np.isfinite(np.asarray(rig_camera["T_cam_base"], float)).all():
        return CalibrationResult(camera_id, "rejected", [f"rig calibration of {camera_id} has no pose "
                                                         f"(status {rig_camera.get('status')!r})"], None, None,
                                 "rig_shared", {"method": "rig_shared", "fixedness": motion,
                                                "rig": rig_camera.get("provenance")})
    T = np.asarray(rig_camera["T_cam_base"], float)
    reasons = []
    pooled = {int(i) for i in pooled_frames}
    fit_frames = [f for f in frames if int(f.t_index) in pooled]
    held = [f for f in frames if int(f.t_index) not in pooled]
    rows = _frame_stats(T, held, K, cfg)
    visible = [f for r, f in zip(rows, held) if r["robot_in_view_fraction"] >= cfg.min_visible_fraction
               and f.depth is not None]
    vis_ids = {id(f) for f in visible}
    vals = [r["depth_abs_median_m"] for r, f in zip(rows, held) if r.get("depth_abs_median_m") is not None
            and id(f) in vis_ids]
    vals = [v for v in vals if np.isfinite(v)]
    depth_med = float(np.median(vals)) if vals else None
    ef = explained_fraction(T, visible, K, cfg) if visible else None
    fit_rows = _frame_stats(T, fit_frames, K, cfg) if fit_frames else []
    fit_vals = [r["depth_abs_median_m"] for r in fit_rows if r.get("depth_abs_median_m") is not None]
    spread = rig_camera.get("empirical_spread") or {}
    diag = {"config": asdict(cfg), "camera_id": camera_id, "frames": len(frames), "visible_frames": len(visible),
            "method": "rig_shared", "init_method": "rig_shared", "rig": rig_camera.get("provenance"),
            "episode_in_rig_pool": bool(pooled), "fixedness": motion, "motion_coverage": motion_coverage(visible),
            "per_frame_heldout": rows, "per_frame_fit": fit_rows,
            "depth_abs_median_m": float(np.median(fit_vals)) if fit_vals else depth_med,
            "heldout_depth_abs_median_m": depth_med,
            "explained_fraction_fit": explained_fraction(T, fit_frames, K, cfg) if fit_frames else ef,
            "explained_fraction_heldout": ef,
            # uncertainty = the rig's empirical spread (episode-half replicates, configuration groups), not a covariance
            "std_rot_deg": spread.get("max_pair_rot_deg"), "std_trans_m": spread.get("max_pair_centre_m"),
            "uncertainty": "empirical: max pairwise spread of the pooled solve over episode-half replicates and "
                           "configuration groups",
            "empirical_spread": spread, "rig_validation": rig_camera.get("validation")}
    if rig_camera.get("status") != "validated":
        reasons.append(f"rig calibration of {camera_id} is {rig_camera.get('status')!r}, not validated")
    if not motion.get("fixed"):
        reasons.append(f"camera not verified fixed against the rig reference: {motion.get('reason')}")
    if len(visible) < cfg.min_visible_frames:
        reasons.append(f"insufficient robot visibility on frames outside the rig pool: {len(visible)} frames < "
                       f"{cfg.min_visible_frames}")
    if depth_med is None or depth_med > cfg.max_heldout_depth_median_m:
        reasons.append(f"robot-depth residual median {depth_med} m > {cfg.max_heldout_depth_median_m}")
    if ef is None or not np.isfinite(ef) or ef < cfg.min_explained_fraction:
        reasons.append(f"measured robot depth explained by the posed robot: {ef} < {cfg.min_explained_fraction}")
    return CalibrationResult(camera_id, "rejected" if reasons else "accepted", reasons, T, T, "rig_shared", diag)


# ------------------------------------------------------------------ release documents (calibration release 2026-11-08)
# A rig document with ``fixedness_rule == FIXEDNESS_RULE_G5AB`` (published by ``rig_loma_publish``) is judged per episode
# by the rule of its verdict (rig_dw DESIGN v1.1, G5 amendment) instead of the strict segment test: (G5a) the whole-
# episode-median registration to the rig reference detects motion only when it is VERIFIABLE (>= min_inlier_patches and
# >= min_inlier_fraction consistent patches); (G5b) the per-episode epipolar consistency of the rig solve, stored per
# camera and episode in ``fixedness_evidence``, must not flag motion; an episode without stored evidence is rejected.
# Cameras held ONLY by G5a at publication (``held_by == ["G5a"]``) stay held - and every episode is rejected on the
# registration-fixedness check - unless the document carries a valid ``fixedness_exception`` naming the camera.
FIXEDNESS_RULE_G5AB = "g5ab/1"
_SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$")


def registration_class(motion: dict, cfg: FixednessConfig) -> dict:
    """G5a reading of an ``episode_motion`` result: per segment VERIFIABLE iff a registration exists with >=
    ``min_inlier_patches`` and >= ``min_inlier_fraction`` consistent patches; MOVED iff verifiable and not fixed."""
    segs = motion.get("segments") or []
    verifiable, moved = [], []
    for k, m in enumerate(segs):
        tot, inl = m.get("patches", 0), m.get("inlier_patches", 0)
        ok = "shift_px" in m and inl >= cfg.min_inlier_patches and inl >= cfg.min_inlier_fraction * max(1, tot)
        verifiable.append(bool(ok))
        if ok and not m.get("fixed", False):
            moved.append(f"segment {k}: {m.get('reason')}")
    return {"verifiable": bool(segs) and all(verifiable), "verifiable_segments": int(sum(verifiable)),
            "segments": len(segs), "moved": bool(moved), "reason": "; ".join(moved) or None,
            "max_shift_px": motion.get("max_shift_px")}


def fixedness_exception_problem(doc: dict, camera_id: str) -> str | None:
    """Why the document's ``fixedness_exception`` does NOT cover ``camera_id`` (None: it does). Valid only when the
    camera is eligible (named at publication), the decision is "accepted", the camera is listed, every evidence entry
    names a file and its sha256, and the decision is attributed (decided_by, decided_at, reference)."""
    ex = doc.get("fixedness_exception") or {}
    if camera_id not in (ex.get("eligible_cameras") or []):
        return "camera not eligible for a fixedness exception"
    if ex.get("decision") != "accepted":
        return f"exception decision {ex.get('decision')!r} (empty until the coordinator fills it from FIXREG evidence)"
    if camera_id not in (ex.get("cameras") or []):
        return "camera not listed in the exception"
    evidence = ex.get("evidence") or []
    if not evidence or not all(isinstance(e, dict) and e.get("path") and _SHA256.match(str(e.get("sha256", "")))
                               for e in evidence):
        return "exception evidence missing or without sha256"
    missing = [k for k in ("decided_by", "decided_at", "reference") if not ex.get(k)]
    if missing:
        return f"exception provenance missing {missing}"
    return None


def rig_camera_status(doc: dict, camera_id: str) -> tuple[str, list, dict | None]:
    """(effective status, reasons, exception summary or None) of a release document's camera: "validated" when
    published validated, or when held only by G5a and covered by a valid fixedness exception; else "held"."""
    entry = doc["cameras"][camera_id]
    if entry.get("status") == "validated":
        return "validated", [], None
    if entry.get("status") == "held" and list(entry.get("held_by") or []) == ["G5a"]:
        why = fixedness_exception_problem(doc, camera_id)
        if why is None:
            ex = doc["fixedness_exception"]
            return "validated", [], {k: ex.get(k) for k in ("decision", "cameras", "evidence", "decided_by",
                                                              "decided_at", "reference")}
        return "held", [f"held by the registration-fixedness check G5a ({'; '.join(entry.get('reasons') or [])}); "
                        f"no valid fixedness exception: {why}"], None
    return str(entry.get("status") or "held"), list(entry.get("reasons") or []), None


def release_fixedness(doc: dict, camera_id: str, episode_id: str, motion: dict, cfg: FixednessConfig) -> dict:
    """Per-episode fixedness of a camera under ``FIXEDNESS_RULE_G5AB`` (see above); the registration metrics are always
    reported (``registration``, the ``episode_motion`` segments), also when the camera is rejected."""
    status, why, exception = rig_camera_status(doc, camera_id)
    reg = registration_class(motion, cfg)
    ev = ((doc.get("fixedness_evidence") or {}).get(camera_id) or {}).get(episode_id)
    problems, notes = [], []
    if status != "validated" and list(doc["cameras"][camera_id].get("held_by") or []) == ["G5a"]:
        problems.append(why[0] if why else "camera held by G5a")
    if reg["moved"]:
        if exception is not None:
            notes.append(f"G5a registration detects motion ({reg['reason']}); covered by the fixedness exception")
        else:
            problems.append(f"G5a: registration detects motion ({reg['reason']})")
    if ev is None:
        problems.append("episode not covered by the rig document's per-episode fixedness evidence (G5b)")
    elif ev.get("epipolar") == "moved":
        problems.append("G5b: the per-episode epipolar consistency of the rig solve flags motion")
    return {**motion, "fixed": not problems, "strict_segment_test_fixed": bool(motion.get("fixed")),
            "rule": FIXEDNESS_RULE_G5AB, "registration": reg, "evidence": ev, "exception": exception, "notes": notes,
            **({"reason": "; ".join(problems)} if problems else {"reason": None})}


def replicate_spread(T_full: np.ndarray, replicates: list[np.ndarray]) -> dict:
    """Max pairwise rotation / camera-centre distance among the full pooled solve and its replicates."""
    Ts = [np.asarray(T_full, float)] + [np.asarray(T, float) for T in replicates]
    if not all(np.isfinite(T).all() for T in Ts):
        return {"solves": len(Ts), "max_pair_rot_deg": float("inf"), "max_pair_centre_m": float("inf")}
    d = [pose_distance(a, b) for i, a in enumerate(Ts) for b in Ts[i + 1:]]
    return {"solves": len(Ts), "max_pair_rot_deg": max((x[0] for x in d), default=0.0),
            "max_pair_centre_m": max((x[1] for x in d), default=0.0)}


@dataclass
class RigConfig:
    """Pooled rig solve + EMPIRICAL acceptance. Tolerances from the training data: 128x128 crops of the 480x480 centre
    square (1 training px = 3.75 px at 640x480, fx ~609 px) of a scene at ~1.3 m: 1 training px = 0.35 deg of camera
    rotation or ~8 mm of camera translation; lifted metric points move by ~23 mm per degree at 1.3 m."""

    frames_total: int = 200  # pooled frames per camera solve, spread evenly over the pool episodes
    replicates: int = 2  # solves on random halves of the pool episodes
    seed: int = 0
    holdout_every: int = 2  # every holdout_every-th episode (recording order) validates instead of fitting
    max_spread_rot_deg: float = 0.35
    max_spread_centre_m: float = 0.008
    max_holdout_depth_median_m: float = 0.012
    min_holdout_explained: float = 0.70


def _pooled_arrays(parts: list[dict]) -> dict:
    keys = [k for k in parts[0] if k not in ("K", "overlay_frames", "overlay_images")]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["K"] = parts[0]["K"]
    for p in parts[1:]:
        if not np.allclose(p["K"], out["K"]):
            raise ValueError("pooled episodes disagree on the camera intrinsics")
    return out


def build_rig_calibration(episodes: list, out_path: str | Path, *, name: str, cfg: CalibrationConfig,
                          rig_cfg: RigConfig | None = None, groups: dict | None = None, every: int = 2,
                          max_frames: int | None = 200, depth_stage: str = "sensor_depth",
                          robot_mask_stage: str = "sam3_masks", runner=None, reference: int = 0,
                          fixedness: FixednessConfig | None = None, provenance: dict | None = None) -> dict:
    """Pooled robot-grounded calibration of cameras over ``episodes`` [(root, episode_id), recording order; ids
    unique]. Per camera: every episode is first registered to the reference episode (``episode_motion``); only
    verified-unmoved episodes are used. Then the ``calibrate_camera`` kernel on frames pooled from the POOL episodes,
    on random halves of them (empirical spread) and per configuration group (``groups``: episode_id -> label, e.g. the
    task, whose arm workspaces differ: the configuration-dependent bias test), and the validation of the pooled pose on
    every HELD-OUT episode's own frames. A camera is "validated" only if every solve is accepted and finite and every
    gate passes, else "held" (consumers must not use it). Publishes ``out_path`` + ``out_path.reference.npz``."""
    import h5py

    from s4d.realworld import io as rio
    from s4d.realworld.pipeline import RobotGeometry, camera_views, load_meta, read_frames, to_model_rgb
    from s4d.realworld.runners import LocalRunner
    from s4d.realworld.stages.calibrate import camera_arrays

    rig_cfg = rig_cfg or RigConfig()
    fixedness = fixedness or FixednessConfig()
    out_path = Path(out_path)
    episodes = [(Path(r), e) for r, e in episodes]
    if len({e for _, e in episodes}) != len(episodes):
        raise ValueError("episode ids must be unique across roots")
    if rig_cfg.holdout_every < 2:
        raise ValueError("holdout_every must be >= 2: a rig calibration is validated on held-out episodes")
    held_all = [x for i, x in enumerate(episodes) if i % rig_cfg.holdout_every == 1]
    pool_all = [x for x in episodes if x not in held_all]
    runner = runner or LocalRunner(out_path.parent / f".{out_path.stem}.work")
    metas = {e: load_meta(r, e) for r, e in episodes}
    geoms = {e: RobotGeometry(metas[e]) for _, e in episodes}
    depth_recs = {e: rio.read_record(rio.stage_dir(r, e, depth_stage)) for r, e in episodes}
    r0, e0 = episodes[reference]
    cams = [v.stream_id for v in camera_views(metas[e0])]

    def stage_idx(e, cam):
        n = len(metas[e].stream_timestamps_ns[cam])
        idx = list(range(0, n, every))
        if max_frames is not None and len(idx) > max_frames:
            idx = [idx[k] for k in np.linspace(0, len(idx) - 1, int(max_frames)).round().astype(int)]
        return idx

    def arrays_of(r, e, cam, idx):
        return camera_arrays(r, e, metas[e], geoms[e], cam, idx, depth_rec=depth_recs[e], depth_stage=depth_stage,
                             robot_mask_stage=robot_mask_stage, cfg=cfg)

    def appearance(r, e, cam):
        n = len(metas[e].stream_timestamps_ns[cam])
        pick = np.linspace(0, n - 1, min(fixedness.frames, n)).round().astype(int)
        rgb = to_model_rgb(read_frames(r, e, cam, pick), metas[e].stream(cam))
        with h5py.File(rio.stage_dir(r, e, robot_mask_stage) / f"{rio.safe_name(cam)}.h5", "r") as f:
            cls = json.loads(f.attrs.get("class_names", '["background", "robot", "object", "fixture"]'))
            sem = f["semantic"][()]
        return rgb, np.isin(sem, [cls.index(c) for c in ("robot", "object") if c in cls])

    def solve(cam, subset, tag, per_ep, frames_log=None):
        parts, local_cfg = [], None
        for pos, (r, e) in enumerate(subset):
            idx = stage_idx(e, cam)
            pick = [idx[k] for k in np.linspace(0, len(idx) - 1, min(per_ep, len(idx))).round().astype(int)]
            arrays, kept, _, _, has_depth, local_cfg = arrays_of(r, e, cam, pick)
            if not has_depth:
                raise ValueError(f"{e} {cam}: rig calibration needs metric depth ({depth_stage})")
            if frames_log is not None:
                frames_log[e] = [int(i) for i in kept]
            arrays["t_index"] = arrays["t_index"] + 1_000_000 * (pos + 1)  # unique across episodes (diagnostics)
            parts.append(arrays)
        out, kmeta, handle = runner.kernel("calibrate_camera", {"camera_id": cam, "cfg": asdict(local_cfg)},
                                           _pooled_arrays(parts), chunk_id=f"rig/{name}/{cam}/{tag}")
        runner.ack(handle)
        return kmeta["result"], local_cfg

    def usable(res):
        T = res.get("T_cam_base")
        return res.get("status") == "accepted" and T is not None and np.isfinite(np.asarray(T, float)).all()

    doc = {"format": FORMAT, "name": name, "reference_episode": e0, "episodes": [e for _, e in episodes],
           "rig_config": asdict(rig_cfg), "fixedness_config": asdict(fixedness), "calibration_config": asdict(cfg),
           "provenance": provenance or {}, "pool_frames": {}, "cameras": {}}
    ref_arrays = {}
    rng = np.random.default_rng(rig_cfg.seed)
    for cam in cams:
        rgb, dynamic = appearance(r0, e0, cam)
        ref_grey, ref_dyn = static_image(rgb, dynamic, fixedness)
        key = cam.replace("/", "__")
        ref_arrays[f"{key}__grey"], ref_arrays[f"{key}__dynamic"] = ref_grey, ref_dyn
        K = metas[e0].stream(cam).intrinsics.K()
        motion = {}
        for r, e in episodes:  # the reference too: its temporal segments against its own median
            motion[e] = episode_motion(ref_grey, ref_dyn, rgb, dynamic, K, fixedness) if e == e0 else \
                episode_motion(ref_grey, ref_dyn, *appearance(r, e, cam), K, fixedness)
        pool, held = pool_all, held_all
        reasons = []
        unfixed = sorted(e for e, m in motion.items() if not m["fixed"])
        if unfixed:  # the premise (one fixed camera for every listed episode) is false: hold, never filter silently
            reasons.append(f"{len(unfixed)} episodes not verified fixed against the reference: {unfixed[:5]}")
        entry = {"status": None, "T_cam_base": None, "pool_episodes": [e for _, e in pool],
                 "held_out_episodes": [e for _, e in held], "not_verified_fixed": unfixed,
                 "fixedness": {e: {"fixed": m["fixed"], "max_shift_px": m.get("max_shift_px"), "reason": m.get("reason")}
                               for e, m in motion.items()}}
        if len(pool) < 4 or not held:  # >= 4 pool episodes: episode-half replicates are proper subsets
            reasons.append(f"too few episodes: {len(pool)} pool (need >= 4), {len(held)} held out (need >= 1)")
            entry.update(status="held", reasons=reasons, provenance={"rig": name})
            doc["cameras"][cam] = entry
            continue
        per_ep = max(4, rig_cfg.frames_total // len(pool))
        frames_log = {}
        full, local_cfg = solve(cam, pool, "all", per_ep, frames_log)
        for e, ids in frames_log.items():
            doc["pool_frames"].setdefault(e, {})[cam] = ids
        if not usable(full):
            reasons.append(f"pooled solve {full['status']}: {full['reasons']}")
        T = np.asarray(full["T_cam_base"], float) if full.get("T_cam_base") is not None else None
        entry["T_cam_base"] = None if T is None else T.tolist()
        entry["pooled_solve"] = {k: full["diagnostics"].get(k) for k in (
            "frames", "depth_abs_median_m", "heldout_depth_abs_median_m", "explained_fraction_heldout", "std_rot_deg",
            "std_trans_m", "empirical_spread", "motion_coverage", "condition_number")}
        halves = [sorted(rng.choice(len(pool), max(2, len(pool) // 2), replace=False).tolist())
                  for _ in range(rig_cfg.replicates)]
        reps = [solve(cam, [pool[k] for k in h], f"half{i}", per_ep)[0] for i, h in enumerate(halves)]
        labels = sorted({groups[e] for _, e in pool}) if groups else []
        split = {g: solve(cam, [x for x in pool if groups[x[1]] == g], f"group-{g}", per_ep)[0] for g in labels} \
            if len(labels) > 1 else {}
        if T is not None and np.isfinite(T).all():
            bad_reps = [i for i, x in enumerate(reps) if not usable(x)]
            if bad_reps or len(reps) < 1:
                reasons.append(f"episode-half replicate solves not accepted: {bad_reps or 'none run'}")
            spread = replicate_spread(T, [np.asarray(x["T_cam_base"], float) for x in reps if usable(x)])
            spread["near_best_of_full_solve"] = full["diagnostics"].get("empirical_spread")
            entry["empirical_spread"] = spread
            checks = [("episode-half replicates", spread)]
            if groups is not None:
                if len(labels) < 2:
                    reasons.append(f"configuration groups: {labels} - need >= 2 groups among the pool episodes")
                bad_groups = [g for g, x in split.items() if not usable(x)]
                if bad_groups:
                    reasons.append(f"configuration group solves not accepted: {bad_groups}")
                sp = replicate_spread(T, [np.asarray(x["T_cam_base"], float) for x in split.values() if usable(x)])
                entry["configuration_split"] = {"groups": {g: {"T_cam_base": x.get("T_cam_base"), "status": x["status"]}
                                                           for g, x in split.items()}, **sp}
                checks.append((f"configuration groups {labels}", sp))
            for label, sp in checks:
                if not sp["max_pair_rot_deg"] <= rig_cfg.max_spread_rot_deg:
                    reasons.append(f"{label} differ by {sp['max_pair_rot_deg']:.3f} deg > {rig_cfg.max_spread_rot_deg}")
                if not sp["max_pair_centre_m"] <= rig_cfg.max_spread_centre_m:
                    reasons.append(f"{label} differ by {sp['max_pair_centre_m'] * 1000:.1f} mm > "
                                   f"{rig_cfg.max_spread_centre_m * 1000:.1f}")
            vcfg = CalibrationConfig(**{**asdict(local_cfg),
                                        "max_heldout_depth_median_m": rig_cfg.max_holdout_depth_median_m,
                                        "min_explained_fraction": rig_cfg.min_holdout_explained})
            val = []
            for r, e in held:
                arrays, *_ = arrays_of(r, e, cam, stage_idx(e, cam))
                frames = [CalibrationFrame(int(arrays["t_index"][k]), int(arrays["timestamp_ns"][k]),
                                           arrays["landmarks_base"][k], arrays["landmarks_uv"][k],
                                           arrays["landmarks_conf"][k], arrays["surface_base"][k],
                                           arrays["flange_base"][k], arrays["depth"][k],
                                           arrays["robot_mask"][k] if "robot_mask" in arrays else None)
                          for k in range(len(arrays["t_index"]))]
                res = validate_episode(cam, frames, np.asarray(arrays["K"], float), vcfg,
                                       {"T_cam_base": T.tolist(), "status": "validated"}, motion[e])
                val.append({"episode": e, "status": res.status, "reasons": res.reasons,
                            "depth_abs_median_m": res.diagnostics["heldout_depth_abs_median_m"],
                            "explained_fraction": res.diagnostics["explained_fraction_heldout"]})
            failed = [v["episode"] for v in val if v["status"] != "accepted"]
            dm = [v["depth_abs_median_m"] for v in val if v["depth_abs_median_m"] is not None]
            entry["validation"] = {"held_out": val, "failed": failed,
                                   "depth_abs_median_m_median": float(np.median(dm)) if dm else None}
            if failed:
                reasons.append(f"{len(failed)}/{len(val)} held-out episodes reject the pooled pose: {failed[:5]}")
        else:
            reasons.append("pooled solve produced no finite pose")
        entry["status"] = "held" if reasons else "validated"
        entry["reasons"] = reasons
        entry["provenance"] = {"rig": name, "pool_episodes": len(pool), "held_out_episodes": len(held)}
        doc["cameras"][cam] = entry
    return publish_rig(out_path, doc, ref_arrays)


