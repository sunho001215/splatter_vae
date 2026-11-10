"""Deployability checks of trained real-world FQL policies (realworld-ops/serving/POL-CRITERIA.md, fixed in advance).

Read-only: run directories, FQL datasets, raw recordings and checkpoints are only read; nothing here trains, launches,
writes run files or talks to a robot. Torch / h5py are imported inside the functions that need them.

- ``pretraining_complete`` (a), ``fql_complete`` (b): completion record vs the run's own configured budget.
- ``export_chain`` (c): pretrain ``encoder.pt`` == FQL dataset ``export_sha256`` == policy meta ``export_sha256``.
- ``feature_conditioning`` (e1): the coordinator's frozen-feature check (``serving/tools/fql_feature_check.py``,
  same arithmetic and random stream) on the training split of the FQL dataset.
- ``probe_samples`` + ``visual_probe`` (e2): held-out recorded frames through the deploy path; action change under
  visual-input replacement vs the action spread and the FQL-noise-only change.
- ``stability`` (f): finite metrics, no Q runaway, feasible late-phase Q.
- ``classify``: rule set of POL-CRITERIA.md -> deployable / incomplete / quarantined (+ flags).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# POL-CRITERIA.md thresholds (never tuned on results)
E1_OFFSET_MAX = 10.0
E1_LN_UNIT_STD_MIN = 0.5
E2_SAMPLES = 192
E2_SPREAD_FRACTION = 0.2
E2_NOISE_FRACTION = 0.25
F2_Q_RUNAWAY = 100.0
F3_Q_FEASIBLE = 1.0
F3_LATE_FRACTION = 0.8
Q_KEYS = ("training/critic/q_mean", "validation/critic/q_mean")
ATTENUATED = "attenuated_visual_input"


def check(passed: bool | None, **detail) -> dict:
    """One criterion result: ``passed`` None = not evaluated (reason in ``detail``)."""
    return {"pass": passed, **detail}


# ------------------------------------------------------------------------------------------------ a, b
def _budget(config: dict) -> tuple[str, int] | None:
    """The run's configured pretraining budget: (kind, value) from its own config.yaml."""
    train, run = config.get("train") or {}, config.get("run") or {}
    if train.get("num_epochs") is not None and train.get("max_global_steps") is None:
        return "epochs", int(train["num_epochs"])
    for key in ("max_global_steps", "max_steps"):
        if train.get(key) is not None:
            return "steps", int(train[key])
    if run.get("total_steps") is not None:  # epoch budget resolved by the trainer (epochs x steps_per_epoch)
        return "steps", int(run["total_steps"])
    return None


def pretraining_complete(config: dict, completion: dict | None) -> dict:
    """a: completion status ``completed`` and progress reaching the configured budget (steps >= budget - 1 for the
    step-index convention; epochs exactly)."""
    if completion is None:
        return check(False, reason="no completion.json (running, failed or not synced)")
    budget = _budget(config)
    if budget is None:
        return check(False, reason="no budget key in config.yaml")
    kind, value = budget
    if completion.get("status") != "completed":
        return check(False, reason=f"completion status {completion.get('status')!r}", budget=value, kind=kind)
    done = completion.get("epochs_completed") if kind == "epochs" else completion.get("steps")
    ok = done is not None and (int(done) >= value if kind == "epochs" else int(done) >= value - 1)
    return check(ok, kind=kind, budget=value, done=done,
                 **({} if ok else {"reason": f"{kind} {done} < budget {value}"}))


def fql_complete(config: dict, completion: dict | None) -> dict:
    """b (file part): completion ``completed`` with steps == train.offline_steps."""
    budget = int((config.get("train") or {}).get("offline_steps", -1))
    if completion is None:
        return check(False, reason="no completion.json (running or not synced)", budget=budget)
    ok = completion.get("status") == "completed" and int(completion.get("steps", -1)) == budget
    return check(ok, budget=budget, steps=completion.get("steps"), status=completion.get("status"),
                 **({} if ok else {"reason": f"status {completion.get('status')!r}, steps {completion.get('steps')} "
                                             f"vs budget {budget}"}))


# ------------------------------------------------------------------------------------------------ c
def export_chain(meta: dict, dataset_attrs: dict | None, encoder_sha256: str | None) -> dict:
    """c (identity part): the policy's export is the one its FQL dataset was built with and the one its pretraining
    run exported. e2e_cnn: no export (dataset condition must match)."""
    if meta.get("condition") == "e2e_cnn":
        cond = None if dataset_attrs is None else dataset_attrs.get("condition")
        return check(cond == "e2e_cnn", export=None, dataset_condition=cond,
                     **({} if cond == "e2e_cnn" else {"reason": "dataset is not an e2e_cnn dataset"}))
    shas = {"policy_meta": meta.get("export_sha256"),
            "dataset": None if dataset_attrs is None else dataset_attrs.get("export_sha256"),
            "pretrain_encoder": encoder_sha256}
    ok = None not in shas.values() and len(set(shas.values())) == 1
    return check(ok, export_path=meta.get("export_path"), export_sha256=shas,
                 **({} if ok else {"reason": "export sha256 chain differs or a link is missing"}))


# ------------------------------------------------------------------------------------------------ e1
def feature_norm_of(payload: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """The input standardisation a checkpoint persists (meta ``feature_norm`` + the encoder buffers ``feat_mean`` /
    ``feat_std``, fitted on the training split), or None. POL-STATUS amendment A1: e1 is computed on the features as
    the FQL head sees them."""
    if not (payload.get("meta") or {}).get("feature_norm"):
        return None
    found = {}

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            elif isinstance(k, str) and k.rsplit(".", 1)[-1] in ("feat_mean", "feat_std"):
                found.setdefault(k.rsplit(".", 1)[-1], v)

    walk(payload.get("agent") or {})
    if set(found) != {"feat_mean", "feat_std"}:
        raise ValueError("checkpoint declares feature_norm but carries no feat_mean / feat_std buffers")
    return tuple(np.asarray(found[k].detach().cpu().numpy() if hasattr(found[k], "detach") else found[k],
                            np.float64).reshape(-1) for k in ("feat_mean", "feat_std"))


def feature_conditioning(dataset_path, train_episodes: list[str], norm=None) -> dict:
    """e1 = ``serving/tools/fql_feature_check.py`` (coordinator, 2026-11-09): offset_to_signal = ||mean|| /
    sqrt(sum var) over the training split, LayerNorm unit std after a random xavier projection of <= 20 rows per
    timeline. Same arithmetic and the same ``default_rng(0)`` stream, so the numbers equal that tool's evidence.
    ``norm``: (mean, std) persisted with the policy (``feature_norm_of``), applied to every row first."""
    import h5py

    rng = np.random.default_rng(0)
    n, s1, s2, sub = 0, None, None, []
    with h5py.File(dataset_path, "r") as f:
        for ep in train_episodes:
            for _, g in f["episodes"][ep].items():
                x = g["features"][:].astype(np.float64)
                if norm is not None:
                    x = (x - norm[0]) / norm[1]
                if s1 is None:
                    s1, s2 = np.zeros(x.shape[1]), np.zeros(x.shape[1])
                n += len(x)
                s1 += x.sum(0)
                s2 += (x ** 2).sum(0)
                sub.append(x[rng.choice(len(x), min(len(x), 20), replace=False)])
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0))
    sub = np.concatenate(sub)
    W = rng.uniform(-1, 1, (256, sub.shape[1])) * np.sqrt(6 / (sub.shape[1] + 256))
    H = sub @ W.T
    L = (H - H.mean(1, keepdims=True)) / H.std(1, keepdims=True)
    offset = float(np.linalg.norm(mu) / np.sqrt((sd ** 2).sum()))
    ln = float(L.std(0).mean())
    return check(offset <= E1_OFFSET_MAX and ln >= E1_LN_UNIT_STD_MIN, rows=int(n), dim=int(sub.shape[1]),
                 offset_to_signal=offset, ln_unit_std_raw=ln, offset_pass=offset <= E1_OFFSET_MAX,
                 ln_pass=ln >= E1_LN_UNIT_STD_MIN, normalized=norm is not None)


# ------------------------------------------------------------------------------------------------ e2
def raw_episode_path(raw_root, store_episode: str) -> Path:
    """``drawer-open/drawer-open-000007`` -> ``<raw_root>/drawer-open/episode_000007.h5`` (gate report naming)."""
    task, name = store_episode.split("/")
    return Path(raw_root) / task / f"episode_{name.rsplit('-', 1)[1]}.h5"


def _regular(stamps, numbers, idx, period_s: float, tolerance: float) -> bool:
    gaps = np.diff(stamps[idx]) / 1e9
    return bool(np.all(np.diff(numbers[idx]) == 1) and np.all(np.abs(gaps - period_s) <= tolerance * period_s))


def probe_samples(episodes: list[str], raw_root, cameras: list[str], *, n: int = E2_SAMPLES, seed: int = 0,
                  history: int = 3, period_s: float = 1 / 15, tolerance: float = 0.01, views: list[str] | None = None,
                  depth: bool = False, rig_document=None, alignment: str = "rs_align") -> list[dict]:
    """n held-out samples, each a complete trained history: a uniformly drawn episode and colour frame index whose
    ``history`` consecutive frames follow the trained spacing (device frame numbers consecutive, intervals within
    1 %), with the measured state bound by the training rule. Single-view (``views`` None): a uniformly drawn trained
    camera. Multi-view (``views`` = the trained camera order): every step is a mutual-nearest set within half a
    period around the reference (first) camera's frame. ``depth``: native depth aligned like the live client
    (``alignment``) with the rig's validated native correction; ``rig_document``: the task's released rig (session
    calibration). Frames are the client's preprocessed ``Frame`` objects."""
    from s4d.rwlearn.deploy.realsense import load_rig_session
    from s4d.rwlearn.deploy.replay import RecordedEpisode

    rng = np.random.default_rng(seed)
    recs, out, attempts = {}, [], 0
    while len(out) < n:
        attempts += 1
        if attempts > 20 * n:
            raise RuntimeError(f"only {len(out)} of {n} probe samples with a complete trained history")
        ep = episodes[int(rng.integers(len(episodes)))]
        if ep not in recs:
            recs[ep] = RecordedEpisode(raw_episode_path(raw_root, ep))
        rec = recs[ep]
        if views:
            cams = list(views)
            if any(c not in rec.cameras() for c in cams):
                raise ValueError(f"{ep}: trained cameras {cams} not all recorded ({rec.cameras()})")
        else:
            avail = [c for c in cameras if c in rec.cameras()]
            if not avail:
                raise ValueError(f"{ep}: none of the trained cameras {cameras} recorded")
            cams = [avail[int(rng.integers(len(avail)))]]
        ref = cams[0]
        stamps = {c: rec.timestamps_ns(c) for c in cams}
        numbers = {c: rec.frame_numbers(c) for c in cams}
        end = int(rng.integers(history - 1, len(stamps[ref])))
        idx = {ref: list(range(end - history + 1, end + 1))}
        ok = _regular(stamps[ref], numbers[ref], idx[ref], period_s, tolerance)
        for c in cams[1:]:
            if not ok:
                break
            near = [int(np.argmin(np.abs(stamps[c] - stamps[ref][i]))) for i in idx[ref]]
            back = [int(np.argmin(np.abs(stamps[ref] - stamps[c][j]))) for j in near]
            ok = back == idx[ref] and all(abs(int(stamps[c][j]) - int(stamps[ref][i])) <= period_s * 5e8
                                          for i, j in zip(idx[ref], near)) \
                and _regular(stamps[c], numbers[c], near, period_s, tolerance)
            idx[c] = near
        if not ok:
            continue
        rig = load_rig_session(rig_document, cams, {c: rec.K_native(c) for c in cams}) if rig_document else None
        corr = (rig or {}).get("native_depth_correction", {})
        frames = {c: [rec.frame(c, i, depth=depth, correction=corr.get(c), alignment=alignment) for i in idx[c]]
                  for c in cams}
        t = max(f[-1].timestamp_ns for f in frames.values())
        try:
            state = rec.state_at(t)
        except ValueError:
            continue
        if t - state["pose_time_ns"] > 15_000_000 or t - state["gripper_time_ns"] > 100_000_000:
            continue  # the training binding would not accept this observation
        out.append({"episode": ep, "cameras": cams, "index": end, "native_hw": {c: rec.native_hw(c) for c in cams},
                    "frames": frames, "session": rig, **state})
    return out


def _observations(policy, samples: list[dict]) -> list[dict]:
    """Each sample's observation through a bound session + ``HistoryWindow`` (the deploy assembly path). Multi-view
    sets get M5 evidence 0.0: the probe measures sensitivity on recorded sets, it does not certify their sync (the
    observation tensor does not depend on the M5 value)."""
    import uuid

    from s4d.rwlearn.deploy import HistoryWindow

    spec0 = policy.input_spec
    obs, sessions = [], {}
    for s in samples:
        cams = tuple(s["cameras"])
        key = (cams, tuple(tuple(s["native_hw"][c]) for c in cams))
        if key not in sessions:
            choices = {"native_hw": {c: tuple(s["native_hw"][c]) for c in cams}}
            if spec0.views == 1:
                choices["camera_id"] = cams[0]
            if spec0.needs_calibration:
                choices["calibration"] = {c: s["session"]["calibration"][c] for c in cams}
            if "depth" in spec0.modalities:
                choices["native_depth_correction"] = {c: s["session"]["native_depth_correction"][c] for c in cams}
            sessions[key] = policy.reset_session(session_id=f"probe-{uuid.uuid4().hex}", concurrent=True, **choices)
        spec = sessions[key]
        h = HistoryWindow(spec)
        for c in cams:
            for f in s["frames"][c]:
                h.push(c, f)
        end = {c: s["frames"][c][-1].seq for c in cams}
        disp = [0.0] * spec.history_length if spec.views > 1 else None
        obs.append(h.assemble(end, surface_disp_m=disp).observation())
    for spec in sessions.values():
        policy.end_session(spec.session_id)
    return obs


def visual_probe(policy, samples: list[dict], *, seed: int = 0) -> dict:
    """e2 on a loaded ``DeployPolicy`` (single-view conditions): S = mean_d std_i a_i, D_vis = mean |a(obs_pi(i)) -
    a(obs_i)| (same proprio, same noise), D_noise = mean |a(obs_i, n'_i) - a(obs_i, n_i)|. Actions are
    ``DeployPolicy.act`` normalised actions; frozen features are computed once per observation through the same calls
    as ``DeployPolicy.act`` (checked bitwise against it on the first samples)."""
    import torch

    from s4d.rwlearn import actions as A

    rng = np.random.default_rng(seed)
    n = len(samples)
    noise = rng.standard_normal((n, 1, A.ACTION_DIM)).astype(np.float32)
    noise2 = rng.standard_normal((n, 1, A.ACTION_DIM)).astype(np.float32)
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
    obs = _observations(policy, samples)
    off = policy.offline
    pixel = off.condition == "e2e_cnn"
    feats = None if pixel else [off.features(o).to(torch.float16).float() for o in obs]

    def act(i_obs: int, i_state: int, nz) -> np.ndarray:
        s = samples[i_state]
        nz = torch.as_tensor(nz)
        if pixel:
            return policy.act(obs[i_obs], s["pose7"], s["gripper_register"], noise=nz)["action_norm"]
        proprio = A.measured_proprio(s["pose7"], s["gripper_register"])
        return off.act_from_features(feats[i_obs], proprio, s["pose7"], noise=nz)["action_norm"]

    for i in range(min(4, n)):  # the cached-feature path IS DeployPolicy.act
        ref = policy.act(obs[i], samples[i]["pose7"], samples[i]["gripper_register"],
                         noise=torch.as_tensor(noise[i]))["action_norm"]
        if not np.array_equal(ref, act(i, i, noise[i])):
            raise AssertionError("probe path differs from DeployPolicy.act")
    real = np.stack([act(i, i, noise[i]) for i in range(n)])
    vis = np.stack([act(int(perm[i]), i, noise[i]) for i in range(n)])
    nz2 = np.stack([act(i, i, noise2[i]) for i in range(n)])
    spread = float(real.std(0).mean())
    d_vis = float(np.abs(vis - real).mean())
    d_noise = float(np.abs(nz2 - real).mean())
    ok = d_vis >= E2_SPREAD_FRACTION * spread and d_vis >= E2_NOISE_FRACTION * d_noise
    return check(bool(ok), samples=n, action_spread=spread, d_vis=d_vis, d_noise=d_noise,
                 d_vis_over_spread=d_vis / spread if spread else math.inf,
                 d_vis_over_noise=d_vis / d_noise if d_noise else math.inf,
                 episodes=sorted({s["episode"] for s in samples}),
                 cameras=sorted({c for s in samples for c in s["cameras"]}),
                 **({"m5_note": "multi-view probe sets: recorded mutual-nearest sets, M5 not certified"}
                    if policy.input_spec.views > 1 else {}))


# ------------------------------------------------------------------------------------------------ f
def stability(rows: list[dict], budget: int) -> dict:
    """f1 finite logged values, f2 no q_mean > 100 (training and validation), f3 median and final training q_mean
    <= +1 over steps >= 0.8 budget. Also reports first q_max > 0, peak q_mean and the final validation errors."""
    if not rows:
        return check(False, reason="no metrics")
    bad = sorted({k for r in rows for k, v in r.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isfinite(v)})
    runaway = [(r["step"], k, r[k]) for r in rows for k in Q_KEYS if k in r and r[k] > F2_Q_RUNAWAY]
    train_q = [(r["step"], r["training/critic/q_mean"]) for r in rows if "training/critic/q_mean" in r]
    late = [q for s, q in train_q if s >= F3_LATE_FRACTION * budget]
    final_q = train_q[-1][1] if train_q else None
    late_median = float(np.median(late)) if late else None
    f1, f2 = not bad, not runaway
    f3 = late_median is not None and late_median <= F3_Q_FEASIBLE and final_q <= F3_Q_FEASIBLE
    qmax0 = next((r["step"] for r in rows if r.get("training/critic/q_max", -1) > 0), None)
    peak = max(train_q, key=lambda x: x[1]) if train_q else None
    last_val = next((r for r in reversed(rows) if "validation/onestep_pos_err_mm" in r), {})
    reasons = ([f"non-finite {bad}"] if bad else []) + ([f"q_mean > {F2_Q_RUNAWAY} at {runaway[:3]}"] if runaway
                                                        else []) + \
        ([] if f3 else [f"late q_mean median {late_median} / final {final_q} > {F3_Q_FEASIBLE}"])
    return check(f1 and f2 and f3, f1_finite=f1, f2_no_runaway=f2, f3_late_feasible=f3, last_step=rows[-1]["step"],
                 final_q_mean=final_q, late_q_mean_median=late_median, peak_q_mean=peak, first_q_max_positive=qmax0,
                 final_validation={k: last_val.get(k) for k in ("validation/onestep_pos_err_mm",
                                                                "validation/bcflow_pos_err_mm",
                                                                "validation/critic/critic_loss")},
                 **({"reason": "; ".join(reasons)} if reasons else {}))


# ------------------------------------------------------------------------------------------------ classification
def classify(condition: str, checks: dict, *, quarantine: str | None = None) -> dict:
    """POL-CRITERIA.md classes. ``checks``: a, b, c, d, e1, e2, f (each a ``check`` dict; a/e1 may be absent where
    the condition has none). ``quarantine``: an explicit coordinator/user decision (reason)."""
    if condition == "ours":
        return {"status": "fixture-only", "reasons": ["ours: training deferred; interfaces + labelled fixtures only"],
                "flags": []}
    reasons, flags = [], []
    for k in ("a", "b"):
        if k in checks and not checks[k]["pass"]:
            reasons.append(f"{k}: {checks[k].get('reason', 'failed')}")
    if reasons:
        return {"status": "incomplete", "reasons": reasons, "flags": flags}
    e1, e2 = checks.get("e1"), checks.get("e2")
    if condition != "e2e_cnn":
        if e1 is None or e1["pass"] is None:
            reasons.append("e1: not evaluated")
        elif not e1["offset_pass"]:
            reasons.append(f"e1: offset_to_signal {e1['offset_to_signal']:.3g} > {E1_OFFSET_MAX}")
        elif not e1["ln_pass"] and e2 is not None and e2["pass"]:
            flags.append(ATTENUATED)
        elif not e1["ln_pass"]:
            reasons.append(f"e1: LN unit std {e1['ln_unit_std_raw']:.3g} < {E1_LN_UNIT_STD_MIN}")
    if e2 is None or e2["pass"] is None:
        reasons.append(f"e2: not evaluated ({(e2 or {}).get('reason', 'missing')})")
    elif not e2["pass"]:
        reasons.append(f"e2: D_vis {e2['d_vis']:.3g} vs spread {e2['action_spread']:.3g} / noise {e2['d_noise']:.3g}")
    for k in ("c", "d", "f"):
        if k not in checks or checks[k]["pass"] is None:
            reasons.append(f"{k}: not evaluated")
        elif not checks[k]["pass"]:
            reasons.append(f"{k}: {checks[k].get('reason', 'failed')}")
    if quarantine:
        reasons.insert(0, f"quarantined by decision: {quarantine}")
    return {"status": "quarantined" if reasons else "deployable", "reasons": reasons, "flags": flags}
