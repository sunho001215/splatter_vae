"""Read-only CPU deployment evidence for the real-world FQL policies, separate from the release test suite (real
checkpoints and recordings; missing files FAIL, never skip). No GPU, network, job/campaign imports or robot/camera
I/O; outputs only the JSON given by --out (or stdout). Subcommands:

  contracts   FQL dataset attrs -> <run>.deploy_contract.json sidecars (trained camera ids etc.) for servers that hold
              only policy.pt + encoder export (``DeployPolicy(..., training=<sidecar>)``).
  preprocess  recorded episode (raw H5) through the CPU client preprocessing vs the released training-store episode:
              RGB / K / c2w / depth-resampling bit-exactness and the native-RealSense vs FoundationStereo depth gap.
  equality    real checkpoints on recorded observations (client path): DeployPolicy.infer == DeployPolicy.act (the
              OfflinePolicy methods with training-precision float16 features) bitwise with identical FQL noise; the
              action difference to the unrounded OfflinePolicy.act; diagnostics vs the FQL training dataset (features,
              proprio, action) and, with a training store, the train (FS depth) vs deploy (native depth) effect.

Examples: docs/realworld_deployment.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def local(runs_root: Path, recorded: str) -> Path:
    """A run file recorded with another host's path -> the local backup ``<runs_root>/<run>/<file>``."""
    p = Path(recorded)
    return runs_root / p.parent.name / p.name


def stats(x) -> dict:
    x = np.asarray(x, np.float64).ravel()
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "mean": float(x.mean()),
            "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)), "max_abs": float(np.abs(x).max())}


# ------------------------------------------------------------------------------------------------ contracts
def cmd_contracts(args) -> dict:
    import h5py
    import torch

    from s4d.rwlearn.deploy.contracts import read_training_contract, training_contract

    out = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        meta = torch.load(args.runs_root / run / "policy.pt", map_location="cpu", weights_only=False)["meta"]
        dataset = local(args.runs_root, meta["dataset"])
        with h5py.File(dataset, "r") as f:
            training = training_contract(dict(f.attrs))
        training["source_dataset"] = {"recorded": meta["dataset"], "local": str(dataset), "sha256": sha256(dataset)}
        read_training_contract(meta, training)  # checked against the checkpoint
        path = args.out_dir / f"{run}.deploy_contract.json"
        path.write_text(json.dumps(training, indent=1) + "\n")
        out[run] = {"sidecar": str(path), "sha256": sha256(path), "camera_orders": training["camera_orders"],
                    "training_depth": training.get("training_depth")}
    return out


# ------------------------------------------------------------------------------------------------ preprocessing
def cmd_preprocess(args) -> dict:
    from s4d.realworld.stages.depth import load_view_depth
    from s4d.realworld.train_store import EpisodeFile
    from s4d.rwlearn.deploy import preprocess_frame
    from s4d.rwlearn.deploy.realsense import load_rig_session
    from s4d.rwlearn.deploy.replay import RecordedEpisode

    rec = RecordedEpisode(args.raw)
    store = EpisodeFile(args.store)
    try:
        attrs = store.attrs()
        raw_sha = sha256(args.raw)
        if attrs.get("raw_sha256") and attrs["raw_sha256"] != raw_sha:
            raise ValueError("training store episode was built from another recording")
        cams = list(store.cameras)
        session = load_rig_session(args.rig, cams, {c: rec.K_native(c) for c in cams})
        scales = json.loads(attrs.get("calibration_derived") or "{}").get("depth_scale", {}).get("factors", {})
        result = {"raw": str(args.raw), "raw_sha256": raw_sha, "store": str(args.store),
                  "store_attrs": {k: attrs.get(k) for k in ("format", "schema_version", "writer_version", "code_commit",
                                                            "depth_stage", "upstream", "calibration_derived")},
                  "rig_document": str(args.rig), "processed_root": str(args.root) if args.root else None,
                  "training_depth": attrs.get("depth_stage"), "deployment_depth": "native_realsense",
                  "deployment_alignment": args.alignment, "cameras": {}}
        for cam in cams:
            t_store, t_rec = store.timestamps(cam), rec.timestamps_ns(cam)
            if not np.array_equal(t_store, t_rec):
                raise ValueError(f"{cam}: store and recording frame identity differ")
            n = len(t_store)
            idx = np.unique(np.linspace(0, n - 1, min(n, args.frames)).round().astype(int)) if args.frames else \
                np.arange(n)
            corr = session["native_depth_correction"][cam]
            fs_scale = float(scales.get(f"{cam}/color", 1.0))
            c2w_doc = np.asarray(session["calibration"][cam]["c2w"])
            per = {"frames": int(len(idx)), "store_frames": int(n), "native_correction": corr,
                   "fs_depth_scale_store": fs_scale,
                   "c2w_rig_minus_store_max_abs": float(np.abs(c2w_doc - np.linalg.inv(store.w2c(cam))).max())}
            rgb_err, kerr, fs_err, rows = 0, 0.0, 0.0, {"signed_m": [], "abs_m": [], "rel": []}
            vn, vf, only_native, only_fs = [], [], [], []
            for i in idx:
                nat = rec.native(cam, int(i), depth=True, alignment=args.alignment)
                got = preprocess_frame(nat["rgb"], depth=nat["depth"], valid=nat["valid"], K_native=rec.K_native(cam),
                                       native_correction=corr)
                rgb_err = max(rgb_err, int(np.abs(got["rgb"].astype(int) - store.rgb(cam, [i])[0]).max()))
                kerr = max(kerr, float(np.abs(got["K"] - store.intrinsics(cam)).max()))
                train = store.depth(cam, [i])[0]
                if args.root:  # the training depth path through the CLIENT resampling (FS stage x store scale)
                    fs, fs_valid, K_fs = load_view_depth(args.root, args.episode, f"{cam}/color", [int(i)],
                                                         attrs["depth_stage"])
                    np.testing.assert_array_equal(K_fs, rec.K_native(cam))
                    fs_depth = fs[0] * np.float32(fs_scale) if fs_scale != 1.0 else fs[0]
                    via_client = preprocess_frame(nat["rgb"], depth=fs_depth, valid=fs_valid[0])["depth"]
                    fs_err = max(fs_err, float(np.abs(via_client - train).max()))
                both = got["valid"] & (train > 0)
                d = (got["depth"] - train)[both]
                rows["signed_m"].append(d)
                rows["abs_m"].append(np.abs(d))
                rows["rel"].append(d / train[both])
                vn.append(got["valid"].mean())
                vf.append((train > 0).mean())
                only_native.append((got["valid"] & ~(train > 0)).mean())
                only_fs.append((~got["valid"] & (train > 0)).mean())
            per.update(rgb_client_minus_store_max_abs=rgb_err, K_client_minus_store_max_abs=kerr,
                       fs_depth_via_client_minus_store_max_abs_m=fs_err if args.root else None,
                       native_valid_fraction=float(np.mean(vn)), fs_valid_fraction=float(np.mean(vf)),
                       native_only_fraction=float(np.mean(only_native)), fs_only_fraction=float(np.mean(only_fs)),
                       native_minus_fs_m=stats(np.concatenate(rows["signed_m"])),
                       abs_native_minus_fs_m=stats(np.concatenate(rows["abs_m"])),
                       relative_native_minus_fs=stats(np.concatenate(rows["rel"])))
            result["cameras"][cam] = per
    finally:
        store.close()
    return result


# ------------------------------------------------------------------------------------------------ equality
def _proxy_m5(rec, step_times: np.ndarray) -> float:
    """Flange-origin displacement across a set's frame times (PROXY for the URDF-surface M5 when no schema >= 1.1
    store is local; labelled as such in the evidence)."""
    from s4d.rwlearn import actions as A

    i = A.latest_at_or_before(rec.raw.robot_t, np.asarray(step_times, np.int64))
    p = rec.raw.measured[i, :3]
    return float(max(np.linalg.norm(p[a] - p[b]) for a in range(len(p)) for b in range(a + 1, len(p))))


def _windows(policy, spec, rec, args, m5):
    """Recorded windows through the client path: [(window, info)]."""
    from s4d.realworld.train_store import mutual_sync
    from s4d.rwlearn.deploy import HistoryWindow

    depth = "depth" in spec.modalities
    corr = spec.native_depth_correction
    cams = spec.camera_ids
    out = []
    if spec.views == 1:
        n = len(rec.timestamps_ns(cams[0]))
        ends = np.unique(np.linspace(2, n - 1, args.windows).round().astype(int))
        for end in ends:
            h = HistoryWindow(spec)
            for i in range(end - 2, end + 1):
                h.push(cams[0], rec.frame(cams[0], int(i), depth=depth, correction=corr.get(cams[0])))
            out.append((h.assemble({cams[0]: int(rec.frame_numbers(cams[0])[end])}),
                        {"stream_index": {cams[0]: int(end)}, "m5": None}))
        return out
    if m5 is not None:
        steps, disp, source = m5["steps"], m5["disp"], "training store sync/surface_disp_m (M5)"
        if m5["cameras"] != list(cams):
            raise ValueError("store camera order differs from the trained order")
    else:
        frames, _ = mutual_sync([rec.timestamps_ns(c) for c in cams], None)
        steps = [dict(zip(cams, map(int, row))) for row in frames if (row >= 0).all()]
        disp, source = None, "PROXY flange-origin displacement across the set times (no local M5 store)"
    for s in np.unique(np.linspace(2, len(steps) - 1, args.windows).round().astype(int)):
        chain = steps[s - 2:s + 1]
        if any(chain[j + 1][c] - chain[j][c] != 1 for j in range(2) for c in cams):
            continue  # not three consecutive synchronised steps
        h = HistoryWindow(spec)
        for st in chain:
            for c in cams:
                h.push(c, rec.frame(c, st[c], depth=depth, correction=corr.get(c)))
        ref = rec.frame_numbers(cams[0])
        if disp is not None:
            evidence = {int(ref[st[cams[0]]]): disp[int(ref[st[cams[0]]])] for st in chain}
        else:
            evidence = {int(ref[st[cams[0]]]): _proxy_m5(rec, [rec.timestamps_ns(c)[st[c]] for c in cams])
                        for st in chain}
        try:
            w = h.assemble({c: int(rec.frame_numbers(c)[chain[-1][c]]) for c in cams}, surface_disp_m=evidence)
        except ValueError as e:
            if "M5" in str(e):
                continue  # a moving set the training loader would also have excluded
            raise
        out.append((w, {"stream_index": chain[-1], "m5": source}))
    return out


def _dataset_reference(dataset, rec, policy, spec, window, info, episode):
    """Training-dataset features/proprio at this observation time (None when the timeline/time is absent)."""
    import h5py

    from s4d.rwlearn.transitions import history_indices

    tl = "sync" if spec.views > 1 else spec.camera_ids[0]
    with h5py.File(dataset, "r") as f:
        g = f.get(f"episodes/{episode}/{tl}")
        if g is None:
            return None
        k = np.flatnonzero(g["times"][:] == window.observation_time_ns)
        if len(k) != 1:
            return None
        k = int(k[0])
        hist = [k] if spec.sequence_feature else history_indices(g["run"][:], np.array([k]), 3)[0]
        ref = {"k": k, "proprio": g["proprio"][k], "measured_pose": g["measured_pose"][k],
               "transition_ok": bool(g["transition_ok"][k]), "history": [int(x) for x in hist]}
        if "features" in g:
            ref["features"] = g["features"][:][hist]
        ref["frames"] = g["frames"][:][hist].tolist()
    return ref


def cmd_equality(args) -> dict:
    import torch

    from s4d.realworld.train_store import EpisodeFile
    from s4d.rwlearn import actions as A
    from s4d.rwlearn.deploy import DeployPolicy
    from s4d.rwlearn.deploy.realsense import load_rig_session
    from s4d.rwlearn.deploy.replay import RecordedEpisode, m5_from_store

    torch.set_num_threads(args.threads)
    rec = RecordedEpisode(args.raw)
    raw_sha = sha256(args.raw)
    m5 = m5_from_store(args.store, rec) if args.store else None
    report = {"raw": str(args.raw), "raw_sha256": raw_sha, "store": str(args.store) if args.store else None,
              "rig_document": str(args.rig), "observations": "recorded episode through the client path "
              "(PIL-decoded JPEG; native z16 -> emulated rs.align -> colour-camera z -> x rig sensor_depth scale)",
              "comparison": "infer vs DeployPolicy.act (OfflinePolicy methods, float16-rounded frozen features as in "
                            "training); unrounded OfflinePolicy.act reported separately", "runs": {}}
    for run in args.runs:
        ckpt = args.runs_root / run / "policy.pt"
        meta = torch.load(ckpt, map_location="cpu", weights_only=False)["meta"]
        export = local(args.runs_root, meta["export_path"]) if meta["export_path"] else None
        sidecar = args.contracts / f"{run}.deploy_contract.json" if args.contracts else None
        dataset = local(args.runs_root, meta["dataset"])
        policy = DeployPolicy(ckpt, export_path=export, training=sidecar if sidecar else dataset)
        base = policy.input_spec
        cams = base.trained_camera_orders[0] if base.views > 1 else (base.trained_camera_orders[0][0],)
        choices = {"session_id": f"evidence-{run}", "native_hw": {c: rec.native_hw(c) for c in cams}}
        if base.views == 1:
            choices["camera_id"] = cams[0]
        if base.needs_calibration or "depth" in base.modalities:
            session = load_rig_session(args.rig, cams, {c: rec.K_native(c) for c in cams})
            if base.needs_calibration:
                choices["calibration"] = session["calibration"]
            if "depth" in base.modalities:
                choices["native_depth_correction"] = session["native_depth_correction"]
        spec = policy.reset_session(**choices)
        policy.warm_up()
        episode = next((ep for ep, r in json.loads(_attr(dataset, "raw_identities")).items()
                        if r["raw_sha256"] == raw_sha), None)
        rows = []
        store = EpisodeFile(args.store) if args.store else None
        for w, info in _windows(policy, spec, rec, args, m5):
            st = rec.state_at(w.observation_time_ns)
            ids = w.binding(f"robot@{st['pose_time_ns']}", pose_time_ns=st["pose_time_ns"],
                            gripper_time_ns=st["gripper_time_ns"])
            noise = torch.randn(1, 7, generator=torch.Generator().manual_seed(int(w.observation_time_ns % 2**31)))
            obs = w.observation()
            with torch.inference_mode():
                direct = policy.act(obs, st["pose7"], st["gripper_register"], noise=noise)
                again = policy.act(obs, st["pose7"], st["gripper_register"], noise=noise)
                unrounded = policy.offline.act(obs, st["pose7"], st["gripper_register"], noise=noise)
            deployed = policy.infer(w, st["pose7"], st["gripper_register"], ids, noise=noise)
            exact = all(np.array_equal(np.asarray(deployed[k]), np.asarray(direct[k])) for k in direct)
            row = {"frame_ids": w.frame_ids, "stream_index": info["stream_index"], "m5_source": info["m5"],
                   "padded": w.padded, "infer_equals_training_precision_act_bitwise": exact,
                   "repeat_act_bitwise": all(np.array_equal(np.asarray(again[k]), np.asarray(direct[k]))
                                             for k in direct),
                   "unrounded_offline_act_action_max_abs": float(
                       np.abs(deployed["action_norm"] - unrounded["action_norm"]).max()),
                   "unrounded_offline_act_bitwise": all(np.array_equal(np.asarray(deployed[k]),
                                                                       np.asarray(unrounded[k])) for k in unrounded),
                   "action_norm": deployed["action_norm"].tolist(), "timings_cpu_ms": deployed["timings"]}
            proprio = A.measured_proprio(st["pose7"], st["gripper_register"]).astype(np.float32)
            ref = _dataset_reference(dataset, rec, policy, spec, w, info, episode) if episode else None
            if ref is not None:
                row["dataset_k"] = ref["k"]
                row["proprio_minus_dataset_max_abs"] = float(np.abs(proprio - ref["proprio"]).max())
                row["dataset_history_frames"] = ref["frames"]
                if "features" in ref:
                    with torch.inference_mode():
                        feat = policy.offline.features(obs).cpu().numpy().astype(np.float16)  # training precision
                        f16 = torch.from_numpy(ref["features"].astype(np.float32))
                        f16 = f16 if spec.sequence_feature else f16[None]
                        trained = policy.offline.act_from_features(f16, ref["proprio"], st["pose7"], noise=noise)
                    flat = ref["features"].reshape(feat.shape)
                    diff = feat.astype(np.float32) - flat.astype(np.float32)
                    row["features_fp16_minus_dataset"] = {
                        "max_abs": float(np.abs(diff).max()), "equal_fraction": float((feat == flat).mean()),
                        "rel_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(flat.astype(np.float32)), 1e-12))}
                    row["action_minus_dataset_features_action_max_abs"] = float(
                        np.abs(deployed["action_norm"] - trained["action_norm"]).max())
            if store is not None and "depth" in spec.modalities:  # same window with the training store's FS depth
                obs = w.observation()  # fresh copy
                idx = info["stream_index"]
                hist = {c: [int(np.flatnonzero(rec.frame_numbers(c) == s)[0]) for s in w.frame_ids[c]] for c in cams}
                for v, c in enumerate(cams):
                    obs["depth"][:, v] = store.depth(c, hist[c])
                    np.testing.assert_array_equal(obs["rgb"][:, v], store.rgb(c, hist[c]))
                with torch.inference_mode():
                    fs_out = policy.act(obs, st["pose7"], st["gripper_register"], noise=noise)
                dp = deployed["delta_phys"] - fs_out["delta_phys"]
                row["native_vs_fs_depth"] = {
                    "action_norm_max_abs": float(np.abs(deployed["action_norm"] - fs_out["action_norm"]).max()),
                    "translation_mm": float(1e3 * np.linalg.norm(dp[:3])),
                    "rotation_deg": float(np.degrees(np.linalg.norm(dp[3:6]))),
                    "gripper_closure_abs": float(abs(deployed["gripper_closure"] - fs_out["gripper_closure"])),
                    "stream_index": idx}
            rows.append(row)
        if store is not None:
            store.close()
        if not rows:
            raise ValueError(f"{run}: no recorded window assembled")
        report["runs"][run] = {
            "condition": spec.condition, "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt),
            "export": str(export) if export else None, "contract_source": str(sidecar or dataset),
            "dataset_episode": episode, "InputSpec": spec.to_json(), "windows": rows,
            "all_bitwise_equal": all(r["infer_equals_training_precision_act_bitwise"] for r in rows),
            "unrounded_offline_act_action_max_abs": max(r["unrounded_offline_act_action_max_abs"] for r in rows),
            "deterministic_given_noise": all(r["repeat_act_bitwise"] for r in rows)}
    return report


def _attr(path, key):
    import h5py

    with h5py.File(path, "r") as f:
        v = f.attrs[key]
    return v.decode() if isinstance(v, bytes) else v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="JSON report path (default stdout)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("contracts")
    c.add_argument("--runs-root", type=Path, required=True)
    c.add_argument("--out-dir", type=Path, required=True)
    c.add_argument("runs", nargs="+")
    p = sub.add_parser("preprocess")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True, help="released train_data episode.h5 of this recording")
    p.add_argument("--rig", type=Path, required=True, help="the task's released rig document")
    p.add_argument("--root", type=Path, help="processed root holding the FS depth stage of the episode")
    p.add_argument("--episode", help="episode id inside --root")
    p.add_argument("--frames", type=int, default=0, help="evenly spaced frames per camera (0 = all)")
    p.add_argument("--alignment", choices=("rs_align", "sensor_depth"), default="rs_align",
                   help="native depth registration: live rs.align (emulated) -> colour z, or the sensor_depth stage")
    e = sub.add_parser("equality")
    e.add_argument("--runs-root", type=Path, required=True)
    e.add_argument("--raw", type=Path, required=True)
    e.add_argument("--rig", type=Path, required=True)
    e.add_argument("--store", type=Path, help="released train_data episode.h5 (M5 sets; FS-depth comparison)")
    e.add_argument("--contracts", type=Path, help="directory of <run>.deploy_contract.json sidecars")
    e.add_argument("--windows", type=int, default=6)
    e.add_argument("--threads", type=int, default=8)
    e.add_argument("runs", nargs="+")
    args = ap.parse_args()
    report = {"contracts": cmd_contracts, "preprocess": cmd_preprocess, "equality": cmd_equality}[args.cmd](args)
    text = json.dumps(report, indent=1, allow_nan=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
