"""Dataset-contract and model-interface checks for one processed real-world episode (inference only, no training),
PER CONSUMER FAMILY (2026-11-06): an episode is contract-OK for a family iff that family passes, and a failure of one
family never aborts or decides the other.

* ``report["singleview"]`` - single-view consumers (FQL single view, e2e CNN + FQL, DynaMo): each camera's own frames
  in the compact store, its robot <-> camera sync (``robot/ok``) and the ``rwlearn.transitions.timelines(multiview=
  False)`` transitions of the action contract on the raw recording. No cross-camera set is involved. CPU only.
* ``report["multiview"]`` - multi-view consumers: our model through ``RealworldWindowDataset`` over the compact
  training store (criterion M5: the store's mutual-nearest sets accepted iff the robot surface moved <= 5 mm;
  ``validate_batch`` per sample) and ``s4d.data.contract.collate``; batch checks (shapes, finite values, temporal
  sampling, K/w2c consistency); ``build_model`` + ``forward_losses`` under ``torch.no_grad`` on the inference server
  (``forward_check`` kernel); and, when the dataset declares SNeRL, real scenes in SNeRL's own format (128x128 squares,
  normalised OpenGL poses, SAM 3 semantics) from the SAME M5 sets (``depth_stages.episode_frame_sets``) through its
  ``training_step`` and the DINO teacher equality check (``snerl_check`` kernel). An empty multi-view split (no M5
  window) fails only this family. The former window export (legacy roots) is checked as the multi-view family.

The process exits 0 once both families are evaluated and recorded; the QA step (``rwprep_local.py qa --families``)
turns the families into per-family verdicts. Contract violations inside a family's data path (``validate_batch``,
shapes) and tooling errors still raise. These checks show data/forward compatibility only, not learning performance.

    .venv/bin/python -I ops/inference_server/contract_checks.py --pipeline configs/realworld/mug_hanging.yaml \\
        --server configs/realworld/inference_server.yaml --episode mug-hanging-test-000001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from ops.inference_server.command_support import pool_config  # noqa: E402
from ops.inference_server.run_episode import local_head, spec  # noqa: E402
from ops.inference_server.session import RemoteRunner, Server  # noqa: E402
from ops.inference_server.stream_client import RemoteWorkerPool  # noqa: E402


def axes_lookat(T_cam_base: np.ndarray) -> np.ndarray:
    """Least-squares point closest to every camera's optical axis (base frame): the scene centre the cameras view."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for T in T_cam_base:
        R, t = T[:3, :3], T[:3, 3]
        c, d = -R.T @ t, R[2]  # camera centre and optical axis (base frame)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def batch_checks(batch: dict) -> dict:
    import torch

    out = {"shapes": {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)},
           "dtypes": {k: str(v.dtype) for k, v in batch.items() if torch.is_tensor(v)},
           "finite": {k: bool(torch.isfinite(v.float()).all()) for k, v in batch.items()
                      if torch.is_tensor(v) and v.is_floating_point()}}
    K, w2c, c2w = batch["K"].double(), batch["w2c"].double(), batch["c2w"].double()
    out["w2c_c2w_inverse_max_err"] = float((w2c @ c2w - torch.eye(4, dtype=torch.float64)).abs().max())
    out["K_positive_focal"] = bool((K[..., 0, 0] > 0).all() and (K[..., 1, 1] > 0).all())
    depth = batch["depth"]
    out["depth_valid_fraction"] = float((depth > 0).float().mean())
    out["depth_range_m"] = [float(depth[depth > 0].min()), float(depth[depth > 0].max())] if (depth > 0).any() else None
    ts = []
    for t_idx, stride in zip(batch["meta"]["t_indices"], batch["meta"]["stride"]):
        if list(np.diff(t_idx)) != [stride, stride]:
            ts.append({"t_indices": list(t_idx), "stride": stride, "problem": "index steps != stride"})
    out["temporal_index_problems"] = ts
    return out


def singleview_family(store, ep: str, raw=None, expected_cameras=None) -> dict:
    """Single-view consumer family of one store episode: per camera the frame count, the robot <-> camera sync
    fraction (``robot/ok`` of a compact store; absent for stores without per-camera robot state) and the transitions of
    ``rwlearn.transitions.timelines(store, ep, multiview=False)`` under the action contract of the raw recording
    (``raw`` default: ``store.raw_recording(ep)``). A camera passes iff it has frames, some robot-synchronised frames
    and at least one valid transition; the family passes iff the recording satisfies the action contract and every
    camera passes and every recording camera (``expected_cameras``) is in the store. Data properties (the recording
    outside the action contract, commands outside its bounds, cameras excluded from the store) are recorded as
    problems; anything else raises (a tooling error is never turned into a family verdict)."""
    from s4d.rwlearn import actions as A
    from s4d.rwlearn.transitions import episode_transitions, timelines

    out = {"passed": False, "problems": [], "cameras": {}}
    contract = None
    try:
        raw = raw if raw is not None else A.load_raw(store.raw_recording(ep))
        A.check_contract_preconditions(raw)
        contract = A.contract_for(raw)
    except (ValueError, KeyError) as exc:  # recording validation (format, preconditions, period, teleop limits)
        out["problems"].append(f"action contract: {type(exc).__name__}: {exc}")
    if expected_cameras is not None:
        missing = sorted(set(expected_cameras) - set(store.cameras(ep)))
        if missing:
            out["problems"].append(f"recording cameras missing from the training store: {missing}")
    episode = getattr(store, "episode", None)  # CompactStore: per-camera robot state of the store
    lines = {tl.name: tl for tl in timelines(store, ep, multiview=False)}
    for cam in store.cameras(ep):
        n = len(store.timestamps(ep, cam))
        row = {"frames": int(n)}
        if episode is not None:
            ok = np.asarray(episode(ep).robot(cam)["ok"], bool)
            row["robot_ok_fraction"] = float(ok.mean()) if n else 0.0
        if contract is not None:
            try:
                tr = episode_transitions(raw, lines[cam], contract)
                row.update(observations=int(len(tr["times"])), transitions=int(tr["transition_ok"].sum()),
                           state_ok_fraction=float(np.mean(tr["state_ok"])) if len(tr["times"]) else 0.0,
                           label=tr["label"])
            except ValueError as exc:
                if "outside the contract bounds" not in str(exc):  # the one data property raised here
                    raise
                row["error"] = f"{type(exc).__name__}: {exc}"
        row["passed"] = bool(n > 0 and row.get("robot_ok_fraction", 1.0) > 0 and row.get("transitions", 0) > 0)
        out["cameras"][cam] = row
    out["passed"] = bool(out["cameras"]) and not out["problems"] and all(r["passed"] for r in out["cameras"].values())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--output-root", help="override the pipeline config's output_root (e.g. a fresh baseline run)")
    ap.add_argument("--model-config", default="configs/droid/pretrain.yaml")
    ap.add_argument("--model-override", action="append", default=[],
                    help="config override for the forward check (the encoder image size follows the data unless given)")
    ap.add_argument("--strides", default="1,2", help="window strides of the training-store loader")
    ap.add_argument("--snerl-config", default="configs/baselines/snerl/hammer.yaml")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--snerl-scenes", type=int, default=6)
    args = ap.parse_args(argv)
    if args.batches < 1 or args.snerl_scenes < 1:
        ap.error("--batches and --snerl-scenes must be positive")

    import torch
    from torch.utils.data import DataLoader

    from s4d.data.contract import collate
    from s4d.realworld import io as rio
    from s4d.realworld.dataset import RealworldCacheDataset
    from s4d.realworld.depth_stages import configured_sets_policy, episode_frame_sets
    from s4d.realworld.pipeline import accepted_calibration, camera_views, load_meta, read_frames
    from s4d.realworld.train_store import FILE as STORE_FILE
    from s4d.realworld.train_store import MAX_SET_MOTION_M, CompactStore, sync_counts
    from s4d.realworld.train_windows import RealworldWindowDataset

    cfg = yaml.safe_load(Path(args.pipeline).read_text())
    if args.output_root:
        cfg["output_root"] = args.output_root
    root, ep = Path(cfg["output_root"]), args.episode
    out_dir = root / "reports" / ep / "contracts"
    out_dir.mkdir(parents=True, exist_ok=True)
    head = local_head()
    compact = root / "train_data" / ep
    compact_record = rio.read_record(compact)
    if (compact / STORE_FILE).is_file() or compact_record is not None:
        if compact_record is None:
            raise ValueError(f"{compact}: incomplete training store (no valid completion record); re-export train_data")
        cache, data_file = compact, compact / STORE_FILE
    else:  # former window export
        cache = root / "export" / ep
        data_file = cache / "manifest.json"
    strides = tuple(int(x) for x in args.strides.split(","))
    if cache == compact:  # the encoder input size follows the stored image size unless overridden explicitly
        with h5py.File(data_file, "r") as f:
            size = int(f.attrs["image_size"])
        given = {o.split("=", 1)[0] for o in args.model_override}
        overrides = [f"model.encoder.{k}={size}" for k in ("image_height", "image_width")
                     if f"model.encoder.{k}" not in given] + list(args.model_override)
    else:
        overrides = list(args.model_override)
    # the SNeRL scenes read the normalized frames, calibration and SAM 3 masks under the root's frame-set policy
    snerl_inputs = {}
    if "snerl" in (cfg.get("baselines") or []):
        dirs = {"ingest": rio.episode_dir(root, ep) / "normalized", "calibrate": rio.stage_dir(root, ep, "calibrate"),
                "sam3_masks": rio.stage_dir(root, ep, "sam3_masks")}
        snerl_inputs = {"sets_policy": configured_sets_policy(root, ep, cfg),
                        **{k: (rio.read_record(d) or {}).get("build_id") for k, d in dirs.items()}}
    store = rio.ChunkStore(out_dir / "calls", rio.stage_fingerprint(
        "contract_checks", "3", {"model_config": args.model_config, "model_overrides": overrides,
                                 "snerl_config": args.snerl_config, "batches": args.batches,
                                 "scenes": args.snerl_scenes,
                                 "loader": {"strides": list(strides), "sync_criterion": "M5",
                                            "max_set_motion_m": MAX_SET_MOTION_M} if cache == compact else "export"},
        {"manifest": rio.sha256_file(data_file), "source": head, **snerl_inputs}))
    report = {"episode": ep, "source_commit": head, "cache": str(cache), "model_config": args.model_config,
              "model_overrides": overrides, "families": ["singleview", "multiview"]}
    # ---- single-view family: each camera's own frames and robot sync (CPU, no cross-camera set, no server)
    if cache == compact:
        sv_store = CompactStore([compact])
        recording = sorted({v.rig_id for v in camera_views(load_meta(root, ep))})
        try:
            report["singleview"] = singleview_family(sv_store, sv_store.episodes()[0], expected_cameras=recording)
        finally:
            sv_store.close()
    else:
        report["singleview"] = {"passed": False, "problems": ["legacy window export: no per-camera training store"],
                                "cameras": {}}
    # ---- multi-view family: M5 windows + forward + SNeRL scenes from the same M5 sets
    mv = {"passed": False, "problems": []}
    report["multiview"] = mv
    report["dataset"], report["forward"], report["snerl"] = None, [], None
    server = pool = audit = ds = None
    report["persistence_audit"] = {"ok": True, "skipped": "no remote calls"}
    try:
        if cache == compact:
            cs = CompactStore([compact])
            try:
                ef = cs.episode(cs.episodes()[0])
                missing = sorted(set(recording) - set(ef.cameras))
                if "surface_disp_m" not in ef.sync():
                    mv["problems"].append("store has no per-set robot motion (schema < 1.1.0); re-export train_data "
                                          "for M5; legacy loaders require sync_criterion='time' explicitly")
                else:
                    sets = cs.sync_sets(cs.episodes()[0])
                    mv["sync"] = sync_counts(np.asarray(sets["frames"]),
                                             np.asarray(sets["surface_disp_m"], np.float64))
            finally:
                cs.close()
            if not mv["problems"]:
                try:
                    ds = RealworldWindowDataset([cache], split="train", strides=strides,
                                                max_set_motion_m=MAX_SET_MOTION_M)  # = viewer / SNeRL sets
                except ValueError as exc:
                    if str(exc) != "empty split train":
                        raise
                    mv["problems"].append(f"no multi-view training window under M5 ({exc}): {mv['sync']}")
            if ds is not None:
                report["dataset"] = {"format": "train_data", "samples": len(ds), "cameras": ds.cams[0],
                                     "image_size": ds.store.image_hw, **ds.counts}
            if missing:  # multi-view consumers need every recording camera (e.g. ReViWo / SNeRL: three views)
                mv["problems"].append(f"recording cameras missing from the training store: {missing}")
        else:
            manifest = json.loads((cache / "manifest.json").read_text())
            ds = RealworldCacheDataset(cache, split="train")  # validate_batch runs per sample
            report["dataset"] = {"format": "export", "samples": len(ds), "windows": len(manifest["windows"]),
                                 "cameras": manifest["cameras"], "resolutions": manifest["resolutions"],
                                 "aspect": manifest.get("aspect"),
                                 "windows_dropped_for_time_gaps": manifest.get("windows_dropped_for_time_gaps")}
        if ds is not None:
            for i in range(len(ds)):  # the whole split through validate_batch, one sample at a time
                ds[i]
            report["dataset"]["validate_batch"] = f"passed for all {len(ds)} samples"
            # No server dependency for single-view checks or episodes with no M5 training window.
            server = Server.load(Path(args.server))
            server.verify_container()
            server.verify_source(head)
            audit = server.audit_start()
            pool = RemoteWorkerPool(pool_config(server.config, head), log_dir=out_dir / "worker_logs")
            runner = RemoteRunner(pool, spec(cfg, "s4d_kernels", expected_commit=head))
            loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
            for b, batch in enumerate(loader):
                if b >= args.batches:
                    break
                checks = batch_checks(batch)
                arrays = {k: v.numpy() for k, v in batch.items() if torch.is_tensor(v)}
                meta = {k: [list(x) if isinstance(x, tuple) else x for x in v] for k, v in batch["meta"].items()}
                got = store.get(f"forward/{b}")
                if got is None:
                    _, kmeta, handle = runner.kernel("forward_check", {"config": args.model_config, "meta": meta,
                                                                       "overrides": overrides, "seed": 0}, arrays,
                                                     chunk_id=f"forward/{b}")
                    store.put(f"forward/{b}", {}, kmeta)
                    runner.ack(handle)
                    got = store.get(f"forward/{b}")
                kmeta = got[1]
                report["forward"].append({"batch": b, "checks": checks, "model": kmeta})
            # SNeRL scenes: only when the dataset declares SNeRL (its SAM 3 semantics + DINO teacher exist only then)
            if "snerl" in (cfg.get("baselines") or []) and not mv["problems"]:  # (missing cameras: no SNeRL scene)
                report["dino_weights"] = server.stage_weights(workers={"dino_snerl"})  # SNeRL teacher (online targets)
                meta_ep = load_meta(root, ep)
                streams = [v.stream_id for v in camera_views(meta_ep)]
                sets = episode_frame_sets(meta_ep, root, ep, streams, cfg)  # the store's M5 sets (single source)
                T = np.stack([accepted_calibration(root, ep, s) for s in streams])
                lookat = axes_lookat(T)
                pick = np.linspace(0, len(sets["index"]) - 1, args.snerl_scenes).round().astype(int)
                images, sem = [], []
                for k in pick:
                    idx = sets["index"][k]
                    images.append(np.stack([read_frames(root, ep, s, [int(idx[v])])[0] for v, s in enumerate(streams)]))
                    row = []
                    for v, s in enumerate(streams):
                        with h5py.File(rio.stage_dir(root, ep, "sam3_masks") / f"{rio.safe_name(s)}.h5", "r") as f:
                            row.append(f["semantic"][int(idx[v])])
                    sem.append(np.stack(row))
                Ks = np.stack([meta_ep.stream(s).intrinsics.K() for s in streams])
                got = store.get("snerl_check")
                if got is None:
                    out, kmeta, handle = runner.kernel("snerl_check", {"config": args.snerl_config,
                                                                       "lookat": lookat.tolist(), "pose_scale": 4.0,
                                                                       "seed": 0},
                                                       {"images": np.stack(images), "semantic": np.stack(sem), "K": Ks,
                                                        "cam2world": np.linalg.inv(T)}, chunk_id="snerl_check")
                    store.put("snerl_check", out, kmeta)
                    runner.ack(handle)
                    got = store.get("snerl_check")
                out, kmeta = got
                report["snerl"] = {**kmeta, "scene_sets": pick.tolist(), "set_criterion": sets.get("criterion", "time"),
                                   "scene_frames": sets["index"][pick].tolist(),
                                   "camera_distance_to_lookat_m": [float(np.linalg.norm(-t[:3, :3].T @ t[:3, 3] - lookat))
                                                                   for t in T]}
                from PIL import Image

                Image.fromarray(np.concatenate([np.concatenate(list(x), 1) for x in out["images128"]], 0)).save(
                    out_dir / "snerl_scenes_128.png")
            # total and every loss finite (as the QA): moving-region METRICS are legitimately NaN when no target
            # pixel moves and are recorded, not failed. The stream protocol encodes NaN/Inf as strings.
            def finite_scalar(value):
                return isinstance(value, (int, float)) and bool(np.isfinite(value))

            finite = [finite_scalar(f["model"].get("total")) and bool(f["model"].get("losses")) and
                      all(finite_scalar(v) for v in f["model"]["losses"].values()) for f in report["forward"]]
            if not finite or not all(finite):
                mv["problems"].append(f"forward total/losses not finite for every batch: {finite}")
            if "snerl" in (cfg.get("baselines") or []) and report["snerl"] is None and not mv["problems"]:
                mv["problems"].append("SNeRL declared but no SNeRL scene was checked")
            if report["snerl"] is not None:
                scenes = report["snerl"].get("scenes") or []
                if not scenes or not all(x.get("finite") is True and x.get("dino_worker_equals_consumer") is True
                                         for x in scenes):
                    mv["problems"].append("SNeRL scenes not finite or DINO teacher mismatch")
            mv["windows"] = len(ds)
        mv["passed"] = ds is not None and len(ds) > 0 and not mv["problems"]
    except Exception as exc:
        mv["passed"] = False
        mv["problems"].append(f"{type(exc).__name__}: {exc}")
        raise  # tooling/contract errors stay visible; retain the independent single-view verdict below
    finally:
        try:
            if ds is not None and hasattr(ds, "store"):
                ds.store.close()
            if pool is not None:
                pool.close()
            if audit is not None:
                report["persistence_audit"] = server.audit_end(audit)
        except Exception as exc:
            mv["passed"] = False
            mv["problems"].append(f"cleanup: {type(exc).__name__}: {exc}")
            raise
        finally:
            rio.atomic_write_json(out_dir / "contract_checks.json", json.loads(json.dumps(report, default=str)))
    fw = report["forward"]
    print(json.dumps({"singleview_passed": report["singleview"]["passed"],
                      "singleview_problems": report["singleview"]["problems"],
                      "multiview_passed": mv["passed"], "multiview_problems": mv["problems"],
                      "validate_batch": (report["dataset"] or {}).get("validate_batch"),
                      "forward_finite": [f["model"].get("finite") for f in fw],
                      "forward_total": [f["model"].get("total") for f in fw],
                      "snerl_finite": [s["finite"] for s in report["snerl"]["scenes"]] if report["snerl"] else
                      "not run (SNeRL not declared for this dataset or no multi-view window)",
                      "dino_equal": [s["dino_worker_equals_consumer"] for s in report["snerl"]["scenes"]]
                      if report["snerl"] else None,
                      "audit_ok": report["persistence_audit"]["ok"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
