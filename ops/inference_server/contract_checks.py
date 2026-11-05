"""Dataset-contract and model-interface checks for one processed real-world episode (inference only, no training).

1. Our model: the compact training store (``<root>/train_data/<episode>``) through ``RealworldWindowDataset`` (the
   former window export through ``RealworldCacheDataset`` when only that exists; ``validate_batch`` per sample) and
   ``s4d.data.contract.collate``; batch checks (shapes, finite values, temporal sampling against the recorded
   timestamps, K/w2c consistency); then ``build_model`` + ``forward_losses`` under ``torch.no_grad`` on the inference
   server (``forward_check`` kernel, config ``--model-config``).
2. SNeRL: real scenes in SNeRL's own format (128x128 squares, normalised OpenGL poses, SAM 3 semantics) through its
   ``training_step`` under ``torch.no_grad`` and the DINO teacher equality check (``snerl_check`` kernel).
These show data/forward compatibility only; they are not evidence of learning performance.

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

    import torch
    from torch.utils.data import DataLoader

    from s4d.data.contract import collate
    from s4d.realworld import io as rio
    from s4d.realworld.dataset import RealworldCacheDataset
    from s4d.realworld.depth_stages import configured_sets_policy, frame_sets
    from s4d.realworld.pipeline import accepted_calibration, camera_views, load_meta, read_frames
    from s4d.realworld.train_store import FILE as STORE_FILE
    from s4d.realworld.train_windows import RealworldWindowDataset

    cfg = yaml.safe_load(Path(args.pipeline).read_text())
    if args.output_root:
        cfg["output_root"] = args.output_root
    server = Server.load(Path(args.server))
    root, ep = Path(cfg["output_root"]), args.episode
    out_dir = root / "reports" / ep / "contracts"
    out_dir.mkdir(parents=True, exist_ok=True)
    head = local_head()
    server.verify_container()
    server.verify_source(head)
    audit = server.audit_start()
    pool_cfg = pool_config(server.config, head)
    pool = RemoteWorkerPool(pool_cfg, log_dir=out_dir / "worker_logs")
    runner = RemoteRunner(pool, spec(cfg, "s4d_kernels", expected_commit=head))
    compact = root / "train_data" / ep
    if rio.read_record(compact) is not None:  # the training store (2026-11-06 format)
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
        "contract_checks", "2", {"model_config": args.model_config, "model_overrides": overrides,
                                 "snerl_config": args.snerl_config, "batches": args.batches,
                                 "scenes": args.snerl_scenes, "loader": {"strides": list(strides)}
                                 if cache == compact else "export"},
        {"manifest": rio.sha256_file(data_file), "source": head, **snerl_inputs}))
    report = {"episode": ep, "source_commit": head, "cache": str(cache), "model_config": args.model_config,
              "model_overrides": overrides}
    try:
        if cache == compact:
            ds = RealworldWindowDataset([cache], split="train", strides=strides)
            report["dataset"] = {"format": "train_data", "samples": len(ds), "cameras": ds.cams[0],
                                 "image_size": ds.store.image_hw, **ds.counts}
        else:
            manifest = json.loads((cache / "manifest.json").read_text())
            ds = RealworldCacheDataset(cache, split="train")  # validate_batch runs per sample
            report["dataset"] = {"format": "export", "samples": len(ds), "windows": len(manifest["windows"]),
                                 "cameras": manifest["cameras"], "resolutions": manifest["resolutions"],
                                 "aspect": manifest.get("aspect"),
                                 "windows_dropped_for_time_gaps": manifest.get("windows_dropped_for_time_gaps")}
        for i in range(len(ds)):  # the whole split through validate_batch, one sample at a time
            ds[i]
        report["dataset"]["validate_batch"] = f"passed for all {len(ds)} samples"
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
        report["forward"] = []
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
        report["snerl"] = None
        if "snerl" in (cfg.get("baselines") or []):
            report["dino_weights"] = server.stage_weights(workers={"dino_snerl"})  # SNeRL teacher (online targets)
            meta_ep = load_meta(root, ep)
            streams = [v.stream_id for v in camera_views(meta_ep)]
            sets = frame_sets(meta_ep, streams, **configured_sets_policy(root, ep, cfg))
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
                out, kmeta, handle = runner.kernel("snerl_check", {"config": args.snerl_config, "lookat": lookat.tolist(),
                                                                   "pose_scale": 4.0, "seed": 0},
                                                   {"images": np.stack(images), "semantic": np.stack(sem), "K": Ks,
                                                    "cam2world": np.linalg.inv(T)}, chunk_id="snerl_check")
                store.put("snerl_check", out, kmeta)
                runner.ack(handle)
                got = store.get("snerl_check")
            out, kmeta = got
            report["snerl"] = {**kmeta, "scene_sets": pick.tolist(),
                               "camera_distance_to_lookat_m": [float(np.linalg.norm(-t[:3, :3].T @ t[:3, 3] - lookat))
                                                               for t in T]}
            from PIL import Image

            Image.fromarray(np.concatenate([np.concatenate(list(x), 1) for x in out["images128"]], 0)).save(
                out_dir / "snerl_scenes_128.png")
    finally:
        pool.close()
        report["persistence_audit"] = server.audit_end(audit)
    rio.atomic_write_json(out_dir / "contract_checks.json", json.loads(json.dumps(report, default=str)))
    fw = report.get("forward", [])
    print(json.dumps({"validate_batch": report["dataset"]["validate_batch"],
                      "forward_finite": [f["model"].get("finite") for f in fw],
                      "forward_total": [f["model"].get("total") for f in fw],
                      "snerl_finite": [s["finite"] for s in report["snerl"]["scenes"]] if report["snerl"] else
                      "not run (SNeRL not declared for this dataset)",
                      "dino_equal": [s["dino_worker_equals_consumer"] for s in report["snerl"]["scenes"]]
                      if report["snerl"] else None,
                      "audit_ok": report["persistence_audit"]["ok"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
