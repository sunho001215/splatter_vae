"""Full-split retrieval and linear probes of a frozen baseline encoder, with our evaluator's metrics and protocol.

Loads the RL export (``vision.export_path``) through the RL-time adapter of each RL encoder config and evaluates the
features the agent caches (``s4d/baselines/representation.py``) on our pretraining config's validation split, for the
pretrained export and for the same encoder at its random initialisation (``encoder_init.pt`` beside the export).
Writes ``<output>/<encoder>/<pretrained|random_init>/summary.json`` (our metric names, ``@s<stride>``), the combined
``<output>/summary.json`` and ``<output>/provenance.json``. The 3D-MVP conditions are two encoder configs:

CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/eval_baseline_representation.py --task hammer \
    --encoder mvp3d_multiview mvp3d_single --export runs/s4-mvp3d-hammer-rem-b/encoder.pt \
    --output runs/s4-mvp3d-hammer-rem-b-repr
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401
from _bootstrap import REPO, guard_gpus

GPU_MAPPING = guard_gpus()

import torch  # noqa: E402
from train import build_data  # noqa: E402

from s4d.baselines.representation import (  # noqa: E402
    BASELINE_ENCODERS,
    FEATURES,
    BaselineEvaluator,
    input_views,
    load_adapter,
)
from s4d.config import get, load_config  # noqa: E402
from s4d.gpu_guard import GPUIsolationError, validate_runtime_path  # noqa: E402
from s4d.run_identity import record_run_identity  # noqa: E402

ARMS = ("pretrained", "random_init")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_identity(cfg: dict, val_loaders: dict, probe_loaders: dict, n_train: int) -> dict:
    """The episode split, windows and probe draw shared with our evaluator for this config."""
    val = next(iter(val_loaders.values())).dataset
    train = next(iter(probe_loaders.values())).dataset
    data = Path(get(cfg, "data.root")) / f"{get(cfg, 'data.task')}.hdf5"
    manifest = data.parent / "splits" / f"{data.stem}_seed{int(get(cfg, 'data.split_seed', 0))}.json"
    heldout = getattr(val, "heldout", None)
    return {
        "data_file": str(data),
        "split_manifest": str(manifest),
        "split_manifest_sha256": sha256(manifest) if manifest.is_file() else None,
        "train_ratio": float(get(cfg, "data.train_ratio", 0.96)),
        "split_seed": int(get(cfg, "data.split_seed", 0)),
        "validation_episodes": list(val.episodes),
        "train_episodes": len(train.episodes),
        "heldout_sets_file": None if heldout is None else heldout.path,
        "val_strides": [int(s) for s in val_loaders],
        "val_windows": {str(s): len(loader.dataset) for s, loader in val_loaders.items()},
        "batch_size": int(get(cfg, "train.batch_size", 16)),
        "probe_batches": int(get(cfg, "eval.probe_batches", 24)),
        "probe_seed": int(get(cfg, "eval.probe_seed", 0)),
        "train_cameras": n_train,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--encoder", nargs="+", required=True, choices=BASELINE_ENCODERS, help="RL encoder config(s)")
    ap.add_argument("--export", required=True, help="pretrained export, the RL runs' vision.export_path")
    ap.add_argument("--init-export", help="random-initialisation export (default: encoder_init.pt beside --export)")
    ap.add_argument("--no-init", action="store_true", help="skip the random-initialisation arm")
    ap.add_argument("--output", required=True, help="output directory (inside the repository or runtime root)")
    ap.add_argument("--config", nargs="*", help="our pretraining config (default: metaworld base + task)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides of our config (data/eval keys)")
    ap.add_argument("--feature", default="stack", choices=sorted(FEATURES))
    ap.add_argument("--no-heldout-sets", action="store_true", help="skip the near/trajectory camera sets")
    ap.add_argument("--max-batches", type=int, help="smoke only: first N validation batches per stride")
    ap.add_argument("--chunk", type=int, default=32, help="RL observations per encoder call (3D-MVP A: 4.7 GB reserved)")
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = ap.parse_args()
    try:
        output = validate_runtime_path(Path(args.output), repository=REPO)
    except GPUIsolationError as exc:
        raise ValueError("outputs must remain inside an authorized runtime root") from exc
    torch.set_num_threads(args.cpu_threads)  # bounded, so concurrent jobs do not oversubscribe the CPUs
    torch.backends.cudnn.benchmark = True  # as at RL time (scripts/train_rl.py)
    device = torch.device("cuda", 0) if args.device == "cuda" else torch.device("cpu")
    paths = args.config or [REPO / "configs/metaworld/base.yaml", REPO / f"configs/metaworld/tasks/{args.task}.yaml"]
    cfg = load_config(paths, args.set)
    if get(cfg, "data.task") != args.task:
        raise ValueError(f"config task {get(cfg, 'data.task')!r} differs from --task {args.task!r}")
    cfg.setdefault("eval", {})["heldout_sets"] = not args.no_heldout_sets
    if args.max_batches is not None:
        cfg["eval"]["max_batches"] = args.max_batches
    exports = {"pretrained": Path(args.export)}
    if not args.no_init:
        default = Path(args.export).with_name("encoder_init.pt")
        exports["random_init"] = Path(args.init_export) if args.init_export else default
    for path in exports.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    identity = record_run_identity(cfg, REPO, output)
    combined = {}
    for encoder in args.encoder:
        for arm, path in exports.items():
            t0 = time.time()
            # Fresh loaders for every evaluation: the shuffled probe loader's generator advances when it is drawn, so
            # every arm fits its probes on the same windows as one scripts/evaluate.py run of our model.
            _, val_loaders, probe_loaders, n_train = build_data(cfg)
            split = split_identity(cfg, val_loaders, probe_loaders, n_train)
            evaluator = BaselineEvaluator(cfg, val_loaders, probe_loaders, device, n_train, args.feature, args.chunk)
            adapter, rl_cfg = load_adapter(encoder, args.task, path)
            adapter.to(device)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if bool(payload.get("random_init", arm == "random_init")) != (arm == "random_init"):
                raise ValueError(f"{path} has random_init={payload.get('random_init')}, expected the {arm} arm")
            print(f"[{encoder}/{arm}] {path} on {split['val_windows']} validation windows; GPUs {GPU_MAPPING}", flush=True)
            summary, num_batches = evaluator(adapter, full=args.max_batches is None)
            provenance = {
                "encoder_config": encoder,
                "encoder_type": rl_cfg["vision"]["encoder_type"],
                "condition": getattr(adapter.backbone, "label", None),
                "observation": adapter.observation or {"views": 1, "depth": False, "cameras": False},
                "input_views": input_views(adapter),
                "feature": args.feature,
                "feature_definition": FEATURES[args.feature] if not adapter.replay_atom_is_stack_feature
                else "the backbone's state of the 3-frame history",
                "replay_atom_shape": list(adapter.replay_atom_shape),
                "export": str(path),
                "export_sha256": sha256(path),
                "export_step": payload.get("step"),
                "random_init": arm == "random_init",
                "git_commit": identity.get("git_commit"),
                "source_dirty": identity.get("source_dirty"),
                "split": split,
                "full": args.max_batches is None,
                "device": str(device),
                "gpus": GPU_MAPPING,
                "torch": torch.__version__,
                "seconds": round(time.time() - t0, 1),
            }
            record = {"arm": arm, "num_batches": num_batches, **summary, "provenance": provenance}
            arm_dir = output / encoder / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            (arm_dir / "summary.json").write_text(json.dumps(record, indent=1))
            combined[f"{encoder}/{arm}"] = record
            print(f"[{encoder}/{arm}] done in {provenance['seconds']} s -> {arm_dir / 'summary.json'}", flush=True)
            del adapter
            if device.type == "cuda":
                torch.cuda.empty_cache()
    (output / "summary.json").write_text(json.dumps(combined, indent=1))


if __name__ == "__main__":
    main()
