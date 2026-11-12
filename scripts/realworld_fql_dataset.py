"""Build an offline-FQL dataset: frozen baseline features once per observation time + transitions (actions, rewards,
masks) from the read-only raw recordings (s4d/rwlearn/fql_data.py, docs/realworld_fql.md). No training, no robot.
For ``--condition e2e_cnn`` (no ``--export``): the transitions and frame indices only; the trainer reads the frames
from the store.

    CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/realworld_fql_dataset.py build --store <store.yaml> \
        --condition dynamo --export runs/pretrain/rw-dynamo-drawer-open-pre/encoder.pt \
        --out runs/realworld_fql/data/dynamo-drawer-open.h5 --gate-report <gate_report.json> \
        --outcomes <outcomes.json>

The gate is checked at the condition's level (geometry-dependent conditions also need the calibration hold released
and a store built from the validated calibration; s4d/rwlearn/gate.py).

Every store episode is used (train and validation; the split manifest selects at training time). Without a passed gate
report only ``--smoke`` builds of at most 3 episodes are allowed (tests / plumbing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()

from ops.resources.gpu_guard import validate_runtime_path  # noqa: E402
from s4d.rwlearn.baseline_data import check_gate  # noqa: E402
from s4d.rwlearn.fql_data import CONDITIONS, build_fql_dataset  # noqa: E402
from s4d.rwlearn.gate import FQL_CONSUMER  # noqa: E402
from s4d.rwlearn.store import open_store  # noqa: E402


def mapped_raw_recordings(store, episodes, mapping: str) -> dict:
    """{episode: local raw path} for ``RECORDED_ROOT=LOCAL_ROOT``: the recorded path must lie below RECORDED_ROOT, the
    mapped file must be a regular file (no symlink) inside LOCAL_ROOT, and its sha256 must equal the episode's recorded
    raw identity (the ingestion manifest's sha256, written into the training store and bound by the gate report)."""
    import hashlib

    recorded, sep, local = mapping.partition("=")
    if not sep or not recorded.startswith("/") or not local.startswith("/"):
        raise SystemExit("--raw-root must be RECORDED_ROOT=LOCAL_ROOT (absolute paths)")
    recorded_root, local_root = Path(recorded), Path(local)
    if any(p.is_symlink() for p in (local_root, *local_root.parents)) or local_root.resolve(strict=True) != local_root:
        raise SystemExit(f"--raw-root local root {local_root} must be a real directory path without symlinks")
    out = {}
    for ep in episodes:
        source = Path(store.raw_recording(ep))
        try:
            relative = source.relative_to(recorded_root)
        except ValueError:
            raise SystemExit(f"{ep}: raw recording {source} is not below {recorded_root}") from None
        if ".." in relative.parts:
            raise SystemExit(f"{ep}: unsafe raw recording path {source}")
        path = local_root / relative
        if path.is_symlink() or not path.is_file() or local_root not in path.resolve(strict=True).parents:
            raise SystemExit(f"{ep}: mapped raw recording {path} is not a regular file inside {local_root}")
        want = str(getattr(store, "raw_sha256", lambda _ep: "")(ep) or "")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 2**20), b""):
                digest.update(block)
        if not want or digest.hexdigest() != want:
            raise SystemExit(f"{ep}: mapped raw recording {path} sha256 {digest.hexdigest()} != recorded {want or None}")
        out[ep] = str(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--store", required=True)
    b.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    b.add_argument("--export", help="frozen encoder export (every condition except e2e_cnn)")
    b.add_argument("--out", required=True)
    b.add_argument("--gate-report")
    b.add_argument("--reward", default="ogbench_final_success")
    b.add_argument("--device", default="cuda")
    b.add_argument("--smoke", action="store_true")
    b.add_argument("--episodes", nargs="*")
    b.add_argument("--outcomes", help="rw-outcomes document (scripts/realworld_learning.py inputs) or JSON "
                                     "{episode: true|false} (true = successful demonstration)")
    b.add_argument("--assume-final-success", action="store_true",
                   help="treat every recording as a successful demonstration (no outcome marker in the files)")
    b.add_argument("--raw-root", metavar="RECORDED_ROOT=LOCAL_ROOT",
                   help="read each episode's raw recording (actions) from LOCAL_ROOT/<path below RECORDED_ROOT> instead "
                        "of the recorded path; every mapped file must be a regular file whose sha256 equals the "
                        "episode's recorded raw identity (fail closed). Local dataset construction (ops/jobs/"
                        "realworld_local.py) maps the remote copies to the read-only ingested originals")
    args = ap.parse_args()

    store = open_store(args.store)
    episodes = args.episodes or store.episodes()
    if args.smoke:
        if len(episodes) > 3:
            raise SystemExit("--smoke builds use at most 3 episodes")
    else:
        if args.episodes:
            raise SystemExit("a full build uses every store episode (the gate covers the whole store)")
        check_gate(store, json.loads(Path(args.gate_report).read_text()) if args.gate_report else None, steps=None,
                   split=None, consumer=FQL_CONSUMER[args.condition])
    out = validate_runtime_path(Path(args.out) if Path(args.out).is_absolute() else REPO / args.out, repository=REPO)
    out.parent.mkdir(parents=True, exist_ok=True)
    if (CONDITIONS[args.condition].encoder_type is None) != (args.export is None):
        raise SystemExit("--export is required for frozen-encoder conditions and refused for e2e_cnn")
    raw_paths = mapped_raw_recordings(store, episodes, args.raw_root) if args.raw_root else None
    print(json.dumps(build_fql_dataset(store, args.condition, args.export, out, episodes=episodes,
                                       reward=args.reward, device=args.device, raw_paths=raw_paths,
                                       outcomes=json.loads(Path(args.outcomes).read_text()) if args.outcomes else None,
                                       assume_final_success=args.assume_final_success), indent=1))


if __name__ == "__main__":
    main()
