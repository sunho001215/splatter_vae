"""Index the released DROID RLDS TFRecord shards in place (byte spans of every step's JPEGs and state payloads).

Reads each shard once, one episode record in memory at a time; writes only small index files to ``--out`` (never
inside the RLDS root). Resumable and parallelizable: run disjoint ``--range a:b`` slices with ``--no-finalize``, then
once without ``--range`` to finalize (already-indexed shards are skipped). ``--partial`` finalizes over the shards
indexed so far (validation subsets; shard ids stay global). Episode counts are checked against the release's
``dataset_info.json`` ``shardLengths``, which also gives global ordinals.

    python scripts/droid_ls_build_rlds_index.py --rlds-root <droid/1.0.1> --out <index dir> [--range 0:64 --no-finalize]
    python scripts/droid_ls_build_rlds_index.py --rlds-root <droid/1.0.1> --out <index dir> --range 463:465 --partial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s4d.data.droid.largescale.rlds_index import build_index, load_shard_lengths  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlds-root", required=True, help="directory holding droid_101-train.tfrecord-*-of-* shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default="droid_101-train.tfrecord-*-of-*")
    ap.add_argument("--range", default=None, help="a:b shard ids to (re)index")
    ap.add_argument("--no-finalize", action="store_true")
    ap.add_argument("--partial", action="store_true", help="finalize over the shards indexed so far (validation subsets)")
    ap.add_argument("--verify-payload", action="store_true", help="also CRC-check every payload (slow, pure Python)")
    args = ap.parse_args(argv)
    root, out = Path(args.rlds_root).resolve(strict=True), Path(args.out).resolve()
    if out == root or root in out.parents:
        raise SystemExit("the index must not be written inside the read-only RLDS release")
    shards = sorted(root.glob(args.pattern))
    if not shards:
        raise SystemExit(f"no shards matching {args.pattern} in {root}")
    only = range(*map(int, args.range.split(":"))) if args.range else None
    lengths = load_shard_lengths(root)
    if lengths is not None and len(lengths) != len(shards):
        raise SystemExit(f"dataset_info.json lists {len(lengths)} shards but {len(shards)} match {args.pattern}")
    build_index(
        shards,
        out,
        only=only,
        verify_payload=args.verify_payload,
        finalize=not args.no_finalize,
        partial=args.partial,
        shard_lengths=lengths,
    )
    print(f"indexed {len(shards) if only is None else len(only)} shard(s) into {out}")


if __name__ == "__main__":
    main()
