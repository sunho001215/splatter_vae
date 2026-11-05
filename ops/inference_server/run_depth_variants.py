"""Controlled depth comparison: the five depth configurations of one episode on identical inputs.

    B0  native RealSense depth registered to each RGB view            (base stage sensor_depth)
    B1  FoundationStereo on each device IR pair -> RGB view            (base stage foundation_stereo_depth)
    E1  MapAnything (3 synchronized RGB views, B0 depth, K, accepted poses) -> per-view depth
    E2  as E1 with the exact B1 depth
    E3  as E2 with B1 after the left-right filter (invalidated pixels only; retained values unchanged)

The base root (``--pipeline`` config, e.g. configs/realworld/mug_hanging_ep1.yaml) must hold the shared stages:
ingest, sensor_depth, foundation_stereo_depth, foundation_stereo_filtered, sam3_masks, calibrate (FROZEN: computed and
accepted once, reused by every variant, never re-fitted per variant) and track2d. Each variant gets its own versioned
root ``<base parent>/<variant>`` whose shared stages are symlinks to the base (identical records and fingerprints, same
image grids), a resolved ``pipeline.yaml`` and only its depth-dependent stages: (mapanything_depth,) lift3d,
track_motion, export, all with the variant's depth. MapAnything settings/checkpoint are identical for E1-E3.

    .venv/bin/python -I ops/inference_server/run_depth_variants.py --pipeline configs/realworld/mug_hanging_ep1.yaml \\
        --server configs/realworld/inference_server.yaml --episode mug-hanging-test-000001 [--variants B0,E1]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

SHARED = ("sensor_depth", "foundation_stereo_depth", "foundation_stereo_filtered", "sam3_masks", "calibrate", "track2d")
VARIANTS = {  # name -> (MapAnything input depth stage or None, depth stage used for lifting/export)
    "B0": (None, "sensor_depth"),
    "B1": (None, "foundation_stereo_depth"),
    "E1": ("sensor_depth", "mapanything_depth"),
    "E2": ("foundation_stereo_depth", "mapanything_depth"),
    "E3": ("foundation_stereo_filtered", "mapanything_depth"),
}


def variant_config(base: dict, name: str, root: Path) -> dict:
    ma_input, depth = VARIANTS[name]
    cfg = copy.deepcopy(base)
    cfg["output_root"] = str(root)
    st = cfg["stages"]
    for stage in ("foundation_stereo_depth", "foundation_stereo_filtered"):  # shared (symlinked), never re-run here
        st.pop(stage, None)
    st["lift3d"] = {**st.get("lift3d", {}), "depth_stage": depth}
    st["export"] = {**st.get("export", {}), "depth_stage": depth}
    if ma_input:
        st["mapanything_depth"] = {**st["mapanything_depth"], "input_depth_stage": ma_input}
    else:
        st.pop("mapanything_depth", None)
    cfg["depth_variant"] = {"name": name, "mapanything_input": ma_input, "depth_stage": depth,
                            "base_root": base["output_root"], "shared_stages": list(SHARED)}
    return cfg


def link_shared(base_root: Path, root: Path, episode: str) -> None:
    from s4d.realworld import io as rio

    src_ep, dst_ep = rio.episode_dir(base_root, episode), rio.episode_dir(root, episode)
    (dst_ep / "derived").mkdir(parents=True, exist_ok=True)
    missing = [s for s in SHARED if rio.read_record(rio.stage_dir(base_root, episode, s)) is None]
    if rio.read_record(src_ep / "normalized") is None or missing:
        raise SystemExit(f"base root lacks complete shared stages: {missing or ['ingest']}")
    for name, src in [("normalized", src_ep / "normalized")] + [(f"derived/{s}", src_ep / "derived" / s) for s in SHARED]:
        dst = dst_ep / name
        if dst.is_symlink():
            if os.readlink(dst) != str(src):
                raise SystemExit(f"{dst} links elsewhere ({os.readlink(dst)})")
            continue
        if dst.exists():
            raise SystemExit(f"{dst} exists and is not the shared symlink; refusing to replace it")
        dst.symlink_to(src)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--skip-weights", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="create roots/configs only")
    args = ap.parse_args(argv)
    from ops.inference_server import run_episode

    base = yaml.safe_load(Path(args.pipeline).read_text())
    base_root = Path(base["output_root"])
    for name in args.variants.split(","):
        if name not in VARIANTS:
            ap.error(f"unknown variant {name}; known {list(VARIANTS)}")
        root = base_root.parent / name
        link_shared(base_root, root, args.episode)
        cfg = variant_config(base, name, root)
        path = root / "pipeline.yaml"
        path.write_text("# resolved by ops/inference_server/run_depth_variants.py from " + args.pipeline + "\n"
                        + yaml.safe_dump(cfg, sort_keys=False))
        stages = (["mapanything_depth"] if VARIANTS[name][0] else []) + ["lift3d", "track_motion", "export"]
        print(json.dumps({"variant": name, "root": str(root), "stages": stages}), flush=True)
        if args.dry_run:
            continue
        rc = run_episode.main(["--pipeline", str(path), "--server", args.server, "--episode", args.episode,
                               "--stages", ",".join(stages)] + (["--skip-weights"] if args.skip_weights else []))
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
