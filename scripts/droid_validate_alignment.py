"""Recompute real-sample alignment and retain local plus optional W&B evidence.

Invoke with python -I. W&B defaults to offline unless WANDB_MODE is supplied.
Read-only input links let the existing overlay utilities render inside the new
repository without modifying downloaded data, cache arrays, or training code.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()

from s4d.data.droid.convert import sample_paths, sha256  # noqa: E402
from s4d.data.droid.inspect import (  # noqa: E402
    alignment_overlays,
    cached_coordinate_overlays,
    dense_sanity_overlays,
    inspection_report,
)
from s4d.diag.wandb_log import init_wandb  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Validate DROID alignment with local and optional W&B evidence")
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO / "docs/droid_validation")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    sample, cache, output = (
        args.sample_root.resolve(strict=True),
        args.cache_root.resolve(strict=True),
        args.output_root.resolve(),
    )
    if not output.is_relative_to(REPO) or output == REPO:
        raise ValueError("alignment outputs must be in a subdirectory of the new repository")
    output.mkdir(parents=True, exist_ok=True)
    overlay_output = output / "overlays"
    overlay_output.mkdir(exist_ok=True)
    inputs = ("manifest.json", "matched_raw_external.npz", "scratch.h5", "dinov2.h5")
    before = {name: sha256(cache / name) for name in inputs}
    native_before = {name: sha256(path) for name, path in sample_paths(sample).items()}
    dense_evidence = []
    report = inspection_report(sample, cache)
    # Overlay functions write relative to their cache argument. Stage only links
    # to read-only inputs, so all newly rendered outputs remain inside this repo.
    with tempfile.TemporaryDirectory(prefix="droid_alignment_", dir=REPO / ".cache") as staging:
        staged = Path(staging)
        for name in inputs:
            (staged / name).symlink_to(cache / name)
        rendered = (
            alignment_overlays(sample, staged) + cached_coordinate_overlays(staged) + dense_sanity_overlays(sample, staged)
        )
        overlays = []
        for source in rendered:
            destination = overlay_output / source.name
            shutil.copyfile(source, destination)
            overlays.append(destination)
        for name in ("dense_depth_report.json", "fused_gt_cloud.npy"):
            destination = output / name
            shutil.copyfile(staged / "dense_sanity" / name, destination)
            dense_evidence.append(destination)
    report["overlays"] = [str(path.relative_to(REPO)) for path in overlays]
    report_path = output / "alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    evidence = [report_path, *dense_evidence]
    for name in (
        "coordinate_correction.json",
        "adapter_verification.json",
        "loader_validation_workers0.json",
        "loader_validation_workers2.json",
        "manifest.json",
    ):
        destination = output / name
        shutil.copyfile(cache / name, destination)
        evidence.append(destination)
    after = {name: sha256(cache / name) for name in inputs}
    native_after = {name: sha256(path) for name, path in sample_paths(sample).items()}
    if before != after or native_before != native_after:
        raise RuntimeError("alignment validation unexpectedly modified an input data file")
    scalars = {
        "alignment/dense_cross_camera_median_relative_error": report["dense_cross_camera"]["median_relative_error"],
        "alignment/dense_cross_camera_unfiltered_relative_error": report["dense_cross_camera"][
            "unfiltered_median_relative_error"
        ],
        "alignment/scene_depth_median_m": report["scene_depth"]["median_residual_m"],
        "alignment/cross_camera_median_relative_error": report["cross_camera"]["median_relative_error"],
        "alignment/cross_camera_unfiltered_relative_error": report["cross_camera"]["unfiltered_median_relative_error"],
        "alignment/gripper_heldout_median_m": report["gripper"]["heldout_median_residual_m"],
        "alignment/gripper_fit_median_m": report["gripper"]["median_residual_m"],
        "alignment/match_rate_selected_sample": report["matching"]["rate"],
        "alignment/raw_pose_max_error_m": report["state_alignment"]["xyz_max_m"],
        "alignment/raw_rotation_max_error_rad": report["state_alignment"]["euler_xyz_rotation_max_rad"],
        "alignment/raw_gripper_max_error": report["state_alignment"]["gripper_direct_max"],
        "alignment/initial_rgb_exact_index_matches": sum(
            row["expected_raw_index"] == row["best_local_raw_index"] for row in report["initial_rgb_alignment"]
        ),
        **{f"acceptance/{key}": int(value) for key, value in report["acceptance"].items()},
    }
    summary = {
        "episode": report["episode"],
        "scope": "one real matched episode; disjoint intra-episode validation clip",
        "metrics": scalars,
        "acceptance": report["acceptance"],
        "gpu_mapping": GPU_MAPPING,
        "input_cache_unchanged": before == after,
        "native_sample_unchanged": native_before == native_after,
        "input_sha256": after,
        "local_evidence": [str(path.relative_to(REPO)) for path in evidence + overlays],
        "wandb": {
            "enabled": not args.no_wandb,
            "mode": "disabled" if args.no_wandb else os.environ.get("WANDB_MODE", "offline"),
            "run_url": None,
            "offline_path": None,
            "finished": False,
        },
    }
    os.environ.setdefault("WANDB_MODE", "offline")
    for variable, leaf in (
        ("WANDB_CACHE_DIR", "wandb_cache"),
        ("WANDB_CONFIG_DIR", "wandb_config"),
        ("WANDB_DATA_DIR", "wandb_data"),
    ):
        directory = REPO / ".cache" / leaf
        directory.mkdir(exist_ok=True)
        os.environ[variable] = str(directory)
    os.environ["WANDB_DIR"] = str(output)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    run = init_wandb(
        {"episode": report["episode"], "scope": summary["scope"], "source_sha256": report["source_sha256"]},
        "droid-validate-alignment",
        "splatter4d-droid",
        not args.no_wandb,
        output,
    )
    if run is not None:
        import wandb

        summary["wandb"].update(run_id=run.id, run_directory=str(Path(run.dir).relative_to(REPO)))
        if summary["wandb"]["mode"] == "offline":
            summary["wandb"]["offline_path"] = str(Path(run.dir).parent.relative_to(REPO))
        else:
            summary["wandb"]["run_url"] = run.url
        try:
            rows = [[key, value] for key, value in scalars.items()]
            run.log(
                {
                    **scalars,
                    "alignment/measurements": wandb.Table(columns=["measurement", "value"], data=rows),
                    **{f"alignment/{path.stem}": wandb.Image(str(path), caption=path.stem) for path in overlays},
                },
                step=0,
            )
            artifact = wandb.Artifact("droid-alignment-evidence", type="validation")
            for path in evidence + overlays:
                artifact.add_file(str(path), name=str(path.relative_to(output)))
            run.log_artifact(artifact)
            run.summary.update(scalars)
        finally:
            run.finish()
        summary["wandb"]["finished"] = True
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    if not all(report["acceptance"].values()):
        raise SystemExit("one or more alignment acceptance checks failed; evidence retained")


if __name__ == "__main__":
    main()
