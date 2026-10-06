#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.droid.shards import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine audited pilot reports into the immutable full-job gate."
    )
    parser.add_argument(
        "--pilot-root",
        default="/home/ws/data/droid_stage0_preprocessed/pilot",
    )
    parser.add_argument("--quality-report", default=None)
    parser.add_argument("--benchmark-report", default=None)
    parser.add_argument("--training-profile", default=None)
    parser.add_argument("--teacher-audit", default=None)
    parser.add_argument("--jpeg-quality", type=int, choices=(95, 97), required=True)
    parser.add_argument(
        "--visual-quality-reviewed",
        action="store_true",
        help="Confirm manual inspection of codec/DA3/flow/Lager visual panels.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = Path(args.pilot_root).expanduser().resolve()
    quality = _load(
        Path(args.quality_report or root / "reports" / "quality_audit.json")
    )
    benchmark = _load(
        Path(args.benchmark_report or root / "reports" / "benchmark.json")
    )
    profile = _load(
        Path(
            args.training_profile
            or root / "reports" / "cached_training_profile.json"
        )
    )
    teacher = _load(
        Path(args.teacher_audit or root / "reports" / "teacher_model_audit.json")
    )
    if len(
        {
            quality["dataset_schema_signature"],
            benchmark["pilot_schema_signature"],
            profile["dataset_schema_signature"],
            teacher["pilot_schema_signature"],
        }
    ) != 1:
        raise ValueError("Pilot reports refer to different dataset schemas.")
    pipeline_signatures = {
        quality["pipeline_signature"],
        benchmark["pilot_pipeline_signature"],
        profile["pipeline_signature"],
        teacher["pipeline_signature"],
    }
    if len(pipeline_signatures) != 1:
        raise ValueError("Pilot reports refer to different preprocessing pipelines.")
    if benchmark["pilot_pipeline_signature"] != benchmark["full_pipeline_signature"]:
        raise ValueError(
            "The pilot and exact full manifests use different preprocessing contracts."
        )

    selected = benchmark["jpeg"]["quality_comparison"]["by_kind"]
    jpeg_metrics_pass = True
    for kind in ("rgb", "lagernvs"):
        values = selected[kind][f"q{int(args.jpeg_quality)}"]
        jpeg_metrics_pass &= (
            float(values["psnr_db"]) >= 35.0
            and float(values["ssim"]) >= 0.97
            and float(values["edge_psnr_db"]) >= 30.0
        )
    storage = benchmark["storage_projection"]
    teacher_gates = teacher["gates"]
    quality_gates = quality["gates"]
    gates = {
        "data_selection": bool(quality_gates["data_selection"]),
        "jpeg": bool(jpeg_metrics_pass and args.visual_quality_reviewed),
        "da3_metric_geometry": bool(
            quality_gates["da3_metric_geometry"]
            and teacher_gates["da3_two_view_metric_path"]
        ),
        "megaflow_quality": bool(
            quality_gates["megaflow_quality"]
            and teacher_gates["megaflow_identical_frame"]
            and teacher_gates["megaflow_real_gap6"]
        ),
        "lagernvs_four_targets": bool(
            quality_gates["lagernvs_four_targets"]
            and teacher_gates["lagernvs_amortized_four_target"]
            and args.visual_quality_reviewed
        ),
        "storage_capacity": bool(storage["capacity_pass"]),
        "cached_loader": bool(
            float(benchmark["loader"]["samples_per_second"]) > 0.0
            and quality["integrity"]["loader"]["windows_checked"] > 0
        ),
        "model_forward_backward": bool(
            profile["all_losses_finite"]
            and profile["all_gradients_finite"]
            and not profile["foundation_model_modules_loaded"]
        ),
    }
    report = {
        "schema_version": 1,
        "pilot_schema_signature": quality["dataset_schema_signature"],
        "pipeline_signature": quality["pipeline_signature"],
        "gates": gates,
        "passed": all(gates.values()),
        "jpeg_quality": int(args.jpeg_quality),
        "jpeg_subsampling": benchmark["jpeg"]["subsampling"],
        "visual_quality_reviewed": bool(args.visual_quality_reviewed),
        "safe_required_bytes": int(storage["safe_required_bytes"]),
        "projected_final_bytes": int(storage["projected_final_bytes"]),
        "filesystem_free_bytes": int(storage["filesystem_free_bytes"]),
        "reports": {
            "quality": str(
                Path(
                    args.quality_report
                    or root / "reports" / "quality_audit.json"
                ).resolve()
            ),
            "benchmark": str(
                Path(
                    args.benchmark_report or root / "reports" / "benchmark.json"
                ).resolve()
            ),
            "training_profile": str(
                Path(
                    args.training_profile
                    or root / "reports" / "cached_training_profile.json"
                ).resolve()
            ),
            "teacher_audit": str(
                Path(
                    args.teacher_audit or root / "reports" / "teacher_model_audit.json"
                ).resolve()
            ),
        },
    }
    output = Path(
        args.output or root / "reports" / "pilot_gate.json"
    ).expanduser().resolve()
    write_json_atomic(output, report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        failed = [name for name, value in gates.items() if not value]
        raise SystemExit(f"Pilot gate remains closed: {failed}")


if __name__ == "__main__":
    main()
