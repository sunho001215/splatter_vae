#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dataset.droid.calibration import load_calibration_manifest
from dataset.droid.codecs import JPEGCodecConfig, NumericCodecConfig
from dataset.droid.integrity import (
    IntegrityConfig,
    classify_partial_artifacts,
    verify_stage0_dataset,
    write_final_manifest,
)
from dataset.droid.preprocessed_manifest import (
    DEFAULT_LAGER_SCENE_CENTER,
    build_stage0_manifest,
    load_stage0_manifest,
    shard_path,
)
from dataset.droid.records import lager_record_contract
from dataset.droid.safety import (
    assert_no_preprocessing_quality_hold,
    assert_source_fingerprint_unchanged,
    source_tree_fingerprint,
    validate_derived_root,
)
from dataset.droid.shards import shard_is_complete, write_json_atomic
from preprocessing.lagernvs.pose import pose_sampler_contract
from preprocessing.stage0.workflow import (
    STAGES,
    Stage0WorkerConfig,
    environment_metadata,
    run_stage_worker,
)

AUTHORIZED_GPUS = (
    "GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce",
    "GPU-d09f0338-71b9-d915-3c7f-e99754a3b639",
    "GPU-6b33eef2-80c5-547e-6a61-7ab81c14d63b",
)
DEFAULT_ROOT = "/home/ws/data/droid_stage0_preprocessed"
DEFAULT_CALIBRATION = (
    "/ws/data/ws/droid_splattervae/manifests/canonical-full/calibration.jsonl.gz"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parse_nvidia_smi_row(value: str) -> dict[str, float | str]:
    fields = [item.strip() for item in value.strip().split(",")]
    if len(fields) != 5:
        raise ValueError(f"Unexpected nvidia-smi telemetry row: {value!r}")
    return {
        "uuid": fields[0],
        "gpu_utilization_percent": float(fields[1]),
        "memory_utilization_percent": float(fields[2]),
        "memory_used_mib": float(fields[3]),
        "memory_total_mib": float(fields[4]),
    }


class _GPUTelemetry:
    """Persist low-frequency read-only metrics for only the authorized GPUs."""

    def __init__(self, root: Path, attempt: int, interval_seconds: float) -> None:
        if float(interval_seconds) <= 0.0:
            raise ValueError("GPU telemetry interval must be positive.")
        self.interval_seconds = float(interval_seconds)
        self.path = (
            root
            / "logs"
            / "preprocessing"
            / f"gpu-telemetry-attempt-{int(attempt):02d}.jsonl"
        )
        self.summary_path = (
            root / "reports" / f"gpu-telemetry-attempt-{int(attempt):02d}.json"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="droid-stage0-gpu-telemetry",
            daemon=True,
        )
        self._stage = "startup"
        self._started = time.time()
        self._statistics: dict[tuple[str, str], dict[str, float]] = {}
        self._errors = 0

    def set_stage(self, stage: str) -> None:
        self._stage = str(stage)

    def start(self) -> None:
        self._thread.start()

    def _record(self, payload: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _sample(self) -> None:
        for uuid in AUTHORIZED_GPUS:
            command = [
                "nvidia-smi",
                f"--id={uuid}",
                "--query-gpu=uuid,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                values = _parse_nvidia_smi_row(result.stdout)
                if values["uuid"] != uuid:
                    raise ValueError(
                        f"nvidia-smi returned {values['uuid']} for requested {uuid}."
                    )
                stage = self._stage
                row = {"time_unix": time.time(), "stage": stage, **values}
                self._record(row)
                key = (stage, uuid)
                stats = self._statistics.setdefault(
                    key,
                    {
                        "samples": 0.0,
                        "gpu_utilization_sum": 0.0,
                        "memory_utilization_sum": 0.0,
                        "memory_used_mib_peak": 0.0,
                        "memory_total_mib": float(values["memory_total_mib"]),
                    },
                )
                stats["samples"] += 1.0
                stats["gpu_utilization_sum"] += float(values["gpu_utilization_percent"])
                stats["memory_utilization_sum"] += float(
                    values["memory_utilization_percent"]
                )
                stats["memory_used_mib_peak"] = max(
                    stats["memory_used_mib_peak"], float(values["memory_used_mib"])
                )
            except Exception as error:  # noqa: BLE001 - telemetry is diagnostic.
                self._errors += 1
                self._record(
                    {
                        "time_unix": time.time(),
                        "stage": self._stage,
                        "uuid": uuid,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            self._errors += 1
        by_stage: dict[str, dict[str, Any]] = {}
        for (stage, uuid), values in sorted(self._statistics.items()):
            count = max(1.0, values["samples"])
            by_stage.setdefault(stage, {})[uuid] = {
                "samples": int(values["samples"]),
                "gpu_utilization_percent_mean": values["gpu_utilization_sum"] / count,
                "memory_utilization_percent_mean": values["memory_utilization_sum"]
                / count,
                "memory_used_mib_peak": values["memory_used_mib_peak"],
                "memory_total_mib": values["memory_total_mib"],
            }
        summary = {
            "schema_version": 1,
            "authorized_gpu_uuids": list(AUTHORIZED_GPUS),
            "sampling_interval_seconds": self.interval_seconds,
            "started_unix": self._started,
            "finished_unix": time.time(),
            "telemetry_errors": self._errors,
            "jsonl_path": str(self.path),
            "by_stage": by_stage,
        }
        write_json_atomic(self.summary_path, summary)
        return {**summary, "summary_path": str(self.summary_path)}


def _read_episode_ids(path: str | None) -> list[str] | None:
    if path is None:
        return None
    values = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as stream:
        for line in stream:
            value = line.strip()
            if value and not value.startswith("#"):
                values.append(value)
    if not values:
        raise ValueError("Episode-id file is empty.")
    return values


def select_representative_pilot(
    calibration_manifest: str,
    *,
    count: int,
) -> list[str]:
    """Stratify by lab, camera pair, split, episode length, and baseline."""

    valid = [
        dict(entry)
        for entry in load_calibration_manifest(calibration_manifest)
        if bool(entry.get("valid"))
    ]
    if int(count) <= 0 or int(count) > len(valid):
        raise ValueError("Pilot episode count is outside the valid manifest size.")
    groups: dict[tuple[str, tuple[str, str]], list[dict[str, Any]]] = defaultdict(list)
    for entry in valid:
        lab = str(entry.get("rlds_path", "unknown")).split("/", 1)[0]
        serials = tuple(str(camera["serial"]) for camera in entry["exterior_cameras"])
        groups[(lab, serials)].append(entry)
    for values in groups.values():
        values.sort(
            key=lambda item: (
                int(item["num_steps"]),
                float(item.get("exterior_baseline_m", 0.0)),
                str(item["episode_id"]),
            )
        )
    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    # Avoid pathological one-step records while still spanning short, long,
    # median, and interquartile demonstrations across camera/lab strata.
    quantiles = (0.05, 0.95, 0.50, 0.25, 0.75)
    round_index = 0
    while len(chosen) < int(count):
        made_progress = False
        for group_index, (_key, values) in enumerate(ordered_groups):
            if len(chosen) >= int(count):
                break
            quantile = quantiles[
                (round_index * len(ordered_groups) + group_index) % len(quantiles)
            ]
            position = round(quantile * (len(values) - 1))
            candidates = values[position:] + values[:position]
            entry = next(
                (value for value in candidates if str(value["episode_id"]) not in used),
                None,
            )
            if entry is None:
                continue
            used.add(str(entry["episode_id"]))
            chosen.append(entry)
            made_progress = True
        if not made_progress:
            break
        round_index += 1
    if len(chosen) != int(count):
        raise RuntimeError(
            f"Pilot selector found only {len(chosen)} of {count} episodes."
        )
    # Ensure the tiny canonical validation split is represented with a typical,
    # rather than pathological shortest, validation trajectory.
    if not any(value.get("dataset_split") == "validation" for value in chosen):
        validation = sorted(
            (
                value
                for value in valid
                if value.get("dataset_split") == "validation"
                and str(value["episode_id"]) not in used
            ),
            key=lambda value: (int(value["num_steps"]), str(value["episode_id"])),
        )
        replacement = validation[len(validation) // 2]
        chosen[-1] = replacement
    return [str(value["episode_id"]) for value in chosen]


def _manifest_command(args: argparse.Namespace) -> None:
    root = validate_derived_root(args.root, args.droid_root)
    ids = _read_episode_ids(args.episode_ids)
    if args.mode == "pilot" and ids is None:
        ids = select_representative_pilot(
            args.calibration_manifest, count=int(args.pilot_episodes)
        )
        selection_path = root / "manifests" / "pilot_episode_ids.txt"
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = selection_path.with_suffix(".txt.partial")
        temporary.write_text("\n".join(ids) + "\n", encoding="utf-8")
        os.replace(temporary, selection_path)
    manifest = build_stage0_manifest(
        args.calibration_manifest,
        root,
        droid_root=args.droid_root,
        target_retained_per_shard=int(args.target_retained_per_shard),
        episode_ids=ids,
        mode=args.mode,
        jpeg_quality=int(args.jpeg_quality),
        jpeg_subsampling=str(args.jpeg_subsampling),
        numeric_compression_level=int(args.zstd_level),
    )
    print(
        json.dumps(
            {
                "root": str(root),
                "schema_signature": manifest["schema_signature"],
                "counts": manifest["counts"],
                "theoretical_uncompressed_bytes": manifest[
                    "theoretical_uncompressed_bytes"
                ],
                "shard_count": len(manifest["shards"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _worker_command(args: argparse.Namespace) -> None:
    result = run_stage_worker(
        Stage0WorkerConfig(
            root=args.root,
            stage=args.stage,
            worker_id=int(args.worker_id),
            worker_count=int(args.worker_count),
            droid_root=args.droid_root,
            jpeg=JPEGCodecConfig(
                quality=int(args.jpeg_quality),
                subsampling=args.jpeg_subsampling,
                optimize=False,
                progressive=False,
            ),
            numeric=NumericCodecConfig(compression_level=int(args.zstd_level)),
            model_cache=args.model_cache,
            scene_center=tuple(float(value) for value in args.scene_center),
            codec_audit_images=int(args.codec_audit_images),
        )
    )
    print(json.dumps(result, sort_keys=True), flush=True)


def _environment_command(args: argparse.Namespace) -> None:
    payload = {"stage_environment": str(args.stage), **environment_metadata()}
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "stage_environment": args.stage,
                "output": str(Path(args.output).expanduser().resolve()),
                "resolved_package_count": len(payload["resolved_packages"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _python_for_stage(args: argparse.Namespace, stage: str) -> str:
    values = {
        "rgb": args.python_training,
        "da3": args.python_da3,
        "megaflow": args.python_megaflow,
        "lagernvs": args.python_lagernvs,
        "compose": args.python_training,
    }
    # Do not resolve this symlink: invoking a uv virtualenv through its resolved
    # base-interpreter target bypasses the virtualenv's pyvenv.cfg and packages.
    path = Path(values[stage]).expanduser().absolute()
    if not path.is_file():
        raise FileNotFoundError(
            f"Python environment for {stage} does not exist: {path}. "
            "Run scripts/create_droid_preprocessing_envs.sh first."
        )
    return str(path)


def _record_environment_inventories(args: argparse.Namespace, root: Path) -> None:
    interpreters = {
        "training": str(Path(args.python_training).expanduser().absolute()),
        "da3": _python_for_stage(args, "da3"),
        "megaflow": _python_for_stage(args, "megaflow"),
        "lagernvs": _python_for_stage(args, "lagernvs"),
    }
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for name, interpreter in interpreters.items():
        path = Path(interpreter)
        if not path.is_file():
            raise FileNotFoundError(f"Python environment for {name} is missing: {path}")
        output = root / "metadata" / "environments" / f"{name}.json"
        subprocess.run(
            [
                str(path),
                str(Path(__file__).resolve()),
                "environment",
                "--stage",
                name,
                "--output",
                str(output),
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
        )


def _check_full_gate(args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
    assert_no_preprocessing_quality_hold(args.root)
    if manifest["mode"] != "full" or args.allow_missing_pilot_gate:
        return
    gate_path = Path(args.pilot_gate).expanduser().resolve()
    if not gate_path.is_file():
        raise FileNotFoundError(
            f"Full preprocessing is gated on a passing pilot report: {gate_path}"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    required = (
        "data_selection",
        "jpeg",
        "da3_metric_geometry",
        "megaflow_quality",
        "lagernvs_four_targets",
        "storage_capacity",
        "cached_loader",
        "model_forward_backward",
    )
    failed = [name for name in required if gate.get("gates", {}).get(name) is not True]
    if failed:
        raise RuntimeError(f"Full preprocessing pilot gates have not passed: {failed}")
    if int(gate.get("jpeg_quality", -1)) != int(args.jpeg_quality):
        raise RuntimeError(
            "Full JPEG quality differs from the validated pilot decision."
        )
    if gate.get("pipeline_signature") != manifest.get("pipeline_signature"):
        raise RuntimeError(
            "Full preprocessing contract differs from the audited pilot pipeline."
        )
    required_bytes = int(gate.get("safe_required_bytes", -1))
    available = os.statvfs(args.root)
    free_bytes = int(available.f_bavail * available.f_frsize)
    if required_bytes <= 0 or free_bytes < required_bytes:
        raise RuntimeError(
            f"Storage gate failed: free={free_bytes}, safe required={required_bytes}."
        )


def _check_lagernvs_pose_audit(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    dataset_signature = str(manifest["schema_signature"])
    pose_contract = pose_sampler_contract()
    pose_signature = str(pose_contract["signature"])
    report_root = root / "reports" / "lagernvs_pose_safety"
    skipped_completed = []
    audited = []
    totals = {
        "retained_timestamps": 0,
        "targets": 0,
        "fallback_targets": 0,
        "exceptional_translation_targets": 0,
        "exceptional_safety_threshold_targets": 0,
    }
    errors = []
    for shard in manifest["shards"]:
        shard_id = int(shard["shard_id"])
        if shard_is_complete(
            shard_path(root, "lagernvs", shard_id),
            schema_signature=dataset_signature,
            verify_checksums=True,
        ):
            skipped_completed.append(shard_id)
            continue
        report_path = report_root / f"shard-{shard_id:05d}.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"shard {shard_id}: missing/unreadable report ({error})")
            continue
        if (
            report.get("status") != "pass"
            or int(report.get("pose_audit_schema_version", -1)) != 2
            or report.get("schema_signature") != dataset_signature
            or report.get("pose_contract_signature") != pose_signature
            or int(report.get("shard_id", -1)) != shard_id
        ):
            errors.append(f"shard {shard_id}: stale or failed pose-audit report")
            continue
        statistics = report.get("statistics", {})
        expected_timestamps = int(shard["retained_count"])
        if (
            int(statistics.get("retained_timestamps", -1)) != expected_timestamps
            or int(statistics.get("targets", -1)) != 4 * expected_timestamps
        ):
            errors.append(f"shard {shard_id}: incomplete pose-audit counts")
            continue
        for name in totals:
            totals[name] += int(statistics.get(name, 0))
        audited.append(shard_id)
    if errors:
        raise RuntimeError(
            "LagerNVS launch is gated on current all-shard pose audits; "
            f"{len(errors)} shard(s) failed, examples={errors[:10]}."
        )
    expected = len(manifest["shards"])
    if len(audited) + len(skipped_completed) != expected:
        raise RuntimeError("LagerNVS pose-audit accounting is incomplete.")
    return {
        "status": "pass",
        "pose_audit_schema_version": 2,
        "dataset_schema_signature": dataset_signature,
        "pose_contract_signature": pose_signature,
        "pose_sampler_contract": pose_contract,
        "expected_shards": expected,
        "audited_shards": len(audited),
        "skipped_previously_completed_lager_shards": len(skipped_completed),
        "statistics": totals,
        "verified_unix": time.time(),
    }


def _stage_worker_command(
    args: argparse.Namespace, stage: str, worker_id: int, worker_count: int
) -> list[str]:
    return [
        _python_for_stage(args, stage),
        str(Path(__file__).resolve()),
        "worker",
        "--root",
        str(Path(args.root).expanduser().resolve()),
        "--droid-root",
        str(Path(args.droid_root).expanduser().resolve()),
        "--stage",
        stage,
        "--worker-id",
        str(worker_id),
        "--worker-count",
        str(worker_count),
        "--jpeg-quality",
        str(args.jpeg_quality),
        "--jpeg-subsampling",
        args.jpeg_subsampling,
        "--zstd-level",
        str(args.zstd_level),
        "--model-cache",
        str(Path(args.model_cache).expanduser().resolve()),
        "--scene-center",
        *(str(value) for value in args.scene_center),
        "--codec-audit-images",
        str(args.codec_audit_images),
    ]


def _run_one_stage(args: argparse.Namespace, stage: str, worker_count: int) -> None:
    root = Path(args.root).expanduser().resolve()
    logs = root / "logs" / "preprocessing"
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen[bytes], Any, Path]] = []
    state: list[dict[str, Any]] = []
    for worker_id in range(worker_count):
        command = _stage_worker_command(args, stage, worker_id, worker_count)
        log_path = logs / f"{stage}-worker-{worker_id:02d}.log"
        stream = log_path.open("ab", buffering=0)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
        environment["CUDA_VISIBLE_DEVICES"] = AUTHORIZED_GPUS[worker_id]
        environment["TOKENIZERS_PARALLELISM"] = "false"
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, stream, log_path))
        state.append(
            {
                "stage": stage,
                "worker_id": worker_id,
                "pid": process.pid,
                "start_time_unix": time.time(),
                "gpu": AUTHORIZED_GPUS[worker_id],
                "command": command,
                "log": str(log_path),
                "exit_code": None,
            }
        )
    state_path = root / "progress" / f"orchestrator-{stage}.json"
    write_json_atomic(state_path, {"workers": state})
    failed = False
    unfinished = set(range(len(processes)))
    while unfinished:
        for index in tuple(unfinished):
            process, stream, _log_path = processes[index]
            code = process.poll()
            if code is None:
                continue
            stream.close()
            state[index]["exit_code"] = int(code)
            state[index]["end_time_unix"] = time.time()
            failed |= code != 0
            unfinished.remove(index)
            # Record a worker failure promptly even when another worker is
            # still finishing valid, atomic shards assigned to another GPU.
            write_json_atomic(state_path, {"workers": state})
        if unfinished:
            time.sleep(1.0)
    if failed:
        failures = [item for item in state if item["exit_code"] != 0]
        raise RuntimeError(f"Preprocessing stage {stage} failed: {failures}")


def _run_command(args: argparse.Namespace) -> None:
    root = validate_derived_root(args.root, args.droid_root)
    validate_derived_root(args.model_cache, args.droid_root)
    manifest = load_stage0_manifest(root)
    _check_full_gate(args, manifest)
    stages = tuple(args.stages)
    invalid = set(stages) - set(STAGES)
    if invalid:
        raise ValueError(f"Unknown preprocessing stages: {sorted(invalid)}")
    worker_count = int(args.workers)
    if not 1 <= worker_count <= len(AUTHORIZED_GPUS):
        raise ValueError("Preprocessing supports one to three authorized GPU workers.")
    pose_audit = None
    if args.require_lagernvs_pose_audit and "lagernvs" in stages:
        pose_audit = _check_lagernvs_pose_audit(root, manifest)
    plan = {
        "root": str(root),
        "mode": manifest["mode"],
        "schema_signature": manifest["schema_signature"],
        "counts": manifest["counts"],
        "shards": len(manifest["shards"]),
        "stages": list(stages),
        "workers": worker_count,
        "gpus": list(AUTHORIZED_GPUS[:worker_count]),
        "lagernvs_pose_audit": pose_audit,
        "commands": {
            stage: [
                _stage_worker_command(args, stage, worker, worker_count)
                for worker in range(worker_count)
            ]
            for stage in stages
        },
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    expected_visible = list(AUTHORIZED_GPUS[:worker_count])
    actual_visible = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if actual_visible != expected_visible:
        raise RuntimeError(
            "Full preprocessing must expose the authorized GPU UUIDs in worker "
            f"order: expected={expected_visible}, actual={actual_visible}."
        )
    source_before = source_tree_fingerprint(args.droid_root)
    historical_source_path = root / "metadata" / "source-tree-before-final-lager.json"
    if historical_source_path.is_file():
        historical_source = json.loads(
            historical_source_path.read_text(encoding="utf-8")
        )
        assert_source_fingerprint_unchanged(historical_source, source_before)
    write_json_atomic(
        root / "metadata" / "source-tree-before-full-run.json", source_before
    )
    _record_environment_inventories(args, root)
    progress_root = root / "progress"
    write_json_atomic(
        root / "metadata" / "lagernvs-pose-safety-contract.json",
        {
            **pose_sampler_contract(),
            "dataset_schema_signature": str(manifest["schema_signature"]),
            "lager_record_contract": lager_record_contract(),
            "maximum_manifest_theoretical_metadata_adjustment_bytes": (
                int(manifest["counts"]["retained_timesteps"])
                * int(
                    lager_record_contract()[
                        "additional_uncompressed_bytes_per_timestamp_vs_v2"
                    ]
                )
            ),
            "manifest_contract_note": (
                "The signed manifest retains the pilot-approved ordinary limits; "
                "this sidecar records bounded exceptional tiers used only after "
                "ordinary pose candidates are exhausted."
            ),
        },
    )
    if pose_audit is not None:
        write_json_atomic(
            root / "metadata" / "lagernvs-pose-safety-audit.json", pose_audit
        )
    run_state_path = progress_root / "orchestrator-run.json"
    if run_state_path.is_file():
        previous_state = json.loads(run_state_path.read_text(encoding="utf-8"))
        attempts = list(previous_state.get("attempts", []))
    else:
        attempts = []
    attempt = {
        "attempt": len(attempts) + 1,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "start_time_unix": time.time(),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "gpu_assignment": expected_visible,
        "dataset_schema_version": int(manifest["schema_version"]),
        "dataset_schema_signature": str(manifest["schema_signature"]),
        "pipeline_signature": str(manifest["pipeline_signature"]),
        "pilot_gate": str(Path(args.pilot_gate).expanduser().resolve()),
        "lagernvs_pose_audit_required": bool(args.require_lagernvs_pose_audit),
        "lagernvs_pose_contract_signature": (
            None if pose_audit is None else pose_audit["pose_contract_signature"]
        ),
        "source_tree_fingerprint_before": source_before,
        "persistent_log": str(root / "logs" / "preprocessing" / "orchestrator.log"),
        "status": "running",
        "exit_code": None,
    }
    attempts.append(attempt)

    def update_run_state() -> None:
        write_json_atomic(
            run_state_path,
            {
                "schema_version": 1,
                "current_attempt": int(attempt["attempt"]),
                "attempts": attempts,
            },
        )

    update_run_state()
    write_json_atomic(progress_root / "orchestrator-plan.json", plan)
    telemetry = _GPUTelemetry(
        root,
        int(attempt["attempt"]),
        float(args.gpu_telemetry_interval_seconds),
    )
    telemetry.start()
    try:
        for stage in stages:
            telemetry.set_stage(stage)
            _run_one_stage(args, stage, worker_count)
        source_after = source_tree_fingerprint(args.droid_root)
        write_json_atomic(
            root / "metadata" / "source-tree-after-full-run.json", source_after
        )
        assert_source_fingerprint_unchanged(source_before, source_after)
        if "compose" in stages and args.integrity != "none":
            full_scan = args.integrity == "full"
            report = verify_stage0_dataset(
                IntegrityConfig(
                    root=args.root,
                    random_samples=int(args.integrity_samples),
                    verify_shard_checksums=True,
                    full_payload_scan=full_scan,
                    decode_all_jpegs=full_scan,
                    loader_windows=int(args.integrity_loader_windows),
                )
            )
            report_path = root / "reports" / "integrity.json"
            write_json_atomic(report_path, report)
            final_path = write_final_manifest(args.root, report)
            print(
                json.dumps(
                    {
                        "integrity": report["status"],
                        "integrity_report": str(report_path),
                        "final_manifest": str(final_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    except BaseException as error:
        attempt.update(
            {
                "status": "failed",
                "exit_code": 1,
                "end_time_unix": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        update_run_state()
        raise
    else:
        attempt.update(
            {
                "status": "complete",
                "exit_code": 0,
                "end_time_unix": time.time(),
            }
        )
        update_run_state()
    finally:
        attempt["gpu_telemetry"] = telemetry.stop()
        update_run_state()


def _status_command(args: argparse.Namespace) -> None:
    manifest = load_stage0_manifest(args.root)
    signature = str(manifest["schema_signature"])
    total = len(manifest["shards"])
    output: dict[str, Any] = {
        "root": str(Path(args.root).expanduser().resolve()),
        "schema_signature": signature,
        "expected_shards": total,
        "stages": {},
    }
    for stage in (*STAGES[:-1], "final"):
        completed = []
        for shard in manifest["shards"]:
            shard_id = int(shard["shard_id"])
            path = shard_path(args.root, stage, shard_id)
            if shard_is_complete(path, schema_signature=signature):
                completed.append(shard_id)
        output["stages"][stage] = {
            "complete": len(completed),
            "expected": total,
            "missing": total - len(completed),
        }
    partials, archived_partials = classify_partial_artifacts(args.root)
    output["unexpected_partial_files"] = partials
    output["archived_partial_files"] = archived_partials
    print(json.dumps(output, indent=2, sort_keys=True))


def _common_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--droid-root", default="/home/ws/data/droid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable offline Stage-0 DROID preprocessing."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    _common_root(manifest)
    manifest.add_argument("--calibration-manifest", default=DEFAULT_CALIBRATION)
    manifest.add_argument("--mode", choices=("pilot", "full"), required=True)
    manifest.add_argument("--episode-ids")
    manifest.add_argument("--pilot-episodes", type=int, default=12)
    manifest.add_argument("--target-retained-per-shard", type=int, default=4096)
    manifest.add_argument("--jpeg-quality", type=int, default=95)
    manifest.add_argument("--jpeg-subsampling", default="4:4:4")
    manifest.add_argument("--zstd-level", type=int, default=3)

    worker = subparsers.add_parser("worker")
    _common_root(worker)
    worker.add_argument("--stage", choices=STAGES, required=True)
    worker.add_argument("--worker-id", type=int, required=True)
    worker.add_argument("--worker-count", type=int, required=True)
    worker.add_argument("--jpeg-quality", type=int, default=95)
    worker.add_argument("--jpeg-subsampling", default="4:4:4")
    worker.add_argument("--zstd-level", type=int, default=3)
    worker.add_argument("--model-cache", required=True)
    worker.add_argument("--scene-center", nargs=3, type=float, required=True)
    worker.add_argument("--codec-audit-images", type=int, default=32)

    environment = subparsers.add_parser(
        "environment", description="Record one resolved Python environment."
    )
    environment.add_argument(
        "--stage", choices=("training", "da3", "megaflow", "lagernvs"), required=True
    )
    environment.add_argument("--output", required=True)

    run = subparsers.add_parser("run")
    _common_root(run)
    run.add_argument("--stages", nargs="+", default=list(STAGES))
    run.add_argument("--workers", type=int, default=3)
    run.add_argument("--jpeg-quality", type=int, default=95)
    run.add_argument("--jpeg-subsampling", default="4:4:4")
    run.add_argument("--zstd-level", type=int, default=3)
    run.add_argument(
        "--model-cache",
        default="/home/ws/data/droid_stage0_preprocessed/metadata/model_cache",
    )
    run.add_argument(
        "--scene-center",
        nargs=3,
        type=float,
        default=DEFAULT_LAGER_SCENE_CENTER,
    )
    run.add_argument("--codec-audit-images", type=int, default=32)
    run.add_argument(
        "--python-training", default=str(REPOSITORY_ROOT / ".venv/bin/python")
    )
    run.add_argument(
        "--python-da3",
        default=str(REPOSITORY_ROOT / ".preprocessing-envs/da3/bin/python"),
    )
    run.add_argument(
        "--python-megaflow",
        default=str(REPOSITORY_ROOT / ".preprocessing-envs/megaflow/bin/python"),
    )
    run.add_argument(
        "--python-lagernvs",
        default=str(REPOSITORY_ROOT / ".preprocessing-envs/lagernvs/bin/python"),
    )
    run.add_argument(
        "--pilot-gate",
        default="/home/ws/data/droid_stage0_preprocessed/pilot/reports/pilot_gate.json",
    )
    run.add_argument("--allow-missing-pilot-gate", action="store_true")
    run.add_argument(
        "--require-lagernvs-pose-audit",
        action="store_true",
        help=(
            "Refuse LagerNVS launch until every unfinished shard has a current "
            "signed pose-only audit report."
        ),
    )
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--integrity",
        choices=("full", "sample", "none"),
        default="full",
        help="Final payload audit after compose; full decodes every numeric/JPEG payload.",
    )
    run.add_argument("--integrity-samples", type=int, default=512)
    run.add_argument("--integrity-loader-windows", type=int, default=32)
    run.add_argument(
        "--gpu-telemetry-interval-seconds",
        type=float,
        default=60.0,
        help="Persistent read-only utilization sampling for the three authorized GPUs.",
    )

    status = subparsers.add_parser("status")
    _common_root(status)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        _manifest_command(args)
    elif args.command == "worker":
        _worker_command(args)
    elif args.command == "environment":
        _environment_command(args)
    elif args.command == "run":
        _run_command(args)
    elif args.command == "status":
        _status_command(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
