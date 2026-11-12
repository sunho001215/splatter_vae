"""Reviewed resource classes and release-63f5165 per-file JUnit seconds.

Unknown files reserve a GPU slot; resource regexes can promote them to serial.
At 32 threads: 10 CPU + 3 GPU + 1 serial gives two threads per worker.
Whole files retain fixture/order semantics; serial files never overlap each other.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

FILE_CLASSES = {
    "tests/test_arm4_configs.py": "cpu",
    "tests/test_baseline_representation.py": "gpu",
    "tests/test_baselines.py": "gpu",
    "tests/test_baselines_dynamo.py": "gpu",
    "tests/test_baselines_snerl.py": "gpu",
    "tests/test_camera_suites.py": "gpu",
    "tests/test_chamfer_lean.py": "gpu",
    "tests/test_checkpoint.py": "gpu",
    "tests/test_cli.py": "gpu",
    "tests/test_contract.py": "cpu",
    "tests/test_cross_host_equivalence.py": "gpu",
    "tests/test_cst_configs.py": "cpu",
    "tests/test_cuda_render.py": "gpu",
    "tests/test_cuda_ssim.py": "gpu",
    "tests/test_dataset_downloads.py": "serial",  # Concurrent downloader workers and timed interruption/recovery.
    "tests/test_ddp.py": "serial",  # Real Gloo ranks and distributed rendezvous.
    "tests/test_decoder.py": "gpu",
    "tests/test_depth.py": "cpu",
    "tests/test_diagnostics.py": "cpu",
    "tests/test_droid.py": "cpu",
    "tests/test_droid_crop.py": "cpu",
    "tests/test_droid_csw4_profile.py": "cpu",
    "tests/test_droid_far4_profiles.py": "cpu",
    "tests/test_droid_largescale.py": "cpu",
    "tests/test_droid_protocols.py": "cpu",
    "tests/test_droid_vits_profile.py": "cpu",
    "tests/test_encoder.py": "cpu",
    "tests/test_encoder_pilots.py": "cpu",
    "tests/test_export.py": "cpu",
    "tests/test_fair_share.py": "cpu",
    "tests/test_future_stride.py": "cpu",
    "tests/test_geometry.py": "cpu",
    "tests/test_gpu_guard.py": "cpu",
    "tests/test_gpu_isolation.py": "serial",  # Host GPU/process isolation probes use global host resources.
    "tests/test_heldout.py": "gpu",
    "tests/test_heldout_replay_check.py": "gpu",
    "tests/test_host_roles.py": "cpu",
    "tests/test_hybrid_masking.py": "gpu",
    "tests/test_jobs.py": "serial",  # Live scheduler subprocess ownership and termination timing.
    "tests/test_lifecycle_cleanup.py": "cpu",
    "tests/test_masking_perframe.py": "cpu",
    "tests/test_metaworld_checks.py": "cpu",
    "tests/test_motion.py": "gpu",
    "tests/test_motion3d.py": "gpu",
    "tests/test_mujoco_guard.py": "gpu",
    "tests/test_mvp3d.py": "gpu",
    "tests/test_non_metaworld_geometry.py": "cpu",
    "tests/test_nvidia_query.py": "serial",  # Live PID ownership, query deadlines and deferred process reaping.
    "tests/test_occlusion.py": "gpu",
    "tests/test_ops_helpers.py": "cpu",
    "tests/test_ops_layout.py": "cpu",
    "tests/test_policy_serving.py": "serial",  # Loopback policy sockets, session expiry and latency assertions.
    "tests/test_policy_staging.py": "serial",  # Serving threads and bounded shutdown/latency assertions.
    "tests/test_policy_staging_fql_settings.py": "cpu",
    "tests/test_pool_pin.py": "cpu",
    "tests/test_realworld_baselines.py": "cpu",
    "tests/test_realworld_calibration.py": "cpu",
    "tests/test_realworld_calibration_board.py": "cpu",
    "tests/test_realworld_calibration_release.py": "cpu",
    "tests/test_realworld_calibration_rig.py": "cpu",
    "tests/test_realworld_cli.py": "cpu",
    "tests/test_realworld_contract_workers.py": "cpu",
    "tests/test_realworld_fs_stage.py": "cpu",
    "tests/test_realworld_ir_stereo.py": "cpu",
    "tests/test_realworld_layout.py": "cpu",
    "tests/test_realworld_local.py": "cpu",
    "tests/test_realworld_m5_consumers.py": "cpu",
    "tests/test_realworld_mounts.py": "cpu",
    "tests/test_realworld_pipeline.py": "cpu",
    "tests/test_realworld_refactor.py": "cpu",
    "tests/test_realworld_rgbd.py": "cpu",
    "tests/test_realworld_rig_dw_extext.py": "cpu",
    "tests/test_realworld_rig_dw_robot.py": "cpu",
    "tests/test_realworld_rig_loma_render.py": "cpu",
    "tests/test_realworld_rig_loma_solve.py": "cpu",
    "tests/test_realworld_stereo.py": "cpu",
    "tests/test_realworld_stream_transport.py": "serial",  # Worker process sessions, disconnect and timeout measurements.
    "tests/test_realworld_sync_m5.py": "cpu",
    "tests/test_realworld_track_motion.py": "cpu",
    "tests/test_realworld_trackeverything.py": "cpu",
    "tests/test_realworld_train_store.py": "cpu",
    "tests/test_realworld_viewer_data.py": "cpu",
    "tests/test_remote_access.py": "serial",  # Timed access lock contention between live processes.
    "tests/test_remote_artifact_check.py": "cpu",
    "tests/test_remote_jobs.py": "cpu",
    "tests/test_remote_launch.py": "cpu",
    "tests/test_remote_runtime.py": "cpu",
    "tests/test_render_guard.py": "gpu",
    "tests/test_retired_algorithms.py": "cpu",
    "tests/test_review_options.py": "gpu",
    "tests/test_rl.py": "gpu",
    "tests/test_rl_cli.py": "gpu",
    "tests/test_robot_pc.py": "serial",  # Robot/policy loopback sockets and live deadline assertions.
    "tests/test_robot_pc_hwtest.py": "serial",  # Live controller timing, RTU/port ownership and signal handling.
    "tests/test_robot_pc_infer.py": "serial",  # Live setpoint deadlines and controller transport lifecycle.
    "tests/test_robot_pc_recorder.py": "serial",  # Paced recorder throughput and producer latency measurements.
    "tests/test_robot_pc_warmup.py": "serial",  # Warmup deadlines and controller transport lifecycle.
    "tests/test_run_identity.py": "cpu",
    "tests/test_rwlearn_actions.py": "cpu",
    "tests/test_rwlearn_bcflow.py": "serial",  # Policy server loopback sessions and timed inference overload.
    "tests/test_rwlearn_deploy.py": "cpu",
    "tests/test_rwlearn_fql.py": "cpu",
    "tests/test_rwlearn_fql_pipeline.py": "cpu",
    "tests/test_rwlearn_frame_mapping.py": "cpu",
    "tests/test_rwlearn_gate.py": "cpu",
    "tests/test_rwlearn_native_depth.py": "cpu",
    "tests/test_rwlearn_ours.py": "gpu",
    "tests/test_rwlearn_ours_realdata.py": "gpu",
    "tests/test_rwlearn_outcomes.py": "cpu",
    "tests/test_rwlearn_pixel.py": "cpu",
    "tests/test_rwlearn_snerl_eval.py": "gpu",
    "tests/test_rwlearn_store_baselines.py": "cpu",
    "tests/test_rwlearn_trainers_gpu.py": "gpu",
    "tests/test_scheduler_daemon_reexec.py": "serial",  # Process-wide child reaping and daemon PID ownership.
    "tests/test_selfrender_radial.py": "gpu",
    "tests/test_selfrender_replace.py": "gpu",
    "tests/test_selfrender_v4.py": "gpu",
    "tests/test_snerl_sac.py": "gpu",
    "tests/test_stop_step.py": "gpu",
    "tests/test_sync_media.py": "cpu",
    "tests/test_third_party.py": "cpu",
    "tests/test_vision_only.py": "gpu",
    "tests/test_watch.py": "cpu",
    "tests/test_workers.py": "gpu",
    "tests/test_run_tests_sharding.py": "cpu",
}

# Summed setup/call/teardown times from the supplied native release report.
FILE_SECONDS = {
    "tests/test_arm4_configs.py": 4.908,
    "tests/test_baseline_representation.py": 13.307,
    "tests/test_baselines.py": 37.449,
    "tests/test_baselines_dynamo.py": 40.860,
    "tests/test_baselines_snerl.py": 67.049,
    "tests/test_camera_suites.py": 42.150,
    "tests/test_chamfer_lean.py": 0.999,
    "tests/test_checkpoint.py": 18.849,
    "tests/test_cli.py": 0.065,
    "tests/test_contract.py": 0.239,
    "tests/test_cross_host_equivalence.py": 1.852,
    "tests/test_cst_configs.py": 0.440,
    "tests/test_cuda_render.py": 0.601,
    "tests/test_cuda_ssim.py": 0.027,
    "tests/test_dataset_downloads.py": 12.367,
    "tests/test_ddp.py": 5.810,
    "tests/test_decoder.py": 0.266,
    "tests/test_depth.py": 0.062,
    "tests/test_diagnostics.py": 0.523,
    "tests/test_droid.py": 0.215,
    "tests/test_droid_crop.py": 9.114,
    "tests/test_droid_csw4_profile.py": 0.601,
    "tests/test_droid_far4_profiles.py": 3.995,
    "tests/test_droid_largescale.py": 21.848,
    "tests/test_droid_protocols.py": 0.221,
    "tests/test_droid_vits_profile.py": 0.754,
    "tests/test_encoder.py": 0.051,
    "tests/test_encoder_pilots.py": 8.710,
    "tests/test_export.py": 0.273,
    "tests/test_fair_share.py": 0.696,
    "tests/test_future_stride.py": 4.683,
    "tests/test_geometry.py": 0.018,
    "tests/test_gpu_guard.py": 0.546,
    "tests/test_gpu_isolation.py": 0.234,
    "tests/test_heldout.py": 1.447,
    "tests/test_heldout_replay_check.py": 0.021,
    "tests/test_host_roles.py": 4.883,
    "tests/test_hybrid_masking.py": 1.504,
    "tests/test_jobs.py": 7.677,
    "tests/test_lifecycle_cleanup.py": 1.314,
    "tests/test_masking_perframe.py": 0.802,
    "tests/test_metaworld_checks.py": 0.869,
    "tests/test_motion.py": 0.033,
    "tests/test_motion3d.py": 0.283,
    "tests/test_mujoco_guard.py": 0.508,
    "tests/test_mvp3d.py": 36.911,
    "tests/test_non_metaworld_geometry.py": 0.046,
    "tests/test_nvidia_query.py": 3.159,
    "tests/test_occlusion.py": 0.084,
    "tests/test_ops_helpers.py": 0.120,
    "tests/test_ops_layout.py": 20.493,
    "tests/test_policy_serving.py": 30.490,
    "tests/test_policy_staging.py": 3.077,
    "tests/test_policy_staging_fql_settings.py": 0.101,
    "tests/test_pool_pin.py": 0.006,
    "tests/test_realworld_baselines.py": 0.987,
    "tests/test_realworld_calibration.py": 36.765,
    "tests/test_realworld_calibration_board.py": 157.920,
    "tests/test_realworld_calibration_release.py": 12.577,
    "tests/test_realworld_calibration_rig.py": 35.569,
    "tests/test_realworld_cli.py": 0.365,
    "tests/test_realworld_contract_workers.py": 2.435,
    "tests/test_realworld_fs_stage.py": 0.030,
    "tests/test_realworld_ir_stereo.py": 0.096,
    "tests/test_realworld_layout.py": 19.188,
    "tests/test_realworld_local.py": 14.245,
    "tests/test_realworld_m5_consumers.py": 0.814,
    "tests/test_realworld_mounts.py": 0.158,
    "tests/test_realworld_pipeline.py": 6.509,
    "tests/test_realworld_refactor.py": 1.462,
    "tests/test_realworld_rgbd.py": 10.161,
    "tests/test_realworld_rig_dw_extext.py": 3.435,
    "tests/test_realworld_rig_dw_robot.py": 23.607,
    "tests/test_realworld_rig_loma_render.py": 0.161,
    "tests/test_realworld_rig_loma_solve.py": 1.496,
    "tests/test_realworld_stereo.py": 0.058,
    "tests/test_realworld_stream_transport.py": 48.779,
    "tests/test_realworld_sync_m5.py": 0.253,
    "tests/test_realworld_track_motion.py": 6.156,
    "tests/test_realworld_trackeverything.py": 21.137,
    "tests/test_realworld_train_store.py": 20.544,
    "tests/test_realworld_viewer_data.py": 0.021,
    "tests/test_remote_access.py": 5.911,
    "tests/test_remote_artifact_check.py": 0.050,
    "tests/test_remote_jobs.py": 18.789,
    "tests/test_remote_launch.py": 8.699,
    "tests/test_remote_runtime.py": 0.533,
    "tests/test_render_guard.py": 0.028,
    "tests/test_retired_algorithms.py": 1.043,
    "tests/test_review_options.py": 0.800,
    "tests/test_rl.py": 26.538,
    "tests/test_rl_cli.py": 90.228,
    "tests/test_robot_pc.py": 29.333,
    "tests/test_robot_pc_hwtest.py": 54.208,
    "tests/test_robot_pc_infer.py": 0.825,
    "tests/test_robot_pc_recorder.py": 10.050,
    "tests/test_robot_pc_warmup.py": 0.431,
    "tests/test_run_identity.py": 3.366,
    "tests/test_rwlearn_actions.py": 0.597,
    "tests/test_rwlearn_bcflow.py": 1.437,
    "tests/test_rwlearn_deploy.py": 3.526,
    "tests/test_rwlearn_fql.py": 7.090,
    "tests/test_rwlearn_fql_pipeline.py": 10.920,
    "tests/test_rwlearn_frame_mapping.py": 0.742,
    "tests/test_rwlearn_gate.py": 4.588,
    "tests/test_rwlearn_native_depth.py": 38.549,
    "tests/test_rwlearn_ours.py": 12.543,
    "tests/test_rwlearn_outcomes.py": 0.540,
    "tests/test_rwlearn_pixel.py": 0.405,
    "tests/test_rwlearn_snerl_eval.py": 3.683,
    "tests/test_rwlearn_store_baselines.py": 1.804,
    "tests/test_rwlearn_trainers_gpu.py": 143.405,
    "tests/test_scheduler_daemon_reexec.py": 0.070,
    "tests/test_selfrender_radial.py": 2.479,
    "tests/test_selfrender_replace.py": 3.108,
    "tests/test_selfrender_v4.py": 2.622,
    "tests/test_snerl_sac.py": 1.818,
    "tests/test_stop_step.py": 3.515,
    "tests/test_sync_media.py": 0.074,
    "tests/test_third_party.py": 1.523,
    "tests/test_vision_only.py": 19.907,
    "tests/test_watch.py": 0.785,
    "tests/test_workers.py": 1.848,
}

SERIAL_RESOURCE = re.compile(
    r"torch\.distributed|DistributedDataParallel|\bDDP\b|\bgloo\b|"
    r"\b(?:socket|zmq)\b|MASTER_PORT|\.bind\(|\.listen\(|"
    r"time\.(?:sleep|monotonic|perf_counter)|SIG(?:TERM|INT)"
)


def classify(file: str, root: Path) -> str:
    if file in FILE_CLASSES:
        return FILE_CLASSES[file]
    path = root / file
    if path.is_file() and SERIAL_RESOURCE.search(path.read_text()):
        return "serial"
    return "gpu"


def partition(nodeids: list[str], root: Path, cpu: int, gpu: int,
              durations: dict[str, float] | None = None) -> list[dict]:
    """Deterministic LPT, never splitting a file or creating an empty shard."""
    if cpu < 1 or gpu < 1:
        raise ValueError("Shard counts must be positive")
    counts = Counter(node.split("::", 1)[0] for node in nodeids)
    weights = FILE_SECONDS if durations is None else durations
    shards = []
    for kind, limit in (("serial", 1), ("gpu", gpu), ("cpu", cpu)):
        files = [file for file in counts if classify(file, root) == kind]
        bins = [{"class": kind, "files": [], "tests": 0, "estimated_seconds": 0.0}
                for _ in range(min(limit, len(files)))]
        for file in sorted(files, key=lambda f: (-weights.get(f, counts[f]), f)):
            target = min(bins, key=lambda b: b["estimated_seconds"])
            target["files"].append(file)
            target["tests"] += counts[file]
            target["estimated_seconds"] += weights.get(file, counts[file])
        shards.extend(bins)
    return [{"index": i, **shard} for i, shard in enumerate(shards)]
