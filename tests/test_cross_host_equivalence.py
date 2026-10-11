from __future__ import annotations

import ast
import copy
import hashlib
import math
from pathlib import Path

import pytest

from s4d.cross_host_equivalence import (
    CRITERION,
    DATA_IDENTITY,
    DATA_ROOT,
    FRESH_STEPS,
    MIGRATION_STEPS,
    PROTOCOL,
    REMOTE_DATA,
    STATE_HASHES,
    artifact_path,
    case_overrides,
    compare_reports,
    curve_distance,
    input_files,
    json_bytes,
    sha256_file,
    step0_comparison,
    validate_config,
    validate_data_manifest,
    validate_report,
    verify_data_unchanged,
)


def _manifest():
    entries = (
        ("hammer.hdf5", 19940581894, "e314fdfb0ee6a2e7f2d0e8dcb4120a59fc44ba9af4f1518048127adefbe83b68"),
        ("heldout_sets/hammer.hdf5", 1773703148, "7ed9e75c9cadee0e067c62f69ecbfbff42221e6ad146fac51dae2dc7d9dbef8d"),
        ("splits/hammer_seed0.json", 2845, "86dd4340f819f4daa394c136312294e9bb88d7b21be85b414c1161d45e647bb3"),
        ("workspace_stats/hammer.json", 5467, "66d4e7499e5dbc7f74687cd0fdce7e7ecf641139c88e49605530e5812d30d41f"),
        ("pick-place.hdf5", 17291959202, "97546b78952724f2878062b80357e26ca2ddc06f505553d9462788e12e27310d"),
        ("heldout_sets/pick-place.hdf5", 1604908380, "f2f400f3aa241ebee9bc5efad156a054148c8024718e04ca6ea454619d0053f7"),
        ("splits/pick-place_seed0.json", 2849, "546721a5bcb45deca203b332218cf9e7fdedf15731686e5f0f6ce4dd66911e1a"),
        ("workspace_stats/pick-place.json", 5465, "f0e1acab64e6453d715629df3053adff9e7eea64074bcd8379bbf9fd2b34d0f9"),
    )
    return {
        "identity_sha256": DATA_IDENTITY,
        "files": [
            {"source": str(DATA_ROOT / name), "target": str(REMOTE_DATA / name), "bytes": size, "sha256": digest}
            for name, size, digest in entries
        ],
    }


def _config(task="hammer", variant="d0", seed=0):
    cfg = {
        "data": {
            "regime": "metaworld",
            "task": task,
            "root": "<checksummed-data>",
            "train_ratio": 0.96,
            "strides": [2, 4, 6],
            "val_strides": [2, 6],
            "episodes": None,
        },
        "train": {
            "steps": 200000,
            "stop_step": 100000,
            "seed": seed,
            "batch_size": 16,
            "bf16": True,
            "num_workers": 6,
            "prefetch_factor": 2,
        },
        "aug": {"synth_views": 2, "synth_render": True},
        "loss": {
            "depth_valid_far": True,
            "depth_grad": 0.0 if variant in ("d2", "d3") else 0.5,
            "depth_hard": 0.0 if variant in ("d2", "d4") else 0.5,
        },
    }
    if variant != "d0":
        cfg["loss"].update(occlusion=1.0, occlusion_margin=0.02)
    return cfg


def _reports(mode="fresh", task="hammer", variant="d0", seed=0):
    start, count = (0, FRESH_STEPS) if mode == "fresh" else (40000, MIGRATION_STEPS)
    files, cfg = validate_data_manifest(_manifest()), _config(task, variant, seed)
    initial = {name: hashlib.sha256(name.encode()).hexdigest() for name in STATE_HASHES}
    run = {
        "initial_model_sha256": initial["model_sha256"],
        "initial_state": initial,
        "optimizer_steps_completed": count,
        "trace": [
            {
                "step": step,
                "batch_sha256": hashlib.sha256(str(step).encode()).hexdigest(),
                "losses": {"total": 1.0, "component": 0.0},
                "learning_rates": [0.0001],
            }
            for step in range(start, start + count)
        ],
    }
    verifier = {"producer": "e" * 64, "validator": "f" * 64}
    local = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "purpose": "cross_host_equivalence",
        "campaign_eligible": False,
        "measurement_complete": True,
        "source_changed_during_check": False,
        "source_dirty": False,
        "host": "local",
        "gpu_uuids": ["GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce"],
        "git_commit": "a" * 40,
        "image_digest": "sha256:" + "b" * 64,
        "source_sha256": {"s4d/train/loop.py": "c" * 64},
        "source_sha256_before": {"s4d/train/loop.py": "c" * 64},
        "verifier_sha256": verifier,
        "verifier_sha256_before": copy.deepcopy(verifier),
        "task": task,
        "variant": variant,
        "seed": seed,
        "mode": mode,
        "criterion": CRITERION,
        "numeric_config": cfg,
        "config_sha256": hashlib.sha256(json_bytes(cfg)).hexdigest(),
        "data_identity": DATA_IDENTITY,
        "data_manifest_sha256": "d" * 64,
        "data_manifest_files": files,
        "data_files": input_files(files, task),
        "resume_step": start,
        "steps": count,
        "checkpoint_sha256": "0" * 64 if mode == "migration" else None,
        "checkpoint_state": copy.deepcopy(initial) if mode == "migration" else None,
        "runs": [run],
        "passed": False,
        "cross_host_acceptance": False,
    }
    remote = copy.deepcopy(local)
    remote.update(host="remote", gpu_uuids=["GPU-4391fcee-537f-d408-d138-10d8a3866eea"])
    remote["runs"].append(copy.deepcopy(run))
    return local, remote


def test_fresh_protocol_has_one_case_and_two_remote_repetitions():
    local, remote = _reports()
    result = compare_reports(local, remote)
    assert result["passed"] and result["step0_identical"] and result["short_run_within_noise"]
    assert result["checkpoint_resume_within_noise"] is None
    assert result["cross_host_curve"]["rmse"] == result["same_host_repeat_curve"]["rmse"] == 0
    assert local["passed"] is remote["passed"] is False
    assert result["criterion"] == CRITERION
    assert case_overrides("hammer", "d0", 0, "fresh") == [
        "train.steps=200000",
        "train.stop_step=100000",
        "train.seed=0",
        "aug.synth_views=2",
        "aug.synth_render=true",
        "loss.depth_valid_far=true",
    ]


@pytest.mark.parametrize("task,variant,seed", [("pick-place", "d0", 0), ("hammer", "d1", 0), ("hammer", "d0", 1)])
def test_other_fresh_cases_are_not_required_or_admitted(task, variant, seed):
    with pytest.raises(ValueError, match="only C"):
        case_overrides(task, variant, seed, "fresh")
    assert case_overrides(task, variant, seed, "migration")


@pytest.mark.parametrize("task", ["hammer", "pick-place"])
@pytest.mark.parametrize("variant", ["d0", "d1", "d2", "d3", "d4"])
@pytest.mark.parametrize("seed", [0, 1])
def test_migration_checks_actual_checkpoint_case_and_50_steps(task, variant, seed):
    local, remote = _reports("migration", task, variant, seed)
    result = compare_reports(local, remote)
    assert result["passed"] and result["checkpoint_resume_within_noise"]
    assert result["resume_step"] == 40000 and result["checkpoint_sha256"] == local["checkpoint_sha256"]
    assert len(local["runs"][0]["trace"]) == 50


def test_step0_relative_rule_has_no_absolute_floor():
    assert step0_comparison({"total": 1.0, "zero": 0.0}, {"total": 1.00005, "zero": 0.0})[0]
    assert not step0_comparison({"total": 1.0}, {"total": 1.0002})[0]
    assert not step0_comparison({"total": 0.0}, {"total": 1e-300})[0]
    assert not step0_comparison({"total": -1e-8}, {"total": -1e-8 + 2e-12})[0]
    assert step0_comparison({"total": -1.0}, {"total": -1.00005})[0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1"])
def test_nonfinite_or_nonnumeric_losses_fail_closed(value):
    with pytest.raises(ValueError):
        step0_comparison({"total": 1.0}, {"total": value})


def test_curve_gate_uses_measured_same_host_rmse_without_smoothing_or_floor():
    local, remote = _reports()
    remote["runs"][0]["trace"][100]["losses"]["total"] += 0.01
    remote["runs"][1]["trace"][100]["losses"]["total"] += 0.01
    assert not compare_reports(local, remote)["passed"]
    remote["runs"][1]["trace"][100]["losses"]["total"] += 0.01
    result = compare_reports(local, remote)
    assert result["passed"]
    assert result["cross_host_curve"]["rmse"] == result["same_host_repeat_curve"]["rmse"]
    assert result["cross_host_curve"]["rmse"] == pytest.approx(0.01 / math.sqrt(200))


def test_zero_same_host_noise_has_no_numerical_floor():
    local, remote = _reports()
    for run in remote["runs"]:
        run["trace"][100]["losses"]["total"] += 1e-12
    result = compare_reports(local, remote)
    assert result["same_host_repeat_curve"]["rmse"] == 0
    assert result["cross_host_curve"]["rmse"] > 0
    assert not result["passed"]


@pytest.mark.parametrize("value", [None, True, 199])
def test_forward_traces_do_not_replace_actual_optimizer_updates(value):
    local, remote = _reports()
    remote["runs"][0]["optimizer_steps_completed"] = value
    with pytest.raises(ValueError):
        compare_reports(local, remote)


def test_pointwise_max_is_only_a_diagnostic_not_an_extra_gate():
    local, remote = _reports()
    remote["runs"][0]["trace"][100]["losses"]["total"] = 2.0
    for index in range(1, 200):
        remote["runs"][1]["trace"][index]["losses"]["total"] += 0.1
    result = compare_reports(local, remote)
    assert result["passed"]
    assert (
        result["cross_host_curve"]["max_absolute_difference"] > result["same_host_repeat_curve"]["max_absolute_difference"]
    )
    assert result["cross_host_curve"]["rmse"] < result["same_host_repeat_curve"]["rmse"]


def test_component_step0_failure_cannot_hide_behind_total_curve():
    local, remote = _reports()
    remote["runs"][0]["trace"][0]["losses"]["component"] = 1e-300
    result = compare_reports(local, remote)
    assert result["short_run_within_noise"] and not result["step0_identical"] and not result["passed"]


def test_migration_has_measured_50_step_noise_not_a_second_fresh_step0_threshold():
    local, remote = _reports("migration")
    remote["runs"][0]["trace"][0]["losses"]["total"] += 0.01
    remote["runs"][1]["trace"][0]["losses"]["total"] += 0.02
    assert compare_reports(local, remote)["checkpoint_resume_within_noise"]


@pytest.mark.parametrize(
    "key",
    [
        "git_commit",
        "image_digest",
        "config_sha256",
        "source_sha256",
        "verifier_sha256",
        "data_manifest_sha256",
        "checkpoint_sha256",
    ],
)
def test_changed_cross_host_provenance_rejects_comparison(key):
    local, remote = _reports("migration")
    if key == "git_commit":
        remote[key] = "b" * 40
    elif key == "image_digest":
        remote[key] = "sha256:" + "c" * 64
    elif key == "config_sha256":
        remote["numeric_config"]["train"]["lr"] = 0.002
        remote[key] = hashlib.sha256(json_bytes(remote["numeric_config"])).hexdigest()
    elif key in ("source_sha256", "verifier_sha256"):
        first = next(iter(remote[key]))
        remote[key][first] = "0" * 64
        remote[key + "_before"] = copy.deepcopy(remote[key])
    else:
        remote[key] = "1" * 64
    result = compare_reports(local, remote)
    assert not result["passed"] and key in result["identity_mismatches"]


@pytest.mark.parametrize("name", STATE_HASHES)
def test_restored_checkpoint_state_must_be_actual_and_complete(name):
    local, remote = _reports("migration")
    remote["runs"][0]["initial_state"][name] = "1" * 64
    with pytest.raises(ValueError):
        compare_reports(local, remote)


@pytest.mark.parametrize("index", [0, 24, 49, 199])
def test_actual_batch_sequence_differences_reject(index):
    local, remote = _reports()
    remote["runs"][1]["trace"][index]["batch_sha256"] = "1" * 64
    assert not compare_reports(local, remote)["passed"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("measurement_complete", False),
        ("source_dirty", True),
        ("source_changed_during_check", True),
        ("schema_version", True),
        ("protocol", "meta-world-cross-host-v1"),
        ("steps", 50),
        ("criterion", "fixed-tolerance"),
        ("data_identity", "0" * 64),
        ("resume_step", True),
        ("campaign_eligible", True),
        ("error", "native failure"),
    ],
)
def test_missing_or_changed_required_report_evidence_fails_closed(key, value):
    local, _ = _reports()
    local[key] = value
    with pytest.raises(ValueError):
        validate_report(local)


@pytest.mark.parametrize("mutation", ["truncated", "nan", "reordered", "missing_component", "missing_lr", "bad_hash"])
def test_incomplete_trace_cannot_claim_200_real_steps(mutation):
    local, remote = _reports()
    trace = remote["runs"][1]["trace"]
    if mutation == "truncated":
        trace.pop()
    elif mutation == "nan":
        trace[100]["losses"]["total"] = float("nan")
    elif mutation == "reordered":
        trace[10], trace[11] = trace[11], trace[10]
    elif mutation == "missing_component":
        trace[5]["losses"].pop("component")
    elif mutation == "missing_lr":
        trace[5].pop("learning_rates")
    else:
        trace[5]["batch_sha256"] = "not-a-hash"
    with pytest.raises(ValueError):
        compare_reports(local, remote)


def test_remote_must_really_repeat_and_reference_must_be_local():
    local, remote = _reports()
    with pytest.raises(ValueError):
        compare_reports(remote, local)
    remote["runs"].pop()
    with pytest.raises(ValueError):
        compare_reports(local, remote)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("train", "steps", 200),
        ("train", "bf16", False),
        ("train", "num_workers", 0),
        ("train", "prefetch_factor", 4),
        ("train", "batch_size", 1),
        ("data", "strides", [2]),
        ("data", "split_seed", 1),
        ("data", "fixed_window_indices", [0]),
        ("loss", "depth_hard", 0.0),
        ("aug", "synth_views", 0),
    ],
)
def test_production_numeric_settings_cannot_be_substituted(section, key, value):
    cfg = _config()
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg, "hammer", "d0", 0, "fresh")


def test_development_manifest_reproduces_actual_transfer_identity():
    manifest = _manifest()
    files = validate_data_manifest(manifest)
    assert len(files) == 8 and len(input_files(files, "hammer")) == 3
    manifest["files"].reverse()
    assert validate_data_manifest(manifest) == files


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "extra", "source", "traversal", "absolute", "hash", "size", "identity"]
)
def test_bad_development_manifests_reject(mutation):
    manifest = _manifest()
    first = manifest["files"][0]
    if mutation == "duplicate":
        manifest["files"].append(copy.deepcopy(first))
    elif mutation == "missing":
        manifest["files"].pop()
    elif mutation == "extra":
        first["target"] = str(REMOTE_DATA / "other.hdf5")
    elif mutation == "source":
        first["source"] = str(DATA_ROOT / "pick-place.hdf5")
    elif mutation == "traversal":
        first["target"] = str(REMOTE_DATA) + "/../splatter4d_v1/hammer.hdf5"
    elif mutation == "absolute":
        first["target"] = "/foreign/hammer.hdf5"
    elif mutation == "hash":
        first["sha256"] = "0" * 64
    elif mutation == "size":
        first["bytes"] = True
    else:
        manifest["identity_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_data_manifest(manifest)


def test_runtime_paths_reject_symlinks_traversal_and_outside_roots(tmp_path):
    file = tmp_path / "input.json"
    file.write_text("{}")
    assert artifact_path(str(file), (tmp_path,), existing=True) == file
    alias = tmp_path / "alias.json"
    alias.symlink_to(file)
    for path in (alias, str(tmp_path / "../input.json"), "relative.json", tmp_path, tmp_path.parent / "outside.json"):
        with pytest.raises(ValueError):
            artifact_path(path, (tmp_path,), existing=True)
    with pytest.raises(ValueError):
        artifact_path(str(tmp_path) + "/./input.json", (tmp_path,), existing=True)
    assert sha256_file(file) == hashlib.sha256(b"{}").hexdigest()


def test_data_changes_are_not_silently_accepted(tmp_path):
    file = tmp_path / "input"
    file.write_bytes(b"first")
    state = file.stat()
    identities = {str(file): (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)}
    verify_data_unchanged(identities)
    file.write_bytes(b"different")
    with pytest.raises(ValueError):
        verify_data_unchanged(identities)


def test_curve_metric_rejects_missing_lengths_and_nonfinite():
    with pytest.raises(ValueError):
        curve_distance([], [])
    with pytest.raises(ValueError):
        curve_distance([{"losses": {"total": 1}}], [])
    with pytest.raises(ValueError):
        curve_distance([{"losses": {"total": float("inf")}}], [{"losses": {"total": 1}}])


def test_producer_defers_native_imports_and_uses_production_training_paths():
    path = Path(__file__).resolve().parents[1] / "scripts/check_metaworld_equivalence.py"
    tree = ast.parse(path.read_text())
    top_imports = [name.name for node in tree.body if isinstance(node, ast.Import) for name in node.names]
    assert not {"torch", "gsplat", "numpy", "mujoco", "metaworld"}.intersection(top_imports)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    main = functions["main"]
    direct_calls = [
        node.value.func.id
        for node in main.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
    ]
    assert "atomic_report" in direct_calls
    guarded = next(node for node in main.body if isinstance(node, ast.Try))
    assert isinstance(guarded.body[0], ast.Assign) and guarded.body[0].value.func.id == "guard_gpus"
    assert isinstance(guarded.body[2], ast.Expr) and guarded.body[2].value.func.id == "require_passed_tests"
    source = ast.get_source_segment(path.read_text(), functions["production_run"])
    assert "forward_losses(model, batch, cfg, step)" in source
    assert "make_train_loader(dataset, cfg, ctx, seed)" in source
    assert "infinite(loader, start_step=start)" in source
    assert "optimizer.step()" in source and "scheduler.step()" in source
    assert "load_checkpoint(checkpoint, model.encoder, model.decoder, optimizer, scheduler, rank=0)" in source
    assert "_shutdown_workers()" in source and "batches.close()" in source
    assert "view(torch.uint8)" in ast.get_source_segment(path.read_text(), functions["tree_sha256"])
