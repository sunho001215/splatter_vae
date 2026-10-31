"""Pure validation of the preregistered Meta-World cross-host diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

PROTOCOL = "meta-world-cross-host-v2"
FRESH_STEPS = 200
MIGRATION_STEPS = 50
SCHEDULE_STEPS = 200000
STEP0_RELATIVE_TOLERANCE = 1e-4
DATA_IDENTITY = "56bd8ce9c59e81077061f577888f708541c2c14d605248f0cbdf81521e54bae1"
DATA_ROOT = Path("/home/ws/data/metaworld/splatter4d_v1")
REMOTE_ROOT = PurePosixPath("/home/compu/kaist/sunho")
REMOTE_DATA = REMOTE_ROOT / "data/metaworld/splatter4d_v1"
TASKS = ("hammer", "pick-place")
VARIANTS = ("d0", "d1", "d2", "d3", "d4")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
STATE_HASHES = ("model_sha256", "optimizer_sha256", "scheduler_sha256", "rng_sha256")
CRITERION = "unsmoothed-total-loss-rmse<=remote-same-host-repeat-rmse"


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_overrides(task: str, variant: str, seed: int, mode: str) -> list[str]:
    if task not in TASKS or variant not in VARIANTS or type(seed) is not int or seed not in (0, 1):
        raise ValueError("Unknown development task, item-1 variant or pretraining seed")
    if mode not in ("fresh", "migration") or (mode == "fresh" and (task, variant, seed) != ("hammer", "d0", 0)):
        raise ValueError("Fresh acceptance is only C+D0 hammer seed 0; other cases are migration-only")
    values = [
        f"train.steps={SCHEDULE_STEPS}",
        "train.stop_step=100000",
        f"train.seed={seed}",
        "aug.synth_views=2",
        "aug.synth_render=true",
        "loss.depth_valid_far=true",
    ]
    if variant != "d0":
        values.extend(("loss.occlusion=1.0", "loss.occlusion_margin=0.02"))
    if variant in ("d2", "d3"):
        values.append("loss.depth_grad=0.0")
    if variant in ("d2", "d4"):
        values.append("loss.depth_hard=0.0")
    return values


def validate_config(cfg: dict, task: str, variant: str, seed: int, mode: str) -> None:
    case_overrides(task, variant, seed, mode)
    if not isinstance(cfg, dict):
        raise ValueError("Actual numerical config is required")
    expected = {
        "data.regime": "metaworld",
        "data.task": task,
        "data.train_ratio": 0.96,
        "data.strides": [2, 4, 6],
        "data.val_strides": [2, 6],
        "data.episodes": None,
        "train.steps": SCHEDULE_STEPS,
        "train.stop_step": 100000,
        "train.seed": seed,
        "train.batch_size": 16,
        "train.bf16": True,
        "train.num_workers": 6,
        "train.prefetch_factor": 2,
        "aug.synth_views": 2,
        "aug.synth_render": True,
        "loss.depth_valid_far": True,
        "loss.depth_grad": 0.0 if variant in ("d2", "d3") else 0.5,
        "loss.depth_hard": 0.0 if variant in ("d2", "d4") else 0.5,
    }
    for dotted, value in expected.items():
        section, key = dotted.split(".")
        node = cfg.get(section)
        if (
            not isinstance(node, dict)
            or key not in node
            or node[key] != value
            or isinstance(node[key], bool) != isinstance(value, bool)
        ):
            raise ValueError(f"Production case config differs: {dotted}")
    data, loss = cfg["data"], cfg["loss"]
    if (
        data.get("split_seed", 0) != 0
        or data.get("fixed_window_indices") is not None
        or loss.get("occlusion", 0.0) != (0.0 if variant == "d0" else 1.0)
        or (variant != "d0" and loss.get("occlusion_margin") != 0.02)
    ):
        raise ValueError("Production split, sampling or item-1 objective differs")


def artifact_path(path: str | Path, roots: tuple[Path, ...], *, existing: bool = False) -> Path:
    candidate = Path(path)
    if (
        not candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.resolve() != candidate
        or (isinstance(path, str) and str(candidate) != path)
        or not any(root.resolve() in candidate.parents for root in roots)
    ):
        raise ValueError("Artifacts must be symlink-free descendants of an approved runtime root")
    if existing and not candidate.is_file():
        raise ValueError("Required artifact is not a regular file")
    return candidate


def _hash(value, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"Missing or invalid {name}")
    return value


def _finite(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Missing or nonfinite {name}")
    return float(value)


def validate_data_manifest(payload: dict) -> list[dict]:
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise ValueError("Verified development-data manifest is required")
    wanted = {
        name
        for task in TASKS
        for name in (
            f"{task}.hdf5",
            f"heldout_sets/{task}.hdf5",
            f"splits/{task}_seed0.json",
            f"workspace_stats/{task}.json",
        )
    }
    observed, normalized = set(), []
    for entry in payload["files"]:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("target"), str)
            or not isinstance(entry.get("source"), str)
        ):
            raise ValueError("Development-data file paths are missing")
        remote, local = PurePosixPath(entry["target"]), PurePosixPath(entry["source"])
        if (
            ".." in remote.parts
            or ".." in local.parts
            or str(remote) != entry["target"]
            or str(local) != entry["source"]
            or REMOTE_DATA not in remote.parents
        ):
            raise ValueError("Development-data paths are outside the fixed transferred dataset")
        relative = str(remote.relative_to(REMOTE_DATA))
        if relative not in wanted or relative in observed or local != PurePosixPath(DATA_ROOT) / relative:
            raise ValueError("Development-data source/target identity differs")
        if type(entry.get("bytes")) is not int or entry["bytes"] <= 0:
            raise ValueError("Development-data size must be positive integer bytes")
        observed.add(relative)
        normalized.append(
            {
                "target": str(remote.relative_to(REMOTE_ROOT)),
                "sha256": _hash(entry.get("sha256"), "data checksum"),
                "bytes": entry["bytes"],
            }
        )
    if observed != wanted:
        raise ValueError("Transferred development-data manifest must cover exactly its eight files")
    normalized.sort(key=lambda entry: entry["target"])
    identity = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
    if identity != DATA_IDENTITY or payload.get("identity_sha256") != identity:
        raise ValueError("Development-data manifest does not match the transferred campaign identity")
    return normalized


def input_files(files: list[dict], task: str) -> list[dict]:
    names = {f"{task}.hdf5", f"splits/{task}_seed0.json", f"workspace_stats/{task}.json"}
    return [
        entry for entry in files if str(PurePosixPath(entry["target"]).relative_to("data/metaworld/splatter4d_v1")) in names
    ]


def verify_data_files(payload: dict, task: str) -> tuple[list[dict], dict[str, tuple]]:
    selected = input_files(validate_data_manifest(payload), task)
    identities = {}
    for entry in selected:
        relative = PurePosixPath(entry["target"]).relative_to("data/metaworld/splatter4d_v1")
        path = artifact_path(DATA_ROOT / relative, (DATA_ROOT,), existing=True)
        before = path.stat()
        if before.st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Input data checksum or size differs: {path.name}")
        original = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identities[str(path)] = original
    verify_data_unchanged(identities)
    return selected, identities


def verify_data_unchanged(identities: dict[str, tuple]) -> None:
    for name, expected in identities.items():
        current = Path(name).stat()
        if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != expected:
            raise ValueError("Input data changed during numerical verification")


def step0_comparison(reference: dict, candidate: dict) -> tuple[bool, dict]:
    _losses(reference)
    _losses(candidate)
    if candidate.keys() != reference.keys():
        raise ValueError("Complete matching total/component losses are required")
    differences = {}
    for name in sorted(reference):
        expected, actual = float(reference[name]), float(candidate[name])
        difference = _finite(abs(actual - expected), "step-0 difference")
        tolerance = STEP0_RELATIVE_TOLERANCE * abs(expected)
        differences[name] = {
            "absolute_difference": difference,
            "allowed_difference": tolerance,
            "within_noise": difference <= tolerance,
        }
    return all(value["within_noise"] for value in differences.values()), differences


def _losses(value: dict) -> None:
    if not isinstance(value, dict) or not value or "total" not in value or any(not isinstance(name, str) for name in value):
        raise ValueError("All observed total/component losses are required")
    for loss in value.values():
        _finite(loss, "observed loss")


def curve_distance(reference: list[dict], candidate: list[dict]) -> dict:
    if not reference or len(reference) != len(candidate):
        raise ValueError("Matching nonempty unsmoothed curves are required")
    differences = [
        _finite(
            abs(_finite(a["losses"]["total"], "total loss") - _finite(b["losses"]["total"], "total loss")),
            "curve difference",
        )
        for a, b in zip(reference, candidate)
    ]
    rmse = _finite(math.hypot(*differences) / math.sqrt(len(differences)), "curve RMSE")
    return {"rmse": rmse, "max_absolute_difference": max(differences), "pointwise_absolute_differences": differences}


def _state_hashes(value: dict) -> None:
    if not isinstance(value, dict) or value.keys() != set(STATE_HASHES):
        raise ValueError("Complete model/optimizer/scheduler/RNG identities are required")
    for name, digest in value.items():
        _hash(digest, name)


def validate_report(report: dict) -> None:
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != 2
        or report.get("protocol") != PROTOCOL
        or report.get("purpose") != "cross_host_equivalence"
        or report.get("campaign_eligible") is not False
        or report.get("measurement_complete") is not True
        or report.get("source_changed_during_check") is not False
        or report.get("source_dirty") is not False
        or "error" in report
    ):
        raise ValueError("Complete immutable-source native verification evidence is required")
    task, variant, seed, mode = (report.get(key) for key in ("task", "variant", "seed", "mode"))
    case_overrides(task, variant, seed, mode)
    host = report.get("host")
    if host not in ("local", "remote"):
        raise ValueError("Verification host must be explicit")
    uuids = report.get("gpu_uuids")
    if not isinstance(uuids, list) or len(uuids) != 1 or not isinstance(uuids[0], str) or not uuids[0].startswith("GPU-"):
        raise ValueError("Verification must record exactly one guarded GPU UUID")
    if not isinstance(report.get("git_commit"), str) or not COMMIT.fullmatch(report["git_commit"]):
        raise ValueError("Verification commit is missing")
    image = report.get("image_digest")
    if not isinstance(image, str) or not image.startswith("sha256:"):
        raise ValueError("The verified image digest is required on both hosts")
    _hash(image.removeprefix("sha256:"), "image digest")
    sources = report.get("source_sha256")
    if not isinstance(sources, dict) or not sources or report.get("source_sha256_before") != sources:
        raise ValueError("Unchanged before/after source fingerprints are required")
    for name, digest in sources.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Source paths are missing")
        _hash(digest, "source checksum")
    verifier = report.get("verifier_sha256")
    if (
        not isinstance(verifier, dict)
        or verifier.keys() != {"producer", "validator"}
        or report.get("verifier_sha256_before") != verifier
    ):
        raise ValueError("The separately frozen operational verifier must remain unchanged")
    for digest in verifier.values():
        _hash(digest, "operational verifier checksum")
    normalized = report.get("numeric_config")
    validate_config(normalized, task, variant, seed, mode)
    if hashlib.sha256(json_bytes(normalized)).hexdigest() != _hash(report.get("config_sha256"), "config checksum"):
        raise ValueError("Actual numerical config checksum differs")
    if report.get("criterion") != CRITERION or report.get("data_identity") != DATA_IDENTITY:
        raise ValueError("Preregistered criterion and data identity must not change")
    _hash(report.get("data_manifest_sha256"), "data manifest checksum")
    files = report.get("data_manifest_files")
    if (
        not isinstance(files, list)
        or len(files) != 8
        or hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() != DATA_IDENTITY
        or report.get("data_files") != input_files(files, task)
    ):
        raise ValueError("Actual input files and transferred manifest identity must match")
    start = report.get("resume_step")
    steps = FRESH_STEPS if mode == "fresh" else MIGRATION_STEPS
    if type(start) is not int or start < 0 or type(report.get("steps")) is not int or report["steps"] != steps:
        raise ValueError("Actual optimizer step range is missing")
    if mode == "fresh":
        if start != 0 or report.get("checkpoint_sha256") is not None or report.get("checkpoint_state") is not None:
            raise ValueError("Fresh acceptance must not substitute a resumed experiment")
    else:
        if start <= 0 or start + steps > normalized["train"]["stop_step"]:
            raise ValueError("Migration diagnostics must follow the actual checkpoint within its original schedule")
        _hash(report.get("checkpoint_sha256"), "original checkpoint checksum")
        _state_hashes(report.get("checkpoint_state"))
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != (1 if host == "local" else 2):
        raise ValueError("One local run and two same-host remote repetitions are required")
    for run in runs:
        if (
            not isinstance(run, dict)
            or type(run.get("optimizer_steps_completed")) is not int
            or run["optimizer_steps_completed"] != steps
        ):
            raise ValueError("Every native optimizer update must actually complete")
        _hash(run.get("initial_model_sha256"), "initial model checksum")
        _state_hashes(run.get("initial_state"))
        if run["initial_model_sha256"] != run["initial_state"]["model_sha256"]:
            raise ValueError("Actual initial model identity differs")
        if mode == "migration" and run["initial_state"] != report["checkpoint_state"]:
            raise ValueError("Loaded model/optimizer/scheduler/RNG differs from the original checkpoint")
        trace = run.get("trace")
        if not isinstance(trace, list) or len(trace) != steps:
            raise ValueError("Every actual optimizer step must be measured")
        names = None
        for step, measurement in enumerate(trace, start):
            if not isinstance(measurement, dict) or type(measurement.get("step")) is not int or measurement["step"] != step:
                raise ValueError("Trace must retain the actual consecutive pre-update optimizer positions")
            _hash(measurement.get("batch_sha256"), "observed batch checksum")
            _losses(measurement.get("losses"))
            current = set(measurement["losses"])
            if names is not None and names != current:
                raise ValueError("Observed component losses changed unexpectedly")
            names = current
            rates = measurement.get("learning_rates")
            if not isinstance(rates, list) or not rates or any(_finite(rate, "learning rate") < 0 for rate in rates):
                raise ValueError("Actual production optimizer learning rates are required")


def compare_reports(reference: dict, candidate: dict) -> dict:
    validate_report(reference)
    validate_report(candidate)
    if reference["host"] != "local" or candidate["host"] != "remote":
        raise ValueError("An original local reference and two actual remote repetitions are required")
    fields = (
        "mode",
        "task",
        "variant",
        "seed",
        "git_commit",
        "image_digest",
        "source_sha256",
        "verifier_sha256",
        "config_sha256",
        "data_identity",
        "data_manifest_sha256",
        "data_files",
        "checkpoint_sha256",
        "checkpoint_state",
        "resume_step",
    )
    mismatches = [name for name in fields if reference[name] != candidate[name]]
    local, remote, repeat = reference["runs"][0], candidate["runs"][0], candidate["runs"][1]
    initial_identical = local["initial_state"] == remote["initial_state"] == repeat["initial_state"]
    inputs_identical = all(
        a["batch_sha256"] == b["batch_sha256"] == c["batch_sha256"]
        and a["learning_rates"] == b["learning_rates"] == c["learning_rates"]
        and a["losses"].keys() == b["losses"].keys() == c["losses"].keys()
        for a, b, c in zip(local["trace"], remote["trace"], repeat["trace"])
    )
    step0_matches, step0 = step0_comparison(local["trace"][0]["losses"], remote["trace"][0]["losses"])
    cross_host = curve_distance(local["trace"], remote["trace"])
    same_host = curve_distance(remote["trace"], repeat["trace"])
    curve_matches = cross_host["rmse"] <= same_host["rmse"]
    fresh = reference["mode"] == "fresh"
    intact = not mismatches and initial_identical and inputs_identical
    passed = intact and curve_matches and (step0_matches if fresh else True)
    return {
        "passed": passed,
        "protocol": PROTOCOL,
        "mode": reference["mode"],
        "criterion": CRITERION,
        "step0_identical": intact and (step0_matches if fresh else True),
        "short_run_within_noise": intact and curve_matches,
        "checkpoint_resume_within_noise": passed if not fresh else None,
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "resume_step": reference["resume_step"],
        "data_identity": DATA_IDENTITY,
        "identity_mismatches": mismatches,
        "initial_state_identical": initial_identical,
        "batch_sequence_identical": inputs_identical,
        "step0_loss_differences": step0,
        "cross_host_curve": cross_host,
        "same_host_repeat_curve": same_host,
    }
