"""Measure real production fresh or checkpoint-resumed Meta-World equivalence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import REPO, guard_gpus, require_passed_tests, source_fingerprints

import ops.resources.host_equivalence as equivalence
from ops.resources.host_equivalence import (
    CRITERION,
    DATA_IDENTITY,
    DATA_ROOT,
    FRESH_STEPS,
    MIGRATION_STEPS,
    PROTOCOL,
    TASKS,
    VARIANTS,
    artifact_path,
    case_overrides,
    compare_reports,
    json_bytes,
    sha256_file,
    validate_config,
    validate_data_manifest,
    validate_report,
    verify_data_files,
    verify_data_unchanged,
)
from ops.resources.gpu_guard import HOST_NAME, RUNTIME_ROOT, validate_runtime_path, visible_device_uuids
from ops.jobs.run_identity import numeric_config, verify_clean_release


def tree_sha256(value) -> str:
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def token(payload: bytes):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def visit(node):
        if torch.is_tensor(node):
            tensor = node.detach().cpu().contiguous()
            token(json_bytes(["tensor", list(tensor.shape), str(tensor.dtype)]))
            token(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(node, np.ndarray):
            token(json_bytes(["numpy", list(node.shape), str(node.dtype)]))
            token(memoryview(np.ascontiguousarray(node)).cast("B"))
        elif isinstance(node, dict):
            token(b"dict")
            token(str(len(node)).encode())
            for key in sorted(node, key=lambda key: json_bytes([type(key).__name__, key])):
                visit(key)
                visit(node[key])
        elif isinstance(node, (tuple, list)):
            token(type(node).__name__.encode())
            token(str(len(node)).encode())
            for item in node:
                visit(item)
        elif isinstance(node, bytes):
            token(b"bytes")
            token(node)
        elif node is None or type(node) in (str, int, float, bool):
            token(json_bytes([type(node).__name__, node]))
        elif isinstance(node, np.generic):
            visit(node.item())
        else:
            raise TypeError(f"Unsupported identity value: {type(node).__name__}")

    visit(value)
    return digest.hexdigest()


def state_identity(encoder: dict, decoder: dict, optimizer: dict, scheduler: dict, rng: dict) -> dict:
    return {
        "model_sha256": tree_sha256({"encoder": encoder, "decoder": decoder}),
        "optimizer_sha256": tree_sha256(optimizer),
        "scheduler_sha256": tree_sha256(scheduler),
        "rng_sha256": tree_sha256(rng),
    }


def original_checkpoint(path: Path, task: str, variant: str, seed: int) -> tuple[dict, int, dict]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or any(
        key not in checkpoint for key in ("step", "encoder", "decoder", "optimizer", "scheduler", "config", "rng")
    ):
        raise ValueError("Original production checkpoint states are missing")
    step, cfg = checkpoint["step"], checkpoint["config"]
    validate_config(cfg, task, variant, seed, "migration")
    if type(step) is not int or step <= 0 or step + MIGRATION_STEPS > cfg["train"]["stop_step"]:
        raise ValueError("Original checkpoint step is outside its unchanged production schedule")
    if (
        Path(cfg["data"]["root"]) != DATA_ROOT
        or Path(cfg["model"]["anchor_stats"]) != DATA_ROOT / "workspace_stats" / f"{task}.json"
    ):
        raise ValueError("Original checkpoint must retain the checksummed production data and anchor statistics")
    ranks = checkpoint.get("rng_by_rank")
    if ranks is not None and (not isinstance(ranks, list) or len(ranks) != 1):
        raise ValueError("Migration requires the original single-GPU per-rank RNG state")
    rng = ranks[0] if ranks is not None else checkpoint["rng"]
    if (
        not isinstance(rng, dict)
        or set(rng) != {"torch", "cuda", "numpy", "python"}
        or not isinstance(rng["cuda"], list)
        or len(rng["cuda"]) != 1
    ):
        raise ValueError("Original CUDA/Python/NumPy/Torch RNG states are required")
    identity = state_identity(
        checkpoint["encoder"], checkpoint["decoder"], checkpoint["optimizer"], checkpoint["scheduler"], rng
    )
    return cfg, step, identity


def production_run(
    cfg: dict,
    checkpoint: Path | None,
    start: int,
    steps: int,
    checkpoint_state: dict | None,
    measurement: dict,
    report: dict,
    output: Path,
) -> None:
    import torch

    from s4d.data.metaworld.dataset import MetaworldWindowDataset, split_episodes
    from s4d.train.checkpoint import capture_rng_state, load_checkpoint
    from s4d.train.ddp import DistContext
    from s4d.train.loop import (
        build_model,
        build_optimizer,
        forward_losses,
        infinite,
        make_train_loader,
        move_batch,
        seed_everything,
    )

    loader, batches, model, optimizer, scheduler, dataset = (None,) * 6
    try:
        seed = cfg["train"]["seed"]
        seed_everything(seed)
        device = torch.device("cuda", 0)
        ctx = DistContext(rank=0, local_rank=0, world_size=1, device=device)
        model = build_model(cfg).to(device)
        model.decoder.set_anchor_statistics(json.loads(Path(cfg["model"]["anchor_stats"]).read_text()))
        optimizer, scheduler = build_optimizer(model, cfg)
        if checkpoint is not None:
            if load_checkpoint(checkpoint, model.encoder, model.decoder, optimizer, scheduler, rank=0) != start:
                raise ValueError("Actual loaded optimizer step differs from the original checkpoint")
        initial = state_identity(
            model.encoder.state_dict(),
            model.decoder.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict(),
            capture_rng_state(),
        )
        measurement.update(initial_model_sha256=initial["model_sha256"], initial_state=initial, trace=[])
        if checkpoint_state is not None and initial != checkpoint_state:
            raise ValueError("Actually loaded model/optimizer/scheduler/RNG differs from original checkpoint")
        path = DATA_ROOT / f"{cfg['data']['task']}.hdf5"
        episodes, _ = split_episodes(path, cfg["data"]["train_ratio"], cfg["data"].get("split_seed", 0))
        dataset = MetaworldWindowDataset(path, episodes, strides=tuple(cfg["data"]["strides"]), seed=seed)
        loader = make_train_loader(dataset, cfg, ctx, seed)
        batches = infinite(loader, start_step=start)
        model.train()
        for step in range(start, start + steps):
            batch = next(batches)
            batch_hash = tree_sha256(batch)
            batch = move_batch(batch, device)
            out = forward_losses(model, batch, cfg, step)
            losses = {name: float(value.detach().cpu()) for name, value in out["losses"].items()}
            losses["total"] = float(out["total"].detach().cpu())
            if not all(math.isfinite(value) for value in losses.values()):
                report["nonfinite_measurement"] = {
                    "step": step,
                    "batch_sha256": batch_hash,
                    "losses": {name: value if math.isfinite(value) else str(value) for name, value in losses.items()},
                }
                raise FloatingPointError(f"Nonfinite production loss at step {step}")
            measurement["trace"].append(
                {
                    "step": step,
                    "batch_sha256": batch_hash,
                    "losses": losses,
                    "learning_rates": [group["lr"] for group in optimizer.param_groups],
                }
            )
            optimizer.zero_grad(set_to_none=True)
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            del out, batch
            atomic_report(output, report)
            if step == start or (step - start + 1) % 50 == 0:
                print(
                    json.dumps({"run": len(report["runs"]), "optimizer_step": step + 1, "total_loss": losses["total"]}),
                    flush=True,
                )
        torch.cuda.synchronize(device)
        measurement["optimizer_steps_completed"] = steps
    finally:
        if batches is not None:
            batches.close()
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
        loader = batches = model = optimizer = scheduler = dataset = None
        gc.collect()
        torch.cuda.empty_cache()


def verifier_fingerprints() -> dict:
    return {"producer": sha256_file(Path(__file__)), "validator": sha256_file(Path(equivalence.__file__))}


def atomic_report(directory: Path, report: dict) -> None:
    temporary = directory / ".equivalence.json.tmp"
    with temporary.open("w") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, directory / "equivalence.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="hammer")
    parser.add_argument("--variant", choices=VARIANTS, default="d0")
    parser.add_argument("--seed", choices=(0, 1), type=int, default=0)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--checkpoint", help="Original campaign checkpoint; selects migration-only 50-step diagnostics")
    args = parser.parse_args()
    if not os.environ.get("S4D_HOST_CONFIG") or len(visible_device_uuids()) != 1:
        raise ValueError("An explicit host config and exactly one allowed UUID are required")
    if any(
        os.environ.get(name, default) != default
        for name, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"))
    ):
        raise ValueError("Equivalence uses the production single-GPU path, not DDP")
    mode = "migration" if args.checkpoint else "fresh"
    overrides = case_overrides(args.task, args.variant, args.seed, mode)
    if args.reference and HOST_NAME != "remote":
        raise ValueError("Only remote comparison mode accepts an original local reference")
    roots = (REPO,) if RUNTIME_ROOT is None else (REPO, RUNTIME_ROOT)
    output_roots = (REPO / "runs",) if RUNTIME_ROOT is None else (REPO / "runs", RUNTIME_ROOT)
    output = artifact_path(args.output_dir, output_roots)
    validate_runtime_path(output)
    if output.exists():
        raise FileExistsError("Verification output must be a new directory")
    manifest_path = artifact_path(args.data_manifest, roots, existing=True)
    checkpoint = artifact_path(args.checkpoint, roots, existing=True) if args.checkpoint else None
    reference_path = artifact_path(args.reference, roots, existing=True) if args.reference else None
    if not output.parent.is_dir():
        raise FileNotFoundError("Verification output parent must already exist")
    output.mkdir()
    report = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "purpose": "cross_host_equivalence",
        "campaign_eligible": False,
        "host": HOST_NAME,
        "task": args.task,
        "variant": args.variant,
        "seed": args.seed,
        "mode": mode,
        "criterion": CRITERION,
        "steps": FRESH_STEPS if mode == "fresh" else MIGRATION_STEPS,
        "measurement_complete": False,
        "passed": False,
        "cross_host_acceptance": False,
        "source_sha256_before": source_fingerprints(),
        "verifier_sha256_before": verifier_fingerprints(),
        "runs": [],
    }
    atomic_report(output, report)
    try:
        mapping = guard_gpus()
        report["gpu_uuids"] = [item["uuid"] for item in mapping]
        require_passed_tests()
        from s4d.config import load_config
        from s4d.model.render import require_prebuilt_renderer

        commit = os.environ.get("S4D_EXPECTED_COMMIT", "")
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise ValueError("S4D_EXPECTED_COMMIT must identify the clean pushed source")
        verify_clean_release(REPO, commit)
        subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, "origin/splatter4d"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        image = os.environ.get("S4D_IMAGE_DIGEST", "")
        if not image.startswith("sha256:") or equivalence.SHA256.fullmatch(image.removeprefix("sha256:")) is None:
            raise ValueError("S4D_IMAGE_DIGEST must identify the actually verified immutable image on either host")
        report.update(git_commit=commit, source_dirty=False, image_digest=image)
        manifest_hash = sha256_file(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        report.update(
            data_identity=DATA_IDENTITY,
            data_manifest_sha256=manifest_hash,
            data_manifest_files=validate_data_manifest(manifest),
        )
        files, data_stats = verify_data_files(manifest, args.task)
        report["data_files"] = files
        checkpoint_hash = sha256_file(checkpoint) if checkpoint else None
        reference_bytes = reference_path.read_bytes() if reference_path else None
        reference = json.loads(reference_bytes) if reference_bytes is not None else None
        if reference is not None:
            validate_report(reference)
            for name, expected in {
                "mode": mode,
                "task": args.task,
                "variant": args.variant,
                "seed": args.seed,
                "git_commit": commit,
                "image_digest": image,
                "data_identity": DATA_IDENTITY,
                "data_manifest_sha256": manifest_hash,
                "checkpoint_sha256": checkpoint_hash,
            }.items():
                if reference[name] != expected:
                    raise ValueError(f"Original local reference differs before checkpoint loading: {name}")
            if (
                reference["host"] != "local"
                or reference["source_sha256"] != report["source_sha256_before"]
                or reference["verifier_sha256"] != report["verifier_sha256_before"]
            ):
                raise ValueError("Original local reference must use identical production source and frozen verifier")
            report["reference_sha256"] = hashlib.sha256(reference_bytes).hexdigest()
        if checkpoint is None:
            cfg = load_config(
                [REPO / "configs/metaworld/base.yaml", REPO / f"configs/metaworld/tasks/{args.task}.yaml"], overrides
            )
            start, checkpoint_state = 0, None
        else:
            cfg, start, checkpoint_state = original_checkpoint(checkpoint, args.task, args.variant, args.seed)
        validate_config(cfg, args.task, args.variant, args.seed, mode)
        normalized = numeric_config(cfg)
        report.update(
            numeric_config=normalized,
            config_sha256=hashlib.sha256(json_bytes(normalized)).hexdigest(),
            resume_step=start,
            checkpoint_sha256=checkpoint_hash,
            checkpoint_state=checkpoint_state,
        )
        if reference is not None:
            for name in ("config_sha256", "checkpoint_state", "resume_step"):
                if reference[name] != report[name]:
                    raise ValueError(f"Original local reference differs before native measurement: {name}")
        require_prebuilt_renderer()
        for _ in range(1 if HOST_NAME == "local" else 2):
            measured = {}
            report["runs"].append(measured)
            production_run(cfg, checkpoint, start, report["steps"], checkpoint_state, measured, report, output)
            verify_data_unchanged(data_stats)
            if sha256_file(manifest_path) != manifest_hash or (checkpoint and sha256_file(checkpoint) != checkpoint_hash):
                raise ValueError("Original manifest or checkpoint changed during measurement")
            atomic_report(output, report)
        report["source_sha256"] = source_fingerprints()
        report["source_changed_during_check"] = report["source_sha256"] != report["source_sha256_before"]
        report["verifier_sha256"] = verifier_fingerprints()
        if report["source_changed_during_check"] or report["verifier_sha256"] != report["verifier_sha256_before"]:
            raise ValueError("Production source or frozen verifier changed during native numerical verification")
        verify_clean_release(REPO, commit)
        report["measurement_complete"] = True
        validate_report(report)
        if reference is not None:
            if sha256_file(reference_path) != report["reference_sha256"]:
                raise ValueError("Original local report changed during comparison")
            comparison = compare_reports(reference, report)
            report.update(comparison)
            report["cross_host_acceptance"] = comparison["passed"]
            if not comparison["passed"]:
                raise ValueError("Measured cross-host numerical comparison did not pass the preregistered gate")
        atomic_report(output, report)
        return 0
    except BaseException as exc:
        report.update(passed=False, cross_host_acceptance=False, error=f"{type(exc).__name__}: {exc}")
        report["verifier_sha256"] = verifier_fingerprints()
        report["source_sha256"] = source_fingerprints()
        report["source_changed_during_check"] = report["source_sha256"] != report["source_sha256_before"]
        atomic_report(output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
