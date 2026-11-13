"""Task/variant Ours 400k chains. Stdout-only unless the coordinator supplies --execute.

Row generation is pure; see realworld-ops/dsreset-20261113/ours400k/INTERFACE.md.
Publication reuses the reviewed ourx local-release transaction and exact-ID placement hold.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
SOURCE = next(p for p in Path(__file__).resolve().parents if (p / "s4d").is_dir())
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(SOURCE))
import register_ours_ext_20261113 as reviewed  # noqa: E402
import yaml  # noqa: E402

from s4d.config import load_config  # noqa: E402
from s4d.train.continuation import file_sha, science_config, validate_resume_state  # noqa: E402

TOOL = "runs/setup/tools/register_ours400k_20261113.py"
SCHEDULE = "configs/realworld/ours_ext/continuation400k.yaml"
LOCAL_PAUSE = "ours400k-await-local-placement-20261113"
STEM = "rw-registration-ours400k-20261113"
MAIN = reviewed.MAIN
require = reviewed.require


def build_rows(task, variant, data_manifest, base200k_source=None, *, current, release):
    """Return a fresh list of queue rows without reading/writing files or importing drivers.

    ``data_manifest`` supplies the pinned task recipe, four input paths, outcomes,
    template task/pre row, data hashes and readiness dependencies. ``base200k_source``
    is None (create a standard 200k pre row) or {id, checkpoint_sha256} of an already
    validated identical-data 200k producer. The caller validates/pins all evidence.
    """
    require(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", task or ""), "unsafe task")
    require(variant in ("S", "SF"), "variant must be S or SF")
    m = copy.deepcopy(data_manifest)
    needed = {
        "baseline_config",
        "baseline_sha256",
        "store",
        "split_manifest",
        "gate_report",
        "anchor_stats",
        "stats_id",
        "outcomes",
        "gate_sha256",
        "manifest_sha256",
        "recipe_source",
    }
    require(needed <= m.keys(), f"missing manifest fields: {sorted(needed - m.keys())}")
    require(
        m["baseline_config"] == f"configs/realworld/ours_ext/baseline200k-{task}-{variant}.yaml",
        "baseline must be a task/variant science snapshot",
    )
    require(m["anchor_stats"] == f"runs/{m['stats_id']}/workspace_stats.json", "stats path/dependency mismatch")
    for key in ("baseline_sha256", "gate_sha256", "manifest_sha256"):
        require(re.fullmatch("[0-9a-f]{64}", m[key] or ""), f"invalid {key}")
    if base200k_source is not None:
        require(
            set(base200k_source) == {"id", "checkpoint_sha256"}
            and re.fullmatch("[0-9a-f]{64}", base200k_source["checkpoint_sha256"] or ""),
            "validated source id and checkpoint digest required",
        )
    by_id = {r["id"]: r for r in current["jobs"]}
    template_pre = m["recipe_source"]
    require(template_pre in by_id, "recipe source row missing")
    rows = []

    def job(stage, seed=None):
        return f"rwv2-20261113-{stage}-ours-{task}-{variant}" + (f"-s{seed}" if seed is not None else "")

    metadata = dict(
        realworld_target=task,
        realworld_variant=variant,
        realworld_inputs=str(Path(m["store"]).parent),
        realworld_gate_sha256=m["gate_sha256"],
        realworld_manifest_sha256=m["manifest_sha256"],
        inputs=[m[k] for k in ("store", "split_manifest", "gate_report")],
    )

    # Never inherit live placement/retry history or drawer-specific gate identities.
    def derive(source_id, identity, script=None, args=None, deps=(), priority=-1, results=None):
        row = copy.deepcopy(by_id[source_id])
        for key in (
            "pin_host",
            "placement",
            "local_placement_pause",
            "release_source",
            "supersedes",
            "plan_row_id",
            "protocol_template",
            "purpose",
            "local_repinned_from",
            "calibration_release",
        ):
            row.pop(key, None)
        row.update(
            metadata,
            id=identity,
            derived_from=source_id,
            host="remote",
            gpu="any",
            gpu_profile="1g.24gb",
            cpu_threads=8,
            gpu_mem_gb=row["mem_gb"],
            planned_commit=release["commit"],
            planned_image=release["image"],
            evidence_manifest=release["evidence_manifest"],
            registration_tool=TOOL,
            priority=priority,
            deps=list(dict.fromkeys(deps)),
            calibration_dependent=True,
            gate_level="geometry (full-data gate + calibration_validated)",
        )
        if release.get("source") == "local":
            row.update(release_source="local", local_placement_pause=LOCAL_PAUSE)
        if script is not None:
            row["script"] = script
        if args is not None:
            row["args"] = args
        if results is not None:
            row["required_results"] = results
        rows.append(row)
        return row

    cfg_inputs = [
        f"{key}={m[name]}"
        for key, name in (
            ("data.store", "store"),
            ("data.split_manifest", "split_manifest"),
            ("data.gate_report", "gate_report"),
            ("model.anchor_stats", "anchor_stats"),
        )
    ]
    readiness = [m["stats_id"], *m.get("deps", [])]
    if base200k_source is None:
        pre = job("pre")
        derive(
            template_pre,
            pre,
            "scripts/train.py",
            [
                "--config",
                m["baseline_config"],
                "--name",
                pre,
                "--output-root",
                "runs",
                "--resume",
                "auto",
                "--set",
                *cfg_inputs,
            ],
            readiness,
            -3,
            ["completion.json", "metrics.jsonl", "checkpoints/step_0200000.pt"],
        )
    else:
        pre = base200k_source["id"]
    pre400, export, dataset = job("pre400k"), job("export400k"), job("fqldata400k")
    args = [
        "--name",
        pre400,
        "--output-root",
        "runs",
        "--source-checkpoint",
        f"runs/{pre}/checkpoints/step_0200000.pt",
        "--baseline-config",
        m["baseline_config"],
        "--baseline-sha256",
        m["baseline_sha256"],
        "--schedule-config",
        SCHEDULE,
        "--start-step",
        "200000",
        "--stop-step",
        "400000",
    ]
    if base200k_source:
        args += ["--source-sha256", base200k_source["checkpoint_sha256"]]
    args += ["--set", *cfg_inputs]
    derive(
        template_pre,
        pre400,
        "scripts/train_ours_continuation.py",
        args,
        [pre, *readiness],
        -3,
        ["completion.json", "continuation.json", "metrics.jsonl", "resume_source.json", "checkpoints/step_0400000.pt"],
    )
    derive(
        "rwcal-20261111-ours-drawer-open-export",
        export,
        args=[
            "--run",
            f"runs/{pre400}",
            "--checkpoint",
            f"runs/{pre400}/checkpoints/step_0400000.pt",
            "--out",
            f"runs/{export}/encoder.pt",
        ],
        deps=[pre400],
        results=["encoder.pt"],
    )
    old_data = "rwcal-20261111-fqldata-ours-drawer-open"
    old = by_id[old_data]["args"]
    replacements = {
        "--store": m["store"],
        "--gate-report": m["gate_report"],
        "--outcomes": m["outcomes"],
        "--out": f"runs/{dataset}/dataset.h5",
        "--export": f"runs/{export}/encoder.pt",
    }
    args = list(old)
    for flag, value in replacements.items():
        args[args.index(flag) + 1] = value
    derive(old_data, dataset, args=args, deps=[export, *m.get("deps", [])])
    for seed in range(3):
        source = f"rwnq-20261111-fql-ours-drawer-open-s{seed}"
        identity = job("fql400k", seed)
        old = by_id[source]["args"]
        require(
            {
                "agent.alpha=10.0",
                "agent.normalize_q_loss=true",
                "agent.batch_size=256",
                "train.offline_steps=1000000",
                f"train.seed={seed}",
            }
            <= set(old),
            "unexpected FQL recipe",
        )
        changes = {
            "data.dataset": f"runs/{dataset}/dataset.h5",
            "data.split_manifest": m["split_manifest"],
            "data.gate_report": m["gate_report"],
        }
        args = [
            identity
            if x == source
            else f"{x.partition('=')[0]}={changes[x.partition('=')[0]]}"
            if x.partition("=")[0] in changes
            else x
            for x in old
        ]
        derive(source, identity, args=args, deps=[dataset])
    if m.get("include_eval", True):
        identity = job("eval400k")
        row = derive(
            template_pre,
            identity,
            "scripts/evaluate_ours_length.py",
            [
                "--name",
                identity,
                "--output-root",
                "runs",
                "--checkpoint",
                f"runs/{pre400}/checkpoints/step_0400000.pt",
                "--config",
                m["baseline_config"],
                "--set",
                *cfg_inputs,
            ],
            [pre400, *readiness],
            results=["evaluation.json", "grasp_sites.json", "completion.json"],
        )
        row.update(max_restarts=1, disk_gb=2)
    return rows


def verify_checkpoint(path, cfg):
    """Read-only, CPU full AdamW/scheduler/RNG and exact recipe validation."""
    import torch

    from s4d.train.loop import build_model
    from s4d.train.schedule import build_optimizer

    before = file_sha(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(cfg)
    optimizer, _ = build_optimizer(model, cfg)
    validate_resume_state(state, cfg, optimizer=optimizer, lower=200000, upper=400000, exact_step=200000)
    require(
        state["scheduler"]["base_lrs"] == [cfg["train"]["lr"]] * len(optimizer.param_groups),
        "checkpoint base LRs differ from original recipe",
    )
    require(
        all(g["lr"] == cfg["train"]["min_lr"] for g in state["optimizer"]["param_groups"])
        and state["scheduler"]["_last_lr"] == [cfg["train"]["min_lr"]] * len(optimizer.param_groups),
        "200k checkpoint did not reach original minimum LR",
    )
    require(file_sha(path) == before, "checkpoint changed during validation")
    return dict(
        sha256=before,
        step=state["step"],
        adamw_states=len(state["optimizer"]["state"]),
        last_epoch=state["scheduler"]["last_epoch"],
        group_lrs=state["scheduler"]["_last_lr"],
        rank_rng_states=len(state.get("rng_by_rank", [state["rng"]])),
    )


def prepare(repo, source_root, commit, image, specs, release_source="local"):
    require(re.fullmatch("[0-9a-f]{40}", commit or ""), "full --commit required")
    require(re.fullmatch("sha256:[0-9a-f]{64}", image or ""), "image digest required")
    require(release_source == "local", "these rows require --release-source local")
    # Candidate dry-runs pin prospective local release; only publication claims acceptance.
    release = dict(commit=commit, image=image, source="local", evidence_manifest=reviewed.LOCAL_EVIDENCE)
    if repo.resolve() == source_root.resolve():
        release = reviewed.local_release_pin(repo, commit, image)
    before = (repo / "experiments/queue.yaml").read_bytes()
    registry = (repo / "experiments/registry.jsonl").read_bytes()
    current = reviewed.queue_document(before)
    rows, inputs, evidence = [], {}, {}
    for path in specs:
        path = Path(path)
        spec = json.loads(path.read_bytes())
        inputs[str(path.resolve())] = file_sha(path)
        task, variant, m = spec["task"], spec["variant"], spec["data_manifest"]
        baseline = source_root / m["baseline_config"]
        require(file_sha(baseline) == m["baseline_sha256"], "baseline digest mismatch")
        cfg = load_config([baseline])
        require(cfg["train"]["steps"] == 200000 and cfg["wandb"]["project"] == "splatter4d-realworld", "wrong baseline")
        require(
            all(
                cfg[s][k] == m[name]
                for s, k, name in (
                    ("data", "store", "store"),
                    ("data", "split_manifest", "split_manifest"),
                    ("data", "gate_report", "gate_report"),
                    ("model", "anchor_stats", "anchor_stats"),
                )
            ),
            "manifest science mismatch",
        )
        source = spec.get("base200k_source")
        recipe = repo / spec["recipe_config"] if spec.get("recipe_config") else None
        if recipe:
            require(science_config(load_config([recipe])) == science_config(cfg), "actual 200k recipe changed")
            inputs[str(recipe)] = file_sha(recipe)
        else:
            by_id = {j["id"]: j for j in current["jobs"]}
            args = by_id[m["recipe_source"]]["args"]
            saved = load_config([source_root / args[args.index("--config") + 1]], args[args.index("--set") + 1 :])
            for section, key, name in (
                ("data", "store", "store"),
                ("data", "split_manifest", "split_manifest"),
                ("data", "gate_report", "gate_report"),
                ("model", "anchor_stats", "anchor_stats"),
            ):
                saved[section][key] = m[name]
            saved["data"].update(task=task, target=task)
            require(saved == cfg, "paused pretraining row recipe changed")
            inputs[str(source_root / args[args.index("--config") + 1])] = file_sha(
                source_root / args[args.index("--config") + 1]
            )
        if source:
            require(recipe is not None, "reusing a checkpoint requires its actual saved run config")
            checkpoint = repo / "runs" / source["id"] / "checkpoints/step_0200000.pt"
            evidence[f"{task}-{variant}"] = verify_checkpoint(checkpoint, load_config([baseline, source_root / SCHEDULE]))
            require(evidence[f"{task}-{variant}"]["sha256"] == source["checkpoint_sha256"], "checkpoint digest mismatch")
            inputs[str(checkpoint)] = source["checkpoint_sha256"]
        rows.extend(build_rows(task, variant, m, source, current=current, release=release))
        inputs[str(baseline)] = file_sha(baseline)
    # Retain the reviewed proof that all seed templates still resolve to their saved full science.
    by_id = {j["id"]: j for j in current["jobs"]}
    for seed in range(3):
        identity = f"rwnq-20261111-fql-ours-drawer-open-s{seed}"
        args = by_id[identity]["args"]
        config = source_root / args[args.index("--config") + 1]
        saved = repo / "runs" / identity / "config.yaml"
        saved_cfg = yaml.safe_load(saved.read_bytes())
        saved_cfg.pop("run", None)
        require(load_config([config], args[args.index("--set") + 1 :]) == saved_cfg, "FQL template science changed")
        inputs.update({str(p): file_sha(p) for p in (config, saved)})
    known = {j["id"] for j in current["jobs"]} | reviewed.registry_ids(registry)
    require(len(rows) == len({j["id"] for j in rows}), "duplicate proposed ids")
    require(not known.intersection(j["id"] for j in rows), "proposed ids already exist")
    require(
        all(not (repo / "runs" / j["id"]).exists() and not (repo / "runs" / j["id"]).is_symlink() for j in rows),
        "proposed run path already exists",
    )
    require(not (repo / f"runs/setup/{STEM}.pending.json").exists(), "pending journal requires inspection")
    for name in (SCHEDULE, TOOL, reviewed.TOOL):
        inputs[str(source_root / name)] = file_sha(source_root / name)
    if repo.resolve() == source_root.resolve():
        inputs[str(repo / "docs/tests.json")] = file_sha(repo / "docs/tests.json")
    candidate = reviewed.queue_candidate(before, rows)
    return dict(
        schema="rw-registration/v1",
        status="dry-run",
        registered=False,
        prepared_at=reviewed.now(),
        repo=str(repo.resolve()),
        source_root=str(source_root.resolve()),
        release=release,
        rows=rows,
        rows_sha256=reviewed.json_sha(rows),
        priorities=dict(pretraining=-3, ours_fql=-1),
        input_sha256=inputs,
        checkpoint_evidence=evidence,
        local_pause=LOCAL_PAUSE,
        journal_stem=STEM,
        holds=[reviewed.local_hold(rows, LOCAL_PAUSE)],
        queue_sha256_before=reviewed.sha(before),
        queue_sha256_after=reviewed.sha(candidate),
        registry_sha256=reviewed.sha(registry),
        queue_diff=dict(added_rows=len(rows), modified_rows=0, removed_rows=0, added_ids=[j["id"] for j in rows]),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=MAIN)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--spec", type=Path, action="append", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--release-source", choices=["local"], default="local")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    record = prepare(args.repo, args.source_root, args.commit, args.image, args.spec, args.release_source)
    if args.execute:
        record = reviewed.execute(args.repo, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
