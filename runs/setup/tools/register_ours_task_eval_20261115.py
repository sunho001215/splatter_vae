"""Replace non-drawer Ours length evaluations. Stdout-only dry-run; coordinator --execute.

The established repin tools preserve script/argv, so this transaction appends -v2
rows and permanently holds never-launched originals. Failed covering S stays intact.
Publication installs exact-ID holds before appending, under both registrar locks.
No registry writes, placement, launch, driver calls or network access.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(SOURCE), str(Path(__file__).resolve().parent)]

import register_ours400k_20261113 as reg  # noqa: E402

from ops.jobs import lifecycle, scheduler  # noqa: E402
from ops.jobs import realworld_local as rw  # noqa: E402
from s4d.config import load_config  # noqa: E402
from s4d.train.continuation import file_sha  # noqa: E402

R = reg.reviewed
MAIN = reg.MAIN
TOOL = "runs/setup/tools/register_ours_task_eval_20261115.py"
STEM = "rw-registration-ours-task-eval-20261115"
OLD_SCRIPT = "scripts/evaluate_ours_length.py"
NEW_SCRIPT = "scripts/evaluate_ours_task_length.py"
COVERING = "rwv2-20261113-eval400k-ours-covering-S"
OBSOLETE_PAUSE = "ours-task-eval-superseded-20261115"
PAUSES = "runs/setup/admission-pauses.json"
TOOLS = (TOOL, reg.TOOL, R.TOOL, "runs/setup/tools/register_dsreset_20261113.py",
         "runs/setup/tools/place_ours400k_20261113.py", "runs/setup/tools/register_fql_nq_20261111.py")


def build_rows(sources, release):
    """Copy science, inputs, dependencies and resource declarations exactly."""
    rows = []
    for source in sources:
        identity = source["id"] + "-v2"
        R.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*-v2", identity), "unsafe replacement ID")
        row = copy.deepcopy(source)
        for key in ("pin_host", "placement", "local_repinned_from", "supersedes", "plan_row_id",
                    "protocol_template", "purpose"):
            row.pop(key, None)
        R.require(row["script"] == OLD_SCRIPT, "source entry point changed")
        args = row["args"]
        R.require(args.count("--name") == 1 and args[args.index("--name") + 1] == source["id"],
                  "source output identity differs")
        args[args.index("--name") + 1] = identity
        row.update(id=identity, script=NEW_SCRIPT, derived_from=source["id"], host="remote", gpu="any",
                   priority=-1, gpu_profile="1g.24gb", planned_commit=release["commit"],
                   planned_image=release["image"], evidence_manifest=release["evidence_manifest"],
                   release_source="local", local_placement_pause=reg.LOCAL_PAUSE, registration_tool=TOOL)
        rw.needed_inputs(row)  # Same strict argv/input contract as explicit placement.
        rows.append(row)
    return rows


def pause_candidate(before, holds):
    """Merge exact IDs; preserve other holds and IDs already awaiting Ours placement."""
    document = json.loads(before)
    R.require(isinstance(document.get("pauses"), list), "invalid admission pauses")
    for proposed in holds:
        matches = [h for h in document["pauses"] if h.get("name") == proposed["name"]]
        R.require(len(matches) <= 1, "duplicate named admission hold")
        if not matches:
            document["pauses"].append(copy.deepcopy(proposed))
            continue
        hold = matches[0]
        R.require(not hold.get("prefixes"), "placement/superseded holds must have exact IDs only")
        ids = sorted(set(hold.get("ids", [])) | set(proposed["ids"]))
        hold.update(active=True, ids=ids, rows=ids)
    return R.json_bytes(document)


def require_holds(repo, record):
    active = lifecycle.load_pauses(repo / PAUSES)
    for proposed in record["holds"]:
        holds = [h for h in active if h.get("name") == proposed["name"]]
        R.require(len(holds) == 1 and not holds[0].get("prefixes")
                  and set(proposed["ids"]) <= set(holds[0].get("ids", [])),
                  f"missing active exact-ID hold: {proposed['name']}")


def queue_candidate(before, rows):
    candidate = R.queue_candidate(before, rows)
    # Validate the entire queue, including unchanged historical rows, with the scheduler's real reader.
    with tempfile.TemporaryDirectory(prefix="ours-task-eval-") as directory:
        path = Path(directory) / "queue.yaml"
        path.write_bytes(candidate)
        scheduler.load_queue(path)
    R.require(R.json_sha(R.queue_document(candidate)) == R.json_sha(
        {**R.queue_document(before), "jobs": R.queue_document(before)["jobs"] + rows}),
        "full-queue equality failed")
    return candidate


def prepare(repo, source_root, commit, image, release_source="local"):
    repo, source_root = Path(repo), Path(source_root)
    R.require(re.fullmatch("[0-9a-f]{40}", commit or ""), "full --commit required")
    R.require(re.fullmatch("sha256:[0-9a-f]{64}", image or ""), "image digest required")
    R.require(release_source == "local", "task evaluations require local release")
    release = dict(commit=commit, image=image, source="local", evidence_manifest=R.LOCAL_EVIDENCE)
    if repo.resolve() == source_root.resolve():
        release = R.local_release_pin(repo, commit, image)
    before = (repo / "experiments/queue.yaml").read_bytes()
    registry = (repo / "experiments/registry.jsonl").read_bytes()
    pauses = (repo / PAUSES).read_bytes()
    current = R.queue_document(before)
    by_id = {j["id"]: j for j in current["jobs"]}
    seen = R.registry_ids(registry)
    state = scheduler.read_registry(repo / "experiments/registry.jsonl")
    R.require(COVERING in by_id and state.get(COVERING, {}).get("status") == "failed",
              "covering S must remain registered failed history")
    sources, obsolete, inputs, skipped = [], [], {}, []
    for source in current["jobs"]:
        if source.get("script") != OLD_SCRIPT:
            continue
        known = rw._arguments(source)
        configs = [source_root / p for p in known.config]
        cfg = load_config(configs, known.set)
        task = cfg["data"]["task"]
        R.require(task == source.get("realworld_target"), "source config/target differs")
        if task == "drawer-open":  # Historical drawer protocol remains pinned.
            continue
        identity = source["id"]
        if identity != COVERING:
            if identity in seen:
                skipped.append(dict(id=identity, reason="registry history; preserve launched/attempted row"))
                continue
            run = repo / "runs" / identity
            R.require(not run.exists() and not run.is_symlink(), f"never-launched source has output: {identity}")
            obsolete.append(source)
        else:
            pre = "rwv2-20261113-pre400k-ours-covering-S"
            stats = "rwcal-20261111-ours-covering-stats"
            R.require(task == "covering" and known.checkpoint == f"runs/{pre}/checkpoints/step_0400000.pt"
                      and cfg["model"]["anchor_stats"] == f"runs/{stats}/workspace_stats.json"
                      and pre in source["deps"] and stats in source["deps"]
                      and all(state.get(i, {}).get("status") == "done" for i in (pre, stats)),
                      "covering S must reuse the done pre400k checkpoint and original stats")
        for path in configs:
            inputs[str(path.resolve())] = file_sha(path)
        # Hash available done dependencies, including covering's original checkpoint and stats.
        for dep, name in rw.needed_inputs(source)["dependency_files"]:
            if state.get(dep, {}).get("status") == "done":
                path = repo / "runs" / dep / name
                inputs[str(path.resolve())] = file_sha(path)
        sources.append(source)
    R.require(any(j["id"] == COVERING for j in sources), "covering S old-script source missing")
    rows = build_rows(sources, release)
    known_ids = set(by_id) | seen
    R.require(len({j["id"] for j in rows}) == len(rows) and not known_ids.intersection(j["id"] for j in rows),
              "replacement IDs already registered")
    R.require(all(not (repo / "runs" / j["id"]).exists() and not (repo / "runs" / j["id"]).is_symlink()
                  for j in rows), "replacement output already exists")
    R.require(not any((repo / f"runs/setup/{STEM}{suffix}").exists()
                      for suffix in (".pending.json", ".json")), "existing journal/proof requires inspection")
    for name in (*TOOLS, NEW_SCRIPT):
        inputs[str((source_root / name).resolve())] = file_sha(source_root / name)
    if repo.resolve() == source_root.resolve():
        inputs[str((repo / "docs/tests.json").resolve())] = file_sha(repo / "docs/tests.json")
    holds = [R.local_hold(rows, reg.LOCAL_PAUSE)]
    if obsolete:
        hold = R.local_hold(obsolete, OBSOLETE_PAUSE)
        hold["reason"] = "Superseded old non-drawer evaluator; retain history and never admit these originals"
        holds.append(hold)
    candidate = queue_candidate(before, rows)
    paused = pause_candidate(pauses, holds)
    return dict(schema="rw-registration/v1", status="dry-run", registered=False, prepared_at=R.now(),
                repo=str(repo.resolve()), source_root=str(source_root.resolve()), release=release, rows=rows,
                rows_sha256=R.json_sha(rows), input_sha256=inputs, holds=holds,
                pauses_sha256_before=R.sha(pauses), pauses_sha256_after=R.sha(paused),
                queue_sha256_before=R.sha(before), queue_sha256_after=R.sha(candidate), registry_sha256=R.sha(registry),
                skipped=skipped, superseded_ids=[j["id"] for j in obsolete],
                repin_policy="established repin tools preserve script/argv; append -v2 and hold originals",
                queue_diff=dict(added_rows=len(rows), modified_rows=0, removed_rows=0,
                                added_ids=[j["id"] for j in rows]),
                placement_command=[".venv/bin/python", "runs/setup/tools/place_ours400k_20261113.py",
                                   *[j["id"] for j in rows]])


def execute(repo, record):
    repo = Path(repo)
    R.require(repo.resolve() == MAIN and Path(__file__).resolve() == MAIN / TOOL,
              "only coordinator released main-checkout copy may execute")
    R.require(Path(record["source_root"]).resolve() == MAIN, "execute must use main sources")
    setup = repo / "runs/setup"
    pending, proof = setup / f"{STEM}.pending.json", setup / f"{STEM}.json"
    candidate_path = setup / f"{STEM}.queue-candidate.yaml"
    with R.locks(repo):
        R.require(not pending.exists() and not proof.exists(), "existing journal/proof requires inspection")

        def check(*, held=False):
            R.require(R.local_release_pin(repo, record["release"]["commit"], record["release"]["image"])
                      == record["release"], "local release changed")
            R.require(file_sha(repo / "experiments/queue.yaml") == record["queue_sha256_before"], "queue changed")
            R.require(file_sha(repo / "experiments/registry.jsonl") == record["registry_sha256"], "registry changed")
            R.require(R.json_sha(record["rows"]) == record["rows_sha256"], "rows changed")
            R.require(all(file_sha(p) == digest for p, digest in record["input_sha256"].items()), "inputs changed")
            R.require(file_sha(repo / PAUSES) == record["pauses_sha256_after" if held else "pauses_sha256_before"],
                      "admission pauses changed")
            for row in record["rows"]:
                run = repo / "runs" / row["id"]
                R.require(not run.exists() and not run.is_symlink(), "replacement output appeared")
            for identity in record["superseded_ids"]:
                run = repo / "runs" / identity
                R.require(not run.exists() and not run.is_symlink(), "never-launched source output appeared")
            if held:
                require_holds(repo, record)

        check()
        # Tools live outside source_fingerprints; publication must use their adopted HEAD bytes too.
        from ops.jobs.migration import commit_reader

        read, _ = commit_reader(repo, record["release"]["commit"])
        R.require(all(read(name) == (repo / name).read_bytes() for name in TOOLS), "registrar/helper differs from HEAD")
        for row in record["rows"]:
            rw.placement_scope(row, root=repo)
        candidate = queue_candidate((repo / "experiments/queue.yaml").read_bytes(), record["rows"])
        paused = pause_candidate((repo / PAUSES).read_bytes(), record["holds"])
        R.require(R.sha(candidate) == record["queue_sha256_after"] and R.sha(paused) == record["pauses_sha256_after"],
                  "candidate changed")
        R.atomic_write(candidate_path, candidate)
        R.atomic_write(pending, R.json_bytes(dict(record, status="pending")))
        check()
        # If interrupted, holds remain active and the journal requires inspection before retry.
        R.atomic_write(repo / PAUSES, paused)
        check(held=True)
        import os

        os.replace(candidate_path, repo / "experiments/queue.yaml")
        R.sync_directory(repo / "experiments")
        result = dict(record, status="published", registered=True, registered_at=R.now())
        R.atomic_write(proof, R.json_bytes(result))
        pending.unlink()
        R.sync_directory(setup)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=MAIN)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--release-source", choices=["local"], default="local")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    record = prepare(args.repo, args.source_root, args.commit, args.image, args.release_source)
    if args.execute:
        record = execute(args.repo, record)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
