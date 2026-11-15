"""Read-only phase-A admission replay. No locks, launches, or live writes.

Run from the candidate worktree; --output must be a new directory under /tmp.
Both runs use one copied queue/registry and one source/ownership snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ops.jobs import host_roles, scheduler  # noqa: E402
from ops.jobs import local_phase_a as a  # noqa: E402
from ops.jobs import realworld_local as rw  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", type=Path, default=Path("/home/ws/ws/splatter4d"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, help="Replay a previous snapshot instead of reading live state")
    parser.add_argument("--cached-only", action="store_true", help="Compare against the snapshot's before.json")
    args = parser.parse_args()
    if args.cached_only and not args.snapshot:
        parser.error("--cached-only requires --snapshot")
    out = args.output.resolve()
    if not out.is_relative_to("/tmp"):
        parser.error("--output must be under /tmp")
    out.mkdir(exist_ok=False)
    live = args.live_root.resolve()
    for name in ("queue.yaml", "registry.jsonl"):
        shutil.copyfile((args.snapshot or live / "experiments") / name, out / name)
    state = scheduler.read_registry(out / "registry.jsonl")
    queue = scheduler.load_queue(out / "queue.yaml")
    by_id = {j["id"]: j for j in queue["jobs"]}
    rows = [
        j
        for j in scheduler.eligible(queue["jobs"], state)
        if j.get("host") == "local" and j["script"] == a.SCRIPT and j["id"].endswith("-hybridA")
    ]
    candidate_source = scheduler.source_identity()
    if args.snapshot:
        context = json.loads((args.snapshot / "context.json").read_text())
        ownership, roles, source = (context[k] for k in ("ownership", "roles", "source"))
        assert [j["id"] for j in rows] == context["ids"]
    else:
        ownership = host_roles.read_ownership(live / "runs/setup/ownership")
        roles = host_roles.load_roles(live / "configs/hosts/roles.yaml")
        scheduler.REPO = live
        source = scheduler.source_identity()
    scoped = {name for row in rows for name in ownership[row["id"]]["execution_scope"]["files"]}

    def scope_hashes(identity, root):
        return {
            name: identity["source_sha256"].get(name)
            or (hashlib.sha256((root / name).read_bytes()).hexdigest() if (root / name).is_file() else None)
            for name in scoped
        }

    baseline_scope = scope_hashes(source, live)
    candidate_scope = scope_hashes(candidate_source, ROOT)

    def differences(found):
        return sorted(
            {
                name
                for row in rows
                for name, expected in ownership[row["id"]]["execution_scope"]["files"].items()
                if found.get(name) != expected
            }
        )

    baseline_differences = differences(baseline_scope)
    candidate_differences = differences(candidate_scope)
    candidate_matches_snapshot = candidate_scope == baseline_scope
    (out / "context.json").write_text(
        json.dumps(dict(ids=[j["id"] for j in rows], ownership=ownership, roles=roles, source=source), indent=2)
    )
    original = scheduler.input_sha256
    results = {"before": json.loads((args.snapshot / "before.json").read_text())} if args.cached_only else {}
    for mode in ("after",) if args.cached_only else ("before", "after"):
        counts = Counter()

        def digest(path, mode=mode, counts=counts):
            path = Path(path)
            cache = scheduler._input_hash_cache.get()
            hit = mode == "after" and cache is not None and scheduler._input_file_identity(path.stat()) in cache
            counts["calls"] += 1
            counts["hits"] += hit
            if not hit:
                counts["bytes"] += path.stat().st_size
                category = "frozen" if "/frozen/" in str(path) else "raw" if "/real_world/" in str(path) else "evidence"
                counts[category + "_bytes"] += path.stat().st_size
            if mode == "after":
                return original(path)
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 2**20), b""):
                    h.update(block)
            return h.hexdigest()

        original_a, original_rw = a.sha, rw._hash
        if mode == "before":
            a.sha = rw._hash = digest
        else:
            scheduler.input_sha256 = digest
        decisions = {}
        start = time.perf_counter()
        with scheduler.input_verification_pass():
            for i, row in enumerate(rows, 1):
                try:
                    effective, record = rw.localize(
                        row, state, roles=roles, by_id=by_id, ownership=ownership, source_identity=source, root=live
                    )
                    decisions[row["id"]] = dict(ready=True, effective=effective, record=record)
                except rw.LocalInputError as exc:
                    decisions[row["id"]] = dict(ready=False, reason=str(exc))
                print(f"{mode}: {i}/{len(rows)} {time.perf_counter() - start:.3f}s", flush=True)
        a.sha, rw._hash, scheduler.input_sha256 = original_a, original_rw, original
        results[mode] = dict(seconds=time.perf_counter() - start, hashes=dict(counts), decisions=decisions)
        (out / f"{mode}.json").write_text(json.dumps(results[mode], indent=2, sort_keys=True))
    same = results["before"]["decisions"] == results["after"]["decisions"]
    summary = dict(
        rows=len(rows),
        identical_decisions=same,
        candidate_matches_placed_scope=not candidate_differences,
        candidate_matches_snapshot=candidate_matches_snapshot,
        baseline_scope_differences=baseline_differences,
        candidate_scope_differences=candidate_differences,
        **{mode: {k: v for k, v in result.items() if k != "decisions"} for mode, result in results.items()},
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not same or not rows or not candidate_matches_snapshot:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
