"""Print the nearest compatible accepted baseline for run_tests.py --changed.

Usage: BASE=$(CUDA_VISIBLE_DEVICES=<free MIG> .venv/bin/python -I scripts/test_baseline.py)
Then pass --changed "$BASE" to the unchanged runner. This is read-only and fails
before collection if there is no suitable baseline; the release gate still decides acceptance.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import test_gate  # noqa: E402
import test_impact  # noqa: E402


def nearest_baseline(repo: Path, cache: Path, environment: dict) -> str:
    """Breadth-first ancestry order (first parent breaks ties); validate cached records against git objects."""
    history = subprocess.run(["git", "-C", str(repo), "rev-list", "--parents", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.splitlines()
    parents = {parts[0]: parts[1:] for line in history if (parts := line.split())}
    pending, seen = deque([next(iter(parents))]), set()
    store = test_gate.Baselines(cache)
    while pending:
        commit = pending.popleft()
        if commit in seen:
            continue
        seen.add(commit)
        record = store.load(commit)
        if record and record.get("environment") == environment:
            nodes = record.get("nodes")
            if isinstance(nodes, dict) and nodes and all(isinstance(v, list) and v for v in nodes.values()):
                if record.get("source_digest") == test_gate.source_digest(test_gate.commit_sources(repo, commit)):
                    return commit
        pending.extend(parents.get(commit, []))
    raise ValueError("no compatible accepted ancestor baseline; run a complete suite or register existing complete "
                     "evidence with scripts/test_gate.py record-baseline")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="also print baseline provenance")
    options = parser.parse_args(argv)
    from ops.resources.gpu_guard import HOST_NAME, enforce_allowed_gpus

    with contextlib.redirect_stdout(sys.stderr):
        mapping = enforce_allowed_gpus()
    import torch

    environment = test_gate.environment(dict(python=sys.version, torch=torch.__version__, cuda_runtime=torch.version.cuda,
                                             host=HOST_NAME, gpu_mapping=mapping))
    try:
        cache = test_impact.cache_root(REPO)
        commit = nearest_baseline(REPO, cache, environment)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if options.json:
        record = test_gate.Baselines(cache).load(commit)
        print(json.dumps({k: record[k] for k in ("commit", "source_digest", "environment", "evidence")}))
    else:
        print(commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
