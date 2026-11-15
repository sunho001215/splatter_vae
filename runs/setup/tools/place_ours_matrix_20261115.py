"""Explicit local placement and exact moved-ID hold release for Ours scratch matrix.

Default previews the reviewed planner; --execute performs placement. Never launches.
Run only after adoption/registration, from the main checkout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "s4d").is_dir())
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import register_ours_matrix_20261115 as reg  # noqa: E402

from ops.jobs import host_roles  # noqa: E402
from ops.jobs import realworld_local as rw  # noqa: E402


def valid_id(identity, by_id=None):
    return bool(re.fullmatch(
        r"rwv3-20261115-(?:pre|export|fqldata|fql|eval)-ours-"
        r"(?:drawer-open-v2|covering|pick-and-place)-(?:S|SF)-(?:orig|ch01)-"
        r"(?:scratch400k|continuation400k)(?:-s[012])?", identity)
        and (by_id or {}).get(identity, {}).get('registration_tool') == reg.TOOL)


def lift_moved(root, moved):
    """Only release an exact set proven to have local queue and ownership records."""
    moved = set(moved)
    if not moved:
        return []
    r = reg.R
    with r.locks(root):
        doc = r.queue_document((root / "experiments/queue.yaml").read_bytes())
        by_id = {j["id"]: j for j in doc["jobs"]}
        owned = host_roles.read_ownership()
        r.require(
            all(
                by_id[i]["host"] == "local"
                and owned[i]["owner_host"] == "local"
                and by_id[i].get("local_placement_pause") == reg.LOCAL_PAUSE
                for i in moved
            ),
            "moved IDs lack verified local ownership",
        )
        path = root / "runs/setup/admission-pauses.json"
        pauses = json.loads(path.read_bytes())
        holds = [h for h in pauses["pauses"] if h.get("name") == reg.LOCAL_PAUSE]
        r.require(len(holds) == 1, "expected exactly one Ours matrix placement hold")
        hold = holds[0]
        r.require(
            hold["active"] is True and not hold.get("prefixes") and moved <= set(hold["ids"]),
            "moved IDs are not in the active exact-ID hold",
        )
        hold["ids"] = [i for i in hold["ids"] if i not in moved]
        hold["rows"] = list(hold["ids"])
        hold["active"] = bool(hold["ids"])
        r.atomic_write(path, r.json_bytes(pauses))
    return sorted(moved)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ids", nargs="+")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args(argv)
    reg.R.require(Path.cwd().resolve() == ROOT == reg.MAIN, "placement requires released main checkout")
    queue = reg.R.queue_document((ROOT / "experiments/queue.yaml").read_bytes())
    by_id = {j["id"]: j for j in queue["jobs"]}
    reg.R.require(
        all(valid_id(i, by_id) for i in a.ids),
        "unexpected row ID",
    )
    if not a.execute:
        # The official CLI is the read-only planner used by the original place_ourx helper.
        import subprocess

        return subprocess.run(
            [
                sys.executable,
                "-m",
                "ops.jobs.realworld_local",
                "place",
                "--dry-run",
                "--allow-pause",
                reg.LOCAL_PAUSE,
                *a.ids,
            ],
            check=False,
        ).returncode
    result = rw.apply(a.ids, "Ours scratch matrix: adopted local release", allow_pauses=[reg.LOCAL_PAUSE], blocking=True)
    # A prior crash after apply but before hold release is recoverable for these explicitly requested IDs.
    queue = reg.R.queue_document((ROOT / "experiments/queue.yaml").read_bytes())
    by_id = {j["id"]: j for j in queue["jobs"]}
    pauses = json.loads((ROOT / "runs/setup/admission-pauses.json").read_bytes())
    held = {i for h in pauses["pauses"] if h.get("name") == reg.LOCAL_PAUSE and h.get("active") for i in h.get("ids", [])}
    moved = set(result.get("moved") or []) | {i for i in a.ids if by_id[i]["host"] == "local" and i in held}
    print(json.dumps(dict(result=result, lifted=lift_moved(ROOT, moved)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
