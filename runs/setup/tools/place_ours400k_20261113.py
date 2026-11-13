"""Explicit local placement and exact moved-ID hold release for Ours 400k.

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
import register_ours400k_20261113 as reg  # noqa: E402

from ops.jobs import host_roles  # noqa: E402
from ops.jobs import realworld_local as rw  # noqa: E402


def lift_moved(root, moved):
    """Only release an exact set proven to have local queue and ownership records."""
    moved = set(moved)
    if not moved:
        return []
    r = reg.reviewed
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
        r.require(len(holds) == 1, "expected exactly one Ours400k placement hold")
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
    reg.require(Path.cwd().resolve() == ROOT == reg.MAIN, "placement requires released main checkout")
    reg.require(
        all(
            re.fullmatch(
                r"rwv2-20261113-(?:pre|pre400k|export400k|fqldata400k|fql400k|eval400k)-ours-"
                r"[a-z0-9-]+-(?:S|SF)(?:-s[012])?",
                i,
            )
            for i in a.ids
        ),
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
    result = rw.apply(a.ids, "Ours 400k: adopted local release", allow_pauses=[reg.LOCAL_PAUSE], blocking=True)
    # A prior crash after apply but before hold release is recoverable for these explicitly requested IDs.
    queue = reg.reviewed.queue_document((ROOT / "experiments/queue.yaml").read_bytes())
    by_id = {j["id"]: j for j in queue["jobs"]}
    pauses = json.loads((ROOT / "runs/setup/admission-pauses.json").read_bytes())
    held = {i for h in pauses["pauses"] if h.get("name") == reg.LOCAL_PAUSE and h.get("active") for i in h.get("ids", [])}
    moved = set(result.get("moved") or []) | {i for i in a.ids if by_id[i]["host"] == "local" and i in held}
    print(json.dumps(dict(result=result, lifted=lift_moved(ROOT, moved)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
