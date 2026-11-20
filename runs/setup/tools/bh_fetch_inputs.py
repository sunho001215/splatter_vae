"""Fetch only registered BH local rows' original verified mirror inputs. No placement/launch.

Preview unless --apply; RemoteClient reads the remote and writes the local mirror.
The original verifier/localizer expands compact-store and calibration closure.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

from bh_isolated_release import IDS, install_aliases  # noqa: E402
from bh_register import MAIN, require  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    try:
        import yaml

        from ops.jobs import realworld_local

        install_aliases()
        proof = json.loads(a.registration.read_bytes())
        actual = {r["id"]: r for r in yaml.safe_load((MAIN / "experiments/queue.yaml").read_bytes())["jobs"]}
        rows = [r for r in proof["rows"] if r["host"] == "local"]
        require(rows and {r["id"] for r in rows} <= IDS, "exact local BH rows required")
        for row in rows:
            require(actual.get(row["id"]) == row, "live row differs from registration proof")
        rels = sorted({name for row in rows for name in realworld_local.needed_inputs(row)["mirror"]})
        from ops.remote.access import RemoteClient

        result = realworld_local.fetch(
            rels,
            client=RemoteClient(repo=MAIN),
            priority="dependency",
            dry_run=not a.apply,
            mirror=MAIN / "runs/realworld-inputs",
        )
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        print("REFUSED: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
