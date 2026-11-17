"""Print a PNP v2 registration proposal. This preparation tool has no queue writer.

Even a valid authorization file cannot turn this checkout into a registrar. The
explicit gates reserve the future interface; actual publication needs a reviewed
follow-up using the established locked registration/placement machinery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ops.jobs.pnpv2_plan import build_rows
from s4d.rwlearn.pnpv2_spec import SPEC, TASK, load_spec

ROOT = Path(__file__).resolve().parents[2]


def proposal(spec_path, root=ROOT):
    spec_path = Path(spec_path)
    if not spec_path.is_absolute():
        spec_path = Path(root) / spec_path
    spec = load_spec(spec_path)
    rows = build_rows(spec, root, spec_path.relative_to(root) if spec_path.is_relative_to(root) else spec_path)
    return dict(
        schema="pick-and-place-v2-registration-proposal/1",
        status="dry-run",
        registered=False,
        launch_authorized=False,
        task=TASK,
        variants=["S"],
        spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        rows=rows,
        rows_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        blockers=[
            "not collected: TODO paths and unmeasured scene/prompts",
            "no full-data publication or accepted task calibration evidence",
            "no adopted release/image pin or scheduler resolver/placement approval",
            "queue publication intentionally unavailable in preparation-only registrar",
        ],
    )


def registration_request(record, authorization):
    if authorization is None or not authorization.is_file() or authorization.is_symlink():
        raise ValueError("--register requires an explicit regular authorization file")
    auth = json.loads(authorization.read_text())
    if (
        auth.get("schema") != "pick-and-place-v2-registration-authorization/1"
        or auth.get("task") != TASK
        or auth.get("authorize_registration") is not True
        or not isinstance(auth.get("authorized_by"), str)
        or not auth["authorized_by"].strip()
        or auth.get("spec_sha256") != record["spec_sha256"]
        or auth.get("rows_sha256") != record["rows_sha256"]
    ):
        raise ValueError("authorization must explicitly bind this task, spec and exact proposed rows")
    raise ValueError("preparation-only registrar: queue publication is not implemented; a reviewed follow-up is required")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path(SPEC))
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--authorization-file", type=Path)
    args = parser.parse_args(argv)
    try:
        record = proposal(args.spec)
        if args.register:
            registration_request(record, args.authorization_file)
        elif args.authorization_file:
            raise ValueError("authorization file alone cannot register; --register is also required")
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"REFUSED: {error}\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
